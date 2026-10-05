"""Build the temporal PK/LK/NK validation set by diffing CAFA-6's train_terms.tsv (t0)
against the already-cached current Swiss-Prot dump (t1). See
docs/superpowers/specs/2026-10-05-temporal-pk-eval-design.md for the full design.
"""
import sys
from pathlib import Path

# Same sys.path fix as pk_eval_common.py (Task 1 ruling) - running this script directly
# puts experiments/pk_eval/scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import obonet
import pandas as pd

from generate_dataset import propagate_protein_terms, restrict_to_true_path
from pk_eval_common import filter_to_vocab, split_by_aspect

DATA_ROOT = Path("data/cafa-6-protein-function-prediction")
TRAIN_TERMS_PATH = DATA_ROOT / "Train/train_terms.tsv"
GO_OBO_PATH = DATA_ROOT / "Train/go-basic.obo"


def build_t0_index(train_terms_path, true_path_graph, label2id, term_to_aspect) -> dict:
  """For every protein in train_terms.tsv: propagate its asserted terms to the full
  true-path closure (reusing generate_dataset.propagate_protein_terms - this is the
  batch, whole-corpus case it was designed for), filter to the trainable vocabulary, then
  split into per-aspect known-term buckets.
  """
  train_df = pd.read_csv(train_terms_path, sep="\t")
  propagated = propagate_protein_terms(train_df, true_path_graph)
  t0_index = {}
  for protein_id, terms in propagated.items():
    vocab_terms = filter_to_vocab(terms, label2id)
    t0_index[protein_id] = split_by_aspect(vocab_terms, term_to_aspect)
  return t0_index


def categorize_protein(known_by_aspect: dict, t1_novel_by_aspect: dict) -> dict:
  has_any_t0_annotation = any(known_by_aspect[a] for a in ("C", "F", "P"))
  result = {}
  for aspect in ("C", "F", "P"):
    if not t1_novel_by_aspect.get(aspect):
      result[aspect] = None
      continue
    if known_by_aspect[aspect]:
      result[aspect] = "PK"
    elif has_any_t0_annotation:
      result[aspect] = "LK"
    else:
      result[aspect] = "NK"
  return result


SWISSPROT_DUMP_PATH = Path("data/uniprot/uniprot_sprot.dat.gz")


def stream_diff(dump_path, t0_index, true_path_graph, label2id, term_to_aspect,
                record_limit=None) -> tuple:
  import gzip
  import itertools

  from Bio import SwissProt

  from pk_eval_common import extract_experimental_go_terms, propagate_terms

  added_rows, unseen_rows = [], []
  ancestor_cache = {}

  with gzip.open(dump_path, "rt") as fh:
    records = SwissProt.parse(fh)
    if record_limit is not None:
      records = itertools.islice(records, record_limit)
    for record in records:
      accession = record.accessions[0]
      raw_terms = extract_experimental_go_terms(record)
      if not raw_terms:
        continue  # e.g. Q6GZX4: 100% electronic annotation, nothing to diff against
      propagated = propagate_terms(raw_terms, true_path_graph, ancestor_cache)
      vocab_terms = filter_to_vocab(propagated, label2id)
      novel_by_aspect = split_by_aspect(vocab_terms, term_to_aspect)
      if not any(novel_by_aspect.values()):
        continue  # every term dropped by vocab filtering -> no scoreable example

      if accession in t0_index:
        known_by_aspect = t0_index[accession]
        category_by_aspect = categorize_protein(known_by_aspect, novel_by_aspect)
        if not any(category_by_aspect.values()):
          continue  # no aspect actually gained a novel term -> no example
        added_rows.append({
            "protein_id": accession,
            "sequence": record.sequence,
            "known_by_aspect": known_by_aspect,
            "novel_by_aspect": novel_by_aspect,
            "category_by_aspect": category_by_aspect,
            "pool": "added_annotation",
        })
      else:
        unseen_rows.append({
            "protein_id": accession,
            "sequence": record.sequence,
            "known_by_aspect": {"C": set(), "F": set(), "P": set()},
            "novel_by_aspect": novel_by_aspect,
            "category_by_aspect": {a: ("NK" if novel_by_aspect[a] else None) for a in ("C", "F", "P")},
            "pool": "unseen_protein",
        })

  return added_rows, unseen_rows


def balance_pools(added_rows: list, unseen_rows: list, seed: int = 42) -> tuple:
  import random
  if not added_rows or not unseen_rows:
    raise ValueError(
        f"cannot balance: added_annotation={len(added_rows)}, unseen_protein={len(unseen_rows)} "
        "- both pools must be non-empty")
  n = min(len(added_rows), len(unseen_rows))
  rng = random.Random(seed)
  balanced_added = rng.sample(added_rows, n) if len(added_rows) > n else list(added_rows)
  rng2 = random.Random(seed)
  balanced_unseen = rng2.sample(unseen_rows, n) if len(unseen_rows) > n else list(unseen_rows)
  return balanced_added, balanced_unseen


def summarize_categories(added_rows: list, unseen_rows: list) -> dict:
  summary = {"added_annotation": {"C": {}, "F": {}, "P": {}}, "unseen_protein": len(unseen_rows)}
  for row in added_rows:
    for aspect in ("C", "F", "P"):
      category = row["category_by_aspect"][aspect]
      if category is not None:
        summary["added_annotation"][aspect][category] = (
            summary["added_annotation"][aspect].get(category, 0) + 1)
  return summary


def _self_test():
  obonet_graph = obonet.read_obo(GO_OBO_PATH)
  true_path_graph = restrict_to_true_path(obonet_graph)
  from pk_eval_common import build_term_to_aspect
  term_to_aspect = build_term_to_aspect(obonet_graph)

  import datasets
  dataset_full = datasets.load_from_disk(str(DATA_ROOT / "dataset_full"))
  label_names = dataset_full["train"].features["labels"].feature.names
  label2id = {name: i for i, name in enumerate(label_names)}

  t0_index = build_t0_index(TRAIN_TERMS_PATH, true_path_graph, label2id, term_to_aspect)

  train_df = pd.read_csv(TRAIN_TERMS_PATH, sep="\t")
  n_unique_proteins = train_df["EntryID"].nunique()
  assert len(t0_index) == n_unique_proteins, (len(t0_index), n_unique_proteins)

  # Q5W0B1 is train_terms.tsv's first protein (GO:0000785 C, GO:0004842 F, GO:0051865 P,
  # GO:0006275 P per the raw file) - expect all three aspects non-empty after propagation.
  q5 = t0_index["Q5W0B1"]
  assert q5["C"], "expected non-empty C bucket for Q5W0B1"
  assert q5["F"], "expected non-empty F bucket for Q5W0B1"
  assert q5["P"], "expected non-empty P bucket for Q5W0B1"

  print(f"[OK] build_t0_index self-test passed ({len(t0_index)} proteins)")

  # --- categorize_protein: all four outcomes, hand-constructed ---
  has_any_known = {"C": {"GO:x"}, "F": set(), "P": set()}
  result = categorize_protein(has_any_known, {"C": set(), "F": {"GO:novel1"}, "P": set()})
  assert result["C"] is None, result  # no novel term in C -> no example
  assert result["F"] == "LK", result  # empty known_F, but known_C is non-empty -> LK
  assert result["P"] is None, result

  no_known_at_all = {"C": set(), "F": set(), "P": set()}
  result2 = categorize_protein(no_known_at_all, {"C": {"GO:novel2"}, "F": set(), "P": set()})
  assert result2["C"] == "NK", result2

  known_in_same_aspect = {"C": set(), "F": {"GO:known1"}, "P": set()}
  result3 = categorize_protein(known_in_same_aspect, {"C": set(), "F": {"GO:novel3"}, "P": set()})
  assert result3["F"] == "PK", result3

  print("[OK] categorize_protein self-test passed")

  added_rows, unseen_rows = stream_diff(
      SWISSPROT_DUMP_PATH, t0_index, true_path_graph, label2id, term_to_aspect,
      record_limit=3000)
  assert len(added_rows) > 0, "expected at least one added_annotation row in first 3000 records"
  assert len(unseen_rows) > 0, "expected at least one unseen_protein row in first 3000 records"
  valid_categories = {"PK", "LK", "NK", None}
  for row in added_rows:
    assert set(row["category_by_aspect"].values()) <= valid_categories
    assert row["pool"] == "added_annotation"
    assert any(row["novel_by_aspect"].values()), "added row must have >=1 novel term somewhere"
  for row in unseen_rows:
    assert row["pool"] == "unseen_protein"
    assert all(not s for s in row["known_by_aspect"].values()), "unseen rows must have empty known"
    assert any(row["novel_by_aspect"].values())
  print(f"[OK] stream_diff smoke test passed ({len(added_rows)} added, {len(unseen_rows)} unseen, first 3000 records)")

  # --- balance_pools: synthetic unequal pools ---
  fake_added = [{"protein_id": f"A{i}"} for i in range(7)]
  fake_unseen = [{"protein_id": f"U{i}"} for i in range(3)]
  balanced_added, balanced_unseen = balance_pools(fake_added, fake_unseen, seed=42)
  assert len(balanced_added) == 3 and len(balanced_unseen) == 3
  balanced_added_2, _ = balance_pools(fake_added, fake_unseen, seed=42)
  assert [r["protein_id"] for r in balanced_added] == [r["protein_id"] for r in balanced_added_2], (
      "same seed must reproduce the same subsample")

  empty_unseen_should_fail = False
  try:
    balance_pools(fake_added, [], seed=42)
  except ValueError:
    empty_unseen_should_fail = True
  assert empty_unseen_should_fail, "balancing against an empty pool must raise, not silently produce 0 rows"

  summary = summarize_categories(added_rows, unseen_rows)
  assert summary["unseen_protein"] == len(unseen_rows)
  assert set(summary["added_annotation"].keys()) <= {"C", "F", "P"}
  print("[OK] balance_pools/summarize_categories self-test passed")


if __name__ == "__main__":
  _self_test()
