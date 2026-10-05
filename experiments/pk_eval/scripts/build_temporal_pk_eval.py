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


if __name__ == "__main__":
  _self_test()
