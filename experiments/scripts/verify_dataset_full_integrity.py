"""Standalone integrity checks for data/cafa-6-protein-function-prediction/dataset_full,
run after generate_dataset_full.py's main(). No pytest in this repo -- see
verify_select_full_terms.py's header for why this is a plain assertion script.
"""
import sys
from pathlib import Path

import datasets
import obonet
import pandas as pd
from Bio import SeqIO

# Add the project root to sys.path so we can import split_utils (same fix
# verify_select_full_terms.py applies, for the same reason: this script lives in
# experiments/scripts/, not the repo root).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate_dataset import compute_depths, propagate_protein_terms, restrict_to_true_path
from generate_dataset_full import select_full_terms
from split_utils import compute_cluster_aware_split

DATA_PATH = "data/cafa-6-protein-function-prediction"


def test_split_matches_existing_variants():
  """dataset_full's train/val/test protein IDs must be IDENTICAL to dataset/dataset_raw's
  (same compute_cluster_aware_split(seed=42) call over the same protein universe) -- a
  mismatch would make future results on this variant incomparable to existing baselines.
  """
  train_df = pd.read_csv(f"{DATA_PATH}/Train/train_terms.tsv", delimiter="\t")
  unique_protein_ids = list(train_df["EntryID"].unique())
  expected_split = compute_cluster_aware_split(unique_protein_ids)

  dataset_full = datasets.load_from_disk(f"{DATA_PATH}/dataset_full")
  for split_name in ("train", "validation", "test"):
    actual_ids = set(dataset_full[split_name]["protein"])
    expected_ids = {pid for pid, split in expected_split.items() if split == split_name}
    unexpected = actual_ids - expected_ids
    assert not unexpected, (
        f"{split_name}: {len(unexpected)} protein IDs don't match the expected cluster-aware "
        f"split, e.g. {list(unexpected)[:5]}")
  print("[OK] test_split_matches_existing_variants")


def test_full_label_space_matches_disk():
  """Combines two checks that both need the same from-scratch recomputation of
  propagated_terms/full_terms, and both need to validate the REAL on-disk dataset_full
  artifact directly rather than just the recomputation in isolation (a from-scratch-only
  check can only ever catch select_full_terms regressing, never a bug in main()'s dataset
  assembly/save/push path):

  1. "no labels removed silently" -- generate_dataset()'s 'no labels in vocab' safety net
     should remove exactly zero proteins here -- every protein propagates to at least an
     aspect root, and select_full_terms includes every term that appears in ANY protein's
     propagated closure (no threshold), so by construction protein_to_labels[pid] ==
     propagated_terms[pid] whenever the latter is non-empty. A nonzero count here would mean
     that invariant broke upstream.

     generate_dataset() reports removals as a single combined counter (sequence
     missing/too-long OR no labels in vocab), which conflates that invariant with an entirely
     separate, expected baseline: the independent max_length=1024 sequence-length truncation
     shared by every generate_dataset*.py variant (out of this task's scope to change --
     generate_dataset.py is shared code). An earlier version of this check asserted on the
     combined counter directly (removed_fraction < 0.05) and failed at the real 9.25% figure
     -- investigated and traced entirely to the truncation baseline: dataset_full's row counts
     (47766/13513/13502) are numerically IDENTICAL to the already-shipped dataset (383-term
     coarse) variant's, despite a completely different label vocabulary, and a manual
     decomposition found exactly 0 proteins dropped for "no labels in vocab" specifically.
     This version decomposes the two removal conditions itself (re-deriving
     prot_sequences/propagated_terms/full_terms the same way main() does) so the assertion
     targets only the failure mode it's meant to catch.

  2. "on-disk artifact matches the recomputed surviving set, per split, exactly" -- every
     surviving protein_id (computed independently of the on-disk dataset) is bucketed by its
     EXPECTED split (compute_cluster_aware_split, the same call main() makes) and then checked
     for exact set equality against dataset_full[split]["protein"]. This is strictly stronger
     than a one-sided subset check: it catches a protein missing from a split (silently
     dropped or reassigned) just as readily as it catches an unexpected extra one, and it
     would catch a split ending up empty (survivors[split] would be non-empty while
     actual_ids was empty, so the equality fails). It also ties the on-disk ClassLabel
     vocabulary (both size and exact name set) back to the real, freshly-recomputed
     full_terms selection, closing the gap where "larger than 668" was true but not
     necessarily EXACTLY the selected vocabulary.
  """
  data_path_obj = Path(DATA_PATH)
  prot_sequences = {}
  for sequence in SeqIO.parse(data_path_obj / "Train/train_sequences.fasta", "fasta"):
    prot_sequences[sequence.id.split("|")[1]] = str(sequence.seq).upper().replace(r"[UZOB]", "X")
  train_df = pd.read_csv(f"{DATA_PATH}/Train/train_terms.tsv", delimiter="\t")
  obonet_graph = obonet.read_obo(data_path_obj / "Train/go-basic.obo")

  true_path_graph = restrict_to_true_path(obonet_graph)
  depths = compute_depths(true_path_graph)
  propagated_terms = propagate_protein_terms(train_df, true_path_graph)
  full_terms = select_full_terms(propagated_terms, depths)

  unique_protein_ids = list(train_df["EntryID"].unique())
  expected_split = compute_cluster_aware_split(unique_protein_ids)

  too_long_or_missing_seq = 0
  no_labels_in_vocab = 0
  survivors = {"train": set(), "validation": set(), "test": set()}
  for protein_id in unique_protein_ids:
    seq = prot_sequences.get(protein_id)
    if seq is None or len(seq) > 1024:
      too_long_or_missing_seq += 1
      continue
    labels = [term for term in propagated_terms.get(protein_id, []) if term in full_terms]
    if not labels:
      no_labels_in_vocab += 1
      continue
    survivors[expected_split[protein_id]].add(protein_id)

  print(f"[INFO] removed - too-long/missing sequence (independent baseline, not asserted "
        f"on): {too_long_or_missing_seq}")
  print(f"[INFO] removed - no labels in vocab: {no_labels_in_vocab}")
  assert no_labels_in_vocab == 0, (
      f"{no_labels_in_vocab} proteins had zero in-vocabulary labels despite a no-threshold "
      "full label space -- every protein should propagate to at least an aspect root, "
      "investigate before trusting this dataset")

  dataset_full = datasets.load_from_disk(f"{DATA_PATH}/dataset_full")
  for split_name in ("train", "validation", "test"):
    actual_ids = set(dataset_full[split_name]["protein"])
    expected_ids = survivors[split_name]
    missing = expected_ids - actual_ids
    unexpected = actual_ids - expected_ids
    assert actual_ids == expected_ids, (
        f"{split_name}: on-disk dataset_full protein IDs don't exactly match the "
        f"recomputed surviving set -- {len(missing)} expected proteins missing from disk "
        f"(e.g. {list(missing)[:5]}), {len(unexpected)} unexpected proteins on disk (e.g. "
        f"{list(unexpected)[:5]}) -- dataset_full may have silently lost, gained, or "
        "misassigned rows relative to select_full_terms/compute_cluster_aware_split")

  vocab_feature = dataset_full["train"].features["labels"].feature
  assert vocab_feature.num_classes == len(full_terms), (
      f"on-disk vocabulary ({vocab_feature.num_classes} terms) does not exactly equal the "
      f"freshly recomputed full_terms selection ({len(full_terms)} terms) -- select_full_terms "
      "may be filtering, or the on-disk dataset is stale relative to the current pipeline")
  assert vocab_feature.names == sorted(full_terms.keys()), (
      "on-disk ClassLabel names do not exactly equal sorted(full_terms.keys()) -- the "
      "artifact's label ordering/content has drifted from the real selection")

  print(f"[OK] test_full_label_space_matches_disk (0 proteins removed for missing vocab "
        f"labels; {too_long_or_missing_seq} removed for the separate, expected "
        f"sequence-length baseline; all 3 splits match exactly; vocab size "
        f"{vocab_feature.num_classes} matches full_terms exactly)")


def test_dependency_graph_selection_rule():
  """export_dependency_graph's own docstring warns that a wrong/default selection_rule
  would silently mislabel the exported graph's metadata for anyone reading it later --
  verify main() actually passed this variant's real rule, not a copy-pasted wrong one.
  """
  import json
  payload = json.loads(
      open(f"{DATA_PATH}/dataset_full/go_dependency_graph.json").read())
  rule = payload["coarse_selection_rule"]
  assert rule == {"min_annotations": None, "propagated": True, "depth_cap": None}, (
      f"go_dependency_graph.json's selection rule is {rule}, expected the full-label-space "
      "rule (min_annotations=None, propagated=True, depth_cap=None) -- check main()'s "
      "export_dependency_graph call")
  print("[OK] test_dependency_graph_selection_rule")


if __name__ == "__main__":
  test_split_matches_existing_variants()
  test_full_label_space_matches_disk()
  test_dependency_graph_selection_rule()
  print("[OK] all integrity checks passed")
