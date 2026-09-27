"""Standalone integrity checks for data/cafa-6-protein-function-prediction/dataset_full,
run after generate_dataset_full.py's main(). No pytest in this repo -- see
verify_select_full_terms.py's header for why this is a plain assertion script.
"""
import sys
from pathlib import Path

import datasets
import pandas as pd

# Add the project root to sys.path so we can import split_utils (same fix
# verify_select_full_terms.py applies, for the same reason: this script lives in
# experiments/scripts/, not the repo root).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


def test_vocabulary_larger_than_existing_variants():
  """The full closure (no threshold) must be a strict superset in spirit of the existing,
  thresholded variants -- at minimum, far larger than dataset_raw's 668 terms.
  """
  dataset_full = datasets.load_from_disk(f"{DATA_PATH}/dataset_full")
  vocab_size = dataset_full["train"].features["labels"].feature.num_classes
  assert vocab_size > 668, (
      f"full-label-space vocabulary ({vocab_size} terms) is not larger than dataset_raw's "
      "668-term thresholded space -- select_full_terms may be filtering when it shouldn't")
  print(f"[OK] test_vocabulary_larger_than_existing_variants ({vocab_size} terms)")


def test_no_labels_removed_silently():
  """generate_dataset()'s 'no labels in vocab' safety net should remove close to zero
  proteins here (every protein propagates to at least an aspect root, which is always in
  a no-threshold vocabulary) -- a large removed count would mean something upstream broke.
  """
  dataset_full = datasets.load_from_disk(f"{DATA_PATH}/dataset_full")
  train_df = pd.read_csv(f"{DATA_PATH}/Train/train_terms.tsv", delimiter="\t")
  n_input_proteins = train_df["EntryID"].nunique()
  n_output_proteins = sum(len(dataset_full[s]) for s in ("train", "validation", "test"))
  removed_fraction = 1 - (n_output_proteins / n_input_proteins)
  assert removed_fraction < 0.05, (
      f"{removed_fraction:.1%} of proteins were dropped (too-long sequence or no labels) -- "
      "expected close to 0% for a no-threshold vocabulary, investigate before trusting this dataset")
  print(f"[OK] test_no_labels_removed_silently ({removed_fraction:.2%} dropped)")


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
  test_vocabulary_larger_than_existing_variants()
  test_no_labels_removed_silently()
  test_dependency_graph_selection_rule()
  print("[OK] all integrity checks passed")
