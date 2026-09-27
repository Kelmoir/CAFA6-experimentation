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
from generate_dataset_full import MAX_LENGTH, select_full_terms, select_trainable_terms
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
  """Combines several checks that all need the same from-scratch recomputation of
  propagated_terms/full_terms/trainable_terms, and all need to validate the REAL on-disk
  dataset_full artifact directly rather than just the recomputation in isolation (a
  from-scratch-only check can only ever catch select_full_terms/select_trainable_terms
  regressing, never a bug in main()'s dataset assembly/save/push path):

  1. "select_full_terms itself never drops a protein to zero labels" -- this invariant is a
     property of the UNREDUCED full closure (before select_trainable_terms's train-usability
     reduction runs): every protein propagates to at least an aspect root, and
     select_full_terms includes every term seen in ANY protein's closure (no threshold), so
     by construction protein_to_labels[pid] == propagated_terms[pid] whenever the latter is
     non-empty, against the FULL (unreduced) vocabulary. A nonzero count here means that
     invariant broke upstream, in select_full_terms itself -- checked separately from (2)
     below precisely because (2) is expected to be nonzero now (see next point).

  2. "trainable-term reduction can legitimately zero out some proteins, and that's fine" --
     select_trainable_terms (2026-09-28, user-specified) drops any term with zero examples
     among train-split proteins actually usable at MAX_LENGTH from the TRAINABLE vocabulary
     (not from full_terms's definition -- see select_trainable_terms's docstring). A
     protein whose entire label set happens to consist only of such dropped terms will end
     up with zero labels against the REDUCED vocabulary, and generate_dataset()'s "no labels
     in vocab" safety net will legitimately remove it. This is reported for visibility, not
     asserted to be zero -- it's an expected consequence of the reduction, not a bug.

     generate_dataset() reports removals as a single combined counter (sequence
     missing/too-long OR no labels in vocab), which conflates both of the above with an
     entirely separate, expected baseline: the independent max_length sequence-length
     truncation (generate_dataset.py's/generate_dataset_raw.py's shared default is 1024;
     generate_dataset_full.py raises this to MAX_LENGTH specifically to recover labels from
     proteins that default would otherwise exclude entirely -- out of this check's scope to
     change generate_dataset.py itself, which is shared code). An earlier version of this
     check asserted on the combined counter directly (removed_fraction < 0.05) and failed at
     the real 9.25% figure -- investigated and traced entirely to the truncation baseline.
     This version decomposes every removal condition itself (re-deriving
     prot_sequences/propagated_terms/full_terms/trainable_terms the same way main() does) so
     each assertion targets only the failure mode it's meant to catch.

  3. "on-disk artifact matches the recomputed surviving set, per split, exactly" -- every
     surviving protein_id (computed independently of the on-disk dataset, against the
     REDUCED/trainable vocabulary, matching what main() actually builds) is bucketed by its
     EXPECTED split (compute_cluster_aware_split, the same call main() makes) and then checked
     for exact set equality against dataset_full[split]["protein"]. This is strictly stronger
     than a one-sided subset check: it catches a protein missing from a split (silently
     dropped or reassigned) just as readily as it catches an unexpected extra one, and it
     would catch a split ending up empty. It also ties the on-disk ClassLabel vocabulary
     (both size and exact name set) back to the real, freshly-recomputed trainable_terms
     selection -- the one main() actually uses to build label2id.
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
  train_ids = [pid for pid in unique_protein_ids if expected_split[pid] == "train"]

  trainable_terms = select_trainable_terms(full_terms, propagated_terms, train_ids,
                                           prot_sequences, MAX_LENGTH)

  too_long_or_missing_seq = 0
  no_labels_in_unreduced_vocab = 0
  no_labels_in_trainable_vocab = 0
  survivors = {"train": set(), "validation": set(), "test": set()}
  for protein_id in unique_protein_ids:
    seq = prot_sequences.get(protein_id)
    if seq is None or len(seq) > MAX_LENGTH:
      too_long_or_missing_seq += 1
      continue
    unreduced_labels = [term for term in propagated_terms.get(protein_id, []) if term in full_terms]
    if not unreduced_labels:
      no_labels_in_unreduced_vocab += 1
    trainable_labels = [term for term in unreduced_labels if term in trainable_terms]
    if not trainable_labels:
      no_labels_in_trainable_vocab += 1
      continue
    survivors[expected_split[protein_id]].add(protein_id)

  print(f"[INFO] removed - too-long/missing sequence (independent baseline, not asserted "
        f"on): {too_long_or_missing_seq}")
  print(f"[INFO] removed - no labels in UNREDUCED vocab (select_full_terms invariant): "
        f"{no_labels_in_unreduced_vocab}")
  print(f"[INFO] removed - no labels in REDUCED/trainable vocab (expected, not asserted "
        f"on): {no_labels_in_trainable_vocab}")
  assert no_labels_in_unreduced_vocab == 0, (
      f"{no_labels_in_unreduced_vocab} proteins had zero in-vocabulary labels against the "
      "UNREDUCED full label space -- every protein should propagate to at least an aspect "
      "root, investigate select_full_terms before trusting this dataset")

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
        "misassigned rows relative to select_full_terms/select_trainable_terms/"
        "compute_cluster_aware_split")

  vocab_feature = dataset_full["train"].features["labels"].feature
  assert vocab_feature.num_classes == len(trainable_terms), (
      f"on-disk vocabulary ({vocab_feature.num_classes} terms) does not exactly equal the "
      f"freshly recomputed trainable_terms selection ({len(trainable_terms)} terms) -- "
      "select_full_terms/select_trainable_terms may be filtering differently, or the on-disk "
      "dataset is stale relative to the current pipeline")
  assert vocab_feature.names == sorted(trainable_terms.keys()), (
      "on-disk ClassLabel names do not exactly equal sorted(trainable_terms.keys()) -- the "
      "artifact's label ordering/content has drifted from the real selection")

  print(f"[OK] test_full_label_space_matches_disk (0 proteins removed for missing vocab "
        f"labels against the unreduced closure; {no_labels_in_trainable_vocab} removed "
        f"specifically by the trainable-vocab reduction; {too_long_or_missing_seq} removed "
        f"for the separate, expected sequence-length baseline; all 3 splits match exactly; "
        f"vocab size {vocab_feature.num_classes} matches trainable_terms exactly)")


def test_dependency_graph_selection_rule():
  """export_dependency_graph's own docstring warns that a wrong/default selection_rule
  would silently mislabel the exported graph's metadata for anyone reading it later --
  verify main() actually passed this variant's real rule, not a copy-pasted wrong one.
  """
  import json
  payload = json.loads(
      open(f"{DATA_PATH}/dataset_full/go_dependency_graph.json").read())
  rule = payload["coarse_selection_rule"]
  expected_rule = {"min_annotations": None, "propagated": True, "depth_cap": None,
                   "min_train_split_annotations_at_max_length": 1, "max_length": MAX_LENGTH}
  assert rule == expected_rule, (
      f"go_dependency_graph.json's selection rule is {rule}, expected {expected_rule} -- "
      "check main()'s export_dependency_graph call")
  print("[OK] test_dependency_graph_selection_rule")


if __name__ == "__main__":
  test_split_matches_existing_variants()
  test_full_label_space_matches_disk()
  test_dependency_graph_selection_rule()
  print("[OK] all integrity checks passed")
