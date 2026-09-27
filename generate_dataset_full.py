"""Full GO-term dataset variant -- every annotated term, no arbitrary threshold.

Label vocabulary starts as every GO term that appears anywhere in any protein's propagated
ancestor closure (is_a/part_of propagation, the same true-path rule generate_dataset.py
already applies) -- no depth cap, no annotation-count threshold, nothing excluded for being
rare. This is the full label space required by the 2026-09-26 decision in
experiments/cafa6_next_steps_plan.md ("Scope constraint: full label space, no partial
GO-term selection").

One further, principled reduction applies on top (2026-09-28, user-specified): a term with
ZERO examples among train-split proteins actually usable at this variant's max_length cutoff
cannot produce any gradient signal under any architecture, so select_trainable_terms drops
it from the classifier head specifically (not from the label space's definition -- a term
this drops is still real/annotated in the full population, just unlearnable given the
current data and compute budget). This is a data-driven annotation-completeness
distinction, not the arbitrary depth/frequency convenience threshold the 2026-09-26 scope
constraint rules out.

(For context only, not something this variant does: generate_dataset.py's coarse space
additionally restricts to depth<=2 terms with >=50 propagated annotations -- 383 terms --
and generate_dataset_raw.py drops propagation entirely and restricts to >=100
directly-asserted annotations -- 668 terms. This variant applies neither restriction.)

Reuses generate_dataset.py's sequence loading, propagation (restrict_to_true_path,
compute_depths, propagate_protein_terms), generate_dataset() (unmodified schema),
export_dependency_graph(), and split_utils.compute_cluster_aware_split (same
protein-universe split as every other generate_dataset*.py variant, so this dataset's
train/val/test protein IDs match kelmoir/cafa6 and kelmoir/cafa6-raw exactly).
generate_dataset.py's own pipeline is guarded behind `if __name__ == "__main__":`, so
importing it here does not re-run or re-push the original kelmoir/cafa6 dataset.
"""
from collections import Counter
from pathlib import Path

import datasets
import obonet
import pandas as pd
from Bio import SeqIO

from generate_dataset import (
    compute_depths,
    export_dependency_graph,
    generate_dataset,
    propagate_protein_terms,
    restrict_to_true_path,
)
from split_utils import compute_cluster_aware_split

# Sequence-length ceiling for dataset inclusion (generate_dataset()'s max_length filter).
# Raised from generate_dataset.py's/generate_dataset_raw.py's shared 1024 default: those
# proteins aren't just truncated, they're excluded from the dataset ENTIRELY, taking every
# one of their labels with them -- disproportionately hurting rare/singleton terms whose only
# supporting proteins happen to be long. Analysis (2026-09-28): raising the cutoff from 1024
# recovers proteins fast at first, then flattens out hard -- only 191/29,386 vocabulary terms
# (0.65%) are EVER zero-train-positive purely because of a length cutoff (the rest are zero
# regardless of length), and 81% of those 191 are already recovered by ~4096; going past that
# needs including proteins up to 35,213 residues long for a handful more terms, a bad
# cost/benefit trade under quadratic self-attention. 4192 (a user-specified value, not a
# round number chosen for its own sake) sits just past that practical ceiling. See
# experiments/scripts/verify_dataset_full_integrity.py, which imports this same constant so
# the two can't drift out of sync with each other.
MAX_LENGTH = 4192


def select_full_terms(propagated_terms: dict, depths: dict) -> dict:
  """EVERY GO term that appears anywhere in any protein's propagated ancestor closure --
  no depth cap, no annotation-count threshold, no filtering step at all. Unlike
  select_coarse_terms/select_raw_terms, there is no `if count >= min_annotations` (or
  any other) condition here: a term backed by a single protein is included exactly the
  same way as one backed by every protein in the dataset.

  Returns {term_id: {"depth", "count"}}, matching select_coarse_terms's/select_raw_terms's
  shape so export_dependency_graph's payload stays consistent across variants. depth is
  None for a term outside the true-path graph entirely (obsolete/unreachable term IDs that
  can still appear as directly-asserted train_terms.tsv rows and end up in a propagated
  closure even though propagate_protein_terms's ancestors_of() has no edges for them --
  the same edge case generate_dataset_raw.py's select_raw_terms already documents).
  """
  counts = {}
  for terms in propagated_terms.values():
    for term in terms:
      counts[term] = counts.get(term, 0) + 1
  # No threshold, no filter: every key in `counts` -- i.e. every term seen even once --
  # is returned. This dict comprehension only attaches depth metadata; it excludes nothing.
  return {
      term: {"depth": depths.get(term), "count": count}
      for term, count in counts.items()
  }


def select_trainable_terms(full_terms: dict, propagated_terms: dict, train_ids: list,
                           prot_sequences: dict, max_length: int) -> dict:
  """Further restrict full_terms to terms with >=1 supporting protein among the ones the
  model can actually train on: TRAIN split, present, and not excluded by max_length. A term
  can be genuinely annotated somewhere in the full population (so select_full_terms keeps
  it) and still be un-trainable here (zero examples in the *usable train* subset) -- no
  architecture can learn from zero examples, so keeping such a term in the classifier head
  is pure waste (it just converges to "always predict near-0" and dilutes the BCE loss's
  gradient signal to the labels that do have real examples). This does not touch the
  scoring/evaluation label space (that stays the full closure from select_full_terms, per
  the 2026-09-26 "no partial GO-term selection" scope constraint) -- it only decides what
  the model's own classifier head is asked to learn; a dropped term would just be predicted
  at a fixed prior at inference time, which is the best any training could ever do for it.

  Returns a dict with the same {term_id: {"depth", "count"}} shape as full_terms, filtered.
  """
  trainable_protein_ids = {
      pid for pid in train_ids
      if pid in prot_sequences and len(prot_sequences[pid]) <= max_length
  }
  counts = Counter()
  for pid in trainable_protein_ids:
    for term in propagated_terms.get(pid, ()):
      if term in full_terms:
        counts[term] += 1
  return {term: meta for term, meta in full_terms.items() if counts.get(term, 0) >= 1}


def main():
  data_path = Path("./data/cafa-6-protein-function-prediction")
  prot_sequences = {}
  for sequence in SeqIO.parse(data_path / "Train/train_sequences.fasta", "fasta"):
    prot_sequences[sequence.id.split("|")[1]] = str(sequence.seq).upper().replace(r"[UZOB]", "X")
  train_df = pd.read_csv(data_path / "Train/train_terms.tsv", delimiter="\t")
  obonet_graph = obonet.read_obo(data_path / "Train/go-basic.obo")

  true_path_graph = restrict_to_true_path(obonet_graph)
  depths = compute_depths(true_path_graph)
  propagated_terms = propagate_protein_terms(train_df, true_path_graph)
  full_terms = select_full_terms(propagated_terms, depths)
  print(f"[INFO] Selected {len(full_terms)} full GO terms "
        f"(every propagated term, no depth cap, no annotation-count threshold)")

  unique_protein_ids = list(train_df["EntryID"].unique())
  split_assignment = compute_cluster_aware_split(unique_protein_ids)
  train_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "train"]
  val_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "validation"]
  test_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "test"]
  print(f"Splitting IDs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

  trainable_terms = select_trainable_terms(full_terms, propagated_terms, train_ids,
                                           prot_sequences, MAX_LENGTH)
  print(f"[INFO] Reduced vocabulary from {len(full_terms)} to {len(trainable_terms)} terms "
        f"with >=1 train-split annotation at max_length={MAX_LENGTH} (dropped "
        f"{len(full_terms) - len(trainable_terms)} terms with zero train-usable support)")
  full_terms = trainable_terms

  label2id = {term: idx for idx, term in enumerate(sorted(full_terms))}

  protein_to_labels = {
      protein_id: sorted(terms & full_terms.keys())
      for protein_id, terms in propagated_terms.items()
  }

  dataset_dict = datasets.DatasetDict({
      "train": generate_dataset(train_ids, protein_to_labels, label2id, prot_sequences, max_length=MAX_LENGTH),
      "validation": generate_dataset(val_ids, protein_to_labels, label2id, prot_sequences, max_length=MAX_LENGTH),
      "test": generate_dataset(test_ids, protein_to_labels, label2id, prot_sequences, max_length=MAX_LENGTH),
  })

  out_path = data_path / "dataset_full"
  dataset_dict.save_to_disk(out_path)
  export_dependency_graph(obonet_graph, full_terms, out_path / "go_dependency_graph.json",
                          selection_rule={"min_annotations": None, "propagated": True, "depth_cap": None,
                                          "min_train_split_annotations_at_max_length": 1,
                                          "max_length": MAX_LENGTH})
  dataset_dict.push_to_hub("kelmoir/cafa6-full", private=False)

  print("\nFull-label-space dataset created successfully.")


if __name__ == "__main__":
  main()
