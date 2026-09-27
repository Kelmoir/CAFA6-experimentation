"""Full GO-term dataset variant -- every term, no filtering of any kind.

Label vocabulary = every GO term that appears anywhere in any protein's propagated
ancestor closure (is_a/part_of propagation, the same true-path rule generate_dataset.py
already applies). No depth cap. No annotation-count threshold. Nothing is excluded for
being rare -- a term with exactly one supporting protein is in the vocabulary on exactly
the same footing as one with a hundred thousand. This is the full label space required by
the 2026-09-26 decision in experiments/cafa6_next_steps_plan.md ("Scope constraint: full
label space, no partial GO-term selection").

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

  label2id = {term: idx for idx, term in enumerate(sorted(full_terms))}

  protein_to_labels = {
      protein_id: sorted(terms & full_terms.keys())
      for protein_id, terms in propagated_terms.items()
  }

  unique_protein_ids = list(train_df["EntryID"].unique())
  split_assignment = compute_cluster_aware_split(unique_protein_ids)
  train_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "train"]
  val_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "validation"]
  test_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "test"]
  print(f"Splitting IDs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

  dataset_dict = datasets.DatasetDict({
      "train": generate_dataset(train_ids, protein_to_labels, label2id, prot_sequences),
      "validation": generate_dataset(val_ids, protein_to_labels, label2id, prot_sequences),
      "test": generate_dataset(test_ids, protein_to_labels, label2id, prot_sequences),
  })

  out_path = data_path / "dataset_full"
  dataset_dict.save_to_disk(out_path)
  export_dependency_graph(obonet_graph, full_terms, out_path / "go_dependency_graph.json",
                          selection_rule={"min_annotations": None, "propagated": True, "depth_cap": None})
  dataset_dict.push_to_hub("kelmoir/cafa6-full", private=False)

  print("\nFull-label-space dataset created successfully.")


if __name__ == "__main__":
  main()
