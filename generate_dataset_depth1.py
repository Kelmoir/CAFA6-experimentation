"""Depth-1-only GO-term dataset variant for the aspect-conditioning ablation
(trial-077: no aspect input, trial-078: ground-truth aspect input to the classifier -
see experiments/ancestor_sweep/sweep_log.md). Label space is restricted to the 39 depth-1 GO terms
(no depth-0 aspect roots, no depth-2+ terms) instead of generate_dataset.py's full
383-term coarse space, and each example additionally carries a 3-way aspect_labels
multi-hot column ([C, F, P] - 1 where that aspect root is in the protein's true-path
ancestor closure) as an optional model INPUT feature, not a prediction target.

Reuses generate_dataset.py's GO-graph/propagation logic (restrict_to_true_path,
compute_depths, propagate_protein_terms) rather than duplicating it - only the label
selection (depth==1 only) and the emitted dataset schema differ. generate_dataset.py's
own pipeline is guarded behind `if __name__ == "__main__":`, so importing it here does
not re-run or re-push the original kelmoir/cafa6 dataset.
"""
import random
from pathlib import Path

import datasets
import obonet
import pandas as pd
from Bio import SeqIO
from progressbar import progressbar

from generate_dataset import (
    COARSE_MIN_ANNOTATIONS,
    ROOT_TERM_IDS,
    compute_depths,
    export_dependency_graph,
    propagate_protein_terms,
    restrict_to_true_path,
)

DEPTH1 = 1
# [C, F, P] - alphabetical on the aspect letter, fixed so the multi-hot column's
# column order is unambiguous downstream (training_run.py must agree on this order).
ASPECT_ORDER = sorted(ROOT_TERM_IDS.items(), key=lambda kv: kv[1])  # [(GO:0005575,C), (GO:0003674,F), (GO:0008150,P)]


def select_depth1_terms(propagated_terms: dict, depths: dict,
                        min_annotations: int = COARSE_MIN_ANNOTATIONS) -> dict:
  """Depth-1-only counterpart to generate_dataset.py's select_coarse_terms - same
  propagation/threshold logic, restricted to depth==1 (not 1<=depth<=2), and the 3
  depth-0 roots are deliberately NOT included (they're exposed as aspect_labels
  instead, an input feature rather than a prediction target for this variant).
  """
  candidate_terms = {term for term, depth in depths.items() if depth == DEPTH1}
  counts = {}
  for terms in propagated_terms.values():
    for term in terms & candidate_terms:
      counts[term] = counts.get(term, 0) + 1
  return {
      term: {"depth": DEPTH1, "count": count}
      for term, count in counts.items()
      if count >= min_annotations
  }


def generate_dataset_with_aspect(protein_ids, propagated_terms: dict, label2id: dict,
                                 prot_sequences: dict, max_length: int = 1024) -> datasets.Dataset:
  features, sequences, label_ids_list, aspect_multihot = [], [], [], []
  removed = 0
  for protein_id in progressbar(protein_ids, desc="creating depth-1 dataset with aspect labels"):
    sequence = prot_sequences.get(protein_id)
    if sequence is None or len(sequence) > max_length:
      removed += 1
      continue
    closure = propagated_terms.get(protein_id, set())
    labels = sorted(label2id[term] for term in closure if term in label2id)
    if not labels:
      # A protein with any depth-2 (or deeper) annotation necessarily has a depth-1
      # ancestor on its shortest path to the root, so this is expected to be as rare
      # as generate_dataset.py's own equivalent safety net, not a normal path.
      removed += 1
      continue
    aspect_vec = [1 if root_id in closure else 0 for root_id, _aspect in ASPECT_ORDER]
    features.append(protein_id)
    sequences.append(sequence)
    label_ids_list.append(labels)
    aspect_multihot.append(aspect_vec)

  print(f"[INFO] Generating depth-1 dataset with {len(features)} items.")
  print(f"[INFO] Number of proteins removed (too long / no depth-1 labels in vocab): {removed}")
  label_feature = datasets.ClassLabel(names=sorted(label2id.keys()))
  dataset_dict = {
      "protein": features,
      "labels": label_ids_list,
      "sequence": sequences,
      "aspect_labels": aspect_multihot,
  }
  dataset_features = datasets.Features({
      "protein": datasets.Value("string"),
      "labels": datasets.Sequence(label_feature),
      "sequence": datasets.Value("string"),
      "aspect_labels": datasets.Sequence(datasets.Value("int64"), length=3),
  })
  dataset_info = datasets.DatasetInfo(
      description=(
          "CAFA-6 depth-1-only GO-term variant, for the aspect-conditioning ablation "
          "(see experiments/ancestor_sweep/sweep_log.md trial-077/trial-078). aspect_labels is a "
          "3-way multi-hot [C, F, P] derived from the ground-truth true-path ancestor "
          "closure - intended as an optional model INPUT feature, not a prediction "
          "target."
      ),
      citation=(
          "Iddo Friedberg, Predrag Radivojac, Paul D Thomas, An Phan, M. Clara De Paolis Kaluza, "
          "Damiano Piovesan, Parnal Joshi, Chris Mungall, Martyna Plomecka, Walter Reade, and "
          "María Cruz. CAFA 6 Protein Function Prediction. "
          "https://kaggle.com/competitions/cafa-6-protein-function-prediction, 2025. Kaggle."
      ),
      features=dataset_features,
  )
  return datasets.Dataset.from_dict(dataset_dict, dataset_features, info=dataset_info)


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
  depth1_terms = select_depth1_terms(propagated_terms, depths)
  print(f"[INFO] Selected {len(depth1_terms)} depth-1 GO terms "
        f"(>={COARSE_MIN_ANNOTATIONS} annotations after propagation, no depth-0 roots, no depth-2+ terms)")

  label2id = {term: idx for idx, term in enumerate(sorted(depth1_terms))}

  # Same seed/shuffle/split code as generate_dataset.py, applied to the same
  # train_terms.tsv - produces the identical train/val/test protein ID sets as the
  # main kelmoir/cafa6 dataset, just with a different label/feature schema attached.
  unique_protein_ids = list(train_df["EntryID"].unique())
  random.seed(42)
  random.shuffle(unique_protein_ids)
  test_size = int(len(unique_protein_ids) * 0.18)
  train_val_ids = unique_protein_ids[:-test_size]
  test_ids = unique_protein_ids[-test_size:]
  val_size = int(len(train_val_ids) * 0.22)
  train_ids = train_val_ids[:-val_size]
  val_ids = train_val_ids[-val_size:]
  print(f"Splitting IDs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

  dataset_dict = datasets.DatasetDict({
      "train": generate_dataset_with_aspect(train_ids, propagated_terms, label2id, prot_sequences),
      "validation": generate_dataset_with_aspect(val_ids, propagated_terms, label2id, prot_sequences),
      "test": generate_dataset_with_aspect(test_ids, propagated_terms, label2id, prot_sequences),
  })

  out_path = data_path / "dataset_depth1"
  dataset_dict.save_to_disk(out_path)
  export_dependency_graph(obonet_graph, depth1_terms, out_path / "go_dependency_graph.json",
                          selection_rule={"depth": DEPTH1, "min_annotations": COARSE_MIN_ANNOTATIONS})
  dataset_dict.push_to_hub("kelmoir/cafa6-depth1", private=False)

  print("\nDepth-1 datasets created successfully.")


if __name__ == "__main__":
  main()
