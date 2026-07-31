"""Aspect-only (3-way C/F/P) GO dataset variant, for training a real aspect predictor
- the first stage a sequential cascade would need (see experiments/sweep_log.md
trial-078: a ground-truth aspect input gave a decisive win over no aspect input on
depth-1 GO-term prediction, but that used an oracle input; this dataset is for
training the model that would actually predict those aspects at inference time).

Label space is just the 3 GO namespace roots (molecular_function, cellular_component,
biological_process - ROOT_TERM_IDS in generate_dataset.py), each protein's label a
3-way multi-hot (most proteins have more than one aspect simultaneously - verified
directly: only ~29% of proteins have exactly one aspect, 43.7% have all three).

Also carries a total_terms column: each protein's RAW (unpropagated) GO term count
across all three aspects combined, from train_terms.tsv directly. This is the
annotation-completeness proxy from the trial-080 investigation (see
experiments/sweep_log.md) - proteins missing a given aspect are systematically
under-annotated everywhere (63-75% of aspect-missing proteins have <=2 total terms,
vs a full-dataset median of 4), consistent with GO's own "Open World Assumption"
(absence of annotation != absence of function; Gaudet & Dessimoz, GO Handbook).
training_run.py's --mask-sparse-negatives-max-total-terms uses this column to exclude
untrustworthy "negative" aspect labels from the loss on sparsely-annotated proteins,
while still trusting every positive label regardless of total_terms (an asserted
positive doesn't depend on how much else is known about the protein).

Reuses generate_dataset.py's GO-graph/propagation logic (restrict_to_true_path,
propagate_protein_terms) rather than duplicating it - generate_dataset.py's own
pipeline is guarded behind `if __name__ == "__main__":`, so importing it here does not
re-run or re-push the original kelmoir/cafa6 dataset.
"""
import random
from pathlib import Path

import datasets
import obonet
import pandas as pd
from Bio import SeqIO
from progressbar import progressbar

from generate_dataset import (
    ROOT_TERM_IDS,
    propagate_protein_terms,
    restrict_to_true_path,
)


def generate_dataset_with_total_terms(protein_ids, protein_to_labels: dict, label2id: dict,
                                      total_terms_by_protein: dict, prot_sequences: dict,
                                      max_length: int = 1024) -> datasets.Dataset:
  features, sequences, label_ids_list, total_terms_list = [], [], [], []
  removed = 0
  for protein_id in progressbar(protein_ids, desc="creating aspect dataset with total_terms"):
    sequence = prot_sequences.get(protein_id)
    if sequence is None or len(sequence) > max_length:
      removed += 1
      continue
    labels = sorted(label2id[term] for term in protein_to_labels.get(protein_id, []) if term in label2id)
    if not labels:
      removed += 1
      continue
    features.append(protein_id)
    sequences.append(sequence)
    label_ids_list.append(labels)
    total_terms_list.append(int(total_terms_by_protein.get(protein_id, 0)))

  print(f"[INFO] Generating aspect dataset with {len(features)} items.")
  print(f"[INFO] Number of proteins removed (too long / no labels in vocab): {removed}")
  label_feature = datasets.ClassLabel(names=sorted(label2id.keys()))
  dataset_dict = {
      "protein": features,
      "labels": label_ids_list,
      "sequence": sequences,
      "total_terms": total_terms_list,
  }
  dataset_features = datasets.Features({
      "protein": datasets.Value("string"),
      "labels": datasets.Sequence(label_feature),
      "sequence": datasets.Value("string"),
      "total_terms": datasets.Value("int64"),
  })
  dataset_info = datasets.DatasetInfo(
      description=(
          "CAFA-6 aspect-only (3-way F/C/P) GO dataset variant, for training a real "
          "aspect predictor. total_terms is each protein's raw (unpropagated) GO term "
          "count across all aspects - an annotation-completeness proxy for masking "
          "untrustworthy negative labels in training_run.py."
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
  propagated_terms = propagate_protein_terms(train_df, true_path_graph)

  # Raw (unpropagated) GO term count per protein, across ALL aspects - the
  # annotation-completeness proxy. Computed from train_terms.tsv directly, not from
  # propagated_terms (propagation would deterministically inflate counts regardless of
  # how much a protein was actually studied, defeating the point of the proxy).
  total_terms_by_protein = train_df.groupby("EntryID").size().to_dict()

  # The 3 roots only - unconditionally included, same convention as
  # generate_dataset.py's select_coarse_terms (which always adds them regardless of
  # the depth/count filter that governs every other coarse term).
  label2id = {term: idx for idx, term in enumerate(sorted(ROOT_TERM_IDS))}
  print(f"[INFO] Aspect-only label space: {label2id}")

  protein_to_labels = {
      protein_id: sorted(terms & ROOT_TERM_IDS.keys())
      for protein_id, terms in propagated_terms.items()
  }

  # Same seed/shuffle/split as generate_dataset.py, applied to the same
  # train_terms.tsv - produces the identical train/val/test protein ID sets as the
  # main kelmoir/cafa6 dataset and kelmoir/cafa6-depth1, just with a different
  # label/feature schema attached.
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
      "train": generate_dataset_with_total_terms(train_ids, protein_to_labels, label2id, total_terms_by_protein, prot_sequences),
      "validation": generate_dataset_with_total_terms(val_ids, protein_to_labels, label2id, total_terms_by_protein, prot_sequences),
      "test": generate_dataset_with_total_terms(test_ids, protein_to_labels, label2id, total_terms_by_protein, prot_sequences),
  })

  out_path = data_path / "dataset_aspect"
  dataset_dict.save_to_disk(out_path)
  dataset_dict.push_to_hub("kelmoir/cafa6-aspect", private=False)

  print("\nAspect-only datasets created successfully.")


if __name__ == "__main__":
  main()
