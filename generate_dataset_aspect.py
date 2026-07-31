"""Aspect-only (3-way C/F/P) GO dataset variant, for training a real aspect predictor
- the first stage a sequential cascade would need (see experiments/sweep_log.md
trial-078: a ground-truth aspect input gave a decisive win over no aspect input on
depth-1 GO-term prediction, but that used an oracle input; this dataset is for
training the model that would actually predict those aspects at inference time).

Label space is just the 3 GO namespace roots (molecular_function, cellular_component,
biological_process - ROOT_TERM_IDS in generate_dataset.py), each protein's label a
3-way multi-hot (most proteins have more than one aspect simultaneously - verified
directly: only ~29% of proteins have exactly one aspect, 43.7% have all three).

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

from generate_dataset import (
    ROOT_TERM_IDS,
    generate_dataset,
    propagate_protein_terms,
    restrict_to_true_path,
)


def main():
  data_path = Path("./data/cafa-6-protein-function-prediction")
  prot_sequences = {}
  for sequence in SeqIO.parse(data_path / "Train/train_sequences.fasta", "fasta"):
    prot_sequences[sequence.id.split("|")[1]] = str(sequence.seq).upper().replace(r"[UZOB]", "X")
  train_df = pd.read_csv(data_path / "Train/train_terms.tsv", delimiter="\t")
  obonet_graph = obonet.read_obo(data_path / "Train/go-basic.obo")

  true_path_graph = restrict_to_true_path(obonet_graph)
  propagated_terms = propagate_protein_terms(train_df, true_path_graph)

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
      "train": generate_dataset(train_ids, protein_to_labels, label2id, prot_sequences=prot_sequences),
      "validation": generate_dataset(val_ids, protein_to_labels, label2id, prot_sequences=prot_sequences),
      "test": generate_dataset(test_ids, protein_to_labels, label2id, prot_sequences=prot_sequences),
  })

  out_path = data_path / "dataset_aspect"
  dataset_dict.save_to_disk(out_path)
  dataset_dict.push_to_hub("kelmoir/cafa6-aspect", private=False)

  print("\nAspect-only datasets created successfully.")


if __name__ == "__main__":
  main()
