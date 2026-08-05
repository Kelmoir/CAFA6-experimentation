"""Select the working set for the positional-molecular_function task.

Originally picked only the 5000 "best known" proteins (highest total_terms - the
annotation-completeness proxy validated during the trial-080/081 investigation, see
experiments/sweep_log.md); now selects every protein in train_terms.tsv, tagging each with
the train/val/test split it already belongs to in the main pipeline (generate_dataset.py's
seed=42 shuffle), so this task never disagrees with the rest of the project about which
proteins are train vs. val vs. test. total_terms is still recorded per protein (visualize_protein.py
and downstream review both use it as an at-a-glance annotation-completeness signal), just no
longer used to filter the working set down.

No network access - reads only local Kaggle files already on disk.
"""
import random
from pathlib import Path

import pandas as pd

DATA_PATH = Path("./data/cafa-6-protein-function-prediction")


def compute_split_assignment(unique_protein_ids: list[str]) -> dict[str, str]:
  """Reproduce generate_dataset.py's main() split exactly (same seed, same slicing)."""
  ids = list(unique_protein_ids)
  random.seed(42)
  random.shuffle(ids)

  test_size = int(len(ids) * 0.18)
  train_val_ids = ids[:-test_size]
  test_ids = ids[-test_size:]

  val_size = int(len(train_val_ids) * 0.22)
  train_ids = train_val_ids[:-val_size]
  val_ids = train_val_ids[-val_size:]

  assignment = {}
  assignment.update({pid: "train" for pid in train_ids})
  assignment.update({pid: "validation" for pid in val_ids})
  assignment.update({pid: "test" for pid in test_ids})
  return assignment


def main():
  train_df = pd.read_csv(DATA_PATH / "Train" / "train_terms.tsv", sep="\t")
  total_terms_by_protein = train_df.groupby("EntryID").size().to_dict()

  unique_protein_ids = list(train_df["EntryID"].unique())
  split_assignment = compute_split_assignment(unique_protein_ids)

  by_split = {"train": [], "validation": [], "test": []}
  for pid in unique_protein_ids:
    by_split[split_assignment[pid]].append(pid)

  print(f"Selected {len(unique_protein_ids)} proteins (all of train_terms.tsv)")
  for split, pids in by_split.items():
    print(f"  {split}: {len(pids)}")

  out = {
      "n_proteins": len(unique_protein_ids),
      "selection_rule": "all proteins in train_terms.tsv",
      "split_source": "generate_dataset.py main() seed=42 shuffle, 82/18 then 78/22 split",
      "by_split": by_split,
      "total_terms_by_protein": total_terms_by_protein,
  }
  import json
  out_path = Path("experiments") / "positional_mf_protein_split.json"
  out_path.write_text(json.dumps(out, indent=2))
  print(f"\nWrote {out_path}")


if __name__ == "__main__":
  main()
