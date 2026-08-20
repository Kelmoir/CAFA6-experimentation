"""Select the working set for the positional-molecular_function task.

Originally picked only the 5000 "best known" proteins (highest total_terms - the
annotation-completeness proxy validated during the trial-080/081 investigation, see
experiments/sweep_log.md); now selects every protein in train_terms.tsv, tagging each with
the train/val/test split it already belongs to in the main pipeline (generate_dataset.py's
cluster-aware split -- see split_utils.py), so this task never disagrees with the rest of the
project about which proteins are train vs. val vs. test. total_terms is still recorded per
protein (visualize_protein.py and downstream review both use it as an at-a-glance
annotation-completeness signal), just no longer used to filter the working set down.

No network access - reads only local Kaggle files already on disk.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from split_utils import compute_cluster_aware_split

DATA_PATH = Path("./data/cafa-6-protein-function-prediction")


def main():
  train_df = pd.read_csv(DATA_PATH / "Train" / "train_terms.tsv", sep="\t")
  total_terms_by_protein = train_df.groupby("EntryID").size().to_dict()

  unique_protein_ids = list(train_df["EntryID"].unique())
  split_assignment = compute_cluster_aware_split(unique_protein_ids)

  by_split = {"train": [], "validation": [], "test": []}
  for pid in unique_protein_ids:
    by_split[split_assignment[pid]].append(pid)

  print(f"Selected {len(unique_protein_ids)} proteins (all of train_terms.tsv)")
  for split, pids in by_split.items():
    print(f"  {split}: {len(pids)}")

  out = {
      "n_proteins": len(unique_protein_ids),
      "selection_rule": "all proteins in train_terms.tsv",
      "split_source": "generate_dataset.py main() cluster-aware split (split_utils.py), seed=42, "
                      "82/18 then 78/22 target proportions, CD-HIT clusters kept intact",
      "by_split": by_split,
      "total_terms_by_protein": total_terms_by_protein,
  }
  import json
  out_path = Path("experiments") / "positional_mf_protein_split.json"
  out_path.write_text(json.dumps(out, indent=2))
  print(f"\nWrote {out_path}")


if __name__ == "__main__":
  main()
