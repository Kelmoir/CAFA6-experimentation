"""Two-tier discrete relabeling for GO's Open World Assumption problem -- the ORIGINAL PU-learning
design this project's plan was built around before it got restructured onto nnPU (see
pu_learning_plan.md's restructure history), revived after nnPU's training-time loss mechanism
racked up THREE independent failure modes without ever beating plain BCE: bad calibration (KM2,
naive spy mean, centroid ratio all failed; a genuine held-out spy retrain and TIcE both eventually
worked, see estimate_class_priors_tice.py), an anchor-batch/shared-backbone gradient leak (fixed and
verified with a direct gradient-isolation test), and a still-unexplained aggregate AUROC cost that
survived fixing the leak (see project_pu_learning_nnpu_km2.md v1.2-v1.4 for the full trail).

Unlike nnPU (a training-time LOSS change with its own per-step anchor-batch machinery), this is a
purely offline DATA change, with no new training-time code path beyond what already exists:
  1. For each term with a trustworthy TIcE prior (>= --min-source-positives, same floor nnPU used),
     estimate how many of its currently-"unlabeled" (label 0) proteins are likely hidden true
     positives: n_hidden_hat = clip(round(pi_tice * N_total) - n_known_positives, 0, n_unlabeled).
  2. Rank that term's unlabeled proteins by the baseline classifier's own predicted probability
     (protein_probs.npy, from the SAME pretrain-reduced-full-protein-v8 checkpoint TIcE's
     embeddings came from) and mark the top n_hidden_hat (most-likely-secretly-positive) as
     EXCLUDED from the loss for that term -- neither trusted as negative nor asserted as positive.
  3. Everything else -- known positives, the remaining lower-ranked "reliably negative" unlabeled
     proteins, and every non-eligible term -- is completely unmodified: ordinary hard 0/1 BCE,
     weight 1.

Output: adds a sparse `excluded_labels` column (list of term indices to mask out of the loss for
that row) to a copy of dataset_raw, saved to --output-path. `training_run.py --loss-mask-from-dataset`
(new flag, reuses the EXISTING loss_mask/compute_loss machinery built for
--mask-sparse-negatives-max-total-terms) turns this into a per-cell loss_mask at train time.

Usage:
  env/bin/python experiments/scripts/build_two_tier_dataset.py \\
      > experiments/logs/build_two_tier_dataset.out 2>&1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset_raw")
    parser.add_argument("--output-path", default="data/cafa-6-protein-function-prediction/dataset_raw_two_tier")
    parser.add_argument("--embeddings-dir", default="experiments/pu_cache",
                         help="Where protein_probs.npy/protein_meta.json live (extract_protein_embeddings.py output)")
    parser.add_argument("--tice-cache-path", default="experiments/pu_cache/class_priors_tice.json")
    parser.add_argument("--min-source-positives", type=int, default=2000,
                         help="Matches --class-priors-min-source-positives, the floor nnPU used")
    args = parser.parse_args()

    start = time.time()
    embeddings_dir = Path(args.embeddings_dir)
    probs = np.load(embeddings_dir / "protein_probs.npy")
    meta = json.loads((embeddings_dir / "protein_meta.json").read_text())
    label_names = meta["label_names"]
    protein_ids = meta["protein_ids"]
    protein_row = {pid: i for i, pid in enumerate(protein_ids)}
    n_total = len(protein_ids)
    print(f"[INFO] Loaded {n_total} scored proteins, {len(label_names)} labels")

    tice_cache = json.loads(Path(args.tice_cache_path).read_text())["priors"]
    eligible = {t: v for t, v in tice_cache.items() if v.get("n_positives", 0) >= args.min_source_positives}
    print(f"[INFO] {len(eligible)} terms eligible (>= {args.min_source_positives} source positives): {list(eligible.keys())}")

    dataset = load_from_disk(args.data_path)
    label2id = {name: i for i, name in enumerate(label_names)}

    # (split, row_idx) for every embedded protein, and the true hard-label vector, so we know
    # which rows are "unlabeled" for a term and can map back to dataset_raw's own row indices.
    split_of_row = [None] * n_total
    idx_in_split = [None] * n_total
    cursor = 0
    hard_labels = np.zeros((n_total, len(label_names)), dtype=np.float32)
    for split in meta["splits"]:
        rows = dataset[split]
        labels_col = rows["labels"]
        for i, labels in enumerate(labels_col):
            global_idx = cursor + i
            split_of_row[global_idx] = split
            idx_in_split[global_idx] = i
            for t in labels:
                hard_labels[global_idx, int(t)] = 1.0
        cursor += len(rows)
    assert cursor == n_total

    # excluded_by_row[(split, row_idx)] -> list of term indices to mask out of the loss there.
    excluded_by_row = {}
    for term_name, entry in eligible.items():
        term_idx = label2id[term_name]
        pi_hat = entry["pi_tice"]
        n_known = int(hard_labels[:, term_idx].sum())
        unlabeled_rows = np.where(hard_labels[:, term_idx] == 0)[0]
        n_hidden_hat = int(np.clip(round(pi_hat * n_total) - n_known, 0, len(unlabeled_rows)))
        term_probs = probs[unlabeled_rows, term_idx]
        top_order = np.argsort(-term_probs)[:n_hidden_hat]
        excluded_global_rows = unlabeled_rows[top_order]
        for g in excluded_global_rows:
            key = (split_of_row[g], idx_in_split[g])
            excluded_by_row.setdefault(key, []).append(term_idx)
        print(f"[STEP] {term_name}: pi_tice={pi_hat:.4f} n_known={n_known} "
              f"n_hidden_hat={n_hidden_hat}/{len(unlabeled_rows)} unlabeled excluded from loss")

    total_excluded_cells = sum(len(v) for v in excluded_by_row.values())
    print(f"[INFO] {total_excluded_cells} (protein, term) cells excluded from the loss across "
          f"{len(excluded_by_row)} distinct proteins")

    new_splits = {}
    for split in ["train", "validation"]:
        def add_excluded(row, idx, split=split):
            row["excluded_labels"] = excluded_by_row.get((split, idx), [])
            return row

        new_splits[split] = dataset[split].map(add_excluded, with_indices=True)
    new_splits["test"] = dataset["test"].map(lambda row: {**row, "excluded_labels": []})
    new_dataset = DatasetDict(new_splits)

    output_path = Path(args.output_path)
    new_dataset.save_to_disk(str(output_path))
    print(f"[DONE] Wrote two-tier dataset to {output_path} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
