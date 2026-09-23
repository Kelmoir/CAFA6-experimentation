"""Build a spy-holdout variant of `dataset_raw` for a genuine held-out Elkan-Noto spy retrain.

Why this exists: every prior-estimator attempt so far (KM2, classifier-probability spy mean,
centroid ratio -- raw and mean-centered) has been checked against true per-term prevalence and
found unusable (see project_pu_learning_nnpu_km2.md v1.1/v1.2 and pu_learning_plan.md). The
classifier-probability spy mean's specific failure mode (median 14.4x overestimation) was traced to
the SAME checkpoint being scored that was trained on those exact "spy" positives -- Elkan-Noto's
method requires E[g(x) | y=1] to be estimated from a classifier that has genuinely never seen those
positives labeled as positive, which reusing `pretrain-reduced-full-protein-v8` cannot provide.

This script builds the training data for that real retrain: for every term with
>= --min-term-positives positives (across train+validation, matching
estimate_class_priors_km2.py's convention), it randomly selects --spy-fraction of that term's
positive rows (seeded, reproducible) and removes ONLY that term's label from those rows (all other
labels on the same row are untouched -- a row can be a spy for one term and a normal positive for
another). The 'test' split is passed through unmodified (never used for training or extraction).

A model trained on the resulting dataset has, for every eligible term, a genuinely-held-out set of
known positives it was trained to treat as negative/unknown. Scoring that model on exactly those
held-out rows (via extract_protein_embeddings.py against a --checkpoint-path pointed at the new
run, same 668-term label space) gives an unbiased Elkan-Noto c = E[g(x) | y=1] for the ratio
estimator pi_hat = E[g(x) | unlabeled] / c -- see estimate_class_priors_spy_heldout.py, which reads
this script's spy_holdout_meta.json to know exactly which (protein, term) pairs to treat as the
held-out positive set for each term, and to exclude from the "unlabeled" set for the ratio's
denominator.

IMPORTANT: any retrain on this dataset must start from a FRESH backbone init (the plain HF
checkpoint, e.g. facebook/esm2_t12_35M_UR50D), never from --init-from-checkpoint/
--resume-from-revision pointed at pretrain-reduced-full-protein-v8 or any checkpoint already
fine-tuned on the unmodified labels -- doing so would carry the contamination straight back into
the retrain's weights even though the labels themselves are held out at the data level.

Usage:
  env/bin/python experiments/scripts/build_spy_holdout_dataset.py \\
      > experiments/logs/build_spy_holdout_dataset.out 2>&1
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
    parser.add_argument("--output-path", default="data/cafa-6-protein-function-prediction/dataset_raw_spy_holdout")
    parser.add_argument("--min-term-positives", type=int, default=100,
                         help="Matches estimate_class_priors_km2.py/estimate_class_priors_centroid.py's default")
    parser.add_argument("--spy-fraction", type=float, default=0.2, help="Matches the km2 script's spy fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meta-output-path", default="experiments/pu_cache/spy_holdout_meta.json")
    args = parser.parse_args()

    start = time.time()
    dataset = load_from_disk(args.data_path)
    train_val_splits = ["train", "validation"]
    label_names = dataset["train"].features["labels"].feature.names
    n_labels = len(label_names)

    # (split, row_idx) -> list of protein-term positives, flattened for eligibility counting.
    split_labels = {split: dataset[split]["labels"] for split in train_val_splits}
    split_proteins = {split: dataset[split]["protein"] for split in train_val_splits}

    positive_counts = np.zeros(n_labels, dtype=np.int64)
    # locations[term_idx] = list of (split, row_idx)
    locations = {t: [] for t in range(n_labels)}
    for split in train_val_splits:
        for row_idx, labels in enumerate(split_labels[split]):
            for t in labels:
                positive_counts[t] += 1
                locations[t].append((split, row_idx))

    eligible_terms = [t for t in range(n_labels) if positive_counts[t] >= args.min_term_positives]
    print(f"[INFO] {len(eligible_terms)}/{n_labels} terms eligible (>= {args.min_term_positives} positives)")

    rng = np.random.default_rng(args.seed)
    # removals[split][row_idx] = set of term indices to strip from that row's labels
    removals = {split: {} for split in train_val_splits}
    spy_meta = {}
    for t in eligible_terms:
        locs = locations[t]
        n_spy = max(1, int(round(len(locs) * args.spy_fraction)))
        spy_idx = rng.choice(len(locs), size=n_spy, replace=False)
        spy_locs = [locs[i] for i in spy_idx]
        spy_protein_ids = []
        for split, row_idx in spy_locs:
            removals[split].setdefault(row_idx, set()).add(t)
            spy_protein_ids.append(split_proteins[split][row_idx])
        spy_meta[label_names[t]] = {
            "n_positives": int(len(locs)),
            "n_spy": int(n_spy),
            "spy_protein_ids": spy_protein_ids,
        }

    total_pairs = sum(len(v["spy_protein_ids"]) for v in spy_meta.values())
    print(f"[INFO] {total_pairs} (protein, term) pairs held out as spies across {len(eligible_terms)} terms")

    new_splits = {}
    for split in train_val_splits:
        removal_map = removals[split]

        def strip_spy_labels(row, idx, removal_map=removal_map):
            to_remove = removal_map.get(idx)
            if not to_remove:
                return row
            row["labels"] = [t for t in row["labels"] if t not in to_remove]
            return row

        new_splits[split] = dataset[split].map(strip_spy_labels, with_indices=True)
        n_modified = len(removal_map)
        print(f"[INFO] split={split}: modified {n_modified}/{len(dataset[split])} rows")

    new_splits["test"] = dataset["test"]
    new_dataset = DatasetDict(new_splits)

    output_path = Path(args.output_path)
    new_dataset.save_to_disk(str(output_path))
    print(f"[DONE] Wrote spy-holdout dataset to {output_path} in {time.time() - start:.0f}s")

    meta_output_path = Path(args.meta_output_path)
    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_output_path, "w") as f:
        json.dump({
            "data_path": args.data_path,
            "output_path": str(output_path),
            "min_term_positives": args.min_term_positives,
            "spy_fraction": args.spy_fraction,
            "seed": args.seed,
            "eligible_terms": [label_names[t] for t in eligible_terms],
            "spy_holdout": spy_meta,
        }, f)
    print(f"[INFO] Wrote spy-holdout metadata to {meta_output_path}")


if __name__ == "__main__":
    main()
