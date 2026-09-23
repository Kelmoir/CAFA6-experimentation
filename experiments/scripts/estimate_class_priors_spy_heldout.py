"""Elkan-Noto class-prior estimation from a GENUINE held-out spy retrain -- the fix attempted after
three prior estimators (KM2, classifier-probability spy mean, centroid ratio) all failed against
true per-term prevalence (see project_pu_learning_nnpu_km2.md v1.1/v1.2 and pu_learning_plan.md's
centroid-approach note).

Two, not one, bugs in the original `estimate_class_priors_km2.py`'s "pi_spy" are fixed here:

  1. **Not a real spy technique.** The old script scored "spy" positives with the SAME checkpoint
     that was directly trained on those exact rows as hard positive (BCE target 1.0) -- of course a
     well-fit classifier is confident on its own training positives; that's most of where the 14.4x
     median overestimation came from. `experiments/scripts/build_spy_holdout_dataset.py` fixes this
     properly: it builds a dataset where each eligible term's spy positives are REMOVED from that
     term's training labels (row's other labels untouched), and a model trained on it (from a FRESH
     backbone init, never from a checkpoint that already saw the unmodified labels) has genuinely
     never been told these specific rows are positive for that term.
  2. **Not actually Elkan-Noto's ratio.** The old script set `pi_spy = c_spy` directly (the mean
     predicted probability on spy positives), not the paper's `pi_hat = E[g(x) | unlabeled] /
     E[g(x) | known positive]`. Skipping the ratio and using the numerator alone silently assumes
     E[g(x) | unlabeled] ~= pi (only true if unlabeled-set exposure and prior scale happen to
     coincide) -- this script computes the actual ratio.

Reads:
  --embeddings-dir: output of extract_protein_embeddings.py run against the NEW spy-holdout-trained
    checkpoint (protein_probs.npy + protein_meta.json) -- probabilities only; embeddings aren't
    needed for this estimator.
  --spy-holdout-meta-path: build_spy_holdout_dataset.py's spy_holdout_meta.json, giving each
    eligible term's held-out spy protein ids.
  --data-path: the ORIGINAL (unmodified) dataset_raw, to get each term's TRUE labels -- both for
    the ratio's "unlabeled" denominator (true label == 0, same convention as every other estimator
    in this pipeline) and for the true_prevalence column used to validate this estimator directly
    (a research-only luxury; a real deployment wouldn't have this check).

Usage (after the retrain + a fresh extraction against its checkpoint have both finished):
  env/bin/python experiments/scripts/estimate_class_priors_spy_heldout.py \\
      --embeddings-dir experiments/pu_cache/spy_holdout \\
      > experiments/logs/estimate_class_priors_spy_heldout.out 2>&1
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", default="experiments/pu_cache/spy_holdout",
                         help="extract_protein_embeddings.py output against the spy-holdout-retrained checkpoint")
    parser.add_argument("--spy-holdout-meta-path", default="experiments/pu_cache/spy_holdout_meta.json")
    parser.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset_raw",
                         help="ORIGINAL (unmodified) dataset, for true labels/prevalence")
    parser.add_argument("--output-path", default="experiments/pu_cache/class_priors_spy_heldout.json")
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    probs = np.load(embeddings_dir / "protein_probs.npy")
    meta = json.loads((embeddings_dir / "protein_meta.json").read_text())
    label_names = meta["label_names"]
    protein_ids = meta["protein_ids"]
    protein_row = {pid: i for i, pid in enumerate(protein_ids)}
    print(f"[INFO] Loaded {probs.shape[0]} scored proteins, {len(label_names)} labels")

    spy_meta = json.loads(Path(args.spy_holdout_meta_path).read_text())
    print(f"[INFO] {len(spy_meta['eligible_terms'])} eligible terms in spy-holdout metadata "
          f"(spy_fraction={spy_meta['spy_fraction']}, min_term_positives={spy_meta['min_term_positives']})")

    dataset = load_from_disk(args.data_path)
    protein_to_labels = {}
    for split in ["train", "validation"]:
        rows = dataset[split]
        for pid, labels in zip(rows["protein"], rows["labels"]):
            protein_to_labels[pid] = labels

    n = probs.shape[0]
    label2id = {name: i for i, name in enumerate(label_names)}
    hard_labels = np.zeros((n, len(label_names)), dtype=np.float32)
    for pid, row_idx in protein_row.items():
        for term_idx in protein_to_labels.get(pid, []):
            hard_labels[row_idx, int(term_idx)] = 1.0

    results = {}
    ratios = []
    for term_name, entry in spy_meta["spy_holdout"].items():
        term_idx = label2id[term_name]
        spy_rows = [protein_row[pid] for pid in entry["spy_protein_ids"] if pid in protein_row]
        if len(spy_rows) < len(entry["spy_protein_ids"]):
            print(f"[WARN] {term_name}: {len(entry['spy_protein_ids']) - len(spy_rows)} spy proteins "
                  f"missing from the extraction output")
        c = float(probs[spy_rows, term_idx].mean())
        unlabeled_rows = np.where(hard_labels[:, term_idx] == 0)[0]
        u_mean = float(probs[unlabeled_rows, term_idx].mean())
        pi_hat = float(np.clip(u_mean / c, 0.0, 1.0)) if c > 1e-8 else float("nan")
        true_prevalence = float(hard_labels[:, term_idx].sum() / n)
        ratio = (pi_hat / true_prevalence) if true_prevalence > 0 and pi_hat == pi_hat else float("nan")
        if ratio == ratio:
            ratios.append(ratio)
        results[term_name] = {
            "pi_spy_heldout": pi_hat,
            "c_spy_heldout": c,
            "u_mean": u_mean,
            "n_spy": len(spy_rows),
            "n_positives": entry["n_positives"],
            "true_prevalence": true_prevalence,
            "ratio": ratio,
        }

    ratios = np.array(ratios)
    print(f"[DONE] {len(results)} terms estimated")
    print(f"[SUMMARY] pi_spy_heldout / true_prevalence: median={np.median(ratios):.2f}x "
          f"mean={ratios.mean():.2f}x min={ratios.min():.2f}x max={ratios.max():.2f}x")
    within2x = int(((ratios >= 0.5) & (ratios <= 2.0)).sum())
    print(f"[SUMMARY] within [0.5x, 2x] of true prevalence: {within2x}/{len(ratios)} ({100 * within2x / len(ratios):.1f}%)")

    output = {
        "embeddings_dir": str(embeddings_dir),
        "checkpoint_path": meta.get("checkpoint_path"),
        "spy_holdout_meta_path": args.spy_holdout_meta_path,
        "data_path": args.data_path,
        "priors": results,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    main()
