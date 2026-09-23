"""Centroid-based per-term class prior (pi) estimation -- an alternative to KM2 for
experiments/pu_learning_plan.md's nnPU pipeline, tried after both KM2 and the classifier-probability
spy mean failed empirically (see project_pu_learning_nnpu_km2.md, v1.1/v1.2).

Motivation: experiments/scripts/diagnose_mf_pooling.py found that a trivial nearest-centroid cosine
similarity probe ranks a term's held-out positives above background well (AUROC 0.71-0.99 across 6
MF terms tested), even though KM2's kernel-mean-reconstruction test (D(kappa)) found nothing to
exploit in the SAME embedding space (monotonic decrease to kappa=1 for every term checked). This is
not necessarily a contradiction: AUROC is scale-invariant (it only needs positives to rank above
background), while both KM2 and Elkan-Noto's ratio estimator need to recover an actual SCALE (the
class prior itself), which a tiny-but-consistent mean shift can fail to support even when it
supports good ranking.

So this script does NOT assume "ranking works => a centroid-based prior will work". It implements
the natural centroid-based prior estimator and checks its output directly against TRUE per-term
prevalence -- available here only because this is a fully-labeled research dataset used to validate
the estimator; a real deployment wouldn't have this check. Two variants, both an Elkan-Noto-style
ratio (pi_hat = E[g(x) | unlabeled] / E[g(x) | held-out known positive]), differing only in g:

  1. `pi_centroid_raw`: g(x) = cosine similarity of x to the L2-normalized centroid of a term's
     positive embeddings (half held out as "spies", per --spy-fraction).
  2. `pi_centroid_centered`: same, but every embedding has the global population mean subtracted
     before normalizing/projecting -- removing a shared "generic protein" direction that could
     otherwise inflate g(x) on true negatives too and bias the ratio upward, the same failure mode
     already diagnosed for the classifier-probability spy mean (14.4x median overestimation, see
     project_pu_learning_nnpu_km2.md v1.2).

Elkan-Noto's ratio is only unbiased when E[g | true negative] ~= 0; centering is a cheap attempt to
get closer to that, without the O(n^2) kernel machinery KM2 needs.

Reuses the SAME cached embeddings extract_protein_embeddings.py already produced (no new GPU forward
pass) -- pure CPU linear algebra over a (54395, 480) matrix, a few seconds total even across all
eligible terms, so this runs directly (no GPU handoff needed).

Usage:
  env/bin/python experiments/scripts/estimate_class_priors_centroid.py \\
      > experiments/logs/estimate_class_priors_centroid.out 2>&1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, eps, None)


def estimate_term(hard_labels_col: np.ndarray, normed_raw: np.ndarray, normed_centered: np.ndarray,
                   spy_fraction: float, rng: np.random.Generator) -> dict:
    pos_rows = np.where(hard_labels_col == 1)[0]
    n_spy = max(1, int(round(len(pos_rows) * spy_fraction)))
    spy_rows = rng.choice(pos_rows, size=n_spy, replace=False)
    centroid_rows = np.setdiff1d(pos_rows, spy_rows)
    if len(centroid_rows) == 0:
        centroid_rows = pos_rows
    unlabeled_rows = np.where(hard_labels_col == 0)[0]

    result = {"n_spy": int(n_spy), "n_positives": int(len(pos_rows))}
    for tag, normed in (("raw", normed_raw), ("centered", normed_centered)):
        centroid = normed[centroid_rows].mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-8:
            result[f"pi_centroid_{tag}"] = float("nan")
            result[f"c_{tag}"] = 0.0
            continue
        centroid = centroid / centroid_norm
        scores = normed @ centroid
        c = float(scores[spy_rows].mean())
        pi = float(np.clip(scores[unlabeled_rows].mean() / c, 0.0, 1.0)) if abs(c) > 1e-8 else float("nan")
        result[f"pi_centroid_{tag}"] = pi
        result[f"c_{tag}"] = c
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", default="experiments/pu_cache")
    parser.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset_raw")
    parser.add_argument("--min-term-positives", type=int, default=100,
                         help="Matches estimate_class_priors_km2.py's default, for direct comparability")
    parser.add_argument("--max-terms", type=int, default=None)
    parser.add_argument("--spy-fraction", type=float, default=0.2, help="Matches the km2 script's spy fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="experiments/pu_cache/class_priors_centroid.json")
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    embeddings = np.load(embeddings_dir / "protein_embeddings.npy")
    meta = json.loads((embeddings_dir / "protein_meta.json").read_text())
    label_names = meta["label_names"]
    protein_ids = meta["protein_ids"]
    print(f"[INFO] Loaded {embeddings.shape[0]} embedded proteins, {len(label_names)} labels")

    dataset = load_from_disk(args.data_path)
    protein_to_labels = {}
    for split in meta["splits"]:
        rows = dataset[split]
        for pid, labels in zip(rows["protein"], rows["labels"]):
            protein_to_labels[pid] = labels

    n = embeddings.shape[0]
    hard_labels = np.zeros((n, len(label_names)), dtype=np.float32)
    for row_idx, pid in enumerate(protein_ids):
        for term_idx in protein_to_labels.get(pid, []):
            hard_labels[row_idx, int(term_idx)] = 1.0

    positive_counts = hard_labels.sum(axis=0)
    eligible_term_indices = np.where(positive_counts >= args.min_term_positives)[0]
    if args.max_terms is not None:
        eligible_term_indices = eligible_term_indices[: args.max_terms]
    print(f"[INFO] {len(eligible_term_indices)}/{len(label_names)} terms eligible (>= {args.min_term_positives} positives)")

    start = time.time()
    mixture_mean = embeddings.mean(axis=0)
    normed_raw = l2_normalize(embeddings)
    normed_centered = l2_normalize(embeddings - mixture_mean)
    print(f"[INFO] Precomputed normalized embeddings in {time.time() - start:.1f}s")

    rng = np.random.default_rng(args.seed)
    results = {}
    raw_ratios, centered_ratios = [], []
    for i, term_idx in enumerate(eligible_term_indices):
        term_name = label_names[term_idx]
        col = hard_labels[:, term_idx]
        r = estimate_term(col, normed_raw, normed_centered, args.spy_fraction, rng)
        true_prevalence = float(col.sum() / n)
        r["true_prevalence"] = true_prevalence
        for tag in ("raw", "centered"):
            pi = r[f"pi_centroid_{tag}"]
            r[f"ratio_{tag}"] = (pi / true_prevalence) if true_prevalence > 0 and pi == pi else float("nan")
        if r["ratio_raw"] == r["ratio_raw"]:
            raw_ratios.append(r["ratio_raw"])
        if r["ratio_centered"] == r["ratio_centered"]:
            centered_ratios.append(r["ratio_centered"])
        results[term_name] = r
        if (i + 1) % 50 == 0 or (i + 1) == len(eligible_term_indices):
            print(f"[STEP] term {i + 1}/{len(eligible_term_indices)} ({term_name}): "
                  f"pi_raw={r['pi_centroid_raw']:.4f} pi_centered={r['pi_centroid_centered']:.4f} "
                  f"true={true_prevalence:.4f} elapsed={time.time() - start:.1f}s")

    raw_ratios = np.array(raw_ratios)
    centered_ratios = np.array(centered_ratios)
    print(f"[DONE] {len(results)} terms estimated in {time.time() - start:.1f}s")
    print(f"[SUMMARY] pi_centroid_raw / true_prevalence: median={np.median(raw_ratios):.2f}x "
          f"mean={raw_ratios.mean():.2f}x min={raw_ratios.min():.2f}x max={raw_ratios.max():.2f}x")
    print(f"[SUMMARY] pi_centroid_centered / true_prevalence: median={np.median(centered_ratios):.2f}x "
          f"mean={centered_ratios.mean():.2f}x min={centered_ratios.min():.2f}x max={centered_ratios.max():.2f}x")

    output = {
        "checkpoint_path": meta["checkpoint_path"],
        "data_path": args.data_path,
        "min_term_positives": args.min_term_positives,
        "spy_fraction": args.spy_fraction,
        "priors": results,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    main()
