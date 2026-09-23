"""TIcE (Tree Induction for c Estimation -- Bekker & Davis, AAAI 2018) per-term class-prior
estimation, the last established PU-learning prior estimator in this project's original candidate
list not yet tried (see project_pu_learning_nnpu_km2.md: "TIcE (cheaper, ~10x faster than KM2) and
Elkan-Noto's simple mean-c were both explicitly considered and rejected as primary estimators" back
when the plan was written -- KM2, the classifier-probability spy mean, a centroid-ratio estimator,
and finally a GENUINE held-out spy retrain have since all been tried and validated against true
prevalence; see pu_learning_plan.md's running log). This is that last untried estimator, tried on
its own merits, independent of whether nnPU's training-time mechanism (a separate, now-diagnosed
issue -- the positive-anchor batch's backward pass perturbs the shared backbone regardless of how
many/which terms are nnPU-eligible) ever gets fixed. A good TIcE prior would still be useful for the
project's ORIGINAL discrete two-step tiering design (relabeling some "unknown" cells via a
threshold/prior), which never touches nnPU's problematic training loop at all.

Why TIcE might behave differently from KM2/spy/centroid: those all estimate a quantity's SCALE from
a single fixed statistic (a kernel-reconstruction residual, a classifier's mean predicted
probability, a raw cosine similarity) computed over the WHOLE population or a fixed random sample.
TIcE instead recursively SEARCHES the feature space for the purest achievable subset (the region
where the local labeled-vs-total ratio is highest), which can in principle find a small,
genuinely-well-separated pocket even when the population-wide statistic shows no signal -- closer in
spirit to why nearest-centroid AUROC found real ranking structure that KM2's global reconstruction
test missed (see diagnose_mf_pooling.py's finding).

--- Implementation notes / where this diverges from the paper ---

TIcE estimates c = P(labeled | positive) (the "propensity"/labeling frequency under SCAR), then
recovers the class prior via pi_hat = P(labeled) / c_hat -- exactly Elkan-Noto's ratio, just with c
estimated by decision-tree search instead of a spy sample or a classifier's mean probability.

The core recursive search (this project's simplified stand-in for the paper's tree induction,
disclosed the same way estimate_class_priors_km2.py disclosed its elbow-fit substitution for
"gradient thresholding"):
  - At each node (starting from the full mixture-sample-and-positives set for a term), compute a
    LOWER-CONFIDENCE-BOUND-corrected labeled fraction for the current subset (a Hoeffding bound,
    `labeled_frac - sqrt(log(1/delta) / (2*n))`), guarding against a tiny subset's ratio looking
    good purely from sampling noise (the exact failure mode --spy-fraction's small held-out sets hit
    in project_pu_learning_nnpu_km2.md v1.3).
  - Greedily searches a random subset of features (--features-per-split) x a few quantile
    thresholds each, picks whichever split produces the child with the highest corrected estimate,
    and recurses into that child up to --max-depth or until a subset drops below --min-leaf-size.
  - Returns the MAX corrected estimate seen at ANY node visited (root, internal, or leaf) -- not
    just leaves, since the purest region TIcE is looking for isn't guaranteed to be a terminal node.
  - Repeated --k-folds times with independent random feature subsets/seeds and averaged, mirroring
    the paper's stability-via-averaging step (a simplification of its actual bootstrap procedure).

Uses the SAME cached embeddings extract_protein_embeddings.py already produced (no new GPU pass).

Usage:
  # Smoke test
  env/bin/python experiments/scripts/estimate_class_priors_tice.py --max-terms 5 --k-folds 2

  # Full run (timed first on a handful of terms to extrapolate total cost -- see the module's
  # own timing note in pu_learning_plan.md before running unattended)
  env/bin/python experiments/scripts/estimate_class_priors_tice.py \\
      > experiments/logs/estimate_class_priors_tice.out 2>&1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def corrected_estimate(labeled_frac: float, n: int, delta: float) -> float:
    if n == 0:
        return 0.0
    bound = np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    return max(0.0, labeled_frac - bound)


def tice_search(
    x: np.ndarray, y_labeled: np.ndarray, delta: float, max_depth: int, min_leaf_size: int,
    features_per_split: int, rng: np.random.Generator, depth: int = 0,
) -> float:
    n = len(y_labeled)
    labeled_frac = float(y_labeled.mean())
    best = corrected_estimate(labeled_frac, n, delta)
    if depth >= max_depth or n < 2 * min_leaf_size:
        return best

    n_features = x.shape[1]
    feature_idx = rng.choice(n_features, size=min(features_per_split, n_features), replace=False)
    best_child_estimate = -1.0
    best_split = None
    for f in feature_idx:
        col = x[:, f]
        quantiles = np.quantile(col, [0.25, 0.5, 0.75])
        for q in quantiles:
            left_mask = col <= q
            n_left, n_right = int(left_mask.sum()), int((~left_mask).sum())
            if n_left < min_leaf_size or n_right < min_leaf_size:
                continue
            left_frac = float(y_labeled[left_mask].mean())
            right_frac = float(y_labeled[~left_mask].mean())
            # Greedily follow whichever side looks purer (uncorrected -- the correction is applied
            # once we actually recurse into it, and at every node along the way).
            candidate_frac, candidate_mask, candidate_n = (
                (left_frac, left_mask, n_left) if left_frac >= right_frac else (right_frac, ~left_mask, n_right)
            )
            if candidate_frac > best_child_estimate:
                best_child_estimate = candidate_frac
                best_split = candidate_mask

    if best_split is not None:
        child_best = tice_search(x[best_split], y_labeled[best_split], delta, max_depth, min_leaf_size,
                                  features_per_split, rng, depth + 1)
        best = max(best, child_best)
    return best


def estimate_term_tice(
    x: np.ndarray, y_labeled: np.ndarray, delta: float, max_depth: int, min_leaf_size: int,
    features_per_split: int, k_folds: int, seed: int,
) -> float:
    estimates = []
    for k in range(k_folds):
        rng = np.random.default_rng(seed + k)
        estimates.append(tice_search(x, y_labeled, delta, max_depth, min_leaf_size, features_per_split, rng))
    return float(np.mean(estimates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", default="experiments/pu_cache")
    parser.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset_raw")
    parser.add_argument("--min-term-positives", type=int, default=100,
                         help="Matches the other estimator scripts' default, for direct comparability")
    parser.add_argument("--max-terms", type=int, default=None)
    parser.add_argument("--delta", type=float, default=0.05, help="Confidence-bound parameter for the corrected estimate")
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--min-leaf-size", type=int, default=50)
    parser.add_argument("--features-per-split", type=int, default=30, help="Random feature subset tried at each split node")
    parser.add_argument("--k-folds", type=int, default=5, help="Independent random-seed repeats, averaged")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="experiments/pu_cache/class_priors_tice.json")
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
    results = {}
    ratios = []
    for i, term_idx in enumerate(eligible_term_indices):
        term_name = label_names[term_idx]
        y_labeled = hard_labels[:, term_idx]
        c_hat = estimate_term_tice(embeddings, y_labeled, args.delta, args.max_depth, args.min_leaf_size,
                                    args.features_per_split, args.k_folds, args.seed)
        labeled_rate = float(y_labeled.mean())
        pi_hat = float(np.clip(labeled_rate / c_hat, 0.0, 1.0)) if c_hat > 1e-8 else float("nan")
        true_prevalence = labeled_rate  # every observed positive IS the true label here (fully-labeled research data)
        ratio = (pi_hat / true_prevalence) if true_prevalence > 0 and pi_hat == pi_hat else float("nan")
        if ratio == ratio:
            ratios.append(ratio)
        results[term_name] = {
            "pi_tice": pi_hat,
            "c_tice": c_hat,
            "n_positives": int(y_labeled.sum()),
            "true_prevalence": true_prevalence,
            "ratio": ratio,
        }
        if (i + 1) % 25 == 0 or (i + 1) == len(eligible_term_indices):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(eligible_term_indices) - i - 1) / max(rate, 1e-9)
            print(f"[STEP] term {i + 1}/{len(eligible_term_indices)} ({term_name}): "
                  f"pi_tice={pi_hat:.4f} c={c_hat:.4f} true={true_prevalence:.4f} ratio={ratio:.2f} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    ratios = np.array(ratios)
    print(f"[DONE] {len(results)} terms estimated in {time.time() - start:.0f}s")
    print(f"[SUMMARY] pi_tice / true_prevalence: median={np.median(ratios):.2f}x mean={ratios.mean():.2f}x "
          f"min={ratios.min():.2f}x max={ratios.max():.2f}x")
    within2x = int(((ratios >= 0.5) & (ratios <= 2.0)).sum())
    print(f"[SUMMARY] within [0.5x, 2x] of true prevalence: {within2x}/{len(ratios)} ({100 * within2x / len(ratios):.1f}%)")

    output = {
        "checkpoint_path": meta["checkpoint_path"],
        "data_path": args.data_path,
        "min_term_positives": args.min_term_positives,
        "delta": args.delta,
        "max_depth": args.max_depth,
        "min_leaf_size": args.min_leaf_size,
        "features_per_split": args.features_per_split,
        "k_folds": args.k_folds,
        "priors": results,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    main()
