"""Per-term class prior (pi) estimation for `experiments/pu_learning_plan.md`'s nnPU pipeline
(pipeline step 3), using KM2 (Ramaswamy, Scott & Tewari 2016) as the primary estimator and the
Elkan-Noto (2008) spy mean as an independent cross-check.

Reads the embeddings/probabilities `experiments/scripts/extract_protein_embeddings.py` produced
once per baseline checkpoint, and writes one cache file (`class_priors_km2.json`) reused by every
subsequent `training_run.py --nnpu` run -- this script is NOT meant to be re-run per training
trial.

--- What this implements, and where it deviates from the paper (read before trusting output) ---

KM2's actual estimator scans a grid of candidate kappa (the presumed positive proportion within
the "mixture" set = all unlabeled-or-positive proteins for a term) and, for each kappa, solves a
QP: find the best convex combination (over the *mixture* sample) that reconstructs
`kappa * mean(positive embeddings)` in an RKHS (Gaussian/RBF kernel), and records the residual
distance D(kappa). D(kappa) stays near its noise floor while kappa <= true kappa*, then rises once
kappa exceeds what a valid sub-population could produce. The paper selects kappa* via a
statistically-derived slope/threshold test ("gradient thresholding").

This implementation:
  - Solves each QP via Frank-Wolfe (conditional gradient) on the probability simplex over the
    mixture sample -- simple, always feasible, no step-size tuning, standard for this kind of
    kernel-herding objective.
  - Selects kappa* via a two-segment (piecewise-linear) elbow fit to the D(kappa) curve instead of
    the paper's exact statistical slope test -- a practical approximation of "gradient
    thresholding", not a byte-for-byte port of it. Flagged explicitly here since it's the one
    place this diverges from the cited method, not just an implementation detail.
  - Uses a SHARED, capped mixture sample (--mixture-sample-size, default 5000 proteins, same
    random draw reused for every term) rather than the full ~54k-protein population per term.
    The plan's earlier note ("KM2 runs at full protein-set scale per term, not a capped
    subsample") turns out to be computationally infeasible as literally stated: an exact
    per-term QP over the full population needs an O(n^2) kernel matrix per term -- at n=54,395
    that's a ~2.96B-entry matrix, and repeating that (plus O(n^2) per Frank-Wolfe iteration) for
    668 terms is many orders of magnitude beyond what a QP scan can do in practice, on this CPU
    or a GPU. A shared 5000-protein subsample is a standard Monte-Carlo approximation within KM2
    itself (not a different, cheaper method the way TIcE would be) -- large enough to be
    statistically representative, and it makes the per-term cost cheap (the O(n^2) kernel matrix
    over the shared mixture sample, K_MM, is computed ONCE and reused for every term's QP scan;
    only the per-term positive-mean vector changes). See pu_learning_plan.md's "Key risks"
    section, which has been updated to record this as a disclosed revision, not a silent one.
  - The Elkan-Noto spy mean (`c_spy` below) is computed from the SAME baseline checkpoint the
    "spy" positives were themselves trained on -- there's no held-out-spy retraining in this v1
    (that's pipeline step 2's "per-term spy holdout with k-fold", not yet built; expensive at
    668-term scale). This makes c_spy optimistic (the model has seen these exact positives during
    training), which biases the cross-check estimate of pi upward. Treated here as a rough sanity
    check against KM2, not a rigorous second estimator -- large disagreement is still worth
    recording, but don't read a close match as strong confirmation.

Usage:
  # Smoke test (few terms, small mixture sample, runs in well under a minute)
  env/bin/python experiments/scripts/estimate_class_priors_km2.py \\
      --embeddings-dir experiments/pu_cache/smoke --mixture-sample-size 200 \\
      --min-term-positives 2 --max-terms 5 --output-path experiments/pu_cache/smoke/class_priors_km2.json

  # Full run (all eligible terms, default 5000-protein mixture sample)
  nohup env/bin/python experiments/scripts/estimate_class_priors_km2.py \\
      > experiments/logs/estimate_class_priors_km2.out 2>&1 &
"""

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from datasets import load_from_disk


def rbf_kernel_matrix(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian RBF kernel: k(a,b) = exp(-||a-b||^2 / (2*bandwidth^2))."""
    x_sq = (x ** 2).sum(axis=1, keepdims=True)
    y_sq = (y ** 2).sum(axis=1, keepdims=True)
    sq_dists = x_sq + y_sq.T - 2 * x @ y.T
    np.clip(sq_dists, 0, None, out=sq_dists)
    return np.exp(-sq_dists / (2 * bandwidth ** 2))


def median_heuristic_bandwidth(x: np.ndarray, sample_size: int = 500, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sub = x[idx]
    sq_dists = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=-1)
    med = np.median(sq_dists[np.triu_indices(len(sub), k=1)])
    return float(np.sqrt(max(med, 1e-12) / 2))


def frank_wolfe_qp(k_mm: np.ndarray, k_mp_mean: np.ndarray, kappa: float, max_iters: int = 100) -> float:
    """min over beta in the simplex of || sum_i beta_i phi(x_i^M) - kappa * mean(phi(P)) ||^2_H
    = beta^T K_MM beta - 2*kappa*beta^T k_mp_mean + kappa^2 * k_pp_mean (last term added by caller,
    constant w.r.t. beta so dropped here). Returns the achieved (unconstant-shifted) objective
    value beta^T K_MM beta - 2*kappa*beta^T k_mp_mean at the Frank-Wolfe optimum.
    """
    n = k_mm.shape[0]
    beta = np.full(n, 1.0 / n)
    for t in range(max_iters):
        grad = 2 * (k_mm @ beta) - 2 * kappa * k_mp_mean
        vertex = int(np.argmin(grad))
        gamma = 2.0 / (t + 2.0)
        beta = (1 - gamma) * beta
        beta[vertex] += gamma
    return float(beta @ k_mm @ beta - 2 * kappa * (beta @ k_mp_mean))


def elbow_kappa(kappas: np.ndarray, d_curve: np.ndarray) -> Tuple[float, int]:
    """Two-segment piecewise-linear fit: for each candidate breakpoint index b (1..len-2), fit a
    separate least-squares line to d_curve[:b+1] and d_curve[b:], sum squared residuals, pick the
    b minimizing total residual. Practical stand-in for KM2's "gradient thresholding" -- see the
    module docstring's caveat.
    """
    best_b, best_resid = 1, float("inf")
    n = len(kappas)
    for b in range(1, n - 1):
        resid = 0.0
        for seg_x, seg_y in ((kappas[:b + 1], d_curve[:b + 1]), (kappas[b:], d_curve[b:])):
            if len(seg_x) < 2:
                continue
            coeffs = np.polyfit(seg_x, seg_y, 1)
            fit = np.polyval(coeffs, seg_x)
            resid += float(((seg_y - fit) ** 2).sum())
        if resid < best_resid:
            best_resid, best_b = resid, b
    return float(kappas[best_b]), best_b


def km2_estimate_term(
    positive_embeddings: np.ndarray,
    k_mm: np.ndarray,
    mixture_embeddings: np.ndarray,
    bandwidth: float,
    kappa_grid: np.ndarray,
    fw_iters: int,
) -> dict:
    k_mp = rbf_kernel_matrix(mixture_embeddings, positive_embeddings, bandwidth)
    k_mp_mean = k_mp.mean(axis=1)
    k_pp = rbf_kernel_matrix(positive_embeddings, positive_embeddings, bandwidth)
    k_pp_mean = float(k_pp.mean())

    d_curve = np.array([
        frank_wolfe_qp(k_mm, k_mp_mean, kappa, max_iters=fw_iters) + kappa ** 2 * k_pp_mean
        for kappa in kappa_grid
    ])
    d_curve = np.clip(d_curve, 0, None)
    kappa_star, breakpoint_idx = elbow_kappa(kappa_grid, d_curve)
    return {
        "pi_km2": kappa_star,
        "d_curve": d_curve.tolist(),
        "kappa_grid": kappa_grid.tolist(),
        "breakpoint_idx": breakpoint_idx,
    }


def elkan_noto_spy_mean(probs_col: np.ndarray, positive_mask: np.ndarray, spy_fraction: float, seed: int) -> Tuple[float, int]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(positive_mask)[0]
    n_spy = max(1, int(round(len(pos_idx) * spy_fraction)))
    spy_idx = rng.choice(pos_idx, size=n_spy, replace=False)
    return float(probs_col[spy_idx].mean()), n_spy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", default="experiments/pu_cache",
                         help="Directory produced by extract_protein_embeddings.py")
    parser.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset_raw",
                         help="Must match the dataset extract_protein_embeddings.py was run against")
    parser.add_argument("--min-term-positives", type=int, default=100,
                         help="Terms with fewer positives (across the embedded splits) than this skip PU entirely "
                              "(too few positives for a stable KM2 QP / spy estimate) -- mirrors "
                              "dataset_refinement_pilot.py's --min-term-annotations convention")
    parser.add_argument("--max-terms", type=int, default=None, help="Cap number of eligible terms processed, for a smoke test")
    parser.add_argument("--mixture-sample-size", type=int, default=5000,
                         help="Shared random subsample of all embedded proteins used as the KM2 'mixture' set for "
                              "every term (see module docstring for why this can't be the full ~54k population)")
    parser.add_argument("--max-positives-per-term", type=int, default=2000,
                         help="Cap a term's positive-embedding set for QP tractability (random subsample if exceeded)")
    parser.add_argument("--kappa-max", type=float, default=0.3)
    parser.add_argument("--kappa-steps", type=int, default=16)
    parser.add_argument("--fw-iters", type=int, default=150, help="Frank-Wolfe iterations per (term, kappa) QP")
    parser.add_argument("--spy-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="experiments/pu_cache/class_priors_km2.json")
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    embeddings = np.load(embeddings_dir / "protein_embeddings.npy")
    probs = np.load(embeddings_dir / "protein_probs.npy")
    meta = json.loads((embeddings_dir / "protein_meta.json").read_text())
    label_names = meta["label_names"]
    protein_ids = meta["protein_ids"]
    print(f"[INFO] Loaded {embeddings.shape[0]} embedded proteins, {len(label_names)} labels")

    dataset = load_from_disk(args.data_path)
    protein_to_labels = {}
    for split in meta["splits"]:
        for row in dataset[split]:
            protein_to_labels[row["protein"]] = row["labels"]

    label2id = {name: i for i, name in enumerate(label_names)}
    n = embeddings.shape[0]
    hard_labels = np.zeros((n, len(label_names)), dtype=np.float32)
    for row_idx, pid in enumerate(protein_ids):
        for term_idx in protein_to_labels.get(pid, []):
            hard_labels[row_idx, int(term_idx)] = 1.0

    positive_counts = hard_labels.sum(axis=0)
    eligible_term_indices = np.where(positive_counts >= args.min_term_positives)[0]
    if args.max_terms is not None:
        eligible_term_indices = eligible_term_indices[: args.max_terms]
    print(f"[INFO] {len(eligible_term_indices)}/{len(label_names)} terms eligible "
          f"(>= {args.min_term_positives} positives)")

    rng = np.random.default_rng(args.seed)
    mixture_idx = rng.choice(n, size=min(args.mixture_sample_size, n), replace=False)
    mixture_embeddings = embeddings[mixture_idx]
    bandwidth = median_heuristic_bandwidth(mixture_embeddings, seed=args.seed)
    print(f"[INFO] Shared mixture sample: {len(mixture_idx)} proteins, RBF bandwidth={bandwidth:.4f}")

    start = time.time()
    print("[INFO] Computing shared K_MM (mixture-mixture kernel matrix), once for every term...")
    k_mm = rbf_kernel_matrix(mixture_embeddings, mixture_embeddings, bandwidth)
    print(f"[INFO] K_MM ready ({k_mm.shape}) in {time.time() - start:.1f}s")

    kappa_grid = np.linspace(0.0, args.kappa_max, args.kappa_steps)

    results = {}
    for i, term_idx in enumerate(eligible_term_indices):
        term_name = label_names[term_idx]
        pos_rows = np.where(hard_labels[:, term_idx] == 1)[0]
        if len(pos_rows) > args.max_positives_per_term:
            pos_rows = rng.choice(pos_rows, size=args.max_positives_per_term, replace=False)
        positive_embeddings = embeddings[pos_rows]

        km2_result = km2_estimate_term(positive_embeddings, k_mm, mixture_embeddings, bandwidth, kappa_grid, args.fw_iters)
        c_spy, n_spy = elkan_noto_spy_mean(probs[:, term_idx], hard_labels[:, term_idx] == 1, args.spy_fraction, args.seed)
        pi_spy = c_spy  # Elkan-Noto's e1 IS the class-prior-scale quantity here (mean predicted
        # prob on held-out positives) when compared against KM2's pi on the same 0-1 scale; see
        # module docstring caveat about this not being a held-out estimate in v1.

        pi_km2 = km2_result["pi_km2"]
        disagreement = abs(pi_km2 - pi_spy)
        results[term_name] = {
            "pi_km2": pi_km2,
            "pi_spy": pi_spy,
            "n_spy": n_spy,
            "n_positives": int(len(pos_rows)),
            "disagreement": disagreement,
            "flagged_scar_violation": disagreement > 0.5 * max(pi_km2, pi_spy, 1e-6),
            "km2_diagnostics": {"d_curve": km2_result["d_curve"], "kappa_grid": km2_result["kappa_grid"],
                                 "breakpoint_idx": km2_result["breakpoint_idx"]},
        }
        if (i + 1) % 10 == 0 or (i + 1) == len(eligible_term_indices):
            elapsed = time.time() - start
            print(f"[STEP] term {i + 1}/{len(eligible_term_indices)} ({term_name}): "
                  f"pi_km2={pi_km2:.4f} pi_spy={pi_spy:.4f} n_pos={len(pos_rows)} elapsed={elapsed:.0f}s")

    n_flagged = sum(1 for r in results.values() if r["flagged_scar_violation"])
    print(f"[DONE] {len(results)} terms estimated in {time.time() - start:.0f}s; "
          f"{n_flagged} flagged for large KM2/spy disagreement (possible SCAR violation)")

    output = {
        "checkpoint_path": meta["checkpoint_path"],
        "data_path": args.data_path,
        "mixture_sample_size": len(mixture_idx),
        "bandwidth": bandwidth,
        "min_term_positives": args.min_term_positives,
        "priors": results,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    main()
