"""Pareto-principle check on the real CAFA-6 GO-term label space.

Two histograms over the *actual* human-annotated data (IA.tsv + train_terms.tsv),
not the depth-limited/ancestor-propagated coarse label space used for training:

1. Information-accretion (IA) weight per GO term, from IA.tsv -- not every term is
   scored equally by the competition metric.
2. Annotation count per GO term -- how many proteins in train_terms.tsv carry each
   term, as asserted (no ancestor propagation).

Both are plotted on a log-x axis since GO term frequency/weight distributions are
famously long-tailed (the actual Pareto question being asked).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ia-tsv", default="data/cafa-6-protein-function-prediction/IA.tsv")
    ap.add_argument("--train-terms-tsv",
                     default="data/cafa-6-protein-function-prediction/Train/train_terms.tsv")
    ap.add_argument("--out-prefix", default="experiments/logs/go_term_pareto")
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    ia = pd.read_csv(args.ia_tsv, sep="\t", header=None, names=["term", "ia_weight"])
    terms = pd.read_csv(args.train_terms_tsv, sep="\t")

    print(f"[INFO] IA.tsv: {len(ia)} terms")
    print(f"[INFO] train_terms.tsv: {len(terms)} annotation rows, "
          f"{terms['term'].nunique()} unique terms, {terms['EntryID'].nunique()} unique proteins")

    # --- Chart 1: IA weight histogram ---
    ia_nonzero = ia[ia["ia_weight"] > 0]["ia_weight"]
    zero_count = (ia["ia_weight"] == 0).sum()
    print(f"[INFO] IA weight: {zero_count} terms at exactly 0.0 (excluded from log-x histogram), "
          f"{len(ia_nonzero)} with weight > 0")
    print(f"[INFO] IA weight (nonzero): mean={ia_nonzero.mean():.4f} median={ia_nonzero.median():.4f} "
          f"max={ia_nonzero.max():.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    log_bins = np.logspace(np.log10(ia_nonzero.min()), np.log10(ia_nonzero.max()), args.bins)
    ax.hist(ia_nonzero, bins=log_bins, color="tab:blue", edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Information accretion (IA) weight [log scale]")
    ax.set_ylabel("Number of GO terms")
    ax.set_title(f"IA.tsv weight distribution across {len(ia)} GO terms "
                 f"({zero_count} zero-weight terms excluded from plot)")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_ia_weight_histogram.png", dpi=150)
    plt.close(fig)

    # --- Chart 2: annotation-count-per-term histogram ---
    counts = terms.groupby("term").size()
    print(f"[INFO] annotation count per term: mean={counts.mean():.2f} median={counts.median():.0f} "
          f"max={counts.max()} min={counts.min()}")

    top10_share = counts.sort_values(ascending=False).head(10).sum() / counts.sum()
    print(f"[INFO] top-10 most-annotated terms account for {top10_share:.1%} of all annotation rows "
          f"(Pareto check)")

    fig, ax = plt.subplots(figsize=(10, 6))
    log_bins = np.logspace(np.log10(counts.min()), np.log10(counts.max()), args.bins)
    ax.hist(counts, bins=log_bins, color="tab:orange", edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of proteins annotated with the term [log scale]")
    ax.set_ylabel("Number of GO terms [log scale]")
    ax.set_title(f"Annotation-count distribution across {terms['term'].nunique()} GO terms "
                 f"(train_terms.tsv, asserted only, no ancestor propagation)")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_annotation_count_histogram.png", dpi=150)
    plt.close(fig)

    print(f"[INFO] Wrote {args.out_prefix}_ia_weight_histogram.png")
    print(f"[INFO] Wrote {args.out_prefix}_annotation_count_histogram.png")


if __name__ == "__main__":
    main()
