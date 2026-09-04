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

    # --- Ordering comparison: same term set, ranked by IA weight vs. by annotation count ---
    merged = pd.merge(ia, counts.rename("annotation_count"), left_on="term", right_index=True,
                       how="inner")
    print(f"[INFO] {len(merged)} terms present in both IA.tsv and train_terms.tsv "
          f"({len(ia) - len(merged)} IA.tsv-only, {len(counts) - len(merged)} train_terms-only, excluded)")

    spearman_rho = merged["ia_weight"].rank().corr(merged["annotation_count"].rank())
    print(f"[INFO] Spearman rank correlation (IA weight vs. annotation count): {spearman_rho:.3f}")

    def top20_share(df, sort_col, value_col):
        n_top = max(1, round(len(df) * 0.2))
        ranked = df.sort_values(sort_col, ascending=False)
        return ranked.head(n_top)[value_col].sum() / df[value_col].sum(), n_top

    ia_share_own, n_top = top20_share(merged, "ia_weight", "ia_weight")
    counts_share_own, _ = top20_share(merged, "annotation_count", "annotation_count")
    print(f"[INFO] Top 20% of terms by IA weight (n={n_top}) hold {ia_share_own:.1%} of total IA weight")
    print(f"[INFO] Top 20% of terms by annotation count (n={n_top}) hold {counts_share_own:.1%} "
          f"of total annotations")

    # Cross shares: does the top-20%-by-one-metric set also dominate the other metric?
    ia_top_set = merged.sort_values("ia_weight", ascending=False).head(n_top)
    counts_top_set = merged.sort_values("annotation_count", ascending=False).head(n_top)
    ia_top_counts_share = ia_top_set["annotation_count"].sum() / merged["annotation_count"].sum()
    counts_top_ia_share = counts_top_set["ia_weight"].sum() / merged["ia_weight"].sum()
    overlap = len(set(ia_top_set["term"]) & set(counts_top_set["term"])) / n_top
    print(f"[INFO] Top-20%-by-IA-weight terms hold {ia_top_counts_share:.1%} of total annotations "
          f"(vs. {counts_share_own:.1%} for the term set actually chosen for annotation count)")
    print(f"[INFO] Top-20%-by-annotation-count terms hold {counts_top_ia_share:.1%} of total IA weight "
          f"(vs. {ia_share_own:.1%} for the term set actually chosen for IA weight)")
    print(f"[INFO] Top-20% sets overlap by {overlap:.1%} ({len(set(ia_top_set['term']) & set(counts_top_set['term']))}/{n_top} terms in common)")

    # --- Chart 3: Lorenz-style cumulative-share curves for both metrics ---
    def cumulative_curve(df, col):
        vals = df[col].sort_values(ascending=False).to_numpy()
        cum = np.cumsum(vals) / vals.sum()
        x = np.arange(1, len(vals) + 1) / len(vals)
        return x, cum

    x_ia, y_ia = cumulative_curve(merged, "ia_weight")
    x_counts, y_counts = cumulative_curve(merged, "annotation_count")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], color="#999990", linewidth=1.5, linestyle="--", label="equal share (no concentration)")
    ax.plot(x_ia, y_ia, color="#2a78d6", linewidth=2, label="IA weight")
    ax.plot(x_counts, y_counts, color="#eb6834", linewidth=2, label="annotation count")
    ax.axvline(0.2, color="#c9c8bc", linewidth=1, linestyle=":")
    ax.set_xlabel("Cumulative share of GO terms (sorted descending by the metric itself)")
    ax.set_ylabel("Cumulative share of the metric's total")
    ax.set_title(f"Concentration of IA weight vs. annotation count across {len(merged)} GO terms\n"
                 f"Spearman ρ={spearman_rho:.2f} between the two rankings")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_concentration_curve.png", dpi=150)
    plt.close(fig)
    print(f"[INFO] Wrote {args.out_prefix}_concentration_curve.png")


if __name__ == "__main__":
    main()
