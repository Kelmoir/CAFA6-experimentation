"""Standalone check (no pytest in this repo): validates build_rank_features() on the
cheap meta-eval slice (already cached, no new compute). Unlike build_features(), this
rank-transforms esm_prob/blast_score per term before flattening -- the fix for the
diagnosed mechanism (a pointwise tree scorer can't infer each term's wildly different
raw-value scale on its own; rank-transforming normalizes that away by construction).
Run via:
    env/bin/python experiments/ensembling/check_build_rank_features.py
"""
import numpy as np

from stacking_pipeline import build_rank_features, load_cached_meta_eval

meta_eval = load_cached_meta_eval()
X, y, qid = build_rank_features(meta_eval)

n_rows, n_labels = meta_eval["val_labels"].shape
assert X.shape == (n_rows * n_labels, 3), X.shape
assert y.shape == (n_rows * n_labels,), y.shape
assert qid.shape == (n_rows * n_labels,), qid.shape
assert np.isfinite(X).all(), "non-finite value in constructed rank features"
assert set(np.unique(y).tolist()) <= {0.0, 1.0}, "y should be binary 0/1"

# Columns 0/1 (rank-transformed esm_prob/blast_score) must be rank fractions in (0, 1],
# unlike raw build_features() which allows exact 0.0 (e.g. a protein with no BLAST hits).
assert X[:, 0].min() > 0.0 and X[:, 0].max() <= 1.0, (X[:, 0].min(), X[:, 0].max())
assert X[:, 1].min() > 0.0 and X[:, 1].max() <= 1.0, (X[:, 1].min(), X[:, 1].max())

# The rank transform must preserve each term's ordering exactly (AUROC-invariant):
# picking one term, the row with the highest raw esm_prob must also have the highest
# rank-transformed value.
term_j = 5
raw_col = meta_eval["esm_prob"][:, term_j]
rank_col = X[:, 0].reshape(n_rows, n_labels)[:, term_j]
assert np.argmax(raw_col) == np.argmax(rank_col), "rank transform did not preserve ordering"

print(f"[PASS] build_rank_features check: X.shape={X.shape}, ranks in (0,1], ordering preserved")
