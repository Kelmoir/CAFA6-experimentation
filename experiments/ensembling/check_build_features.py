"""Standalone check (no pytest in this repo): validates build_features() on the cheap
meta-eval slice (already cached, no new compute) before it's trusted on meta-train's
~54M-row flatten. Run via:
    env/bin/python experiments/ensembling/check_build_features.py
"""
import numpy as np

from stacking_pipeline import build_features, load_cached_meta_eval

meta_eval = load_cached_meta_eval()
X, y, qid = build_features(meta_eval)

n_rows, n_labels = meta_eval["val_labels"].shape
assert X.shape == (n_rows * n_labels, 3), X.shape
assert y.shape == (n_rows * n_labels,), y.shape
assert qid.shape == (n_rows * n_labels,), qid.shape
assert np.isfinite(X).all(), "non-finite value in constructed features"
assert set(np.unique(y).tolist()) <= {0.0, 1.0}, "y should be binary 0/1"
# qid is the term index per row, row-major by protein: for row i the term index is
# i % n_labels, and every term appears in exactly n_rows rows.
assert np.array_equal(qid, np.arange(n_rows * n_labels) % n_labels), "qid != term index per row"
assert (np.bincount(qid) == n_rows).all(), "every term (qid) should appear in exactly n_rows rows"

print(f"[PASS] build_features check: X.shape={X.shape}, y.shape={y.shape}, qid.shape={qid.shape}, all finite, y binary")
