"""Standalone check (no pytest in this repo): validates build_features() on the cheap
meta-eval slice (already cached, no new compute) before it's trusted on meta-train's
~54M-row flatten. Run via:
    env/bin/python experiments/ensembling/check_build_features.py
"""
import numpy as np

from stacking_pipeline import build_features, load_cached_meta_eval

meta_eval = load_cached_meta_eval()
X, y = build_features(meta_eval)

n_rows, n_labels = meta_eval["val_labels"].shape
assert X.shape == (n_rows * n_labels, 3), X.shape
assert y.shape == (n_rows * n_labels,), y.shape
assert np.isfinite(X).all(), "non-finite value in constructed features"
assert set(np.unique(y).tolist()) <= {0.0, 1.0}, "y should be binary 0/1"

print(f"[PASS] build_features check: X.shape={X.shape}, y.shape={y.shape}, all finite, y binary")
