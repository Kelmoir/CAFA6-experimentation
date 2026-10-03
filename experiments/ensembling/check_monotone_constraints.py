"""Standalone check (no pytest in this repo): unit-tests fit_meta_model()'s new
optional `monotone_constraints` parameter with synthetic data, fast and deterministic.
Confirms (1) the constraint is actually applied to the fitted model, not just accepted
and ignored, and (2) predictions are empirically non-decreasing in a constrained
feature when the other features are held fixed -- the real property this is meant to
buy, motivated by the final-review finding (2026-10-03) that a flexible, unconstrained
tree ensemble underperformed a strictly-monotonic closed-form baseline even in the
best case (in-sample, pre-rank-normalized features).

Run via: env/bin/python experiments/ensembling/check_monotone_constraints.py
"""
import numpy as np

from stacking_pipeline import fit_meta_model

n_rows, n_labels = 200, 50
rng = np.random.default_rng(42)

# Each label gets exactly 100 positives / 100 negatives.
y = np.zeros((n_rows, n_labels), dtype=np.float32)
half = np.array([1.0] * (n_rows // 2) + [0.0] * (n_rows // 2))
for j in range(n_labels):
    y[:, j] = rng.permutation(half)
y_flat = y.ravel()

feat_a = (y + rng.normal(scale=0.3, size=y.shape)).ravel().astype(np.float32)
feat_b = (y + rng.normal(scale=0.3, size=y.shape)).ravel().astype(np.float32)
context = rng.normal(size=y.shape).ravel().astype(np.float32)  # unconstrained
X = np.stack([feat_a, feat_b, context], axis=1)
qid = np.tile(np.arange(n_labels), n_rows)

model = fit_meta_model(X, y_flat, qid, n_labels, monotone_constraints=(1, 1, 0))

# 1. The constraint was actually applied to the fitted booster, not silently dropped.
config = model.get_booster().save_config()
assert '"monotone_constraints":"(1,1,0)"' in config, (
    f"monotone_constraints not found in the fitted booster's config: {config[:500]}"
)
print("[PASS] monotone_constraints was applied to the fitted booster")

# 2. Empirical monotonicity: holding feat_b/context fixed, predictions must be
# non-decreasing as feat_a increases, for every point on a grid.
grid = np.linspace(-2, 2, 50).astype(np.float32)
fixed_b, fixed_ctx = 0.0, 0.0
probe_X = np.stack([grid, np.full_like(grid, fixed_b), np.full_like(grid, fixed_ctx)], axis=1)
preds = model.predict(probe_X)
diffs = np.diff(preds)
assert (diffs >= -1e-6).all(), (
    f"predictions decreased as the constrained feature increased: {diffs[diffs < -1e-6]}"
)
print("[PASS] predictions are empirically non-decreasing in the constrained feature")
