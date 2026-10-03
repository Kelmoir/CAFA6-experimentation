"""Standalone check (no pytest in this repo): unit-tests _check_ranking_signal()
directly with synthetic data, fast and deterministic -- the TDD proof for
fit_meta_model()'s degeneracy guard. An earlier version of this guard compared raw
per-label distinct-prediction-value counts against a reference feature; that proxy
falsely flagged a healthy rank:pairwise fit (tree-based rankers legitimately produce
few distinct values without losing ranking signal -- see the real in-sample result:
median 109 distinct values/label, yet mean AUROC 0.8839, beating the model-alone
baseline). This guard now measures per-label AUROC directly against y_train instead.
Run via:
    env/bin/python experiments/ensembling/check_ranking_signal_guard.py
"""
import numpy as np

from stacking_pipeline import _check_ranking_signal

n_rows, n_labels = 200, 50
rng = np.random.default_rng(42)

# Each label gets exactly 100 positives / 100 negatives (own random permutation per
# label), so every label is evaluable regardless of draw.
y = np.zeros((n_rows, n_labels), dtype=np.float32)
half = np.array([1.0] * (n_rows // 2) + [0.0] * (n_rows // 2))
for j in range(n_labels):
    y[:, j] = rng.permutation(half)
y_flat = y.ravel()

# Healthy case: predictions track the labels closely per label -> high AUROC.
healthy_preds = (y + rng.normal(scale=0.1, size=y.shape)).ravel()
_check_ranking_signal(healthy_preds, y_flat, n_labels)  # must not raise
print("[PASS] healthy case: no exception raised")

# Degenerate case: predictions are pure noise, uncorrelated with the label within any
# given label column -> near-chance AUROC per label.
collapsed_preds = rng.random(n_rows * n_labels)
try:
    _check_ranking_signal(collapsed_preds, y_flat, n_labels)
    raise SystemExit("[FAIL] expected AssertionError on the collapsed case")
except AssertionError as e:
    print(f"[PASS] collapsed case correctly raised: {e}")
