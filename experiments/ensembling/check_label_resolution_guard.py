"""Standalone check (no pytest in this repo): unit-tests _check_label_resolution()
directly with synthetic data, fast and deterministic -- the TDD proof for the final-
review fix to fit_meta_model()'s degeneracy guard. The original guard only checked
pooled prediction std, which stays nonzero even when per-label ranking resolution has
collapsed (the real failure found in the stacking result -- see
cafa6_next_steps_plan.md's corrected write-up). Run via:
    env/bin/python experiments/ensembling/check_label_resolution_guard.py
"""
import numpy as np

from stacking_pipeline import _check_label_resolution

n_rows, n_labels = 200, 50
rng = np.random.default_rng(42)

# Healthy case: predictions vary about as much as the reference feature, per label.
healthy_preds = rng.random(n_rows * n_labels)
healthy_ref = rng.random(n_rows * n_labels)
_check_label_resolution(healthy_preds, healthy_ref, n_labels)  # must not raise
print("[PASS] healthy case: no exception raised")

# Degenerate case: predictions take only 2 distinct values per label (resolution
# collapsed), while the reference feature still varies continuously per label.
collapsed_preds = np.tile(np.repeat([0.1, 0.9], n_rows // 2), n_labels)
healthy_ref2 = rng.random(n_rows * n_labels)
try:
    _check_label_resolution(collapsed_preds, healthy_ref2, n_labels)
    raise SystemExit("[FAIL] expected AssertionError on the collapsed case")
except AssertionError as e:
    print(f"[PASS] collapsed case correctly raised: {e}")
