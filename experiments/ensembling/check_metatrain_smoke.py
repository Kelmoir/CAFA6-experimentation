"""Standalone smoke check (no pytest in this repo): runs compute_meta_train_slice() on
a tiny 10-row range so BLAST+inference finish in seconds, to validate the function's
wiring before spending real wall-clock on the full 2000-row meta-train range (Task 3's
real run). Run via:
    env/bin/python experiments/ensembling/check_metatrain_smoke.py
"""
import numpy as np

from stacking_pipeline import MAX_VAL_SAMPLES, compute_meta_train_slice

start, end = MAX_VAL_SAMPLES, MAX_VAL_SAMPLES + 10
result = compute_meta_train_slice(start, end)

n_rows = end - start
n_labels = len(result["label_names"])
assert result["esm_prob"].shape == (n_rows, n_labels), result["esm_prob"].shape
assert result["blast_score"].shape == (n_rows, n_labels), result["blast_score"].shape
assert result["val_labels"].shape == (n_rows, n_labels), result["val_labels"].shape
assert np.isfinite(result["esm_prob"]).all(), "non-finite value in smoke esm_prob"
assert np.isfinite(result["blast_score"]).all(), "non-finite value in smoke blast_score"

print(f"[PASS] metatrain smoke check: shapes and finiteness OK for rows [{start}:{end})")
