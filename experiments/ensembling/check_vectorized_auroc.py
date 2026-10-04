"""Standalone check (no pytest in this repo): proves _vectorized_per_label_auroc()
(the Wilcoxon-Mann-Whitney rank-sum identity for AUROC, computed for every label at
once via a single rankdata(axis=0) call) matches sklearn.metrics.roc_auc_score
exactly, column by column, including with ties -- before trusting it to replace a
per-term roc_auc_score() loop in cv_score() (whose fixed per-call overhead, not just
the earlier O(n_terms*n_rows) masking bug, was still costing ~9s for just 2000 labels
after that bug was fixed). Run via:
    env/bin/python experiments/ensembling/check_vectorized_auroc.py
"""
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from local_hpo_search import _vectorized_per_label_auroc

n_rows, n_labels = 200, 500
rng = np.random.default_rng(42)

labels = (rng.random((n_rows, n_labels)) < 0.3).astype(np.float32)
preds = labels + rng.normal(scale=0.5, size=labels.shape)
# Force real ties by rounding some columns to a coarse grid.
preds[:, ::3] = np.round(preds[:, ::3] * 2) / 2

vectorized = _vectorized_per_label_auroc(preds, labels)
assert vectorized.shape == (n_labels,), vectorized.shape

for j in range(n_labels):
    col_labels = labels[:, j]
    if col_labels.min() == col_labels.max():
        assert np.isnan(vectorized[j]), f"col {j} has one class, expected NaN, got {vectorized[j]}"
        continue
    expected = roc_auc_score(col_labels, preds[:, j])
    assert abs(vectorized[j] - expected) < 1e-9, (
        f"col {j}: vectorized={vectorized[j]:.6f} vs sklearn={expected:.6f}"
    )
print(f"[PASS] vectorized AUROC matches sklearn.roc_auc_score exactly on all {n_labels} columns (incl. ties)")

# Performance: at n_labels=2000 (smaller than this check's 500), the old per-term
# sklearn-call loop took ~9s even after the masking bug was fixed. This must be a
# small fraction of that at a comparable or larger scale.
n_labels_big = 2000
labels_big = (rng.random((n_rows, n_labels_big)) < 0.3).astype(np.float32)
preds_big = labels_big + rng.normal(scale=0.5, size=labels_big.shape)
t0 = time.time()
_vectorized_per_label_auroc(preds_big, labels_big)
elapsed = time.time() - t0
assert elapsed < 1.0, f"vectorized AUROC took {elapsed:.2f}s for {n_labels_big} labels -- expected well under 1s"
print(f"[PASS] vectorized AUROC on {n_labels_big} labels took {elapsed:.3f}s (vs. ~9s for the per-term loop)")
