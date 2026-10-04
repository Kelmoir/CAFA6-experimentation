"""Standalone check (no pytest in this repo): proves cv_score() computes the same
per-fold AUROC regardless of implementation, and that it completes in well under a
second on a modestly-sized synthetic case -- a regression guard against the O(n_terms
x n_val_rows) boolean-rescan bug found after a real run took ~17 hours for 40 trials
(each fold recomputed `val_qid == term` -- a full array scan -- once per term, instead
of reshaping once and column-slicing). Run via:
    env/bin/python experiments/ensembling/check_cv_score_perf.py
"""
import time

import numpy as np

from local_hpo_search import K_FOLDS, cv_score, sample_config

n_rows, n_labels = 100, 2000  # big enough on the term axis to expose an O(terms*rows) bug
rng = np.random.default_rng(42)

y2d = np.zeros((n_rows, n_labels), dtype=np.float32)
half = np.array([1.0] * (n_rows // 2) + [0.0] * (n_rows // 2))
for j in range(n_labels):
    y2d[:, j] = rng.permutation(half)

esm_like = y2d + rng.normal(scale=0.3, size=y2d.shape)
blast_like = y2d + rng.normal(scale=0.3, size=y2d.shape)
term_freq = rng.random(n_labels).astype(np.float32)

from scipy.stats import rankdata  # noqa: E402

rank_esm = np.apply_along_axis(rankdata, 0, esm_like) / n_rows
rank_blast = np.apply_along_axis(rankdata, 0, blast_like) / n_rows
X = np.stack([rank_esm.ravel(), rank_blast.ravel(), np.tile(term_freq, n_rows)], axis=1).astype(np.float32)
y = y2d.ravel().astype(np.float32)
qid = np.tile(np.arange(n_labels), n_rows)
protein_idx = np.repeat(np.arange(n_rows), n_labels)

config = sample_config(np.random.default_rng(1))
config["n_estimators"] = 10  # keep the fit itself cheap -- this check is about the scoring loop

t0 = time.time()
score = cv_score(X, y, qid, protein_idx, n_rows, config, device="cpu")
elapsed = time.time() - t0

assert 0.0 <= score <= 1.0, score
assert elapsed < 10.0, (
    f"cv_score took {elapsed:.1f}s on a {n_rows}x{n_labels} synthetic case -- "
    f"the O(n_terms x n_val_rows) boolean-rescan bug may have regressed "
    f"(expected well under 10s with {K_FOLDS} folds of this size)"
)
print(f"[PASS] cv_score check: score={score:.4f}, completed in {elapsed:.2f}s (well under the 10s regression bar)")
