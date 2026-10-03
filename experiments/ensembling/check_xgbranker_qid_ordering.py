"""Standalone check (no pytest in this repo): verifies, against the real installed
xgboost (3.2.0), what XGBRanker.fit(X, y, qid=...) actually requires of row order --
the stacking pipeline's flattened (protein, term) rows are row-major by protein, so
same-term rows are scattered, not contiguous. Docs across xgboost versions have been
inconsistent about whether pre-sorting is required, so this confirms it directly
rather than assuming (same spirit as the peft/LoRA checks noted in CLAUDE.md).

First finding (confirmed, not assumed): this xgboost build raises
`XGBoostError: qid must be sorted in non-decreasing order` on unsorted qid --
pre-sorting by qid before fit is required, not optional. This check sorts by qid
(stable, so within-group row order is preserved) before fitting, then confirms
rank:pairwise actually recovers a per-group ranking signal on synthetic data where
groups have different base rates -- the exact scenario that collapsed the pooled
binary:logistic fit in the real result.

Run via: env/bin/python experiments/ensembling/check_xgbranker_qid_ordering.py
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)

n_groups, n_rows_per_group = 20, 80
qid = np.repeat(np.arange(n_groups), n_rows_per_group)
# Per-group base rate varies a lot (mimics term_freq varying across GO terms) --
# a pooled fit can "solve" this by base rate alone without learning within-group order.
base_rate = rng.uniform(0.05, 0.95, size=n_groups)
signal = rng.random(n_groups * n_rows_per_group)  # the real within-group ranking signal
noise = rng.normal(scale=0.05, size=n_groups * n_rows_per_group)
X = np.stack([signal + noise, np.repeat(base_rate, n_rows_per_group)], axis=1).astype(np.float32)
y = (signal > (1 - base_rate[qid])).astype(np.float32)  # higher base rate -> easier to be positive

# Shuffle rows so same-qid rows start out scattered, not contiguous -- matches the
# real pipeline's row-major-by-protein layout.
perm = rng.permutation(len(y))
X_shuf, y_shuf, qid_shuf = X[perm], y[perm], qid[perm]

try:
    xgb.XGBRanker(objective="rank:pairwise", tree_method="hist", random_state=42).fit(
        X_shuf, y_shuf, qid=qid_shuf
    )
    raise SystemExit("[FAIL] expected XGBoostError on unsorted qid -- did this xgboost version change?")
except xgb.core.XGBoostError as e:
    assert "sorted in non-decreasing order" in str(e), f"raised for a different reason: {e}"
    print("[INFO] confirmed: this xgboost build requires qid pre-sorted before fit (not optional)")

# Sort by qid (stable, so ties preserve shuffled order -- proves sorting alone is
# sufficient, no other row-order assumption needed) and retry.
order = np.argsort(qid_shuf, kind="stable")
X_sorted, y_sorted, qid_sorted = X_shuf[order], y_shuf[order], qid_shuf[order]

model = xgb.XGBRanker(objective="rank:pairwise", tree_method="hist", random_state=42)
model.fit(X_sorted, y_sorted, qid=qid_sorted)
preds_sorted = model.predict(X_sorted)

# Map predictions back to the original row identity (via perm/order) to score
# per-group AUROC against ground truth.
preds = np.empty_like(preds_sorted)
preds[perm[order]] = preds_sorted
per_group_auroc = np.array([
    roc_auc_score(y[qid == g], preds[qid == g]) for g in range(n_groups)
])

assert per_group_auroc.mean() > 0.9, (
    f"XGBRanker with sorted qid failed to learn within-group ranking: "
    f"mean per-group AUROC={per_group_auroc.mean():.4f} -- rank:pairwise isn't "
    "working as expected even with correctly sorted input"
)
print(
    f"[PASS] XGBRanker.fit(qid=...) with qid-sorted rows correctly ranks within "
    f"groups: mean per-group AUROC={per_group_auroc.mean():.4f}"
)
