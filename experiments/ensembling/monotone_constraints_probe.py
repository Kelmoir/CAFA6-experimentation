"""One-off diagnostic probe (not a correctness check): tests whether monotone
constraints close the gap to the untrained rank-average baseline, out-of-sample.
Builds directly on rank_features_probe.py's result (stacked mean/median AUROC
0.8688/0.9225, worse than raw features, still below both model-alone and baseline)
and the follow-up in-sample check that ruled out overfitting and raw-scale
miscalibration as explanations, pointing at tree-ensemble non-monotonicity instead.
A quick in-sample check with monotone_constraints=(1,1,0) on rank features already
showed the model beating the baseline in-sample (0.8861 vs. 0.8853) for the first
time -- this script tests whether that holds out-of-sample, which is what actually
matters. Reuses the exact same pipeline as rank_features_probe.py, adding only
monotone_constraints to fit_meta_model() -- one variable changed.

Run via: env/bin/python experiments/ensembling/monotone_constraints_probe.py
"""
from datasets import load_from_disk

from stacking_pipeline import (
    DATA_PATH, META_TRAIN_END, META_TRAIN_START, assert_meta_eval_matches_logged,
    build_rank_features, compute_meta_train_slice, fit_meta_model, load_cached_meta_eval,
    report_results,
)

dataset = load_from_disk(str(DATA_PATH))

print("[INFO] Loading cached meta-eval slice (rows 0:2000)...")
meta_eval = load_cached_meta_eval(dataset=dataset)
assert_meta_eval_matches_logged(meta_eval)

print(f"[INFO] Computing/loading meta-train slice (rows {META_TRAIN_START}:{META_TRAIN_END})...")
meta_train = compute_meta_train_slice(META_TRAIN_START, META_TRAIN_END, dataset=dataset)

print("[INFO] Building per-term rank-transformed features...")
X_train, y_train, qid_train = build_rank_features(meta_train)
X_eval, _, _ = build_rank_features(meta_eval)

print("[INFO] Fitting XGBoost ranking meta-model with monotone_constraints=(1,1,0) on meta-train...")
model = fit_meta_model(
    X_train, y_train, qid_train, len(meta_train["label_names"]), monotone_constraints=(1, 1, 0)
)

print("[INFO] Evaluating on meta-eval...")
report_results(model, X_eval, meta_eval)
