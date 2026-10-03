"""One-off diagnostic probe (not a correctness check): confirms or refutes the
diagnosed mechanism for why the trained XGBRanker underperformed the untrained
rank-average baseline on raw features -- a pointwise tree scorer can't infer each
term's wildly different raw-value scale/sparsity from raw values alone (confirmed
directly, 2026-10-03: blast_score is nonzero for <1% of rows on 6061/9401 evaluable
terms, but >50% of rows on others). Reuses the exact same pipeline as main()
(stacking_pipeline.py) -- same meta-train/meta-eval slices, same fit_meta_model(),
same report_results() -- swapping only build_features() for build_rank_features(),
which rank-transforms esm_prob/blast_score per term before handing them to the model.
If this closes (or reverses) the gap to the baseline, the mechanism is confirmed and
a better algorithm genuinely could help; if it doesn't, the baseline's edge is coming
from somewhere else.

Run via: env/bin/python experiments/ensembling/rank_features_probe.py
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

print("[INFO] Building per-term RANK-TRANSFORMED features (not raw values)...")
X_train, y_train, qid_train = build_rank_features(meta_train)
X_eval, _, _ = build_rank_features(meta_eval)

print("[INFO] Fitting XGBoost ranking meta-model on rank-transformed meta-train features...")
model = fit_meta_model(X_train, y_train, qid_train, len(meta_train["label_names"]))

print("[INFO] Evaluating on meta-eval (report_results() recomputes the raw-feature "
      "model/blast/baseline numbers from meta_eval directly -- only 'stacked' here "
      "reflects the rank-transformed-feature fit)...")
report_results(model, X_eval, meta_eval)
