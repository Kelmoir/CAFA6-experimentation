"""One-off diagnostic probe (not a correctness check): resolves whether the 639-term
"bright-spot" subset (terms where BLAST-KNN beat the ESM-2 model by >=0.05 AUROC on
meta-eval) reflects a real, generalizable term-level property, or is a selection-bias/
winner's-curse artifact of defining the subset using the same 2000-row sample it was
then scored on. Freezes the bright-spot term *indices* exactly as derived from
meta-eval (no re-selection on new data), then evaluates model/BLAST/baseline/stacked
AUROC on an independent 5000-row slice of the held-out "test" split
(generate_dataset.py's locally-generated labeled test split -- real ground-truth
labels, distinct from Kaggle's own unlabeled submission test set). If BLAST's edge
over the model replicates out-of-sample on these pre-defined terms, the bright spot is
real and a better ranking approach could exploit it; if it doesn't, it was likely a
winner's-curse artifact of the small meta-eval sample.

Run via: env/bin/python experiments/ensembling/bright_spot_test_set_probe.py
(resumable -- BLAST/inference progress checkpoints to blast_work/, safe to re-run
after a timeout)
"""
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score

from stacking_pipeline import (
    DATA_PATH, META_TRAIN_END, META_TRAIN_START, _per_term_rank_average,
    build_features, compute_meta_train_slice, fit_meta_model, load_cached_meta_eval,
)

N_TEST_SAMPLES = 5000

print("[INFO] Loading dataset once for both the meta-train fit and the test probe...")
dataset = load_from_disk(str(DATA_PATH))

print("[INFO] Loading cached meta-eval slice and freezing the bright-spot term indices...")
meta_eval = load_cached_meta_eval(dataset=dataset)
val_labels_eval = meta_eval["val_labels"]
n_labels = val_labels_eval.shape[1]
evaluable_eval = [j for j in range(n_labels) if val_labels_eval[:, j].min() != val_labels_eval[:, j].max()]
model_auroc_eval = np.array(
    [roc_auc_score(val_labels_eval[:, j], meta_eval["esm_prob"][:, j]) for j in evaluable_eval]
)
blast_auroc_eval = np.array(
    [roc_auc_score(val_labels_eval[:, j], meta_eval["blast_score"][:, j]) for j in evaluable_eval]
)
diff_eval = blast_auroc_eval - model_auroc_eval
bright_spot_terms = [evaluable_eval[i] for i in range(len(evaluable_eval)) if diff_eval[i] >= 0.05]
assert len(bright_spot_terms) == 639, f"expected 639 bright-spot terms (per the logged result), got {len(bright_spot_terms)}"
print(f"[INFO] Frozen {len(bright_spot_terms)} bright-spot term indices from meta-eval (not re-derived from test).")
print(
    f"[REFERENCE] on meta-eval itself, these terms: model mean_auroc="
    f"{model_auroc_eval[[evaluable_eval.index(j) for j in bright_spot_terms]].mean():.4f} "
    f"blast mean_auroc={blast_auroc_eval[[evaluable_eval.index(j) for j in bright_spot_terms]].mean():.4f}"
)

print(f"[INFO] Computing/loading meta-train slice (rows {META_TRAIN_START}:{META_TRAIN_END}) to fit the real stacked model...")
meta_train = compute_meta_train_slice(META_TRAIN_START, META_TRAIN_END, dataset=dataset)
X_train, y_train, qid_train = build_features(meta_train)
print("[INFO] Fitting the real XGBoost ranking meta-model on meta-train...")
model = fit_meta_model(X_train, y_train, qid_train, n_labels)

print(f"[INFO] Computing/loading test slice (rows 0:{N_TEST_SAMPLES}) -- the independent sample...")
test_slice = compute_meta_train_slice(0, N_TEST_SAMPLES, dataset=dataset, split_name="test")
X_test, _, _ = build_features(test_slice)
assert np.isfinite(X_test).all(), "non-finite value in test-slice features"

stacked_test_scores = model.predict(X_test).reshape(N_TEST_SAMPLES, n_labels)
baseline_test_scores = _per_term_rank_average(test_slice["esm_prob"], test_slice["blast_score"])

val_labels_test = test_slice["val_labels"]
evaluable_test = {j for j in range(n_labels) if val_labels_test[:, j].min() != val_labels_test[:, j].max()}
bright_spot_terms_set = set(bright_spot_terms)
bright_spot_evaluable_on_test = [j for j in bright_spot_terms if j in evaluable_test]
other_evaluable_on_test = [j for j in evaluable_test if j not in bright_spot_terms_set]

print(
    f"[RESULT] {len(bright_spot_evaluable_on_test)}/{len(bright_spot_terms)} frozen bright-spot "
    f"terms have both classes present (are evaluable) in this {N_TEST_SAMPLES}-row test sample"
)


def _aurocs(score_matrix, term_indices):
    return np.array([roc_auc_score(val_labels_test[:, j], score_matrix[:, j]) for j in term_indices])


summaries = {}
for label, subset in [
    ("bright-spot (frozen from meta-eval)", bright_spot_evaluable_on_test),
    ("other evaluable test terms", other_evaluable_on_test),
]:
    if not subset:
        print(f"[RESULT] on {label}: no evaluable terms in this test sample")
        continue
    model_a = _aurocs(test_slice["esm_prob"], subset)
    blast_a = _aurocs(test_slice["blast_score"], subset)
    stacked_a = _aurocs(stacked_test_scores, subset)
    baseline_a = _aurocs(baseline_test_scores, subset)
    summaries[label] = (model_a, blast_a, stacked_a, baseline_a)
    print(
        f"[RESULT] on {label} ({len(subset)} terms): "
        f"model mean={model_a.mean():.4f} median={np.median(model_a):.4f} | "
        f"blast mean={blast_a.mean():.4f} median={np.median(blast_a):.4f} | "
        f"stacked mean={stacked_a.mean():.4f} median={np.median(stacked_a):.4f} | "
        f"baseline mean={baseline_a.mean():.4f} median={np.median(baseline_a):.4f}"
    )

if "bright-spot (frozen from meta-eval)" in summaries:
    model_a, blast_a, _, _ = summaries["bright-spot (frozen from meta-eval)"]
    test_diff = blast_a.mean() - model_a.mean()
    eval_diff = diff_eval[[evaluable_eval.index(j) for j in bright_spot_evaluable_on_test]].mean()
    replicates = test_diff >= 0.03  # same direction, not fully decayed
    print(
        f"[VERDICT] BLAST-beats-model gap on these frozen terms: meta-eval={eval_diff:+.4f}, "
        f"test={test_diff:+.4f} -- {'REPLICATES out-of-sample' if replicates else 'DOES NOT CLEARLY REPLICATE (likely winner-s-curse on meta-eval)'}"
    )
