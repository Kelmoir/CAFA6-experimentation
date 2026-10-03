"""Standalone check (no pytest in this repo): fits the meta-model on the cheap,
already-cached meta-eval slice. Originally written to confirm predictions aren't
collapsed to a pooled constant; the final whole-branch review (2026-10-02) found that
check alone couldn't detect per-label ranking-resolution collapse, which the old
pooled binary:logistic fit actually had on this slice (see cafa6_next_steps_plan.md's
corrected write-up and _check_label_resolution() in stacking_pipeline.py). Now that
fit_meta_model fits a rank:pairwise objective grouped by term instead, this confirms
the resolution-collapse guard passes cleanly on this slice -- i.e. the fix actually
fixes the failure this guard exists to catch, not just that the guard still fires.
Run via:
    env/bin/python experiments/ensembling/check_fit_degeneracy.py
"""
from stacking_pipeline import build_features, fit_meta_model, load_cached_meta_eval

meta_eval = load_cached_meta_eval()
X, y, qid = build_features(meta_eval)
n_labels = len(meta_eval["label_names"])

fit_meta_model(X, y, qid, n_labels)
print("[PASS] fit_meta_model check: resolution-collapse guard passes cleanly with the rank:pairwise objective")
