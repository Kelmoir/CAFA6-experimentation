"""Standalone check (no pytest in this repo): two real-data proofs against the cheap,
already-cached meta-eval slice.

1. fit_meta_model() (the real rank:pairwise fit) must pass its own in-sample
   degeneracy guard cleanly.
2. The guard itself (_check_ranking_signal()) must still correctly raise on the
   original, historically-collapsed pooled XGBClassifier(binary:logistic) fit. A
   final-review pass (2026-10-03) found the first version of this check deleted
   exactly this proof when the guard's threshold changed from an absolute cutoff to
   a relative one -- and the *absolute* cutoff (median AUROC > 0.55) it replaced had
   turned out to pass the real collapsed classifier undetected (in-sample median
   AUROC ~0.70, comfortably above 0.55). This check fits that classifier for real
   and confirms the relative guard does catch it.

Run via:
    env/bin/python experiments/ensembling/check_fit_degeneracy.py
"""
from stacking_pipeline import _check_ranking_signal, build_features, fit_meta_model, load_cached_meta_eval

meta_eval = load_cached_meta_eval()
X, y, qid = build_features(meta_eval)
n_labels = len(meta_eval["label_names"])

fit_meta_model(X, y, qid, n_labels)
print("[PASS] fit_meta_model (rank:pairwise) passes its own in-sample degeneracy guard cleanly")

import xgboost as xgb  # noqa: E402

old_model = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", random_state=42)
old_model.fit(X, y)
old_preds = old_model.predict_proba(X)[:, 1]
try:
    _check_ranking_signal(old_preds, y, X[:, 0], n_labels)
    raise SystemExit(
        "[FAIL] expected _check_ranking_signal to raise on the original collapsed "
        "pooled classifier -- did the guard regress again?"
    )
except AssertionError as e:
    print(f"[PASS] guard correctly raises on the real historically-collapsed classifier: {e}")
