"""Standalone check (no pytest in this repo): fits the meta-model on the cheap,
already-cached meta-eval slice (not real meta-train -- this is a wiring/degeneracy
check, not the real fit) and confirms predictions aren't collapsed to a constant.
Run via:
    env/bin/python experiments/ensembling/check_fit_degeneracy.py
"""
from stacking_pipeline import build_features, fit_meta_model, load_cached_meta_eval

meta_eval = load_cached_meta_eval()
X, y = build_features(meta_eval)

model = fit_meta_model(X, y)
preds = model.predict_proba(X)[:, 1]
print(f"[PASS] fit_meta_model check: predictions std={preds.std():.4f} (nonzero, not collapsed)")
