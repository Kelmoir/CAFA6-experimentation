"""Standalone check (no pytest in this repo): fits the meta-model on the cheap,
already-cached meta-eval slice. Originally written to confirm predictions aren't
collapsed to a pooled constant -- the final whole-branch review (2026-10-02) found
that check alone can't detect per-label ranking-resolution collapse, which this exact
slice actually has (see cafa6_next_steps_plan.md's corrected write-up and
_check_label_resolution() in stacking_pipeline.py). This check now confirms the
strengthened guard correctly raises on this real, previously-undetected failure.
Run via:
    env/bin/python experiments/ensembling/check_fit_degeneracy.py
"""
from stacking_pipeline import build_features, fit_meta_model, load_cached_meta_eval

meta_eval = load_cached_meta_eval()
X, y = build_features(meta_eval)
n_labels = len(meta_eval["label_names"])

try:
    fit_meta_model(X, y, n_labels)
    raise SystemExit(
        "[FAIL] expected fit_meta_model to raise on this slice's real per-label "
        "resolution collapse -- did the guard regress?"
    )
except AssertionError as e:
    assert "resolution collapsed" in str(e), f"raised for the wrong reason: {e}"
    print(f"[PASS] fit_meta_model check: resolution-collapse guard correctly raised: {e}")
