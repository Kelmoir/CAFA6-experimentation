"""Standalone check for run_diagnostic's loss_mask-aware metric computation (final-review
fix, Important #2 of docs/superpowers/plans/2026-10-08-known-annotation-conditioning.md's
code review: with --use-known-input, revealed/known cells are visible to the model as
INPUT and must not also be scored as diagnostic AUROC/F1 targets -- they're trivially
"predictable" since the model was literally handed them, which inflates/deflates the
comparison against a pure-NK diagnostic like trial-013's). Pure-numpy, no model needed.
Run directly:
    env/bin/python experiments/scripts/check_run_diagnostic_known_mask.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.metrics import roc_auc_score

from training_run import _diagnostic_metrics_from_arrays


def check_no_mask_matches_old_unmasked_behavior():
    # loss_mask=None must reproduce the exact pre-fix behavior: every cell scored.
    # Ground truth computed independently with sklearn directly over every cell
    # (the pre-fix formula), not a hand-derived number -- decouples this test from
    # arithmetic mistakes in picking the fixture.
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    probs = np.array([[0.9, 0.2], [0.1, 0.8], [0.1, 0.9], [0.9, 0.1]], dtype=float)
    expected_mean_auroc = np.mean([
        roc_auc_score(labels[:, j], probs[:, j]) for j in range(labels.shape[1])
    ])
    result, per_label_auroc = _diagnostic_metrics_from_arrays(probs, labels, loss_mask=None)
    assert abs(result["mean_auroc"] - expected_mean_auroc) < 1e-9, result
    assert result["n_val_examples"] == 4, result
    print(f"[OK] loss_mask=None scores every cell (mean_auroc={result['mean_auroc']:.4f}, "
          f"matches pre-fix unmasked formula)")


def check_masked_known_cells_are_excluded_from_metrics():
    # Rows 2-3 are deliberately WRONG (model "knows" these cells were handed as input,
    # not predicted) but fully masked out (loss_mask=0) -- must not drag the score down.
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    probs = np.array([[0.9, 0.2], [0.1, 0.8], [0.1, 0.9], [0.9, 0.1]], dtype=float)
    loss_mask = np.array([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=float)
    result, per_label_auroc = _diagnostic_metrics_from_arrays(probs, labels, loss_mask=loss_mask)
    assert abs(result["mean_auroc"] - 1.0) < 1e-9, result
    assert abs(result["pooled_auroc"] - 1.0) < 1e-9, result
    assert abs(result["best_f1"] - 1.0) < 1e-9, result
    print(f"[OK] masked-out known cells excluded -> mean_auroc={result['mean_auroc']:.4f} "
          f"(unmasked on these same deliberately-wrong-but-masked cells would be 0.375)")


def check_fully_masked_label_column_is_skipped_not_crashed():
    # Label 1's column is masked out in every row -> 0 unmasked examples for it ->
    # must be skipped as "not evaluable", not raise/crash on an empty array.
    labels = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    probs = np.array([[0.9, 0.2], [0.1, 0.8], [0.8, 0.3]], dtype=float)
    loss_mask = np.array([[1, 0], [1, 0], [1, 0]], dtype=float)
    result, per_label_auroc = _diagnostic_metrics_from_arrays(probs, labels, loss_mask=loss_mask)
    assert result["n_labels_evaluable"] == 1, result
    assert result["n_labels_total"] == 2, result
    print("[OK] a fully-masked label column is skipped (not evaluable), no crash")


if __name__ == "__main__":
    check_no_mask_matches_old_unmasked_behavior()
    check_masked_known_cells_are_excluded_from_metrics()
    check_fully_masked_label_column_is_skipped_not_crashed()
    print("[OK] all checks passed")
