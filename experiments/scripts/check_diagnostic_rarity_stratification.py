"""Standalone check for _diagnostic_metrics_from_arrays's label_is_rare parameter (Task 3
of docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md) -- a
rarity-stratified AUROC breakdown, added because this item's claimed effect (a GO-DAG
hierarchy-consistency loss) concentrates on rare terms and a pooled/mean-over-everything
number could hide it (see docs/superpowers/specs/2026-10-11-go-dag-hierarchy-consistency-
design.md and [[feedback_check_metric_can_see_the_effect]]). Pure-numpy, no model needed.
Run directly:
    env/bin/python experiments/scripts/check_diagnostic_rarity_stratification.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.metrics import roc_auc_score

from training_run import _diagnostic_metrics_from_arrays


def check_label_is_rare_none_is_byte_for_byte_unchanged():
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    probs = np.array([[0.9, 0.2], [0.1, 0.8], [0.1, 0.9], [0.9, 0.1]], dtype=float)
    result_without, _ = _diagnostic_metrics_from_arrays(probs, labels)
    result_with_none, _ = _diagnostic_metrics_from_arrays(probs, labels, label_is_rare=None)
    assert result_without == result_with_none, (result_without, result_with_none)
    assert "rare_mean_auroc" not in result_without, result_without
    print("[OK] omitting label_is_rare (or passing None) adds no new keys -- unchanged behavior")


def check_rare_and_common_buckets_match_hand_computed_values():
    # 3 labels: label 0 (rare), label 1 (rare), label 2 (common).
    labels = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    probs = np.array([
        [0.9, 0.2, 0.9],
        [0.1, 0.8, 0.1],
        [0.2, 0.9, 0.8],
        [0.8, 0.1, 0.2],
    ], dtype=float)
    label_is_rare = np.array([True, True, False])
    expected_auroc = [roc_auc_score(labels[:, j], probs[:, j]) for j in range(3)]
    result, _ = _diagnostic_metrics_from_arrays(probs, labels, label_is_rare=label_is_rare)
    expected_rare_mean = np.mean(expected_auroc[:2])
    expected_common_mean = expected_auroc[2]
    assert abs(result["rare_mean_auroc"] - expected_rare_mean) < 1e-9, result
    assert abs(result["common_mean_auroc"] - expected_common_mean) < 1e-9, result
    assert result["rare_n_labels_evaluable"] == 2, result
    assert result["common_n_labels_evaluable"] == 1, result
    print(f"[OK] rare/common buckets match hand-computed AUROC "
          f"(rare_mean={result['rare_mean_auroc']:.4f}, common_mean={result['common_mean_auroc']:.4f})")


def check_empty_rare_bucket_reports_none_not_crash():
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    probs = np.array([[0.9, 0.2], [0.1, 0.8], [0.1, 0.9], [0.9, 0.1]], dtype=float)
    label_is_rare = np.array([False, False])  # no rare labels at all
    result, _ = _diagnostic_metrics_from_arrays(probs, labels, label_is_rare=label_is_rare)
    assert result["rare_mean_auroc"] is None, result
    assert result["rare_n_labels_evaluable"] == 0, result
    print("[OK] an empty rare bucket reports None/0, not a crash")


if __name__ == "__main__":
    check_label_is_rare_none_is_byte_for_byte_unchanged()
    check_rare_and_common_buckets_match_hand_computed_values()
    check_empty_rare_bucket_reports_none_not_crash()
    print("[OK] all checks passed")
