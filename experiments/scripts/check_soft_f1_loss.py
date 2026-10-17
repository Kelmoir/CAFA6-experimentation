"""Standalone check for soft_f1_loss (Tier 2 item 4: IA-weighted soft-F1 loss, protein-level
only -- see experiments/cafa6_next_steps_plan.md). No GPU/dataset needed.

Run directly:
    env/bin/python experiments/scripts/check_soft_f1_loss.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_run


def check_perfect_predictions_give_near_zero_loss():
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    logits = torch.tensor([[20.0, -20.0, 20.0, -20.0]])  # sigmoid -> ~1, ~0, ~1, ~0
    ia_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss = training_run.soft_f1_loss(logits, labels, ia_weights)
    assert loss.item() < 1e-3, loss.item()
    print("[OK] near-perfect predictions give near-zero soft-F1 loss")


def check_inverted_predictions_give_near_max_loss():
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    logits = torch.tensor([[-20.0, 20.0, -20.0, 20.0]])  # every prediction flipped
    ia_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss = training_run.soft_f1_loss(logits, labels, ia_weights)
    assert loss.item() > 0.99, loss.item()
    print("[OK] fully inverted predictions give near-maximal (~1.0) soft-F1 loss")


def check_ia_weighting_changes_the_result():
    labels = torch.tensor([[1.0, 1.0, 0.0]])
    logits = torch.tensor([[10.0, -5.0, 5.0]])  # term 0 right, term 1 wrong, term 2 false positive
    uniform = training_run.soft_f1_loss(logits, labels, torch.tensor([1.0, 1.0, 1.0]))
    skewed = training_run.soft_f1_loss(logits, labels, torch.tensor([1.0, 10.0, 0.1]))
    assert abs(uniform.item() - skewed.item()) > 1e-3, (uniform.item(), skewed.item())
    print("[OK] IA weighting actually changes the loss value vs. uniform weights")


def check_gradient_flows_and_is_finite():
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    logits = torch.tensor([[0.3, -0.1, 0.8]], requires_grad=True)
    ia_weights = torch.tensor([2.0, 1.0, 5.0])
    loss = training_run.soft_f1_loss(logits, labels, ia_weights)
    loss.backward()
    assert torch.isfinite(logits.grad).all(), logits.grad
    assert logits.grad.abs().sum() > 0
    print("[OK] gradient flows back to logits and is finite")


def check_rows_with_no_true_annotation_are_excluded_not_eps_dominated():
    # Row 0 has a true annotation and a perfect prediction -> contributes ~0.
    # Row 1 has NO true annotation among scored labels (all-zero labels) -- should be
    # excluded from the mean entirely, not contribute a meaningless ~1.0 via the eps term.
    labels = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    logits = torch.tensor([[20.0, -20.0], [-20.0, -20.0]])
    ia_weights = torch.tensor([1.0, 1.0])
    loss = training_run.soft_f1_loss(logits, labels, ia_weights)
    assert loss.item() < 1e-3, loss.item()
    print("[OK] rows with zero IA-weighted true annotation are excluded from the mean")


def check_all_rows_unscored_returns_zero_not_nan():
    labels = torch.zeros((2, 3))
    logits = torch.zeros((2, 3))
    ia_weights = torch.tensor([1.0, 1.0, 1.0])
    loss = training_run.soft_f1_loss(logits, labels, ia_weights)
    assert loss.item() == 0.0, loss.item()
    print("[OK] a batch with no scored rows at all returns 0.0, not NaN")


def check_load_ia_weights_importable_from_both_modules_no_circular_import():
    # The real regression this guards: evaluate_cafa_score imports training_run at module
    # level, so training_run must never import FROM evaluate_cafa_score.
    import evaluate_cafa_score
    assert evaluate_cafa_score.load_ia_weights is training_run.load_ia_weights
    print("[OK] load_ia_weights lives in training_run.py and evaluate_cafa_score imports it back (no circular import)")


if __name__ == "__main__":
    check_perfect_predictions_give_near_zero_loss()
    check_inverted_predictions_give_near_max_loss()
    check_ia_weighting_changes_the_result()
    check_gradient_flows_and_is_finite()
    check_rows_with_no_true_annotation_are_excluded_not_eps_dominated()
    check_all_rows_unscored_returns_zero_not_nan()
    check_load_ia_weights_importable_from_both_modules_no_circular_import()
    print("[OK] all checks passed")
