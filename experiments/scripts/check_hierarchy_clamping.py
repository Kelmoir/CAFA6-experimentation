"""Standalone check for clamp_hierarchy_consistency (Tier 2 item 5 of
experiments/cafa6_next_steps_plan.md: GO-DAG-consistency post-processing at inference).
Pure numpy, no model needed. Run directly:
    env/bin/python experiments/scripts/check_hierarchy_clamping.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from training_run import clamp_hierarchy_consistency, compute_hierarchy_violations


def check_single_violation_gets_clamped_up():
    probs = np.array([[0.9, 0.3]])  # child(0)=0.9 > parent(1)=0.3 -- a violation
    edge_index = np.array([[0, 1]])
    clamped = clamp_hierarchy_consistency(probs, edge_index)
    assert clamped[0, 1] == 0.9, clamped
    assert clamped[0, 0] == 0.9, clamped  # child untouched
    print("[OK] a single direct violation is clamped: parent raised to match child")


def check_already_consistent_input_is_unchanged():
    probs = np.array([[0.2, 0.9]])  # child(0)=0.2 <= parent(1)=0.9 -- already consistent
    edge_index = np.array([[0, 1]])
    clamped = clamp_hierarchy_consistency(probs, edge_index)
    assert np.allclose(clamped, probs), clamped
    print("[OK] already-consistent input is left unchanged")


def check_multi_hop_chain_propagates_fully():
    # chain: 0 (deepest child) -> 1 -> 2 (root). 0's high score must propagate all the way
    # up to 2, even though edge_index only lists the two DIRECT edges (0,1) and (1,2).
    probs = np.array([[0.9, 0.1, 0.05]])
    edge_index = np.array([[0, 1], [1, 2]])
    clamped = clamp_hierarchy_consistency(probs, edge_index)
    assert clamped[0, 0] == 0.9, clamped
    assert clamped[0, 1] == 0.9, clamped
    assert clamped[0, 2] == 0.9, clamped
    print("[OK] a multi-hop chain fully propagates the max up to the root in one call")


def check_multi_parent_term_raises_both_parents():
    # term 0 has two direct parents, 1 and 2 -- both must be raised independently.
    probs = np.array([[0.9, 0.1, 0.2]])
    edge_index = np.array([[0, 1], [0, 2]])
    clamped = clamp_hierarchy_consistency(probs, edge_index)
    assert clamped[0, 1] == 0.9, clamped
    assert clamped[0, 2] == 0.9, clamped
    print("[OK] a term with two direct parents raises both of them")


def check_shared_parent_with_multiple_children_takes_the_max_of_all():
    # parent (idx 2) has two direct children (0 and 1) via two separate edges. A naive
    # fancy-index assignment is last-write-wins on duplicate targets and would silently
    # drop child 0's contribution here -- this pins the fix (np.maximum.at).
    probs = np.array([[0.9, 0.3, 0.05]])
    edge_index = np.array([[0, 2], [1, 2]])
    clamped = clamp_hierarchy_consistency(probs, edge_index)
    assert clamped[0, 2] == 0.9, clamped
    print("[OK] a parent with multiple direct children takes the max over ALL of them")


def check_clamping_never_mutates_the_input_array():
    probs = np.array([[0.9, 0.3]])
    original = probs.copy()
    edge_index = np.array([[0, 1]])
    clamp_hierarchy_consistency(probs, edge_index)
    assert np.array_equal(probs, original), probs
    print("[OK] clamp_hierarchy_consistency never mutates the caller's array")


def check_empty_edge_index_is_a_no_op():
    probs = np.array([[0.9, 0.3]])
    edge_index = np.zeros((0, 2), dtype=int)
    clamped = clamp_hierarchy_consistency(probs, edge_index)
    assert np.array_equal(clamped, probs), clamped
    print("[OK] empty edge_index is a no-op")


def check_result_has_zero_violations_after_clamping():
    rng = np.random.default_rng(42)
    probs = rng.random((50, 10))
    edge_index = np.array([[0, 1], [1, 2], [2, 3], [4, 3], [5, 3]])
    clamped = clamp_hierarchy_consistency(probs, edge_index)
    result = compute_hierarchy_violations(clamped, edge_index)
    assert result["violation_rate"] == 0.0, result
    print("[OK] violation_rate is exactly 0.0 on clamped probabilities, even with random inputs")


def check_max_iterations_bound_is_not_silently_wrong():
    # A chain of depth 5 needs at least 4 sweeps to fully propagate end to end; a
    # generous default (32) must still converge correctly on a deeper-than-usual chain.
    n = 6
    edge_index = np.array([[i, i + 1] for i in range(n - 1)])
    probs = np.zeros((1, n))
    probs[0, 0] = 0.95
    clamped = clamp_hierarchy_consistency(probs, edge_index, max_iterations=32)
    assert np.all(clamped[0] == 0.95), clamped
    print("[OK] a longer chain still fully converges within the default max_iterations")


if __name__ == "__main__":
    check_single_violation_gets_clamped_up()
    check_already_consistent_input_is_unchanged()
    check_multi_hop_chain_propagates_fully()
    check_multi_parent_term_raises_both_parents()
    check_shared_parent_with_multiple_children_takes_the_max_of_all()
    check_clamping_never_mutates_the_input_array()
    check_empty_edge_index_is_a_no_op()
    check_result_has_zero_violations_after_clamping()
    check_max_iterations_bound_is_not_silently_wrong()
    print("[OK] all checks passed")
