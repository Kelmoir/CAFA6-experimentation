"""Standalone check for the GO-DAG true-path-rule consistency helpers (Task 1 of
docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md). Pure numpy/torch, no
model needed. Run directly:
    env/bin/python experiments/scripts/check_hierarchy_consistency.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx
import numpy as np
import torch

from training_run import (
    build_hierarchy_edge_index,
    compute_hierarchy_violations,
    hierarchy_consistency_penalty,
)


def _write_graph_fixture(path: Path):
    graph = nx.MultiDiGraph()
    graph.add_edge("GO:1", "GO:2", key="is_a")       # kept: both endpoints in-vocab
    graph.add_edge("GO:1", "GO:5", key="part_of")    # second parent of GO:1, also kept
    graph.add_edge("GO:1", "GO:4", key="part_of")    # dropped: GO:4 not in label2id
    graph.add_edge("GO:3", "GO:1", key="regulates")  # dropped: not a true-path relationship
    graph.add_edge("GO:2", "GO:6", key="is_a")       # chain: GO:1 -> GO:2 -> GO:6
    payload = {
        "graph": nx.readwrite.json_graph.node_link_data(graph, edges="links"),
        "true_path_relationships": ["is_a", "part_of"],
    }
    path.write_text(json.dumps(payload))


def check_build_hierarchy_edge_index_restricts_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "go_dependency_graph.json"
        _write_graph_fixture(graph_path)
        # GO:4 deliberately absent from label2id.
        label2id = {"GO:1": 0, "GO:2": 1, "GO:3": 2, "GO:5": 3, "GO:6": 4}
        edge_index = build_hierarchy_edge_index(label2id, graph_path)
        pairs = {tuple(row) for row in edge_index.tolist()}
        assert pairs == {(0, 1), (0, 3), (1, 4)}, pairs
    print("[OK] build_hierarchy_edge_index keeps only true-path edges with both endpoints in-vocab")


def check_build_hierarchy_edge_index_multi_parent_term_gets_both_edges():
    # GO:1 has two direct parents (GO:2 via is_a, GO:5 via part_of) -- both must survive,
    # not just the first one found (Review Focus: multi-parent terms).
    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "go_dependency_graph.json"
        _write_graph_fixture(graph_path)
        label2id = {"GO:1": 0, "GO:2": 1, "GO:5": 3}
        edge_index = build_hierarchy_edge_index(label2id, graph_path)
        pairs = {tuple(row) for row in edge_index.tolist()}
        assert pairs == {(0, 1), (0, 3)}, pairs
    print("[OK] a term with two direct parents gets an edge for each parent")


def check_build_hierarchy_edge_index_no_matching_edges_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "go_dependency_graph.json"
        _write_graph_fixture(graph_path)
        label2id = {"GO:3": 2}  # only GO:3, which has no surviving edges
        edge_index = build_hierarchy_edge_index(label2id, graph_path)
        assert edge_index.shape == (0, 2), edge_index.shape
    print("[OK] no matching edges -> empty (0, 2) tensor, not an error")


def check_hierarchy_consistency_penalty_matches_hand_computed_value():
    logits = torch.tensor([
        [2.0, 1.0, 0.0],  # child(0)=2.0 > parent(1)=1.0 -> violation margin 1.0
        [0.5, 1.0, 0.0],  # child(0)=0.5 < parent(1)=1.0 -> no violation
    ])
    edge_index = torch.tensor([[0, 1]], dtype=torch.long)
    penalty = hierarchy_consistency_penalty(logits, edge_index)
    # relu(2.0-1.0)=1.0, relu(0.5-1.0)=0 -> mean over 2 examples * 1 edge = 0.5
    assert abs(penalty.item() - 0.5) < 1e-6, penalty.item()
    print("[OK] hierarchy_consistency_penalty matches a hand-computed hinge value")


def check_hierarchy_consistency_penalty_mid_chain_term_applies_both_edges():
    # GO:1 (idx 0) is a child of GO:2 (idx 1), which is itself a child of GO:6 (idx 2):
    # chain edges (0,1) and (1,2). Both constraints must apply in one forward pass
    # (Review Focus: mid-chain terms).
    logits = torch.tensor([[3.0, 1.0, 5.0]])  # violates (0,1): 3.0>1.0; satisfies (1,2): 1.0<5.0
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    penalty = hierarchy_consistency_penalty(logits, edge_index)
    # relu(3.0-1.0)=2.0, relu(1.0-5.0)=0.0 -> mean over 1 example * 2 edges = 1.0
    assert abs(penalty.item() - 1.0) < 1e-6, penalty.item()
    print("[OK] a mid-chain term's two edges are both applied, independently")


def check_hierarchy_consistency_penalty_empty_edge_index_is_zero():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    edge_index = torch.zeros((0, 2), dtype=torch.long)
    penalty = hierarchy_consistency_penalty(logits, edge_index)
    assert penalty.item() == 0.0, penalty.item()
    print("[OK] empty edge_index -> zero penalty, no nan")


def check_hierarchy_consistency_penalty_fp16_logits_stay_finite():
    # Review Focus: --mixed-precision fp16/bf16 combined with --hierarchy-loss-weight.
    logits = torch.tensor([[60000.0, -60000.0, 0.0]], dtype=torch.float16)
    edge_index = torch.tensor([[0, 1]], dtype=torch.long)
    penalty = hierarchy_consistency_penalty(logits, edge_index)
    assert torch.isfinite(penalty), penalty
    print(f"[OK] fp16 logits at the dtype's extremes stay finite (penalty={penalty.item()})")


def check_compute_hierarchy_violations_matches_hand_computed_values():
    probs = np.array([
        [0.9, 0.3],  # child > parent -> violation, magnitude 0.6
        [0.2, 0.8],  # child < parent -> no violation
    ])
    edge_index = np.array([[0, 1]])
    result = compute_hierarchy_violations(probs, edge_index)
    assert result["violation_rate"] == 0.5, result
    assert abs(result["mean_violation_magnitude"] - 0.6) < 1e-9, result
    assert result["n_edges"] == 1, result
    print("[OK] compute_hierarchy_violations matches hand-computed rate/magnitude")


def check_compute_hierarchy_violations_zero_violations_no_div_by_zero():
    probs = np.array([[0.1, 0.9], [0.2, 0.8]])
    edge_index = np.array([[0, 1]])
    result = compute_hierarchy_violations(probs, edge_index)
    assert result["violation_rate"] == 0.0, result
    assert result["mean_violation_magnitude"] == 0.0, result
    print("[OK] zero violations -> rate/magnitude both 0.0, no division error")


def check_compute_hierarchy_violations_empty_edge_index_is_zero():
    probs = np.array([[0.1, 0.9], [0.2, 0.8]])
    edge_index = np.zeros((0, 2), dtype=int)
    result = compute_hierarchy_violations(probs, edge_index)
    assert result["violation_rate"] == 0.0, result
    assert result["mean_violation_magnitude"] == 0.0, result
    assert result["n_edges"] == 0, result
    print("[OK] empty edge_index -> zero rate/magnitude, not a crash")


def check_compute_hierarchy_violations_chunking_is_consistent():
    rng = np.random.default_rng(42)
    probs = rng.random((10, 4))
    edge_index = np.array([[0, 1], [2, 3]])
    full = compute_hierarchy_violations(probs, edge_index, chunk_size=10)
    chunked = compute_hierarchy_violations(probs, edge_index, chunk_size=3)
    assert abs(full["violation_rate"] - chunked["violation_rate"]) < 1e-12
    assert abs(full["mean_violation_magnitude"] - chunked["mean_violation_magnitude"]) < 1e-12
    print("[OK] chunk_size doesn't change the computed violation rate/magnitude")


if __name__ == "__main__":
    check_build_hierarchy_edge_index_restricts_correctly()
    check_build_hierarchy_edge_index_multi_parent_term_gets_both_edges()
    check_build_hierarchy_edge_index_no_matching_edges_returns_empty()
    check_hierarchy_consistency_penalty_matches_hand_computed_value()
    check_hierarchy_consistency_penalty_mid_chain_term_applies_both_edges()
    check_hierarchy_consistency_penalty_empty_edge_index_is_zero()
    check_hierarchy_consistency_penalty_fp16_logits_stay_finite()
    check_compute_hierarchy_violations_matches_hand_computed_values()
    check_compute_hierarchy_violations_zero_violations_no_div_by_zero()
    check_compute_hierarchy_violations_empty_edge_index_is_zero()
    check_compute_hierarchy_violations_chunking_is_consistent()
    print("[OK] all checks passed")
