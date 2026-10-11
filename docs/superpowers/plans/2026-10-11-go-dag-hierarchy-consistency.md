# GO-DAG True-Path-Rule Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a training-time hinge penalty to `training_run.py` that discourages the
model from scoring a GO term's child higher than its parent (GO's true-path rule), plus
the diagnostic additions needed to actually see this item's claimed effect (a
rarity-stratified AUROC breakdown and a direct violation-rate measurement), then validate
with one real GPU trial against the existing `trial-013` baseline.

**Architecture:** A static `(child_idx, parent_idx)` edge list, built once from
`generate_dataset_full.py`'s already-exported `go_dependency_graph.json` restricted to
the trainable vocabulary, drives a new loss term added to (not replacing) the existing
BCE loss. `run_diagnostic` gains two independently-gated reporting additions that run
whenever `--run-diagnostic` is set, regardless of whether the new loss term is enabled.

**Tech Stack:** `networkx` (already a project dependency, used by `generate_dataset.py`)
to parse/restrict the existing graph artifact. No new third-party dependency, no new
model class, no architecture change — pure loss-function and diagnostic additions.

**Spec:** `docs/superpowers/specs/2026-10-11-go-dag-hierarchy-consistency-design.md`

## Global Constraints

- `--hierarchy-loss-weight` defaults to `0.0`; at `0.0` the mechanism must be a complete
  no-op — no edge index built, no behavior change for any existing invocation.
- Penalty formula is `relu(child_logit - parent_logit)`, averaged over every
  (example, edge) pair, computed in **logit space** (not probability space).
- Only **direct** `is_a`/`part_of` edges where **both** endpoints are in the current
  trainable vocabulary (`label2id`) — never the full transitive ancestor closure (see
  spec's "Why direct edges, not full ancestor closure").
- The penalty is added to **training** loss only — validation loss (`val_loss`, what
  `--checkpoint-selection val_loss` uses) must stay pure BCE, matching
  `--label-smoothing`/`--unknown-target-value`'s existing precedent.
- Rarity bucket boundary is `--rare-term-threshold`, default `100`.
- `_diagnostic_metrics_from_arrays` and `run_diagnostic`'s new parameters default to
  `None` and must produce byte-for-byte-identical output to today's behavior when
  omitted — no existing caller or test may change behavior.
- `--go-dependency-graph-path`: raise immediately if `--hierarchy-loss-weight > 0` and
  the resolved file doesn't exist. If only `--run-diagnostic` needs it and it's missing,
  degrade gracefully — one `[WARN]` line, skip the violation-rate fields, do not crash.
- No `--push-to-hub` in the GPU trial hand-off command (per the known-annotation-
  conditioning incident and its fix already landed in this repo's history).
- The GPU trial hand-off command (Task 6) must pass `--max-length 4192` explicitly —
  `training_run.py`'s own default (`1024`) silently truncates ~9.15% of the train split
  relative to `trial-013`'s actual config, which already passed this flag.

## Review Focus

- A GO term with **more than one direct parent** (e.g. both an `is_a` and a `part_of`
  edge to different parents): `build_hierarchy_edge_index` must emit one edge per
  parent, not just the first found — a reasonable person reading "parent score >= child
  score" expects every parent to be constrained, not an arbitrary single one.
- A term that is a **parent in one edge and a child in another** (mid-chain term, e.g.
  grandparent→parent→child): `hierarchy_consistency_penalty` must apply both
  constraints independently in the same forward pass without one masking the other.
- `--hierarchy-loss-weight` combined with `--resume-from-revision` (continuing a
  checkpoint trained with a different `--hierarchy-loss-weight` setting, including
  `0.0`): since this mechanism adds no new parameters or modules, a reasonable person
  expects this combination to just work with no shape mismatch — never silently broken
  or requiring a special case.
- `--mixed-precision fp16`/`bf16` combined with `--hierarchy-loss-weight > 0`: the hinge
  computed on reduced-precision logits must stay finite (no silent NaN/Inf feeding into
  `GradScaler`) — a reasonable person enabling both expects training to behave the same
  as with the penalty off, not degrade numerically.
- An **empty edge index** (e.g. a vocabulary slice with no true-path edges at all) fed
  to either the loss term or `compute_hierarchy_violations`: must return a clean `0.0`,
  never `NaN` from an empty-tensor `.mean()` or a division by zero.

---

### Task 1: Hierarchy edge index, penalty, and violation-rate helpers

**Files:**
- Modify: `training_run.py` (add `import networkx as nx` near the top; add
  `build_hierarchy_edge_index` and `hierarchy_consistency_penalty` right after
  `compute_loss` ends, currently `training_run.py:710`, before `def nnpu_loss(` at
  `training_run.py:712`; add `compute_hierarchy_violations` right before
  `_diagnostic_metrics_from_arrays`, currently `training_run.py:863`)
- Create: `experiments/scripts/check_hierarchy_consistency.py`

**Interfaces:**
- Produces: `build_hierarchy_edge_index(label2id: Dict[str, int], graph_path: Path) ->
  torch.LongTensor` (shape `(E, 2)`, columns `(child_idx, parent_idx)`, `(0, 2)` when no
  edges match); `hierarchy_consistency_penalty(logits: torch.Tensor, edge_index:
  torch.Tensor) -> torch.Tensor` (scalar); `compute_hierarchy_violations(probs:
  np.ndarray, edge_index: np.ndarray, chunk_size: int = 256) -> Dict[str, float]` (keys
  `violation_rate`, `mean_violation_magnitude`, `n_edges`) — all consumed by Task 2
  (training loop) and Task 4 (`run_diagnostic`).

- [ ] **Step 1: Write the failing tests**

Create `experiments/scripts/check_hierarchy_consistency.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `env/bin/python experiments/scripts/check_hierarchy_consistency.py`
Expected: `ImportError: cannot import name 'build_hierarchy_edge_index' from 'training_run'`
(none of the three functions exist yet).

- [ ] **Step 3: Implement the helpers**

Add `import networkx as nx` to `training_run.py`'s import block (next to `import numpy
as np` at `training_run.py:20`).

Insert after `compute_loss` (ends at `training_run.py:710`), before `def nnpu_loss(`:

```python
def build_hierarchy_edge_index(label2id: Dict[str, int], graph_path: Path) -> torch.LongTensor:
    """Restrict generate_dataset_full.py's exported GO graph (go_dependency_graph.json)
    to direct is_a/part_of edges where both endpoints are in the current trainable
    vocabulary (label2id) -- the transitive ancestor closure is NOT used here (see the
    design spec's "Why direct edges, not full ancestor closure"): if every direct edge
    satisfies parent_score >= child_score, transitive consistency over any longer chain
    follows automatically, so penalizing the full closure would roughly square the edge
    count for no additional constraint power. Returns a (E, 2) LongTensor of
    (child_label_idx, parent_label_idx) pairs, or a (0, 2) tensor if nothing matches --
    cheap enough (well under a second, ~47k edges at the current 27,191-term vocabulary)
    to build once per run and keep resident on device for the whole training loop.
    """
    payload = json.loads(graph_path.read_text())
    graph = nx.readwrite.json_graph.node_link_graph(payload["graph"], edges="links")
    true_path_relationships = set(payload["true_path_relationships"])
    edges = [
        (label2id[u], label2id[v])
        for u, v, key in graph.edges(keys=True)
        if key in true_path_relationships and u in label2id and v in label2id
    ]
    if not edges:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long)


def hierarchy_consistency_penalty(logits: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean hinge violation of GO's true-path rule over edge_index's direct
    (child_idx, parent_idx) pairs: relu(child_logit - parent_logit), averaged over every
    edge and every example in the batch. Zero when every edge already satisfies
    parent_logit >= child_logit for every example; monotonic with sigmoid, so this
    constrains predicted probabilities (parent_prob >= child_prob) without needing to
    compute probabilities at all.
    """
    if edge_index.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    child_logits = logits[:, edge_index[:, 0]]
    parent_logits = logits[:, edge_index[:, 1]]
    return torch.relu(child_logits - parent_logits).mean()
```

Insert right before `def _diagnostic_metrics_from_arrays` (currently `training_run.py:863`):

```python
def compute_hierarchy_violations(probs: np.ndarray, edge_index: np.ndarray,
                                  chunk_size: int = 256) -> Dict[str, float]:
    """Direct measurement of what hierarchy_consistency_penalty targets, computed on
    predicted PROBABILITIES (one level removed from the logit-space loss itself) over
    every (validation example, edge) pair: a violation is child_prob > parent_prob,
    regardless of margin. Processed in example-chunks rather than materializing the
    full (n_examples, n_edges) array at once -- negligible at today's ~2000x47068 scale
    but keeps this safe if --diagnostic-max-val-samples is raised well beyond that.
    """
    if len(edge_index) == 0:
        return {"violation_rate": 0.0, "mean_violation_magnitude": 0.0, "n_edges": 0}
    child_idx, parent_idx = edge_index[:, 0], edge_index[:, 1]
    n_violations, violation_sum, total_pairs = 0, 0.0, 0
    for start in range(0, len(probs), chunk_size):
        chunk = probs[start:start + chunk_size]
        diff = chunk[:, child_idx] - chunk[:, parent_idx]
        violated = diff > 0
        n_violations += int(violated.sum())
        violation_sum += float(diff[violated].sum())
        total_pairs += diff.size
    return {
        "violation_rate": n_violations / total_pairs if total_pairs else 0.0,
        "mean_violation_magnitude": violation_sum / n_violations if n_violations else 0.0,
        "n_edges": int(len(edge_index)),
    }
```

- [ ] **Step 4: Run it to verify it passes**

Run: `env/bin/python experiments/scripts/check_hierarchy_consistency.py`
Expected: all 11 `[OK]` lines print, ending with `[OK] all checks passed`.

- [ ] **Step 5: Commit**

```bash
git add training_run.py experiments/scripts/check_hierarchy_consistency.py
git commit -m "feat: add GO-DAG hierarchy edge index, penalty, and violation helpers"
```

---

### Task 2: Wire `--hierarchy-loss-weight` into the training loop

**Files:**
- Modify: `training_run.py` (new CLI flags in `main()`'s argparse setup, currently
  ending around `training_run.py:1147`; argument validation right after
  `args = parser.parse_args()`, currently `training_run.py:1149-1157`; edge-index
  construction after `label2id` is built, currently `training_run.py:1174`; training-loop
  wiring inside the `with torch.autocast(...)` block, currently
  `training_run.py:1464-1513`)

**Interfaces:**
- Consumes: `build_hierarchy_edge_index`, `hierarchy_consistency_penalty` (Task 1).
- Produces: `args.hierarchy_loss_weight`, `args.go_dependency_graph_path`,
  `args.rare_term_threshold` (new CLI flags); a `hierarchy_edge_index` local variable in
  `main()` (CPU `torch.LongTensor` or `None`) that Task 4 reuses for the diagnostic.

- [ ] **Step 1: Write the failing test (CLI flags + no-op default + fail-fast validation)**

Create `experiments/scripts/check_hierarchy_loss_cli.py`:

```python
"""Standalone check for --hierarchy-loss-weight's CLI wiring (Task 2 of
docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md): the default must be
a complete no-op, and a missing graph file must fail fast when the flag is actually
requested. Runs training_run.py as a subprocess (no GPU, tiny CPU smoke configs) rather
than importing main() directly, since main() has no return value to assert against --
this checks real process exit codes and stdout, as a user running the CLI would see.
Run directly:
    env/bin/python experiments/scripts/check_hierarchy_loss_cli.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = "data/cafa-6-protein-function-prediction/dataset_full"


def _run(extra_args):
    return subprocess.run(
        [sys.executable, "training_run.py", "--data-path", DATA_PATH,
         "--max-train-samples", "20", "--max-val-samples", "10",
         "--epochs", "1", "--device", "cpu", *extra_args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )


def check_missing_graph_file_fails_fast_when_weight_set():
    result = _run(["--hierarchy-loss-weight", "0.1",
                    "--go-dependency-graph-path", "/nonexistent/go_dependency_graph.json"])
    assert result.returncode != 0, result.stdout + result.stderr
    assert "go-dependency-graph-path" in (result.stdout + result.stderr), result.stderr
    print("[OK] --hierarchy-loss-weight > 0 with a missing graph file fails fast, doesn't train silently")


def check_default_weight_zero_builds_no_edge_index():
    result = _run([])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Built hierarchy edge index" not in result.stdout, result.stdout
    print("[OK] default --hierarchy-loss-weight=0.0 never builds the edge index (no-op)")


if __name__ == "__main__":
    check_missing_graph_file_fails_fast_when_weight_set()
    check_default_weight_zero_builds_no_edge_index()
    print("[OK] all checks passed")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `env/bin/python experiments/scripts/check_hierarchy_loss_cli.py`
Expected: `error: unrecognized arguments: --hierarchy-loss-weight 0.1 --go-dependency-graph-path ...`
(the flags don't exist yet) on the first check.

- [ ] **Step 3: Add the CLI flags**

In `main()`'s argparse block, immediately after the `--known-embed-dim` argument
(currently `training_run.py:1103-1105`), insert:

```python
    parser.add_argument("--hierarchy-loss-weight", type=float, default=0.0,
                         help="Weight on a training-time hinge penalty (relu(child_logit - "
                              "parent_logit), see hierarchy_consistency_penalty) that discourages "
                              "scoring a GO term's child higher than its parent -- GO's true-path "
                              "rule, a logical fact independent of annotation completeness (see "
                              "docs/superpowers/specs/2026-10-11-go-dag-hierarchy-consistency-"
                              "design.md). 0.0 (default) disables the mechanism entirely -- no edge "
                              "index is built, no change to any existing invocation.")
    parser.add_argument("--go-dependency-graph-path", default=None,
                         help="Path to generate_dataset_full.py's exported GO graph (see "
                              "export_dependency_graph in generate_dataset.py). Defaults to "
                              "<--data-path>/go_dependency_graph.json. Required (and must exist) "
                              "when --hierarchy-loss-weight > 0; optional for --run-diagnostic's "
                              "violation-rate reporting, which degrades gracefully (a [WARN] and "
                              "skipped fields) if the file isn't found.")
```

Immediately after the `--diagnostic-max-val-samples` argument (currently
`training_run.py:1121`), insert:

```python
    parser.add_argument("--rare-term-threshold", type=int, default=100,
                         help="Train-split annotation-count boundary between 'rare' and 'common' "
                              "for --run-diagnostic's rarity-stratified AUROC breakdown (a term "
                              "with fewer than this many positive examples in the train split is "
                              "'rare'). Matches this item's roadmap source's own rare-term "
                              "definition (docs/superpowers/specs/2026-10-11-go-dag-hierarchy-"
                              "consistency-design.md).")
```

- [ ] **Step 4: Add the fail-fast validation and edge-index construction**

Immediately after the existing `--use-known-input` validation block (currently
`training_run.py:1149-1157`, ending with the `--known-reveal-max` check), insert:

```python
    go_dependency_graph_path = (Path(args.go_dependency_graph_path) if args.go_dependency_graph_path
                                 else Path(args.data_path) / "go_dependency_graph.json")
    if args.hierarchy_loss_weight > 0 and not go_dependency_graph_path.exists():
        raise ValueError(
            f"--hierarchy-loss-weight={args.hierarchy_loss_weight} requires the GO dependency "
            f"graph at {go_dependency_graph_path} (see --go-dependency-graph-path) -- not found.")
```

Immediately after `label2id = {value: key for key, value in id2label.items()}` (currently
`training_run.py:1174`), insert:

```python
    hierarchy_edge_index = None
    if args.hierarchy_loss_weight > 0 or args.run_diagnostic:
        if go_dependency_graph_path.exists():
            hierarchy_edge_index = build_hierarchy_edge_index(label2id, go_dependency_graph_path)
            print(f"[INFO] Built hierarchy edge index: {hierarchy_edge_index.shape[0]} true-path "
                  f"edges within the {len(label2id)}-term trainable vocabulary")
        elif args.hierarchy_loss_weight == 0:
            print(f"[WARN] --run-diagnostic requested violation-rate reporting but "
                  f"{go_dependency_graph_path} was not found -- skipping hierarchy-violation "
                  f"diagnostic fields (rarity-stratified AUROC is unaffected).")
```

(The `args.hierarchy_loss_weight > 0` + missing-file case is unreachable here — it
already raised above.)

- [ ] **Step 5: Wire the penalty into the training loop**

Immediately after `criterion = torch.nn.BCEWithLogitsLoss(...)` (currently
`training_run.py:1413`), insert:

```python
    hierarchy_edge_index_device = hierarchy_edge_index.to(device) if hierarchy_edge_index is not None else None
```

Inside the training loop's `with torch.autocast(...)` block, immediately after the
existing `if args.nnpu: ... else: loss = compute_loss(criterion, logits, labels,
loss_mask)` (the `else` branch currently ends at `training_run.py:1513`), insert, at the
same 16-space indentation as the `if args.nnpu:` line:

```python
                if args.hierarchy_loss_weight > 0:
                    loss = loss + args.hierarchy_loss_weight * hierarchy_consistency_penalty(
                        logits, hierarchy_edge_index_device)
```

- [ ] **Step 6: Run it to verify it passes**

Run: `env/bin/python experiments/scripts/check_hierarchy_loss_cli.py`
Expected: both `[OK]` lines print, ending with `[OK] all checks passed`.

- [ ] **Step 7: Run the end-to-end CPU smoke test**

Run:

```bash
env/bin/python training_run.py \
  --data-path data/cafa-6-protein-function-prediction/dataset_full \
  --hierarchy-loss-weight 0.1 \
  --max-train-samples 100 --max-val-samples 50 --epochs 1 --device cpu
```

Expected: prints `[INFO] Built hierarchy edge index: 47068 true-path edges within the
27191-term trainable vocabulary` (edge/vocab counts may differ slightly if
`dataset_full` has been regenerated since this plan was written), trains the single
epoch to completion with a decreasing or stable `train_loss`, no crash.

- [ ] **Step 8: Review Focus — `--resume-from-revision` compatibility test**

Create `experiments/scripts/check_hierarchy_loss_no_new_params.py`:

```python
"""Standalone check that --hierarchy-loss-weight adds no new model parameters or
modules (Task 2 of docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md) --
unlike --use-known-input/--use-aspect-input, this mechanism is a pure loss-function
addition, so a checkpoint trained with it on or off must have an identical state_dict
shape either way, and --resume-from-revision must work regardless of which setting the
checkpoint being resumed was trained with (Review Focus item). Run directly:
    env/bin/python experiments/scripts/check_hierarchy_loss_no_new_params.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training_run

LABEL_NAMES = [f"GO:{i:07d}" for i in range(10)]
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


def check_state_dict_keys_identical_regardless_of_hierarchy_flag():
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type="facebook/esm2_t6_8M_UR50D",
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    keys = set(model.state_dict().keys())
    # hierarchy_consistency_penalty is a pure function of (logits, edge_index) -- it is
    # never attached to the model as a module/parameter, so create_esm's own signature
    # and output are untouched by this mechanism; this test pins that invariant directly
    # rather than just asserting by reading the (unchanged) create_esm source.
    assert "hierarchy_edge_index" not in str(keys), keys
    assert len(keys) > 0
    print(f"[OK] create_esm's state_dict ({len(keys)} keys) has no hierarchy-related "
          f"parameters -- --resume-from-revision needs no special case for this flag")


if __name__ == "__main__":
    check_state_dict_keys_identical_regardless_of_hierarchy_flag()
    print("[OK] all checks passed")
```

Run: `env/bin/python experiments/scripts/check_hierarchy_loss_no_new_params.py`
Expected: `[OK] all checks passed`.

- [ ] **Step 9: Commit**

```bash
git add training_run.py experiments/scripts/check_hierarchy_loss_cli.py \
  experiments/scripts/check_hierarchy_loss_no_new_params.py
git commit -m "feat: wire --hierarchy-loss-weight into the training loop"
```

---

### Task 3: Rarity-stratified AUROC in `_diagnostic_metrics_from_arrays`

**Files:**
- Modify: `training_run.py:863-930` (`_diagnostic_metrics_from_arrays`)
- Create: `experiments/scripts/check_diagnostic_rarity_stratification.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks (pure function change).
- Produces: `_diagnostic_metrics_from_arrays(..., label_is_rare: Optional[np.ndarray] =
  None)` — when provided (shape `(num_labels,)`, boolean), the returned result dict
  gains `rare_mean_auroc`, `rare_median_auroc`, `rare_n_labels_evaluable`,
  `common_mean_auroc`, `common_median_auroc`, `common_n_labels_evaluable` (each `None`/
  `0` when that bucket has no evaluable labels). Consumed by Task 4.

- [ ] **Step 1: Write the failing test**

Create `experiments/scripts/check_diagnostic_rarity_stratification.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `env/bin/python experiments/scripts/check_diagnostic_rarity_stratification.py`
Expected: `TypeError: _diagnostic_metrics_from_arrays() got an unexpected keyword argument 'label_is_rare'`

- [ ] **Step 3: Implement**

Change the signature at `training_run.py:863-864` from:

```python
def _diagnostic_metrics_from_arrays(probs: np.ndarray, labels: np.ndarray,
                                     loss_mask: Optional[np.ndarray] = None):
```

to:

```python
def _diagnostic_metrics_from_arrays(probs: np.ndarray, labels: np.ndarray,
                                     loss_mask: Optional[np.ndarray] = None,
                                     label_is_rare: Optional[np.ndarray] = None):
```

Replace the per-label loop (`training_run.py:884-891`):

```python
    per_label_auroc = []
    for j in range(labels.shape[1]):
        col_mask = loss_mask[:, j]
        y_true = labels[col_mask, j]
        if y_true.size == 0 or y_true.min() == y_true.max():
            continue  # not evaluable: no unmasked examples, or single-class
        per_label_auroc.append(roc_auc_score(y_true, probs[col_mask, j]))
    per_label_auroc = np.array(per_label_auroc)
```

with:

```python
    per_label_auroc = []
    rare_auroc, common_auroc = [], []
    for j in range(labels.shape[1]):
        col_mask = loss_mask[:, j]
        y_true = labels[col_mask, j]
        if y_true.size == 0 or y_true.min() == y_true.max():
            continue  # not evaluable: no unmasked examples, or single-class
        auroc_j = roc_auc_score(y_true, probs[col_mask, j])
        per_label_auroc.append(auroc_j)
        if label_is_rare is not None:
            (rare_auroc if label_is_rare[j] else common_auroc).append(auroc_j)
    per_label_auroc = np.array(per_label_auroc)
```

Immediately before `return result, per_label_auroc` (currently `training_run.py:930`),
insert:

```python
    if label_is_rare is not None:
        rare_auroc = np.array(rare_auroc)
        common_auroc = np.array(common_auroc)
        result["rare_mean_auroc"] = float(rare_auroc.mean()) if rare_auroc.size else None
        result["rare_median_auroc"] = float(np.median(rare_auroc)) if rare_auroc.size else None
        result["rare_n_labels_evaluable"] = int(rare_auroc.size)
        result["common_mean_auroc"] = float(common_auroc.mean()) if common_auroc.size else None
        result["common_median_auroc"] = float(np.median(common_auroc)) if common_auroc.size else None
        result["common_n_labels_evaluable"] = int(common_auroc.size)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `env/bin/python experiments/scripts/check_diagnostic_rarity_stratification.py`
Expected: all 3 `[OK]` lines print, ending with `[OK] all checks passed`.

- [ ] **Step 5: Run the existing regression suite**

Run: `env/bin/python experiments/scripts/check_run_diagnostic_known_mask.py`
Expected: `[OK] all checks passed` (unchanged — confirms the `label_is_rare=None` default
preserves every existing caller's behavior).

- [ ] **Step 6: Commit**

```bash
git add training_run.py experiments/scripts/check_diagnostic_rarity_stratification.py
git commit -m "feat: add rarity-stratified AUROC breakdown to the diagnostic"
```

---

### Task 4: Violation-rate reporting in `run_diagnostic`, wired into `main()`

**Files:**
- Modify: `training_run.py:933-1028` (`run_diagnostic`); `training_run.py` `main()`'s
  `run_diagnostic(...)` call site, currently `training_run.py:1745-1759`

**Interfaces:**
- Consumes: `compute_hierarchy_violations` (Task 1), `_diagnostic_metrics_from_arrays`'s
  `label_is_rare` parameter (Task 3), the `hierarchy_edge_index`/`label_counts` local
  variables already present in `main()` (Task 2 and the pre-existing `--pos-weight`
  computation at `training_run.py:1242-1246`, respectively).
- Produces: `run_diagnostic(..., hierarchy_edge_index: Optional[np.ndarray] = None,
  label_is_rare: Optional[np.ndarray] = None)`.

- [ ] **Step 1: Write the failing test**

Create `experiments/scripts/check_run_diagnostic_hierarchy_fields.py`:

```python
"""Standalone check that run_diagnostic's new hierarchy_edge_index/label_is_rare
parameters are accepted and threaded through correctly (Task 4 of
docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md). Uses a tiny real ESM
model (facebook/esm2_t6_8M_UR50D) on CPU with a fabricated in-memory dataset -- small
enough to run quickly without GPU, exercising the real forward pass rather than mocking
it. Run directly:
    env/bin/python experiments/scripts/check_run_diagnostic_hierarchy_fields.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import datasets
import numpy as np
import torch

import training_run

LABEL_NAMES = [f"GO:{i:07d}" for i in range(6)]
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


def _fabricate_val_dataset():
    return datasets.Dataset.from_dict({
        "sequence": ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "MSEQKLISEEDLNS", "MKVLAT"],
        "labels": [[0, 1, 2], [1, 3], [2, 4, 5]],
    })


def check_run_diagnostic_accepts_and_reports_hierarchy_fields():
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type="facebook/esm2_t6_8M_UR50D",
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    tokenizer = training_run.create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    val_ds = _fabricate_val_dataset().with_transform(
        transform=lambda ex: training_run.preprocess_batch(
            ex, tokenizer=tokenizer, unique_items=LABEL_NAMES, label2id=LABEL2ID))
    device = torch.device("cpu")
    edge_index = np.array([[0, 1], [2, 3]])  # fabricated, doesn't need to be "real" GO edges
    label_is_rare = np.array([True, True, False, False, True, False])

    result = training_run.run_diagnostic(
        model, tokenizer, val_ds, LABEL_NAMES, device,
        max_val_samples=10, batch_size=2,
        hierarchy_edge_index=edge_index, label_is_rare=label_is_rare,
    )
    assert "violation_rate" in result, result
    assert "mean_violation_magnitude" in result, result
    assert result["n_edges"] == 2, result
    assert "rare_n_labels_evaluable" in result, result
    print(f"[OK] run_diagnostic reports hierarchy fields "
          f"(violation_rate={result['violation_rate']:.4f}, "
          f"rare_n={result['rare_n_labels_evaluable']})")


def check_run_diagnostic_without_hierarchy_args_is_unchanged():
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type="facebook/esm2_t6_8M_UR50D",
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    tokenizer = training_run.create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    val_ds = _fabricate_val_dataset().with_transform(
        transform=lambda ex: training_run.preprocess_batch(
            ex, tokenizer=tokenizer, unique_items=LABEL_NAMES, label2id=LABEL2ID))
    device = torch.device("cpu")
    result = training_run.run_diagnostic(
        model, tokenizer, val_ds, LABEL_NAMES, device, max_val_samples=10, batch_size=2,
    )
    assert "violation_rate" not in result, result
    assert "rare_mean_auroc" not in result, result
    print("[OK] omitting the new kwargs leaves run_diagnostic's output unchanged")


if __name__ == "__main__":
    check_run_diagnostic_accepts_and_reports_hierarchy_fields()
    check_run_diagnostic_without_hierarchy_args_is_unchanged()
    print("[OK] all checks passed")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `env/bin/python experiments/scripts/check_run_diagnostic_hierarchy_fields.py`
Expected: `TypeError: run_diagnostic() got an unexpected keyword argument 'hierarchy_edge_index'`

- [ ] **Step 3: Implement — `run_diagnostic`'s signature and body**

Change the signature (`training_run.py:933-945`) by adding two parameters after
`use_known_input: bool = False,`:

```python
    hierarchy_edge_index: Optional[np.ndarray] = None,
    label_is_rare: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
```

Change the `_diagnostic_metrics_from_arrays` call (currently `training_run.py:1001`)
from:

```python
    result, per_label_auroc = _diagnostic_metrics_from_arrays(probs, labels, loss_mask)
```

to:

```python
    result, per_label_auroc = _diagnostic_metrics_from_arrays(probs, labels, loss_mask, label_is_rare)
    if hierarchy_edge_index is not None:
        result.update(compute_hierarchy_violations(probs, hierarchy_edge_index))
```

Immediately before `print(f"[DIAGNOSTIC] {json.dumps(result, default=str)}")` (currently
`training_run.py:1026`), insert:

```python
    if "rare_mean_auroc" in result:
        rare_str = f"{result['rare_mean_auroc']:.4f}" if result["rare_mean_auroc"] is not None else "n/a"
        common_str = f"{result['common_mean_auroc']:.4f}" if result["common_mean_auroc"] is not None else "n/a"
        print(f"Rarity-stratified AUROC: rare mean={rare_str} (n={result['rare_n_labels_evaluable']})  "
              f"common mean={common_str} (n={result['common_n_labels_evaluable']})")
    if "violation_rate" in result:
        print(f"Hierarchy violations: rate={result['violation_rate']:.4f} "
              f"mean_magnitude={result['mean_violation_magnitude']:.4f} (n_edges={result['n_edges']})")
```

- [ ] **Step 4: Implement — `main()`'s call site**

Replace the `run_diagnostic(...)` call (currently `training_run.py:1747-1759`):

```python
        run_diagnostic(
            best_model,
            tokenizer,
            full_validation_dataset,
            label_names,
            device,
            autocast_dtype=autocast_dtype,
            use_amp=use_amp,
            max_val_samples=args.diagnostic_max_val_samples,
            batch_size=args.batch_size,
            use_aspect_input=args.use_aspect_input,
            use_known_input=args.use_known_input,
        )
```

with:

```python
        label_is_rare = (label_counts.numpy() < args.rare_term_threshold)
        run_diagnostic(
            best_model,
            tokenizer,
            full_validation_dataset,
            label_names,
            device,
            autocast_dtype=autocast_dtype,
            use_amp=use_amp,
            max_val_samples=args.diagnostic_max_val_samples,
            batch_size=args.batch_size,
            use_aspect_input=args.use_aspect_input,
            use_known_input=args.use_known_input,
            hierarchy_edge_index=hierarchy_edge_index.numpy() if hierarchy_edge_index is not None else None,
            label_is_rare=label_is_rare,
        )
```

(`label_counts` and `hierarchy_edge_index` are both already in scope here — `label_counts`
from the pre-existing `--pos-weight` computation at `training_run.py:1242-1246`,
`hierarchy_edge_index` from Task 2.)

- [ ] **Step 5: Run it to verify it passes**

Run: `env/bin/python experiments/scripts/check_run_diagnostic_hierarchy_fields.py`
Expected: both `[OK]` lines print, ending with `[OK] all checks passed`.

- [ ] **Step 6: Run the existing regression suite**

Run: `env/bin/python experiments/scripts/check_run_diagnostic_known_mask.py && env/bin/python experiments/scripts/check_known_input_masking.py`
Expected: both print `[OK] all checks passed` (confirms Task 4's changes didn't disturb
the known-annotation-conditioning item's own tests, which call these same functions
without the new kwargs).

- [ ] **Step 7: Run the end-to-end CPU smoke test**

Run:

```bash
env/bin/python training_run.py \
  --data-path data/cafa-6-protein-function-prediction/dataset_full \
  --hierarchy-loss-weight 0.1 \
  --max-train-samples 100 --max-val-samples 50 --epochs 1 --device cpu \
  --run-diagnostic --diagnostic-max-val-samples 50
```

Expected: training completes as in Task 2's smoke test, and the diagnostic output
additionally includes a `Rarity-stratified AUROC: ...` line and a `Hierarchy violations:
rate=... mean_magnitude=... (n_edges=47068)` line (edge count may differ slightly if
`dataset_full` has been regenerated), no crash.

- [ ] **Step 8: Commit**

```bash
git add training_run.py experiments/scripts/check_run_diagnostic_hierarchy_fields.py
git commit -m "feat: report hierarchy-violation rate and rarity AUROC from run_diagnostic"
```

---

### Task 5: Backfill script for the existing `trial-013` baseline

**Files:**
- Create: `experiments/scripts/rerun_diagnostic_full.py`

**Interfaces:**
- Consumes: `training_run.load_training_data`, `training_run.build_hierarchy_edge_index`
  (Task 1), `training_run.create_esm_tokenizer`, `training_run.create_esm`,
  `training_run.preprocess_batch`, `training_run.run_diagnostic` (Task 4) — all already
  public module-level functions, no new interface to design.
- Produces: no new interface (one-shot script); its output is the control numbers Task 6
  compares the new trial against.

`trial-013` (`kelmoir/cafa6-esm@main`) was trained and diagnosed before this item
existed, so its `trial_metrics.json` has neither rarity-stratified AUROC nor a violation
rate — there is no real control number for a hierarchy-loss trial's new metrics without
re-running the (now-extended) diagnostic against it. This script does that, read-only,
no retraining, no re-push.

This script is a thin orchestration wrapper around already-tested functions (Tasks 1-4),
modeled directly on the existing `experiments/ancestor_sweep/scripts/
diagnostic_eval_stage2.py` pattern — there is nothing here to unit-test in isolation; its
own completion contract is a real invocation against a real checkpoint in Step 2.

- [ ] **Step 1: Write the script**

Create `experiments/scripts/rerun_diagnostic_full.py`:

```python
"""Post-hoc diagnostic backfill for a checkpoint trained BEFORE run_diagnostic gained the
rarity-stratified AUROC / hierarchy-violation-rate fields (Task 5 of
docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md) -- most importantly
kelmoir/cafa6-esm@main (trial-013), the pure-BCE baseline a hierarchy-loss trial needs a
real control number against (see docs/superpowers/specs/2026-10-11-go-dag-hierarchy-
consistency-design.md's "Backfill script" section). Modeled directly on
experiments/ancestor_sweep/scripts/diagnostic_eval_stage2.py's re-download-and-reload
pattern, but targeting dataset_full (not that script's stale legacy dataset path) and
passing the new hierarchy_edge_index/label_is_rare arguments unconditionally. No
retraining, no re-push to the Hub -- read-only.

Run directly:
    env/bin/python experiments/scripts/rerun_diagnostic_full.py <revision> [max_val_samples] [rare_term_threshold]
e.g.:
    env/bin/python experiments/scripts/rerun_diagnostic_full.py main 2000 100
"""
import json
import sys
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_run

CHECKPOINT_REVISION = sys.argv[1] if len(sys.argv) > 1 else "main"
HF_REPO_ID = "kelmoir/cafa6-esm"
MAX_VAL_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
RARE_TERM_THRESHOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 100
DATA_PATH = Path("data/cafa-6-protein-function-prediction/dataset_full")

trial_metrics_path = hf_hub_download(repo_id=HF_REPO_ID, filename="trial_metrics.json", revision=CHECKPOINT_REVISION)
trial_config = json.loads(Path(trial_metrics_path).read_text())["config"]
MODEL_NAME = trial_config["model_name"]
USE_LORA = trial_config.get("use_lora", False)
USE_ASPECT_INPUT = trial_config.get("use_aspect_input", False)
USE_KNOWN_INPUT = trial_config.get("use_known_input", False)
KNOWN_EMBED_DIM = trial_config.get("known_embed_dim", 64)
MIXED_PRECISION = trial_config.get("mixed_precision", "no")
AUTOCAST_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(MIXED_PRECISION)

dataset = training_run.load_training_data(data_path=DATA_PATH)
label_names = [str(name) for name in dataset["train"].features["labels"].feature.names]
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in id2label.items()}

label_counts = torch.zeros(len(label_names))
for labs in dataset["train"]["labels"]:
    for idx in labs:
        label_counts[int(idx)] += 1
label_is_rare = (label_counts.numpy() < RARE_TERM_THRESHOLD)

graph_path = DATA_PATH / "go_dependency_graph.json"
hierarchy_edge_index = None
if graph_path.exists():
    hierarchy_edge_index = training_run.build_hierarchy_edge_index(label2id, graph_path).numpy()
else:
    print(f"[WARN] {graph_path} not found -- skipping hierarchy-violation diagnostic fields")

tokenizer = training_run.create_esm_tokenizer(MODEL_NAME)
preprocess = partial(
    training_run.preprocess_batch,
    tokenizer=tokenizer,
    max_length=training_run.MAX_LENGTH,
    unique_items=label_names,
    label2id=label2id,
    use_known_input=USE_KNOWN_INPUT,
    known_terms_from_dataset=False,
)
val_ds = dataset["validation"].with_transform(transform=preprocess)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = AUTOCAST_DTYPE is not None and device.type == "cuda"

if USE_LORA:
    model = training_run.create_esm(
        model_type=MODEL_NAME, use_lora=True, lora_resume_repo_id=HF_REPO_ID,
        revision=CHECKPOINT_REVISION, id2label=id2label, label2id=label2id,
        lora_base_dtype=AUTOCAST_DTYPE,
    ).to(device)
else:
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type=HF_REPO_ID, revision=CHECKPOINT_REVISION,
        id2label=id2label, label2id=label2id,
        use_aspect_input=USE_ASPECT_INPUT, use_known_input=USE_KNOWN_INPUT,
        known_embed_dim=KNOWN_EMBED_DIM,
    ).to(device)
model.eval()
print(f"[INFO] Loaded {HF_REPO_ID}@{CHECKPOINT_REVISION} ({MODEL_NAME}, use_lora={USE_LORA}, "
      f"use_known_input={USE_KNOWN_INPUT}) for eval, device={device}")

training_run.run_diagnostic(
    model,
    tokenizer,
    val_ds,
    label_names,
    device,
    autocast_dtype=AUTOCAST_DTYPE,
    use_amp=use_amp,
    max_val_samples=MAX_VAL_SAMPLES,
    batch_size=16,
    use_aspect_input=USE_ASPECT_INPUT,
    use_known_input=USE_KNOWN_INPUT,
    hierarchy_edge_index=hierarchy_edge_index,
    label_is_rare=label_is_rare,
)
```

- [ ] **Step 2: Run it for real against `trial-013`**

Run: `env/bin/python experiments/scripts/rerun_diagnostic_full.py main 2000 100`
Expected: loads `kelmoir/cafa6-esm@main` (no GPU required — `esm2_t12_35M_UR50D` is small
enough for a one-off CPU diagnostic pass), prints the same `DIAGNOSTIC`/`Rarity-
stratified AUROC`/`Hierarchy violations` output shape as Task 4's smoke test, with real
numbers. Record the printed `mean_auroc`, `pooled_auroc`, `best_f1`, `rare_mean_auroc`,
`common_mean_auroc`, and `violation_rate`/`mean_violation_magnitude` values — Task 6
needs them as the control.

- [ ] **Step 3: Commit**

```bash
git add experiments/scripts/rerun_diagnostic_full.py
git commit -m "feat: add dataset_full-aware diagnostic backfill script for existing checkpoints"
```

---

### Task 6: Real GPU validation trial and roadmap logging

**Files:**
- Modify: `experiments/cafa6_next_steps_plan.md` (log the result under Tier 1 item 3)

**Interfaces:** none new — this task only runs the already-built mechanism (Tasks 1-4)
and logs the outcome; no code changes beyond the roadmap doc.

- [ ] **Step 1: Hand the user the GPU trial command**

This step needs real GPU throughput (per `hardware_local_gpu.md`: run GPU-assisted work
locally via a `!`-command, not in-sandbox). Hand the user this command to run via `!`:

```bash
env/bin/python training_run.py \
  --data-path data/cafa-6-protein-function-prediction/dataset_full \
  --model-name facebook/esm2_t12_35M_UR50D --unfreeze-last-n 4 --max-length 4192 \
  --esm-dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --batch-size 2 --grad-accum-steps 8 --epochs 30 --checkpoint-selection f1 \
  --hierarchy-loss-weight 0.1 \
  --run-diagnostic --run-name hierarchy-loss-trial-001 \
  --hf-repo-id kelmoir/cafa6-esm --checkpoint-mode final
```

**`--max-length 4192` is required, not optional, for this comparison to be valid**:
`training_run.py --max-length` defaults to `1024` (truncates sequences at tokenization
time, independent of `generate_dataset_full.py`'s own `MAX_LENGTH=4192` dataset-inclusion
filter), but `trial-013` explicitly passed `--max-length 4192`
(`experiments/full_label_sweep/sweep_log.md`'s trial index table). ~9.15% of the train
split (4,813/52,579 proteins) is longer than 1024 residues and would be silently
truncated without this flag — a real confound this plan's author caught (this session)
only while writing this task, after discovering the already-shelved known-annotation-
conditioning item's own GPU trials (`known-input-trial-001`/`-002`) omitted this flag
entirely (confirmed via their logged `[CONFIG]` output: `"max_length": 1024`). That is a
separate, already-closed item and not reopened by this plan, but is worth flagging to
the user as a caveat on that item's conclusion, independent of this task.

This is `trial-013`'s exact hyperparameter config (`experiments/full_label_sweep/
sweep_log.md`) plus `--hierarchy-loss-weight 0.1`, so `trial-013` doubles as the
pooled/mean/median-AUROC control; Task 5's backfill script supplies the matching
rarity/violation-rate control numbers `trial-013` never originally computed. No
`--push-to-hub` — this is a first exploratory trial of a new mechanism, not a config
already confirmed HEALTHY at target scale (same reasoning the known-annotation-
conditioning plan's incident already established).

Expected: a completed trial pushed to `kelmoir/cafa6-esm@hierarchy-loss-trial-001`, with
a `--run-diagnostic` summary (including the new rarity/violation fields) printed and
saved in that branch's `trial_metrics.json`. Confirm with the user that this ran to
completion and share the diagnostic numbers before proceeding.

- [ ] **Step 2: Log the comparison and commit**

Log the result in `experiments/cafa6_next_steps_plan.md` under Tier 1 item 3: the new
trial's pooled/mean/median AUROC against `trial-013`'s own (already logged) numbers, and
the new rarity-stratified AUROC / violation-rate numbers against Task 5's backfilled
`trial-013` control — stating plainly whether the rare-term AUROC gap narrowed and
whether the violation rate dropped, without rounding a flat or negative result into a
positive one (same discipline the known-annotation-conditioning item's own negative
result was logged with).

```bash
git add experiments/cafa6_next_steps_plan.md
git commit -m "docs: log hierarchy-loss-trial-001 vs trial-013 control"
```
