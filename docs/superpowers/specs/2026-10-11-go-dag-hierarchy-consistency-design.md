# GO-DAG True-Path-Rule Consistency — Training-Time Loss Term (design)

**Goal:** Per `experiments/cafa6_next_steps_plan.md` Tier 1 item 3 — add a lightweight,
training-time penalty that discourages the model from scoring a child GO term higher
than its parent, independent of annotation completeness (GO's true-path rule is a
logical/definitional fact, not a statistical one — unlike the OWA/PU-learning problem,
this isn't gated on label-completeness work). The expected benefit, per the roadmap's
source report, concentrates on rare/near-zero-shot terms (<100 annotated proteins), not
the bulk of the label space — so the diagnostic used to judge this item must be able to
see a rare-term-only effect, not just an aggregate that could dilute it to invisible
([[feedback_check_metric_can_see_the_effect]]).

**Architecture:** Extend `training_run.py` in place — no new scripts for the mechanism
itself (one small standalone script is added for backfilling a comparison number against
the existing baseline checkpoint — see "Backfill script" below). A new helper builds a
static `(child_label_idx, parent_label_idx)` edge list, restricted to `is_a`/`part_of`
edges where both endpoints are in the current trainable vocabulary, from the GO graph
`generate_dataset_full.py` already exports
(`data/cafa-6-protein-function-prediction/dataset_full/go_dependency_graph.json`). A new
loss term, added to (not replacing) the existing BCE loss, penalizes `relu(child_logit -
parent_logit)` summed over that edge list. `run_diagnostic` gains two new, independently
useful reporting capabilities: a rarity-stratified AUROC breakdown, and a direct
hierarchy-violation-rate measurement — both computed whenever `--run-diagnostic` runs,
regardless of whether the new loss term is enabled, so a plain BCE control run (including
the existing `trial-013` checkpoint, re-diagnosed) produces numbers directly comparable to
a trial that trains with the penalty on.

**Tech Stack:** Reuses the existing `go_dependency_graph.json` artifact and `networkx`
(already a `generate_dataset_full.py` dependency) to restrict/parse it — no new data
artifact, no new third-party dependency. Pure-tensor operations (`index_select`/fancy
indexing + `relu`) for the loss term itself — no new model class, no architecture change.

**Spec context:** `experiments/cafa6_next_steps_plan.md` Tier 1 item 3. Decided in this
session's brainstorming:
- Penalty formulation: hinge directly on logits — `relu(child_logit - parent_logit)`,
  not a probability-space hinge or a squared hinge (cheapest, most numerically stable,
  monotonic with sigmoid so `parent_prob >= child_prob` follows from `parent_logit >=
  child_logit`).
- Combination with BCE: `total_loss = bce_loss + hierarchy_loss_weight *
  hierarchy_penalty`, a single new flag (`--hierarchy-loss-weight`, default `0.0` — opt
  in, matches the project's existing per-mechanism-flag style), first trial picks a
  value by inspection (`0.1`), no sweep.
- `run_diagnostic` must be able to see a rare-term-only effect: add a rarity-stratified
  (`<100` vs `>=100` train-split annotations) AUROC breakdown.
- `run_diagnostic` must also directly measure what the loss term targets (violation rate
  and magnitude), independent of whether AUROC moves — added as its own reporting
  capability, not folded into the AUROC numbers.

## Background

The dataset's label vocabulary is already a true-path-rule-consistent *target* — every
protein's positive labels are the full ancestor-propagated closure
(`generate_dataset_full.py`'s `propagate_protein_terms`), so a parent term is always
positive whenever any of its descendants are. But nothing today constrains the model's
*predicted* scores to respect that same structure: `training_run.py`'s classifier head
is a flat `Linear(hidden_size, num_labels)` (or its LoRA/aspect/known-input variants),
and `compute_loss`'s BCE term scores every label independently — a model is free to
predict a child term confidently while scoring its parent near zero, a logical
impossibility under GO's true-path rule that the current training signal does nothing to
discourage.

`generate_dataset_full.py` already exports the full GO graph once, at dataset-build
time, specifically so a later item could look up ancestor/descendant relationships
without recomputing them (`export_dependency_graph`'s own docstring: "A later curriculum
stage can load this... without recomputing the graph from scratch"). This item is that
later stage, for the parent-child (direct-edge) case specifically — see "Why direct
edges, not full ancestor closure" below for why direct edges are the right level, not the
full transitive closure.

A quick feasibility check (this session) confirmed the artifact is usable as-is: loading
`go_dependency_graph.json` and restricting to `is_a`/`part_of` edges where both endpoints
are in the current 27,191-term trainable vocabulary (`label2id`) takes well under a
second (~0.6s graph load + ~0.05s restriction) and yields 47,068 edges — small enough to
keep resident on `device` for the whole training run, no per-batch recomputation.

## Scope

**In scope:**
- A new helper, `build_hierarchy_edge_index`, producing the restricted
  `(child_idx, parent_idx)` edge tensor from `go_dependency_graph.json` + `label2id`.
- A new loss term, `hierarchy_consistency_penalty`, and its wiring into `main()`'s
  training loop (added to the existing BCE loss, training only — not validation loss,
  see "Interaction with validation loss" below).
- A new `--hierarchy-loss-weight` CLI flag (default `0.0`, opt-in) and a new
  `--go-dependency-graph-path` CLI flag (default: derived from `--data-path`) controlling
  where the graph artifact is read from.
- Two new, independently-gated `run_diagnostic` reporting additions: rarity-stratified
  AUROC, and hierarchy-violation rate/magnitude — both computed whenever
  `--run-diagnostic` runs and the graph artifact (for violations) / train-split label
  counts (for rarity) are available, regardless of `--hierarchy-loss-weight`.
- A new `--rare-term-threshold` CLI flag (default `100`, matching the roadmap source's
  own rare-term definition) controlling the rarity bucket boundary.
- A new small standalone script (modeled on the existing
  `experiments/ancestor_sweep/scripts/diagnostic_eval_stage2.py` pattern, but targeting
  `dataset_full`) to re-run the new diagnostic additions against the existing
  `kelmoir/cafa6-esm@main` (`trial-013`) checkpoint with no retraining — needed because
  `trial-013` predates this item and its original diagnostic JSON has neither new field;
  without this, there is no real control number to compare a hierarchy-loss trial's
  rarity/violation numbers against.
- One real GPU validation trial: `trial-013`'s exact config + `--hierarchy-loss-weight
  0.1`, pushed to its own branch.

**Out of scope (explicit follow-ups, not blockers for this item):**
- Tier 2 item 5 (GO-DAG-consistency post-processing at *inference* time — clamping
  predicted scores so parents never score below children, as a post-hoc numpy pass with
  no retraining). The roadmap's own text calls these two complementary, not redundant;
  this item is the training-time half only.
- A hyperparameter sweep over `--hierarchy-loss-weight` — the first trial picks `0.1` by
  inspection; a plateau-first sweep, if warranted, is a follow-on.
- Any change to the full graph-embedding plan (`experiments/graph_embedding_plan.md`),
  which stays gated on the OWA/PU-learning fix per the roadmap's own explicit
  distinction — this item's weaker, logic-only consistency claim is not a step toward
  that mechanism.
- Combining this with the now-shelved known-annotation-conditioning mechanism (Tier 1
  item 2B) — orthogonal (the penalty operates on classifier-head logits regardless of
  which head variant produced them), so no explicit interaction code is needed, but no
  joint trial is planned either.

## Why direct edges, not full ancestor closure

The restricted edge list uses only *direct* `is_a`/`part_of` edges (a term and its
immediate parent(s)), not the full transitive ancestor closure each term also has
available via `nx.descendants`. This is sufficient: if every direct edge satisfies
`parent_score >= child_score`, transitive consistency across any longer chain follows
automatically (the relation composes), so penalizing the full transitive closure would
roughly square the edge count (and the per-step cost) for no additional constraint power
once direct edges are already satisfied — a worse lazy-ladder outcome for the same
result.

## Edge index construction

New helper, next to `create_esm`/other model-setup helpers:

```python
def build_hierarchy_edge_index(label2id: Dict[str, int], graph_path: Path) -> torch.LongTensor:
    """Restrict generate_dataset_full.py's exported GO graph (go_dependency_graph.json)
    to direct is_a/part_of edges where both endpoints are in the current trainable
    vocabulary (label2id) -- see the design spec's "Why direct edges" section for why
    the transitive closure isn't used. Returns a (E, 2) LongTensor of
    (child_label_idx, parent_label_idx) pairs; cheap enough (well under a second,
    ~47k edges at the current 27,191-term vocabulary) to build once per run and keep
    resident on device for the whole training loop.
    """
    payload = json.loads(graph_path.read_text())
    graph = nx.readwrite.json_graph.node_link_graph(payload["graph"], edges="links")
    true_path_relationships = set(payload["true_path_relationships"])
    edges = [
        (label2id[u], label2id[v])
        for u, v, key in graph.edges(keys=True)
        if key in true_path_relationships and u in label2id and v in label2id
    ]
    return torch.tensor(edges, dtype=torch.long)
```

Built once in `main()`, immediately before the `if args.run_diagnostic:` block that
already exists near the end of training, AND once before the training loop starts when
`args.hierarchy_loss_weight > 0` (same tensor either way — built at most once per run,
not twice, by sharing one local variable computed as soon as either condition is true).
Moved to `device` for the training-loop use; the diagnostic's own use (see below) stays
on CPU/numpy since it runs against already-`.cpu()`'d probabilities.

New CLI flags:
- `--hierarchy-loss-weight` (float, default `0.0`): weight on the penalty term in the
  training loss. `0.0` (default) means the whole mechanism is a no-op — no edge index is
  built for training purposes, no behavior change for any existing invocation.
- `--go-dependency-graph-path` (str, default `None`): path to the exported GO graph.
  When unset, resolved as `Path(args.data_path) / "go_dependency_graph.json"`. If
  `--hierarchy-loss-weight > 0` and the resolved path doesn't exist, raise a clear
  `RuntimeError` at startup (fail fast — silently training without the requested penalty
  would be a worse failure mode than crashing immediately). If only `--run-diagnostic` is
  set (no hierarchy loss) and the path doesn't exist, degrade gracefully: print one
  `[WARN]` line and omit the violation-rate diagnostic fields, the same graceful-skip
  pattern the project already uses for optional `trackio` ("Experiment tracking" in
  `CLAUDE.md`).
- `--rare-term-threshold` (int, default `100`): train-split annotation-count boundary
  between "rare" and "common" for the diagnostic's rarity-stratified AUROC.

## Loss term

```python
def hierarchy_consistency_penalty(logits: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean hinge violation of GO's true-path rule over edge_index's direct
    (child_idx, parent_idx) pairs: relu(child_logit - parent_logit), averaged over every
    edge and every example in the batch. Zero when every edge already satisfies
    parent_logit >= child_logit for every example; monotonic with sigmoid, so this
    constrains predicted probabilities (parent_prob >= child_prob) without needing to
    compute probabilities at all.
    """
    child_logits = logits[:, edge_index[:, 0]]
    parent_logits = logits[:, edge_index[:, 1]]
    return torch.relu(child_logits - parent_logits).mean()
```

Wired into `main()`'s training loop (`training_run.py`'s `loss = compute_loss(...)`
line, both the plain-BCE branch and the `--nnpu` branch — the penalty operates on
`logits` directly, upstream of either branch's own loss, so it's added once after
whichever branch runs, not duplicated into each):

```python
if args.hierarchy_loss_weight > 0:
    loss = loss + args.hierarchy_loss_weight * hierarchy_consistency_penalty(logits, hierarchy_edge_index)
```

Computed inside the same `torch.autocast(...)` block the rest of the step already runs
under — no new dtype handling needed beyond what `compute_loss`'s own BCE term already
relies on.

This is compatible with every existing flag with no special-casing required:
`--use-lora`, `--use-aspect-input`, `--use-known-input`, `--nnpu`, `--label-smoothing`,
`--unknown-target-value` all only affect what `logits` the classifier head produces or
how `labels`/`loss_mask` are built — the hierarchy penalty is a pure function of
`logits` and the static `edge_index`, so no interaction code or mutual-exclusion check is
needed (contrast with `--use-known-input`'s own mutual exclusions with the other
loss-mask flags, which existed because those mechanisms all write the same `loss_mask`
key).

### Interaction with validation loss

The hierarchy penalty is **not** added to `validation_loss` (the value
`--checkpoint-selection val_loss` uses, and what gets printed/logged as `val_loss` every
epoch). This matches existing precedent: `--label-smoothing` and
`--unknown-target-value` already only modify the training-loop's local `labels`
variable, never the validation loop's (`training_run.py`'s validation loop re-reads
`batch["labels"]` directly, unmodified) — validation loss is kept as a "pure" measure of
the actual task throughout this codebase, regardless of what training-time-only
regularizers are active. Keeping this penalty out of `val_loss` preserves that
convention and keeps `val_loss` numbers comparable across trials regardless of whether
`--hierarchy-loss-weight` was set.

## Diagnostic additions (`run_diagnostic` / `_diagnostic_metrics_from_arrays`)

### Rarity-stratified AUROC

`compute_label_annotation_counts(train_labels_column, num_labels) -> np.ndarray` (new
helper): flattens the train split's raw per-example label-id lists and counts
occurrences per label index (`np.bincount`-style) — a property of the training data's
label distribution, independent of the hierarchy mechanism itself. Computed once in
`main()`, inside the existing `if args.run_diagnostic:` block (not needed otherwise).

`_diagnostic_metrics_from_arrays` gains an optional `label_is_rare: Optional[np.ndarray]
= None` parameter (shape `(num_labels,)`, boolean). When provided, in addition to the
existing pooled/mean/median AUROC, the function also partitions the per-label AUROC
values it already computes into a rare subset and a common subset (using the same
evaluable-label membership, not a separate pass) and adds `rare_mean_auroc`,
`rare_median_auroc`, `rare_n_labels_evaluable`, `common_mean_auroc`,
`common_median_auroc`, `common_n_labels_evaluable` to the returned result dict. When
`None` (the default — every existing call site, including the full
`check_run_diagnostic_known_mask.py` test suite, is unaffected), these keys are simply
absent from the result dict, not present-as-`None` — preserves exact backward
compatibility with every existing caller/test.

### Hierarchy violation rate

New helper, pure-numpy (testable without a model):

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

`run_diagnostic` gains two new optional parameters, `hierarchy_edge_index:
Optional[np.ndarray] = None` and `label_is_rare: Optional[np.ndarray] = None` (both
default `None`, preserving every existing call site unmodified). When
`hierarchy_edge_index` is provided, `run_diagnostic` calls
`compute_hierarchy_violations(probs, hierarchy_edge_index)` and merges its three keys
into the printed/returned result; when `label_is_rare` is provided, it's threaded into
`_diagnostic_metrics_from_arrays` as described above. `main()`'s existing
`run_diagnostic(...)` call site passes both whenever `args.run_diagnostic` is set and the
corresponding precomputation succeeded (edge index unless the graph file was missing and
only the graceful-skip `[WARN]` path was taken; rarity counts always, since they need
only the train split already in memory).

## Backfill script

`trial-013` (the pure-BCE baseline, `kelmoir/cafa6-esm@main`) was trained and diagnosed
before this item existed, so its existing `trial_metrics.json` has neither
rarity-stratified AUROC nor a violation rate — there is no real control number for the
hierarchy-loss trial's new metrics without one. New script,
`experiments/scripts/rerun_diagnostic_full.py`, modeled directly on the existing
`experiments/ancestor_sweep/scripts/diagnostic_eval_stage2.py` pattern (download a
checkpoint's `trial_metrics.json` for its config, reload the checkpoint, call
`run_diagnostic` against it, no retraining) but pointed at `dataset_full` (the stale
script's hardcoded legacy `dataset` path is not reused) and passing the new
`hierarchy_edge_index`/`label_is_rare` arguments unconditionally. Invocation:
`env/bin/python experiments/scripts/rerun_diagnostic_full.py kelmoir/cafa6-esm main` —
produces a `[DIAGNOSTIC]` line with the new fields for `trial-013`'s own checkpoint, used
purely as the control number; does not modify or re-push anything to the Hub.

## Files

- `training_run.py` — new `build_hierarchy_edge_index`, `hierarchy_consistency_penalty`,
  `compute_label_annotation_counts`, `compute_hierarchy_violations` helpers;
  `_diagnostic_metrics_from_arrays` and `run_diagnostic` gain the new optional
  parameters described above; `main()` gains the new CLI flags, the edge-index/rarity
  precomputation, the training-loop loss wiring, and the `run_diagnostic(...)` call
  site's two new kwargs.
- `experiments/scripts/rerun_diagnostic_full.py` — new standalone backfill script.
- No changes to `generate_dataset_full.py` (the graph artifact it already exports is
  reused as-is), `experiments/pk_eval/*` (this item is not PK-eval-specific — its
  expected effect is on ordinary rare-term AUROC, not the PK gap item 2 targeted), or
  `experiments/ancestor_sweep/scripts/diagnostic_eval_stage2.py` (left as-is; superseded
  for this purpose by the new script, not modified in place, since it still serves its
  original legacy-coarse-space callers).

## Testing / verification

Per the project's TDD convention, pure-numpy/no-GPU standalone `check_*.py` scripts
(mirroring `experiments/scripts/check_known_input_masking.py` /
`check_run_diagnostic_known_mask.py`):
- `build_hierarchy_edge_index`: a fabricated small graph fixture (a few nodes, a mix of
  `is_a`, `part_of`, and a non-true-path relationship like `regulates`, plus a node
  outside `label2id`) confirming only true-path edges with both endpoints in-vocabulary
  survive, no duplicates, correct `(child_idx, parent_idx)` orientation.
- `hierarchy_consistency_penalty`: fabricated logits confirming zero penalty when every
  edge already satisfies `parent_logit >= child_logit`, and a hand-computed nonzero
  value when some edges are violated.
- `compute_label_annotation_counts`: a fabricated small label-id-list column confirming
  correct per-label counts, including a label with zero occurrences.
- `compute_hierarchy_violations`: a fabricated small `probs`/`edge_index` fixture with a
  hand-computed expected violation rate and mean magnitude, plus a case with zero
  violations (rate and magnitude both `0.0`, no division-by-zero).
- `_diagnostic_metrics_from_arrays`'s new `label_is_rare` parameter: confirms the
  existing (no-argument) behavior is byte-for-byte unchanged, and that a fabricated
  rare/common split produces the expected per-bucket mean/median/count.
- An end-to-end CPU smoke run (`--hierarchy-loss-weight 0.1 --max-train-samples 100
  --max-val-samples 50 --epochs 1 --device cpu --run-diagnostic
  --diagnostic-max-val-samples 50`), confirming training completes, loss reporting
  includes the combined term without crashing, and the diagnostic prints the new
  rarity/violation fields.
- A real GPU trial (user's local GPU, `!`-command per `hardware_local_gpu.md`):
  `trial-013`'s exact config (`--model-name facebook/esm2_t12_35M_UR50D
  --unfreeze-last-n 4 --esm-dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01
  --batch-size 2 --grad-accum-steps 8 --epochs 30 --checkpoint-selection f1`) plus
  `--hierarchy-loss-weight 0.1 --run-diagnostic`, pushed to its own
  `kelmoir/cafa6-esm@hierarchy-loss-trial-001` branch — then the backfill script against
  `kelmoir/cafa6-esm@main` for the control numbers, and a roadmap log entry comparing
  pooled AUROC, rare-term AUROC, and violation rate between the two.

## Risks & mitigations

- **`--hierarchy-loss-weight`'s value (`0.1`) has no sweep history** — picked by
  inspection, same caveat the known-annotation-conditioning spec already flagged for its
  own new hyperparameters. Mitigation: treat as a starting point, not a tuned
  conclusion; open to a follow-on sweep if the first trial shows promise but an
  obviously wrong scale.
- **Rarity-stratified AUROC could still be noisy at the current val-set scale**
  (`--diagnostic-max-val-samples 2000` — a 27,191-term vocabulary's rare-term subset may
  have very few evaluable (both-classes-present) labels within that sample). Mitigation:
  the diagnostic already reports `n_labels_evaluable` per bucket; a near-zero rare `n`
  would itself be a finding (the diagnostic sample is too small to see this item's
  claimed effect at all), not silently hidden.
- **Validation-loss exclusion (see above) means `--checkpoint-selection val_loss` never
  reflects the hierarchy penalty** — intentional (keeps `val_loss` comparable across
  trials), but means a user selecting checkpoints by `val_loss` while this flag is on
  would pick the epoch with the best task loss, not the best combined objective. Not a
  concern for the validation trial itself (`--checkpoint-selection f1`, matching
  `trial-013`), but worth remembering if someone reaches for `val_loss` selection later
  with this flag on.

## Non-goals / explicit follow-ups

- Tier 2 item 5 (inference-time post-processing clamp) — separate, complementary, not
  part of this item.
- A `--hierarchy-loss-weight` sweep — first trial only, per the brainstorming decision.
- Extending the rarity-stratified/violation-rate diagnostic additions to any other
  mechanism's own evaluation (e.g. re-adding them to the now-shelved known-input
  diagnostic path) — out of scope; they're general `run_diagnostic` capabilities but
  this item only wires them into the plain/hierarchy-loss path's own call site.
