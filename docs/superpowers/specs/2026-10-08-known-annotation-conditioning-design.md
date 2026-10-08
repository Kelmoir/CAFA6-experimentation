# Known-Annotation (PK) Conditioning — Training-Time Input Channel (design)

**Goal:** Give the model a real partial-knowledge (PK) input channel at training time —
per `experiments/cafa6_next_steps_plan.md` Tier 1 item 2, part (B) — so it can learn to
predict a protein's *remaining* unannotated GO terms conditioned on a subset of terms
already known for that protein, instead of always predicting the full annotation set
from sequence alone (today's pure-NK setup). This is the training-time counterpart to
part (A), the temporal PK/LK/NK evaluation harness (`experiments/pk_eval/`,
`docs/superpowers/specs/2026-10-05-temporal-pk-eval-design.md`), already built and
verified. Part (A) is NOT changed by this work — it stays the post-training holdout
check, scored against each protein's real t0-known terms rather than a synthetic split.

**Architecture:** Extend `training_run.py` in place — no new scripts, no new dataset
build step. A new on-the-fly masking transform inside `preprocess_batch` synthesizes a
"known" / "novel" split per example per draw. A new `KnownAwareClassificationHead` /
`KnownConditionedEsmForSequenceClassification` pair (direct extension of the existing
`AspectAwareClassificationHead` / `AspectConditionedEsmForSequenceClassification`
pattern) injects a pooled known-term embedding into the classifier head. Known/revealed
terms are excluded from that example's loss via the existing `loss_mask`/`compute_loss`
machinery — no loss-function change.

**Tech Stack:** PyTorch (`torch.nn.EmbeddingBag` for variable-size-set pooling —
framework-native, no manual pooling/padding code). Reuses `generate_dataset_full.py`'s
existing `dataset_full` (27,191-term full ancestor-closure label space) completely
unchanged — no new dataset artifact.

**Spec context:** `experiments/cafa6_next_steps_plan.md` Tier 1 item 2 (Partial-Knowledge
conditioning), part (B). Decided in this session's brainstorming (superseding any
earlier sketch in the roadmap doc that assumed a precomputed per-protein known/novel
split would be needed):
- Train on t0 data only; masking is synthetic, not temporal (temporal diffing is part
  (A)'s job, run after training as a separate holdout check, not for generating training
  data).
- Mix in a tunable fraction of pure-zero-known (NK) examples alongside randomly-revealed
  ones, so the model doesn't forget how to perform the original NK task.
- Known/revealed terms are removed from the loss entirely for that example (not kept as
  targets) — avoids teaching a trivial "echo the input" shortcut, and keeps the training
  objective aligned with what part (A)'s eval harness actually scores (novel terms only).
- Model-input mechanism is a learned per-term embedding + sum-pooling (not a raw
  27,191-dim one-hot concatenation) — chosen partly because the resulting per-GO-term
  embedding table may be reusable input signal for a later term-relation/DAG-consistency
  item on the same roadmap, not only for this item.

## Background

The current pipeline (`generate_dataset_full.py` → `training_run.py`) is pure NK: every
training example's target is a protein's full ancestor-propagated annotation closure
(27,191-term label space, one dense multi-hot `labels` tensor per protein), predicted
from sequence alone. There is no "already known, predict the rest" signal anywhere in
the pipeline — part (A)'s own prerequisite check (`experiments/cafa6_next_steps_plan.md`,
2026-10-04 entry) confirmed this directly.

The project already has one precedent for a non-sequence auxiliary model input: the
aspect-conditioning ablation (`AspectAwareClassificationHead`,
`AspectConditionedEsmForSequenceClassification`, trial-077/078 —
`experiments/ancestor_sweep/sweep_log.md`). It concatenates a small ground-truth
auxiliary vector (3-dim `[C, F, P]` one-hot) onto the pooled CLS representation before
the classifier head's dense layer, leaving the ESM backbone's own forward pass
untouched. That pattern is the right injection *point* for a known-terms signal, but its
direct-concatenation *mechanism* does not scale as-is: `aspect_input` is 3-dim, a
known-terms vector is 27,191-dim. A literal copy (`Linear(hidden_size + 27191,
hidden_size)`) would add roughly `27191 * hidden_size` parameters to one layer — at
`esm2_t12_35M_UR50D` (`hidden_size=480`, the Tier-0 baseline's architecture — see
`experiments/full_label_sweep/sweep_log.md`), that is ~13M new parameters in a single
dense layer, comparable to ~40% of the entire current model's size, and raw one-hot
concatenation of a huge sparse vector tends to memorize specific known-term
combinations rather than generalize across related GO terms.

## Scope

**In scope:**
- A new on-the-fly known/novel masking transform inside `preprocess_batch`, operating
  only on `dataset_full`'s existing `labels` column — no new dataset build script, no
  new HF Hub dataset push.
- A new learned per-term embedding + `EmbeddingBag(mode="sum")` pooling mechanism for
  the known-terms input, injected at the classifier head via a new
  `KnownAwareClassificationHead` / `KnownConditionedEsmForSequenceClassification` pair.
- Excluding known/revealed terms from the loss via the existing `loss_mask` /
  `compute_loss` path.
- New `training_run.py` CLI flags gating all of the above (default off, so every
  existing invocation is unaffected).
- Minimal changes to `score_temporal_pk_eval.py` (part (A)'s scorer) so a
  `--use-known-input`-trained checkpoint can be scored with each protein's *real*
  t0-known terms fed through the same `known_input` path — not a training-data change,
  just wiring the existing eval harness to the new optional model input.

**Out of scope (explicit follow-ups, not blockers for this item):**
- Any change to part (A)'s evaluation harness methodology itself (categorization,
  Fmax formula, cafaeval cross-check) — already built, verified, and left untouched.
- Combining this with PU-learning's two-tier relabeling — PU-learning is parked
  (`experiments/cafa6_next_steps_plan.md` item 7, `project_pu_learning_nnpu_km2.md`);
  may be reintroduced later in the training pipeline, not as part of this item.
- A cross-attention or backbone-level conditioning mechanism — rejected for this item,
  see "Alternatives considered" below.
- Retraining/re-sweeping regularization hyperparameters from scratch for the new
  architecture variant — follow-on work once this lands, not part of the design itself.
- A learned initialization or warm-start for the new known-term embedding table — starts
  randomly initialized, same as the classifier head's own weights.

## Data pipeline — on-the-fly masking (`preprocess_batch`)

No new dataset artifact. The split is derived at preprocessing time, every time an
example is drawn, directly from `dataset_full`'s existing dense `labels`/raw label-id
list for that protein (the full t0 ancestor-propagated closure — already exactly what
`preprocess_batch` decodes today at `training_run.py:394-418`). Because preprocessing
runs lazily via `dataset.with_transform(...)` (`training_run.py:961-965`), this
naturally re-randomizes on every access — a different known/novel split each epoch for
the same protein, for free, rather than a frozen split baked into a dataset file.

Per example, when `--use-known-input` is set:

1. Draw `is_zero_known ~ Bernoulli(known_mix_zero_frac)`.
2. If `is_zero_known`: `known = {}` (pure NK — identical to today's task; every
   positive term stays a loss target).
3. Else: draw `r ~ Uniform(0, known_reveal_max)`, then sample
   `floor(r * |positives|)` of that protein's positive term ids uniformly at random
   (without replacement) as `known`. The remaining positives (plus all negatives) stay
   loss targets.
4. Output, alongside the existing `labels` tensor:
   - `known_term_ids`: the sampled positive term ids (variable-length per example — fed
     to `EmbeddingBag` as `(indices, offsets)`, see below; empty for a zero-known draw).
   - `loss_mask`: dense `(num_labels,)` tensor, same shape/semantics as today's
     `loss_mask` (`training_run.py:432-454`) — 0 at each `known` position, 1 everywhere
     else (novel positives and all negatives).

This reuses the exact `loss_mask` contract `compute_loss` already consumes
(`training_run.py:552-562`) — no loss-function change, and composes the same way
`--mask-sparse-negatives-max-total-terms` and `--loss-mask-from-dataset`'s two-tier
exclusion already do (mutually exclusive with those two for now — see "Interaction with
existing loss-mask flags" below, not a hard technical conflict, just undefined combined
semantics until someone needs it).

New CLI flags (mirroring `--use-aspect-input`'s existing style, default off/no-op):
- `--use-known-input` (bool): gates the masking transform, the new model class, and the
  new loss-mask source.
- `--known-mix-zero-frac` (float, default `0.3`): fraction of draws forced to pure NK.
- `--known-reveal-max` (float, default `1.0`): upper bound of the reveal-fraction
  sampling range for non-zero draws.

### Interaction with existing loss-mask flags

`--mask-sparse-negatives-max-total-terms` and `--loss-mask-from-dataset` already write
`output_dict["loss_mask"]` (`training_run.py:420-454`). `--use-known-input`'s masking
step becomes a third writer of the same key. For this item, `--use-known-input` is
documented as mutually exclusive with the other two (raise at argument-parsing time if
more than one is set) — combining "exclude sparse negatives" / "exclude two-tier-flagged
cells" / "exclude revealed-known terms" into one mask is a real question (union? which
takes precedence?) with no forcing use case yet. Revisit if a later item needs the
combination.

## Model — known-term embedding, pooling, and injection

New classes, direct siblings of the existing aspect-conditioning pair
(`training_run.py:134-197`):

```python
class KnownAwareClassificationHead(torch.nn.Module):
    """Like AspectAwareClassificationHead (see above), but the auxiliary input is a
    variable-size set of known GO term ids rather than a fixed 3-dim aspect vector:
    pooled via a learned per-term embedding + sum (EmbeddingBag), not concatenated
    directly. A raw 27,191-dim one-hot concat was rejected -- see the design spec's
    "Alternatives considered" section for why.
    """

    def __init__(self, config, num_labels: int, known_embed_dim: int = 64):
        super().__init__()
        self.known_embedding = torch.nn.EmbeddingBag(num_labels, known_embed_dim, mode="sum")
        self.dense = torch.nn.Linear(config.hidden_size + known_embed_dim, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, known_indices=None, known_offsets=None, **kwargs):
        if known_indices is None or known_offsets is None:
            raise ValueError("KnownAwareClassificationHead requires known_indices/known_offsets")
        x = features[:, 0, :]
        x = self.dropout(x)
        known_vec = self.known_embedding(known_indices, known_offsets)
        x = torch.cat([x, known_vec.to(dtype=x.dtype, device=x.device)], dim=-1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
```

`EmbeddingBag(mode="sum")` is the framework-native fit for "pool a variable-size set
from a large vocabulary into one fixed-size vector" — the standard bag-of-items
pattern, taking flattened `indices` plus per-example `offsets` (identical calling
convention to any bag-of-words usage), rather than requiring manual padding + masked
sum code. A fully zero-known example is simply an empty bag at that offset, which
`EmbeddingBag` already handles (sums to the zero vector), matching today's behavior
with no known-input at all.

`KnownConditionedEsmForSequenceClassification` mirrors
`AspectConditionedEsmForSequenceClassification` (`training_run.py:166-197`) exactly,
substituting `known_indices`/`known_offsets` for `aspect_input` in its `forward()` and
applying the same manual Linear re-initialization the aspect variant already does after
`post_init()` replaces the head (`training_run.py:176-187`).

`create_esm` (`training_run.py:200-350`) gains a `use_known_input: bool = False` /
`known_embed_dim: int = 64` pair alongside the existing `use_aspect_input`, selecting
`KnownConditionedEsmForSequenceClassification` the same way `use_aspect_input` selects
`AspectConditionedEsmForSequenceClassification` at `training_run.py:323`. Combining
`--use-known-input` with `--use-aspect-input` is not handled by this design (a model
needing both inputs would need a head that concatenates all three pieces — deferred,
no current use case requires both at once) and is rejected at argument-parsing time
for now, same as the mutual exclusion with the other loss-mask flags above.
`--use-known-input` + `--use-lora` is rejected for the same reason
`--use-aspect-input` + `--use-lora` already is (`training_run.py:223-227`): LoRA's
`modules_to_save=["classifier"]` path assumes the stock classifier head shape.

New CLI flag: `--known-embed-dim` (int, default `64`) — embedding dimension per GO
term. At `known_embed_dim=64` and 27,191 terms, the embedding table itself is ~1.74M
parameters (`27191 * 64`) — small and constant regardless of label-space size, unlike
the rejected one-hot-concat approach's `hidden_size`-scaling cost.

### Collation

`data_collate_function` (`training_run.py:471-...`) gains a fourth optional batch key,
alongside `aspect_input`/`loss_mask`: flatten each example's `known_term_ids` into one
`known_indices` tensor plus a `known_offsets` tensor (cumulative start position of each
example's bag within the flattened tensor) — the standard `EmbeddingBag` batching
convention, analogous to how `torch.nn.utils.rnn.pad_sequence` is already used for
`input_ids`/`attention_mask` just above it, but offset-based instead of padding-based
since `EmbeddingBag` doesn't need padding at all.

## Alternatives considered

**Direct one-hot concatenation** (literal extension of `AspectAwareClassificationHead`,
`Linear(hidden_size + 27191, hidden_size)`): rejected. ~13M new parameters in one dense
layer at the Tier-0 baseline's scale (`esm2_t12_35M_UR50D`, hidden_size 480) —
disproportionate to the whole model's current size — and raw one-hot concatenation of a
large sparse vector tends to memorize specific known-term combinations rather than
generalize across related GO terms the way a learned embedding can (terms that are
biologically or hierarchically related can converge toward similar embeddings; one-hot
dimensions are independent by construction).

**A dense `Linear` projection of the multi-hot vector** instead of `EmbeddingBag`:
mathematically equivalent to sum-pooled embeddings (a `Linear` layer applied to a
multi-hot vector sums exactly the rows of its weight matrix corresponding to set bits —
the weight matrix *is* the embedding table), but computationally wasteful at this
vocabulary size (materializing a dense 27,191-dim one-hot vector per example just to
immediately matrix-multiply it, instead of a sparse index lookup). Not pursued as a
separate design, since it collapses to the chosen `EmbeddingBag` approach once the
inefficiency is recognized.

**Cross-attention between the sequence representation and known-term embeddings inside
the backbone** (treating each known term as an extra attended-to "token" rather than a
pooled side-channel concatenated at the classifier head): more expressive in principle,
but breaks the project's established precedent that the ESM backbone's own forward pass
stays untouched when adding an auxiliary input (`AspectAwareClassificationHead`'s own
docstring, `training_run.py:140-143`) — the explicit point of that precedent is keeping
any quality delta attributable to the new signal, not to a changed encoder. It would
also require new transformer-layer-scale engineering for an unproven additional
benefit. Rejected for this item; worth reconsidering only if the simpler
classifier-head-level mechanism turns out insufficient.

## Eval-harness integration (`score_temporal_pk_eval.py`)

Part (A)'s scorer (`experiments/pk_eval/scripts/score_temporal_pk_eval.py`) currently
runs inference with no known-term input at all (today's pure-NK model). For a
`--use-known-input`-trained checkpoint, the scorer's `run_inference` step needs to:

1. Load the checkpoint via `KnownConditionedEsmForSequenceClassification` instead of
   `AutoModelForSequenceClassification` (mirroring how a `--use-aspect-input` checkpoint
   would already need to be loaded — `score_temporal_pk_eval.py` does not currently
   handle either case, since nothing trained with either flag has been scored by it
   yet).
2. For each row's `known` category pool (PK/LK rows, per part (A)'s own categorization —
   `experiments/pk_eval/scripts/build_temporal_pk_eval.py`), feed that row's *real*
   t0-known term ids as `known_indices`/`known_offsets` — the genuine temporal known-set,
   not a synthetic draw. For NK rows, feed an empty bag (zero known terms), matching
   training's zero-known case exactly.
3. No change to the categorization logic, the Fmax formula, or the cafaeval cross-check
   — this is purely a model-loading and inference-input change, confined to
   `run_inference`.

This wiring is listed here for completeness (it's required before part (B)'s trained
checkpoints can be meaningfully compared against part (A)'s existing pure-NK baseline
numbers), but is not re-specified in detail — the implementation plan should treat it as
a small, final task once the training-side changes above are built and tested.

## Files

- `training_run.py` — `preprocess_batch`, `data_collate_function`, `compute_loss`
  (unchanged, reused as-is), `create_esm`, new `KnownAwareClassificationHead` /
  `KnownConditionedEsmForSequenceClassification` classes, new CLI flags in `main()`'s
  `argparse` setup and the `create_esm(...)` / `model_kwargs` call sites
  (`training_run.py:959-1010`, `:1174-1272`, `:1320-1325`, `:1483-1504` — the same
  call sites `--use-aspect-input` already threads through).
- `experiments/pk_eval/scripts/score_temporal_pk_eval.py` — `run_inference` gains the
  known-conditioned loading/inference path described above.
- No changes to `generate_dataset_full.py`, `experiments/pk_eval/scripts/
  build_temporal_pk_eval.py`, `experiments/pk_eval/scripts/pk_eval_common.py`, or
  `experiments/pk_eval/scripts/crosscheck_cafaeval.py`.

## Testing / verification

Per the project's TDD convention (every function gets a RED→GREEN cycle):
- `preprocess_batch`'s new masking branch: unit tests for the pure-zero-known draw, a
  partial-reveal draw (`loss_mask` has exactly the revealed positions zeroed, everything
  else 1), and the edge case of a protein with very few positive terms (reveal-fraction
  rounding to 0 revealed terms).
- `KnownAwareClassificationHead`/`KnownConditionedEsmForSequenceClassification`: a
  smoke-forward test (fixed small `config`, fabricated `known_indices`/`known_offsets`)
  confirming output shape `(batch, num_labels)` and that an empty bag produces the same
  logits as omitting the known-input path entirely (zero-vector equivalence).
- `data_collate_function`'s new `known_indices`/`known_offsets` batching: a unit test
  with variable-length per-example known-sets confirming the offsets are correct
  (standard `EmbeddingBag` contract — offset[i] is the flattened start index of
  example i's bag).
- An end-to-end CPU smoke run (`--use-known-input --max-train-samples 100
  --max-val-samples 50 --device cpu`, mirroring the existing smoke-test invocation
  pattern in "Common commands") confirming training runs to completion and loss
  decreases, before any real GPU run.
- A real GPU run (via the user's local GPU, `!`-command — per
  `hardware_local_gpu.md`) to confirm the architecture trains at the Tier-0 baseline's
  scale, before any scoring comparison against part (A)'s existing numbers.

## Risks & mitigations

- **New embedding table starts randomly initialized** — the model has to learn useful
  per-term representations from scratch, same as any other newly-initialized head
  component (no different in kind from the classifier head's own `out_proj` weights).
  Mitigation: none needed beyond normal training; flagged only so a low-signal early
  epoch isn't mistaken for a bug.
- **`known_mix_zero_frac` / `known_reveal_max` are new hyperparameters with no sweep
  history** — unlike `--unfreeze-last-n`/dropout/weight-decay, which the Tier-0 plateau
  sweep (`experiments/full_label_sweep/sweep_log.md`) already settled. Mitigation: start
  from the defaults proposed here (`0.3`/`1.0`), confirmed only by the smoke test above;
  treat as open to a follow-on plateau-first sweep rather than assumed-correct.
- **Loss-mask flag interaction** (see "Interaction with existing loss-mask flags")
  — mitigated by rejecting the combination outright at argument-parsing time rather than
  silently picking a precedence order nobody asked for.

## Non-goals / explicit follow-ups

- Reintroducing PU-learning's two-tier relabeling alongside this mechanism — explicitly
  parked (`project_pu_learning_nnpu_km2.md`); revisit only if a later item needs it.
- A combined aspect+known input head — no current use case; `--use-known-input` +
  `--use-aspect-input` is rejected, not silently ignored.
- A hyperparameter sweep over `known_mix_zero_frac`/`known_reveal_max`/`known_embed_dim`
  — out of scope for the initial implementation; the defaults above are a starting
  point, not a tuned conclusion.
- Reusing the learned known-term embedding table for a future term-relation/DAG-
  consistency roadmap item — noted as a motivating consideration for choosing the
  embedding approach (see spec context above), not committed to or designed here.
