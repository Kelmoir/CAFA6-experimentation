# Sequential cascade architecture — plan for later use

Not yet implemented. Written up 2026-07-28 after a design discussion, for reference once
the current flat-classifier sweep (`esm2_t36_3B_lora_scaleup`, see `sweep_log.md`) has a
confirmed, promoted result to build on. Gate for starting this: the current pipeline's
flat 383-term classifier must be tuned properly and confirmed at production scale first
(a `--push-to-hub`'d, HEALTHY result — trial-074 or its successor) before any of this
starts. Don't begin implementation before that's true.

## What this is (and isn't)

A **sequential cascade** of classifiers, ordered coarse-to-fine through the GO hierarchy's
depth levels — not a mixture-of-experts in the usual deep-learning sense (MoE = a gating
network routes each input to a few *parallel* experts whose outputs get combined; this is
neither parallel nor gated). Each stage ("expert") is conditioned on every previous stage's
output. Closest established precedent: **Progressive Neural Networks** (Rusu et al., 2016)
— freeze each previously-trained column, give the next column lateral connections from all
previous columns' activations. This design is a simplified variant: one injection point
(the classification layer) instead of PNN's per-layer lateral connections, and "activation"
here specifically means the sigmoid'd prediction probability vector (bounded [0,1]), not a
hidden representation — closer to the classic "feed the parent's predicted label
distribution to the child" pattern from the hierarchical multi-label classification (HMC)
literature (survey: Silla & Freitas, 2011) than to PNN's richer representation transfer.

## Goals

1. **Compute/memory efficiency** via sharing the frozen backbone across experts.
2. **Better performance on deeper/harder GO terms**, by giving each finer expert direct
   access to every coarser ancestor level's prediction as context, instead of relying on
   the loss function alone to learn hierarchy consistency.

## Architecture

- **Shared frozen backbone, per-expert LoRA adapters.** All experts share the exact same
  `--lora-layers-to-transform` boundary (e.g. all share layers 0–25 frozen/adapter-free,
  each expert has its own independent LoRA adapters only on layers 26–35). Fixing the same
  boundary across every expert is what makes the "compute the shared part once" optimization
  clean — if experts used different depths, the shared-prefix win gets messier to realize.
  Each expert's adapters are independent (not stacked on top of previous experts' adapters —
  previous experts are frozen and contribute only via their forward pass).
- **Per-expert label space**: each expert predicts only the terms newly introduced at its
  own depth (not re-predicting its ancestors' terms too) — standard HMC practice, and
  matches how `generate_dataset.py` already thinks about depth-based term selection.
- **Cross-expert input**: expert *i*'s classification layer receives
  `[shared_backbone_representation ; expert_1_probs ; expert_2_probs ; ... ; expert_{i-1}_probs]`
  — **all** previous experts' sigmoid'd probability vectors concatenated, not just the
  immediately preceding one. Cheap: each contributes only a small probability vector (one
  float per label at that level), not a full hidden state, so the growing input stays small
  even for the last expert in the chain. Exact fusion mechanism (plain concatenation into
  the existing linear classifier head vs. a small dedicated fusion MLP) is an implementation
  detail to settle when actually building this, not fixed here.
- **Training order**: strictly staged, matching PNN's own protocol. Train and fully confirm
  expert 1 (coarsest level), freeze it completely, then train expert 2 with expert 1 fixed
  (forward-only, contributing input features) — repeat down the chain. This is forced by the
  architecture (expert *i+1* can't train without expert *i*'s output existing), not a real
  choice. No joint/end-to-end fine-tuning of earlier experts once frozen, at least for this
  initial version — a possible future refinement, not part of the first build.

## Efficiency mechanisms

- **Frozen-prefix reuse is not automatic.** `--lora-layers-to-transform` today only makes
  autograd skip *backward* for early layers during training of a single adapter — it doesn't
  give a mechanism for computing the shared prefix once and branching for multiple
  simultaneously-active experts at inference. Realizing the actual compute saving needs new
  code: extract the hidden state at the shared/adapted boundary layer, run it through each
  expert's adapted suffix separately, rather than re-running the full end-to-end forward pass
  once per expert.
- **peft already supports holding multiple named adapters on one base model** via
  `add_adapter(name, config)` / `set_adapter(name)` — avoids duplicating the frozen 2.85B-param
  backbone in memory once per expert. Doesn't by itself give the "compute prefix once" saving
  (still need the manual boundary-splitting above), but is the right primitive to build on.
- **Inference-time short-circuiting** (bonus, not yet confirmed with the user beyond being
  proposed): skip a finer expert entirely for a given protein if its parent term wasn't
  predicted positive — the true-path rule guarantees deeper terms can't apply if their parent
  doesn't. Free additional savings on top of the shared-backbone win.

## Staging (confirmed via discussion 2026-07-28)

**Phase 1 — validate the cascade mechanism on the existing label space.** Restructure the
current 383-term depth≤2 space (3 aspect roots + depth-1 + depth-2 terms) into a 3-expert
cascade (root → depth-1 → depth-2), reusing the already-working dataset/infra — no
`generate_dataset.py` changes needed. Purpose: find out whether chaining coarse→fine outputs
actually helps at all, using data we already trust, before also taking on a second,
independent unknown (deeper-label sparsity). Isolates variables, consistent with how every
other experiment in this sweep has been staged (one axis at a time).

**Phase 2 — extend to real leaf terms (contingent on phase 1 showing the mechanism helps).**
Only after phase 1 validates the architecture: raise `COARSE_MAX_DEPTH` or otherwise expose
deeper/leaf GO terms in `generate_dataset.py`, and extend the cascade with additional
expert(s) for those deeper levels. This is where the original motivating observation (models
failing at the very leaves) actually gets tested directly — phase 1 alone doesn't touch true
leaves, since depth≤2 terms aren't leaves in most branches of the GO graph.

## Open questions to resolve at implementation time (deliberately not fixed now)

- Exact fusion mechanism for combining shared-backbone features with previous experts'
  probability vectors (concatenation into the existing classifier vs. a small dedicated
  fusion layer).
- Whether/how to enforce hard hierarchy consistency at inference (e.g., zeroing a child
  prediction if its parent wasn't predicted positive) vs. leaving it soft/learned via the
  cross-expert input alone.
- Whether error propagation from an earlier expert's mistakes needs an explicit mitigation
  (e.g., a gated/residual fusion so a later expert can learn to discount bad upstream signal)
  or whether the standard cross-entropy training signal handles this adequately in practice —
  something to observe empirically in phase 1, not pre-solve architecturally.
- Per-expert evaluation: track both isolated per-expert AUROC/F1 (to see which stage is
  weak) and end-to-end pipeline metrics (reflecting real deployed behavior, including
  propagated errors) once this is built.
