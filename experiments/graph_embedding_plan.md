# GNN / graph-embedding approach for GO-term relationships — future direction

**Not yet implemented. Written up 2026-09-04 after a design discussion.**

**Gate for starting this: the Open World Assumption (OWA) problem must be tackled to at least
some extent first.** The related sequential-cascade idea
(`experiments/sequential_cascade_plan.md`) was tried in an aspect-first form and failed for
exactly this reason: even in the coarser 3-aspect prediction space, the training loss still
didn't distinguish a *true* negative from an *unknown*, so the first stage of the cascade was
learning from contaminated labels before any graph-aware machinery even entered the picture
(see `experiments/ancestor_sweep/sweep_log.md` trial-080/081: masking sparse-protein negatives out of the loss
made AUROC measurably *worse*, -7%, because it starved an already-scarce negative signal instead
of fixing its reliability). A graph-embedding model built on top of the same contaminated labels
would just learn a more sophisticated way to fit the same noise — it doesn't route around the
problem, it inherits it. This session's `experiments/scripts/dataset_refinement_pilot.py` (soft
targets for unknown cells, testing whether resolving a few leaves a detectable trace) is the
current attempt at that prerequisite fix. Don't start building this before that line of work has
a confirmed, workable mitigation.

## Motivation

This session's Pareto analysis of the real, human-annotated CAFA-6 label space
(`experiments/scripts/analyze_go_term_pareto.py`) found that scoring weight (information
accretion, IA) and annotation frequency are concentrated in almost disjoint term sets
(Spearman ρ = -0.10 between the two rankings across 26,125 terms present in both `IA.tsv` and
`train_terms.tsv`):

- Terms annotated to **exactly 1 protein** are 20.6% of the vocabulary but carry **25.5%** of
  total IA weight, off just **1.0%** of annotation rows.
- Terms annotated **≤5 times** are 55.8% of the vocabulary and carry **64.9%** of total IA
  weight, off just **6.4%** of annotation rows.

So a majority of the challenge's actual scoring value sits on terms with only a handful of
positive examples — real few-shot territory, not ordinary class imbalance. A flat multi-label
classifier (the current architecture) has no mechanism to generalize to a term it has seen once
or never: it can only fit per-label frequency statistics, and a singleton term barely has any.
The one plausible route to generalizing onto that tail is **sharing statistical strength across
structurally related terms** — which is exactly what the GO graph encodes and the current
pipeline currently throws away at training time (it's used only as a label-propagation rule in
`generate_dataset.py`, never seen by the model itself).

## What this is (and isn't)

Not a change to how labels are *generated* (propagation via `is_a`/`part_of` stays as-is — that's
governed by GO's true-path rule and is a separate, already-solved concern). This is about giving
the **model** access to the GO graph's structure at training/inference time, so it can use a rare
term's position in the graph (its parents, siblings, and non-hierarchical relations) as a
substitute for the direct examples it doesn't have.

`generate_dataset.py` already exports everything needed for this:
`go_dependency_graph.json` contains the full GO graph across **all 5 relationship types**
(`is_a`, `part_of`, `regulates`, `positively_regulates`, `negatively_regulates`), not just the
2 propagation-safe ones — written specifically so a future stage could use it directly without
recomputing the graph.

## Two approaches, cheapest first

### 1. Static label-embedding (cheap, validate-the-mechanism-first)

Compute a fixed embedding per GO term from the graph alone (e.g. `node2vec`, or a
relation-aware method like `TransE`/`RotatE` treating `is_a`/`part_of`/`regulates`-family edges
as distinct relation types), independent of any protein data. Replace the classifier's flat
per-term output layer with a **bilinear/dot-product compatibility score** between the protein's
ESM2 representation and each term's fixed graph embedding — this is the standard "label
embedding" pattern used for exactly this long-tail multi-label setting (see Learning points
below). A rare/unseen-in-training term still gets a meaningful embedding purely from its graph
neighborhood, so the model can score it via similarity to better-observed neighbors even with
zero or one direct positive example.

This is the cheap version to try first, per this project's own established pattern (see
`sequential_cascade_plan.md`'s phased staging, and the "find the plateau at cheap scale before
scaling up" sweep strategy already in use): it needs no new training loop, just a different
classifier head, and answers "does graph-based sharing help at all" before committing to a full
GNN build.

### 2. Learned GNN encoder over the label graph (full version, only if #1 shows signal)

Instead of a *fixed* embedding, train a graph neural network (e.g. a relational GCN/GAT, since
`is_a`/`part_of` and the `regulates` family are qualitatively different relations and shouldn't
share the same message-passing weights) end-to-end alongside the protein encoder, so term
embeddings adapt to what actually helps prediction rather than staying fixed from a
graph-structure-only pretraining pass. Two-tower architecture: the existing ESM2 protein encoder
on one side, a GNN term encoder on the other, joined by the same
bilinear-compatibility mechanism as approach 1.

## Why `regulates`-family edges matter specifically here

`is_a`/`part_of` propagation already gets a term's ancestor structure baked into the label space
today — a graph embedding built only from those two relation types would be somewhat redundant
with what depth-based coarse-term selection already encodes. The `regulates`/
`positively_regulates`/`negatively_regulates` edges are the actually-new information: they're
excluded from label *propagation* (true-path rule doesn't support them — a protein doing the
regulator process doesn't necessarily do the regulated one), but they still carry real biological
correlation signal — proteins annotated for regulating a process are plausibly more likely to
co-occur with annotations for the process itself, a pattern no current part of the pipeline
gives the model access to. This is the concrete answer to "would `regulates` and shared-ancestry
help a rare-term cascade": shared ancestry is mostly already-exploited signal, `regulates` is the
genuinely unexploited one.

## Learning points / prior art

This isn't a novel idea in the GO-prediction literature — worth grounding the design in what's
already been tried before committing engineering time:

- **DeepGO / DeepGOPlus** — among the first to combine a sequence encoder with a GO-graph-derived
  component for function prediction; establishes the basic two-tower pattern.
- **DeepGraphGO** — GNN directly over the GO graph combined with sequence/network features,
  explicitly targeting the long-tail rare-term problem this session's analysis surfaced.
- **TALE / ProTranslator** — label-embedding approaches (the "static embedding + compatibility
  score" pattern in approach 1 above) built specifically to handle GO terms with few or zero
  training examples, which is precisely this project's singleton/≤5-annotation tail.

Read these before designing the exact GNN/embedding mechanics — no need to rediscover
architectural choices (relation-aware message passing, compatibility-score heads, cold-start
handling for never-seen terms) that published work has already worked through.

## Open questions to resolve at implementation time (deliberately not fixed now)

- Whether a static embedding (approach 1) already captures most of the benefit, or whether the
  full learned GNN (approach 2) is needed to see a real gain — resolve empirically, cheap version
  first.
- How much actual graph coverage the `regulates` family has in `go-basic.obo` relative to
  `is_a`/`part_of` — if it's sparse, its contribution may be marginal regardless of architecture.
  Worth a quick count against `go_dependency_graph.json` before investing further.
- Cold-start handling for a genuinely novel term never seen positively anywhere in training (the
  true zero-shot case, one level beyond the singleton/rare-but-present case this analysis
  covered).
- Evaluation methodology: need to isolate whether any observed gain is from graph-awareness
  specifically, versus just added model capacity/parameters — a matched-parameter-count ablation
  (flat classifier vs. graph-aware head at equal added parameter budget) would be the honest
  comparison once this is actually built.
- Interaction with the still-unresolved OWA problem: even with graph-aware sharing, the model is
  only as trustworthy as the labels it trains on — this doesn't replace fixing label reliability,
  it's a separate, additive improvement that assumes that prerequisite is already handled.
