# Temporal PK/LK/NK Evaluation Harness (design)

**Goal:** Build a genuinely temporal Partial-Knowledge evaluation set — per
`experiments/cafa6_next_steps_plan.md` Tier 1 item 2, part (A) — by diffing CAFA-6's own
`train_terms.tsv` (t0) against the already-cached current UniProt/Swiss-Prot dump (t1),
and score the current trained model's predictions against it with the real CAFA
IA-weighted Fmax metric, broken down by aspect and by PK/LK/NK/unseen category. This is
the measurement step; it does not change the model or training pipeline. It also
produces, as a byproduct, the "known-annotation" vector shape that Tier 1 item 2's part
(B) (training-time PK conditioning) will need later.

**Architecture:** Two new scripts under a new `experiments/pk_eval/` directory —
`build_temporal_pk_eval.py` (the t0/t1 diff → labeled validation set) and
`score_temporal_pk_eval.py` (loads a checkpoint, runs inference, computes the
aspect×category Fmax breakdown) — plus a one-time cross-check against the community
`cafaeval`/`protea` package to validate the hand-rolled Fmax implementation.

**Tech Stack:** Python, Biopython's `Bio.SwissProt.parse()` (already the project's
proven pattern for streaming `uniprot_sprot.dat.gz`, from `experiments/mf_span`),
`datasets` (HF `Dataset`/`save_to_disk`), reuses `generate_dataset.py`'s
`propagate_protein_terms()` (imported, not duplicated). `cafaeval`/`protea` is a new,
one-time-use dependency for the cross-check only — not a standing pipeline dependency.

**Spec context:** `experiments/cafa6_next_steps_plan.md` Tier 1 item 2 (Partial-Knowledge
conditioning), its 2026-10-04 prerequisite-check entry (confirmed current pipeline is
pure NK) and CAFA-5-shortcut-refuted entry (confirmed no ready-made PK dataset exists
anywhere, real PK ground truth requires diffing two independently-dated UniProt-GOA
snapshots).

## Background

The project's roadmap flags PK conditioning as a Tier 1 item, but the current pipeline
(`generate_dataset.py`/`training_run.py`) is pure NK: every protein's label is its full
true-path-propagated annotation closure, with no held-out "known" subset anywhere. CAFA's
own PK/LK/NK categories are normally built by the official assessors diffing two
independently-dated UniProt-GOA experimental-annotation snapshots — work this repo cannot
shortcut via CAFA-5's Kaggle release (confirmed: structurally identical single-snapshot
shape to CAFA-6's own data, no ready split).

The key unlock found during design: this repo already has a **second, later** UniProt
snapshot sitting locally — `data/uniprot/uniprot_sprot.dat.gz`, fetched 2026-08-01 for the
unrelated `mf_span` side-project, whose entries carry version timestamps up to June 2026.
`train_terms.tsv` is dated October 2025. That is ~9-10 months of genuine temporal distance,
with zero new fetching required — exactly the two-snapshot diff CAFA's own evaluator does,
built entirely from data already on disk.

## Scope

**In scope:**
- Diffing `train_terms.tsv` (t0) against the cached `uniprot_sprot.dat.gz` (t1) to build a
  genuine, per-aspect PK/LK/NK/unseen-categorized validation set.
- Scoring the current trained model's predictions against it with IA-weighted Fmax,
  broken down by aspect and category, with the official known-terms-exclusion semantics.
- A one-time cross-check of the hand-rolled Fmax math against `cafaeval`/`protea`.

**Out of scope (explicitly deferred, not forgotten):**
- Part (B): the `training_run.py` architecture change (new "known annotations" model
  input channel) and the much larger diffing job that would be needed across the full
  training set to actually train on genuine known/novel pairs. This spec only builds the
  evaluation harness and the label-vector shape (B) will reuse.
- TrEMBL inclusion — Swiss-Prot (reviewed) only, per explicit decision; CAFA-6's own
  training proteins are themselves drawn from the reviewed/experimentally-curated subset,
  so this doesn't undercount in a way that matters here.
- Refreshing `data/uniprot/uniprot_sprot.dat.gz` to an even newer snapshot — this spec
  works against the dump exactly as currently cached; if/when to refresh it is a separate,
  later decision.
- Secondary/historical accession matching — primary-accession matching only; revisit only
  if it turns out to miss a suspicious fraction of `train_terms.tsv` IDs.

## Data pipeline (`build_temporal_pk_eval.py`)

**Matching.** `train_terms.tsv`'s `EntryID` against the first accession on each Swiss-Prot
record's `AC` line (primary accession only).

**Evidence filtering (t1 only — t0 has no evidence column, already pre-filtered to
experimental by Kaggle, same convention as CAFA-5's release).** Parse each record's
`DR   GO; GO:xxxxxxx; <aspect>:<name>; <EVIDENCE>:<source>.` line; keep only the standard
experimental-evidence group: `EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP`.

**Propagation.** Both t0 and t1 per-protein term sets are true-path-propagated to their
full ancestor closure via `generate_dataset.py`'s existing `propagate_protein_terms()`
before diffing — matches CAFA's own assessor behavior, avoids crediting a generic
ancestor term as "novel" when a more specific descendant was already known at t0.

**Per-aspect categorization**, for every protein present in both files, for each aspect
S ∈ {C, F, P}:
- `known_S` = propagated t0 terms in S; `novel_S` = propagated t1 terms in S minus
  `known_S`.
- `novel_S` empty → no example for this (protein, aspect); skip.
- `known_S` non-empty → **PK(S)**.
- `known_S` empty but protein has t0 terms in some other aspect → **LK(S)**.
- No t0 terms in any aspect → **NK(S)**.

**Unseen-protein pool.** Swiss-Prot accessions absent from `train_terms.tsv` entirely
(therefore also absent from every existing train/val/test split — no extra leakage check
needed), filtered to ones with ≥1 current experimental annotation: `known` = empty across
all aspects, `novel` = full current propagated experimental closure. A pure NK example
across all three aspects at once.

**Composition.** Two pools: `added_annotation` (existing CAFA-6 proteins with ≥1 novel
term in some aspect) and `unseen_protein`. Run the diff, observe the real yields, then
balance 50/50 by subsampling the larger pool down to the smaller one (seed=42, per repo
convention). No inflating a thin pool to hit a pre-set number.

## Output artifacts

**Output 1 — HF Dataset (primary),** `experiments/pk_eval/temporal_pk_validation/`. One
row per protein:
- `protein_id`, `sequence`
- `known_labels`: multi-hot vector over the **full** GO term space — reusing
  `data/cafa-6-protein-function-prediction/dataset_full`'s existing
  `features["labels"].feature.names` vocabulary (27,191 terms, confirmed via direct
  inspection), not a newly invented one, so this validation set's label indices line up
  with any full-label-space model's output. Zero vector for `unseen_protein` rows.
- `novel_labels`: multi-hot vector over the same vocabulary — the scoring ground truth.
- `category_C`/`category_F`/`category_P` ∈ {PK, LK, NK, none}
- `pool` ∈ {added_annotation, unseen_protein}

**Output 2 — flat CAFA-assessor files (secondary, cross-check only),**
`experiments/pk_eval/temporal_pk_validation_flat/`: `ground_truth.tsv` (novel terms, same
3-column shape as `train_terms.tsv` itself) + `known_terms.tsv` (t0 subset, non-empty only
for `added_annotation` rows) — in `cafaeval`/`protea`'s expected input format.

**Sidecar:** `experiments/pk_eval/category_summary.json` — per-aspect×category protein
counts, for sanity-checking the real yields before scoring.

## Scoring harness (`score_temporal_pk_eval.py`)

**Metric:** CAFA's own IA-weighted Fmax, reusing `IA.tsv`'s per-term weights directly. For
each threshold τ swept 0.01–1.00:
- per protein, `predicted(τ)` = model's full-term-space output terms with score ≥ τ, with
  known terms stripped out first (mirrors the official `-known`/`--exclude` behavior); `true`
  = `novel_labels`.
- weighted precision `wpr(τ)` = Σ IA(term) over `predicted ∩ true` / Σ IA(term) over
  `predicted`; weighted recall `wrc(τ)` likewise over `true`. Precision averaged only over
  proteins with ≥1 prediction at τ; recall averaged over all proteins with nonempty `true`
  (standard CAFA convention, avoids divide-by-zero).
- `wF(τ) = 2·wpr·wrc / (wpr+wrc)`; **Fmax = max over τ**.

**Breakdown:** computed separately per aspect (C/F/P) **and** per category within aspect
(PK/LK/NK/unseen) — the actual point of this harness.

**Checkpoint loading:** same `--model-name`/`--revision` conventions `training_run.py`
already uses (including the LoRA-adapter-vs-full-model branch where applicable) — no new
loading convention introduced.

**Cross-check:** convert the model's raw per-protein term/score output into
`cafaeval`/`protea`'s expected prediction-file format (protein-ID header line + term/score
rows, the standard CAFA submission shape), run it once against Output 2's flat files,
confirm it matches the hand-rolled Fmax within tolerance.

**No label-space mismatch to worry about:** current checkpoints (trial-013 onward, the
basis for all active ensembling/HPO work) are trained on `dataset_full`/`kelmoir/cafa6-full`
— the full 27,191-term label space (`n_labels_evaluable=9401`) — the same vocabulary this
validation set's `known_labels`/`novel_labels` vectors already use. The 383-term coarse
space only applied to the superseded `ancestor_sweep` trials (001-049) and isn't relevant
to any checkpoint this harness would actually score.

## Files

- **New:** `experiments/pk_eval/scripts/build_temporal_pk_eval.py`
- **New:** `experiments/pk_eval/scripts/score_temporal_pk_eval.py`
- **New data:** `experiments/pk_eval/temporal_pk_validation/` (HF Dataset),
  `experiments/pk_eval/temporal_pk_validation_flat/` (two TSVs),
  `experiments/pk_eval/category_summary.json`
- **Modified:** `experiments/cafa6_next_steps_plan.md` — log the result under Tier 1 item 2
  once run, same pattern as other roadmap entries.

## Testing / verification

No test framework is configured in this repo. Following the established pattern
(`blast_knn_probe_real.py`, `generate_dataset_full.py`'s `select_trainable_terms`):
standalone in-script assertions in `build_temporal_pk_eval.py` — assert per-protein
category counts sum consistently across the three aspects, assert the 50/50 balance is
achieved post-subsampling, assert the label vocabulary length matches `dataset_full`'s
27,191. The `cafaeval` cross-check itself is the main one-time verification of the Fmax
math in `score_temporal_pk_eval.py`.

## Risks & mitigations

- **Swiss-Prot cache staleness over time:** this validation set is pinned to the
  2026-08-01 cache; refreshing it later is explicitly a separate, future decision (not
  this spec's concern) — but means this validation set has a known best-by freshness and
  isn't a permanently-current benchmark.
- **Primary-accession-only matching may miss renamed/merged entries:** low-volume edge
  case; revisit only if the match rate against `train_terms.tsv`'s ~142k IDs looks
  suspiciously low.
- **Streaming the 700MB `uniprot_sprot.dat.gz`:** one-time cost, reuses the exact
  `Bio.SwissProt.parse()` streaming pattern already proven in `experiments/mf_span`
  (never fully decompressed into memory at once).

## Non-goals / explicit follow-ups

Part (B) — the `training_run.py` input-channel change and the full-training-set diffing
job it needs — is a deliberate, explicit follow-up once (A)'s numbers are in hand, not
part of this build. TrEMBL inclusion and Swiss-Prot cache refresh are both explicitly
out of scope per the discussion that shaped this spec.
