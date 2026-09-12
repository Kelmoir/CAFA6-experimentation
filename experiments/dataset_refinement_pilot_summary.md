# Dataset-refinement pilot — summary & findings (closed out 2026-09-13)

**Status: closed out.** The question this pilot was built to answer has been answered (see
"Bottom line" below) — not by finding a usable signal, but by establishing that this
instrumentation *can't* produce one. Follow-up work moves to a positive-unlabeled (PU) learning
approach instead; see `pu_learning_plan.md`. Raw per-run logs (40 swept configs, ~700 individual
training runs, ~217 log/chart files) have been deleted after this summary was written — this
document plus the one histogram below (`dataset_refinement_pilot_summary_histogram.png`) are the
retained documentation of record.

## The question

`training_run.py` treats every non-asserted `(protein, GO-term)` cell as a hard negative. Per
`experiments/ancestor_sweep/sweep_log.md`'s trial-080/081 finding, that's GO's own Open World
Assumption (OWA) problem: absence of annotation isn't confirmed absence of function. Masking
those cells out of the loss entirely made things measurably worse (AUROC -7%) — it starves an
already-scarce negative signal. `dataset_refinement_pilot.py` was built to test a gentler
alternative directly: if a currently-"unknown" cell gets soft-labeled (`--unknown-target-value`,
not hard 0) and then a small batch of cells gets "resolved" one way or another, does that
resolution leave a measurable trace in validation loss / AUROC — and could that trace be turned
into a practical tool for identifying which real "unknown" cells are actually mislabeled?

Three conditions, repeated many times each: **base** (no overrides), **faithful** (100% of
overridden cells forced to 0 — what the model already assumes), **false_negative** (all but one
cell forced to 0, the remaining one a real, already-confirmed positive forced to 0 anyway —
simulating a curation mistake).

## What was run

Three phases, roughly 700 individual training runs total:

1. **Fresh-model sweep** (2026-09-04 – 09-05, 24 configs: `speed-*`, `*-raw-*`) — smallest ESM2
   backbone, varying `--max-train-samples` (200-1000), `--items-per-trial` (10-1000),
   `--unknown-target-value` (0.0002-0.2), always training from the raw pretrained HF checkpoint
   (no prior exposure to any CAFA-6 label).
2. **Pretrained-checkpoint sweep** (2026-09-07 – 09-10, `pretrained1`-`pretrained5`) — same
   design, but starting from a locally fine-tuned `training_run.py` checkpoint
   (`--init-from-checkpoint`) instead of the raw backbone, at 200/128-sample scale.
   **Caveat found afterward**: since both the checkpoint's own pretraining and the pilot use the
   repo's fixed seed=42 and (by default) the same dataset, the pilot's fixed training subset is
   very likely a *subset* of what the checkpoint was already fine-tuned on — meaning these 5
   configs are not a clean "fresh model" test. A model that already learned the correct answer
   during pretraining is a different (probably harder-to-perturb) question than a naive model
   seeing the label space for the first time.
3. **Bias-corrected sweep** (2026-09-12, `bias-*`, 11 configs) — after adding a classifier
   bias-prior init (log-odds of each label's positive rate, mirroring `training_run.py`'s own
   fresh-start behavior) to `dataset_refinement_pilot.py`'s `build_base_model`, which was
   previously missing. Without it, BCE under this task's severe class imbalance (see below) wastes
   a 1-3 epoch pilot run's whole gradient budget just walking the bias down from a random init,
   on every run including base — a real confound against detecting anything. Fresh HF backbone
   throughout this phase (no checkpoint-contamination issue).

## Bottom line

**At the population level, no config in any of the three phases showed a detectable effect.**
Pooled validation loss and mean/global AUROC differences between `faithful`/`false_negative` and
`base` were consistently 3-4 orders of magnitude below the metrics' own natural scale (e.g.
val_loss diffs ~1e-6 against a ~0.03 baseline; AUROC diffs ~1e-4 against a ~0.55-0.85 baseline,
depending on config) — indistinguishable from the base condition's own zero-noise floor (base
repeats are bit-identical by construction: fixed seed, fixed init, fixed data order, no
overrides). This held true regardless of model scale, checkpoint pretraining, `--unknown-target-value`,
or the bias-prior fix.

**At the individual-repeat level, a real, reproducible minority effect exists — but only in
`false_negative`, and it doesn't get you a usable detector.** The most rigorous single
measurement (`bias-250-256-0.001`, bias-corrected, fresh model, n=1100 repeats per condition
after resuming): `faithful` repeats cluster in an extremely tight band (val_loss diff vs base:
0 to +4e-7). `false_negative` repeats spread much wider (-1.8e-6 to +6.1e-6), and **~18-20% of
them fall entirely outside `faithful`'s observed range** — a real, detectable tail. The same
right-skewed tail pattern reproduced independently against two different locally-pretrained
checkpoints (v3 and v6, phase 2) at a comparable scale, and did *not* appear at a smaller
27-eligible-term scale (`pretrained5`) — suggesting the effect is scale/leverage-dependent, not
universal.

The catch: **~80% of `false_negative` repeats are statistically indistinguishable from a
`faithful` repeat.** There is no threshold on pooled val_loss (or pooled AUROC) that separates
"this run had a real annotation error in it" from "this run didn't" without either missing most
real errors (conservative threshold) or occasionally flagging a genuinely-faithful run (loose
threshold). This isn't a tuning problem — it's structural: pooled per-run loss aggregates the
effect of *every* overridden cell in that run (up to 250 of them in the `bias-*` phase) into one
number, so it can only ever say "this whole run's mix of overrides mattered a bit," never "this
one specific cell is why."

A secondary, more fixable issue compounded phase 3: `--items-per-trial` was swept at 39-98% of
`--max-train-samples` (vs. the original plan's ~2.5%, 100-of-4000) — at that density, almost every
training example has *some* cell overridden, leaving too little clean signal for the model to
learn the eligible terms well in 1-3 epochs at all (AUROC ~0.52-0.56, barely above chance) — on
top of, not instead of, the structural problem above.

## Why this closes out the aggregate-loss approach

The raw/unpropagated CAFA-6 label space (`kelmoir/cafa6-raw`, what this pilot trained against)
has a **0.556% average positive rate** (42,369 proteins × 668 terms) — even sparser than the
ancestor-propagated coarse space's 3.58% (`training_run.py`'s own docs). Under that sparsity,
a per-cell signal diluted across a whole training run's pooled loss was never going to surface
reliably. Getting a number worth thresholding requires a signal computed *per candidate cell*,
not per training run — which is the premise `pu_learning_plan.md` starts from.

## Example evidence

`dataset_refinement_pilot_summary_histogram.png` (kept from `bias-250-256-0.001`, the largest-n,
bias-corrected, fresh-model config) shows exactly the pattern described above: `faithful`'s
per-repeat loss-diff-from-base packed into a needle-thin spike near zero, `false_negative`'s
spread much wider with a one-sided tail toward worse outcomes and no comparable better-tail —
real, but with the bulk of its mass overlapping `faithful` completely.
