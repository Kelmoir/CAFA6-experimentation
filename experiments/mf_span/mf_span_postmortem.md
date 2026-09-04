# MF Span-Localization: Postmortem and Status

**Status: SHELVED (2026-09-03).** Not deleted, not "paused pending a quick fix" — shelved on the
strength of a real result that says the approach as built doesn't do the job it was meant to do.
This document is the full account of why, for whoever (human or Claude) picks this back up later.
See [`mf_span_sweep_log.md`](mf_span_sweep_log.md) for the trial-by-trial record this summarizes.

## What this was

A side-experiment (never part of the main CAFA6 pipeline — see `CLAUDE.md`'s "Data layout"
section) to localize GO `molecular_function` (MF) terms to specific residue spans within a
protein sequence: given a protein and one of its GO MF terms, predict *which residues* are the
evidence for that term. Built `2026-08-20` as `experiments/mf_span/scripts/train_mf_span_model.py` — a
FiLM-conditioned ESM-2 model (frozen backbone + unfrozen last-N layers, same convention as the
main pipeline's classifier) taking a semantic text embedding of the GO term (name + definition,
via `sentence-transformers/all-MiniLM-L6-v2`) as a conditioning signal, with a per-token binary
span head. The design rationale (multi-span support, semantic rather than lookup-table term
conditioning, sampled negatives) is documented in that script's own module docstring and was
sound reasoning at the time — none of it is what went wrong.

## The sweep looked genuinely successful

`2026-08-23` through `2026-09-02`, 22 trials of one-axis-at-a-time coordinate descent (same
methodology as the main pipeline's `sweep_log.md`), tracked in `mf_span_sweep_log.md`. At least
**~83 GPU-hours** directly measurable from surviving per-epoch logs (trials 15-22 alone; trials
1-14 predate per-epoch log-file capture and aren't fully reconstructable, so the true total is
higher). The sweep's own metric — token-level precision/recall/F1 — climbed steadily and
believably:

| Milestone | Trial | Metric |
|---|---|---|
| Baseline | 1 | F1 0.6295 |
| More data (biggest single win) | 20 | F1 0.8080 |
| Epoch-budget extension (30→50 epochs) | 20-ext | F1 0.8203 |
| Bigger model (150M, rejected on cost) | 21 | F1 0.8111 |
| Capacity retest (unfreeze=4, post-more-data) | 22 | F1 0.8110 |

Every reported jump had a real, isolated cause (more training data, label smoothing, epoch
budget, `unfreeze_last_n`), each verified one axis at a time, each cross-checked against
`val_loss` divergence, wall-clock cost, and — correctly — a bigger-model trade that was rejected
on cost/benefit grounds (trial 21). The methodology itself wasn't sloppy. The metric it was
climbing was the problem.

## What the real test revealed

`evaluate_cafa_score.py` (built this session, see its own module docstring) scores a model the
way the actual CAFA competition does: information-accretion-weighted precision/recall, swept
across thresholds, reduced to Fmax, against an *open* per-protein candidate set — no hints about
which terms might be right. Run against trial 20's checkpoint, `--go-scope mf-span` (the 66-term
head-to-head slice), full test split:

```
n=6764 proteins, 66 candidate terms
Fmax=0.1779 @ threshold=0.83, precision=0.1510, recall=0.2167
17.8% of the perfect-predictor ceiling
```

**0.18, not 0.82.** Not a rounding difference — the two numbers are measuring different tasks.

## Root cause

`train_mf_span_model.py`'s validation metric (`token_level_metrics`) is computed over a narrow,
curated set of `(protein, go_id)` *pairs*: each protein's actual true term(s), plus only
**`n_negatives_per_protein=2`** sampled negatives (drawn from whatever terms happen to occur
elsewhere in that split). Within those pairs, the metric itself is per-*residue* (did it flag the
right tokens), not per-protein presence/absence. The model is never asked, during training or
validation, "is this protein's function among these N candidates" — it's handed a pair already
known to be relevant (or a couple of arbitrary negatives) and asked to find the right residues.

The real task — and what `evaluate_cafa_score.py` actually scores — is the opposite: for every
protein, decide presence/absence across the *entire* candidate space (66 terms in the
head-to-head scope, ~4,579 in the model's native vocabulary) with no hints at all. Two negatives
per protein is a vanishing fraction of either number. The model has essentially no training
signal telling it to stay silent on the ~60+ terms per protein it was never paired against, so
scored against the full candidate set it fires broadly and semi-confidently — precision 0.151 at
the *best* threshold is exactly what that failure mode looks like. Recall (0.217) is also weak,
so it isn't merely "over-firing but still finding everything" either.

Two compounding factors, smaller than the above but real:

1. **Nothing in the sweep's decision loop ever touched an open-candidate metric.** All 22 trial
   verdicts, every checkpoint-selection decision, every early-stop trigger in
   `mf_span_sweep_log.md` were made against the curated-pair metric. There was no mechanism by
   which the sweep could have noticed this problem on its own — it was optimizing the only signal
   it had, and that signal doesn't reflect the deployment task.
2. **The per-protein reduction (`evaluate_cafa_score.py`'s max-over-residues aggregation) was
   never validated during training either.** It's a reasonable choice for scoring, but it's an
   additional untested assumption layered on top of the calibration gap above, not something the
   training loop was ever shaped around.

## What's still possibly salvageable

The failure is specifically in **term retrieval/presence calibration**, not necessarily in
**span-finding given a known-relevant term** — those are different capabilities, and only the
first was actually stress-tested here. It's plausible (untested) that, hand a trial-20 checkpoint
a term it's told is genuinely present, it still finds sensible residues for it; that's the exact
task its training data shape matches. If so, the architecture could still have a use as a
*downstream* annotator layered on the main classifier's own predictions (classifier: "this
protein probably has GO:X" → mf-span: "here's roughly where") rather than as a standalone
protein-function predictor. The span-representation design choices themselves (per-token binary
head over a single-span QA head, multi-span support, semantic term conditioning over a lookup
table) were empirically motivated in the original design conversation and aren't invalidated by
this finding.

## Route to redemption, if ever revisited

1. **Fix the negative-sampling starvation directly.** Either scale `n_negatives_per_protein` up
   by orders of magnitude (proportional to the real candidate space, not a fixed small constant),
   or move to a genuine hard-negative-mining / listwise-ranking objective per protein-epoch,
   instead of a fixed tiny pair list.
2. **Make an open-candidate metric the actual checkpoint-selection signal**, not an afterthought
   run once at the end. Fold something `evaluate_cafa_score.py`-shaped into
   `train_mf_span_model.py` itself as a `--run-diagnostic`-equivalent (the main pipeline added
   exactly this after its own false-signal incident — see `CLAUDE.md`'s note on `run_diagnostic()`
   and trial-050's AUROC check), so every future trial's verdict is made against the metric that
   actually matters, not a proxy that can drift arbitrarily far from it.
3. **Consider decoupling the two sub-problems explicitly** — a presence/retrieval component and a
   localization component, each trained and validated against a metric that actually matches its
   own job, rather than one token-level metric standing in for both.
4. **If retried, start a fresh sweep from scratch at cheap scale** (this repo's own
   plateau-first-at-cheap-scale convention) rather than resuming from any trial 1-22 checkpoint —
   none of them were selected against a metric that reflects the real task, so none are a
   trustworthy starting point.

## Bottom line

The regularization-sweep *engineering* was careful and the conclusions it drew about its own
metric were correct at every step. The metric itself just didn't measure the thing that mattered,
and nothing in the loop could have caught that without an independent, harder test — which is
exactly what building `evaluate_cafa_score.py` and pointing it at a real checkpoint did. That's a
successful use of the tool it was built for, even though the news it delivered was bad. Shelving
now, rather than continuing to tune a metric that's been shown not to predict the real task, is
the correct call.
