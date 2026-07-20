# CAFA6 Regularization Sweep Log

Goal: find ESM-2 fine-tuning settings (dropout, gradient clipping, unfrozen-layer count, model size)
that train reliably on the CAFA6 GO-aspect classification task, without either dying (no learning
signal) or overfitting immediately. Each trial runs on a Colab T4 for ~1h via `training_run.py`
(see `colab run` invocations in each trial section below). Full background/architecture: see the
plan this was built from and `CLAUDE.md`.

## Rubric

Judge from the *shape* of the per-epoch curve (all epochs together), not a single endpoint.

- **DEAD** — train loss stays flat near chance level (~0.693 for BCE), train/val recall and accuracy
  don't move meaningfully across epochs. No train/val gap because nothing is being learned at all.
  → Raise `--learning-rate` (×3-5) and/or `--unfreeze-last-n`, or reduce dropout/weight-decay if
  either is already aggressive. Resolve DEAD before judging over/underfitting — a config can't be
  judged overfit or healthy until it's confirmed to learn at all. Exception to one-axis-at-a-time:
  DEAD may justify changing LR and unfreeze-count together in the same trial, since it has multiple
  plausible simultaneous causes — call this out explicitly in the trial's "next config change" note.
- **OVERFITTING** — train loss drops (often sharply) while val loss flattens or rises; train/val
  accuracy-recall gap widens epoch over epoch. → Raise `--output-dropout`/`--esm-dropout`/
  `--attention-dropout`, raise `--weight-decay`, reduce `--unfreeze-last-n`, or lower
  `--learning-rate` — one axis per trial, priority to whichever hasn't been tried yet this sweep.
- **HEALTHY** — train and val loss decrease together (val may lag slightly but tracks train, doesn't
  diverge), accuracy/recall rise together on both splits, gap stays roughly constant, no NaN/spikes.
  → Candidate config. If mid-sweep, push one axis further (more epochs, more unfrozen layers, or
  scale up model size) to see how far it holds. If already at target scale, promote: run one
  confirmation trial at full model/dataset size (still ~1h-capped), then if still HEALTHY run the
  uncapped final training run and `--push-to-hub` to `kelmoir/cafa6-esm@main`.
- **UNSTABLE** — loss spikes, NaN/inf, or wild oscillation between epochs. → Lower `--learning-rate`,
  tighten `--grad-clip-norm` (e.g. 1.0 → 0.5), or check `--batch-size` isn't too small for stable
  gradient estimates.

## Search strategy

Coordinate descent from today's hardcoded defaults (`esm2_t12_35M`, `unfreeze=2`, `lr=5e-5`,
`output_dropout=0.2`, `esm/attn dropout=0.0`, `grad_clip=1.0`, `weight_decay=0.01`) — trial-001
itself starts smaller/cheaper (`esm2_t6_8M`) purely to iterate fast. Change exactly one axis per
trial (DEAD exception above), priority order: (1) LR + `--unfreeze-last-n` — decide whether the
model learns at all; (2) dropout/weight-decay/grad-clip — once training moves, resolve overfitting;
(3) model size — scaled up only once a small model is HEALTHY; (4) dataset size/epoch count — used
purely to hit the ~1h wall-clock budget, not a regularization lever.

## Trial index

| id | model | unfreeze | lr | out_drop | esm_drop | attn_drop | grad_clip | wd | epochs | train_n | val_n | wall_clock | verdict | next_action |
|----|-------|----------|----|----|----|----|----|----|----|----|----|----|----|----|
| _(none yet)_ | | | | | | | | | | | | | | |
