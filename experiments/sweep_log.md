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
| trial-001 | esm2_t6_8M | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 2 | 20000 | 2000 | ~18min (2x~450s+overhead) | VOID (data bug) | fix label bug, rerun as trial-002 |

## trial-001

**Config**: `esm2_t6_8M`, unfreeze=2, lr=5e-5, out_drop=0.2, esm_drop=0.0, attn_drop=0.0,
grad_clip=1.0, wd=0.01, epochs=2, train_n=20000, val_n=2000, seed=42, no `--max-minutes` cap.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.0273 | 0.0004 | 0.9982 | 1.0000 | 0.0000 | 0.0000 |
| 2 | 0.0002 | 0.0001 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

**Verdict: VOID — not a regularization result at all, a data-pipeline bug.**

Loss collapsing to ~0 with "accuracy" at 1.0 while recall stays *exactly* 0.0000 on both splits
across both epochs is the signature of the model learning to predict all-negative against an
all-negative target — not real convergence. Traced to `preprocess_batch` in `training_run.py`:
`generate_dataset.py` stores the `labels` column as `datasets.Sequence(datasets.ClassLabel(...))`,
which means `dataset["train"]["labels"]` yields already-encoded **integer** class ids (e.g. `[0, 2]`)
at load time — not the string tokens (`"C"`, `"F"`, `"P"`) the `ClassLabel` was originally built from.
`preprocess_batch` did `label_terms = [str(item) for item in labels_value]` (stringifying the ints to
`"0"`, `"2"`) then checked `if item in label2id`, where `label2id` is keyed on the aspect letters —
never true. Every example's multi-hot label vector has been silently all-zero. This plausibly explains
the user's original complaint (months of manual attempts reading as either overfitting or dead
training regardless of dropout/clip settings) — the model was likely never receiving real supervision,
independent of regularization.

**Fix applied** (`training_run.py`, `preprocess_batch`): when `labels_value` items are not strings,
treat them as already-valid class indices (`labels[int(item)] = 1.0`) instead of re-stringifying and
matching against `label2id`. Verified `unique_items`/`UNIQUE_ASPECTS = ["C", "F", "P"]` ordering matches
`generate_dataset.py`'s `ClassLabel(names=sorted(label2id.keys()))` (already alphabetical, so indices
align). Compiles clean (`py_compile`).

**Next action**: rerun the identical config as trial-002 with the fix in place — this is the real
first data point for the sweep. trial-001's numbers are not meaningful for the DEAD/OVERFITTING/
HEALTHY/UNSTABLE rubric and must not be compared against future trials.
