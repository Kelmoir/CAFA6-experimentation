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

## Operational constraint: colab exec connection ceiling

Discovered across trial-002/003/004: a `colab exec` invocation reliably loses its websocket
connection (`RuntimeError: Connection was lost`, raised by `jupyter_kernel_client`, fatal — colab-cli
never passes a nonzero `reconnect_interval`) somewhere around **~9-10 minutes of total wall clock**
from `colab exec` start, regardless of whether the kernel is producing periodic output (trial-004's
first attempt died ~2-3min in with zero output produced yet; adding a `[STEP]` print roughly every
20s pushed the failure to 96% through the same epoch, i.e. much later wall-clock-wise, but it still
died before completing — ruling out pure output-silence as the cause and pointing to a roughly fixed
connection lifetime for headless, non-UI-driven kernel connections). Until/unless this is fixed
upstream in colab-cli, every trial config must keep **each single `colab exec` invocation's training
volume small enough to finish within that ceiling** — in practice, `--max-train-samples` around
10000-12000 (≈1250-1500 steps at batch-size 8, ~4-5min of training loop time, leaving comfortable
margin under the ~9min ceiling once data/model download overhead is included) rather than the
20000 used in trial-001 through trial-004's first attempt. Multi-epoch trials are stitched together
across separate `colab exec` calls using `--resume-from-revision`/`--start-epoch` (added for
trial-003) — one epoch (or fewer, if needed) per invocation. This changes trial-to-trial sample-size
comparability going forward; note the actual `--max-train-samples` used in each trial's config rather
than assuming 20000.

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
| trial-002 | esm2_t6_8M | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 2 (only 1 completed) | 20000 | 2000 | ~7.3min (438.7s epoch 1) then dropped | INCOMPLETE (connection lost) | identical config, rerun as trial-003 |
| trial-003 | esm2_t6_8M | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 2 (split across 2 sessions via resume) | 20000 | 2000 | ~7.4min + ~7.4min (2 sessions) | DEAD (weak signal) | raise lr to 2e-4 and unfreeze-last-n to 4 for trial-004 |
| trial-004 | esm2_t6_8M | 4 | 2e-4 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 2 (split across 2 sessions via resume) | 12000 | 2000 | ~7.0min + ~6.5min (2 sessions) | DEAD (still, prior-prediction basin) | raise lr to 1e-3 for trial-005; local diagnostic ruled out a pipeline bug |
| trial-005 | esm2_t6_8M | 4 | 1e-3 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 1 | 12000 | 2000 | ~6.4min | WORSE (full collapse, val_recall=1.0) | root-caused to BCE bias-saturation trap; add classifier bias-to-prior init for trial-006 |

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

## trial-002

**Config**: identical to trial-001 (`esm2_t6_8M`, unfreeze=2, lr=5e-5, out_drop=0.2, esm_drop=0.0,
attn_drop=0.0, grad_clip=1.0, wd=0.01, epochs=2, train_n=20000, val_n=2000, seed=42), run via
`colab --auth=adc new/upload/exec` directly on host (no podman) — first real trial after switching
off the buggy `oauth2` auth mode to `adc`, which resolved the recurring `invalid_scope` refresh
failure.

**Per-epoch metrics** (only epoch 1 completed):

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6109 | 0.6018 | 0.6880 | 0.6885 | 0.9826 | 0.9704 |

**Verdict: INCOMPLETE — infra failure (connection lost), not a rubric verdict.** The `colab exec`
websocket connection dropped (`RuntimeError: Connection was lost`) partway through epoch 2, after the
epoch-1 checkpoint had already been pushed successfully to `kelmoir/cafa6-esm@trial-002`. This is a
transient Colab/tunnel reliability issue, unrelated to the auth fix (session provisioning, upload, and
epoch 1 all completed normally under `--auth=adc`) and unrelated to training itself.

Epoch 1's numbers are still meaningful as a sanity check even though the trial is incomplete: loss
(0.6018 val) is already below chance (~0.693) and recall is high and non-zero (0.9704 val) — confirming
the label-encoding fix from trial-001 is working (real supervision signal is reaching the model). Not
enough data yet to judge DEAD/OVERFITTING/HEALTHY/UNSTABLE — one epoch's high-recall/moderate-accuracy
shape is consistent with an early "predict-mostly-positive" phase that a second epoch would clarify
(recall creeping toward precision as it discriminates further, or accuracy failing to rise = overfit
risk on the harder classes). Do not compare this epoch-1 data point against future trials' full 2-epoch
curves.

**Next action**: rerun the identical config as trial-003 (pure infra retry, no config change) to get a
complete 2-epoch curve for the first real rubric judgment.

## trial-003

**Config**: identical to trial-001/002 (`esm2_t6_8M`, unfreeze=2, lr=5e-5, out_drop=0.2, esm_drop=0.0,
attn_drop=0.0, grad_clip=1.0, wd=0.01, epochs=2, train_n=20000, val_n=2000, seed=42). Epoch 1 hit the
same `RuntimeError: Connection was lost` as trial-002, at the identical point (right after the epoch-1
checkpoint push) with identical epoch-1 metrics (fully deterministic given the fixed seed) — confirming
this is a reproducible colab-cli/Colab-tunnel limitation, not random flakiness (see `training_run.py`'s
new `--resume-from-revision`/`--start-epoch` flags, added specifically to work around it: colab-cli's
`exec` never passes a nonzero `reconnect_interval` to its underlying websocket client, so any drop is
fatal to the whole run — likely an idle/session timeout Colab's tunnel enforces on headless,
non-UI-driven kernel connections around the ~9-10min wall-clock mark). Epoch 2 was completed on a fresh
Colab session, resuming model weights from the `trial-003` checkpoint branch pushed after epoch 1
(`--resume-from-revision trial-003 --start-epoch 1 --epochs 1`) — optimizer/LR-schedule state was not
preserved across the resume (fresh AdamW + warmup within the second invocation), a minor documented
deviation from a true single-process run.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6109 | 0.6018 | 0.6880 | 0.6885 | 0.9826 | 0.9704 |
| 2 | 0.5932 | 0.6009 | 0.6978 | 0.6950 | 0.9687 | 0.9815 |

**Verdict: DEAD (weak/near-stagnant learning signal).** Val loss is essentially flat (0.6018 → 0.6009,
a 0.0009 move) across the full 2 epochs; train loss moves a little more (0.6109 → 0.5932) but train/val
track closely with no meaningful gap opening (not overfitting). Both losses sit modestly below chance
(~0.693) but recall is pinned near-ceiling (~0.97-0.98) on both splits both epochs while accuracy barely
moves (~0.69 → ~0.70) — consistent with the model mostly predicting positive across the board rather
than learning to discriminate, and not meaningfully progressing epoch over epoch. Per the rubric this is
closer to DEAD than a slow-but-real HEALTHY start: nothing here suggests two more epochs at this
config would meaningfully change the curve shape.

**Next action** (DEAD → raise LR and/or unfreeze-last-n; multi-axis change permitted for DEAD): trial-004
raises `--learning-rate` from 5e-5 to 2e-4 (×4) and `--unfreeze-last-n` from 2 to 4 (out of 6 layers on
`esm2_t6_8M`), keeping all other hyperparameters and sample sizes unchanged, to establish whether the
model can learn at all at this scale before returning to one-axis-at-a-time regularization tuning.

## trial-004

**Config**: `esm2_t6_8M`, unfreeze=4, lr=2e-4, out_drop=0.2, esm_drop=0.0, attn_drop=0.0, grad_clip=1.0,
wd=0.01, epochs=2, train_n=12000 (reduced from 20000 — see the operational-constraint note above),
val_n=2000, seed=42. Ran as two `colab exec` invocations (fresh session per epoch, resume via
`--resume-from-revision`), after the first attempt at train_n=20000 hit the connection-loss ceiling
mid-epoch-1 (documented above) even with periodic `[STEP]` keepalive logging added.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6180 | 0.6066 | 0.6849 | 0.6933 | 0.9844 | 0.9893 |
| 2 | 0.6028 | 0.6073 | 0.6913 | 0.6910 | 0.9665 | 0.9873 |

**Verdict: DEAD (still).** Despite 4x the LR and 2x the unfrozen layers vs. trial-003, the curve lands
in almost the same place: val_loss flat-to-slightly-up (0.6066 → 0.6073), recall pinned near-ceiling
(~0.97-0.99) both epochs, accuracy barely moving (~0.69). Two materially different configs converging
to the same shape means this isn't primarily an LR/capacity problem — see the diagnostic below.

**Diagnostic (before spending a 3rd trial guessing blind)**: the label distribution is `C=72%, F=67%,
P=67%` positive (checked directly against `kelmoir/cafa6`, 20000-example sample) — moderately imbalanced
toward positive. The theoretical BCE loss of a trivial "always predict the population base rate, ignore
the input" classifier for a ~69% average positive rate is `-[0.69·ln(0.69) + 0.31·ln(0.31)] ≈ 0.619` —
strikingly close to the ~0.60-0.62 losses seen in both trial-003 and trial-004. This is the signature of
the model converging to (and getting stuck at) the "predict the prior" local minimum rather than using
sequence information. Ran a local CPU sanity check (`create_esm`/`preprocess_batch` reused directly,
not a Colab trial) to rule out a real pipeline bug before chasing this further: overfitting a fixed
16-example batch at lr=5e-4 took loss from 0.6931 → 0.0095 in 40 steps, with nonzero, healthy gradient
norms flowing into `classifier.out_proj.weight` throughout. Gradients and the freeze/unfreeze wiring are
correct — the bottleneck is that 1-2 epochs (1500-3000 steps) at lr≤2e-4 isn't enough push to escape the
prior-prediction basin on the full, diverse (not memorizable) dataset, not a code defect.

**Next action**: trial-005 raises `--learning-rate` further to 1e-3 (~5x trial-004, within the rubric's
suggested ×3-5 DEAD response), keeping `--unfreeze-last-n 4` (more capacity didn't hurt) and all other
settings unchanged. Watch the `[STEP]` loss trajectory early in epoch 1 — if it doesn't move
meaningfully below ~0.60 within the first few hundred steps this time either, the next lever is more
epochs/steps rather than a further LR increase (a still-higher LR risks UNSTABLE before it helps).

## trial-005

**Config**: `esm2_t6_8M`, unfreeze=4, lr=1e-3, out_drop=0.2, esm_drop=0.0, attn_drop=0.0, grad_clip=1.0,
wd=0.01, epochs=1, train_n=12000, val_n=2000, seed=42.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6252 | 0.6195 | 0.6853 | 0.6895 | 0.9959 | 1.0000 |

**Verdict: WORSE, not DEAD-but-improving — the LR increase made the collapse more complete.** val_recall
hit a literal 1.0000 (predicting positive on every single validation example) and val_loss rose slightly
vs. trial-004 epoch 1 (0.6195 vs 0.6066). Raising LR pushed the model to converge on the trivial
"always predict positive" solution *faster and more completely*, not to escape it — the opposite of
what the DEAD rubric's "raise LR" response assumes. This ruled out "not enough push" as the mechanism
and prompted a root-cause investigation (see below) rather than a third blind LR trial.

**Root cause**: classic BCE + imbalanced-label saturation trap. The classifier head's `out_proj.bias`
starts at zero at init. Early gradient descent has a strong, easy signal to move the bias toward
matching the label prior (C=72%, F=67%, P=67%) — this happens fast regardless of LR. Once logits
saturate near the prior, the per-example BCE gradient (`sigmoid(logit) - label`) shrinks toward zero on
average (since predictions ≈ the label's mean value), starving the weight matrix of the differential
signal it needs to actually use sequence content. A higher LR just reaches this degenerate fixed point
faster and lands closer to it, worsening rather than fixing the symptom.

**Fix**: initialize `model.classifier.out_proj.bias` to the training label prior's log-odds at model
creation (skipped on `--resume-from-revision`, since that already loads real trained weights) — a
standard technique from the imbalanced-classification literature (e.g. the RetinaNet/focal-loss paper's
bias-init trick). This lets the model start past the saturation point instead of spending its early
gradient budget getting there, hopefully freeing gradient signal for genuine discrimination from step 1.
Verified locally (not a Colab trial): pre-training loss on a 200-example sample moved from ~0.696
(zero-bias, ≈ln(2)) to ~0.656 after the fix — confirms the mechanism engages as intended.

**Next action**: trial-006 retests at a moderate `--learning-rate 2e-4` (trial-004's LR, since 1e-3 is
now known to push toward collapse) with the bias-prior fix in place, `--unfreeze-last-n 4` unchanged.
If the bias fix works, expect val_loss to drop meaningfully below the ~0.60-0.62 plateau and val_recall
to move off its ~0.97-1.0 ceiling within the first epoch.
