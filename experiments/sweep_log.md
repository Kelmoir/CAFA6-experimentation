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
| trial-006 | esm2_t6_8M | 4 | 2e-4 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 2 (via resume) | 12000 | 2000 | ~6.6min + ~6.5min | DEAD (2 epochs, still flat) | pivot to model size for trial-007 |
| trial-007 | esm2_t12_35M | 4 | 2e-4 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 1 | 6000 | 2000 | ~7.6min | DEAD (same plateau, 4x larger model) | pause sweep, consult user — 5 materially different configs all converge to the same plateau |
| trial-008 | esm2_t6_8M | 1 | 2e-4 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 4 (via resume) | 12000 | 2000 | ~3.75min/epoch x4 | OVERFITTING (epochs 1-2 flat, 3-4 diverge) | raise esm/attn dropout to 0.1 for trial-009 |
| trial-009 | esm2_t6_8M | 1 | 2e-4 | 0.2 | 0.1 | 0.1 | 1.0 | 0.01 | 4 (via resume) | 12000 | 2000 | ~3.8min/epoch x4 | OVERFITTING (dropout didn't help) | raise weight-decay to 0.1 for trial-010 |
| trial-010 | esm2_t6_8M | 1 | 2e-4 | 0.2 | 0.0 | 0.0 | 1.0 | 0.1 | 4 (via resume) | 12000 | 2000 | ~3.8min/epoch x4 | OVERFITTING but smallest gap yet (0.133 vs 0.168/0.191) | push weight-decay further to 0.3 for trial-011 |
| trial-011 | esm2_t6_8M | 1 | 2e-4 | 0.2 | 0.0 | 0.0 | 1.0 | 0.3 | 4 (via resume) | 12000 | 2000 | ~3.7min/epoch x4 | OVERFITTING, gap keeps shrinking (0.079) | push weight-decay to 0.5 for trial-012 |
| trial-012 | esm2_t6_8M | 1 | 2e-4 | 0.2 | 0.0 | 0.0 | 1.0 | 0.5 | 4 (via resume; one epoch-2 retry after a transient connection drop) | 12000 | 2000 | ~3.8min/epoch x4 | OVERFITTING, gap down to 0.046 | push weight-decay to 1.0 for trial-013 |
| trial-013 | esm2_t6_8M | 1 | 2e-4 | 0.2 | 0.0 | 0.0 | 1.0 | 1.0 | 4 (via resume) | 12000 | 2000 | ~3.7min/epoch x4 | gap ~0 but train_loss stalled (re-approaching DEAD) | extend trial-012 (wd=0.5) further instead — better tradeoff point |

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

## trial-006

**Config**: `esm2_t6_8M`, unfreeze=4, lr=2e-4, out_drop=0.2, esm_drop=0.0, attn_drop=0.0, grad_clip=1.0,
wd=0.01, train_n=12000, val_n=2000, seed=42, classifier bias initialized to label priors (`C=0.722,
F=0.674, P=0.663` on this 12000-example subset).

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6153 | 0.6070 | 0.6873 | 0.6933 | 0.9916 | 0.9877 |

**Verdict: DEAD — the bias-init fix made essentially no difference.** val_loss (0.6070) and val_recall
(0.9877) after epoch 1 are nearly identical to trial-004's epoch 1 (0.6066, 0.9893) at the same LR
without the fix. The fix demonstrably engages (confirmed both by the printed prior values and the
earlier local check showing it shifts pre-training loss), so this isn't a "the fix didn't run" issue —
bias saturation is not the (sole) real bottleneck. Combined with trial-005 (higher LR made things
worse), both of the DEAD rubric's first-line responses (raise LR, fix bias saturation) have now been
tried without success. What hasn't been tested: patience. Every trial so far has been capped at 1-2
epochs (forced by the connection-ceiling operational constraint). It's possible the weight matrix
genuinely needs more gradient steps to develop discriminative signal beyond the prior, independent of
where the bias starts or how large the LR is.

**Next action**: extend trial-006 for several more epochs via `--resume-from-revision`/`--start-epoch`
(same config, no hyperparameter change) rather than launching a new trial — this directly tests the
"needs more steps" hypothesis. Continue epoch-by-epoch and stop early if the plateau breaks or if
several more epochs still show no movement (at which point the next lever is architecture: more
unfrozen layers or a larger base model, not more patience).

## trial-007

**Config**: `esm2_t12_35M` (4x larger than trial-001-006's `esm2_t6_8M`; 12 layers vs 6), unfreeze=4,
lr=2e-4, out_drop=0.2, esm_drop=0.0, attn_drop=0.0, grad_clip=1.0, wd=0.01, epochs=1, train_n=6000
(reduced further for the larger model's slower per-step time), val_n=2000, seed=42, bias-init fix in
place.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6158 | 0.6089 | 0.6851 | 0.6902 | 0.9894 | 0.9982 |

**Verdict: DEAD — same plateau as every prior config.** val_loss (0.6089) and val_recall (0.9982) are
statistically indistinguishable from trial-003 through trial-006, despite this model having 4x the
parameters and 2x the encoder depth of every trial before it.

## Status: pausing the sweep to reassess

Five materially different configurations have now converged to the same "predict near the population
prior" plateau (val_loss ≈ 0.60-0.62, val_recall ≈ 0.97-1.0, val_acc ≈ 0.69) within one epoch, and stay
there:
- learning rate spanning 5e-5 to 1e-3 (20x range) — trial-003 vs trial-005
- unfrozen-layer count 2 vs 4 — trial-003 vs trial-004
- classifier bias initialized to zero vs. the label prior's log-odds — trial-004 vs trial-006
- 1 epoch vs 2 epochs at an identical config — trial-006 epoch 1 vs epoch 2
- model size 8M vs 35M params (6 vs 12 encoder layers) — trial-006 vs trial-007

A local CPU check (not a GPU trial) confirmed gradients flow correctly and the model can drive loss to
near-zero memorizing a small fixed batch, ruling out a dead-gradient/pipeline bug. Given how consistent
the plateau is across every lever tried so far, further blind hyperparameter trials are unlikely to be
a good use of GPU time/compute credits without a new hypothesis. Plausible remaining directions, roughly
in order of how cheap they are to test:
1. **Much longer training budget** — everything so far has been capped at 1-2 epochs (~1500-3000 steps)
   per the connection-ceiling constraint; genuine generalization on a hard, high-level task (predicting
   only 3 coarse GO aspects from raw sequence) may need many more total steps than tested so far, via
   many more chained `--resume-from-revision` epochs.
2. **Full unfreeze** — `--unfreeze-last-n` has only been tried at 2/6, 4/6, and 4/12; a fully unfrozen
   backbone hasn't been tested.
3. **Loss/architecture changes** — e.g. focal loss instead of plain BCE, or a higher-capacity
   classifier head — bigger code changes, worth trying only after (1)/(2) are exhausted.
4. **Revisit the task framing itself** — verify aspect-level (C/F/P presence) prediction isn't close to
   a genuinely hard-to-learn-from-sequence task at this scale, independent of everything tried so far.

Stopping here to consult the user given the compute already spent (7 trials, several multi-epoch) and
because the next move is a strategic choice about budget/direction rather than a confident next
hyperparameter guess.

**User direction**: run longer (5-10+ epochs, matching what it took in prior manual attempts before the
model "learned things") and keep `--unfreeze-last-n` as small as possible, for efficiency at the final
target scale.

## trial-008

**Config**: `esm2_t6_8M`, unfreeze=1 (the smallest tested so far), lr=2e-4, out_drop=0.2, esm_drop=0.0,
attn_drop=0.0, grad_clip=1.0, wd=0.01, train_n=12000, val_n=2000, seed=42, bias-init fix in place. Run
across 4 chained `colab exec` invocations via `--resume-from-revision`. Epoch time dropped to ~225s
(vs. ~390-420s at unfreeze=4) — unfreeze=1 is meaningfully cheaper, consistent with the user's
efficiency goal.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6132 | 0.6050 | 0.6874 | 0.6877 | 0.9915 | 0.9730 |
| 2 | 0.5949 | 0.6046 | 0.6947 | 0.6937 | 0.9603 | 0.9810 |
| 3 | 0.5604 | 0.6150 | 0.7185 | 0.6877 | 0.9310 | 0.9525 |
| 4 | 0.4945 | 0.6628 | 0.7621 | 0.6725 | 0.9135 | 0.9024 |

**Verdict: OVERFITTING (a qualitatively new, more informative state than DEAD)**. Epochs 1-2 replicate
the familiar flat plateau. Epoch 3 shows the first real train-side movement (loss drops, accuracy rises,
recall moves off the ceiling) with val barely moved. Epoch 4 is unambiguous: train continues improving
sharply while val gets *worse* (val_loss 0.6150 → 0.6628, val_acc 0.6877 → 0.6725) — the train/val gap
opens from ~0.05 (epoch 3) to ~0.17 (epoch 4) in a single epoch. This closely matches the project's
original standing problem (manual attempts "consistently either overfit or failed to learn at all") —
here we see the same model transition almost directly from dead-flat to overfitting within a couple of
epochs, with little sustained middle ground where both splits improve together.

**Next action** (OVERFITTING → one axis per the rubric): trial-009 raises `--esm-dropout` and
`--attention-dropout` together from 0.0 to 0.1 (treated as one axis — both apply to the same single
unfrozen encoder layer, which is presumably where the overfitting-prone adaptation is happening; they
were previously completely unregularized). `--unfreeze-last-n` stays at 1 per the user's efficiency
preference rather than reducing further. Everything else unchanged. Run through several epochs again
watching for whether the healthy (both-improving) window widens before overfitting sets in.

## trial-009

**Config**: identical to trial-008 except `--esm-dropout 0.1 --attention-dropout 0.1` (up from 0.0).
Same 4-epoch chained run via resume.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6142 | 0.6053 | 0.6866 | 0.6912 | 0.9921 | 0.9729 |
| 2 | 0.5985 | 0.6037 | 0.6925 | 0.6953 | 0.9655 | 0.9858 |
| 3 | 0.5622 | 0.6137 | 0.7152 | 0.6905 | 0.9310 | 0.9556 |
| 4 | 0.4815 | 0.6727 | 0.7716 | 0.6603 | 0.9164 | 0.8633 |

**Verdict: OVERFITTING — dropout on the unfrozen encoder layer made no meaningful difference.** Epochs
1-3 track trial-008 almost exactly (train/val losses within ~0.002 of each other at every epoch). Epoch
4's train/val gap (0.4815 vs 0.6727 = 0.191) is essentially the same as trial-008's (0.4945 vs 0.6628 =
0.168) — if anything marginally worse, not better. 0.1 dropout on a single unfrozen layer's
attention/output sublayers isn't a strong enough regularizer here (or isn't the right lever for this
overfitting mode, which may be more about the classifier head's capacity than the tiny amount of
newly-unfrozen encoder capacity).

**Next action**: trial-010 reverts esm/attn dropout to 0.0 (trial-008's baseline, to isolate the next
change cleanly) and instead raises `--weight-decay` from 0.01 to 0.1 (10x) — this axis hasn't been
touched in any trial so far, and weight decay directly penalizes the large weight magnitudes that let a
tiny classifier head memorize training-set specifics.

## trial-010

**Config**: identical to trial-008 except `--weight-decay 0.1` (up from 0.01, 10x). Same 4-epoch
chained run via resume.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6131 | 0.6048 | 0.6873 | 0.6885 | 0.9914 | 0.9732 |
| 2 | 0.5958 | 0.6042 | 0.6941 | 0.6945 | 0.9622 | 0.9820 |
| 3 | 0.5652 | 0.6115 | 0.7153 | 0.6893 | 0.9342 | 0.9555 |
| 4 | 0.5113 | 0.6441 | 0.7520 | 0.6745 | 0.9146 | 0.9154 |

**Verdict: still OVERFITTING, but the smallest gap of the three regularization attempts so far.**
Epoch-4 train/val gap is 0.133 (0.5113 vs 0.6441) — meaningfully smaller than trial-008's 0.168 and
trial-009's 0.191. val_loss at every epoch is the best of the three variants tried. Weight decay is a
real, working lever here (unlike encoder dropout, which did nothing) — it's just not strong enough yet
at 0.1 to produce a genuinely healthy (both-improving) curve; val still turns upward starting epoch 3.

**Next action**: trial-011 pushes `--weight-decay` further to 0.3 (3x trial-010) to see whether more of
the same lever closes the gap further, keeping everything else identical.

## trial-011

**Config**: identical to trial-008/010 except `--weight-decay 0.3` (3x trial-010, 30x the original
default). Same 4-epoch chained run via resume.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6133 | 0.6047 | 0.6872 | 0.6888 | 0.9921 | 0.9759 |
| 2 | 0.5982 | 0.6050 | 0.6921 | 0.6938 | 0.9660 | 0.9832 |
| 3 | 0.5772 | 0.6093 | 0.7060 | 0.6902 | 0.9426 | 0.9596 |
| 4 | 0.5447 | 0.6235 | 0.7289 | 0.6783 | 0.9209 | 0.9214 |

**Verdict: still OVERFITTING, but the trend across trial-008/010/011 is now unambiguous and monotonic.**
Epoch-4 train/val gap: 0.168 (wd=0.01) → 0.133 (wd=0.1) → 0.079 (wd=0.3). val_loss at epoch 4: 0.6628 →
0.6441 → 0.6235, each one the best yet. The shape within trial-011 itself is still classic overfitting
(gap widens epoch over epoch: 0.008 → 0.007 → 0.032 → 0.079) — weight decay is delaying and dampening
the onset, not eliminating it, but the effect is large and consistent enough to keep pushing this same
lever rather than diversify.

**Next action**: trial-012 pushes `--weight-decay` to 0.5 to test whether the monotonic improvement
continues or starts to plateau/hurt (very high weight decay can eventually suppress genuine learning
capacity too, which would show up as train_loss failing to drop at all — watch for that).

## trial-012

**Config**: identical to trial-008/010/011 except `--weight-decay 0.5` (5x trial-011, 50x the original
default). Same 4-epoch chained run via resume. Epoch 2's first attempt hit a transient
`RuntimeError: Connection was lost` failing during session-startup boilerplate (before any config/data
loading), unrelated to hyperparameters — a fresh session + immediate retry succeeded cleanly.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6136 | 0.6046 | 0.6870 | 0.6897 | 0.9928 | 0.9785 |
| 2 | 0.6005 | 0.6053 | 0.6915 | 0.6925 | 0.9722 | 0.9853 |
| 3 | 0.5867 | 0.6077 | 0.7000 | 0.6907 | 0.9534 | 0.9682 |
| 4 | 0.5675 | 0.6132 | 0.7133 | 0.6887 | 0.9339 | 0.9489 |

**Verdict: still OVERFITTING, but the monotonic trend across trial-008/010/011/012 continues cleanly.**
Epoch-4 gap: 0.168 → 0.133 → 0.079 → 0.046 (wd = 0.01/0.1/0.3/0.5). val_loss at epoch 4: 0.6628 → 0.6441
→ 0.6235 → 0.6132, each the best yet. Train_loss is still decreasing meaningfully each epoch (0.614 →
0.600 → 0.587 → 0.568) — weight decay isn't suppressing genuine learning capacity yet, it's purely
closing the generalization gap. No sign of diminishing returns so far.

**Next action**: trial-013 pushes `--weight-decay` to 1.0 to continue probing for where this trend
saturates or reverses.

## trial-013

**Config**: identical to trial-008/010/011/012 except `--weight-decay 1.0` (2x trial-012, 100x the
original default). Same 4-epoch chained run via resume.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6143 | 0.6051 | 0.6875 | 0.6903 | 0.9953 | 0.9842 |
| 2 | 0.6050 | 0.6067 | 0.6898 | 0.6885 | 0.9826 | 0.9907 |
| 3 | 0.6000 | 0.6066 | 0.6928 | 0.6895 | 0.9787 | 0.9873 |
| 4 | 0.5962 | 0.6067 | 0.6944 | 0.6907 | 0.9755 | 0.9839 |

**Verdict: the trend reverses here — this is the other failure mode creeping back in.** The train/val
gap is now nearly zero (0.0105 at epoch 4), continuing the monotonic pattern from trial-008→012. But
unlike every prior trial, train_loss has essentially stalled (0.614 → 0.605 → 0.600 → 0.596, barely
moving after epoch 1) rather than continuing to drop — and recall stays pinned near-ceiling throughout
(0.98-0.99), the same signature as the original DEAD plateau (trial-001 through 007). wd=1.0 is strong
enough to be suppressing genuine learning capacity again, not just closing the generalization gap.
trial-012 (wd=0.5) is the more promising point: still-improving train_loss with only a modest (0.046)
gap, i.e. actively learning without wd=1.0's re-approach to DEAD.

**Next action**: extend trial-012 (wd=0.5) for more epochs (5, 6, ...) rather than testing another wd
value — it's the best candidate seen so far for a genuinely HEALTHY multi-epoch curve, and no config has
been run past 4 epochs yet despite the user's guidance to try 5-10+.

**trial-012 epoch 5** (extension, same config, resumed from the trial-012 checkpoint):

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 5 | 0.5449 | 0.6252 | 0.7264 | 0.6805 | 0.9170 | 0.9214 |

Gap widens back out to 0.080 (from epoch 4's 0.046) — the delay bought by wd=0.5 was real but finite;
overfitting resumes by epoch 5.

## Infra update: `colab exec` connection-drop root cause found, workaround adopted

Researched the recurring `RuntimeError: Connection was lost` against the upstream `google-colab-cli`
and `jupyter-kernel-client` GitHub repos. The VM-idle-pruning bug behind it (CLI-created runtimes hit
`403 USER_PROJECT_DENIED` on their keep-alive RPC, since ordinary accounts lack entitlement on Colab's
internal project) is documented in `googlecolab/google-colab-cli#14` and was fixed upstream in
v0.6.0 (which we already have installed) by switching to a Tunnel Frontend HTTP ping. A related,
apparently-still-open gap affects CLI-driven sessions run as discrete invocations with idle gaps
between them. Separately, the underlying `jupyter_kernel_client` websocket has `reconnect_interval`
disabled by design (`datalayer/jupyter-kernel-client#17`), and `google-colab-cli` never overrides that
default — so any transient blip on that connection during an active `exec` call is unrecoverable.

**Workaround (adopted per user direction)**: manually provision a T4 runtime via the Colab web UI (named
`claude_cafa6`), then adopt it into the CLI's local session store using `colab_cli`'s internal
`list_assignments()`/`SessionState`/`store.add()` APIs (documented as a community workaround in the
same GitHub issue thread). A **browser-owned** runtime is kept alive by the browser tab itself, not the
CLI's keep-alive daemon, sidestepping both known failure paths. Verified working: a GPU smoke test and
then a full 10-epoch training run (trial-015, below) both completed in a single `colab exec` call with
zero connection drops — a first for this sweep. Per user instruction, this session is not stopped
between trials; it stays alive until the user says so or the browser tab/instance is closed.
**Operating pattern going forward**: one `colab exec` call per experiment, up to 10 epochs, reusing the
adopted `claude_cafa6` session — no more per-epoch chunking.

## trial-015

**Config**: `esm2_t6_8M`, unfreeze=1, lr=2e-4, out_drop=0.2, esm_drop=0.0, attn_drop=0.0, grad_clip=1.0,
wd=0.5, train_n=12000/epoch (varying subset per epoch), val_n=2000, seed=42, bias-init fix. First clean,
uninterrupted 10-epoch run in a single `colab exec` call (no resume/optimizer-reset artifacts).

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6046 | 0.6039 | 0.6991 | 0.6918 | 0.9925 | 0.9812 |
| 2 | 0.5938 | 0.6049 | 0.7041 | 0.6938 | 0.9802 | 0.9889 |
| 3 | 0.5937 | 0.6020 | 0.7020 | 0.6947 | 0.9777 | 0.9849 |
| 4 | 0.5936 | 0.6076 | 0.7025 | 0.6943 | 0.9741 | 0.9677 |
| 5 | 0.5885 | 0.6015 | 0.7063 | 0.6885 | 0.9760 | 0.9593 |
| 6 | 0.5887 | 0.6040 | 0.7073 | 0.6870 | 0.9766 | 0.9529 |
| 7 | 0.5887 | 0.6133 | 0.7051 | 0.6953 | 0.9694 | 0.9949 |
| 8 | 0.5928 | 0.5990 | 0.7027 | 0.6947 | 0.9731 | 0.9749 |
| 9 | 0.5850 | 0.6000 | 0.7098 | 0.6912 | 0.9737 | 0.9672 |
| 10 | 0.5926 | 0.6074 | 0.7035 | 0.6947 | 0.9740 | 0.9773 |

**Verdict: the best-behaved trajectory in the sweep so far — a stable plateau, not overfitting or dead.**
train_loss drops from 0.6046 to a ~0.585-0.593 plateau by epoch 5 and stays there (not continuing toward
memorization). val_loss oscillates in a tight band (0.599-0.613) around ~0.605 with no systematic drift
in either direction across all 10 epochs — best value (0.5990) at epoch 8, essentially matching epoch
1's start. The train/val gap stays small (~0.01-0.02) and doesn't widen the way it did in every earlier
multi-epoch trial. This isn't a strong "both improving together" HEALTHY curve (val isn't clearly
trending down), but it's unambiguously stable rather than diverging — and train_loss itself plateauing
suggests the bottleneck is now model **capacity** at unfreeze=1, not regularization.

**Next action**: trial-016 tests `--unfreeze-last-n 2` (still minimal per the user's efficiency
preference, but more than the current capacity-limited 1), same wd=0.5, lr=2e-4, 10 epochs — check
whether more trainable capacity lets train_loss (and hopefully val_loss) push meaningfully past the
current plateau.

**User-confirmed constraints going forward**: (1) every trial from here on runs exactly 10 epochs with
`--max-train-samples` fixed at 12000, so trials stay directly comparable; (2) once a satisfactory config
is found at this model size (`esm2_t6_8M`), scale up to a larger model but re-run the regularization
search from scratch on the larger model rather than assuming trial-015/016/...'s settings transfer —
given how much trouble regularization caused at this scale, nothing about it should be assumed to
generalize across model sizes.

## Infra note: workspace crash, session re-adoption, reusable scripts

The `claude_cafa6` adopted session was lost when the user's browser/workspace crashed mid-trial-016
(first attempt): the `colab exec` call hung indefinitely against the dead runtime rather than erroring
out (consistent with the busy-poll-on-timeout class of bug in `googlecolab/google-colab-cli#82`), and
had to be killed via `TaskStop`. No progress was lost — the crash happened before any epoch completed
or checkpoint pushed. Re-adopted a freshly-reprovisioned browser runtime under the same name and
re-verified with a GPU smoke test before continuing.

Also extracted the repeated launch sequence (generate a `sys.argv` wrapper, upload token, upload
script, run `colab exec`, log output) into reusable scripts, since it had been copy-pasted by hand for
every single trial: `experiments/scripts/run_trial.sh <trial-name> <session-name> [--timeout N] --
<training_run.py args...>`, `experiments/scripts/adopt_session.py <session-name>`, and
`experiments/scripts/check_and_adopt_session.sh <session-name>` (checks liveness, adopts + smoke-tests
if needed). All trials from trial-016 onward are launched via `run_trial.sh`.

## trial-017 (abandoned)

**Config**: `esm2_t6_8M`, unfreeze=6 (full model), else identical to trial-016. Attempted 3 times against
the browser-adopted session — first attempt hung indefinitely after a workspace crash (killed via
`TaskStop`), second attempt found no active session at all, third attempt failed partway through epoch 1
(step ~1260/1500, no epoch completed, `exit code 1`, no captured traceback). All three failures cluster
in the back half of epoch 1, consistent with full-unfreeze's heavier per-step compute pushing epoch 1's
wall-clock past whatever window this connection can reliably sustain — the browser-adoption workaround
fixes VM idle-pruning but not the separate, still-unfixed `jupyter_kernel_client` zero-reconnect-interval
issue.

**Decision**: abandoned rather than retried further. It predates the user's new beam-search methodology
(below) and isn't required to seed it — trial-015/016 are both complete, reliable, and nearly identical.
"More capacity" (unfreeze) is folded into the beam search's own candidate axes instead of chased in
isolation.

## Methodology change: beam search

Per user direction, from here on the search follows a **beam search** (width 2, branching factor 3 per
beam member — 6 experiments per round) instead of ad hoc single-axis guessing:
1. From the best config(s) of the previous round, generate 6 experiments, each tweaking exactly one
   hyperparameter. 2 of the 6 must tweak a regularization axis.
2. Judge the 6, select the best 2, log what helped and what hurt.
3. From those top 2 (3 tweaks from each), generate the next 6. Repeat.
4. Stop when any single epoch hits **val_loss < 0.45** (the user-confirmed success criterion), when all
   imaginable options are exhausted, or when the user says stop — in the last case, keep enough state to
   resume cleanly.

New tracking files: `experiments/beam_search_state.json` (round/beam/queue state, resumable),
`experiments/results_index.csv` (rebuildable full index of every trial's config + best-epoch metrics +
verdict, via `experiments/scripts/build_results_index.py`). The user explicitly authorized adding new
search axes beyond the existing CLI flags and extending `training_run.py` as needed — this isn't meant
to be a pure grid search over already-exposed hyperparameters.

**Seed for Round 1**: trial-015 (unfreeze=1, lr=2e-4, wd=0.5, dropout=0.2/0.0/0.0) — best_val_loss=0.5990,
narrowly ahead of trial-016's 0.5995 (unfreeze=2, otherwise identical) and more efficient per the user's
minimal-unfreeze preference. Given the two are statistically indistinguishable, treating trial-015 as
sole seed rather than both is a simplification (pure width-2 beam would carry both forward) — acceptable
since unfreeze itself is one of the axes Round 1 tests anyway.

## trial-016

**Config**: `esm2_t6_8M`, unfreeze=2 (up from trial-015's 1), lr=2e-4, out_drop=0.2, esm_drop=0.0,
attn_drop=0.0, grad_clip=1.0, wd=0.5, train_n=12000/epoch (per-epoch variation), val_n=2000, seed=42,
10 epochs in one clean `colab exec` call (via `run_trial.sh`, on the re-adopted `claude_cafa6` session).

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 1 | 0.6038 | 0.6031 | 0.6991 | 0.6902 | 0.9897 | 0.9730 |
| 2 | 0.5940 | 0.6063 | 0.7042 | 0.6943 | 0.9783 | 0.9750 |
| 3 | 0.5947 | 0.6021 | 0.7007 | 0.6935 | 0.9744 | 0.9863 |
| 4 | 0.5948 | 0.6077 | 0.7022 | 0.6917 | 0.9742 | 0.9595 |
| 5 | 0.5899 | 0.6017 | 0.7061 | 0.6912 | 0.9759 | 0.9578 |
| 6 | 0.5905 | 0.6053 | 0.7062 | 0.6887 | 0.9742 | 0.9577 |
| 7 | 0.5908 | 0.6185 | 0.7040 | 0.6945 | 0.9688 | 0.9956 |
| 8 | 0.5946 | 0.5997 | 0.7013 | 0.6947 | 0.9728 | 0.9833 |
| 9 | 0.5870 | 0.5995 | 0.7081 | 0.6935 | 0.9724 | 0.9789 |
| 10 | 0.5946 | 0.6080 | 0.7003 | 0.6922 | 0.9735 | 0.9795 |

**Verdict: essentially identical to trial-015 — doubling unfrozen layers (1→2) made no meaningful
difference.** Train_loss plateau (~0.587-0.595) and val_loss oscillation band (0.5995-0.6185) both
closely match trial-015's (~0.585-0.593 / 0.599-0.613). A small capacity increase isn't the lever here.

**Next action**: trial-017 tests a much larger capacity jump — full unfreeze (`--unfreeze-last-n 6`,
i.e. the entire `esm2_t6_8M` encoder) — to decisively check whether capacity matters at all before
concluding this plateau reflects a data/task ceiling rather than a model-size limitation. If full
unfreeze still lands in the same place, the plateau is likely close to what's achievable with the
current `--max-train-samples 12000` budget and this task's difficulty, and it may be time to consider
this model size "satisfactory" per the user's stated criteria before moving to scale-up.

## Important nuance surfaced by the full trial-012 trajectory

Looking at trial-012's val_loss across all 5 epochs: **0.6046 → 0.6053 → 0.6077 → 0.6132 → 0.6252 — it
rises every single epoch, never once dropping below its epoch-1 value.** This pattern holds across every
regularization trial so far (008-013): train_loss reliably decreases (proving the model can learn
*something*), but val_loss has never yet gone meaningfully below where it started. What weight decay has
been doing is slowing the *rate* of val_loss's rise, not producing genuine held-out improvement — every
"OVERFITTING, smaller gap" verdict so far has been relatively better, not absolutely good. This is a more
specific and more concerning framing than plain overfitting: it raises the possibility of a genuine
train/val distribution mismatch (rather than just "not enough regularization"), which no amount of
weight-decay tuning alone would fix. Worth checking directly (e.g. inspect whether train/val splits have
any systematic difference — sequence length distribution, label distribution, near-duplicate proteins
leaking across the split) before continuing to chase weight-decay values further.

**trial-006 epoch 2** (extension, same config, resumed from the trial-006 checkpoint):

| epoch | train_loss | val_loss | train_acc | val_acc | train_recall | val_recall |
|-------|-----------|----------|-----------|---------|--------------|------------|
| 2 | 0.6023 | 0.6074 | 0.6902 | 0.6922 | 0.9686 | 0.9801 |

val_loss is essentially unchanged from epoch 1 (0.6070 → 0.6074, technically a hair worse) after 3000
total steps. recall drifted down slightly (0.9877 → 0.9801, a small move away from the ceiling, but
nowhere near enough to call it real progress). Two full epochs of flat val_loss is strong evidence
against "just needs more steps at this config" — all three quick levers tried so far (raise LR, fix
bias saturation, more patience) have failed to move the needle. **Pivoting to model capacity**: this is
`esm2_t6_8M`, Facebook's smallest ESM-2 checkpoint (8M params) — its own pretraining may simply not have
produced rich enough general protein representations for a small fine-tuning budget to extract
GO-aspect signal from. trial-007 tests the actual project default, `facebook/esm2_t12_35M_UR50D`
(per `SELECTED_MODEL` in `training_run.py`), keeping lr=2e-4 and the bias-init fix, with
`--max-train-samples` reduced (to ~6000) to leave margin for the larger model's slower per-step time
against the same ~9-10min connection ceiling.

## Round 1 results (trial-018 through trial-023)

All 6 ran back-to-back in one manual Colab-notebook session (`experiments/round1_manual.ipynb`,
generated from `experiments/round1_manifest.json`), pasted back into chat as
`experiments/logs/trial018-023.log` and split into per-trial `experiments/logs/trial-0NN.log` files.
Every trial completed all 10 epochs cleanly (no connection drops this batch). Seed: trial-015
(unfreeze=1, lr=2e-4, wd=0.5, dropout=0.2/0.0/0.0, best_val_loss=0.599).

| trial | axis | best epoch | best val_loss | best val_acc | best train_loss |
|-------|------|-----------|---------------|--------------|------------------|
| trial-018 | lr_schedule=cosine | 8 | **0.5959** | 0.6955 | 0.5865 |
| trial-022 | grad_accum_steps=2 | 8 | 0.5978 | 0.6973 | 0.5861 |
| trial-019 | warmup_steps=300 | 8 | 0.5981 | 0.6948 | 0.5925 |
| trial-021 | decouple_bias_wd (reg) | 8 | 0.5990 | 0.6950 | 0.5834 |
| trial-020 | label_smoothing=0.05 (reg) | 8 | 0.5996 | 0.6950 | 0.6033 |
| trial-023 | unfreeze_last_n=3 | 9 | 0.6015 | 0.6917 | 0.5883 |

**Bug found**: trial-020's per-epoch log shows `train_acc=0.0000 train_recall=0.0000` every epoch. Not
a training failure — `compute_metrics` was being called with the label-smoothed targets (0.05/0.95
instead of 0/1), and its exact-equality comparisons (`preds == labels`, `labels == 1`) can never match
a smoothed value, so both metrics silently zero out. val_loss/val_acc (computed without smoothing) were
unaffected, so trial-020's ranking above is still valid. Fixed in `training_run.py`: training-loop
metrics now compute against a separate `hard_labels` tensor captured before smoothing is applied.

**What helped**: both non-regularization schedule/optimization tweaks — cosine LR decay (trial-018) and
a larger effective batch via gradient accumulation (trial-022) — gave small real reductions below the
trial-015 seed. Every trial's best epoch landed at epoch 8/10 (trial-023 at 9), suggesting these configs
are approaching a plateau from above rather than wandering randomly.

**What hurt**: both regularization tweaks (label_smoothing, decouple_bias_wd) underperformed or merely
tied the seed. Combined with the modest, stable train/val gap seen across all of round 1 (~0.01-0.02,
not widening), there's no sign we're currently overfitting hard enough for *more* regularization to
help — round 2 tests *less* wd/dropout instead. `unfreeze_last_n=3` (trial-023) was worse, noisier
(val_loss oscillated 0.60-0.622 across its own 10 epochs, vs. trial-018's tighter 0.5959-0.6044 band),
and ~48% slower per epoch (327s vs 220s) for no benefit — more capacity alone isn't the lever, echoing
trial-016's same finding at unfreeze=2.

**Caveat**: all 6 results (0.5959-0.6015) sit inside a narrow band around the seed (0.599), and a single
trial's own epoch-to-epoch noise (e.g. trial-018 itself ranges 0.5959-0.6044 across its 10 epochs) is
comparable in size to the gap between round-1's best and worst trial. Round 1's "winners" are the best
available signal per the beam-search protocol, not confirmed-robust improvements — and all of them
remain far from the **val_loss < 0.45** target. If round 2 doesn't break this plateau meaningfully, that
would be a signal worth raising with the user before sinking further rounds into `esm2_t6_8M` at
unfreeze<=3 — it may be more of a model-capacity/task-difficulty ceiling than a regularization problem.

**Decision**: beam updated to `[trial-018, trial-022]` (`experiments/beam_search_state.json`, round 2).
Round 2 (`experiments/round2_manifest.json`, trial-024 through trial-029) branches 3 children off each:
- from trial-018 (cosine schedule): `trial-024` raises lr to 3e-4, `trial-025` lowers weight_decay to
  0.2 (regularization axis, testing under- vs over-regularization), `trial-026` bumps unfreeze to 2.
- from trial-022 (grad_accum=2): `trial-027` adds the cosine schedule on top (stacking both round-1
  winners), `trial-028` lowers output_dropout to 0.1 (2nd regularization axis, same under/over question
  from the other beam member), `trial-029` pushes grad_accum to 4 (effective batch 32).

## Round 2 results (trial-024 through trial-029)

All 6 ran back-to-back in one manual Colab-notebook session (`experiments/round2_manual.ipynb`,
generated from `experiments/round2_manifest.json`), pasted back into chat as
`experiments/logs/trial-024-029.log` and split into per-trial `experiments/logs/trial-0NN.log` files.
Every trial completed all 10 epochs cleanly. Beam parents: trial-018 (best_val_loss 0.5959) and
trial-022 (best_val_loss 0.5978).

| trial | parent | axis | best epoch | best val_loss | best val_acc | best train_loss |
|-------|--------|------|-----------|---------------|--------------|------------------|
| trial-025 | trial-018 | weight_decay=0.2 (reg) | 8 | **0.5960** | 0.6945 | 0.5753 |
| trial-026 | trial-018 | unfreeze_last_n=2 | 10 | **0.5961** | 0.6968 | 0.5817 |
| trial-024 | trial-018 | learning_rate=3e-4 | 8 | 0.5966 | 0.6942 | 0.5878 |
| trial-027 | trial-022 | +cosine lr_schedule | 9 | 0.5968 | 0.6960 | 0.5748 |
| trial-029 | trial-022 | grad_accum_steps=4 | 7 | 0.5971 | 0.6982 | 0.5789 |
| trial-028 | trial-022 | output_dropout=0.1 (reg) | 8 | 0.5984 | 0.6958 | 0.5855 |

**What helped**: lowering weight_decay on the trial-018 lineage (trial-025, 0.5960) and bumping
unfreeze to 2 (trial-026, 0.5961) both nudged marginally past their trial-018 parent's own epoch-8/9/10
noise band — but neither beats trial-018's own best (0.5959) in absolute terms. trial-024 (higher LR)
landed between trial-018 and trial-022, no clear signal either way.

**What hurt**: stacking cosine schedule onto trial-022 (trial-027) did *not* combine the two round-1
gains additively — it landed worse than trial-018 alone, suggesting the two round-1 "wins" weren't
independent effects. Pushing grad_accum to 4 (trial-029) was flat-to-worse than grad_accum=2
(trial-022's own 0.5978), so more accumulation isn't a scaling lever either. Lowering output_dropout
further (trial-028, 0.5984) was the worst trial of the round — some regularization is still needed,
even if round 1 already showed *more* isn't the answer.

**Plateau assessment — the round-1 caveat's trigger condition has now been met.** Round 2's best result
(trial-025, 0.5960) is *statistically indistinguishable* from round 1's best (trial-018, 0.5959) — a
0.0001 difference, an order of magnitude smaller than a single trial's own epoch-to-epoch noise
(~0.004-0.008 band width). Across 2 full rounds (12 trials branching from a common trial-015 seed at
0.599), covering both non-regularization axes (lr_schedule, warmup, grad_accum, learning_rate,
unfreeze_last_n) and regularization axes (weight_decay, dropout, label_smoothing, decouple_bias_wd) in
both directions, best val_loss has moved from 0.599 to 0.5959 to 0.5960 — essentially flat after the
first round. Every trial in both rounds also converges to a near-identical val_acc (~0.69-0.70) and
val_recall (~0.94-0.98) regardless of config, which reads less like "still searching for the right
hyperparameters" and more like `esm2_t6_8M` (with `max_train_samples=12000`) has hit a representational
or data-budget ceiling for this task that further regularization/schedule tuning at this scale won't
break. Flagging this to the user now rather than queuing a round 3 that's unlikely to move the needle.

**Decision**: beam updated to `[trial-025, trial-026]` (both trial-018 descendants;
`experiments/beam_search_state.json`, round 2 marked complete). Round 3 design is paused pending user
input on whether to continue hyperparameter search at `esm2_t6_8M` scale or pivot to a larger model
per the project's original plan ("once a satisfactory config is found at this scale, scale up and
re-run the regularization search from scratch") — model-scale is the one major lever untested so far.

## Scale-up phase: esm2_t6_8M -> esm2_t12_35M

User direction (2026-07-24): pivot to model scale rather than continue tuning at `esm2_t6_8M`. The
`esm2_t6_8M` phase is paused (not abandoned) in `experiments/beam_search_state.json`'s `phases` map;
`esm2_t12_35M_scale_up` is now the active phase. `esm2_t12_35M` is the project's actual default model
(`SELECTED_MODEL` in `training_run.py`) and was only tried once before — trial-007, 1 epoch,
pre-beam-search, before the bias-init fix's effects were well understood and before 10-epoch training
was shown to matter — so it isn't a fair prior data point against the now-mature methodology.

The user also flagged something worth acting on directly: across the *entire* `esm2_t6_8M` phase
(trial-008 through trial-029), train/val loss gap stayed small and stable (~0.01-0.02) no matter how
regularization was dialed — the sweep has never actually observed overfitting. That makes "reduce
regularization further" hard to interpret: is 0.596 a true ceiling, or is the metric just not sensitive
to regularization changes in this regime? Per the user's request, one of the two seed-phase trials
strips all regularization to check.

`experiments/round3_manifest.json` (2 trials, notebook: `experiments/round3_manual.ipynb`):
- **trial-030** (seed/timing): the `esm2_t6_8M` phase's actually-validated baseline (unfreeze=1,
  lr=2e-4, wd=0.5, dropout=0.2/0.0/0.0, batch=8, grad_clip=1.0, constant schedule — deliberately *not*
  round-1/2's micro-tweaks, since those never beat this baseline outside noise), model swapped to
  `esm2_t12_35M_UR50D`. Epochs capped at 6 (vs. 10 at 8M) as a wall-clock probe — 35M has 2x the
  encoder depth of 8M and per-epoch time at this scale is unmeasured; its `[TIMING]` lines will size
  epoch count / `max_train_samples` for the rest of this phase.
- **trial-031** (diagnostic, user-requested): identical to trial-030 except `weight_decay=0.0`,
  `output_dropout=0.0`, `esm_dropout=0.0`, `attention_dropout=0.0` — zero regularization. If train/val
  loss finally diverge here, that confirms the metric can detect overfitting at all, calibrating how
  much regularization is actually load-bearing going forward. If the gap stays just as flat as every
  prior trial even at zero regularization and 4x the parameters, that's strong evidence the plateau is
  a data/task ceiling rather than a regularization problem.

**Next step**: once trial-030/031 land, size and launch a full 6-experiment round-1 beam-search batch
at this scale (branching off trial-030, same width-2/branching-3/2-regularization-required protocol as
the 8M phase), informed by trial-031's overfitting-or-not signal.

## Scale-up results: trial-030 (seed/timing), trial-031 (zero-regularization diagnostic)

Both ran in one manual browser Colab session (`experiments/round3_manual.ipynb`), pasted back as
`experiments/logs/trial-030-031.log`, split into per-trial files. Both completed all 6 epochs.

**trial-030** (baseline: unfreeze=1, lr=2e-4, wd=0.5, dropout=0.2/0.0/0.0, batch=8): best val_loss
**0.5946** at epoch 4, edging past the entire `esm2_t6_8M` phase's best (0.5959) in 6 epochs instead of
10. **Correction**: this is not a clean "healthy" result — val_loss rises from 0.5946 (epoch 4) to
0.5959 (epoch 6) while train_loss keeps falling the whole time (0.6027 → 0.5795), the same
overfitting shape as trial-031, just much weaker (train/val gap widens from ~0 to only ~0.016, vs.
trial-031's ~0.03 to ~0.04). `[TIMING]`: ~530s/epoch (vs. ~230s/epoch at 8M scale, ~2.3x slower per
epoch — roughly tracks the 2x encoder-depth difference plus some overhead). Total wall clock 3225s
(~54min) for 6 epochs, in line with the ~1h trial budget.

**Noise floor, per user direction**: val_loss > 0.595 should be treated as noise, not signal. Under
that bar, every single `esm2_t6_8M`-phase result (0.596–0.6015 across 22 trials) was noise — none of
that phase's "winners" were ever real. Only trial-030 (0.5946) and trial-031 (0.5918) have cleared
0.595 so far, both at the 35M scale. This is now the standing bar for judging every trial going
forward, recorded in `beam_search_state.json` as `noise_floor_val_loss`.

**trial-031** (zero regularization: wd=0.0, dropout=0.0/0.0/0.0, otherwise identical to trial-030):
best val_loss **0.5918** at epoch 4 — the best result of the entire sweep so far, both phases. But
critically, this is the **first clear overfitting signal in the whole sweep**: train_loss falls
monotonically every epoch (0.6012 → 0.5875 → 0.5826 → 0.5769 → 0.5660 → 0.5584) while val_loss bottoms
at epoch 4 (0.5918) and then rises through epoch 6 (0.5988) — the train/val gap widens from ~0.03 at
epoch 1 to ~0.04 by epoch 6. Every trial across the entire `esm2_t6_8M` phase kept a flat, stable gap
regardless of regularization; this is the first time removing regularization actually produced a
textbook overfitting curve.

**Interpretation**: the `esm2_t6_8M` plateau wasn't purely a "regularization doesn't matter" ceiling —
it was at least partly a capacity ceiling. At 4x the parameters (`esm2_t12_35M`), the model is finally
expressive enough to overfit, and it does so *even with trial-030's full regularization* (wd=0.5,
dropout=0.2) — just weakly. Zeroing regularization (trial-031) doesn't create the overfitting, it just
amplifies an effect that was already present. This directly answers the diagnostic question raised
before this batch: regularization *is* a real, measurable lever again at this scale — the question is
now *how much*, not *whether*.

**Decision**: the beam-search protocol continues into a proper round-1 at this scale (`trial-030` and
`trial-031` as the two seed points), designed jointly with three additional batch-size trials the user
asked for directly off trial-031. See `experiments/round4_manifest.json` /
`experiments/round4_manual.ipynb` (6 trials, trial-032 through trial-037):

- **trial-032/033/034** (user-requested, off trial-031, `learning_rate` unchanged at 2e-4): batch_size
  32, batch_size 128, and batch_size 128 with `grad_accum_steps=4` (effective batch 512). Primarily a
  throughput/memory probe on the T4 at this model scale — batch=128 cuts optimizer steps/epoch from
  1500 to ~94 (16x fewer), and the accum=4 variant to ~23/epoch, so neither is expected to be
  competitive on val_loss purely from undertraining; the useful signal is `[TIMING]` (does a bigger
  batch use the T4 better?) and whether OOM becomes a constraint.
- **trial-035** (regularization, off trial-030): light-touch middle ground — `output_dropout=0.1`,
  `weight_decay=0.1` — testing whether trial-031's early-epoch gains are reachable without its
  late-epoch blowup.
- **trial-036** (regularization, off trial-030): isolates which of trial-030's two regularizers
  actually matters — keeps `output_dropout=0.2`, zeroes `weight_decay`.
- **trial-037** (off trial-030): `unfreeze_last_n=2` — capacity bump untested so far at this scale.

Per the user's operational note: they're running this batch in a browser Colab session kept alive
manually (not the CLI-driven `colab new`/`exec`/`stop` flow), so there's no session for me to
explicitly stop between trials — the user closes it themselves once the batch is done and they're
ready to hand results back for analysis.

**Mid-batch update (round_1/trial-032..037 still running)**: two user notes, both recorded in
`beam_search_state.json`:
- **Epochs fixed at 10 going forward**, model scale aside — for cross-run comparability, and because
  the user has seen harsh overfitting show up in later epochs (matches trial-031, whose train/val gap
  was still widening at epoch 6, its last epoch). trial-030/031 and this round (trial-032..037) already
  ran at epochs=6 before this policy was set, so their results aren't directly epoch-for-epoch
  comparable to future epochs=10 trials at this scale — only per-epoch trajectory shape is.
- **GPU memory**: trial-033 (batch=128) almost OOM'd the T4; trial-034 (batch=128, grad_accum=4),
  run right after, was surprisingly easier on memory. Checked `training_run.py`: `grad_accum_steps`
  doesn't reduce per-step batch size (the DataLoader uses `batch_size` as-is every micro-step), so
  there's no mechanism in the code for accumulation alone to lower peak memory — trial-034 processes
  identical full batches of 128. Most likely explanation is CUDA's caching allocator having already
  grown its pool to fit batch=128 during trial-033, letting trial-034 reuse it — a run-order artifact
  of one session, not a real property of `grad_accum_steps`. Flagging so this isn't misread as
  "accumulation saves memory" in a future config decision.

## round_1 results (trial-032 through trial-037)

All 6 ran in the user's manually-kept-alive browser Colab session, pasted back as
`experiments/logs/trial-032-037.log`, split per-trial. All completed their (pre-policy) 6 epochs.

| trial | parent | axis | best epoch | best val_loss | notes |
|-------|--------|------|-----------|----------------|-------|
| trial-033 | trial-031 | batch_size=128 | 5 | **0.5902** | new best of the entire sweep; nearly OOM'd the T4 |
| trial-035 | trial-030 | dropout=0.1, wd=0.1 (reg) | 5 | 0.5914 | 2nd best; beats both trial-030 (heavy reg) and trial-031 (zero reg) |
| trial-032 | trial-031 | batch_size=32 | 4 | 0.5916 | ties trial-031 |
| trial-031 | (seed) | zero regularization | 4 | 0.5918 | prior best, for reference |
| trial-037 | trial-030 | unfreeze_last_n=2 | 4 | 0.5936 | ~15% slower/epoch than unfreeze=1 |
| trial-036 | trial-030 | wd=0.0, dropout=0.2 (reg) | 4 | 0.5937 | better than trial-030, worse than trial-035's lighter combo |
| trial-030 | (seed) | baseline (wd=0.5, dropout=0.2) | 4 | 0.5946 | prior baseline, for reference |
| trial-034 | trial-031 | batch=128, grad_accum=4 | 6 | 0.5960 | still falling at epoch 6 (last), undertrained — above the 0.595 noise floor, not yet a real result |

**Batch size, surprising result**: bigger batch (128) gave the best val_loss of the sweep so far
despite ~94 optimizer steps/epoch — 16x fewer than batch=8's 1500. But it did **not** speed up
wall-clock: batch=8 trials ran ~530-540s/epoch, batch=32 ~680s/epoch, batch=128 ~735s/epoch — *larger*
batches were *slower* per epoch, the opposite of the usual "bigger batch, better GPU utilization"
expectation. Likely cause: `data_collate_function` pads each batch to that batch's own longest
sequence (`pad_sequence`, no length-bucketing), and batches are drawn from a random shuffle
(`shuffle=True`, no sorting) — a larger batch has a higher chance of containing at least one
near-`max_length` (1024) sequence, forcing more padding waste per batch as batch size grows. Total
GPU-perceived work per epoch likely rises with batch size even though the underlying token count is
fixed, which would fully explain the trend. Not fixed as part of this batch (would need
length-bucketed batching in `training_run.py`); noting it as a real code-level explanation rather than
something specific to `batch_size=128` itself.

**Regularization**: trial-035's light-touch (dropout=0.1/wd=0.1) beat trial-030 (heavy: wd=0.5/dropout=0.2)
and roughly tied trial-031 (zero) — regularization strength here behaves like a real, tunable
parameter with a genuine sweet spot in the middle, not a monotonic "less is more" or "more is better"
relationship.

**GPU memory (user-reported)**: trial-033 (batch=128) nearly OOM'd the T4; trial-034 (batch=128,
grad_accum=4, same per-step batch size) was easier — consistent with the run-order/allocator-caching
explanation above rather than a real accumulation effect.

**Decision**: new beam = `[trial-033 (0.5902), trial-035 (0.5914)]`. Round 2 at this scale
(`experiments/round5_manifest.json` / `experiments/round5_manual.ipynb`, trial-038 through trial-043)
branches 3 children off each, now at the newly-fixed **epochs=10**:
- from trial-033 (batch=128, zero reg): `trial-038` reruns the exact same config at epochs=10 (its
  own 6-epoch curve was just starting to tick up — need the full 10-epoch picture); `trial-039`
  (regularization) adds trial-035's dropout=0.1/wd=0.1 on top of batch=128, testing whether the two
  independent wins stack; `trial-040` tries batch_size=64, interpolating between trial-032 and
  trial-033 to map the batch-size curve more finely.
- from trial-035 (dropout=0.1/wd=0.1, batch=8): `trial-041` reruns the exact same config at
  epochs=10; `trial-042` (regularization) tries an even lighter touch (dropout=0.05/wd=0.05) to see if
  0.1/0.1 is close to the true optimum or if there's more room; `trial-043` adds `unfreeze_last_n=2` on
  top of the light-regularization config (trial-037 tested unfreeze=2 only under heavy regularization).

**Runtime warning**: at epochs=10, this round is meaningfully longer than round_1. Extrapolating from
round_1's per-epoch timings (~530-540s/epoch at batch=8, ~680s at batch=32, ~735s at batch=128) across
6 trials at 10 epochs each, total wall clock for this batch is likely **~10-11 hours** run back to
back in one session — worth knowing before committing to running it in one sitting, especially via a
manually-kept-alive browser session rather than the CLI's auto-managed one. trial-038/039 repeat
trial-033's near-OOM batch_size=128, now for 10 epochs instead of 6 — same memory risk, longer
exposure.

## round_2 results, part 1 (trial-038 through trial-040; trial-041 interrupted)

Ran in the user's manual browser Colab session, pasted back as `experiments/logs/trial-038-041.log`.
The session **disconnected mid-trial-041**, after epoch 7/10 — this is exactly the long-single-session
risk flagged above. trial-038/039/040 completed all 10 epochs cleanly first.

| trial | axis | best epoch | best val_loss | status |
|-------|------|-----------|----------------|--------|
| trial-038 | batch=128, zero reg (rerun @10ep) | 7 | **0.5891** | complete — new best of the entire sweep |
| trial-039 | batch=128, dropout=0.1/wd=0.1 (rerun @10ep) | 7 | 0.5900 | complete |
| trial-040 | batch_size=64 | 7 | 0.5905 | complete |
| trial-041 | dropout=0.1/wd=0.1, batch=8 (rerun @10ep) | 5 (so far) | 0.5914 (so far) | **interrupted at epoch 7/10** — no checkpoint (`checkpoint_mode=final`), needs a full rerun |

**New best of the sweep**: trial-038 at 0.5891 (epoch 7), edging past trial-033's 0.5902. But the
extra epochs revealed something round_1 (capped at 6 epochs) couldn't show: at batch=128, val_loss
doesn't rise cleanly after its early minimum — it **oscillates** (0.5902 → 0.5922 → 0.5891 → 0.5923 →
0.5939 → 0.5948 across epochs 5–10). trial-039 and trial-040 show the same pattern. Most likely
explanation: batch=128 means only ~94 optimizer steps/epoch (vs. batch=8's 1500), so each epoch's
val_loss is a noisier estimate — this reads more like a wide-noise regime than a textbook overfitting
curve. trial-041's own first 6 epochs reproduced trial-035's original run bit-for-bit (deterministic,
same seed) — useful confirmation the pipeline is reproducible — and its epoch 7 (0.5955) sat close to
epoch 6 (0.5958), no clean overfit turn visible yet before the disconnect.

**Interruption**: with `--checkpoint-mode final` (the sweep's default, chosen specifically to avoid
per-epoch Hub commits), a mid-trial disconnect loses the whole trial — there's no partial checkpoint
to resume from. trial-041 must be fully rerun. Not switching the default to `every-epoch` for this
(that mode exists specifically for the CLI's `--resume-from-revision` retry path, and adds a Hub
commit per epoch); instead, per the user's direction, **future rounds are batched into sets of 3
trials per notebook** (down from 6) — this halves how much work a single disconnect can lose, and
roughly halves session length too (a 3-trial batch-8 round is ~5h vs. 6-trial's ~10-11h).

**Next batch** (`experiments/round5b_manifest.json` / `experiments/round5b_manual.ipynb`, 3 trials):
`trial-041` (full rerun, unchanged config), `trial-042` (regularization: dropout=0.05/wd=0.05, even
lighter than trial-035's 0.1/0.1), `trial-043` (unfreeze_last_n=2 on top of the light-regularization
config). This completes round_2's original 6-trial design, just split across two smaller batches.

## round_2 results, part 2 (trial-041 through trial-043) — full 10-epoch curves complete

All 3 ran cleanly to 10 epochs this time, pasted back as `experiments/logs/trial-041-043.log`.

| trial | axis | best epoch | best val_loss | epoch-10 val_loss |
|-------|------|-----------|----------------|--------------------|
| trial-041 | dropout=0.1/wd=0.1, batch=8 (rerun, complete) | 5 | 0.5914 | 0.6088 |
| trial-042 | dropout=0.05/wd=0.05, batch=8 (reg) | 4 | 0.5929 | 0.6046 |
| trial-043 | unfreeze_last_n=2, dropout=0.1/wd=0.1 | 3 | 0.5923 | 0.6249 |

This is the **harsh late-epoch overfit the user expected** — and exactly why epochs got fixed at 10.
All three curves rise sharply after their best epoch while train_loss keeps falling: trial-041 goes
0.5914 → 0.6088 (epochs 5→10), trial-043 goes 0.5923 → 0.6249 (epochs 3→10, the worst of the round —
capacity (unfreeze=2) makes the overfit worse once regularization isn't holding it back). trial-042
confirms trial-035/041's dropout=0.1/wd=0.1 is closer to the real sweet spot than going lighter still
— 0.05/0.05 landed worse (0.5929 vs 0.5914).

### round_2 complete — full ranking (trial-038 through trial-043)

| trial | axis | best val_loss | shape |
|-------|------|----------------|-------|
| trial-038 | batch=128, zero reg | **0.5891** | noisy oscillation, not monotonic |
| trial-039 | batch=128, dropout=0.1/wd=0.1 | 0.5900 | noisy oscillation, not monotonic |
| trial-040 | batch=64, zero reg | 0.5905 | noisy oscillation, not monotonic |
| trial-041 | batch=8, dropout=0.1/wd=0.1 | 0.5914 | clean rise-then-overfit |
| trial-043 | batch=8, unfreeze=2, dropout=0.1/wd=0.1 | 0.5923 | clean rise-then-overfit, sharpest |
| trial-042 | batch=8, dropout=0.05/wd=0.05 | 0.5929 | clean rise-then-overfit |

**Caveat worth taking seriously**: the top 3 (all batch≥64) show *noisy oscillation*, not the clean
"improve then overfit" shape the batch=8 trials show. That makes "best single epoch" a shakier
statistic for them — with only ~94-188 optimizer steps/epoch, each epoch's val_loss is a noisier
estimate, and the minimum of a noisy sequence is downward-biased vs. its true expected value. It's
not yet clear whether trial-038's 0.5891 reflects a genuinely better-trained model or a lucky low
epoch in a noisier curve. Also interesting: light regularization (0.1/0.1) *helped* at batch=8
(trial-041 0.5914 beats trial-031's zero-reg 0.5918) but *hurt* at batch=128 (trial-039 0.5900 is
worse than trial-038's zero-reg 0.5891) — plausibly a real batch-size interaction, or plausibly just
noise given the gap (0.0009) is smaller than the oscillation band itself. Round 3 is designed to
settle both questions.

**Decision**: new beam = `[trial-038 (0.5891), trial-039 (0.5900)]`, both batch=128. Round 3
(`experiments/round6a_manifest.json` + `round6b_manifest.json`, trial-044 through trial-049, split
into the now-standard two 3-trial batches) branches 3 children off each:

- from trial-038 (batch=128, zero reg): `trial-044` reruns the exact same config at `seed=123` — a
  reproducibility check, not a hyperparameter axis, addressing the noise-vs-real-result question
  above directly; `trial-045` (regularization) tries a lighter touch than trial-039's (dropout=0.05/
  wd=0.05) to see if batch=128 simply wants less regularization than batch=8's optimum, rather than
  concluding regularization doesn't help at this batch size; `trial-046` tries `batch_size=96`,
  further mapping the curve between trial-040 (64) and trial-038 (128).
- from trial-039 (batch=128, dropout=0.1/wd=0.1): `trial-047` reruns at `seed=123`, paired with
  trial-044; `trial-048` (regularization) isolates weight_decay alone (dropout=0.0) at batch=128,
  mirroring trial-036's isolation test for trial-030 at batch=8; `trial-049` adds
  `grad_accum_steps=2` (effective batch 256) — a more moderate accumulation than trial-034's
  inconclusive accum=4 test.

**Runtime note**: this round's leading lineage (batch=128, ~810s/epoch) is much slower per trial than
batch=8's ~580s/epoch. Even at the new 3-trials-per-batch policy, each batch here is likely
**~6-6.5 hours** (3 trials × ~10 epochs × ~800s), not the ~5h a batch=8 round of 3 would take — the
batching policy caps trial *count* per session, not total time, and this round's trials are all
individually expensive. Flagging in case the user wants an even smaller batch (e.g. 2) specifically
for this batch=128-heavy round.

## round_3 part 1 results (trial-044 through trial-046) — the noise question is answered

All 3 completed 10 epochs, pasted back as `experiments/logs/trial-044-046.log`. trial-044 hit a
transient infra hiccup — `HF_TOKEN` fetch from Colab Secrets timed out mid-run (`"Requesting secret
HF_TOKEN timed out"`), so no Hub checkpoint was pushed for it (`[WARN] --checkpoint-mode final
requested but no HF token was found`). Training itself was unaffected — all 10 epochs completed and
the logged metrics are fully valid, there's just no `kelmoir/cafa6-esm@trial-044` branch. Every other
trial resolved its token normally; treating this as a one-off flake, not a new structural issue like
the earlier documented CLI-Secrets unreachability.

| trial | axis | best epoch | best val_loss |
|-------|------|-----------|----------------|
| trial-038 | batch=128, zero reg (seed=42) | 7 | 0.5891 |
| trial-046 | batch_size=96, zero reg | 5 | 0.5893 |
| trial-045 | batch=128, dropout=0.05/wd=0.05 (reg) | 7 | 0.5895 |
| trial-039 | batch=128, dropout=0.1/wd=0.1 (reg) | 7 | 0.5900 |
| trial-040 | batch_size=64, zero reg | 7 | 0.5905 |
| **trial-044** | **batch=128, zero reg (seed=123)** | 6 | **0.5921** |

**trial-044 answers the reproducibility question directly, and the answer is: mostly noise.**
trial-044 is the *exact same config* as trial-038 — only the random seed changed (123 vs 42) — and its
best val_loss (0.5921) is **worse than every other trial in this table**, a 0.0030 gap from trial-038
purely from re-seeding. Every other "improvement" in the table (batch 64→96→128, or the choice of
regularization) spans a narrower band (0.5891–0.5905, just 0.0014) than that single seed-to-seed gap.
In other words: the fine-grained ranking this round has been chasing — is 96 better than 64? is
0.05/0.05 better than zero? — sits **inside** the noise floor set by ordinary seed variance, not above
it. The batch-size/regularization exploration at this scale has reached the point where a single
seed=42 run per config is no longer a reliable enough signal to rank configs this close together.

**What this doesn't undermine**: the *large* effects found earlier in the sweep remain solid —
batch≥64 beating batch=8 by ~0.002-0.003 minimum, the 8M→35M capacity jump, and the harsh late-epoch
overfitting at batch=8 with heavier regularization (trial-041/043's 0.03-0.06 blowups) are all well
outside this noise band. It's specifically the *last* round of fine-tuning (batch 64 vs 96 vs 128,
0.05 vs 0.1 regularization) that this result calls into question.

**Decision**: round_3 part 2 (`experiments/round6b_manifest.json` / `experiments/round6b_manual.ipynb`,
trial-047/048/049 — seed-123 replication of trial-039, weight-decay isolation, grad_accum=2) is
already designed and gets sent as planned, since trial-047 will add a second seed-variance data point
specifically for the light-regularization config. But **this is a natural point to ask the user
whether to keep grinding this axis** — further fine hyperparameter search here would need multi-seed
averaging per config to produce a trustworthy ranking, multiplying cost, for gains that (per this
data) may not be real. Not deciding this unilaterally; raising it once round_3 part 2 is back.

## round_3 part 2 results (trial-047 through trial-049) — noise finding confirmed twice, phase closes

All 3 completed 10 epochs, pasted back as `experiments/logs/trial-047-049.log`.

| trial | axis | best epoch | best val_loss |
|-------|------|-----------|----------------|
| trial-038 | batch=128, zero reg (seed=42, ref) | 7 | 0.5891 |
| trial-048 | weight_decay=0.1 alone, dropout=0.0 (reg) | 7 | 0.5892 |
| trial-046 | batch_size=96, zero reg (ref) | 5 | 0.5893 |
| trial-045 | dropout=0.05/wd=0.05 (reg, ref) | 7 | 0.5895 |
| trial-039 | dropout=0.1/wd=0.1 (seed=42, ref) | 7 | 0.5900 |
| trial-040 | batch_size=64, zero reg (ref) | 7 | 0.5905 |
| trial-049 | dropout=0.1/wd=0.1, grad_accum_steps=2 | 9 | 0.5913 |
| **trial-044** | **batch=128, zero reg (seed=123, replicate of 038)** | 6 | **0.5921** |
| **trial-047** | **dropout=0.1/wd=0.1 (seed=123, replicate of 039)** | 6 | **0.5929** |

**The noise finding is now confirmed twice, independently.** trial-047 reruns trial-039's exact config
at seed=123 and lands at 0.5929 — worse than trial-039's 0.5900 by 0.0029, almost identical to the
0.0030 gap trial-044 showed vs. trial-038. Two different base configs, reseeded independently, both
degrade by ~0.003. Meanwhile every seed=42 result in this family (038, 039, 040, 045, 046, 048, 049)
spans only 0.5891–0.5913 — a 0.0022 band, still narrower than either single seed-swap. This isn't a
one-off anymore: **at this scale, a single seed=42 run per config cannot reliably rank configs that
differ by less than ~0.003 val_loss**, and essentially every hyperparameter variation tried in round_3
(batch 64→96→128, regularization 0→0.05→0.1, grad_accum 1→2) falls inside that band.

**One result does survive as a real (if small) finding**: trial-048 isolates weight_decay alone
(0.1, dropout=0.0) and lands at 0.5892 — indistinguishable from zero regularization (trial-038,
0.5891). Weight_decay alone has essentially no effect at batch=128, mirroring trial-036's finding at
batch=8 (dropout mattered, weight_decay didn't). This is a qualitative conclusion about *which axis
has any effect at all*, not a fine ranking, so it isn't undercut by the noise-floor finding above.

### Closing the esm2_t12_35M_scale_up regularization/batch-size search

Per the user's direction, this phase closes here rather than continuing into more multi-seed-averaged
rounds — the next phase is dataset expansion (see below), and further grinding on batch
size/regularization at single-seed resolution would mostly be chasing noise per the finding above.

**Practical takeaway for whatever config gets carried forward**: anything in the cluster
`{trial-038, trial-039, trial-040, trial-045, trial-046, trial-048}` — batch 64–128, unfreeze=1, zero
to light regularization (dropout 0–0.1, weight_decay optional) — performs equivalently within noise,
best val_loss ≈0.589–0.591. `trial-038`'s exact config (batch=128, zero regularization) is as
reasonable a default as any: simplest (no regularization to tune), among the best measured, and
dropout's the regularizer that would matter if any is added later. Given the model/task itself is
about to change substantially (see below), it's not worth spending more compute narrowing this
further before that change — the overfitting/regularization dynamics may look completely different
once training on ~26,000 real GO terms instead of 3 aspects.

**Full sweep summary, both phases**: `esm2_t6_8M` (rounds 1–2, trial-018–029) plateaued at val_loss
≈0.596, never distinguishable from noise per the later-established 0.595 floor. Pivoting to
`esm2_t12_35M` (trial-030 onward) broke that ceiling immediately (val_loss ≈0.589–0.595, real
signal), confirming the plateau was at least partly a capacity ceiling, not a fundamental limit of the
architecture or data. Batch size turned out to be the single largest lever found (batch≥64 beating
batch=8 by ~0.002-0.003, consistently, outside the noise band) — an unplanned discovery from a
user-requested throughput test, not something the original search design anticipated.

## Next: dataset/task expansion

The whole sweep above optimized `training_run.py`'s current label space — the 3 top-level GO aspects
(C/F/P) — which is a heavy simplification of the actual CAFA task. `generate_dataset.py` currently
computes `unique_go_terms` from `train_terms.tsv` but discards it, building `label2id` from
`unique_aspects` instead. The raw data has **26,125 unique GO terms** (2,651 C / 6,616 F / 16,858 P),
with a long tail (668 terms have ≥100 annotations, 1,585 have ≥50, 7,873 have ≥10) and a median of 4
annotated terms per protein. This likely explains why the aspect-only task plateaus so early: 3
classes carries very little information for the model to differentiate proteins by, regardless of
hyperparameters. Planned next direction (not yet started): a coarse-to-fine curriculum that adds a
term-level prediction head (filtered to a frequency-thresholded subset of terms, not all 26,125) on
top of the existing aspect head, exploiting the GO DAG's hierarchy rather than flattening it away —
full reasoning and alternatives discussed with the user, to be captured in a fresh planning pass
before implementation starts.

## Dataset expansion deployed (2026-07-26) — every trial above predates this

Before touching `generate_dataset.py`, ran a diagnostic (`experiments/diagnostic_eval_stage1.ipynb`,
log in `experiments/logs/diagnostics_1.log`) to check whether the aspect model was worth building on
at all — `compute_metrics()`'s accuracy/recall alone couldn't distinguish "learned real signal" from
"learned the label prior" (a trivial always-positive baseline scores ~0.698 accuracy / 1.0 recall,
suspiciously close to every trial's reported numbers). Result: AUROC clearly above 0.5 for every
aspect (C 0.676, F 0.631, P 0.607, macro 0.639) — the model has real, if modest, discriminative
signal, not just the prior. Not a dead end, though genuinely useful quality would need far more data
and training than this sweep's scale (12,000 train samples, 6-10 epochs) — noted as an open
possibility, not pursued further for now.

`generate_dataset.py` was then rewritten: labels are now **383 coarse GO terms** (the 3 aspect roots
+ every term within 2 hops with ≥50 annotations after true-path propagation via `obonet`), not just
the 3 aspects. Propagation restricted to `is_a`/`part_of` edges (the two relationship types that
support GO's true-path rule); `regulates`/`positively_regulates`/`negatively_regulates` edges are
excluded from label propagation but preserved in a new `go_dependency_graph.json` artifact (the full
40,122-node/77,229-edge graph, all 5 relationship types) pushed alongside the dataset to
`kelmoir/cafa6`, so a later curriculum stage can look up any coarse term's descendants without
recomputing anything. `training_run.py`'s hardcoded `UNIQUE_ASPECTS = ["C","F","P"]` (used to size
the label vector and the bias-prior computation) was removed in favor of reading the dataset's real
label count dynamically — it's now label-space-agnostic. Dataset regenerated and pushed for real
(Train=47,817 / Val=13,485 / Test=13,479); hit and worked around a `hf_xet` version-mismatch bug in
the process (`HF_HUB_DISABLE_XET=1`, documented in `CLAUDE.md`).

**Every trial in this log (trial-001 through trial-049) was run against the old 3-aspect label
space** — none of the regularization/batch-size conclusions above have been validated against the
new 383-term task. First real step from here: an actual training trial against the new dataset.

## New phase: esm2_t12_35M_coarse_go_terms (trial-050)

Checked the new dataset's actual label profile before designing the trial: **3.58% overall positive
rate** (mean 13.7 of 383 labels positive per protein, median 12, range 2–96) — much sparser than the
old 3-aspect task's ~70% per-label positive rate. This changes two things:

- **compute_metrics()'s accuracy is now close to meaningless.** A trivial "always predict negative"
  baseline already scores ~96.4% accuracy at this positive rate — the opposite failure mode from the
  old task's "always predict positive" trivial baseline (~69.8%), but the same underlying problem:
  accuracy alone can't be trusted here either. val_loss and recall are the metrics worth watching
  until precision/F1/AUROC (already built for the Stage-1 diagnostic) get wired into the regular
  training loop.
- **The esm2_t12_35M_scale_up phase's "zero regularization is fine" conclusion is not assumed to
  transfer.** That was reached on a dense, ~70%-positive task; overfitting dynamics on a sparse,
  383-way, rare-class-heavy task are unvalidated. trial-050 (`experiments/round7_manifest.json` /
  `experiments/round7_manual.ipynb`) reuses that phase's *original* validated baseline instead
  (unfreeze=1, lr=2e-4, wd=0.5, dropout=0.2/0.0/0.0, batch=8, epochs=10, matching trial-030's
  config) — the safe, well-tested starting point, not the later fine-tuned conclusion. batch=8 over
  128 for the same reason as before: the batch-size finding was about a quality edge from noisier
  updates, not speed (larger batches were consistently slower per epoch here), so there's no reason
  to start large on an unvalidated task.

The `noise_floor_val_loss` (0.595) from the old task does **not** carry over — this task's BCE loss
sits at a structurally different scale given the different positive rate, and needs its own
calibration once trial-050 lands.

## trial-050 result — looked like a collapse, wasn't, and exposed a real metrics bug

Raw log: val_loss improved cleanly (0.0803 → 0.0772 over 10 epochs, best at epoch 7), but val_recall
stayed pinned at 2.7–3.6% the entire run and val_accuracy (97.2%) sat barely above the trivial
always-negative baseline (96.4%). This is the textbook signature of unweighted BCE collapsing under
severe class imbalance — before concluding that, ran a proper diagnostic
(`experiments/scripts/diagnostic_eval_stage2.py`, locally on CPU) on the same checkpoint.

**The model is actually healthy.** Median per-label AUROC = 0.8366 (mean 0.8283), 87.3% of the 362
evaluable labels clear AUROC > 0.7. Pooled at threshold 0.5: precision 0.6565, recall 0.4583,
F1 0.5398 — nothing like the training log's reported 3.55% recall. Best pooled F1 over a threshold
sweep: 0.5727 @ threshold 0.32.

**Root cause, found and fixed**: `compute_metrics()` computed per-batch recall (`tp/(tp+fn)`) and
averaged those *ratios* across batches — mathematically fine for accuracy (a linear average
aggregates correctly regardless of grouping) but wrong for recall. With `batch_size=8` and 383 labels
at 3.58% positive rate, the expected positive count for any given label in an 8-example batch is
~0.29 — the overwhelming majority of (label, batch) pairs have **zero** true positives, so
`tp/(tp+fn+eps) = 0` for most labels in most batches *regardless of model quality*, dragging every
batch's — and therefore the epoch's — averaged recall toward 0. Confirmed with a synthetic test: a
model constructed to have true recall = 1.0 was reported as 25.7% under the old code, 100% under the
fix. Fixed in `training_run.py`: `compute_metrics()` now returns raw per-batch counts
(correct/total/tp/fn), and a new `finalize_metrics()` accumulates those across a full epoch before
computing the ratio once — applied to both the training and validation loops.

**Scope of the bug**: only ever affected `recall`, never `accuracy` or `val_loss` (both unaffected —
`val_loss` remains trustworthy across the entire sweep, old and new phases alike). Every prior trial
(all on the 3-label task) was only mildly affected, since 3 sparse-but-not-*this*-sparse labels rarely
hit zero-positives-per-8-example-batch the way 383 labels do — not worth re-running retroactively.
`val_recall`/`train_recall` from trial-051 onward are now computed correctly and can be trusted.

**Decision**: trial-050 stands as a genuinely healthy seed for this phase (val_loss 0.0772 best, real
AUROC ~0.83). Next: design a proper first beam-search round here, now that both the pipeline and its
metrics are trustworthy.

## pos_weight added, trial-051 (A/B against trial-050)

trial-050's diagnostic left one real (if moderate) issue on the table: best F1 came at
threshold=0.32, not the standard 0.5, and recall@0.5 was only 45.8% — unweighted BCE naturally
pushes predicted probabilities low when negatives outnumber positives 27-to-1 on average. Added a
`--pos-weight` flag to `training_run.py`: per-label `sqrt(n_negative/n_positive)`, computed from the
training split. Checked the real distribution before picking sqrt over raw inverse-frequency: raw
ratio spans 0.38–4780.7 across the 383 labels (~12,600x — would spike gradients whenever a rare
positive shows up in a batch), sqrt compresses that to 0.61–69.14 (~113x) while staying well-behaved
at both ends without needing a clamp (a naive `log2` version goes negative for the majority-positive
aspect roots, since their raw ratio is <1). Verified the tensor's shape/values and that
`BCEWithLogitsLoss` broadcasts it correctly before launching anything.

`trial-051` (`experiments/round8_manifest.json` / `experiments/round8_manual.ipynb`) is an isolated
A/B against trial-050 — identical config, only `--pos-weight` added — to get a clean read on whether
it actually helps before folding it into a broader beam-search round.

**First attempt crashed**: `Expected all tensors to be on the same device`. Root cause was in the
`compute_metrics()`/`finalize_metrics()` refactor from the trial-050 write-up, not the pos_weight
code — `train_metrics_acc`/`validation_metrics_acc` initialized their `tp`/`fn` accumulator tensors
with plain `torch.zeros(len(label_names))` (CPU by default), while `compute_metrics()`'s per-batch
outputs live on whatever device `logits`/`labels` are on (CUDA during real training). Missed locally
since this sandbox has no CUDA — every tensor happened to be CPU on both sides there, so the mismatch
never had a chance to appear. Fixed with `device=device` on both `torch.zeros(...)` calls, and
`round8_manual.ipynb` regenerated (it embeds `training_run.py`'s source directly) before resending.

## trial-051 result: negative for pos_weight as formulated

trial-051's own `val_loss` (0.24–0.27) and `val_recall` numbers are **not directly comparable** to
trial-050's: `pos_weight` changes the loss function's numeric scale (up to 69x multiplier on
rare-label positive mistakes), and trial-050's recall was computed under the old buggy metric anyway.
Ran the same AUROC diagnostic (`diagnostic_eval_stage2.py`, now parameterized by checkpoint revision)
on both checkpoints for an apples-to-apples read:

| | trial-050 (no pos_weight) | trial-051 (pos_weight) |
|---|---|---|
| median per-label AUROC | 0.8366 | 0.8456 |
| precision @ 0.5 | 0.6565 | 0.4724 |
| recall @ 0.5 | 0.4583 | 0.5286 |
| F1 @ 0.5 | 0.5398 | 0.4989 |
| best F1 (threshold sweep) | **0.5727** @ 0.32 | 0.5007 @ 0.48 |

`pos_weight`'s mechanism worked exactly as designed — median AUROC ticked up slightly (a real, if
small, ranking-quality improvement) and the optimal threshold recalibrated back toward 0.5 (0.32 →
0.48), confirming it counteracts the low-probability bias from class imbalance. But the
best-achievable F1 got **worse** (0.5727 → 0.5007): at threshold 0.5, recall rose but precision fell
harder, a net-negative trade for the metric that matters most practically. Both trials used identical
`seed=42`, so this isn't seed noise — `sqrt(n_neg/n_pos)` as formulated overcorrects on this dataset.

**Decision**: not adopting `pos_weight` as tried. Directions not yet tried: a gentler compression
(`ratio**0.3`, range 0.75–12.7 vs. sqrt's 0.61–69.14) or lighter regularization alongside it. But the
more promising, *cheaper* lever is sitting right there in trial-050's own numbers: post-hoc threshold
tuning (free at inference, no retraining) already recovers 0.5398 → 0.5727 F1 just by picking 0.32
instead of 0.5. The first proper beam-search round for this phase should use unweighted BCE +
threshold tuning as the baseline, not `pos_weight`.

## Round 1 for this phase (trial-052 through trial-057)

Generalized `--pos-weight` before designing the round: `training_run.py` now takes
`--pos-weight-exponent` (default 0.5), so `pos_weight = (n_negative/n_positive)**exponent` — the old
hardcoded `sqrt()` call is just the exponent=0.5 case. Verified exponent=0.5 exactly reproduces
trial-051's tensor via `torch.allclose` against the old code, and exponent=0.3 gives the gentler
0.75–12.70x range identified earlier (vs. 0.5's 0.61–69.14x).

Sole seed for this round is trial-050 — trial-051 underperformed on best-achievable F1 (0.5007 vs.
0.5727), so it's not used as a second beam member to branch off. 6 children, split into the standard
two 3-trial batches:

- **trial-052** (regularization): `pos_weight_exponent=0.3`, the gentler correction — does it recover
  trial-051's small AUROC gain (0.8366→0.8456) without its F1 cost (0.5727→0.5007)?
- **trial-053** (regularization): `weight_decay=0.2` — trial-050 inherited `wd=0.5`/`dropout=0.2`
  unchanged from the old aspect-task phase, untested whether that's actually right for this smaller,
  sparser, 383-label task.
- **trial-054**: `batch_size=32` — re-tests whether the old phase's biggest finding (batch≥64 beating
  batch=8 outside its noise floor) transfers here; trial-050 deliberately started conservative at
  batch=8.
- **trial-055**: `unfreeze_last_n=2` — capacity, untested at this scale/task.
- **trial-056**: `lr_schedule=cosine` — small real win in the old `esm2_t6_8M` phase, cheap to re-test.
- **trial-057**: `learning_rate=3e-4` — untested here, trial-050 used the inherited `2e-4`.

Part 1 (`experiments/round9a_manifest.json` / `experiments/round9a_manual.ipynb`, trial-052/053/054)
sent now; part 2 (`round9b_manifest.json`, trial-055/056/057) held until part 1 lands.

**Evaluation caveat carried into this round**: don't trust raw training-loop val_loss for
cross-config comparison against any trial that changes `pos_weight` settings — it changes the loss's
numeric scale (per the trial-050 vs. trial-051 lesson). trial-052 specifically needs the post-hoc
AUROC/threshold-sweep diagnostic, not a direct val_loss comparison; the other five trials are fine to
compare on val_loss directly.

### Round 1 part 1 results (trial-052/053/054)

All three completed cleanly, 10/10 epochs, checkpoints pushed to their own `trial-NNN` Hub branches.
Combined log arrived as a single `trial-052-054.log` (batched-run convention) — split into individual
`trial-052.log`/`trial-053.log`/`trial-054.log` to match `build_results_index.py`'s one-file-per-trial
parsing convention, and `results_index.csv` rebuilt.

- **trial-052** (`pos_weight_exponent=0.3`): ran the AUROC diagnostic
  (`experiments/scripts/diagnostic_eval_stage2.py trial-052`) for the required apples-to-apples read.
  Median AUROC=0.8473 (best of trial-050/051/052), pooled F1@0.5=0.5445 (beats trial-050's 0.5398 and
  trial-051's 0.4989), best F1 over a threshold sweep=0.5604@0.42 — clearly better than trial-051's
  0.5007 but still short of trial-050's plain-BCE best F1 of 0.5727. Confirms a monotonic trend: as
  the pos_weight exponent goes from 0.5→0.3 (gentler), F1 cost shrinks but so does the AUROC/threshold-
  calibration gain, approaching but not beating plain BCE. Since trial-050 already reaches 0.5727 for
  free via post-hoc threshold tuning (no retraining needed), pos_weight at either exponent tried isn't
  earning its cost for this task. Not carried forward into round 2.
- **trial-053** (`weight_decay=0.2`): best val_loss 0.0754 at epoch 7 — a clear, real improvement over
  trial-050's 0.0772, not confounded by any loss-scale change. trial-050's inherited `wd=0.5` (carried
  over unchanged from the old 3-aspect phase's baseline) was simply too strong for this task's smaller
  (12k example) training set. **New round-1 winner.**
- **trial-054** (`batch_size=32`): best val_loss 0.0754 at epoch 10 — ties trial-053's number, but
  hadn't plateaued yet (still falling at the last epoch) and each epoch took ~742s vs. trial-053's
  ~574s (~29% slower) despite 4x fewer optimizer steps/epoch — the *opposite* of the old
  `esm2_t12_35M_scale_up` phase's finding that larger batch sped up wall-clock. No demonstrated
  advantage over trial-053 within the fixed 10-epoch budget, and it costs more. Not carried forward.

**Part 2 re-parented before sending**: `round9b_manifest.json` (trial-055/056/057) was originally
designed branching off trial-050 (`weight_decay=0.5`). With trial-053 now a confirmed, un-confounded
improvement, running three more axes against the stale `wd=0.5` baseline would have wasted the
batch's ~5h GPU budget testing against a baseline already known to be worse. Edited the manifest
in-place before generating the notebook: parent trial-050→trial-053, `weight_decay` 0.5→0.2 for all
three trials, the axis under test in each is otherwise unchanged (`unfreeze_last_n=2`,
`lr_schedule=cosine`, `learning_rate=3e-4`). Regenerated and sent `round9b_manual.ipynb`.

Beam updated: trial-053 (val_loss=0.0754) replaces trial-050 as the sole seed going into round 2,
pending part 2's results.

### Round 1 part 2 results (trial-055/056/057) — and a notebook mismatch

All three completed cleanly, 10/10 epochs. **Their `[CONFIG]` lines show `weight_decay=0.5`, not the
`0.2` they were re-parented to** before `round9b_manual.ipynb` was sent — the run used the pre-pivot
design branching off trial-050, not trial-053. The committed/sent `.ipynb` was independently verified
to contain `weight_decay: "0.2"` for all three trials, so the mismatch traces to something on the
execution side (most likely a stale local copy predating the correction), not the file that was
delivered. Flagging this plainly rather than silently reinterpreting the results — they're still
valid data, just answering a different question than the one intended:

- **trial-055** (`unfreeze_last_n=2`, off trial-050/wd=0.5): best val_loss 0.0768 @ epoch 7 — mild
  improvement over trial-050 (0.0772), short of trial-053 alone (0.0754).
- **trial-056** (`lr_schedule=cosine`, off trial-050/wd=0.5): best val_loss 0.0762 @ epoch 9 — best of
  this batch, still short of trial-053 alone. The most promising secondary lever found independently
  so far.
- **trial-057** (`learning_rate=3e-4`, off trial-050/wd=0.5): best val_loss 0.0775 @ epoch 9 — ties
  trial-050 within noise, no real improvement.

**Round 1 final ranking** (val_loss for non-pos_weight trials, best-F1 diagnostic for pos_weight
trials): trial-053 (0.0754) ≈ trial-054 (0.0754, not adopted — unconverged + slower) > trial-056
(0.0762) > trial-055 (0.0768) > trial-050 (0.0772) > trial-057 (0.0775) > trial-052 (best F1 0.5604)
> trial-051 (best F1 0.5007). **trial-053 (`weight_decay=0.2`) is the confirmed round winner**,
promoted to round 2's sole seed.

## Round 2 (trial-058 through trial-063)

Seed: trial-053 (`weight_decay=0.2`). Since round 1 part 2's `unfreeze_last_n=2` and `lr_schedule=cosine`
tests never actually got stacked with `wd=0.2` as intended (the mismatch above), two of round 2's
slots re-run those exact stacked combinations properly, closing the loop rather than treating the
mismatch as wasted data — the standalone-at-wd=0.5 results already told us the ranking (cosine >
unfreeze=2 > baseline > higher-lr), so round 2 checks whether those gains compound with `wd=0.2` or
just overlap with it.

- **trial-058** (regularization): `weight_decay=0.1` — only two points on this axis tested so far
  (0.5, 0.2), both favoring lighter. Tests whether the trend continues.
- **trial-059** (regularization): `output_dropout=0.1` — dropout has sat at its inherited 0.2 the
  whole phase while weight_decay turned out to be too strong at its own inherited value; tests the
  same hypothesis on the other regularization knob.
- **trial-060**: `lr_schedule=cosine` properly stacked with `weight_decay=0.2` this time.
- **trial-061**: `unfreeze_last_n=2` properly stacked with `weight_decay=0.2` this time.
- **trial-062**: `learning_rate=3e-4` properly stacked with `weight_decay=0.2` — closing out the
  intended test, though a reversal seems unlikely given the null result at wd=0.5.
- **trial-063**: `warmup_steps=300` — fresh axis for this task; no improvement in the old
  `esm2_t12_35M_scale_up` phase, but that phase's conclusions aren't assumed to transfer.

Part 1 (`round10a_manifest.json`/`round10a_manual.ipynb`, trial-058/059/060) sent now; part 2
(`round10b_manifest.json`, trial-061/062/063) held until part 1 lands.

### Round 2 part 1 results (trial-058/059/060)

`[CONFIG]` lines verified correct this time — no repeat of round 1 part 2's stale-notebook mismatch.

- **trial-058** (`weight_decay=0.1`): best val_loss 0.0749 @ epoch 7 — edges out trial-053 (0.0754) by
  0.0005, continuing the lighter-weight_decay trend.
- **trial-059** (`output_dropout=0.1`): best val_loss 0.0751 @ epoch 8 — edges out trial-053 by 0.0003,
  the smallest gain of the batch.
- **trial-060** (`lr_schedule=cosine` + `weight_decay=0.2`): best val_loss 0.0748 @ epoch 9 — best of
  the batch. Confirms cosine's round 1 gain (found independently at wd=0.5, trial-056) genuinely
  compounds with wd=0.2 rather than just overlapping with it.

**All three margins are smaller than round 1's own margins**, which were themselves already close to
the old `esm2_t12_35M_scale_up` phase's directly-measured seed-noise floor (~0.003). These are real
results, but this round's ranking is provisional until trial-064/065/066 (the noise-floor check
running in parallel — trial-052/053/054's exact configs at `seed=123`) lands. Not picking a round-3
seed off these numbers alone. Part 2 (`round10b_manual.ipynb`, trial-061/062/063) sent.

### Noise-floor check results (trial-064/065/066)

CONFIG lines confirmed each trial is its seed=42 counterpart with only `seed` changed to 123.

| seed=42 (config) | val_loss | seed=123 (trial) | val_loss | delta |
|---|---|---|---|---|
| trial-052 (`pos_weight_exponent=0.3`) | 0.1438 | trial-064 | 0.1449 | 0.0011 |
| trial-053 (`weight_decay=0.2`) | 0.0754 | trial-065 | 0.0755 | **0.0001** |
| trial-054 (`batch_size=32`) | 0.0754 | trial-066 | 0.0762 | 0.0008 |

**Calibrated noise floor for this phase: ~0.0007–0.0011** (mean 0.00067, max 0.0011 across the 3
seed-pairs; n=3, so provisional, but usable) — notably tighter than the old
`esm2_t12_35M_scale_up` phase's ~0.003. trial-065 in particular (the noise sample for trial-053's own
config) came back almost identical to trial-053 (0.0755 vs. 0.0754), which is a strong direct
confirmation of that config's stability.

**Applying the floor**:
- Round 1's trial-053-vs-trial-050 margin (0.0018) **clears the floor — confirmed real.**
- Round 2 part 1's margins over trial-053 (trial-058: 0.0005, trial-059: 0.0003, trial-060: 0.0006)
  **all fall at or below the floor — none should be trusted as real improvements on their own.**

So round 2 part 1 effectively found nothing that beats trial-053 outside noise. Possible read: we're
near a plateau for the regularization/schedule axes tested so far, at this model size and 12k-example
training budget. Applying the same >0.001 bar to round 2 part 2 (trial-061/062/063, pending) once it
lands — if that also comes back within noise, the natural next move is to stop fine-grained axis
search here and treat trial-053 as this phase's config for a longer confirmation run, rather than
continuing to chase deltas smaller than the noise floor.

**Noise-floor check (trial-064/065/066)**: round 1's margins (trial-053 vs. trial-050: Δ0.0018;
trial-053 vs. trial-054: Δ~0.0000) are all smaller than the seed-to-seed noise measured directly in
the old `esm2_t12_35M_scale_up` phase (~0.003, trial-044/047) — a check never run for this phase. User
set up trial-064/065/066 directly: trial-052/053/054's exact configs rerun at `seed=123`. trial-065
(the trial-053 analog) is the one that matters most — if it doesn't land close to 0.0754, round 1's
ranking is noise-dominated and the round-2 design built on top of it needs re-examining. Running in
parallel with round 2 part 1, expected to land around the same time.
