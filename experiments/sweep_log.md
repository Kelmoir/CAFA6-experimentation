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
