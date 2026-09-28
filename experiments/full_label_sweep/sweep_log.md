# Full-Label-Space Regularization Sweep Log

Goal: find ESM-2 fine-tuning settings that train reliably on the full ancestor-closure
GO-term label space (`data/cafa-6-protein-function-prediction/dataset_full` /
`kelmoir/cafa6-full`, see `generate_dataset_full.py`), producing the first genuinely
plateau-tuned baseline for `experiments/cafa6_next_steps_plan.md`'s Tier 1 work. Each trial
runs locally on the user's GPU ([[hardware_local_gpu]]) via `training_run.py
--data-path data/cafa-6-protein-function-prediction/dataset_full`, handed to the user as an
exact `!`-prefixed command per [[feedback_no_long_cpu_runs_prefer_gpu]] (the sandbox has no
GPU device passthrough).

**Dataset specifics as of trial-001 (2026-09-28):** `MAX_LENGTH=4192` (raised from an initial
2048, itself raised from the shared 1024 default — see `generate_dataset_full.py`'s own
comments for the recovery-curve analysis behind each jump), **27,191-term trainable
vocabulary** (down from an initial 29,386 — `select_trainable_terms` drops any term with zero
examples among train-split proteins usable at `MAX_LENGTH`, since those can't produce
gradient signal under any architecture). Removed-protein fraction: 0.22%.

## Rubric

Same DEAD / OVERFITTING / HEALTHY / UNSTABLE judgment as
`experiments/ancestor_sweep/sweep_log.md` (see that file for the full rubric text and
corrective actions per verdict) — reused verbatim, not redefined here.

**Two differences from the coarse-space sweep, worth calling out explicitly:**
1. This label space is far more extreme in class imbalance (27,191 terms, most with very few
   positives, vs. 383 terms with a 50-annotation floor). `--run-diagnostic`'s pooled
   accuracy/recall at threshold 0.5 will look even more misleading here than it did at
   trial-050 in the coarse sweep (see CLAUDE.md's "Post-training diagnostic" note) — judge
   HEALTHY/DEAD/OVERFITTING from per-label AUROC distribution (mean/median) and pooled AUROC,
   not pooled accuracy/recall, from the first trial onward.
2. Report the **evaluable-label count** (`n_labels_evaluable` in the diagnostic's printed
   JSON) alongside every AUROC figure. `run_diagnostic` skips any label with only one class
   present in the validation slice — at `--diagnostic-max-val-samples 2000` and this label
   space's long tail (5,128 terms with exactly 1 train positive), a large share of labels
   will be un-evaluable in any given trial, so a low mean/median AUROC could reflect which
   labels happened to be evaluable, not real model quality. Don't compare two trials' AUROC
   numbers without also comparing their evaluable-label counts.

## Search strategy

Same coordinate-descent approach as `experiments/ancestor_sweep/sweep_log.md`: start from
that sweep's own eventual HEALTHY-config defaults as trial-001's starting point (rather than
re-deriving from scratch) — `esm2_t12_35M_UR50D`, `unfreeze_last_n=2`, `lr=5e-5`,
`output_dropout=0.2`, `esm_dropout=0.0`, `attention_dropout=0.0`, `grad_clip_norm=1.0`,
`weight_decay=0.01` — since this is a new label space on the same backbone/task family, not
a reason to assume DEAD-basin defaults again. Change one axis per trial; priority order:
(1) confirm the model learns at all on this much larger/sparser label space (per-label AUROC
mean clearly above 0.5, on a real evaluable-label count), (2) resolve over/underfitting,
(3) scale dataset size/epochs only once HEALTHY.

**Memory/throughput note specific to this sweep:** `MAX_LENGTH=4192` is ~4.1x
`dataset`/`dataset_raw`'s 1024, and self-attention memory scales roughly quadratically with
sequence length. User-determined (2026-09-28, empirical, own hardware testing): **small
physical `--batch-size` (1) + `--grad-accum-steps 16` (effective batch 16) is both the memory-
and throughput-optimal choice at this sequence length** -- matrix multiplications are actually
faster at smaller physical batch sizes here, and `--gradient-checkpointing` was tried and
rejected for greatly slowing training (trades compute for memory, and memory wasn't actually
the bottleneck once physical batch size was small). Don't reach for gradient checkpointing by
default at this scale going forward -- prefer shrinking physical batch size + raising
`--grad-accum-steps` to hit a target effective batch size instead.

## Trial index

| id | model | unfreeze | lr | out_drop | esm_drop | attn_drop | grad_clip | wd | epochs | max_len | batch | grad_accum | train_n | val_n | verdict | next_action |
|----|-------|----------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| trial-001 | esm2_t12_35M_UR50D | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 3 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, still improving | extend epochs (6) to find the plateau before touching unfreeze/model size |
| trial-002 | esm2_t12_35M_UR50D | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 6 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, val_loss flat but AUROC/F1 still rising | extend epochs (10), switch --checkpoint-selection to f1 |
| trial-003 | esm2_t12_35M_UR50D | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 10 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, plateau reached at this data volume | scale train_n up (more data, not more epochs) before touching unfreeze/model size |
| trial-004 | esm2_t12_35M_UR50D | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 6 | 4192 | 1 | 16 | 32000 | 4000 | HEALTHY, doubling train_n did not beat trial-003's plateau -- capacity-limited, not data-limited | raise --unfreeze-last-n (2->4), revert train_n to 16000 (32k showed no benefit) |
| trial-005 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 10 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, unfreezing more layers broke past trial-003's ceiling, still climbing at epoch 10 | extend epochs (15) to relocate the plateau at unfreeze=4 |
| trial-006 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 15 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, plateau forming but train/val f1 gap widening -- early overfitting warning | retry train_n=32000 at unfreeze=4 (more data may both help and narrow the gap) |
| trial-007 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 10 | 4192 | 1 | 16 | 32000 | 4000 | HEALTHY, best diagnostics of the sweep so far but overfitting onset tied to total gradient exposure, not resolved by more data | revert train_n to 16000 (cheaper, no proven gain), raise esm/attn dropout off 0.0 -- first real regularization-axis trial |

## trial-001

**Config**: `esm2_t12_35M_UR50D`, unfreeze=2, lr=5e-5, out_drop=0.2, esm_drop=0.0, attn_drop=0.0,
grad_clip=1.0, wd=0.01, epochs=3, max_length=4192, batch_size=1, grad_accum_steps=16 (effective
batch 16), train_n=16000, val_n=4000, seed=42, `--data-path dataset_full` / `kelmoir/cafa6-full`
(27,191-term trainable vocabulary). Log: `experiments/full_label_sweep/logs/trial-001.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 | wall_clock |
|-------|-----------|----------|----------|--------|------------|
| 1 | 0.0068 | 0.0069 | 0.2188 | 0.2359 | 561.6s |
| 2 | 0.0066 | 0.0067 | 0.2457 | 0.2490 | 559.8s |
| 3 | 0.0063 | 0.0066 | 0.2583 | 0.2526 | 567.7s |

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable (both classes present).
Per-label AUROC: mean=0.7923 median=0.8364 min=0.0028 max=1.0000 (8,668/9,401 labels >0.5,
7,207/9,401 >0.7). Pooled AUROC=0.9674. Best F1=0.3834@threshold=0.16. Pooled accuracy/recall
at threshold 0.5 (0.9984/0.1532) is not the judging signal here, per this log's own note above
-- only reported for completeness.

**Verdict: HEALTHY.** Train/val loss decrease together every epoch, gap stays small and
constant (0.0001-0.0003), no divergence, no spikes/NaN -- rules out OVERFITTING and UNSTABLE.
Mean/median per-label AUROC well above 0.5 with strong tail performance (77% of evaluable
labels >0.7 AUROC) rules out DEAD, despite this being a much larger, sparser vocabulary
(27,191 terms) than any prior sweep tackled.

**Next action**: per [[feedback_plateau_first_sweep_strategy]], loss was still decreasing at
epoch 3 with no sign of flattening -- extend epochs (e.g. 6) at this same cheap
scale/data-volume to locate the actual plateau before changing `--unfreeze-last-n` or model
size. Not yet at a decision point for model-size/data-volume scaling.

**Also noted**: `[WARN] Final checkpoint push failed: cannot import name 'XetSession' from
'hf_xet'` -- fixed in `training_run.py` (commit `823cf46`, `os.environ.setdefault` before
`import huggingface_hub`) after this trial ran; not expected to recur in trial-002.

## trial-002

**Config**: identical to trial-001 except `--epochs 6` (was 3). Log:
`experiments/full_label_sweep/logs/trial-002.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 | wall_clock |
|-------|-----------|----------|----------|--------|------------|
| 1 | 0.0068 | 0.0069 | 0.2188 | 0.2359 | 564.3s |
| 2 | 0.0066 | 0.0067 | 0.2457 | 0.2490 | 564.4s |
| 3 | 0.0063 | 0.0066 | 0.2583 | 0.2526 | 563.2s |
| 4 | 0.0062 | 0.0065 | 0.2639 | 0.2555 | 571.5s |
| 5 | 0.0061 | 0.0065 | 0.2759 | 0.2731 | 579.0s |
| 6 | 0.0061 | 0.0064 | 0.2789 | 0.2919 | 565.2s |

Epochs 1-3 are bit-for-bit identical to trial-001's epochs 1-3, as expected (same seed, same
data ordering, same hyperparameters up to that point -- confirms determinism, not a bug).

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable (same set as trial-001,
depends on the validation sample, not epoch count). Per-label AUROC: mean=0.8225
median=0.8659 min=0.0000 max=1.0000 (8,903/9,401 labels >0.5, 7,704/9,401 >0.7). Pooled
AUROC=0.9717. Best F1=0.3937@threshold=0.20 -- every one of these numbers improved over
trial-001's (mean AUROC 0.7923->0.8225, median 0.8364->0.8659, pooled 0.9674->0.9717,
>0.7-AUROC share 76.7%->81.9%).

**Notable pattern (not a bug, already documented in CLAUDE.md's checkpoint-selection notes)**:
`val_loss` essentially flattened over the last 3 epochs (0.0065/0.0065/0.0064) while AUROC and
F1 kept improving meaningfully the whole time (val_f1 0.2526->0.2555->0.2731->0.2919). Under
this task's severe class imbalance, BCE loss barely moves once the easy majority-negative
predictions are dialed in, even while real ranking quality on the rare positive terms keeps
improving -- so `--checkpoint-selection val_loss` (this trial's setting) would treat epochs
4-6 as roughly equally "best" despite epoch 6 clearly being the strongest on the metric that
actually matters here.

**No overfitting signal**: train/val loss gap stayed small and roughly constant; val_f1 was
actually *higher* than train_f1 at every epoch, the opposite of what overfitting would show.

**Verdict: still HEALTHY, real plateau not yet reached** -- AUROC/F1 (the metrics this sweep
judges by, per this log's own note above) were still climbing meaningfully at epoch 6, even
though val_loss looked flat. Per [[feedback_plateau_first_sweep_strategy]]: extend epochs
further (10) before touching unfreeze/model size, and switch `--checkpoint-selection f1` for
trial-003 onward given the flat-val-loss-but-rising-F1 dynamic just confirmed empirically on
this label space, not just the coarse one CLAUDE.md originally documented it for.

## trial-003

**Config**: identical to trial-002 except `--epochs 10` (was 6) and
`--checkpoint-selection f1` (was the `val_loss` default). Log:
`experiments/full_label_sweep/logs/trial-003.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 | wall_clock |
|-------|-----------|----------|----------|--------|------------|
| 1 | 0.0068 | 0.0069 | 0.2188 | 0.2359 | -- |
| 2 | 0.0066 | 0.0067 | 0.2457 | 0.2490 | -- |
| 3 | 0.0063 | 0.0066 | 0.2583 | 0.2526 | -- |
| 4 | 0.0062 | 0.0065 | 0.2639 | 0.2555 | -- |
| 5 | 0.0061 | 0.0065 | 0.2759 | 0.2731 | -- |
| 6 | 0.0061 | 0.0064 | 0.2789 | 0.2919 | -- |
| 7 | 0.0060 | 0.0064 | 0.2876 | 0.2692 | -- |
| 8 | 0.0059 | 0.0063 | 0.2899 | 0.2955 | -- |
| 9 | 0.0058 | 0.0063 | 0.2955 | 0.2888 | -- |
| 10 | 0.0058 | 0.0063 | 0.2990 | 0.2896 | -- |

Epochs 1-6 are bit-for-bit identical to trial-002's epochs 1-6 (same seed/data/hyperparameters
up to that point, as expected). `--checkpoint-selection f1` picked **epoch 8** (val_f1=0.2955)
as best, not epoch 10 -- the first trial in this sweep where the selected checkpoint isn't the
final epoch, confirming the switch away from `val_loss` selection was warranted.

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable (same set as trial-001/002).
Per-label AUROC: mean=0.8356 median=0.8791 min=0.0110 max=1.0000. Pooled AUROC=0.9734. Best
F1=0.3993@threshold=0.20. Still an improvement over trial-002 (mean AUROC 0.8225->0.8356,
median 0.8659->0.8791, pooled 0.9717->0.9734), but a much smaller step than trial-002's jump
over trial-001.

**Plateau signature found**: unlike trial-002's smooth monotonic `val_f1` climb through all 6
epochs, trial-003's epochs 6-10 **oscillate** around 0.27-0.30 (0.2919 -> 0.2692 -> 0.2955 ->
0.2888 -> 0.2896) rather than climbing every epoch. The per-epoch rate of mean-AUROC
improvement also slowed sharply: +0.0302 over epochs 3->6 (trial-002's extension) vs. only
+0.0131 over epochs 6->10 (trial-003's extension), despite the same epoch budget. `val_loss`
stayed flat at 0.0063-0.0064 across the last 4 epochs. This combination -- noisy oscillation
around a ceiling instead of a still-climbing curve, on both the loss and the ranking metrics --
is the actual plateau shape this sweep has been hunting for, not just a flat-loss/rising-F1
false plateau like trial-002's.

**No overfitting signal**: train/val loss gap stayed small and roughly constant throughout;
val_f1 tracked train_f1 closely (sometimes exceeding it), the opposite of an overfitting
signature.

**Verdict: HEALTHY, plateau reached at this data volume (16k train / 4k val samples/epoch).**
Per [[feedback_plateau_first_sweep_strategy]]'s own logic -- find the plateau at cheap scale
before scaling up -- this is the decision point: further epochs at 16k samples aren't likely to
move the needle further (the oscillation is now bigger than the trend). Per the sweep's own
search-strategy priority order ("(3) scale dataset size/epochs only once HEALTHY"), the next
trial should scale **train_n** (more data per epoch, since 16k is a small slice of the ~52.7k
protein train split), not add more epochs at the same data volume or touch unfreeze/model size
yet.

**Next action**: trial-004, same config as trial-003 but with a larger `--max-train-samples`
(e.g. the full ~52,579-row train split, or an intermediate step like 32000 first) and enough
epochs to re-locate the plateau at that new scale -- per this sweep's own strategy, re-derive
the plateau at the new scale rather than assuming trial-003's epoch count transfers.

## trial-004

**Config**: identical to trial-003 except `--max-train-samples 32000` (was 16000) and
`--epochs 6` (was 10, since the plan was to re-locate the plateau at the new data volume rather
than assume trial-003's epoch count). Log: `experiments/full_label_sweep/logs/trial-004.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 |
|-------|-----------|----------|----------|--------|
| 1 | 0.0067 | 0.0067 | 0.2327 | 0.2653 |
| 2 | 0.0063 | 0.0065 | 0.2617 | 0.2617 |
| 3 | 0.0061 | 0.0064 | 0.2774 | 0.3010 |
| 4 | 0.0060 | 0.0063 | 0.2866 | 0.2801 |
| 5 | 0.0058 | 0.0063 | 0.2960 | 0.2920 |
| 6 | 0.0057 | 0.0062 | 0.3073 | 0.2825 |

`--checkpoint-selection f1` picked **epoch 3** (val_f1=0.3010) as best -- `val_f1` was already
noisy by epoch 2-3, never showing trial-002's early smooth monotonic climb despite twice the
per-epoch data.

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable (same set every trial).
Per-label AUROC: mean=0.8302 median=0.8740. Pooled AUROC=0.9723. Best F1=0.3941@threshold=0.20.
**Worse than trial-003** on every one of these figures (mean AUROC 0.8356->0.8302, pooled
0.9734->0.9723, best F1 0.3993->0.3941) despite double the train_n.

**No overfitting signal**: train/val gap stayed small throughout; final train_f1 (0.3073) only
modestly exceeds val_f1 (0.2825), consistent with capacity-limited rather than
overfitting-limited behavior.

**Verdict: HEALTHY, but scaling train_n did NOT beat trial-003's plateau.** Doubling per-epoch
data volume produced a *slightly worse* result on every diagnostic metric than trial-003's
16k/10-epoch run, at greater compute cost per epoch. Combined with trial-003's finding (val_f1
oscillation, not a climbing trend, once near this ceiling) and the fact that both trials
converge to essentially the same AUROC/F1 band regardless of data volume, this points at a
**capacity bottleneck, not a data-volume bottleneck**: only the classifier head plus the last 2
encoder layers (`--unfreeze-last-n 2`) are trainable, and that's plausibly saturated for a
27,191-term label space already.

**Next action**: per the sweep's coordinate-descent search strategy, move to the capacity axis
next -- trial-005 should raise `--unfreeze-last-n` (2 -> 4), reverting `--max-train-samples`
back to 16000 (32k showed no benefit, no reason to pay its extra wall-clock cost while probing
this axis) and keeping `--epochs 10` (trial-003's already-located plateau epoch count for this
data volume) as the starting point.

## trial-005

**Config**: identical to trial-003 except `--unfreeze-last-n 4` (was 2). Log:
`experiments/full_label_sweep/logs/trial-005.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 |
|-------|-----------|----------|----------|--------|
| 1 | 0.0068 | 0.0069 | 0.2261 | 0.2485 |
| 2 | 0.0065 | 0.0067 | 0.2529 | 0.2495 |
| 3 | 0.0062 | 0.0065 | 0.2641 | 0.2569 |
| 4 | 0.0061 | 0.0065 | 0.2714 | 0.2613 |
| 5 | 0.0060 | 0.0064 | 0.2856 | 0.2798 |
| 6 | 0.0060 | 0.0063 | 0.2902 | 0.2901 |
| 7 | 0.0058 | 0.0063 | 0.3016 | 0.2778 |
| 8 | 0.0057 | 0.0063 | 0.3081 | 0.2966 |
| 9 | 0.0056 | 0.0063 | 0.3193 | 0.2858 |
| 10 | 0.0055 | 0.0063 | 0.3264 | 0.3001 |

`--checkpoint-selection f1` picked **epoch 10** (val_f1=0.3001) as best -- unlike trial-003,
where the best epoch was 8 and the last 4-5 epochs oscillated without net gain, here `val_f1`
is noisy epoch-to-epoch (small dips at 2, 7, 9) but its *trend* keeps climbing, ending at its
highest value yet on the final epoch.

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable (same set every trial).
Per-label AUROC: mean=0.8476 median=0.8904 min not recorded here. Pooled AUROC=0.9744. Best
F1=0.4027@threshold=0.18. **Beats trial-003 on every figure** (mean AUROC 0.8356->0.8476,
median 0.8791->0.8904, pooled 0.9734->0.9744, best F1 0.3993->0.4027) -- the first trial in
this sweep to clear the ceiling trial-003/004 converged to regardless of data volume.

**No overfitting signal**: train/val loss gap stayed small and constant; val_f1 sometimes dips
below train_f1's trend but never diverges, consistent with a still-improving fit rather than
memorization.

**Verdict: HEALTHY, confirms the capacity-bottleneck hypothesis from trial-004.** Unfreezing 2
more encoder layers (2->4) raised the achievable ceiling on every diagnostic metric, and unlike
trial-003/004, this trial is still climbing (not oscillating around a fixed ceiling) at epoch
10 -- the plateau for `--unfreeze-last-n 4` hasn't been located yet.

**Next action**: per [[feedback_plateau_first_sweep_strategy]], extend epochs further (e.g. 15)
at this same `--unfreeze-last-n 4` / 16k-train_n scale to relocate the new plateau, mirroring
the trial-001->002->003 epoch-extension pattern that worked for `--unfreeze-last-n 2`. Only
after that plateau is found does it make sense to decide whether to push `--unfreeze-last-n`
even higher, scale train_n again (now that capacity is less constrained), or move to a larger
model.

## trial-006

**Config**: identical to trial-005 except `--epochs 15` (was 10). Log:
`experiments/full_label_sweep/logs/trial-006.out`.

**Per-epoch metrics** (epochs 1-10 bit-for-bit identical to trial-005, as expected; only
epochs 11-15 shown beyond that point):

| epoch | train_loss | val_loss | train_f1 | val_f1 |
|-------|-----------|----------|----------|--------|
| 11 | 0.0054 | 0.0063 | 0.3385 | 0.2897 |
| 12 | 0.0054 | 0.0062 | 0.3465 | 0.3105 |
| 13 | 0.0053 | 0.0062 | 0.3538 | 0.3049 |
| 14 | 0.0051 | 0.0062 | 0.3661 | 0.3169 |
| 15 | 0.0051 | 0.0063 | 0.3732 | 0.3030 |

`--checkpoint-selection f1` picked **epoch 14** (val_f1=0.3169) as best; epoch 15 dipped.

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable (same set every trial).
Per-label AUROC: mean=0.8563 median=0.9002. Pooled AUROC=0.9758. Best F1=0.4059@threshold=0.20.
Still an improvement over trial-005 (mean AUROC 0.8476->0.8563, median 0.8904->0.9002, pooled
0.9744->0.9758, best F1 0.4027->0.4059), but a noticeably smaller step than trial-005's jump
over trial-003 -- consistent with a plateau starting to form at this capacity level too.

**Emerging overfitting signal (new, not seen in trial-001 through -005)**: the train/val F1 gap
widened steadily through these 5 epochs -- 0.0263 at epoch 10 (0.3264 vs 0.3001) growing to
0.0702 by epoch 15 (0.3732 vs 0.3030). `train_loss` kept dropping smoothly (0.0055->0.0051)
while `val_loss` flattened and ticked back up slightly at the last epoch (0.0062->0.0063). Not
severe (val_loss isn't clearly diverging, val_f1 isn't collapsing), but the first early warning
of overfitting in this sweep -- worth flagging now rather than waiting for it to worsen with
more epochs at this same data volume.

**Verdict: HEALTHY, plateau forming for `--unfreeze-last-n 4`, but with an early overfitting
warning this sweep hasn't seen before.** Diagnostic metrics still improved, but the shrinking
step size plus the widening train/val gap together suggest more epochs at 16k train_n is the
wrong lever to pull next -- it would likely trade a little more val_f1 for a worse overfitting
profile, not find genuinely new signal.

**Next action**: rather than extending epochs further at this capacity/data combination, retry
`--max-train-samples 32000` now at `--unfreeze-last-n 4` (trial-004 tested 32k only at
unfreeze=2, where it didn't help -- untested whether more data helps once capacity is less
constrained). More data is also the standard remedy for a widening train/val gap, so this trial
doubles as both an axis probe and a check on the overfitting signal above.

## trial-007

**Config**: identical to trial-006 except `--max-train-samples 32000` (was 16000) and
`--epochs 10` (was 15 -- reset conservatively since trial-006's overfitting signal only
appeared past epoch 10, mirroring how trial-004 reset epoch count the first time train_n
changed). Log: `experiments/full_label_sweep/logs/trial-007.out`.

**Per-epoch metrics** (fresh run -- train_n changed, so no epochs are bit-for-bit identical to
a prior trial):

| epoch | train_loss | val_loss | train_f1 | val_f1 | gap (train-val) |
|-------|-----------|----------|----------|--------|------------------|
| 1 | 0.0067 | 0.0067 | 0.2387 | 0.2676 | -0.0289 |
| 2 | 0.0062 | 0.0065 | 0.2677 | 0.2711 | -0.0034 |
| 3 | 0.0060 | 0.0063 | 0.2855 | 0.2994 | -0.0139 |
| 4 | 0.0058 | 0.0063 | 0.2988 | 0.2892 | 0.0096 |
| 5 | 0.0056 | 0.0062 | 0.3153 | 0.3014 | 0.0139 |
| 6 | 0.0055 | 0.0062 | 0.3347 | 0.2811 | 0.0536 |
| 7 | 0.0053 | 0.0062 | 0.3520 | 0.3058 | 0.0462 |
| 8 | 0.0051 | 0.0062 | 0.3685 | 0.3107 | 0.0578 |
| 9 | 0.0050 | 0.0062 | 0.3854 | 0.3254 | 0.0600 |
| 10 | 0.0049 | 0.0063 | 0.4010 | 0.3136 | 0.0874 |

`--checkpoint-selection f1` picked **epoch 9** (val_f1=0.3254) as best; epoch 10 dipped, same
pattern as trial-006's final epoch.

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable (same set every trial).
Per-label AUROC: mean=0.8568 median=0.9025. Pooled AUROC=0.9755. Best F1=0.4096@threshold=0.20.
Best-of-sweep on every figure except pooled AUROC, which ticked down within noise (trial-006:
mean 0.8563, median 0.9002, pooled 0.9758, best F1 0.4059 -> trial-007: 0.8568, 0.9025, 0.9755,
0.4096).

**Overfitting onset tracks total gradient exposure, not epoch count or train_n alone (new
finding)**: a 32k-sample epoch does 2x the gradient steps of a 16k-sample epoch, so "epoch
number" isn't a fair comparison across trial-006 and trial-007 -- total examples processed is.
At matched exposure (trial-007 epoch 5 = 160k examples processed ~= trial-006 epoch 10 = 160k
examples processed), trial-007's train/val gap is 0.0139 vs. trial-006's 0.0263 at that same
point -- doubling train_n roughly halved the gap early on, as hoped. But the gap kept widening
past that: by trial-007 epoch 9 (288k examples processed) the gap is 0.0600, comparable to
trial-006's epoch 15 gap of 0.0702 at 240k examples processed -- a similar order of magnitude at
a similar total-exposure budget. `best_epoch=9` (not the final epoch) confirms the same
plateau-then-regress signature trial-006 showed, now reached at a similar total-gradient-exposure
point rather than a later one. Conclusion: more data delayed the overfitting onset when measured
in epochs (because each epoch does more work), but did not raise the total-exposure budget
before it sets in -- scaling `--max-train-samples` further is unlikely to fix this on its own.

**Verdict: HEALTHY, best diagnostics of the sweep so far, but confirms the overfitting signal
from trial-006 is a real ceiling tied to total training volume rather than a data-scarcity
artifact.** The data axis has now been tested twice (trial-004 at unfreeze=2, trial-007 at
unfreeze=4) without ever clearly buying a better generalization profile -- time to stop pulling
it and move to the regularization axis the sweep is nominally about.

**Next action**: revert `--max-train-samples` to 16000 (cheaper; 32k bought better peak metrics
but not a better overfitting profile) and raise `--esm-dropout`/`--attn-dropout` off their 0.0
floor for the first time this sweep, holding `--unfreeze-last-n 4` and `--epochs 15` (matches
trial-006 exactly except for the new dropout values, to isolate their effect cleanly).
