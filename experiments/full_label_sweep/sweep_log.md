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

| id | model | unfreeze | lr | out_drop | esm_drop | attn_drop | grad_clip | wd | lbl_smooth | epochs | max_len | batch | grad_accum | train_n | val_n | verdict | next_action |
|----|-------|----------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| trial-001 | esm2_t12_35M_UR50D | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 0.0 | 3 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, still improving | extend epochs (6) to find the plateau before touching unfreeze/model size |
| trial-002 | esm2_t12_35M_UR50D | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 0.0 | 6 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, val_loss flat but AUROC/F1 still rising | extend epochs (10), switch --checkpoint-selection to f1 |
| trial-003 | esm2_t12_35M_UR50D | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 0.0 | 10 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, plateau reached at this data volume | scale train_n up (more data, not more epochs) before touching unfreeze/model size |
| trial-004 | esm2_t12_35M_UR50D | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 0.0 | 6 | 4192 | 1 | 16 | 32000 | 4000 | HEALTHY, doubling train_n did not beat trial-003's plateau -- capacity-limited, not data-limited | raise --unfreeze-last-n (2->4), revert train_n to 16000 (32k showed no benefit) |
| trial-005 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 0.0 | 10 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, unfreezing more layers broke past trial-003's ceiling, still climbing at epoch 10 | extend epochs (15) to relocate the plateau at unfreeze=4 |
| trial-006 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 0.0 | 15 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, plateau forming but train/val f1 gap widening -- early overfitting warning | retry train_n=32000 at unfreeze=4 (more data may both help and narrow the gap) |
| trial-007 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 0.0 | 10 | 4192 | 1 | 16 | 32000 | 4000 | HEALTHY, best diagnostics of the sweep so far but overfitting onset tied to total gradient exposure, not resolved by more data | revert train_n to 16000 (cheaper, no proven gain), raise esm/attn dropout off 0.0 -- first real regularization-axis trial |
| trial-008 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.1 | 0.1 | 1.0 | 0.01 | 0.0 | 15 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, overfitting signal fully resolved, matches trial-007's diagnostics at half the data cost, still climbing at epoch 15 | add --label-smoothing 0.1 (user's usual value on this problem), same dropout/epoch settings |
| trial-009 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.1 | 0.1 | 1.0 | 0.01 | 0.1 | 15 | 4192 | 1 | 16 | 16000 | 4000 | DEAD -- label_smoothing=0.1 collapsed ranking ability (mean AUROC 0.856->0.513, near chance) | revert label_smoothing to 0.0, resume trial-008's config (proven good), extend epochs |
| trial-010 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.1 | 0.1 | 1.0 | 0.01 | 0.0 | 25 | 4192 | 1 | 16 | 16000 | 4000 | HEALTHY, best diagnostics of the sweep, but mild overfitting onset appears past epoch ~18 | decide: extend further to confirm true plateau, or accept ~epoch 18-20 as this config's ceiling and move to next axis |
| trial-011 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.1 | 0.1 | 1.0 | 0.05 | 0.0 | 20 | 4192 | 1 | 16 | 16000 | 4000 | NEGATIVE/NEUTRAL -- weight_decay=0.05 doesn't help, diagnostics slightly below trial-008/010, genuine early peak (epoch 18) followed by clean decline | revert weight_decay to 0.01, move to next axis (train_n) |
| trial-012 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.1 | 0.1 | 1.0 | 0.01 | 0.0 | 20 | 4192 | 2 | 8 | 32000 | 4000 | POSITIVE -- doubling train_n improves all diagnostics over trial-010 (mean AUROC +0.0060, median +0.0080, pooled +0.0005, best F1 +0.0059) and reaches a higher val_f1 peak earlier (epoch 15 vs trial-010's epoch 25) | push train_n further (e.g. 64000) to see if the win continues |
| trial-013 | esm2_t12_35M_UR50D | 4 | 5e-5 | 0.2 | 0.1 | 0.1 | 1.0 | 0.01 | 0.0 | 30 | 4192 | 2 | 8 | full (~52.7k) | 4000 | HEALTHY, best diagnostics of the sweep (mean AUROC 0.8731, median 0.9259, best val_f1 0.3741@epoch23) -- user ran 30 epochs intending this as the sweep-closing baseline | pending: promote this checkpoint (push-to-hub) as the Tier-0 baseline for Tier-1 work |

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
but not a better overfitting profile) and raise `--esm-dropout`/`--attention-dropout` off their
0.0 floor for the first time this sweep, holding `--unfreeze-last-n 4` and `--epochs 15` (matches
trial-006 exactly except for the new dropout values, to isolate their effect cleanly).

## trial-008

**Config**: identical to trial-006 except `--esm-dropout 0.1` and `--attention-dropout 0.1`
(both were 0.0 through trial-007; note the correct flag is `--attention-dropout`, not
`--attn-dropout` as first drafted -- fixed before the trial ran). Log:
`experiments/full_label_sweep/logs/trial-008.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 | gap (train-val) |
|-------|-----------|----------|----------|--------|------------------|
| 1 | 0.0068 | 0.0069 | 0.2224 | 0.2494 | -0.0270 |
| 2 | 0.0066 | 0.0067 | 0.2492 | 0.2532 | -0.0040 |
| 3 | 0.0063 | 0.0066 | 0.2581 | 0.2592 | -0.0011 |
| 4 | 0.0062 | 0.0065 | 0.2641 | 0.2602 | 0.0039 |
| 5 | 0.0061 | 0.0065 | 0.2773 | 0.2798 | -0.0025 |
| 6 | 0.0061 | 0.0064 | 0.2801 | 0.2859 | -0.0058 |
| 7 | 0.0060 | 0.0064 | 0.2875 | 0.2788 | 0.0087 |
| 8 | 0.0059 | 0.0063 | 0.2914 | 0.2995 | -0.0081 |
| 9 | 0.0058 | 0.0063 | 0.2956 | 0.2896 | 0.0060 |
| 10 | 0.0058 | 0.0062 | 0.2985 | 0.2991 | -0.0006 |
| 11 | 0.0057 | 0.0063 | 0.3060 | 0.2924 | 0.0136 |
| 12 | 0.0057 | 0.0062 | 0.3087 | 0.3073 | 0.0014 |
| 13 | 0.0056 | 0.0062 | 0.3133 | 0.3035 | 0.0098 |
| 14 | 0.0055 | 0.0062 | 0.3192 | 0.3071 | 0.0121 |
| 15 | 0.0055 | 0.0062 | 0.3226 | 0.3120 | 0.0106 |

`--checkpoint-selection f1` picked **epoch 15** (val_f1=0.3120) as best -- the final epoch, still
climbing, no dip. First trial at `--unfreeze-last-n 4` where `best_epoch` lands on the last
epoch rather than regressing before it.

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable. Per-label AUROC: mean=0.8560
median=0.9021. Pooled AUROC=0.9762. Best F1=0.4095@threshold=0.20. Essentially matches trial-007's
peak diagnostics (mean 0.8568, median 0.9025, pooled 0.9755, best F1 0.4096) -- at half trial-007's
data cost (16k vs 32k train_n) and without trial-007's late-epoch regression.

**Overfitting signal fully resolved**: the train/val F1 gap stayed small and non-monotonic
throughout all 15 epochs (roughly -0.03 to +0.014, no directional trend), a sharp contrast with
trial-006's steady widening (0.0263 at epoch 10 -> 0.0702 at epoch 15) under identical
data/capacity settings. `esm-dropout`/`attention-dropout` at 0.1 each directly targeted the
mechanism (the unfrozen encoder layers memorizing) and it worked cleanly, at effectively no cost
to peak diagnostic quality.

**Verdict: HEALTHY, first fully clean result at `--unfreeze-last-n 4`** -- no overfitting
signal, matches the best diagnostics seen so far, and still improving at the final epoch (not yet
at this config's plateau).

**Next action**: per the user's instruction (2026-09-28: "label smoothing worked really well on
this problem... the next should consider that axis"), add `--label-smoothing 0.1` (the user's
usual value) on top of trial-008's settings, holding everything else fixed (same dropouts, same
15 epochs) to isolate its effect cleanly. Since trial-008 was still climbing at epoch 15, epoch
extension is also owed to this config -- queued for whichever trial comes after label smoothing
is evaluated, rather than confounding both changes in the same trial.

## trial-009

**Config**: identical to trial-008 except `--label-smoothing 0.1` (was 0.0). Log:
`experiments/full_label_sweep/logs/trial-009.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 |
|-------|-----------|----------|----------|--------|
| 1 | 0.2330 | 0.0555 | 0.1915 | 0.1575 |
| 2 | 0.2032 | 0.0576 | 0.1892 | 0.1923 |
| 3 | 0.2030 | 0.0569 | 0.1917 | 0.1910 |
| 4 | 0.2029 | 0.0562 | 0.1893 | 0.1760 |
| 5 | 0.2028 | 0.0581 | 0.1927 | 0.1760 |
| 6 | 0.2027 | 0.0550 | 0.1918 | 0.1760 |
| 7 | 0.2026 | 0.0568 | 0.1896 | 0.1760 |
| 8 | 0.2026 | 0.0578 | 0.1908 | 0.1923 |
| 9 | 0.2025 | 0.0571 | 0.1921 | 0.1760 |
| 10 | 0.2025 | 0.0560 | 0.1938 | 0.1760 |
| 11 | 0.2024 | 0.0565 | 0.1926 | 0.1760 |
| 12 | 0.2024 | 0.0561 | 0.1920 | 0.1760 |
| 13 | 0.2024 | 0.0567 | 0.1877 | 0.1760 |
| 14 | 0.2023 | 0.0557 | 0.1924 | 0.1760 |
| 15 | 0.2023 | 0.0579 | 0.1937 | 0.1760 |

(`train_loss`/`val_loss` aren't comparable to trial-008's -- BCE against smoothed 0.1/0.9
targets has a much higher floor than against hard 0/1 targets, so the ~0.20 vs ~0.055 scale
jump is expected and not itself a signal.) `--checkpoint-selection f1` picked **epoch 2**;
`val_f1` is bit-for-bit identical (0.1760) across 8 of the remaining 13 epochs -- the training
loop found a degenerate, static output distribution almost immediately and never moved off it.

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable (same set every trial).
Per-label AUROC: mean=**0.5133** median=**0.5304** -- both essentially at chance (0.5 = random
ranking). Pooled AUROC collapsed to 0.8814 (from trial-008's 0.9762). Best F1=0.3104@0.18 looks
survivable in isolation, but the AUROC collapse means the model has lost almost all per-label
ranking signal underneath it -- the exact false-alarm pattern this sweep's own diagnostic
tooling exists to catch (see CLAUDE.md's "Post-training diagnostic" section): a single
threshold-0.5-adjacent metric can look tolerable while the ranking model underneath has
effectively died.

**Label smoothing at this scale/value is destructive, not mildly unhelpful**: the old
`ancestor_sweep`'s trial-020 (coarse, 383-term label space) found `label_smoothing=0.05` only
*slightly* underperformed its seed. At the full 27,191-term ancestor-closure label space, with
~3.58% average positive rate, symmetric smoothing at 0.1 pushes essentially every one of a
protein's ~27k mostly-negative targets from a hard 0 up to 0.1 -- at this label-space size that
soft-negative pressure appears to dominate the loss landscape and the model gives up on
per-label discrimination almost immediately (`best_epoch=2`, then flat). This is a much larger
and more label-space-size-sensitive effect than the mild one seen in the old coarse-label sweep.

**Verdict: DEAD.** Ranking ability (AUROC) collapsed to near-chance within 2 epochs and never
recovered; not a case of "underperforms but still healthy," but the pattern CLAUDE.md's
diagnostic section was written to catch: don't trust threshold-0.5 metrics alone at this severe
class imbalance.

**Next action**: revert `--label-smoothing` to 0.0 and resume from trial-008's proven-good
config (esm/attention-dropout=0.1, no label smoothing) -- extend `--epochs` past 15 there, since
trial-008 was still climbing at its final epoch and that thread was interrupted to test label
smoothing. If label smoothing is worth revisiting at all at this label-space scale, it would
need a far smaller value (e.g. 0.01) or the existing `--unknown-target-value` flag (soft-targets
only the *unknown* side, not confirmed negatives) rather than full symmetric smoothing at 0.1 --
flagged for the user to decide, not attempted unprompted.

## trial-010

**Config**: identical to trial-008 (`--esm-dropout 0.1 --attention-dropout 0.1`,
`--label-smoothing` reverted to 0.0/unset) except `--epochs 25` (was 15) -- resuming the epoch-
extension thread that trial-009's label-smoothing detour interrupted. Log:
`experiments/full_label_sweep/logs/trial-010.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 | gap (train-val) |
|-------|-----------|----------|----------|--------|------------------|
| 1 | 0.0068 | 0.0069 | 0.2224 | 0.2494 | -0.0270 |
| 2 | 0.0066 | 0.0067 | 0.2492 | 0.2532 | -0.0040 |
| 3 | 0.0063 | 0.0066 | 0.2581 | 0.2592 | -0.0011 |
| 4 | 0.0062 | 0.0065 | 0.2641 | 0.2602 | 0.0039 |
| 5 | 0.0061 | 0.0065 | 0.2773 | 0.2798 | -0.0025 |
| 6 | 0.0061 | 0.0064 | 0.2801 | 0.2859 | -0.0058 |
| 7 | 0.0060 | 0.0064 | 0.2875 | 0.2788 | 0.0087 |
| 8 | 0.0059 | 0.0063 | 0.2914 | 0.2995 | -0.0081 |
| 9 | 0.0058 | 0.0063 | 0.2956 | 0.2896 | 0.0060 |
| 10 | 0.0058 | 0.0062 | 0.2985 | 0.2991 | -0.0006 |
| 11 | 0.0057 | 0.0063 | 0.3060 | 0.2924 | 0.0136 |
| 12 | 0.0057 | 0.0062 | 0.3087 | 0.3073 | 0.0014 |
| 13 | 0.0056 | 0.0062 | 0.3133 | 0.3035 | 0.0098 |
| 14 | 0.0055 | 0.0062 | 0.3192 | 0.3071 | 0.0121 |
| 15 | 0.0055 | 0.0062 | 0.3226 | 0.3120 | 0.0106 |
| 16 | 0.0055 | 0.0061 | 0.3260 | 0.3148 | 0.0112 |
| 17 | 0.0054 | 0.0061 | 0.3301 | 0.3182 | 0.0119 |
| 18 | 0.0054 | 0.0061 | 0.3328 | 0.3215 | 0.0113 |
| 19 | 0.0053 | 0.0061 | 0.3389 | 0.3118 | 0.0271 |
| 20 | 0.0053 | 0.0061 | 0.3456 | 0.3063 | 0.0393 |
| 21 | 0.0053 | 0.0061 | 0.3474 | 0.3279 | 0.0195 |
| 22 | 0.0053 | 0.0061 | 0.3490 | 0.3130 | 0.0360 |
| 23 | 0.0052 | 0.0061 | 0.3524 | 0.3099 | 0.0425 |
| 24 | 0.0052 | 0.0061 | 0.3560 | 0.3186 | 0.0374 |
| 25 | 0.0052 | 0.0061 | 0.3587 | 0.3401 | 0.0186 |

Epochs 1-15 exactly reproduce trial-008 (same seed, same config up to that point, as expected).
`--checkpoint-selection f1` picked **epoch 25** (val_f1=0.3401) as best -- still the final epoch,
no regression before it, val_loss essentially flat from epoch 16 on (0.0061 throughout).

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable. Per-label AUROC: mean=**0.8629**
median=**0.9097**. Pooled AUROC=0.9773. Best F1=0.4132@threshold=0.22. All four metrics improved
over trial-008 (mean +0.0069, median +0.0076, pooled +0.0011, best F1 +0.0037) -- the best
diagnostics of the sweep so far, but the gains from 10 additional epochs are markedly smaller than
the jump from trial-007->008.

**Mild overfitting onset past epoch ~18, but contained**: the train/val F1 gap stayed in
trial-008's clean -0.03/+0.014 band through epoch 18, then widened to a noisier 0.02-0.04 band
from epoch 19 on (peak 0.0425 at epoch 23) -- a real trend, but far short of trial-006's runaway
widening (0.0263->0.0702 over just 5 epochs at the same unfreeze/data settings with dropout at
0.0). Notably epoch 25 itself pulled the gap back down to 0.0186 on a val_f1 jump, so this isn't a
clean monotonic blowup either -- it looks more like the point where this config's returns start
diminishing and noise increases, not a config that's about to collapse the way trial-006's did.

**Verdict: HEALTHY, best diagnostics of the sweep, with a mild but real overfitting signal
appearing past epoch ~18** -- `best_epoch` still lands on the final epoch (25), so the config
hasn't definitively stopped improving, but the shrinking marginal gain (+0.007 mean AUROC over 10
epochs, vs +0.02+ that dropout alone bought in trial-008) combined with the emerging gap suggests
this config is approaching, not clearly past, its ceiling.

**Next action**: ambiguous between two reasonable paths -- extend further (e.g. 35 epochs) to see
whether the gap keeps widening toward a trial-006-style blowup or the metrics genuinely plateau,
versus accepting ~epoch 18-20 as this config's effective ceiling (e.g. via early stopping at that
point) and moving to the next regularization/scale axis (weight decay, or revisiting train_n now
that dropout is in place). Flagged for the user to decide rather than picked unprompted.

User's decision: accept ~epoch 18-20 as this config's practical ceiling for now and move to the
next axis, but keep the door open to raising epochs again in a later trial specifically if that
trial's curve doesn't show a plateau by epoch 25.

## trial-011

**Config**: identical to trial-010 (`--esm-dropout 0.1 --attention-dropout 0.1`, no label
smoothing) except `--weight-decay 0.05` (was 0.01, untouched default since trial-001) and
`--epochs 20` (was 25, per the user's decision above to cap at this config's identified ceiling
rather than extend further). Log: `experiments/full_label_sweep/logs/trial-011.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 | gap (train-val) |
|-------|-----------|----------|----------|--------|------------------|
| 1 | 0.0068 | 0.0069 | 0.2223 | 0.2491 | -0.0268 |
| 2 | 0.0066 | 0.0067 | 0.2491 | 0.2531 | -0.0040 |
| 3 | 0.0063 | 0.0066 | 0.2580 | 0.2590 | -0.0010 |
| 4 | 0.0062 | 0.0065 | 0.2638 | 0.2597 | 0.0041 |
| 5 | 0.0061 | 0.0065 | 0.2771 | 0.2797 | -0.0026 |
| 6 | 0.0061 | 0.0064 | 0.2798 | 0.2857 | -0.0059 |
| 7 | 0.0060 | 0.0064 | 0.2871 | 0.2782 | 0.0089 |
| 8 | 0.0059 | 0.0063 | 0.2911 | 0.2988 | -0.0077 |
| 9 | 0.0058 | 0.0063 | 0.2951 | 0.2890 | 0.0061 |
| 10 | 0.0058 | 0.0062 | 0.2979 | 0.2992 | -0.0013 |
| 11 | 0.0057 | 0.0063 | 0.3054 | 0.2910 | 0.0144 |
| 12 | 0.0057 | 0.0062 | 0.3081 | 0.3081 | 0.0000 |
| 13 | 0.0056 | 0.0062 | 0.3127 | 0.3025 | 0.0102 |
| 14 | 0.0055 | 0.0062 | 0.3187 | 0.3071 | 0.0116 |
| 15 | 0.0055 | 0.0062 | 0.3218 | 0.3115 | 0.0103 |
| 16 | 0.0055 | 0.0062 | 0.3250 | 0.3110 | 0.0140 |
| 17 | 0.0055 | 0.0061 | 0.3290 | 0.3155 | 0.0135 |
| 18 | 0.0055 | 0.0061 | 0.3318 | 0.3197 | 0.0121 |
| 19 | 0.0053 | 0.0061 | 0.3378 | 0.3119 | 0.0259 |
| 20 | 0.0053 | 0.0062 | 0.3448 | 0.3026 | 0.0422 |

Through epoch 18 this tracks trial-010 almost exactly (e.g. epoch 18: train_f1=0.3318/val_f1=0.3197
here vs 0.3328/0.3215 there -- the higher weight decay isn't visibly doing anything different yet).
`--checkpoint-selection f1` picked **epoch 18** (val_f1=0.3197) as best -- a genuine, decisive peak:
val_f1 then *declines* monotonically through epoch 20 (0.3119, 0.3026), unlike trial-010's noisy
climb/recovery past its own equivalent point.

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable. Per-label AUROC: mean=**0.8543**,
median=**0.9035**. Pooled AUROC=0.9760. Best F1=0.4120@threshold=0.20. All four metrics are
**slightly worse** than trial-010 (mean -0.0086, median -0.0063, pooled -0.0013, best F1 -0.0012)
and mean AUROC is even slightly below trial-008's 0.8560, despite trial-011 training 3 more epochs
than trial-008 did.

**Weight decay increase doesn't resolve the overfitting gap, and mildly hurts**: the gap still
widens past epoch 18 (0.0259 at 19, 0.0422 at 20) just as it did in trial-010 -- `--weight-decay
0.05` bought no visible stabilization of the gap-widening pattern, while the post-peak decline is
cleaner/more monotonic (not noisy-with-recovery like trial-010's). Combined with the across-the-
board diagnostic regression vs. both trial-008 and trial-010, this reads as weight decay at this
value being a step sideways-to-negative, not a fix.

**Verdict: NEGATIVE/NEUTRAL for `--weight-decay 0.05`** -- doesn't resolve the overfitting gap,
diagnostics land slightly below trial-008 and trial-010, and the early peak (epoch 18) is followed
by a clean decline rather than a plateau. Not DEAD (still learns real ranking signal, comparable
scale to prior trials) but not an improvement either.

**Next action**: revert `--weight-decay` to 0.01 (trial-010's still-best value) and move to the
next axis -- `train_n`, per trial-010's own flagged options, now that dropout-based regularization
is locked in and weight decay has been ruled out as a lever. Unambiguous enough to proceed directly
rather than asking the user again.

## trial-012

**Config**: identical to trial-010 (`--esm-dropout 0.1 --attention-dropout 0.1`, `--weight-decay
0.01`, no label smoothing) except `--max-train-samples 32000` (was 16000 -- the train_n axis flagged
by trial-010/011, now retested with dropout regularization in place, unlike its earlier pre-dropout
tests in trial-004/007) and `--batch-size 2 --grad-accum-steps 8` (was `--batch-size 1
--grad-accum-steps 16` -- same effective batch size of 16, purely the user's confirmed
batch-size-2-is-fastest speed finding, not a learning-dynamics change). Log:
`experiments/full_label_sweep/logs/trial-012.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 | gap (train-val) |
|-------|-----------|----------|----------|--------|------------------|
| 1 | 0.0067 | 0.0067 | 0.2347 | 0.2655 | -0.0308 |
| 2 | 0.0063 | 0.0065 | 0.2622 | 0.2686 | -0.0064 |
| 3 | 0.0061 | 0.0064 | 0.2782 | 0.3029 | -0.0247 |
| 4 | 0.0060 | 0.0063 | 0.2868 | 0.2793 | 0.0075 |
| 5 | 0.0058 | 0.0063 | 0.2947 | 0.3032 | -0.0085 |
| 6 | 0.0057 | 0.0062 | 0.3048 | 0.2899 | 0.0149 |
| 7 | 0.0056 | 0.0062 | 0.3134 | 0.2967 | 0.0167 |
| 8 | 0.0055 | 0.0062 | 0.3200 | 0.3061 | 0.0139 |
| 9 | 0.0054 | 0.0062 | 0.3289 | 0.3252 | 0.0037 |
| 10 | 0.0054 | 0.0061 | 0.3361 | 0.3290 | 0.0071 |
| 11 | 0.0053 | 0.0061 | 0.3450 | 0.3145 | 0.0305 |
| 12 | 0.0052 | 0.0061 | 0.3508 | 0.3111 | 0.0397 |
| 13 | 0.0051 | 0.0061 | 0.3577 | 0.3250 | 0.0327 |
| 14 | 0.0051 | 0.0060 | 0.3672 | 0.3374 | 0.0298 |
| 15 | 0.0050 | 0.0061 | 0.3716 | 0.3465 | 0.0251 |
| 16 | 0.0050 | 0.0060 | 0.3775 | 0.3381 | 0.0394 |
| 17 | 0.0049 | 0.0061 | 0.3843 | 0.3356 | 0.0487 |
| 18 | 0.0049 | 0.0061 | 0.3901 | 0.3333 | 0.0568 |
| 19 | 0.0048 | 0.0061 | 0.3952 | 0.3441 | 0.0511 |
| 20 | 0.0048 | 0.0061 | 0.4009 | 0.3357 | 0.0652 |

No reproduction check against trial-010 is possible here (train_n itself changed, so the data seen
each epoch differs). `--checkpoint-selection f1` picked **epoch 15** (val_f1=0.3465) as best --
notably *earlier and higher* than trial-010's epoch-25 peak (val_f1=0.3401): doubling train_n let
the model reach a better validation optimum in 10 fewer epochs. Past epoch 15 val_f1 oscillates down
(0.3381, 0.3356, 0.3333) with a partial recovery at epoch 19 (0.3441) before falling again at 20
(0.3357) -- a noisy decline similar in shape to trial-010's post-peak behavior, not a clean collapse.

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable. Per-label AUROC: mean=**0.8689**,
median=**0.9177**. Pooled AUROC=0.9778. Best F1=0.4191@threshold=0.22. All four metrics **improved**
over trial-010 (mean +0.0060, median +0.0080, pooled +0.0005, best F1 +0.0059) -- the best
diagnostics of the sweep so far, surpassing trial-010 on every axis.

**Train_n finally pays off now that dropout regularization is in place**: unlike its two earlier,
pre-dropout tests (trial-004 at unfreeze=2/no-dropout: no benefit; trial-007 at unfreeze=4/no-dropout:
overfitting tied to gradient exposure, not resolved by more data), doubling train_n to 32000 on top
of trial-008/010's esm/attention-dropout=0.1 config produces a clear win on every diagnostic metric
*and* reaches its validation peak sooner (epoch 15 vs 25) -- more data plus dropout is a
combination that actually helps, where more data alone previously wasn't enough. The train/val F1
gap still widens past the peak (0.0251 at epoch 15 -> 0.0652 at epoch 20), essentially the same
overfitting-onset shape seen in trial-010/011, just arriving earlier in wall-clock epochs because
there's more signal to learn from sooner -- `--checkpoint-selection f1` already neutralizes this by
picking epoch 15's checkpoint rather than the final one. The `--batch-size 2`/`--grad-accum-steps 8`
change ran clean (no OOM, no warnings/errors in the log) and is treated as a pure speed lever, not a
confound, since it holds the effective batch size at 16, same as every prior trial.

**Verdict: POSITIVE for `train_n=32000`** -- best diagnostics of the sweep yet, and the first trial
where increasing train_n has clearly helped now that dropout is locked in. Confirms the axis trial-010
flagged as worth retesting was correctly identified.

**Next action**: push `train_n` further (e.g. 64000) to see whether the win continues or where it
tapers off, keeping `--weight-decay 0.01`, `--epochs 20`, and `--batch-size 2`/`--grad-accum-steps 8`
fixed at trial-012's now-proven values. Unambiguous enough to proceed directly rather than asking the
user again.

## trial-013

**Config**: identical to trial-012 except `--max-train-samples` dropped entirely (uses the full
~52.7k-example train split -- 64000 would have exceeded it anyway, per
`split_utils.py:compute_cluster_aware_split`'s `test_frac=0.18`/`val_frac_of_remainder=0.22` applied
to 82,404 total proteins) and `--epochs 30` (was 20 -- the user's own call, running this as the
intended sweep-closing baseline rather than a quick axis probe). Log:
`experiments/full_label_sweep/logs/trial-013.out`.

**Per-epoch metrics**:

| epoch | train_loss | val_loss | train_f1 | val_f1 | gap (train-val) |
|-------|-----------|----------|----------|--------|------------------|
| 1 | 0.0066 | 0.0066 | 0.2440 | 0.2704 | -0.0264 |
| 2 | 0.0061 | 0.0064 | 0.2749 | 0.2968 | -0.0219 |
| 3 | 0.0059 | 0.0062 | 0.2924 | 0.3053 | -0.0129 |
| 4 | 0.0057 | 0.0062 | 0.3050 | 0.3146 | -0.0096 |
| 5 | 0.0056 | 0.0061 | 0.3185 | 0.3177 | 0.0008 |
| 6 | 0.0054 | 0.0061 | 0.3310 | 0.3283 | 0.0027 |
| 7 | 0.0053 | 0.0061 | 0.3432 | 0.3199 | 0.0233 |
| 8 | 0.0052 | 0.0061 | 0.3552 | 0.3253 | 0.0299 |
| 9 | 0.0051 | 0.0061 | 0.3666 | 0.3175 | 0.0491 |
| 10 | 0.0050 | 0.0060 | 0.3768 | 0.3517 | 0.0251 |
| 11 | 0.0049 | 0.0061 | 0.3865 | 0.3400 | 0.0465 |
| 12 | 0.0048 | 0.0061 | 0.3951 | 0.3495 | 0.0456 |
| 13 | 0.0047 | 0.0061 | 0.4045 | 0.3489 | 0.0556 |
| 14 | 0.0047 | 0.0061 | 0.4130 | 0.3489 | 0.0641 |
| 15 | 0.0046 | 0.0061 | 0.4203 | 0.3442 | 0.0761 |
| 16 | 0.0045 | 0.0061 | 0.4277 | 0.3472 | 0.0805 |
| 17 | 0.0045 | 0.0061 | 0.4359 | 0.3583 | 0.0776 |
| 18 | 0.0044 | 0.0061 | 0.4423 | 0.3463 | 0.0960 |
| 19 | 0.0044 | 0.0061 | 0.4491 | 0.3544 | 0.0947 |
| 20 | 0.0043 | 0.0061 | 0.4549 | 0.3626 | 0.0923 |
| 21 | 0.0043 | 0.0062 | 0.4617 | 0.3675 | 0.0942 |
| 22 | 0.0042 | 0.0062 | 0.4683 | 0.3589 | 0.1094 |
| 23 | 0.0042 | 0.0061 | 0.4733 | 0.3741 | 0.0992 |
| 24 | 0.0041 | 0.0061 | 0.4791 | 0.3717 | 0.1074 |
| 25 | 0.0041 | 0.0062 | 0.4842 | 0.3601 | 0.1241 |
| 26 | 0.0041 | 0.0062 | 0.4897 | 0.3614 | 0.1283 |
| 27 | 0.0040 | 0.0063 | 0.4956 | 0.3557 | 0.1399 |
| 28 | 0.0040 | 0.0063 | 0.4996 | 0.3565 | 0.1431 |
| 29 | 0.0039 | 0.0063 | 0.5043 | 0.3703 | 0.1340 |
| 30 | 0.0039 | 0.0063 | 0.5082 | 0.3680 | 0.1402 |

`--checkpoint-selection f1` picked **epoch 23** (val_f1=0.3741) as best -- the global peak of a
noisy plateau, not a sharp single maximum: val_f1 oscillates in a roughly 0.34-0.37 band from
epoch 10 through 30 without a clean trend either way (epoch 20: 0.3626, 21: 0.3675, 29: 0.3703, 30:
0.3680, all close to epoch 23's 0.3741), while train_f1 climbs steadily and almost monotonically
the entire time (0.2440 -> 0.5082) as train_loss keeps dropping (0.0066 -> 0.0039). The train/val
gap widens far more than any prior trial in absolute terms -- 0.0251 at epoch 10, 0.0923 at epoch
20, 0.1402 at epoch 30 -- because train_f1 keeps climbing on the much larger train set while val_f1
has essentially saturated/plateaued since epoch ~10. This is the same qualitative overfitting-onset
shape seen in trial-010/012 (train keeps improving, val stops), just far more pronounced in gap size
at this scale and epoch count, and harmless in practice since `--checkpoint-selection f1` already
picks the plateau's best point rather than the final epoch.

**Diagnostic** (2000 val examples): 9,401/27,191 labels evaluable. Per-label AUROC: mean=**0.8731**,
median=**0.9259**. Pooled AUROC=0.9773. Best F1=0.4197@threshold=0.26. Mean AUROC (+0.0042), median
AUROC (+0.0082), and best val_f1 (0.3741 vs trial-012's 0.3465, +0.0276) all improve over trial-012;
pooled AUROC is flat/marginally down (-0.0005) and the best-F1-sweep value is essentially flat
(+0.0006) -- both within noise. **Best diagnostics of the entire sweep.**

**Verdict: HEALTHY -- best diagnostics of the sweep, confirms the full train split is this config's
practical scale ceiling**. Going from 32k to the full ~52.7k train examples continues the positive
trend trial-012 found, though with diminishing per-example returns (mean AUROC gain per added
training example is far smaller than the 16k->32k jump bought) and a much wider train/val gap by
epoch 30 -- but `--checkpoint-selection f1`'s epoch-23 pick is unaffected by that gap and represents
the strongest validated checkpoint produced anywhere in this sweep. The user ran this trial at 30
epochs specifically intending it as the sweep-closing baseline, and the plateau shape (noisy-flat
val_f1 from epoch ~10 on, best point at 23, no later epoch meaningfully beating it) supports treating
it that way rather than as one more axis probe.

**Next action**: pending the user's confirmation, promote this checkpoint via `--push-to-hub` to
`kelmoir/cafa6-esm@main` as the Tier-0 baseline `experiments/cafa6_next_steps_plan.md`'s Tier 1 work
(ensembling, PK-conditioning, DAG-consistency) builds on. This is a Hub push (visible, hard to fully
undo), so it's flagged for explicit go-ahead rather than run directly.
