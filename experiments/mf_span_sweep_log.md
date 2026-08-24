# MF Span-Localization Regularization Sweep Log

Tracks manual local-GPU trials of `experiments/scripts/train_mf_span_model.py` (the positional
molecular_function span-localization side-experiment described in `CLAUDE.md`'s "Data layout"
section — not part of the main CAFA6 pipeline). Regularization features (esm/attention dropout, LR
warmup+schedule, decoupled weight decay) were added to the script for this sweep, mirroring
`training_run.py`'s existing surface (commit `2f1b79a`).

All trials share this base config unless noted otherwise in "Changed":
`facebook/esm2_t12_35M_UR50D`, epochs=10, batch_size=16, max_train_pairs=12000, max_val_pairs=2000,
learning_rate=5e-5, unfreeze_last_n=2, output_dropout=0.2, grad_clip_norm=1.0, seed=42, device=cuda,
n_negatives_per_protein=2, weight_decay=0.01, esm_dropout=0.0, attention_dropout=0.0,
lr_schedule=constant, decouple_bias_wd=false. Pushed to `kelmoir/cafa6-mf-span`.

Methodology: one-axis-at-a-time coordinate descent, following the same convention as the main
pipeline's `sweep_log.md`.

## Results table

| Trial | run_name | Changed vs. base | best_epoch | best_val_loss | best/final F1 | Verdict |
|---|---|---|---|---|---|---|
| 1 (baseline) | `mf-span-esm2_t12_35M_UR50D-epochs10-bs16` (auto) | pre-regularization-commit script | 8 | 0.3948 | 0.6295 / 0.6301 | Mild overfit after epoch 8 (val_loss rises 0.395→0.434) |
| 2 (reg-v1) | `mf-span-esm2_t12-reg-v1` | esm/attn dropout 0.1 **+** cosine LR (2 axes at once — methodology slip) | 6 | 0.4270 (worse) | 0.5870 / 0.5906 (worse) | Underfit/plateau — bad trial, cause not isolated |
| 3 (dropout-only) | `mf-span-reg-dropout-only` | esm/attn dropout 0.1 only, constant LR | 8 | 0.3968 (≈baseline) | 0.6299 / 0.6395 (≈baseline) | Isolated: dropout alone is a no-op; confirms trial 2's regression was the cosine LR |
| 4 (reg-wd) | `mf-span-reg-wd` | weight_decay 0.05 + decouple_bias_wd, no dropout, constant LR | 8 | 0.3956 (≈baseline) | 0.6318 / 0.6319 (≈baseline) | Isolated: weight decay alone is also a no-op |
| 5 (reg-wd, name collision) | `mf-span-reg-wd` **(same name as trial 4 — overwrote it)** | n_negatives_per_protein 2→4 (wd/decouple unchanged from trial 4) | 4 | 0.3776 (lowest, but not comparable — see below) | 0.4480 / 0.5160 (worst so far) | Regression: more negatives per protein caused earlier overfit + lower F1, not better generalization |
| 6 (reg-out-dropout-0.2) | `mf-span-reg-out-dropout-0.2` | **none** — identical config to trial 4 (weight_decay 0.05 + decouple_bias_wd, n_negatives=2), just a fresh run_name | 8 | 0.3956 | 0.6319 / 0.6319 | Exact reproduction of trial 4 (bit-for-bit matching val_loss/F1) — recovers a retrievable checkpoint after trial 5 overwrote trial 4's |
| 7 (reg-more-data) | `mf-span-reg-more-data` | esm/attn dropout 0.0→0.1 **+** max_train_pairs 12000→20000, max_val_pairs 2000→5000 (2 axes at once — methodology slip, but dropout is a known no-op from trial 3) | 10 (still falling — no overfit yet) | 0.3532 | 0.6879 / 0.6879 | Genuine improvement — more data (same 42.9% positive rate, so a real volume increase not a composition change) resolves the overfit trials 1/3/4/6 all showed |
| 8 (reg-more-data-less-dropout) | `mf-span-reg-more-data-less-dropout` | isolates trial 7: dropout reverted to 0.0, data volume unchanged (20000/5000) | 10 (still falling — no overfit yet) | 0.3593 (≈trial 7) | 0.6874 / 0.6874 (≈trial 7, **best so far**) | Isolated: confirms trial 7's gain is purely from data volume — dropout remains a no-op even at this larger scale |
| 9 (reg-more-data-unfreeze-1) | `mf-span-reg-more-data-unfreeze-1` | isolates capacity: `unfreeze_last_n` 2→1, data volume unchanged (20000/5000), else identical to trial 8 | 10 (still falling — no overfit yet) | 0.3669 (worse) | 0.6493 / 0.6493 (worse) | Regression: cutting capacity underfits — there was no overfitting at this data scale (per trials 7/8) for it to fix |
| 10 (reg-more-data-unfreeze-4) | `mf-span-reg-more-data-unfreeze-4` | isolates capacity: `unfreeze_last_n` 2→4, data volume unchanged (20000/5000), else identical to trial 8 | 5 (by val_loss; **see note**) | 0.3601 | **0.6441 (pushed checkpoint, epoch 5) / 0.6980 (final epoch — sweep-best, but not the pushed checkpoint)** | First clear OVERFITTING-by-loss in the sweep (val_loss diverges from train_loss after epoch 5), but F1 keeps rising anyway through epoch 10 — val_loss-based checkpoint selection picked a worse-F1 epoch than several later ones |
| 11 (unfreeze-2, epochs=20, crashed) | not captured (crash — see note) | extends epochs on `unfreeze_last_n=2` (trial 7/8's setting, not trial 10's) from 10→20, data volume unchanged (20000/5000), else identical to trial 8 | 11 (confirmed via trackio) | 0.378 (confirmed; epoch 17 = 0.396 — **mild rise, see note**) | **0.710 at epoch 17 (confirmed, new sweep-best)** — precision 0.67, recall 0.74 | **Corrected verdict**: trackio recovery (step counter is 0-indexed, so trackio step *N* = epoch *N*+1 — corrected from an initial off-by-one) shows val_loss *did* rise after epoch 11 (0.378→0.396 by epoch 17, train_loss still falling to 0.16) — a real but much gentler OVERFITTING-by-loss than trial 10's sharp epoch-8 spike, not the "no divergence" read the crash-memory estimate gave. Precision/recall/F1 were all still climbing at epoch 17 despite the val_loss rise — same recurring val_loss/F1 divergence as trials 5 and 10, now with confirmed numbers. **Exposed a real gap: script had no incremental per-epoch log file, only stdout — fixed same day, see note.** |

## Trial notes

**Trial 1 (baseline).** Pre-regularization-commit script, so no dropout/schedule knobs existed yet.
Mild overfit signature: val_loss bottoms out at epoch 8 (0.3948) then climbs to 0.434 by epoch 10.
Sets the target to beat.

**Trial 2 (reg-v1).** Changed two axes simultaneously (esm/attention dropout 0.1 **and** cosine LR
schedule) — a violation of the one-axis-at-a-time convention. Result was strictly worse than
baseline (best_val_loss 0.4270 vs 0.3948, F1 0.587 vs 0.630) but with two changes conflated, the
cause couldn't be attributed. Corrected for trial 3+ by isolating each axis.

**Trial 3 (dropout-only).** Isolates esm/attention dropout 0.1 with constant LR (trial 2's other
axis reverted). Result is within noise of baseline (val_loss 0.3968, F1 0.6299/0.6395 — final F1
slightly *better* than baseline's). Confirms dropout alone was not responsible for trial 2's
regression — the cosine LR schedule was.

**Trial 4 (reg-wd).** Isolates weight_decay 0.05 + decouple_bias_wd, no dropout, constant LR.
Also within noise of baseline (val_loss 0.3956, F1 0.6318/0.6319).

**Trial 5 (reg-wd, run_name collision).** Kept trial 4's weight_decay/decouple_bias_wd config but
raised `--n-negatives-per-protein` from 2 to 4. Two issues:
- **Data-loss bug**: reused `--run-name mf-span-reg-wd` identically to trial 4. `save_checkpoint`/
  `push_checkpoint_to_hub` have no collision guard, so trial 5 silently overwrote trial 4's checkpoint
  both locally (`models/mf-span-reg-wd/best_checkpoint`) and on the Hub (`kelmoir/cafa6-mf-span@main`).
  Trial 4's actual weights (best result so far, F1 0.6318) are no longer retrievable from that
  location. **Use a distinct `--run-name`/`--hub-branch` per trial going forward.**
  Note "results" table below.
- **Metric-comparability trap**: raising negatives-per-protein under the fixed `--max-train-pairs
  12000` cap changes the pair pool's composition, not just its size — val positives dropped from
  858/2000 (42.9%) to 558/2000 (27.9%). More all-zero-target ("easy") pairs mechanically dilutes
  average val_loss downward without reflecting better span-localization quality, so trial 5's
  val_loss (0.3776) is **not directly comparable** to trials 1-4's. F1 stays meaningful across the
  composition change and shows the real story: best_epoch moved earlier (4 vs 8 — the model overfit
  *sooner* with more negatives, not later) and best/final F1 dropped well below every prior trial
  (0.448 / 0.516 vs. ~0.63 baseline). Net: `n_negatives_per_protein=4` is a regression, not a
  regularizer, at least under this fixed pair-count cap.

**Trial 6 (reg-out-dropout-0.2).** Despite the name, no axis was actually changed from trial 4 —
weight_decay=0.05, decouple_bias_wd=true, n_negatives_per_protein=2, esm/attn dropout=0.0, all
identical to trial 4. Results match trial 4's to the last decimal (val_loss 0.3956468...,
F1 0.6318958...), confirming the run is deterministic under this config/seed. Net effect: trial 4's
best result is now retrievable again, on its own branch (`kelmoir/cafa6-mf-span@main`, currently
holding this checkpoint, run_name `mf-span-reg-out-dropout-0.2`) instead of the one trial 5
overwrote. `--output-dropout` above 0.2 is still untested.

**Trial 7 (reg-more-data).** Two axes changed at once again: esm/attention dropout 0.0→0.1 (re-added)
and `--max-train-pairs`/`--max-val-pairs` roughly doubled (12000→20000 / 2000→5000). This does
violate one-axis-at-a-time, but with lower risk of misattribution than trial 2's case, since trial 3
already isolated esm/attn dropout=0.1 as a no-op against this exact baseline — the dropout term is
along for the ride, not a live confound. The val set's positive rate held essentially constant
(2143/5000 = 42.9% vs. trial 4/6's 858/2000 = 42.9%), so unlike trial 5 this is a genuine volume
increase, not a composition shift, and val_loss/F1 are directly comparable to earlier trials.

Result is a clear, unambiguous improvement: best F1 0.6879 (vs. ~0.63 for every prior config), and
critically, val_loss was **still decreasing at epoch 10** (0.3612→0.3532, no sign of the epoch-8
plateau/overfit every prior trial hit). This is the first trial where more data, not a regularizer,
moved the needle — suggesting the earlier "overfitting" was at least partly a data-starvation
artifact (12k/2k pairs) rather than something dropout/weight-decay could fix. Recall (0.7533) now
noticeably exceeds precision (0.6329), a shift from the roughly balanced precision/recall of trials
4/6 — worth watching in future trials, not yet diagnosed.

**Recurring Hub-overwrite pattern (all trials, not just 5)**: every trial's config carries
`hub_branch: "main"` — none has ever overridden `--hub-branch`. Local checkpoints are safely
distinct per `--run-name` (`models/<run_name>/best_checkpoint`), but **the Hub copy at
`kelmoir/cafa6-mf-span@main` is overwritten by every single trial's push**, regardless of run_name.
Trial 7 overwrote trial 6's Hub copy this way (trial 6's checkpoint is still safe locally at
`models/mf-span-reg-out-dropout-0.2/best_checkpoint`, just no longer on the Hub). Only the most
recent trial's checkpoint is ever Hub-retrievable unless `--hub-branch` is explicitly set per trial.

**Trial 8 (reg-more-data-less-dropout).** Direct isolation of trial 7: identical 20000/5000-pair
config with esm/attn dropout reverted to 0.0. Result lands within noise of trial 7's (val_loss 0.3593
vs. 0.3532; F1 0.6874 vs. 0.6879 — statistically indistinguishable, if anything trial 8's F1 is
marginally the higher of the two once rounding is accounted for). Confirms cleanly: the gain from
trial 7 is 100% attributable to data volume, not dropout — dropout remains a no-op even at 20k/5k
scale, consistent with trial 3's finding at 12k/2k scale. Also still hadn't plateaued at epoch 10
(val_loss 0.3651→0.3593), same as trial 7 — the epoch-8 overfit ceiling seen at 12k/2k scale is gone
at this data volume regardless of dropout.

Also overwrote trial 7's Hub copy (`kelmoir/cafa6-mf-span@main`) per the recurring pattern noted
above; trial 7's checkpoint is still safe locally at `models/mf-span-reg-more-data/best_checkpoint`.

**Trial 9 (reg-more-data-unfreeze-1).** Isolates a capacity cut instead of a penalty-based
regularizer: `unfreeze_last_n` 2→1 (only the last backbone layer trainable), data volume unchanged
from trial 8 (20000/5000), dropout still 0.0. This is a real, unconfounded regression — both metrics
land clearly outside trial 8's range (val_loss 0.3669 vs. 0.3593; F1 0.6493 vs. 0.6874, a ~0.038 F1
drop, well past the noise band every dropout/weight-decay trial landed within). Still hadn't plateaued
by epoch 10 either (val_loss 0.3772→0.3669), so this isn't a case of "less capacity, faster
convergence" — it's just a worse model throughout training.

Consistent with trial 7/8's finding: there is no overfitting at the 20k/5k data scale for a capacity
cut to fix (val_loss was still falling in both, not diverging), so removing trainable capacity here
only hurts. The "considerable overfitting" this trial was meant to guard against is a 12k/2k-scale
phenomenon (trials 1/3/4/6) that more data already resolved — `unfreeze_last_n=2` should stay as-is
unless overfitting reappears at a larger data/epoch budget.

Also overwrote trial 8's Hub copy per the recurring pattern; trial 8's checkpoint is still safe
locally at `models/mf-span-reg-more-data-less-dropout/best_checkpoint`.

**Trial 10 (reg-more-data-unfreeze-4).** Isolates capacity in the other direction: `unfreeze_last_n`
2→4, data volume unchanged (20000/5000), dropout still 0.0. Per-epoch curve:

| epoch | train_loss | val_loss | val_f1 |
|---|---|---|---|
| 1 | 0.4905 | 0.4499 | 0.4892 |
| 2 | 0.3726 | 0.4110 | 0.5762 |
| 3 | 0.3245 | 0.3990 | 0.6049 |
| 4 | 0.2902 | 0.3900 | 0.6158 |
| 5 | 0.2627 | **0.3601 (best val_loss)** | 0.6441 |
| 6 | 0.2351 | 0.3743 | 0.6661 |
| 7 | 0.2138 | 0.3680 | 0.6814 |
| 8 | 0.1962 | 0.4061 (spike) | 0.6803 |
| 9 | 0.1821 | 0.3863 | 0.6950 |
| 10 | 0.1650 | 0.3682 | **0.6980 (best F1)** |

Two things at once here:

1. **This is the sweep's first genuine OVERFITTING verdict by the project rubric.** train_loss falls
   smoothly the entire run (0.49→0.17) while val_loss bottoms at epoch 5 and then diverges — flattens,
   spikes, oscillates — a widening train/val gap exactly matching the CLAUDE.md rubric's OVERFITTING
   pattern. Neither trial 7/8 (unfreeze=2) nor trial 9 (unfreeze=1) showed this at the same 20k/5k data
   scale — `unfreeze_last_n=4` is the first setting where this data volume isn't enough to keep the
   extra capacity from overfitting (by loss).
2. **But F1 doesn't reflect that overfitting** — it keeps climbing almost monotonically through
   epoch 10 (0.644→0.698), reaching the best F1 of the entire sweep, *after* val_loss says the model
   started overfitting. Same val_loss/F1-divergence pattern already seen in trial 5 and documented in
   CLAUDE.md's `run_diagnostic()`/trial-050 precedent for the main pipeline: aggregate BCE loss and
   threshold-based F1 don't always move together under class imbalance.
3. **Practical consequence**: `save_checkpoint`'s selection criterion is val_loss-only (lowest wins),
   so the checkpoint actually written to disk and pushed to `kelmoir/cafa6-mf-span@main` is **epoch
   5's** (F1 0.6441) — worse by F1 than trials 7/8 (~0.687-0.688), even though this same run reached
   F1 0.698 at epoch 10. The better-F1 weights were never saved; only the per-epoch log line records
   they existed. If F1/task quality is the metric that matters, val_loss-based early-stopping is
   picking a suboptimal checkpoint whenever this divergence happens.

**Trial 11 (unfreeze-2, epochs=20, crashed).** Rather than trial 10's specific recommendation
(rerun *unfreeze=4* with more epochs), this trial extended epochs on `unfreeze_last_n=2` instead —
trials 7/8's setting, the one that "hadn't found its ceiling within 10 epochs" per the trial-10
conclusion. Same data volume (20000/5000), same non-dropout config as trial 8, `--epochs 20`. The
machine crashed partway through (accidental, not a training divergence) after epoch 17, before the
run's own summary/checkpoint bookkeeping could be inspected, and at the time the script had no
per-epoch log file, only stdout — so most exact numbers were initially lost. Two passes were needed
to reconstruct this note from the run's local trackio db, which survived the crash independently of
stdout:

- **First pass** relied on memory alone ("val_loss stayed at usual rates, no divergence") — wrong,
  corrected below.
- **Second pass** pulled real numbers from trackio but initially reported them against the wrong
  epoch: trackio's step counter is 0-indexed (first `.log()` call = step 0), so **trackio step *N*
  corresponds to epoch *N*+1**, not epoch *N*. Corrected throughout.
- **val_loss bottomed at epoch 11 (0.378, confirmed)** — close to, but slightly higher than, the
  original memory-based estimate (~0.35-0.36).
- **val_loss then rose to 0.396 by epoch 17, while train_loss kept falling to 0.16.** This *is* a
  real overfitting-by-loss signature after all — the first-pass "no divergence" read was wrong. It's
  materially gentler than trial 10's: trial 10 spiked from 0.360→0.406 (Δ0.046) in a single epoch
  (5→8, trial 10's epoch numbers came straight from stdout's already-1-indexed `[EPOCH N/10]` lines,
  no trackio offset involved there); trial 11 rose from 0.378→0.396 (Δ0.018) over six epochs
  (11→17). Same direction, much slower onset — consistent with `unfreeze_last_n=2` overfitting later
  and more gradually than `unfreeze=4` at this data scale, not avoiding it entirely.
- **val F1 at epoch 17 = 0.710 (confirmed), precision 0.67 / recall 0.74** — the sweep's best F1,
  ahead of trial 10's 0.698, and precision/recall/F1 were all *still climbing* at epoch 17 even
  though val_loss had already turned upward six epochs earlier. Same recall > precision skew first
  seen in trial 7 (0.7533/0.6329) persists here. This confirms — not just suggests — the same
  val_loss/F1-divergence pattern as trials 5 and 10: aggregate BCE loss and threshold-0.5 F1 keep
  moving in opposite directions once the model gets confident, exactly CLAUDE.md's `run_diagnostic()`/
  trial-050 precedent.
- No checkpoint near epoch 17 was saved either way — `save_checkpoint`'s val_loss-only selection
  means whatever's on disk from this run is close to the epoch-11 optimum, not the later (higher-F1)
  epochs. Same gap as trial 10, compounded here by the crash.
- **Fixed same day**: `train_mf_span_model.py` previously only printed per-epoch metrics to
  stdout — nothing hit disk until a checkpoint improved, so a crash mid-run (like this one) loses
  everything except what a human happened to be watching or what trackio's local db happened to
  capture. Added incremental per-epoch JSONL logging (`--log-file`, default
  `experiments/logs/mf_span_<run-name>.jsonl`), flushed and `os.fsync`'d after every epoch, so any
  future crash preserves every completed epoch's exact numbers regardless of when it happens.

## Standing conclusion (through trial 11)

The capacity axis (`unfreeze_last_n`) now has a shape at the 20k/5k data scale: 1 underfits; 2
overfits-by-loss too, but later and much more gently than 4 (val_loss bottoms at epoch 11 at 0.378,
then only creeps to 0.396 by epoch 17 — Δ0.018 over six epochs, vs. trial 10's Δ0.046 in a single
epoch, epoch 5→8); 4 overfits sharply starting around epoch 5-6. In both trial 10 and trial 11, F1
kept rising past the point val_loss turned upward — trial 11's F1 confirmed at 0.710 by epoch 17
(precision 0.67, recall 0.74, both still climbing at crash time), ahead of trial 10's 0.698. So more
capacity doesn't straightforwardly buy a *better* ceiling, just a *faster, less stable* approach to a
similar-or-lower one — `unfreeze=2` reaches a higher confirmed F1 with a much gentler val_loss rise.
Every dropout/weight-decay axis remains a no-op; capacity is the only axis that has produced a real,
direction-dependent effect.

Current best *retrievable* checkpoint by F1: still trial 7 or 8 (~0.687-0.688) — trial 10's best-F1
epoch (0.698) was never saved, and trial 11's better result (0.710, confirmed) is unsaved too (the
crash happened before epoch 20, and `save_checkpoint`'s val_loss-only criterion would have kept the
epoch-11 checkpoint anyway, not epoch 17's). Best F1 *confirmed anywhere* in the sweep is trial 11's
epoch 17 (0.710), but since the run crashed before completing its planned 20 epochs, it's still
unknown whether F1 had truly plateaued there or was still rising — that's what the next trial should
settle.

Recommended next trial: **cleanly rerun trial 11's exact config** (`unfreeze_last_n=2`, 20k/5k, no
dropout, `--epochs 20`, ideally a couple past 20 as headroom) now that the crash-logging gap is fixed
(`--log-file`, see trial 11's note) — this both confirms the run reaches/exceeds F1 0.710 without a
crash and locates where F1 actually plateaus past epoch 17. Since `unfreeze=2`'s val_loss rise is much
gentler than `unfreeze=4`'s, this should also produce a checkpoint that's closer to actually
retrievable (val_loss and F1 aren't fighting each other as sharply as they are for `unfreeze=4`). If
it holds, this becomes the sweep's leading config, ahead of trials 7/8/10, without needing the
capacity increase trial 10 introduced. Separately, still worth deciding whether checkpoint selection
should switch from val_loss to F1 (or track both) given the recurring divergence, now confirmed at
both `unfreeze=2` and `unfreeze=4`; flagging, not changing anything without you weighing in. Untested
regularization axes remain lowest priority given how little effect they've had so far:
`--output-dropout` above 0.2, `--learning-rate` lower than 5e-5, `--lr-schedule linear_decay`,
`--pos-weight` override. Remember to give each future trial a distinct `--hub-branch`,
not just a distinct `--run-name`, if the Hub copy should stay retrievable.
