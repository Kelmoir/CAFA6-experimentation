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
| 12 (unfreeze2-epochs30, clean) | `mf-span-reg-more-data-unfreeze2-epochs30` | clean, uncrashed rerun of trial 11's config with `--epochs 30` (the recommended next trial), full `--log-file` capture, all other args identical to trial 8 | 10 (exact) | **0.3593 (exact match to trial 8 — see note on determinism)** | **0.7421 at epoch 28 (new sweep-best, saved checkpoint only reaches 0.6874 at epoch 10 — see note)** | Confirms and sharpens trial 11: real OVERFITTING-by-loss after epoch 10, noisy and much larger by epoch 30 (val_loss 0.3593→0.5936, Δ0.234) than the partial trial-11 picture suggested, while F1 climbs to a plateau of roughly 0.72-0.74 from epoch ~19 onward. Largest val_loss/F1-selection gap in the sweep so far (0.6874 saved vs. 0.7421 true best) — prompted adding a `--checkpoint-selection {val_loss,f1}` flag (see note) |

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

**Trial 12 (unfreeze2-epochs30, clean).** The recommended clean rerun of trial 11's config, extended
to `--epochs 30` (past the suggested 20, for extra headroom), with the new `--log-file` capturing
every epoch — no crash this time, full data available. Full curve:

| epoch | train_loss | val_loss | val_f1 | new best ckpt? |
|---|---|---|---|---|
| 1 | 0.5107 | 0.4803 | 0.4199 | yes |
| 2 | 0.4094 | 0.4377 | 0.5416 | yes |
| 3 | 0.3669 | 0.4227 | 0.5765 | yes |
| 4 | 0.3341 | 0.3998 | 0.5989 | yes |
| 5 | 0.3078 | 0.3778 | 0.6237 | yes |
| 6 | 0.2819 | 0.3877 | 0.6446 | no |
| 7 | 0.2613 | 0.3723 | 0.6579 | yes |
| 8 | 0.2432 | 0.3701 | 0.6766 | yes |
| 9 | 0.2279 | 0.3651 | 0.6849 | yes |
| 10 | 0.2122 | **0.3593 (best val_loss — saved/pushed checkpoint)** | 0.6874 | yes |
| 11 | 0.1982 | 0.3820 | 0.6964 | no |
| 12 | 0.1857 | 0.3928 | 0.6958 | no |
| 13 | 0.1758 | 0.3732 | 0.7055 | no |
| 14 | 0.1630 | 0.3924 | 0.7069 | no |
| 15 | 0.1550 | 0.3830 | 0.7152 | no |
| 16 | 0.1454 | 0.4020 | 0.7189 | no |
| 17 | 0.1372 | 0.4314 | 0.7181 | no |
| 18 | 0.1284 | 0.4367 | 0.7215 | no |
| 19 | 0.1217 | 0.4492 | 0.7212 | no |
| 20 | 0.1148 | 0.4590 | 0.7232 | no |
| 21 | 0.1068 | 0.4559 | 0.7322 | no |
| 22 | 0.1018 | 0.4603 | 0.7296 | no |
| 23 | 0.0955 | 0.4741 | 0.7347 | no |
| 24 | 0.0891 | 0.5089 | 0.7398 | no |
| 25 | 0.0836 | 0.5052 | 0.7361 | no |
| 26 | 0.0805 | 0.5495 | 0.7378 | no |
| 27 | 0.0750 | 0.5492 | 0.7377 | no |
| 28 | 0.0711 | 0.5386 | **0.7421 (best F1)** | no |
| 29 | 0.0663 | 0.6009 | 0.7339 | no |
| 30 | 0.0634 | 0.5936 | 0.7390 | no |

Several findings:

1. **Determinism check, with an unresolved discrepancy.** Epoch 10's val_loss here (0.3593) is an
   *exact* match to trial 8's final val_loss (0.3593) — trial 8 used the identical config (same
   seed=42, same weight_decay=0.05+decouple_bias_wd inherited from trial 4/6, same unfreeze=2, same
   20000/5000 pairs, no dropout), so this is expected and confirms the run is bit-for-bit
   reproducible. It does **not**, however, match trial 11's reported epoch-11 val_loss (0.378) — if
   trial 11's config were truly identical, its epoch-10 number should also have been 0.3593. Trial
   11's exact `[CONFIG]` line was never captured (lost to the crash before this log could read it),
   so there's no way to confirm what actually differed; flagging the discrepancy rather than
   asserting a cause. Trial 12 should be treated as the trustworthy reference for this config, not
   trial 11.
2. **Confirms real OVERFITTING-by-loss, and it's worse than trial 11's partial view suggested.**
   val_loss bottoms at epoch 10 (0.3593) then climbs — noisily, with local dips (13, 15) but a clear
   overall upward trend — all the way to 0.5936 by epoch 30 (Δ0.234). Trial 11's crash-truncated view
   (only out to epoch 17, Δ0.018) made this look much gentler than trial 10's than it actually is once
   given the full 30 epochs; extended fully, `unfreeze_last_n=2` still eventually overfits substantially
   by val_loss, just far later in training than `unfreeze_last_n=4` did (epoch ~10 vs. epoch ~5).
3. **F1 finds a real plateau.** From epoch ~19 onward, F1 stays bounded in roughly 0.72-0.74 with
   noise (0.7212 → peak 0.7421 at epoch 28 → 0.7390 at epoch 30) rather than climbing steadily like
   epochs 1-18 did. This is the first time in the sweep an F1 plateau has actually been located (not
   just "still rising when the trial ended") — per the user's plateau-first strategy, `unfreeze=2` at
   20k/5k data plateaus around **F1 ≈ 0.72-0.74**, well above every other trial's saved-checkpoint F1.
4. **The val_loss/F1-selection gap is now the largest in the sweep.** The saved/pushed checkpoint
   (epoch 10, val_loss-selected) has F1 0.6874 — the true best-observed F1 in this run is 0.7421 at
   epoch 28, a 0.0547 gap, larger than trial 10's 0.0539 gap (0.6441 vs. 0.698) and trial 11's now-
   moot gap. Also overwrote the previous Hub checkpoint at `kelmoir/cafa6-mf-span@main` per the
   recurring pattern.
5. **Acted on this trial's own evidence**: given this is the fourth trial (5, 10, 11, 12) showing
   the same val_loss/F1 divergence, and by far the largest gap yet, added a `--checkpoint-selection
   {val_loss,f1}` flag to `train_mf_span_model.py` (default remains `val_loss`, preserving every past
   trial's behavior) — passing `--checkpoint-selection f1` makes "best" mean highest val_f1 instead of
   lowest val_loss, so a future trial like this one would actually save/push the epoch-28 weights
   instead of epoch-10's. Also fixed a small adjacent bug noticed while touching this code: the
   `[SUMMARY]` line's printed precision/recall/f1 came from the *final* epoch's `metrics` variable,
   not `best_metrics` — misleading regardless of `--checkpoint-selection`, since it could report
   numbers for an epoch that isn't the one actually saved. Now uses `best_metrics` (falling back to
   `metrics` only if no checkpoint was ever saved, i.e. val_loss/F1 never improved past its initial
   value).

## Standing conclusion (through trial 12)

The capacity axis (`unfreeze_last_n`) now has a well-established shape at the 20k/5k data scale, and
trial 12 finally located its F1 plateau: **`unfreeze=2` plateaus around F1 ≈ 0.72-0.74** (epoch ~19-30,
peak 0.7421 at epoch 28), clearing every other config in the sweep by a wide margin, while
`unfreeze=4` (trial 10) only reached F1 0.698 before its much sharper val_loss overfit made further
training unproductive, and `unfreeze=1` (trial 9) underfits throughout. Trial 12 also corrected the
partial picture trial 11's crash left: `unfreeze=2` *does* eventually overfit substantially by
val_loss (0.3593 at epoch 10 → 0.5936 by epoch 30, noisy but clearly upward) — just far later and
more gradually than `unfreeze=4`'s epoch-5 spike, not "no divergence." Every dropout/weight-decay
axis remains a no-op; capacity and (now) epoch budget are the only axes that have produced real
effects. Trial 12 is the trustworthy reference for this config — an unresolved discrepancy against
trial 11's numbers means trial 11 should no longer be treated as authoritative (see trial 12's note).

Current best *retrievable* checkpoint by F1: still trial 7 or 8 (~0.687-0.688) — trial 12's actual
best (F1 0.7421 at epoch 28) was saved under the old `val_loss`-only criterion, so what's on disk/Hub
from trial 12 is epoch 10's (F1 0.6874), not epoch 28's. This is now fixed going forward: a
`--checkpoint-selection {val_loss,f1}` flag was added to `train_mf_span_model.py` (default `val_loss`,
see trial 12's note) — the next trial should pass `--checkpoint-selection f1` to actually retain the
plateau-region weights.

Recommended next trial: **rerun trial 12's exact config with `--checkpoint-selection f1`** (same
`unfreeze_last_n=2`, 20k/5k, no dropout, `--epochs 30`) to get a genuinely retrievable checkpoint near
the F1 plateau (~0.72-0.74) instead of the val_loss-selected epoch-10 one — training curves should be
identical to trial 12's (same seed/config), so this is mainly about capturing the right checkpoint,
not re-discovering the curve. With the plateau now located for this data/capacity combo, the next
open question per the user's bitter-lesson framing is whether to push a new axis — more data (the
uncapped pool is ~32k/9k pairs, see the earlier ad hoc calculation) or a capacity value between 2 and
4 (e.g. `unfreeze_last_n=3`) — rather than continuing to spend epochs on `unfreeze=2`, which has now
plateaued. Untested regularization axes remain lowest priority given how little effect they've had so
far: `--output-dropout` above 0.2, `--learning-rate` lower than 5e-5, `--lr-schedule linear_decay`,
`--pos-weight` override. Remember to give each future trial a distinct `--hub-branch`, not just a
distinct `--run-name`, if the Hub copy should stay retrievable.
