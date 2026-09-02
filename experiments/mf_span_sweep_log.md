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
| 13 (unfreeze2-epochs30-label-smoothing) | `mf-span-reg-more-data-unfreeze2-epochs30-label_smoothing` | trial 12 + `--label-smoothing 0.1` (training targets only), `--checkpoint-selection f1` (first trial to actually use it), `--early-stop-patience 5` (never triggered — ran the full 30 epochs) | 28 | 0.3069 (val_loss at the best-F1 epoch — see note) | **0.7539 at epoch 28 — new sweep-best, and the first trial where the true-best epoch is also the actually saved/pushed checkpoint** | Label smoothing closes the val_loss/F1 divergence almost entirely (val_loss rises only 0.0057 from its epoch-28 min by epoch 30, vs. trial 12's Δ0.234) while also raising absolute F1 above trial 12's at every directly-comparable epoch — a genuine, unconfounded win (the other two changed flags don't affect the training curve itself — see note) |
| 14 (unfreeze2-epochs30-label-smoothing-effbatch32) | `mf-span-reg-more-data-unfreeze2-epochs30-label_smoothing-effbatch32` | trial 13 + `--batch-size 4 --grad-accum-steps 8` (effective batch 32, vs. trial 13's real batch 16), same `--learning-rate 5e-5` (not scaled up for the larger effective batch) | 28 | 0.3140 (worse than trial 13's 0.3069) | **0.7384 at epoch 28 (worse than trial 13's 0.7539 by 0.0155 — trailed trial 13 at every matched epoch, not just the end)** | Doubling effective batch size at a fixed epoch budget and unscaled LR is a real regression, not an improvement — but ~24% faster wall-clock (20719s vs. trial 13's 27241s for the same 30 epochs), consistent with half as many optimizer steps/epoch and unscaled LR under-training relative to trial 13, not evidence that a bigger effective batch is inherently worse (see note) |
| 15 (unfreeze2-epochs30-trial-15, effbatch32+lr-scaled) | `mf-span-reg-more-data-unfreeze2-epochs30-trial-15` | trial 14 + `--learning-rate 1e-4` (2x, scaled with effective batch per the linear scaling rule) | 19 (early-stopped after 24) | **0.3038** | **0.7544 at epoch 19 (barely edges out trial 13's 0.7539), reached 9 epochs earlier than trial 13's peak** | Confirms the linear scaling rule hypothesis: LR-scaling doesn't just close trial 14's gap, it slightly exceeds trial 13's quality — while finishing in 16677.7s, ~39% faster than trial 13's 27241s and ~19% faster than trial 14's 20719s (first trial where early stopping actually fired, on top of the batch-size speed advantage) — see note |
| 16 (unfreeze2-epochs30-trial-16, effbatch32/bs2-ga16+label_smoothing0.2) | `mf-span-reg-trial-16` | trial 15 + `--batch-size 2 --grad-accum-steps 16` (same effective batch 32, different split) + `--label-smoothing 0.2` | — | — | **crashed after epoch 17/30 (cause unknown), no `[SUMMARY]`** | Incomplete run, not a result — superseded by trial 17, which isolates the label-smoothing change alone against trial 15's exact batch/grad-accum split. Notably ~14% faster per epoch than trial 17 (~577s avg vs. ~672s avg) — user suspects a link to the crash, unconfirmed. Log file kept for the record only |
| 17 (reg-trial-17, label_smoothing0.2) | `mf-span-reg-trial-17` | trial 15's exact config (`--batch-size 4 --grad-accum-steps 8 --learning-rate 1e-4`) + `--label-smoothing 0.2` (2x) | 26 | 0.3190 (not the run's true min — see note) | 0.7642 at epoch 26, beats trial 15 at every matched epoch | Higher label smoothing is a further real, unconfounded win — but early stopping didn't fire this time (only 4 flat epochs after 26, one short of patience 5), so it ran the full 30 epochs: 20180.2s, ~21% more expensive than trial 15 though still ~26% cheaper than trial 13 — see note |
| 18 (trial-18, label_smoothing0.3, crashed) | `mf-span-reg-trial-18` | trial 17's exact config + `--label-smoothing` 0.2→0.3 | — | — | **crashed after epoch 14/30, no `[SUMMARY]`** | Incomplete run — a confirmed CUDA Xid 13 GPU exception (via `dmesg`/`journalctl`, timestamp-matched to the log file's last write to the second), not a code bug; see the session's diagnosis. Predates `--resume-from-checkpoint`/`run_mf_span_trial.sh` (added the same session), so nothing to resume from — superseded by trial 19, an identical-config clean rerun |
| 19 (reg-trial-19, unfreeze2, label_smoothing0.3) | `mf-span-reg-trial-19` *(original — overwritten, see 19b)* | trial 17's exact config + `--label-smoothing` 0.2→0.3 (rerun of trial 18's config, clean this time) | 26 | 0.3538 (not the run's true min — see note) | 0.7651 at epoch 26, beats trial 17 at 25/30 epochs | Marginal further win over trial 17: +0.0009 vs. trial 17's own +0.0098 over trial 15 — an order of magnitude smaller. Superseded on disk/Hub by trial 19b below (same run-name/branch accidentally reused) — this run's full 30-epoch curve only survives in git history at commit `d383a6c`, not on disk anymore. See note |
| 19b (reg-trial-19, unfreeze3, label_smoothing0.3) | `mf-span-reg-trial-19` *(current)* | trial 19's exact config + `unfreeze_last_n` 2→3 — recommendation (b) from trial 19's own next-trial list, but launched under trial 19's unchanged `--run-name`/`--hub-branch` instead of a new one | 24 (early-stopped at 29/30) | 0.3531 (not the run's true min — see note) | 0.7751 at epoch 24 | Clean, isolated one-axis win over trial 19: +0.0100 F1 from `unfreeze_last_n` alone (only config change), bigger than trial 17's own smoothing jump. First trial since 15 to actually early-stop. Also: this run's `--run-name`/`--hub-branch` collided with trial 19's, overwriting its local log file and Hub checkpoint — see note |
| 20 (reg-trial-20, more-data) | `mf-span-reg-trial-20` | trial 19b's exact config, uncapped `max_train_pairs`/`max_val_pairs` (32025/9363 pairs, vs. every prior trial's fixed 20000/5000) — recommendation (a) from trial 19's next-trial list, with a distinct `--run-name`/`--hub-branch` this time | 30 (never plateaued — see note) | 0.3166 (the run's true min, coincides exactly with the best-F1 epoch) | 0.8080 at epoch 30 | Biggest single-axis win in the sweep: +0.0329 F1 over trial 19b, more than 3x trial 19b's own gain over trial 19. Still improving at epoch 30 (early stop never armed) and, for the first time since trial 15, val_loss min and best-F1 epoch coincide exactly — more data closes the val_loss/F1 divergence that every capacity/smoothing trial had been widening. **Extended to 50 epochs — see below and the extension note** |
| 21 (reg-trial-21, bigger-model) | `mf-span-reg-trial-21` | trial 20's exact config, `--model-name` `esm2_t12_35M_UR50D` → `esm2_t30_150M_UR50D` (~4.3x params), a distinct `--run-name`/`--hub-branch` again — testing the bigger-model axis ahead of the epoch-budget recommendation, not that recommendation itself | 30 (never plateaued — see note) | 0.3196 (true min, epoch 27 — 3 epochs before best-F1) | 0.8111 at epoch 30 — briefly a sweep-best by number, but see note | +0.0031 over trial 20's original 30-epoch result for ~2.8x the wall-clock (86279s vs. 31005s) — a poor return on cost, and since surpassed outright by trial 20's own epoch extension below (on the cheaper 35M model). User's call: not worth it, staying on the 35M model. See note |
| 20-ext (reg-trial-20, resumed to 50 epochs) | `mf-span-reg-trial-20` *(same run, extended)* | trial 20's exact config resumed via `--resume-from-checkpoint`/`--reset-early-stop-on-resume`, `--epochs` 30→50, after the original 30-epoch run never plateaued | 47 (of 50) | 0.3092 (true min, epoch 35 — 12 epochs before best-F1; see note) | **0.8203 at epoch 47 — new sweep-best, beats trial 21's bigger model too, on the 35M model** | +0.0123 F1 over the original 30-epoch checkpoint for +21287s (~5.9h) more, ~52293s (~14.5h) total — and now the practical *and* technical sweep-best at once, closing the model-size question the same direction the user already called it. See note |
| 22 (reg-trial-22, unfreeze4) | `mf-span-reg-trial-22` | trial 20's config (uncapped data, `label_smoothing=0.3`, `--batch-size 2 --grad-accum-steps 16`, `--learning-rate 1e-4`), `unfreeze_last_n` 3→4, fresh 30-epoch run (not a resume) | — (in progress) | — | — | **In progress.** Retests the capacity axis trial 10 found overfitting badly on at the old 20k/5k data scale — now that trial 20 showed uncapped data closes that overfitting outright, user is checking whether `unfreeze=4`'s earlier failure was a data-scale artifact rather than a fundamental capacity ceiling |

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

**Trial 13 (unfreeze2-epochs30-label-smoothing).** Reruns trial 12's exact config with three new
flags, all added in direct response to trial 12's findings: `--label-smoothing 0.1` (softens BCE
training targets toward 0.5, applied only to the training loss — validation loss still uses hard
targets, so it stays comparable to every prior trial's val_loss), `--checkpoint-selection f1` (first
trial to actually use it, instead of the `val_loss`-only default every prior trial used), and
`--early-stop-patience 5` (never triggered — the run completed all 30 epochs with the longest
no-improvement streak topping out at 2 epochs). Of these three, only label smoothing can affect the
training/validation curves themselves — `--checkpoint-selection` and `--early-stop-patience` only
change which epoch's weights get saved and whether the run stops early, neither of which happened
differently enough to matter here (the run went the full 30 epochs regardless). So despite three
flags differing from trial 12, this is a clean, unconfounded isolation of label smoothing's effect.
Full curve:

| epoch | train_loss | val_loss | val_f1 | new best (by F1)? |
|---|---|---|---|---|
| 1 | 0.5757 | 0.4834 | 0.4181 | yes |
| 2 | 0.5012 | 0.4428 | 0.5441 | yes |
| 3 | 0.4710 | 0.4190 | 0.5825 | yes |
| 4 | 0.4485 | 0.4015 | 0.6058 | yes |
| 5 | 0.4301 | 0.3817 | 0.6301 | yes |
| 6 | 0.4134 | 0.3670 | 0.6519 | yes |
| 7 | 0.3991 | 0.3546 | 0.6678 | yes |
| 8 | 0.3871 | 0.3466 | 0.6782 | yes |
| 9 | 0.3765 | 0.3441 | 0.6856 | yes |
| 10 | 0.3664 | 0.3380 | 0.6950 | yes |
| 11 | 0.3570 | 0.3320 | 0.7021 | yes |
| 12 | 0.3496 | 0.3300 | 0.7044 | yes |
| 13 | 0.3421 | 0.3180 | 0.7155 | yes |
| 14 | 0.3351 | 0.3291 | 0.7101 | no |
| 15 | 0.3303 | 0.3187 | 0.7247 | yes |
| 16 | 0.3239 | 0.3188 | 0.7264 | yes |
| 17 | 0.3185 | 0.3175 | 0.7279 | yes |
| 18 | 0.3141 | 0.3164 | 0.7289 | yes |
| 19 | 0.3101 | 0.3193 | 0.7287 | no |
| 20 | 0.3051 | 0.3182 | 0.7276 | no |
| 21 | 0.3009 | 0.3134 | 0.7427 | yes |
| 22 | 0.2973 | 0.3101 | 0.7392 | no |
| 23 | 0.2936 | 0.3076 | 0.7446 | yes |
| 24 | 0.2898 | 0.3140 | 0.7435 | no |
| 25 | 0.2870 | 0.3134 | 0.7443 | no |
| 26 | 0.2841 | 0.3139 | 0.7455 | yes |
| 27 | 0.2812 | 0.3118 | 0.7420 | no |
| 28 | 0.2787 | **0.3069 (min)** | **0.7539 (best F1 — saved/pushed checkpoint)** | yes |
| 29 | 0.2766 | 0.3209 | 0.7397 | no |
| 30 | 0.2744 | 0.3126 | 0.7481 | no |

Several findings:

1. **The val_loss/F1 divergence is almost gone.** val_loss falls essentially monotonically (with mild
   noise, e.g. epoch 14, 19-20, 24-25, 29) all the way to its epoch-28 minimum (0.3069), then rises
   only to 0.3126 by epoch 30 — a Δ0.0057 uptick, compared to trial 12's Δ0.234 rise over the same
   back half of training. Label smoothing did what it was added to do: it stopped BCE from rewarding
   the model for growing arbitrarily confident on already-correct tokens, so val_loss now tracks F1
   almost the whole way through instead of diverging from it after epoch ~10.
2. **Absolute quality improved too, not just the divergence.** This isn't only a "delays overfitting"
   story — at every directly-comparable epoch, trial 13 beats trial 12 on both metrics (val_loss is
   *lower*, not just later-rising): epoch 10 val_loss 0.3380 vs. trial 12's 0.3593, F1 0.6950 vs.
   0.6874; epoch 18 F1 0.7289 vs. 0.7215; epoch 28 F1 0.7539 vs. trial 12's peak of 0.7421 (itself the
   prior sweep-best). Label smoothing is acting as a real regularizer here, not just a cosmetic loss
   transform.
3. **First trial where the saved/pushed checkpoint actually is the best one.** With
   `--checkpoint-selection f1` finally exercised, the checkpoint written to disk and pushed to
   `kelmoir/cafa6-mf-span@main` is epoch 28's — genuinely the run's best F1 (0.7539), not a
   val_loss-selected earlier epoch sitting well below the run's true peak the way trials 10-12 all
   were. The val_loss/F1-selection gap that motivated the flag in the first place is now 0.0 for this
   trial (epoch 28 happens to be the val_loss minimum too).
4. **Precision/recall are noticeably more balanced than prior trials.** 0.7465/0.7615 at epoch 28 —
   close to even, versus the recall > precision skew seen since trial 7 (e.g. trial 7's 0.6329/0.7533,
   trial 11's 0.67/0.74). Not yet clear whether this is a label-smoothing effect or just noise; worth
   watching in future trials rather than treating as established.
5. **First real epoch-timing data in the sweep**, from this trial's `--compile`-adjacent timing
   feature: ~907-910s/epoch (~15.1 min), 27241.1s total (~7.57h) for the full 30 epochs on this
   machine's GPU at this data scale (20k/5k pairs, `unfreeze_last_n=2`) — useful as a reference for
   estimating future same-scale trials' wall-clock cost.

**Trial 14 (unfreeze2-epochs30-label-smoothing-effbatch32).** Reruns trial 13's exact config with
`--batch-size 4 --grad-accum-steps 8` instead of a plain `--batch-size 16` — an effective batch of
32 (double trial 13's), using the newly-added `--grad-accum-steps` flag and its accompanying
`val_loss`-normalization fix (both added this session in response to a real, measured bug: matched-
effective-batch runs at different `--batch-size` values were producing systematically different
`val_loss`, traced to the validation loop naively averaging per-batch means instead of a true
token-weighted average — see the script's git history for the fix). `--learning-rate` was left at
trial 13's `5e-5`, not scaled up for the larger batch. Full curve:

| epoch | train_loss | val_loss | val_f1 | new best (by F1)? |
|---|---|---|---|---|
| 1 | 0.5936 | 0.4938 | 0.4454 | yes |
| 2 | 0.5162 | 0.4624 | 0.5271 | yes |
| 3 | 0.4866 | 0.4326 | 0.5615 | yes |
| 4 | 0.4647 | 0.4248 | 0.5836 | yes |
| 5 | 0.4481 | 0.4023 | 0.5958 | yes |
| 6 | 0.4325 | 0.3883 | 0.6238 | yes |
| 7 | 0.4193 | 0.3740 | 0.6307 | yes |
| 8 | 0.4082 | 0.3654 | 0.6422 | yes |
| 9 | 0.3978 | 0.3591 | 0.6565 | yes |
| 10 | 0.3884 | 0.3541 | 0.6681 | yes |
| 11 | 0.3801 | 0.3469 | 0.6778 | yes |
| 12 | 0.3738 | 0.3442 | 0.6787 | yes |
| 13 | 0.3654 | 0.3359 | 0.6922 | yes |
| 14 | 0.3584 | 0.3386 | 0.6922 | yes |
| 15 | 0.3535 | 0.3351 | 0.7017 | yes |
| 16 | 0.3472 | 0.3303 | 0.7066 | yes |
| 17 | 0.3418 | 0.3232 | 0.7140 | yes |
| 18 | 0.3367 | 0.3268 | 0.7095 | no |
| 19 | 0.3325 | 0.3243 | 0.7174 | yes |
| 20 | 0.3271 | 0.3225 | 0.7152 | no |
| 21 | 0.3226 | 0.3227 | 0.7230 | yes |
| 22 | 0.3187 | 0.3156 | 0.7301 | yes |
| 23 | 0.3153 | 0.3157 | 0.7304 | yes |
| 24 | 0.3115 | 0.3296 | 0.7190 | no |
| 25 | 0.3084 | 0.3179 | 0.7324 | yes |
| 26 | 0.3050 | 0.3205 | 0.7337 | yes |
| 27 | 0.3019 | 0.3163 | 0.7317 | no |
| 28 | 0.2990 | **0.3140** | **0.7384 (best F1 — saved/pushed checkpoint)** | yes |
| 29 | 0.2965 | 0.3163 | 0.7348 | no |
| 30 | 0.2935 | 0.3194 | 0.7334 | no |

Findings:

1. **Worse quality at every matched epoch, not just at the end.** trial 14 trails trial 13 on both
   val_loss and F1 essentially throughout — epoch 10 (0.3541/0.6681 vs. trial 13's 0.3380/0.6950),
   epoch 20 (0.3225/0.7152 vs. 0.3182/0.7276), epoch 28-best (0.3140/0.7384 vs. 0.3069/0.7539, a
   0.0155 F1 gap). This isn't a late-training divergence like the val_loss/F1 story in trials 5-12 —
   it's a consistent lag from early epochs onward, a different failure signature.
2. **Likely cause: same LR, half as many optimizer steps per epoch, unscaled.** Effective batch 32
   on the same 20000-pair training set means ~625 optimizer steps/epoch vs. trial 13's ~1250 —
   literally half the gradient updates per epoch — while `--learning-rate` stayed at trial 13's
   `5e-5`. This is exactly the setup the "linear scaling rule" (Goyal et al. 2017) warns about:
   doubling batch size without proportionally increasing LR under-trains relative to the smaller-
   batch baseline at a fixed epoch budget. Not yet tested whether a scaled-up LR (or a doubled epoch
   budget, to match total step count instead of epoch count) closes this gap — flagging as the
   natural follow-up before concluding effective batch 32 is a worse setting in general.
3. **Substantially faster wall-clock despite the larger effective batch: 20719s vs. trial 13's
   27241s (~24% less total training time for the same 30 epochs).** Consistent with the earlier
   batch-size speed-test finding (smaller literal `--batch-size` trains faster per epoch due to less
   per-batch padding waste) — `--batch-size 4` here is cheap per step even though `--grad-accum-steps
   8` reaches a bigger effective batch than trial 13's plain `--batch-size 16` ever did. If a
   scaled-LR or matched-step-count rerun closes finding 2's quality gap, this configuration could be
   a strictly better (faster *and* as-good) setting than trial 13's.
4. **Confirms the grad-accum / val_loss fix works correctly in a real, full-scale trial** — not just
   the CPU smoke tests. `--checkpoint-selection f1` and `--push-to-hub` both worked as expected, and
   the checkpoint was pushed to a distinct `--hub-branch trial-14` (not overwriting trial 13's
   `main`), per the sweep's recurring-overwrite note.

**Trial 15 (unfreeze2-epochs30-trial-15, effbatch32+lr-scaled).** Directly answers trial 14's open
question: reruns its exact effective-batch-32 config (`--batch-size 4 --grad-accum-steps 8`) with
`--learning-rate 1e-4` instead of trial 13/14's `5e-5` — a 2x LR scale matching the 2x effective-batch
increase, per the linear scaling rule (Goyal et al. 2017). Full curve:

| epoch | train_loss | val_loss | val_f1 | new best (by F1)? |
|---|---|---|---|---|
| 1 | 0.5686 | 0.4724 | 0.4657 | yes |
| 2 | 0.4884 | 0.4268 | 0.5633 | yes |
| 3 | 0.4531 | 0.3908 | 0.6159 | yes |
| 4 | 0.4251 | 0.3824 | 0.6381 | yes |
| 5 | 0.4059 | 0.3571 | 0.6583 | yes |
| 6 | 0.3899 | 0.3421 | 0.6908 | yes |
| 7 | 0.3744 | 0.3293 | 0.7031 | yes |
| 8 | 0.3617 | 0.3281 | 0.7019 | no |
| 9 | 0.3519 | 0.3223 | 0.7204 | yes |
| 10 | 0.3418 | 0.3154 | 0.7243 | yes |
| 11 | 0.3328 | 0.3109 | 0.7295 | yes |
| 12 | 0.3272 | 0.3068 | 0.7360 | yes |
| 13 | 0.3185 | 0.3022 | 0.7452 | yes |
| 14 | 0.3132 | 0.3087 | 0.7409 | no |
| 15 | 0.3080 | 0.3045 | 0.7506 | yes |
| 16 | 0.3024 | 0.3087 | 0.7449 | no |
| 17 | 0.2975 | 0.3040 | 0.7476 | no |
| 18 | 0.2940 | 0.3056 | 0.7475 | no |
| 19 | 0.2897 | **0.3038 (min)** | **0.7544 (best F1 — saved/pushed checkpoint)** | yes |
| 20 | 0.2865 | 0.3089 | 0.7469 | no |
| 21 | 0.2825 | 0.3083 | 0.7505 | no |
| 22 | 0.2800 | 0.3001 | 0.7540 | no |
| 23 | 0.2765 | 0.3056 | 0.7492 | no |
| 24 | 0.2742 | 0.3086 | 0.7526 | no (5th consecutive epoch w/o improvement — early stop) |

Findings:

1. **Confirms the linear scaling rule hypothesis, and then some.** LR-scaling doesn't just close
   trial 14's gap versus trial 13 — trial 15 beats trial 13 at *every* matched epoch shown in trial
   14's table (epoch 10: F1 0.7243 vs. trial 13's 0.6950; epoch 20: 0.7469 vs. 0.7276), and reaches
   trial 13's final sweep-best (0.7539) by epoch 19, nine epochs before trial 13 got there. Final
   best F1 0.7544 is a new sweep-best, though only marginally above trial 13's 0.7539 — essentially a
   tie within noise, not a large jump.
2. **First trial where early stopping actually fired.** `--early-stop-patience 5` triggered after
   epoch 24 (5 epochs — 20 through 24 — with no F1 improvement over epoch 19's 0.7544). Trials 12-14
   all ran the full 30 epochs despite having the flag available; this is the first real confirmation
   the early-stopping logic works end-to-end on a full-scale GPU run, and it saved 6 epochs of
   otherwise-wasted compute (val_f1 was flat/noisy in the 0.749-0.754 range for those 6 epochs, not
   climbing).
3. **Substantially cheaper than both trial 13 and trial 14, from two compounding factors.** Total
   time 16677.7s vs. trial 13's 27241s (~39% less) and trial 14's 20719s (~19% less) — trial 14's
   smaller-literal-`--batch-size` speed advantage (less per-batch padding waste, see trial 14) plus
   early stopping's 6-epoch cutoff stack together. Combined with finding 1, trial 15's config is now
   strictly better than trial 13's: equal-or-better quality at a fraction of the wall-clock cost.
4. **val_loss stayed well-behaved throughout, same as trial 13/14** — no renewed divergence from
   label smoothing being "diluted" by the larger effective batch; val_loss tracks a tight ~0.30-0.31
   band from epoch ~9 onward, consistent with label smoothing's regularizing effect being orthogonal
   to the batch-size/LR axis.

**Trial 16 (trial-16, effbatch32/bs2-ga16+label_smoothing0.2, crashed).** Same idea as trial 17 below
(test `--label-smoothing 0.2` on top of trial 15's config) but with `--batch-size 2
--grad-accum-steps 16` instead of trial 15's `--batch-size 4 --grad-accum-steps 8` (same effective
batch 32, different split). **Crashed after epoch 17/30 for an unknown reason** (not a manual
interruption) — no `[SUMMARY]` line, so it isn't a completed trial.
`experiments/logs/mf_span_mf-span-reg-trial-16.jsonl` is kept as a record of the partial curve (it
was tracking trial 15 closely through epoch 17: F1 0.7514 vs. trial 15's later epochs), but no row
conclusion is drawn from it — superseded by trial 17, which reruns the same label-smoothing change
against trial 15's exact batch/grad-accum split for a clean isolated comparison instead of
confounding it with a second batch-split change. One notable data point for whoever revisits the
crash: trial 16's per-epoch time was consistently faster than trial 17's (~577s avg over its 17
completed epochs vs. trial 17's ~672s avg — the same batch_size=2-vs-4 padding-waste effect seen
since trial 14, just more pronounced at bs=2). The crash's cause is unconfirmed, but the speed
difference is a real, measured signal (smaller literal batch size means far more optimizer
micro-steps per epoch — 10000 vs. trial 17's 5000 — which is at least a plausible place to look:
e.g. an OOM/edge case tied to a specific micro-batch, or instability from noisier per-step gradients
at that granularity) rather than a coincidence, worth checking first if this batch/grad-accum split
is revisited.

**Trial 17 (reg-trial-17, label_smoothing0.2).** Isolates the label-smoothing axis from trial 15's
"Recommended next trial" list: trial 15's exact config (`--batch-size 4 --grad-accum-steps 8
--learning-rate 1e-4`), only `--label-smoothing` raised 0.1→0.2. Full curve:

| epoch | train_loss | val_loss | precision | recall | val_f1 | new best (by F1)? |
|---|---|---|---|---|---|---|
| 1 | 0.6206 | 0.4928 | 0.5601 | 0.4030 | 0.4687 | yes |
| 2 | 0.5613 | 0.4500 | 0.5307 | 0.6008 | 0.5636 | yes |
| 3 | 0.5354 | 0.4128 | 0.5879 | 0.6502 | 0.6175 | yes |
| 4 | 0.5152 | 0.4077 | 0.5714 | 0.7251 | 0.6391 | yes |
| 5 | 0.5009 | 0.3861 | 0.6156 | 0.7114 | 0.6601 | yes |
| 6 | 0.4891 | 0.3671 | 0.6629 | 0.7296 | 0.6947 | yes |
| 7 | 0.4779 | 0.3561 | 0.6818 | 0.7342 | 0.7071 | yes |
| 8 | 0.4688 | 0.3538 | 0.6810 | 0.7286 | 0.7040 | no |
| 9 | 0.4616 | 0.3497 | 0.7089 | 0.7373 | 0.7228 | yes |
| 10 | 0.4548 | 0.3403 | 0.6922 | 0.7671 | 0.7277 | yes |
| 11 | 0.4482 | 0.3343 | 0.7157 | 0.7515 | 0.7331 | yes |
| 12 | 0.4443 | 0.3313 | 0.6909 | 0.7939 | 0.7388 | yes |
| 13 | 0.4385 | 0.3260 | 0.7238 | 0.7762 | 0.7491 | yes |
| 14 | 0.4347 | 0.3306 | 0.7334 | 0.7534 | 0.7433 | no |
| 15 | 0.4311 | 0.3257 | 0.7550 | 0.7502 | 0.7526 | yes |
| 16 | 0.4272 | 0.3285 | 0.7266 | 0.7648 | 0.7452 | no |
| 17 | 0.4239 | 0.3212 | 0.7297 | 0.7761 | 0.7522 | no |
| 18 | 0.4214 | 0.3232 | 0.7350 | 0.7654 | 0.7499 | no |
| 19 | 0.4185 | 0.3230 | 0.7521 | 0.7623 | 0.7572 | yes |
| 20 | 0.4160 | 0.3278 | 0.7744 | 0.7220 | 0.7473 | no |
| 21 | 0.4137 | 0.3258 | 0.7608 | 0.7471 | 0.7539 | no |
| 22 | 0.4113 | **0.3176 (min)** | 0.7423 | 0.7678 | 0.7548 | no |
| 23 | 0.4092 | 0.3231 | 0.7514 | 0.7550 | 0.7532 | no |
| 24 | 0.4074 | 0.3232 | 0.7708 | 0.7478 | 0.7591 | yes |
| 25 | 0.4053 | 0.3247 | 0.7644 | 0.7473 | 0.7558 | no |
| 26 | 0.4035 | 0.3190 | 0.7667 | 0.7617 | **0.7642 (best F1 — saved/pushed checkpoint)** | yes |
| 27 | 0.4023 | 0.3194 | 0.7542 | 0.7605 | 0.7573 | no |
| 28 | 0.4009 | 0.3208 | 0.7551 | 0.7528 | 0.7540 | no |
| 29 | 0.4000 | 0.3227 | 0.7700 | 0.7378 | 0.7536 | no |
| 30 | 0.3981 | 0.3220 | 0.7871 | 0.7345 | 0.7599 | no (only 4 epochs since last best — early stop never armed) |

Findings:

1. **Higher label smoothing is a further real, unconfounded win.** Trial 17 beats trial 15 at every
   directly matched epoch (epoch 19: F1 0.7572 vs. 0.7544; epoch 24: 0.7591 vs. trial 15's
   early-stopped final of 0.7526), and its peak of **0.7642 at epoch 26 is a new sweep-best**,
   clearing trial 15's 0.7544 by a real margin (+0.0098) — a bigger jump than trial 15's own
   marginal +0.0005 over trial 13.
2. **Early stopping didn't fire this time, so the trial cost more.** The best epoch kept moving
   until 26, leaving only 4 flat epochs (27-30) by the time the 30-epoch budget ran out — one short
   of `--early-stop-patience 5`. Total time 20180.2s, ~21% more than trial 15's 16677.7s, though
   still ~26% cheaper than trial 13's 27241s (the batch-size speed advantage alone, without early
   stopping's help this time).
3. **A small val_loss/F1 divergence reappears.** The run's true val_loss minimum is 0.3176 at epoch
   22, but the best-F1 epoch (26) sits slightly higher at val_loss 0.3190 — in trial 15 these
   coincided exactly (both at epoch 19). Not a real regression (both are in the same tight ~0.32
   band, and F1 is the selection criterion this sweep cares about), but a sign that pushing label
   smoothing higher loosens rather than tightens the val_loss/F1 alignment.
4. Trial 16's abandoned partial curve (batch_size=2/grad_accum=16 split) tracked trial 17 closely
   through its last completed epoch (17: F1 0.7514 vs. trial 17's 0.7522), suggesting the literal
   batch/grad-accum split matters little once the effective batch and LR are fixed — consistent with
   trial 14/15's existing read that the split mainly affects wall-clock (padding waste), not the
   learning curve itself.

**Trial 18 (trial-18, label_smoothing0.3, crashed).** Trial 17's exact config with
`--label-smoothing` raised 0.2→0.3, the first (a) option from trial 17's "Recommended next trial"
list. Completed epoch 14 cleanly (F1 0.7430, matching trial 19's epoch 14 below almost exactly —
see trial 19's note) then crashed during epoch 15 with `CUDA error: unspecified launch failure`.
Diagnosed directly, not left as an unknown this time: `dmesg`/`journalctl -k` showed `NVRM: Xid
(PCI:0000:06:00): 13, Graphics Exception` timestamped to the second against the log file's last
write (00:25:49 vs. 00:25:45) — a GPU-side graphics-engine exception, and on this card (RTX 5060
Ti, driver 595.84 at the time) most plausibly new-architecture driver immaturity under a sustained
multi-hour load rather than anything wrong with this script or this config. This trial predates
`--resume-from-checkpoint`/`experiments/scripts/run_mf_span_trial.sh` (both added the same
session, directly motivated by this crash and trial 16's earlier one) — there was no
`latest_checkpoint` to resume from, so the only option was a full clean rerun. `training_state.pt`
now exists precisely so a future crash like this one costs at most one epoch instead of the whole
trial. `experiments/logs/mf_span_mf-span-reg-trial-18.jsonl` is kept as a record of the partial
curve; no row conclusion is drawn from it beyond the crash itself — superseded by trial 19.

**Trial 19 (reg-trial-19, label_smoothing0.3).** Clean rerun of trial 18's exact config (trial 17 +
`--label-smoothing 0.3`) after the crash, run via the new `run_mf_span_trial.sh` wrapper (didn't
end up needing a retry — completed all 30 epochs on the first attempt). Full curve:

| epoch | train_loss | val_loss | precision | recall | val_f1 | new best (by F1)? |
|---|---|---|---|---|---|---|
| 1 | 0.6631 | 0.5182 | 0.5636 | 0.4043 | 0.4708 | yes |
| 2 | 0.6199 | 0.4793 | 0.5347 | 0.5977 | 0.5644 | yes |
| 3 | 0.6010 | 0.4417 | 0.5959 | 0.6434 | 0.6187 | yes |
| 4 | 0.5863 | 0.4387 | 0.5748 | 0.7237 | 0.6407 | yes |
| 5 | 0.5759 | 0.4210 | 0.6182 | 0.7146 | 0.6629 | yes |
| 6 | 0.5673 | 0.4005 | 0.6676 | 0.7280 | 0.6965 | yes |
| 7 | 0.5592 | 0.3915 | 0.6832 | 0.7359 | 0.7086 | yes |
| 8 | 0.5526 | 0.3884 | 0.6859 | 0.7264 | 0.7056 | no |
| 9 | 0.5475 | 0.3863 | 0.7113 | 0.7370 | 0.7239 | yes |
| 10 | 0.5426 | 0.3762 | 0.6989 | 0.7642 | 0.7301 | yes |
| 11 | 0.5381 | 0.3703 | 0.7195 | 0.7536 | 0.7361 | yes |
| 12 | 0.5353 | 0.3672 | 0.6939 | 0.7918 | 0.7396 | yes |
| 13 | 0.5312 | 0.3628 | 0.7266 | 0.7755 | 0.7503 | yes |
| 14 | 0.5287 | 0.3666 | 0.7376 | 0.7484 | 0.7430 | no |
| 15 | 0.5260 | 0.3605 | 0.7529 | 0.7559 | 0.7544 | yes |
| 16 | 0.5232 | 0.3639 | 0.7321 | 0.7601 | 0.7459 | no |
| 17 | 0.5210 | 0.3572 | 0.7328 | 0.7746 | 0.7531 | no |
| 18 | 0.5192 | 0.3569 | 0.7339 | 0.7707 | 0.7518 | no |
| 19 | 0.5170 | 0.3583 | 0.7566 | 0.7570 | 0.7568 | yes |
| 20 | 0.5156 | 0.3622 | 0.7690 | 0.7278 | 0.7478 | no |
| 21 | 0.5135 | 0.3606 | 0.7587 | 0.7476 | 0.7531 | no |
| 22 | 0.5120 | **0.3505 (min)** | 0.7485 | 0.7670 | 0.7576 | yes |
| 23 | 0.5104 | 0.3566 | 0.7535 | 0.7582 | 0.7559 | no |
| 24 | 0.5092 | 0.3597 | 0.7763 | 0.7407 | 0.7581 | yes |
| 25 | 0.5078 | 0.3565 | 0.7634 | 0.7519 | 0.7576 | no |
| 26 | 0.5064 | 0.3538 | 0.7673 | 0.7630 | **0.7651 (best F1 — saved/pushed checkpoint)** | yes |
| 27 | 0.5055 | 0.3536 | 0.7557 | 0.7674 | 0.7615 | no |
| 28 | 0.5047 | 0.3568 | 0.7631 | 0.7476 | 0.7553 | no |
| 29 | 0.5037 | 0.3563 | 0.7663 | 0.7404 | 0.7531 | no |
| 30 | 0.5023 | 0.3553 | 0.7891 | 0.7341 | 0.7606 | no (only 4 epochs since last best — early stop never armed) |

Findings:

1. **A further win, but a much smaller one — the plateau signature, not a regression.** Trial 19
   beats trial 17 at 25 of 30 epochs (loses at 5, all by ≤0.001 — noise), and its peak of **0.7651
   at epoch 26 is a new sweep-best**, but the margin over trial 17 is only **+0.0009** — an order of
   magnitude smaller than trial 17's own +0.0098 over trial 15. Label smoothing hasn't started
   hurting, but it's stopped paying off.
2. **The val_loss/F1 divergence widened further, as trial 17's note flagged as the thing to watch.**
   True val_loss minimum is 0.3505 at epoch 22 (same epoch as trial 17's minimum, interestingly),
   best-F1 epoch is 26 at val_loss 0.3538 — a gap of 0.0033, more than double trial 17's 0.0014 gap.
   Consistent with trial 17's note 3: pushing smoothing further keeps loosening the val_loss/F1
   alignment rather than tightening it.
3. **Same cost profile as trial 17.** Early stopping didn't fire again (4 flat epochs after the
   epoch-26 peak, one short of patience 5) — 20118.8s, essentially identical to trial 17's 20180.2s.
4. **Verdict: treat label smoothing 0.2-0.3 as the plateau, per trial 17's own pre-declared decision
   rule.** That rule was phrased as "if 0.3 regresses, treat 0.2 as the plateau" — 0.3 didn't
   technically regress, but a 10x collapse in marginal gain plus a widening val_loss/F1 gap is the
   same signal the rule was written to catch. Not worth a 0.4 trial; time to switch axes (see
   "Recommended next trial" below). Trial 18's crash (same config, see its note above) is unrelated
   to this — a confirmed GPU-side Xid 13 exception, not a sign 0.3 destabilizes training.

**Trial 19b (reg-trial-19, unfreeze3, label_smoothing0.3).** Trial 19's exact config with
`--unfreeze-last-n` raised 2→3 — the (b) option from trial 19's own "Recommended next trial" list —
but launched with trial 19's `--run-name`/`--hub-branch` left unchanged instead of given a new one,
which overwrote trial 19's local log file and Hub checkpoint (see finding 4 below). Full curve:

| epoch | train_loss | val_loss | precision | recall | val_f1 | new best (by F1)? |
|---|---|---|---|---|---|---|
| 1 | 0.6590 | 0.5058 | 0.5706 | 0.4039 | 0.4730 | yes |
| 2 | 0.6134 | 0.4650 | 0.5452 | 0.6497 | 0.5929 | yes |
| 3 | 0.5926 | 0.4269 | 0.6206 | 0.6638 | 0.6415 | yes |
| 4 | 0.5768 | 0.4257 | 0.5716 | 0.7801 | 0.6598 | yes |
| 5 | 0.5670 | 0.4115 | 0.6249 | 0.7526 | 0.6829 | yes |
| 6 | 0.5573 | 0.3925 | 0.6650 | 0.7546 | 0.7070 | yes |
| 7 | 0.5498 | 0.3840 | 0.6974 | 0.7431 | 0.7196 | yes |
| 8 | 0.5430 | 0.3790 | 0.7128 | 0.7319 | 0.7222 | yes |
| 9 | 0.5379 | 0.3795 | 0.7075 | 0.7646 | 0.7350 | yes |
| 10 | 0.5329 | 0.3721 | 0.6864 | 0.7894 | 0.7343 | no |
| 11 | 0.5286 | 0.3665 | 0.7234 | 0.7615 | 0.7419 | yes |
| 12 | 0.5254 | 0.3642 | 0.7265 | 0.7764 | 0.7507 | yes |
| 13 | 0.5218 | 0.3550 | 0.7292 | 0.7917 | 0.7592 | yes |
| 14 | 0.5190 | 0.3632 | 0.7374 | 0.7630 | 0.7499 | no |
| 15 | 0.5161 | 0.3596 | 0.7558 | 0.7651 | 0.7604 | yes |
| 16 | 0.5136 | 0.3566 | 0.7445 | 0.7713 | 0.7576 | no |
| 17 | 0.5116 | 0.3584 | 0.7426 | 0.7630 | 0.7526 | no |
| 18 | 0.5097 | 0.3532 | 0.7450 | 0.7730 | 0.7587 | no |
| 19 | 0.5078 | 0.3511 | 0.7598 | 0.7692 | 0.7645 | yes |
| 20 | 0.5061 | 0.3589 | 0.7846 | 0.7369 | 0.7600 | no |
| 21 | 0.5048 | 0.3602 | 0.7542 | 0.7643 | 0.7592 | no |
| 22 | 0.5034 | **0.3487 (min, tied)** | 0.7646 | 0.7719 | 0.7683 | yes |
| 23 | 0.5018 | 0.3516 | 0.7872 | 0.7469 | 0.7665 | no |
| 24 | 0.5013 | 0.3531 | 0.7877 | 0.7630 | **0.7751 (best F1 — saved/pushed checkpoint)** | yes |
| 25 | 0.4998 | **0.3487 (min, tied)** | 0.7718 | 0.7627 | 0.7672 | no |
| 26 | 0.4991 | 0.3492 | 0.7660 | 0.7717 | 0.7688 | no |
| 27 | 0.4984 | 0.3483 | 0.7650 | 0.7770 | 0.7709 | no |
| 28 | 0.4976 | 0.3527 | 0.7886 | 0.7564 | 0.7722 | no |
| 29 | 0.4969 | 0.3508 | 0.7850 | 0.7548 | 0.7696 | no (5 flat epochs since epoch 24 — early stop fires) |

Findings:

1. **A real, isolated capacity win — bigger than trial 17's own smoothing jump.** Only
   `unfreeze_last_n` changed vs. trial 19 (2→3). Peak F1 **0.7751 at epoch 24** beats trial 19's 0.7651
   by **+0.0100** — larger than trial 17's own +0.0098 gain from smoothing 0.1→0.2, and unlike trial
   19's marginal 0.3-vs-0.2 gain, this one isn't plateau-sized.
2. **First trial since 15 to actually early-stop** — 5 flat epochs after the epoch-24 peak (25-29),
   stopping at 29/30 rather than running the full budget.
3. **Val_loss/F1 divergence widens further still.** True val_loss minimum (0.3487) is now hit *twice*
   (epochs 22 and 25), while the best-F1 epoch (24) sits at 0.3531 — a gap of 0.0044, wider than trial
   19's 0.0033 and trial 17's 0.0014. The pattern trial 17 first flagged keeps loosening, now along the
   capacity axis too, not just smoothing.
4. **Cost: early stopping saved an epoch, but it didn't save wall-clock.** 21781.0s total, ~8% *more*
   than trial 19's 20118.8s despite stopping one epoch early — the extra unfrozen layer's
   forward/backward cost outweighs the 1-epoch saving (~750-800s/epoch here vs. trial 19's ~740s).
5. **Run-name/hub-branch collision — trial 19's original checkpoint is gone.** This run reused
   `--run-name mf-span-reg-trial-19 --hub-branch trial-19` unchanged, which is exactly the mistake the
   bottom of this doc already warns against. It overwrote `experiments/logs/mf_span_mf-span-reg-trial-19.jsonl`
   in place (trial 19's original 30-epoch curve now only exists in git history, commit `d383a6c`) and
   pushed over the Hub checkpoint on `kelmoir/cafa6-mf-span@trial-19` — trial 19's original
   (`unfreeze_last_n=2`, F1 0.7651) checkpoint is no longer retrievable from either place. Not a data
   loss that matters much given trial 19b strictly dominates it, but worth naming so it isn't mistaken
   for a live checkpoint later.
6. **`mf-span-reg-trial-20`'s config-only log, resolved: it's the real next (more-data) trial, just
   getting started concurrently.** `experiments/logs/mf_span_mf-span-reg-trial-20.jsonl`'s single
   config line (uncapped `max_train_pairs`/`max_val_pairs`, `--batch-size 2 --grad-accum-steps 16`)
   was being launched by the user while this session was processing trial 19b's paste — its sparse
   state (no epoch lines yet, no process visible when checked) simply reflected how early it was, not
   an interrupted or abandoned run. `models/mf-span-reg-trial-20/`'s checkpoint dir (through epoch 18,
   metrics identical to this run's epochs 15/18 above, under the old capped-pairs `unfreeze_last_n=3`
   config) is a separate, stale leftover from an earlier attempt under the same name — harmless,
   ignore it. Per the user: the Hub-side naming collision this caused for trial 19b (finding 5 above)
   is likewise fine to leave as-is, not worth recovering.

**Trial 20 (reg-trial-20, more-data).** Trial 19b's exact config with `max_train_pairs`/
`max_val_pairs` uncapped (32025/9363 pairs, up from every prior trial's fixed 20000/5000) — the (a)
option from trial 19's "Recommended next trial" list, and this time launched with a genuinely new
`--run-name`/`--hub-branch` (no collision). Batch/grad-accum reverted to trial 16's split
(`--batch-size 2 --grad-accum-steps 16`, same effective batch 32 as trial 19b's `--batch-size 4
--grad-accum-steps 8`) — per the standing read that the split mainly affects wall-clock, not the
curve. Full curve:

| epoch | train_loss | val_loss | precision | recall | val_f1 | new best (by F1)? |
|---|---|---|---|---|---|---|
| 1 | 0.6472 | 0.4721 | 0.5595 | 0.5715 | 0.5654 | yes |
| 2 | 0.5978 | 0.4198 | 0.6251 | 0.7102 | 0.6649 | yes |
| 3 | 0.5780 | 0.4033 | 0.6281 | 0.7812 | 0.6963 | yes |
| 4 | 0.5642 | 0.3886 | 0.7187 | 0.7143 | 0.7165 | yes |
| 5 | 0.5549 | 0.3781 | 0.6836 | 0.7745 | 0.7262 | yes |
| 6 | 0.5477 | 0.3632 | 0.7057 | 0.7926 | 0.7466 | yes |
| 7 | 0.5411 | 0.3569 | 0.7305 | 0.7864 | 0.7574 | yes |
| 8 | 0.5363 | 0.3536 | 0.7238 | 0.8182 | 0.7681 | yes |
| 9 | 0.5314 | 0.3416 | 0.7364 | 0.8249 | 0.7781 | yes |
| 10 | 0.5272 | 0.3439 | 0.7308 | 0.8126 | 0.7696 | no |
| 11 | 0.5241 | 0.3371 | 0.7479 | 0.8166 | 0.7807 | yes |
| 12 | 0.5205 | 0.3404 | 0.7539 | 0.8194 | 0.7853 | yes |
| 13 | 0.5177 | 0.3388 | 0.7588 | 0.8106 | 0.7838 | no |
| 14 | 0.5151 | 0.3362 | 0.7388 | 0.8313 | 0.7823 | no |
| 15 | 0.5124 | 0.3353 | 0.7752 | 0.8160 | 0.7951 | yes |
| 16 | 0.5103 | 0.3262 | 0.7595 | 0.8279 | 0.7922 | no |
| 17 | 0.5086 | 0.3302 | 0.7931 | 0.8098 | 0.8014 | yes |
| 18 | 0.5067 | 0.3268 | 0.7771 | 0.8174 | 0.7968 | no |
| 19 | 0.5053 | 0.3276 | 0.7817 | 0.8115 | 0.7963 | no |
| 20 | 0.5036 | 0.3213 | 0.7872 | 0.8267 | 0.8065 | yes |
| 21 | 0.5025 | 0.3202 | 0.7807 | 0.8225 | 0.8010 | no |
| 22 | 0.5015 | 0.3192 | 0.7790 | 0.8231 | 0.8005 | no |
| 23 | 0.4998 | 0.3257 | 0.7981 | 0.8066 | 0.8023 | no |
| 24 | 0.4995 | 0.3182 | 0.7801 | 0.8208 | 0.7999 | no |
| 25 | 0.4983 | 0.3268 | 0.8138 | 0.7997 | 0.8067 | yes |
| 26 | 0.4976 | 0.3234 | 0.8033 | 0.8046 | 0.8040 | no |
| 27 | 0.4967 | 0.3196 | 0.8032 | 0.8103 | 0.8067 | yes (tied) |
| 28 | 0.4961 | 0.3206 | 0.7917 | 0.8129 | 0.8021 | no |
| 29 | 0.4952 | 0.3178 | 0.8041 | 0.8089 | 0.8065 | no |
| 30 | 0.4948 | **0.3166 (min)** | 0.7898 | 0.8271 | **0.8080 (best F1 — saved/pushed checkpoint)** | yes |

Findings:

1. **By far the largest single-axis win in the sweep.** F1 **0.8080 at epoch 30** beats trial 19b's
   0.7751 by **+0.0329** — more than 3x trial 19b's own +0.0100 gain over trial 19, and easily the
   biggest jump since the very early trials. More data was correctly the highest-leverage untested
   lever, as flagged since trial 15/17.
2. **Never plateaued — the 30-epoch budget is now the binding constraint, not overfitting.** Every
   prior trial's F1 curve flattened or early-stopped well before epoch 30 (trial 15 at 24, trial 17/19
   ran the full 30 already flat, trial 19b stopped at 29). This run's best epoch is the *last* one,
   with val_f1 still net-rising through the final stretch (0.8067→0.8080, epochs 25-30, through some
   noise) — early stop never came close to arming. Strong signal that more epochs (e.g. `--epochs 40`
   or `50`) is now a live, untested lever at this data scale, not wasted budget.
3. **The val_loss/F1 divergence — which every capacity/smoothing trial since 17 had been widening —
   is gone.** True val_loss minimum (0.3166) lands on the exact same epoch as the best-F1 checkpoint
   (30), a Δ0.0000 gap versus trial 19b's 0.0044 and trial 19's 0.0033. Consistent with more training
   data reducing variance/overfitting directly, rather than needing a regularization workaround for
   it.
4. **Substantially more expensive, but proportionate to the win.** 31004.7s total (~8.6h), the most
   expensive trial in the sweep — ~42% more than trial 19b's 21781.0s, roughly tracking the ~60%
   increase in train pairs (32025 vs. 20000) plus `--batch-size 2`'s per-step overhead (~1030s/epoch
   here vs. trial 19b's ~750-800s). Worth it for a +0.0329 F1 win at a new sweep-best, but real enough
   that a further epoch-budget increase (finding 2) should be planned as a deliberate cost tradeoff,
   not assumed free.

**Trial 21 (reg-trial-21, bigger-model).** Trial 20's exact config with `--model-name` swapped
`esm2_t12_35M_UR50D` → `esm2_t30_150M_UR50D` (30 layers vs. 12, ~4.3x the parameters) — testing the
model-size axis directly, ahead of trial 20's own "extend epochs" recommendation, presumably as a
scouting move before committing to the planned bigger-model phase. Everything else unchanged
(uncapped pairs, `unfreeze_last_n=3`, `label_smoothing=0.3`, `--batch-size 2 --grad-accum-steps 16`,
`--learning-rate 1e-4`), and this time launched with a genuinely new `--run-name`/`--hub-branch`
(no collision). Full curve:

| epoch | train_loss | val_loss | precision | recall | val_f1 | new best (by F1)? |
|---|---|---|---|---|---|---|
| 1 | 0.6472 | 0.4629 | 0.5718 | 0.6055 | 0.5882 | yes |
| 2 | 0.5974 | 0.4337 | 0.5619 | 0.7705 | 0.6499 | yes |
| 3 | 0.5769 | 0.4069 | 0.6604 | 0.6914 | 0.6755 | yes |
| 4 | 0.5632 | 0.3919 | 0.6346 | 0.8223 | 0.7164 | yes |
| 5 | 0.5523 | 0.3714 | 0.6730 | 0.8161 | 0.7377 | yes |
| 6 | 0.5444 | 0.3634 | 0.7262 | 0.7923 | 0.7578 | yes |
| 7 | 0.5383 | 0.3539 | 0.7251 | 0.8073 | 0.7640 | yes |
| 8 | 0.5331 | 0.3470 | 0.7410 | 0.8089 | 0.7735 | yes |
| 9 | 0.5285 | 0.3409 | 0.7399 | 0.8276 | 0.7813 | yes |
| 10 | 0.5253 | 0.3459 | 0.7600 | 0.8024 | 0.7806 | no |
| 11 | 0.5218 | 0.3352 | 0.7568 | 0.8161 | 0.7853 | yes |
| 12 | 0.5189 | 0.3375 | 0.7535 | 0.8244 | 0.7874 | yes |
| 13 | 0.5157 | 0.3357 | 0.7660 | 0.8164 | 0.7904 | yes |
| 14 | 0.5138 | 0.3392 | 0.7583 | 0.8217 | 0.7887 | no |
| 15 | 0.5113 | 0.3325 | 0.7613 | 0.8243 | 0.7916 | yes |
| 16 | 0.5091 | 0.3339 | 0.7684 | 0.8108 | 0.7890 | no |
| 17 | 0.5075 | 0.3265 | 0.7695 | 0.8274 | 0.7974 | yes |
| 18 | 0.5057 | 0.3269 | 0.7753 | 0.8231 | 0.7985 | yes |
| 19 | 0.5045 | 0.3256 | 0.7735 | 0.8261 | 0.7989 | yes |
| 20 | 0.5031 | 0.3288 | 0.7795 | 0.8178 | 0.7982 | no |
| 21 | 0.5019 | 0.3235 | 0.7839 | 0.8244 | 0.8037 | yes |
| 22 | 0.5008 | 0.3236 | 0.7826 | 0.8228 | 0.8022 | no |
| 23 | 0.4996 | 0.3334 | 0.7849 | 0.8196 | 0.8019 | no |
| 24 | 0.4989 | 0.3232 | 0.7919 | 0.8135 | 0.8026 | no |
| 25 | 0.4978 | 0.3273 | 0.8084 | 0.8037 | 0.8060 | yes |
| 26 | 0.4972 | 0.3219 | 0.7863 | 0.8213 | 0.8034 | no |
| 27 | 0.4965 | **0.3196 (min)** | 0.7962 | 0.8200 | 0.8079 | yes |
| 28 | 0.4958 | 0.3261 | 0.8046 | 0.8113 | 0.8080 | yes |
| 29 | 0.4950 | 0.3254 | 0.8054 | 0.8072 | 0.8063 | no |
| 30 | 0.4947 | 0.3257 | 0.8168 | 0.8054 | **0.8111 (best F1 — saved/pushed checkpoint)** | yes |

Findings:

1. **A real but marginal win, at a cost far out of proportion to it.** F1 **0.8111 at epoch 30** edges
   out trial 20's 0.8080 by only **+0.0031** — an order of magnitude smaller than trial 20's own
   +0.0329 over trial 19b — while total wall-clock jumped to 86279.0s (~24.0h), **2.78x** trial 20's
   31004.7s, for ~4.3x the backbone parameters. Squarely the "138x slower for far-from-138x-better"
   pattern the main pipeline's own LoRA note already warned about at a much bigger scale (see
   `CLAUDE.md`'s trial-067 caveat) — same shape, first time it's shown up on this side-experiment.
2. **Never plateaued, same as trial 20.** Best epoch is again the last one, val_f1 still net-rising
   through the closing stretch despite several no-improve epochs along the way — early stop (patience
   5) never came close to arming. Confirms the epoch-budget ceiling from trial 20's finding 2 isn't a
   quirk of the smaller model; it would very likely keep climbing past 30 here too. Moot for now, since
   the model-size axis itself is being dropped rather than pushed further (see finding 3 and this
   trial's "next" note below).
3. **The val_loss/F1 divergence trial 20 had closed to zero reopens slightly.** True val_loss min
   (0.3196) lands at epoch 27, 3 epochs before the best-F1 checkpoint (30) — a small Δ, not the sharp
   widening seen across the capacity/smoothing trials (17→19→19b), but a real reappearance after
   trial 20's exact Δ0.0000. Not enough data points yet to say whether this tracks model size or is
   just run-to-run noise at this scale.
4. **User's call: the bigger model isn't worth it, staying on `esm2_t12_35M_UR50D`.** This was a
   scouting trial for the model-size axis, run ahead of trial 20's "extend epochs" recommendation
   rather than as a continuation of it. Given +0.0031 F1 for +178% wall-clock, the user is standing pat
   on the 35M model rather than committing to the bigger one — reopening trial 20's own epoch-budget
   recommendation as the live next step, on the 35M model.

**Trial 20 — extended to 50 epochs (epochs 31-50).** Per trial 20's own "extend epochs" recommendation
(reopened as the live next step after the user declined trial 21's bigger-model trade), the original
30-epoch run was resumed in place via `--resume-from-checkpoint models/mf-span-reg-trial-20/latest_checkpoint
--epochs 40` — actually run to `--epochs 50` — plus `--reset-early-stop-on-resume` (so the streak from the
original run's own still-improving finish didn't immediately arm early stopping). Same `--run-name`/
`--hub-branch` (`trial-20`) by design — this is a continuation of trial 20, not a new trial identity. This
was also the first real exercise of the resume path's RNG-state restoration, which crashed on the first
attempt (`TypeError: RNG state must be a torch.ByteTensor` — a latent device-placement bug in the existing
resume code, not a regression from this session's wrapper/flag changes) and was fixed by moving the
restored RNG tensors to CPU before calling `torch.set_rng_state()`/`torch.cuda.set_rng_state_all()` (commit
`ea9a9c2`) before this extension was rerun successfully. Curve, epochs 31-50 (continuing directly from
epoch 30's row in trial 20's table above):

| epoch | train_loss | val_loss | precision | recall | val_f1 | new best (by F1)? |
|---|---|---|---|---|---|---|
| 31 | 0.4944 | 0.3254 | 0.8246 | 0.7847 | 0.8042 | no |
| 32 | 0.4938 | 0.3251 | 0.8069 | 0.8033 | 0.8051 | no |
| 33 | 0.4932 | 0.3229 | 0.8264 | 0.7897 | 0.8077 | no |
| 34 | 0.4931 | 0.3210 | 0.8284 | 0.7966 | 0.8122 | yes |
| 35 | 0.4924 | **0.3092 (min)** | 0.7930 | 0.8288 | 0.8105 | no |
| 36 | 0.4923 | 0.3183 | 0.8229 | 0.8034 | 0.8131 | yes |
| 37 | 0.4917 | 0.3186 | 0.8219 | 0.8007 | 0.8111 | no |
| 38 | 0.4915 | 0.3198 | 0.8213 | 0.8032 | 0.8121 | no |
| 39 | 0.4912 | 0.3142 | 0.8068 | 0.8157 | 0.8112 | no |
| 40 | 0.4908 | 0.3207 | 0.8331 | 0.7952 | 0.8137 | yes |
| 41 | 0.4906 | 0.3224 | 0.8331 | 0.7902 | 0.8111 | no |
| 42 | 0.4904 | 0.3198 | 0.8315 | 0.7944 | 0.8125 | no |
| 43 | 0.4901 | 0.3266 | 0.8452 | 0.7797 | 0.8111 | no |
| 44 | 0.4896 | 0.3231 | 0.8410 | 0.7877 | 0.8134 | no |
| 45 | 0.4897 | 0.3183 | 0.8360 | 0.7987 | 0.8169 | yes |
| 46 | 0.4893 | 0.3202 | 0.8397 | 0.7953 | 0.8169 | no (tied) |
| 47 | 0.4889 | 0.3156 | 0.8378 | 0.8034 | **0.8203 (best F1 — saved/pushed checkpoint)** | yes |
| 48 | 0.4886 | 0.3187 | 0.8461 | 0.7893 | 0.8167 | no |
| 49 | 0.4887 | 0.3122 | 0.8287 | 0.8044 | 0.8164 | no |
| 50 | 0.4887 | 0.3198 | 0.8394 | 0.7917 | 0.8148 | no |

Findings:

1. **The epoch-budget hypothesis from trial 20's own finding 2 pays off.** F1 climbs from 0.8080 (epoch
   30) to **0.8203 (epoch 47)**, +0.0123 for 20 more epochs — a real, if diminishing-returns, gain: about
   37% of the size of trial 20's own +0.0329 jump over trial 19b, for roughly two-thirds the extra epochs.
   Confirms epoch budget, not overfitting, really was the binding constraint at 30 epochs.
2. **This time it does plateau — genuinely, not just slowing down.** Unlike every prior "the run" in this
   sweep, F1 is now visibly noisy-flat rather than net-rising: three no-improve epochs after the epoch-47
   peak (48/49/50 all below it), and only two "new best" epochs (45, 47) in the last 15. The early-stop
   patience of 5 didn't quite arm before the 50-epoch cap was hit, but it was close — a genuine rerun to
   `--epochs 55-60` would likely find where it actually fires, rather than finding another big jump.
3. **The val_loss/F1 divergence reopens, more sharply than trial 21's.** True val_loss minimum across the
   full 50 epochs is **0.3092 at epoch 35**, a full **12 epochs** before the best-F1 checkpoint (epoch 47,
   val_loss 0.3156) — a bigger gap than trial 21's Δ3, on the same 35M model that had closed this gap to
   Δ0 at 30 epochs. Consistent with continuing past the point more data alone can hold the two metrics
   together — the divergence looks like it comes back with epoch count, not just model size or capacity.
4. **Confirms and outright reverses trial 21's model-size call, on hard numbers.** F1 0.8203 beats trial
   21's 0.8111 (`esm2_t30_150M_UR50D`, 30 epochs, 86279s) outright, on the cheaper 35M model, using
   52293s total (original 31005s + 21287s extension, ~14.5h) — still 1.65x cheaper than trial 21's single
   30-epoch run despite now covering 50. The user's cost/benefit call to stay on the 35M model rather than
   scale up wasn't just "not worth the trade" — extending epochs on the small model was strictly better
   than the trade it was weighed against.

## Standing conclusion (through trial 20's epoch extension)

The capacity axis (`unfreeze_last_n`) has a well-established shape at the 20k/5k data scale:
`unfreeze=2` clears every other config in the sweep by a wide margin, `unfreeze=4` (trial 10) overfits
by val_loss much sooner (epoch ~5) and tops out at F1 0.698, and `unfreeze=1` (trial 9) underfits
throughout. Trial 12 first located `unfreeze=2`'s F1 plateau (~0.72-0.74, peak 0.7421 at epoch 28)
alongside a real, substantial val_loss overfit after epoch 10 (0.3593 → 0.5936 by epoch 30). Trial 13
then showed that plateau wasn't the ceiling: `--label-smoothing 0.1` both closes the val_loss/F1
divergence almost entirely (val_loss rises only Δ0.0057 past its epoch-28 minimum, vs. trial 12's
Δ0.234) **and** raises absolute F1 at every comparable epoch, reaching a new sweep-best of **F1
0.7539 at epoch 28** — and for the first time, thanks to `--checkpoint-selection f1`, that epoch-28
checkpoint is what's actually saved and pushed to the Hub, not a val_loss-selected earlier epoch sitting
well below the run's true peak. Every dropout/weight-decay axis tested pre-label-smoothing remains a
no-op; capacity, epoch budget, and now label smoothing are the axes that have produced real effects.
Trial 14 tested effective batch size as a new axis (32 via `--grad-accum-steps`, vs. trial 13's real
16) at trial 13's unchanged learning rate, and came out worse at every matched epoch (F1 0.7384 vs.
0.7539) despite ~24% faster wall-clock — most plausibly an LR/step-count artifact (half as many
optimizer steps/epoch, unscaled LR) rather than evidence that a bigger effective batch is inherently
worse. Trial 15 confirmed this directly: rerunning trial 14's config with `--learning-rate 1e-4`
(2x, per the linear scaling rule) not only closed the gap but slightly exceeded trial 13's quality
(F1 0.7544 vs. 0.7539, reached 9 epochs earlier) while finishing in ~39% less wall-clock than trial 13
and ~19% less than trial 14 — the first trial where early stopping actually fired, stacking on top of
the smaller-`--batch-size` speed advantage. Trial 17 then tested one of trial 15's own recommended
next axes — `--label-smoothing` 0.1→0.2, everything else held at trial 15's exact config — and it's a
real, larger win: **F1 0.7642 at epoch 26, a new sweep-best**, beating trial 15 at every matched epoch
(+0.0098 at the peak, well outside the noise trial 15's own +0.0005-over-trial-13 margin sat in). The
cost tradeoff flipped, though: early stopping didn't fire this time (only 4 flat epochs after the
epoch-26 peak, one short of patience 5), so trial 17 ran the full 30 epochs and cost ~21% more than
trial 15 (20180.2s vs. 16677.7s) — still ~26% cheaper than trial 13, just not as cheap as trial 15 was.
Trial 16, an abandoned same-idea run with a different batch/grad-accum split (`--batch-size 2
--grad-accum-steps 16`, same effective batch 32), tracked trial 17 closely through its last completed
epoch before being interrupted — further evidence the literal batch/grad-accum split mainly affects
wall-clock, not the learning curve. Trial 17 became the reference config, strictly dominating trial 15
on quality. Trial 18 pushed `--label-smoothing` to 0.3 on top of trial 17's config (trial 17's own
recommended next step) but crashed at epoch 15 with a confirmed CUDA Xid 13 GPU exception — diagnosed
directly via `dmesg`/`journalctl`, unrelated to the config itself, and the direct motivation for adding
`--resume-from-checkpoint`/`run_mf_span_trial.sh` the same session (see trial 18's note). Trial 19
reran that exact config cleanly and confirmed the result: F1 0.7651 at epoch 26, only +0.0009 over
trial 17 — an order of magnitude smaller than trial 17's own +0.0098 over trial 15, alongside a
val_loss/F1 divergence that widened further (Δ0.0033 vs. trial 17's Δ0.0014) — the plateau signature
per trial 17's own pre-declared decision rule. Trial 19b then tested trial 19's own (b)
recommendation — `unfreeze_last_n` 2→3 on top of `--label-smoothing 0.3` — and it's a real, larger
win: **F1 0.7751 at epoch 24, a new sweep-best**, +0.0100 over trial 19 (bigger than trial 17's own
smoothing-driven jump), and the first trial since 15 to actually early-stop. It was launched under
trial 19's unchanged `--run-name`/`--hub-branch`, though, which overwrote trial 19's local log and Hub
checkpoint in place (trial 19's original curve survives only in git history at commit `d383a6c` — see
trial 19b's note). Trial 20 then tested trial 19's own (a) recommendation — uncapping
`max_train_pairs`/`max_val_pairs` (32025/9363, up from the fixed 20000/5000 every prior trial used) —
on top of trial 19b's config, and it dwarfs every other single-axis change in the sweep: **F1 0.8080
at epoch 30, a new sweep-best**, +0.0329 over trial 19b (more than 3x trial 19b's own gain over trial
19), still rising at the epoch budget's edge (early stop never armed) and, for the first time since
trial 15, with val_loss's true minimum landing on the exact same epoch as the best-F1 checkpoint —
more data closed the val_loss/F1 divergence outright rather than needing a regularization workaround.
**Trial 20 is the reference config for the ongoing (35M-model) sweep**, and its own curve suggests
the 30-epoch budget, not the model, is now the binding constraint (see trial 20's finding 2). Trial
21 then tested the model-size axis directly — trial 20's exact config on `esm2_t30_150M_UR50D`
(~4.3x params) instead of `esm2_t12_35M_UR50D` — and, while it edged out trial 20's original 30-epoch
result by number (F1 0.8111 vs. 0.8080), the margin was tiny (+0.0031) against a 2.78x wall-clock cost
(86279.0s vs. 31004.7s). **Per the user's own call, that trade wasn't worth it**: the sweep stayed on
the 35M model rather than moving to the bigger one — and trial 20 was then resumed via
`--resume-from-checkpoint`/`--reset-early-stop-on-resume` and extended from 30 to 50 epochs, per its
own reopened "extend epochs" recommendation. That extension (epochs 31-50, see its note above) is a
genuine further win, **F1 0.8203 at epoch 47** — +0.0123 over the original 30-epoch checkpoint — and,
notably, **now beats trial 21's bigger-model result outright (0.8111) while still costing less overall**
(52293s vs. 86279s). Unlike every prior run in this sweep, this one shows real signs of actually
plateauing (noisy-flat over its last several epochs, close to arming early stopping) rather than just
running out of epoch budget — and the val_loss/F1 divergence, closed to Δ0 at epoch 30, reopened to a
striking Δ12 epochs by epoch 47 (val_loss min at 35, best F1 at 47), the largest such gap seen in the
sweep. The user's cost/benefit call on trial 21 is now doubly validated: not only was the bigger model
not worth its cost, extending the small model past it was strictly better on both quality and cost. A
fresh trial 22 is in progress alongside this write-up, retesting `unfreeze_last_n=4` (trial 10's old
overfitting axis) now that trial 20 has shown uncapped data closes that overfitting outright — not yet
logged with results, since it's still running.

Current best *and* practical-reference checkpoint by F1: **trial 20 (extended), F1 0.8203, epoch 47**
(`kelmoir/cafa6-mf-span@trial-20`, run_name `mf-span-reg-trial-20`, `esm2_t12_35M_UR50D`) —
effective batch 32 (`--batch-size 2 --grad-accum-steps 16`), `--learning-rate 1e-4`,
`--unfreeze-last-n 3`, `--label-smoothing 0.3`, `--checkpoint-selection f1`,
`--early-stop-patience 5`, uncapped `--max-train-pairs`/`--max-val-pairs`, run to `--epochs 50` via
one resume from the original 30-epoch checkpoint. This is the first trial in the sweep where the
technical sweep-best and the practical reference checkpoint are the same thing — trial 21's bigger
model (F1 0.8111, `@trial-21`, `esm2_t30_150M_UR50D`) is now behind on both quality and cost, and
remains retrievable only as a record of the model-size axis having been tried and declined. Trial
19b's checkpoint (F1 0.7751, `@trial-19` — trial 19's original `unfreeze_last_n=2`/F1 0.7651
checkpoint it overwrote is no longer retrievable from the Hub, only from git history), trial 17's (F1
0.7642, `@trial-17`), and trial 15's (F1 0.7544, `@trial-15`) remain retrievable as clearly
lower-quality 35M-model alternatives further back on the epoch/data axes. Trial 13's checkpoint (F1
0.7539, `@main`) and trial 14's (F1 0.7384, `@trial-14`) are both further behind that frontier still.

Recommended next trial: trial 22 (`unfreeze_last_n=4`, in progress as of this write-up) is already the
live next step — once it finishes, log its full curve and findings the same way as the trials above.
If it repeats trial 10's overfitting-by-loss pattern despite the uncapped data, that's a real result
(capacity has a ceiling data volume alone doesn't fix); if it instead tracks or beats trial 20's
extended curve, capacity reopens as a lever worth stacking with a further epoch extension. Beyond that:
trial 20's extension note (finding 2) suggests a short further extension (`--epochs 55-60`) would likely
find where early stopping actually fires, now that the curve shows real plateau signs — lower priority
than trial 22's result, but cheap to check once capacity is settled. The model-size axis stays closed
per trial 21's result and the user's cost/benefit call (now reinforced by the extension beating it
outright) — no further `--model-name` scouting planned. Given the local-GPU crash history (trials
16/18) and the RNG-resume bug this session's extension just exercised and fixed (commit `ea9a9c2`),
keep launching trials via `experiments/scripts/run_mf_span_trial.sh` rather than the script directly —
at 8-14+ hours per trial now, that protection matters more than ever. Untested regularization axes
remain lowest priority given how little effect they've had so far: `--output-dropout` above 0.2,
`--lr-schedule linear_decay`, `--pos-weight` override. **Give each future trial a distinct `--run-name`
AND `--hub-branch`, never reused from a prior trial** — one collision has already happened (trial 19b
over trial 19) from skipping this; trial 20's own resume/extension is the one deliberate, documented
exception to that rule (same run identity, continued on purpose).
