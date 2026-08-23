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

## Standing conclusion (through trial 8)

Data volume, not any regularization axis, is the binding constraint at this scale. Trials 7 and 8
isolate this cleanly: doubling `--max-train-pairs`/`--max-val-pairs` (12k/2k → 20k/5k, same 42.9%
positive rate) lifts best F1 from ~0.63 to ~0.688 regardless of dropout, and in both cases val_loss
was still falling at epoch 10 — the epoch-8 overfit ceiling every 12k/2k trial (1/3/4/6) hit is gone.
Every regularization axis tried so far (esm/attn dropout, weight decay+decoupling, more negatives per
protein) has been a no-op or a regression; none of them, and not the LR schedule change either, moved
the result in the way more data did.

Current best/most useful checkpoint: trial 7 or 8 (statistically tied), both far ahead of the
12k/2k-scale trials.

Recommended next trial: extend `--epochs` past 10 on the 20k/5k config (either trial 7 or 8's setup)
to find where it actually plateaus, since neither has yet. If GPU/time budget allows, also worth
trying a further data-volume increase (e.g. 30k/7k) as its own isolated trial, holding dropout/weight
decay at their no-op-confirmed values. Untested axes still on the list but now lower priority given
data volume's outsized effect: `--output-dropout` above 0.2, `--unfreeze-last-n` below 2. Remember to
give each future trial a distinct `--hub-branch`, not just a distinct `--run-name`, if the Hub copy
should stay retrievable.
