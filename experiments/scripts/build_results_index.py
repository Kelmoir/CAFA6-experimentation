"""Rebuild experiments/results_index.csv from the raw trial logs.

Parses every experiments/logs/trial-*.log (including *_epochN.log continuation
fragments from the old per-epoch-chained runs), groups fragments belonging to the
same trial id, and picks each trial's best (lowest val_loss) epoch across all its
fragments. Safe to re-run any time — it's a full rebuild, not an incremental append,
so the CSV always reflects exactly what's in experiments/logs/.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path("experiments/logs")
OUT_PATH = Path("experiments/results_index.csv")

# Manually curated from sweep_log.md's per-trial verdicts (not reliably derivable from the
# logs alone). Update this alongside sweep_log.md whenever a trial gets a verdict.
VERDICTS = {
    "trial-001": "VOID (label bug)",
    "trial-002": "INCOMPLETE (connection lost)",
    "trial-003": "DEAD",
    "trial-004": "DEAD",
    "trial-005": "WORSE (collapse toward all-positive)",
    "trial-006": "DEAD",
    "trial-007": "DEAD",
    "trial-008": "OVERFITTING",
    "trial-009": "OVERFITTING",
    "trial-010": "OVERFITTING",
    "trial-011": "OVERFITTING",
    "trial-012": "OVERFITTING",
    "trial-013": "DEAD (re-approached via excess wd)",
    "trial-014": "MIXED (epochs 1-4 pre-fix, 5-6 post per-epoch-variation fix)",
    "trial-015": "PLATEAU (stable, beam search round-0 seed)",
    "trial-016": "PLATEAU (matches trial-015, unfreeze bump had no effect)",
    "trial-018": "PLATEAU (round-1 winner: cosine lr_schedule, best val_loss of round)",
    "trial-019": "PLATEAU (round-1: warmup_steps=300, no improvement over seed)",
    "trial-020": "PLATEAU (round-1 reg: label_smoothing=0.05, no improvement; train_acc/recall=0 is a metric-computation artifact, not the real training signal - fixed post-hoc, see training_run.py)",
    "trial-021": "PLATEAU (round-1 reg: decouple_bias_wd, roughly ties seed)",
    "trial-022": "PLATEAU (round-1 runner-up: grad_accum_steps=2, ~seed-level)",
    "trial-023": "PLATEAU (round-1: unfreeze_last_n=3, noisier + worse than seed, slower per-epoch)",
    "trial-024": "PLATEAU (round-2: learning_rate=3e-4 off trial-018, no improvement over parent)",
    "trial-025": "PLATEAU (round-2 winner: weight_decay=0.2 off trial-018, ties trial-018 within noise)",
    "trial-026": "PLATEAU (round-2 runner-up: unfreeze_last_n=2 off trial-018, ties trial-018 within noise, slower per-epoch)",
    "trial-027": "PLATEAU (round-2: cosine+grad_accum=2 stacked off trial-022, no improvement over trial-018)",
    "trial-028": "PLATEAU (round-2 reg: output_dropout=0.1 off trial-022, worst of round 2)",
    "trial-029": "PLATEAU (round-2: grad_accum_steps=4 off trial-022, no improvement over trial-022)",
    "trial-030": "MILD OVERFITTING (35M-scale seed, 6 epochs: best val_loss 0.5946 at epoch 4, then val_loss rises 0.5946->0.5959 while train_loss keeps falling - same shape as trial-031, much weaker; per user's val_loss>0.595 noise-floor call, this is the only 8M/35M-comparable result besides trial-031 to clear it)",
    "trial-031": "OVERFITTING (35M-scale, zero regularization: best val_loss 0.5918 at epoch 4 - then val_loss rises while train_loss keeps falling through epoch 6; first clear overfitting signal in the whole sweep)",
    "trial-032": "MILD OVERFITTING (batch_size=32 off trial-031: best val_loss 0.5916 at epoch 4, ties trial-031; slower per-epoch than batch=8 despite fewer steps, likely padding-waste from random per-batch length padding)",
    "trial-033": "MILD OVERFITTING (batch_size=128 off trial-031: best val_loss 0.5902 at epoch 5 - new best of the entire sweep, despite ~94 opt steps/epoch (16x fewer than trial-031); user reports this trial nearly OOM'd the T4)",
    "trial-034": "STILL CONVERGING (batch_size=128 + grad_accum_steps=4 off trial-031: best val_loss 0.5960 at epoch 6, its last epoch and still falling - above the 0.595 noise floor, so not yet a real result; too few optimizer steps (~23/epoch) to have converged in 6 epochs)",
    "trial-035": "MILD OVERFITTING (light regularization off trial-030, dropout=0.1/wd=0.1: best val_loss 0.5914 at epoch 5 - 2nd best of the sweep, beats both trial-030's heavy reg and roughly ties trial-031's zero reg)",
    "trial-036": "MILD OVERFITTING (wd=0.0 with dropout=0.2 off trial-030: best val_loss 0.5937 at epoch 4 - better than trial-030 but worse than trial-035's lighter-both combo)",
    "trial-037": "MILD OVERFITTING (unfreeze_last_n=2 off trial-030: best val_loss 0.5936 at epoch 4, ~15% slower per epoch than unfreeze=1)",
    "trial-038": "NOISY (batch_size=128 off trial-033, epochs=10 rerun: best val_loss 0.5891 at epoch 7 - new best of the entire sweep - but val_loss oscillates rather than cleanly rising after its epoch-5 local min (0.5902->0.5922->0.5891->0.5923...), consistent with few (~94) optimizer steps/epoch adding noise, not a clean overfitting curve)",
    "trial-039": "NOISY (batch=128, dropout=0.1/wd=0.1 off trial-033, epochs=10 rerun: best val_loss 0.5900 at epoch 7, same oscillating pattern as trial-038)",
    "trial-040": "NOISY (batch_size=64 off trial-033: best val_loss 0.5905 at epoch 7, same oscillating pattern, milder)",
    "trial-041": "OVERFITTING (rerun, complete: dropout=0.1/wd=0.1, batch=8 - best val_loss 0.5914 at epoch 5, then rises clearly through epoch 10 (0.6088) while train_loss keeps falling - the harsh late-epoch overfit the user expected, now visible at the full 10 epochs)",
    "trial-042": "OVERFITTING (dropout=0.05/wd=0.05 off trial-035, batch=8: best val_loss 0.5929 at epoch 4 - worse than trial-035/041's 0.1/0.1, so 0.1/0.1 is closer to the regularization sweet spot than going lighter still; also overfits clearly by epoch 10 (0.6046))",
    "trial-043": "OVERFITTING (unfreeze_last_n=2 + dropout=0.1/wd=0.1 off trial-035: best val_loss 0.5923 at epoch 3, worse than trial-041's unfreeze=1 (0.5914) and the most severe late-epoch overfit of the round - val_loss jumps to 0.6166/0.6249 at epochs 9-10)",
    "trial-044": "NOISE-CHECK RESULT (trial-038 rerun at seed=123, batch=128, zero reg: best val_loss 0.5921 at epoch 6 - notably worse than trial-038's seed=42 result (0.5891), a 0.003 gap from seed alone. Confirms the batch=128 lineage's fine ranking (038/039/040/045/046, spread over only ~0.0014) is within seed-to-seed noise, not a reliable ordering. HF token fetch timed out this run, so no Hub checkpoint was pushed - metrics are still valid, training completed all 10 epochs)",
    "trial-045": "NOISY (dropout=0.05/wd=0.05 off trial-038, batch=128: best val_loss 0.5895 at epoch 7 - beats trial-039's 0.1/0.1 (0.59) and nearly ties trial-038's zero reg (0.5891), suggesting batch=128 wants lighter regularization than batch=8 - but the gap is within the seed-noise band trial-044 just measured, so treat directionally, not conclusively)",
    "trial-046": "NOISY (batch_size=96 off trial-038: best val_loss 0.5893 at epoch 5 - nearly ties trial-038's batch=128 result (0.5891) and beats trial-040's batch=64 (0.5905), suggesting the batch-size curve may be leveling off around 96-128 - again within the seed-noise band, not conclusive on its own)",
    "trial-047": "NOISE-CHECK RESULT (trial-039 rerun at seed=123, batch=128, dropout=0.1/wd=0.1: best val_loss 0.5929 at epoch 6 - worse than trial-039's seed=42 result (0.5900), a 0.0029 gap from seed alone. 2nd independent confirmation of the noise-floor finding from trial-044/038)",
    "trial-048": "NOISY (weight_decay=0.1 alone, dropout=0.0, off trial-039, batch=128: best val_loss 0.5892 at epoch 7 - ties trial-038's zero-reg result almost exactly, confirming weight_decay alone has ~no effect at this batch size, same as trial-036 found at batch=8 - dropout is the regularizer that matters, not weight_decay)",
    "trial-049": "NOISY (grad_accum_steps=2 off trial-039, batch=128 effective 256: best val_loss 0.5913 at epoch 9 - later and less oscillating than other batch=128 trials, consistent with even fewer optimizer steps/epoch (~47) slowing convergence; still within the round's noise band)",
    "trial-050": "HEALTHY, reported metrics were misleading due to a real bug (now fixed) (first trial on the new 383-coarse-GO-term task: val_loss improves cleanly 0.0803->0.0772 over 10 epochs, best at epoch 7. Training-loop val_recall (2.7-3.6% the whole run) looked like a collapse, but a proper diagnostic (experiments/scripts/diagnostic_eval_stage2.py) on the same checkpoint found median per-label AUROC=0.84, pooled precision/recall/F1@0.5=0.66/0.46/0.54 - a genuinely well-performing model. Root cause: compute_metrics() averaged per-batch recall ratios (batch_size=8, 383 sparse labels at 3.58% positive means most (label,batch) pairs have zero true positives, so tp/(tp+fn+eps)=0 dominates the batch average regardless of model quality) instead of pooling tp/fn across the whole epoch first. Fixed in training_run.py (compute_metrics()/finalize_metrics() now accumulate raw counts and compute the ratio once) - confirmed via a synthetic test: a model with true recall=1.0 was reported as 25.7% recall under the old code, 100% under the fix. Every prior trial's accuracy was unaffected by this bug (only recall's ratio-averaging was wrong); recall numbers on prior 3-label trials were only mildly affected since 3 labels rarely hit zero-positives-per-batch the way 383 labels do)",
    "trial-051": "NEGATIVE RESULT for pos_weight as tried (isolated A/B vs trial-050, only --pos-weight added: sqrt(n_neg/n_pos) per label). val_loss/recall from the training log are NOT directly comparable to trial-050's - pos_weight changes the loss function's numeric scale (up to 69x on rare-label positives) and trial-050's own recall was computed under the old buggy metric, so a proper diagnostic (experiments/scripts/diagnostic_eval_stage2.py) was run on both checkpoints for an apples-to-apples read. Result: median per-label AUROC ticked up slightly (0.8456 vs 0.8366) and the optimal threshold recalibrated back toward 0.5 as intended (0.48 vs trial-050's 0.32) - pos_weight's mechanism works - but best-achievable pooled F1 got WORSE (0.5007 vs 0.5727): at threshold 0.5, recall rose (45.8%->52.9%) but precision fell harder (65.7%->47.2%), a net-negative trade. sqrt(n_neg/n_pos) as formulated overcorrects for this dataset; a gentler compression (e.g. ratio**0.3) or no pos_weight (just post-hoc threshold tuning, which is free and doesn't disturb the trained representation) may be the better direction",
    "trial-052": "PARTIAL RECOVERY, still worse than plain BCE on the metric that matters (round_1 part_1, reg: pos_weight_exponent=0.3 off trial-050, gentler than trial-051's exponent=0.5/sqrt). diagnostic_eval_stage2.py: median AUROC=0.8473 (best of trial-050/051/052), pooled F1@0.5=0.5445 (beats trial-050's 0.5398 and trial-051's 0.4989), best F1 over threshold sweep=0.5604@0.42 - better than trial-051's 0.5007 but still short of trial-050's plain-BCE best F1 of 0.5727. Confirms the trend: lighter pos_weight compression trades less F1 for less AUROC/threshold-calibration gain, monotonically approaching (not beating) plain BCE as exponent->0. Since trial-050 already recovers 0.5727 for free via post-hoc threshold tuning (no retraining), pos_weight in either form tested so far isn't earning its added hyperparameter/training cost for this task",
    "trial-053": "IMPROVEMENT (round_1 part_1, reg: weight_decay=0.2 off trial-050, wd=0.5->0.2, dropout unchanged at 0.2: best val_loss 0.0754 at epoch 7-8, clearly below trial-050's 0.0772 - trial-050's inherited wd=0.5 was too strong for this task's smaller 12k-example training set; new round_1 winner so far)",
    "trial-054": "TIES trial-053, WORSE WALL-CLOCK (round_1 part_1: batch_size=32 off trial-050: val_loss still falling at epoch 10 (0.0754, hadn't plateaued), essentially tying trial-053's converged 0.0754 - but each epoch took ~742s vs trial-053's ~574s (~29% slower) despite 4x fewer optimizer steps/epoch, the opposite of the old esm2_t12_35M_scale_up phase's finding that larger batch sped up wall-clock; batch=32 doesn't show trial_030-era phase's benefit transferring to this task and costs more wall-clock for no loss improvement within budget",
    "trial-055": "MILD IMPROVEMENT, WRONG PARENT - RAN STALE CONFIG (round_1 part_2: unfreeze_last_n=2. CONFIG line shows weight_decay=0.5, not the 0.2 this trial was re-parented to before the notebook was sent - the run used the pre-pivot design branching off trial-050, not trial-053. Best val_loss 0.0768 at epoch 7, beats trial-050 (0.0772) but well short of trial-053 alone (0.0754). Not directly informative about unfreeze=2 stacked with wd=0.2, which was the intended test",
    "trial-056": "BEST OF PART_2, WRONG PARENT - RAN STALE CONFIG (round_1 part_2: lr_schedule=cosine. Same mismatch as trial-055 - ran at weight_decay=0.5, not the intended 0.2. Best val_loss 0.0762 at epoch 9-10, the best of this batch and beats trial-050 (0.0772), but still short of trial-053 alone (0.0754). Cosine is the most promising secondary lever seen so far (independently, at wd=0.5) - worth testing stacked with wd=0.2 in round 2",
    "trial-057": "NO IMPROVEMENT, WRONG PARENT - RAN STALE CONFIG (round_1 part_2: learning_rate=3e-4. Same mismatch as trial-055/056 - ran at weight_decay=0.5, not the intended 0.2. Best val_loss 0.0775 at epoch 9, essentially ties trial-050 (0.0772) within noise - no evidence higher LR helps within this 10-epoch budget",
    "trial-058": "PROVISIONAL IMPROVEMENT, MARGIN NARROWER THAN ROUND_1'S (round_2 part_1, reg: weight_decay=0.1 off trial-053. Best val_loss 0.0749 at epoch 7-9, edges out trial-053's 0.0754 by 0.0005 - continues the 'lighter wd helps' trend from round_1, but this margin is smaller than round_1's already-marginal deltas and this phase's noise floor hasn't been measured yet (see noise_floor_check, trial-064/065/066 running in parallel). Treat as provisional until that lands",
    "trial-059": "PROVISIONAL IMPROVEMENT, MARGIN NARROWER THAN ROUND_1'S (round_2 part_1, reg: output_dropout=0.1 off trial-053. Best val_loss 0.0751 at epoch 8, edges out trial-053's 0.0754 by 0.0003 - smallest gain of the batch, confirms the 'inherited dropout may be too strong' hypothesis directionally but same noise-floor caveat as trial-058 applies, even more so given the smaller margin",
    "trial-060": "BEST OF ROUND_2 PART_1, MARGIN NARROWER THAN ROUND_1'S (round_2 part_1: lr_schedule=cosine properly stacked with weight_decay=0.2 this time (CONFIG line confirmed correct, no repeat of round_1 part_2's mismatch). Best val_loss 0.0748 at epoch 9-10, best of the batch, confirms cosine's round_1 gain (found independently at wd=0.5) does compound with wd=0.2 rather than just overlapping with it - but again, same noise-floor caveat as trial-058/059",
    "trial-061": "WITHIN NOISE (round_2 part_2, properly stacked this time: unfreeze_last_n=2 + weight_decay=0.2. Best val_loss 0.0748 at epoch 8, delta=-0.0006 vs trial-053 - ties trial-060 for best of round_2 but still at/below the calibrated ~0.0007-0.0011 noise floor, not a confirmed real improvement",
    "trial-062": "NO IMPROVEMENT (round_2 part_2, properly stacked: learning_rate=3e-4 + weight_decay=0.2. Best val_loss 0.0755 at epoch 9, delta=+0.0001 vs trial-053 - essentially ties the baseline, closes out the intended stacked test (round_1's independent test at wd=0.5 also found no improvement)",
    "trial-063": "WITHIN NOISE (round_2 part_2: warmup_steps=300, fresh axis for this task. Best val_loss 0.0753 at epoch 9, delta=-0.0001 vs trial-053 - within noise, no real effect",
    "trial-064": "NOISE-FLOOR MEASUREMENT (trial-052 rerun at seed=123, otherwise identical: pos_weight_exponent=0.3. Best val_loss 0.1449 vs trial-052's 0.1438 - delta=0.0011, this phase's largest measured seed-noise sample. Part of the user-initiated noise_floor_check batch (trial-064/065/066), see beam_search_state.json",
    "trial-065": "NOISE-FLOOR MEASUREMENT - CONFIRMS TRIAL-053 IS REAL (trial-053 rerun at seed=123, otherwise identical: weight_decay=0.2. Best val_loss 0.0755 vs trial-053's 0.0754 - delta=0.0001, essentially reproduces exactly. This is the noise sample for the round_1 winner's exact config, and it's tiny - confirms trial-053's 0.0018 margin over trial-050 is a real effect, well above this specific config's own seed noise",
    "trial-066": "NOISE-FLOOR MEASUREMENT (trial-054 rerun at seed=123, otherwise identical: batch_size=32. Best val_loss 0.0762 vs trial-054's 0.0754 - delta=0.0008. Combined with trial-064/065, gives a 3-sample noise-floor estimate of ~0.0007-0.0011 for this phase - notably smaller than the old esm2_t12_35M_scale_up phase's ~0.003, but still large enough that round_2 part_1's margins (0.0003-0.0006 over trial-053) don't clearly clear it",
    "trial-067": "TIMING CALIBRATION, not a quality trial (esm2_t36_3B_lora_scaleup phase kickoff, LoRA on a 15GB card: 200 train/100 val, 1 epoch. 5.6843s/example - a 138x slowdown vs esm2_t12_35M scale, far beyond the ~81x parameter-count ratio alone - LoRA solves the memory problem but forward/backward FLOPs still scale with the full frozen 3B backbone. Motivated --lora-layers-to-transform (verified real+free via peft: untouched early layers get requires_grad=False, autograd skips their backward entirely)",
    "trial-068": "FAILED - OutOfMemoryError (combined --no-gradient-checkpointing + --lora-layers-to-transform=18 off trial-067: OOM'd after 21.8s, never completed an epoch, not in this CSV's rows below since the parser requires at least one completed epoch. Proved gradient checkpointing is not optional at 36 layers/batch=8 - trial-067's 9.5GB peak WAS the checkpointed number, not spare headroom)",
    "trial-069": "MODEST IMPROVEMENT, still far too slow (isolated --lora-layers-to-transform=18 off trial-067, checkpointing restored: 5.012s/example, 11.8% faster than trial-067 but still ~122x slower than esm2_t12_35M scale. User pivoted away from 3B/LoRA to esm2_t33_650M full fine-tune after this result - LoRA infra kept in training_run.py for later)",
    "trial-070": "TIMING CALIBRATION for the esm2_t33_650M full-fine-tune pivot (200 train/100 val, 1 epoch, all layers trainable: 1.737s/example - 3.27x faster than trial-067's LoRA/3B attempt, 42.3x slower than esm2_t12_35M scale. Strongly validated the pivot away from 3B/LoRA on this card",
    "trial-071": "HEALTHY, first real quality signal at esm2_t33_650M full-fine-tune scale (3 epochs, 450 train resampled/100 val, learning_rate=2e-5 (10x lower than the carried-forward default, per full-fine-tune convention). Train/val loss decrease together, gap stays small. Diagnostic (2000-sample, run post-hoc on CPU): median AUROC=0.6005, pooled AUROC=0.9354, best F1=0.5319@0.24 - real signal despite only 1350 total example-exposures, though median AUROC sits well below the mature 35M-phase trials",
    "trial-068b": "DECISIVE THROUGHPUT WIN, user-run out-of-band on a rented A100 (5x $/hr vs the 15GB card trial-067/068/069 used): batch_size=16, --lora-layers-to-transform=10, --no-gradient-checkpointing. 0.055s/example - 103.4x faster than trial-067, 31.6x faster than trial-070's 650M/full-FT baseline, only 1.34x slower than esm2_t12_35M's own T4 baseline. ~6.3x more cost-effective (examples/sec/$) than the 650M/T4 path despite 5x higher hourly cost. Reopened esm2_t36_3B_lora_scaleup as the sweep's leading direction; only a 1-epoch/200-example timing calibration, said nothing yet about whether layers=10 has enough capacity for real quality",
    "trial-072": "HEALTHY, DECISIVE WIN - new best result across the entire sweep (first real multi-epoch quality trial at trial-068b's proven A100 config: 8 epochs, 6000 train resampled/1000 val. Median AUROC 0.6005->0.8811 vs trial-071 (3.79x increase in skill-margin-above-chance, despite trial-072 also seeing ~35.6x more total data - the corrected, data-volume-aware bar this margin needed to clear), best F1=0.5974@0.30. Also exceeds trial-053, the esm2_t12_35M_coarse_go_terms phase's own best-ever result (0.8545/0.5810 from 120,000 total exposures + two rounds of regularization tuning) with under half the data and zero regularization tuning - though at ~3x the dollar-normalized cost of trial-053, a genuine quality win, not also a cost win. Resolved the capacity concern around --lora-layers-to-transform=10",
    "trial-073": "NO IMPROVEMENT, more depth doesn't help and costs more (layer-depth test off trial-072: --lora-layers-to-transform=18 (double trial-072's 10), same exact 8-epoch/6000-1000 data budget, batch halved 16->8 to fit. Every quality metric ties trial-072 within <1% (median AUROC +0.61%, best F1 -0.05%) while being 12.1% slower per-example and 14.8% more wall-clock for the same budget. 10 layers already captures essentially all the achievable quality at this data budget - confound: batch drop is entangled with the depth increase, can't cleanly attribute to depth alone",
    "trial-074": "HEALTHY, MILD OVERFITTING past epoch 7, first Hub promotion (production run at trial-072's exact config, scaled to 10 epochs/12,000 train/2,000 val, matching the esm2_t12_35M_coarse_go_terms phase's own seed_config data convention, --push-to-hub set. val_loss bottoms at epoch 7 (0.0719) then rises through epoch 10 (0.0726) while train_loss keeps falling monotonically - the best-checkpoint feature correctly promoted the epoch-7 checkpoint to kelmoir/cafa6-esm@main instead of the overfit epoch-10 one, the first successful --push-to-hub promotion in this project's history. Bug found from this trial's own log: --run-diagnostic evaluated the epoch-10 model, not the promoted epoch-7 one (median AUROC 0.8884/best F1 0.5963 recorded are for epoch 10) - fixed in training_run.py so both diagnostic and promotion now share one resolved best_model; not re-run for trial-074 itself given the epoch7-vs-epoch10 val_loss gap is only 0.0007. vs trial-072: median AUROC +0.0073, best F1 -0.0011 - scaling to more epochs/data bought very little, consistent with the overfitting finding. Superseded on kelmoir/cafa6-esm@main by trial-075, see below",
    "trial-071b": "HEALTHY, esm2_t33_650M_full_finetune moves to the A100 (3 epochs, 450 train resampled/100 val, batch_size=16 + --no-gradient-checkpointing now affordable off T4, learning_rate=2e-5 held from trial-071. unfreeze_last_n=27 (not the full 33) - an undocumented but confirmed-intentional change, needed to fit batch=16/no-checkpointing into A100 memory. val_loss decreases monotonically every epoch (0.0951->0.0945->0.0936), same HEALTHY signature as trial-071 - confirms the config before spending a full 8-epoch/6000-sample budget on it (trial-075)",
    "trial-075": "HEALTHY, F1 parity/edge vs trial-072 at ~23% less cost, real macro-AUROC gap (esm2_t33_650M_full_finetune at trial-072's exact 8-epoch/6000-train/1000-val budget and A100 hardware class - the first genuinely apples-to-apples comparison between the 650M-full-FT and 3B-LoRA phases. Supersedes an incomplete first attempt at this run_name whose push left stale/mixed LoRA files on the branch. best_epoch=8 (last epoch also best - no overfitting within budget), best_val_loss=0.0733. Diagnostic: median AUROC=0.8521 (vs trial-072's 0.8811, a real -3.3% gap), best F1=0.6019 (vs trial-072's 0.5974, +0.75%), pooled AUROC=0.9601. Wall clock 2662.5s vs trial-072's 3453.4s (~23% faster/cheaper on identical A100 hardware). Given F1 parity/edge (the metric closer in shape to CAFA's own Fmax scoring) plus the cost saving, promoted to kelmoir/cafa6-esm@main per user direction, superseding trial-074's 3B LoRA checkpoint - also fixed a real Hub corruption found in the process (main had accumulated trial-074's stale adapter files underneath trial-075's push; push_checkpoint() now passes delete_patterns to prevent recurrence)",
    "trial-076": "COMPLETE, PROMOTED to kelmoir/cafa6-esm@main superseding trial-075, real if modest win over trial-075 on every diagnostic metric, val_loss plateaus (continues training FROM trial-075's own checkpoint via --resume-from-revision trial-075 --start-epoch 8, 4 more epochs (9-12) on freshly-resampled 6000-train/1000-val, every other hyperparameter identical to trial-075, per user direction that trial-075's config is the valid point to build on rather than a fresh restart at a bigger data/epoch budget. First attempt OOM'd at batch_size=16 despite trial-075 running clean there - root-caused to the per-epoch random shuffle occasionally clustering several near-max-length sequences into one batch by chance (resample luck, not a real capacity mismatch); fixed via transformers' LengthGroupedSampler (group_by_length) in training_run.py, though the actual successful run used the user's own batch_size=8/--grad-accum-steps=2 workaround found independently first. val_loss: 0.0733->0.0730->0.0726->0.0727->0.0726 - improves once then plateaus (mild overfitting signature, milder than trial-074's clear reversal) while train_loss keeps falling (0.0695->0.0657). Diagnostic: median AUROC=0.8604 (+0.97% vs trial-075, still -2.35% vs trial-072, narrower than trial-075's -3.3% gap), best F1=0.6046 (+0.45% vs trial-075, +1.20% vs trial-072, wider than trial-075's +0.75% edge), pooled AUROC=0.9618. Every diagnostic metric beats trial-075, none regress. Cost honesty check: marginal cost ~1.93 cost-units (1391.0s), but cumulative cost from trial-075+trial-076 is 5.63 cost-units - +17.4% MORE expensive than trial-072's 4.80. Promoted anyway per user direction ('I consider is really promising, so we stick to it') - consistent diagnostic-metric gains judged worth the extra cost",
    "trial-077": "COMPLETE, HEALTHY/borderline plateau, esm2_t33_650M_depth1_go_terms baseline (no aspect input) - new track testing a much smaller 39-term depth-1-only GO-term label space (kelmoir/cafa6-depth1), exactly trial-075's hyperparameters otherwise. First attempt at batch_size=16 OOM'd consistently even with the LengthGroupedSampler fix - a different memory source, not yet root-caused - switched to batch_size=8/grad-accum-steps=2 (now the standing default for this model class). val_loss decreases monotonically through epoch 7 (0.1746->0.1598) then a tiny uptick at epoch 8 (+0.6%) while train_loss keeps falling (0.1834->0.1433) - borderline mild overfitting, best-checkpoint-reload correctly used epoch 7. Diagnostic: all 39/39 labels evaluable, median AUROC=0.8099, mean=0.8140 (LOWER than the 383-term task's 0.85/0.84-range, likely because depth-1 terms are extremely prevalent/broad by construction - e.g. 71.6% base rate for 'cellular anatomical structure' - making AUROC separation intrinsically harder), but best F1=0.6448 is notably HIGHER than trial-075/076's 0.60/0.60 (+7%), and threshold-0.5 recall (~18%) is far above the 383-term task's ~5-6% (much higher average positive rate). Wall clock 1938.2s, ~2.69 cost-units (cheaper than trial-075's 3.70, smaller 39-label classifier head). Not pushed to main - awaiting trial-078 (same config + --use-aspect-input) for the actual aspect-conditioning comparison",
    "trial-078": "COMPLETE, DECISIVE WIN over trial-077 - every single metric improved, the cleanest result in this entire sweep (unlike every other trial-vs-trial comparison, which all had a real tradeoff). Identical hyperparameters to trial-077 plus --use-aspect-input: classifier head concatenates the ground-truth aspect_labels 3-way multi-hot to the pooled CLS-token representation, ESM backbone forward pass byte-identical to trial-077 so the delta is cleanly attributable to the aspect signal alone. val_loss: 0.1598->0.1499 (-6.16%). mean AUROC: 0.8140->0.8294 (+1.89%). median AUROC: 0.8099->0.8171 (+0.90%). pooled AUROC: 0.9454->0.9509 (+0.58%). pooled F1@0.5: 0.6118->0.6477 (+5.85%). best F1: 0.6448->0.6648 (+3.10%). precision@0.5: +5.36%, recall@0.5: +6.26%. min AUROC 0.6188->0.6343, labels with AUROC>0.7: 34/39->37/39. Cost: +2.38% (negligible, 2.756 vs 2.692 cost-units). Same training-dynamics shape as trial-077 (monotonic through epoch 7, noise-level epoch-8 uptick, best_epoch=7 both), so the comparison isn't confounded by different convergence. Caveat: uses GROUND-TRUTH aspects (oracle upper bound), not predicted - real-world cascade gain depends on first-stage aspect-predictor accuracy, not measured here. Meaningfully de-risks the sequential-cascade prospect (experiments/sequential_cascade_plan.md, paused) - open decision: resume the cascade plan, test with predicted (non-oracle) aspects first, or treat this ablation as complete",
    "trial-079": "COMPLETE, DEAD - disguised by a misleading recall=1.0. New esm2_t33_650M_aspect_prediction phase (kelmoir/cafa6-aspect, 3-way F/C/P), health check adapted from trial-071b (4 epochs instead of 3, --max-train-samples 338 to hold the total budget constant at 1352 vs trial-071b's 1350, batch_size=8/grad-accum-steps=2). val_loss decreased slightly (0.6103->0.5950) and train/val recall was 1.0 every epoch with ~70% accuracy - looked healthy, but all 3 aspect labels have priors ~70-73% (above 50%), so the classifier-bias-to-priors init starts POSITIVE, trivially predicting 'positive' for everything before any training. Verified precisely: theoretical entropy floor for zero-signal prediction is 0.5970, actual final val_loss 0.5950 is statistically indistinguishable from it. Diagnostic confirms: mean AUROC=0.6129, pooled AUROC=0.5654 (barely above chance), 0/3 labels clear 0.7 AUROC. Mirror image of trial-050's original false alarm (imbalance direction flipped: here majority-positive inflates recall/accuracy instead of majority-negative deflating them). No code bug. Per user direction, skipping the usual DEAD-response LR-raise retry (judged as a data-starvation artifact of the tiny 338-example/epoch scale, not a real LR mismatch) - proceeding straight to trial-080's full 6000/8-epoch budget with LR unchanged",
}

CONFIG_RE = re.compile(r"^\[CONFIG\] (\{.*\})$")
EPOCH_RE = re.compile(
    r"^Epoch (\d+)/(\d+) \| train_loss=([\d.]+) \| val_loss=([\d.]+) \| "
    r"train_acc=([\d.]+) \| val_acc=([\d.]+) \| train_recall=([\d.]+) \| val_recall=([\d.]+)"
)

FIELDS = [
    "trial_id", "verdict", "model_name", "batch_size", "unfreeze_last_n", "learning_rate",
    "output_dropout", "esm_dropout", "attention_dropout", "weight_decay",
    "lr_schedule", "warmup_steps", "label_smoothing", "decouple_bias_wd", "grad_accum_steps",
    "max_train_samples", "max_val_samples", "epochs_completed",
    "best_epoch", "best_val_loss", "best_val_acc", "best_val_recall",
    "best_train_loss", "best_train_acc", "n_log_fragments",
]


def trial_id_from_filename(name: str) -> str:
    # trial-012_epoch5.log -> trial-012 ; trial-014_epochs5-14.log -> trial-014 ; trial-015.log -> trial-015
    stem = name[:-4] if name.endswith(".log") else name
    return re.sub(r"_epochs?\d+(-\d+)?$", "", stem)


def parse_log(path: Path):
    config = None
    epochs = []
    for line in path.read_text(errors="replace").splitlines():
        m = CONFIG_RE.match(line)
        if m:
            try:
                config = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
            continue
        m = EPOCH_RE.match(line)
        if m:
            epoch_num, _total, train_loss, val_loss, train_acc, val_acc, train_recall, val_recall = m.groups()
            epochs.append({
                "epoch": int(epoch_num),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "train_recall": float(train_recall),
                "val_recall": float(val_recall),
            })
    return config, epochs


def main() -> None:
    by_trial = defaultdict(lambda: {"configs": [], "epochs": [], "n_fragments": 0})

    for log_path in sorted(LOG_DIR.glob("trial-*.log")):
        trial_id = trial_id_from_filename(log_path.name)
        config, epochs = parse_log(log_path)
        entry = by_trial[trial_id]
        entry["n_fragments"] += 1
        if config is not None:
            entry["configs"].append(config)
        entry["epochs"].extend(epochs)

    rows = []
    for trial_id in sorted(by_trial, key=lambda t: (len(t), t)):
        data = by_trial[trial_id]
        if not data["epochs"] or not data["configs"]:
            continue
        config = data["configs"][0]
        best = min(data["epochs"], key=lambda e: e["val_loss"])
        rows.append({
            "trial_id": trial_id,
            "verdict": VERDICTS.get(trial_id, "UNVERIFIED - add to VERDICTS in this script"),
            "model_name": config.get("model_name"),
            "batch_size": config.get("batch_size"),
            "unfreeze_last_n": config.get("unfreeze_last_n"),
            "learning_rate": config.get("learning_rate"),
            "output_dropout": config.get("output_dropout"),
            "esm_dropout": config.get("esm_dropout"),
            "attention_dropout": config.get("attention_dropout"),
            "weight_decay": config.get("weight_decay"),
            "lr_schedule": config.get("lr_schedule", "constant"),
            "warmup_steps": config.get("warmup_steps"),
            "label_smoothing": config.get("label_smoothing", 0.0),
            "decouple_bias_wd": config.get("decouple_bias_wd", False),
            "grad_accum_steps": config.get("grad_accum_steps", 1),
            "max_train_samples": config.get("max_train_samples"),
            "max_val_samples": config.get("max_val_samples"),
            "epochs_completed": max(e["epoch"] for e in data["epochs"]),
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "best_val_acc": best["val_acc"],
            "best_val_recall": best["val_recall"],
            "best_train_loss": best["train_loss"],
            "best_train_acc": best["train_acc"],
            "n_log_fragments": data["n_fragments"],
        })

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} trial(s) to {OUT_PATH}")


if __name__ == "__main__":
    main()
