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
| trial-001 | esm2_t12_35M_UR50D | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 3 | 4192 | 1 | 16 | 16000 | 4000 | *pending* | *pending* |
