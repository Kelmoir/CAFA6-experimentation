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
