# Ensembling Stacking Pipeline — ESM-2 + BLAST-KNN Meta-Model (design)

**Goal:** Build a learning-to-rank-style stacking meta-model that fuses trial-013's
ESM-2 per-(protein, term) probability with the BLAST-KNN retrieval score, per
`experiments/cafa6_next_steps_plan.md` Tier 1 item 1, and confirm whether the fused
score beats trial-013 alone on held-out per-label AUROC — particularly on the ~7% of
terms where the BLAST-KNN probe already beat the model outright.

**Architecture:** One new script, `experiments/ensembling/stacking_pipeline.py`, that
reuses `blast_knn_probe_real.py`'s FASTA/BLAST/label-transfer/model-inference functions
(imported, not duplicated) to build two independent (protein × term) score-matrix
slices from the validation split — a **meta-eval** slice (rows 0:2000, already fully
computed and cached from the earlier probe) and a new **meta-train** slice (rows
2000:4000, computed by this script) — flattens each into row-per-(protein, term)
feature vectors, fits a single global `xgboost.XGBClassifier` on meta-train, and
evaluates the fused score on meta-eval using the exact same per-label-AUROC methodology
every prior probe in this line of work has used.

**Tech Stack:** Python, numpy, `xgboost` (3.2.0, already installed in `env/`, no new
dependency), the existing NCBI BLAST+ 2.17.0 install in `./blast_env/`, and
`training_run.py`'s model-inference helpers (reused via `blast_knn_probe_real.py`, not
reimplemented). No new conda env needed.

**Spec context:** `experiments/cafa6_next_steps_plan.md` Tier 1 item 1 (ensembling), its
2026-10-02 BLAST-KNN probe entries (k-mer approximation then real BLAST+, both logged
there), and `experiments/ensembling/blast_knn_probe.py` /
`experiments/ensembling/blast_knn_probe_real.py` (the two existing decorrelation probes
this design builds directly on top of).

## Background

Two prior spikes (both already run and logged in the plan doc) established:
- The ESM-2 model (trial-013) and a BLAST-KNN retrieval baseline are measurably
  decorrelated — real BLAST+ beats the model by ≥0.05 AUROC on 639/9,401 evaluable
  terms (6.8%), while the model beats BLAST on 6,362/9,401 (68%). Per-label AUROC
  correlation between the two scorers is 0.3462.
- This is exactly the pattern Tier 1 item 1 says is worth building a fusion pipeline
  around (a secondary but genuinely orthogonal evidence source), not one where "lift is
  near-zero because components are redundant."

This design is the actual fusion build the two spikes were gating.

## Scope

**In scope:**
- Exactly two evidence sources: ESM-2 (`esm_prob`) and BLAST-KNN (`blast_score`), plus
  one engineered feature, `log1p(train_positive_count_for_term)` — a proxy for how
  common/rare each term is, since the probe's bright-spot read ("plausibly terms driven
  by literal close homologs") suggests BLAST's relative value may vary with term
  rarity.
- A single global `XGBClassifier` trained once across all (protein, term) pairs — not
  one model per term (27,191 terms makes that intractable) and not split by aspect
  (no evidence yet that aspect-specific models are needed; adding that split later is
  cheap if the global model's errors turn out to cluster by aspect).
- Evaluation against the full existing evaluable-label filter and against the three
  numbers already on record (model alone: 0.8731/0.9259 mean/median AUROC; BLAST alone:
  0.6854/0.6547; both over `n_labels_evaluable=9401`).

**Out of scope (explicitly deferred, not forgotten):**
- Candidate (d), IEA/electronic-annotation features — a separate future feature source,
  not part of this stacking build.
- The test split — this entire pipeline works only within the validation split (a new
  meta-train/meta-eval split carved out of it), per the project's convention of holding
  test back for final evaluation.
- `training_run.py` itself — untouched. The ESM-2 "component" is reused exactly as
  `blast_knn_probe_real.py` already consumes it (load trial-013's checkpoint, run
  inference), not retrained or modified.
- GO-DAG consistency post-processing (Tier 2 item 5) — a separate, complementary
  post-hoc step, not part of fusing these two score sources.

## Data pipeline

- **meta-eval (rows 0:2000):** Reuse `experiments/ensembling/blast_work/blast_results.tsv`
  and `model_probs_partial.npy` as-is — both already fully computed by
  `blast_knn_probe_real.py` for this exact slice. Never recomputed, never used for
  fitting, so the comparison against the already-logged model-alone/BLAST-alone numbers
  stays apples-to-apples.
- **meta-train (rows 2000:4000):** New. Same BLAST database (`blast_work/train_blastdb`,
  already built — not rebuilt), new FASTA for just these 2000 queries, new
  `blastp` run writing to a separate `blast_results_metatrain.tsv`, and a separate
  resumable model-inference checkpoint (`model_probs_metatrain_partial.npy` /
  `model_probs_metatrain_n_done.txt`) — same checkpoint-every-20-batches pattern already
  proven to survive kills this session. Both the BLAST call and the inference loop are
  imported from `blast_knn_probe_real.py` (`write_fasta`, `labels_to_multihot`, the
  blastp subprocess invocation, the inference loop), not copy-pasted.
- Both slices use `val_split.select(range(start, end))` with the identical
  `label2id`/evaluable-label convention every prior probe in this line used.

## Feature construction

- `term_freq = train_labels.sum(axis=0)` → `log1p` → shape `(n_labels,)`, computed once
  from the train split (already how `blast_knn_probe_real.py` gets `train_labels`).
- For each slice independently: flatten the (n_rows × n_labels) score matrices and tile
  the term-frequency vector down the rows —
  `X = stack([esm_prob.ravel(), blast_score.ravel(), tile(log1p_term_freq, n_rows)])`,
  `y = labels.ravel()`. Pure numpy broadcasting, no per-row Python loop.
- Meta-train's `X` is ≈ 2000 × 27,191 × 3 float32 ≈ 653 MB — large but a flat array, not
  a nested structure; built once, not duplicated.

## Meta-model

- `xgboost.XGBClassifier(objective="binary:logistic", tree_method="hist", eval_metric="auc", random_state=42)`,
  fit once on meta-train's flattened `(X, y)`.
- Known risk: meta-train is heavily negative-dominated (consistent with trial-013's
  ~3.58%-ish positive rate at this label-space size). First attempt fits on the full
  flattened set as-is; if that proves too slow or the fit collapses to a trivial
  always-negative classifier, the documented fallback is negative subsampling (keep all
  positive rows, downsample negatives to a fixed ratio e.g. 20:1) or `scale_pos_weight`
  — not a different model family.
- Seed 42 throughout, per this repo's existing convention.

## Evaluation

- `predict_proba` on meta-eval's flattened `X`, reshape back to `(2000, n_labels)` →
  `stacked_scores`.
- Recompute the evaluable-label filter on this slice (expected to match the already-logged
  `n_labels_evaluable=9401` exactly, since it's the identical 2000 rows and labels) —
  this recomputation doubles as the correctness check described below, not just a metric
  step.
- Report mean/median AUROC for the stacked score, alongside the already-logged
  model-alone and BLAST-alone numbers, plus the same correlation/">=0.05 beats" style
  breakdown the two prior probes used, for direct comparability.
- **Targeted check:** recompute AUROC specifically on the 639-term subset where the real
  BLAST probe beat the model by ≥0.05 (derivable from the cached meta-eval score
  matrices, no new data needed) — stacked vs. model-alone on exactly that subset is the
  plan's concrete hypothesis, and the headline mean/median numbers alone could hide a
  real win there if it's diluted by the other ~93% of terms.

## Files

- **New:** `experiments/ensembling/stacking_pipeline.py` — the full pipeline described
  above (data pipeline → features → fit → evaluate → print `[RESULT]` lines in the same
  style as the two prior probe scripts).
- **Modified:** `experiments/cafa6_next_steps_plan.md` — log the result under Tier 1 item
  1, same pattern as the two prior probe entries.

## Testing / verification

No test framework is configured in this repo (CLAUDE.md). Following the established
pattern (`blast_knn_probe_real.py` itself, `generate_dataset_full.py`'s
`select_trainable_terms`), verification is a standalone in-script assertion, run before
the (expensive) meta-train BLAST call: after building meta-eval's flattened arrays,
assert `len(evaluable) == 9401` and that the recomputed model-alone/BLAST-alone mean
AUROC on meta-eval matches the already-logged 0.8731/0.6854 (within float tolerance).
This catches any row/label-order drift in the flattening logic *before* spending BLAST
and inference wall-clock on the new meta-train slice.

## Risks & mitigations

- **BLAST/CPU inference wall-clock for meta-train:** same foreground, `timeout`-wrapped,
  checkpoint-every-20-batches pattern already validated this session for the 0:2000
  slice (confirmed dramatically faster than `run_in_background`, which was heavily
  throttled). The BLAST database itself is reused unmodified, so only the new 2000
  queries add cost.
- **54M-row fit time/memory:** `tree_method="hist"` is chosen specifically for this
  scale; negative subsampling is the documented fallback if a first attempt is
  infeasible, not a silent model-family change.
- **No GPU passthrough in this sandbox:** doesn't block this design — BLAST and XGBoost
  are CPU-native tools, and the one GPU-shaped step (ESM-2 forward inference) reuses the
  already-proven CPU inference path from `blast_knn_probe_real.py` unmodified; nothing
  here is a training run.

## Non-goals / explicit follow-ups

IEA-feature addition, DAG-consistency post-processing, and per-aspect meta-models are
all plausible next iterations once this global 2-source model's result is in hand, but
none are part of this build.
