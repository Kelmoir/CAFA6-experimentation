# CAFA-6 next-steps plan — aggregated observations, ordered by expected impact

**v1.0, 2026-09-25.** Aggregates two independent sources: (1) `reports/CAFA 6 techniques comparison.md`,
a deep-research report on SOTA CAFA techniques (four research notes under
`research_notes/CAFA 6 techniques comparison/`), and (2) this project's own accumulated lessons
from `experiments/pu_learning_plan.md`, `experiments/graph_embedding_plan.md`,
`experiments/sequential_cascade_plan.md`, and `experiments/ancestor_sweep/sweep_log.md`. Ordering is
by expected impact on the actual CAFA-6 score (information-accretion-weighted Fmax), not by novelty
or ease of implementation alone — a cheap, well-evidenced win outranks an expensive speculative one.

This is a prioritization document, not an implementation plan for any single item — each Tier 1/2
entry below is a candidate for its own dedicated implementation plan once picked.

## Scope constraint: full label space, no partial GO-term selection (added 2026-09-26)

**User correction, 2026-09-26:** selecting a subset of GO terms to evaluate a technique on — the
coarse 383-term depth≤2 space from `experiments/ancestor_sweep/`, or the 437-term space this
session's PU-learning work used — skews results: a technique that looks like it works (or doesn't)
on a hand-picked subset doesn't tell you how it performs on the real task. **The goal is to predict
the complete label space — every GO term, including the ancestor-closure terms `generate_dataset.py`
already propagates — not a depth-limited or annotation-count-thresholded subset chosen for
convenience. Every item below must always have some metric scored against the real, full
dataset/train set, not a reduced label space chosen to make an experiment cheaper to measure.**

**How to apply:**
- Before scoping any Tier 1/2 item into an implementation plan, state what label space it will be
  measured against, and default to the full ancestor-closure unless there's a specific, called-out
  reason a subset is unavoidable at the prototyping stage — and even then, the item isn't done until
  its *final* reported result is against the full space, not the prototyping subset.
- This is most relevant to items 3/4/5 (hierarchy-consistency loss, IA-weighted loss, DAG-consistency
  post-processing) and item 12 (graph-embedding, if reopened) — the GO-DAG-related items are the ones
  most tempted to restrict to a depth-limited slice for tractability. That's fine for a first
  correctness check of the mechanism itself, but not as the reported result.
- Doesn't retroactively reopen the old `ancestor_sweep` coarse-label results — those are already
  documented as a separate, superseded label space (CLAUDE.md's Architecture notes). This is
  forward-looking guidance for new work, not a mandate to redo old trials.

## Open question — RESOLVED 2026-09-26, and it surfaced a bigger blocker

**Original question:** has the current `dataset_raw`-based baseline (`pu-baseline-trial001` and its
lineage) been through its own plateau-first regularization pass, or is it inheriting hyperparameters
tuned for the old, now-superseded 383-term coarse ancestor-propagated label space
(`experiments/ancestor_sweep/`)?

**Answer: no plateau-first pass at all.** Checked `experiments/logs/pu_trial_baseline.out` directly.
`pu-baseline-trial001` is a single one-off run — `esm2_t12_35M_UR50D`, 3 epochs, batch_size=16,
lr=5e-5 constant schedule, `unfreeze_last_n=2`, no LoRA, no mixed precision, 16,000 train /
4,000 val examples (a small subsample, not the full training set) — with no sweep, no config
comparison, no regularization search behind it. `sweep_log.md` has zero references to
`dataset_raw`/`generate_dataset_raw.py` — the entire beam-search sweep infrastructure was run only
against the old coarse propagated space. This checkpoint was built purely as a shared substrate for
comparing PU-learning variants against each other, not as a tuned, production-scale baseline; it
does **not** satisfy `sequential_cascade_plan.md`'s gate ("the current pipeline's flat classifier
must be tuned properly and confirmed at production scale first").

**Bigger blocker found while checking this, directly relevant to the 2026-09-26 full-label-space
scope constraint above:** `dataset_raw` (`generate_dataset_raw.py`) is itself a partial GO-term
selection, not the full label space. It selects only the 668 terms with ≥100 **directly-asserted,
unpropagated** annotations (`RAW_MIN_ANNOTATIONS = 100` in `select_raw_terms`), and — by explicit
design (see the module docstring) — does **no ancestor-closure propagation at all**, the opposite of
what the new scope constraint asks for. `generate_dataset.py`'s own coarse space (383 terms,
depth≤2, ≥50 annotations after propagation) is thresholded and depth-capped too. **Neither existing
dataset variant is the full label space (all GO terms in `train_terms.tsv` + ancestor closure) the
2026-09-26 scope constraint now requires.** This means Tier 1 item 1 (ensembling) can't be scoped
yet against a compliant baseline — a new dataset-generation pass (full ancestor closure, no
depth/annotation-count ceiling on the vocabulary) is a prerequisite, followed by a genuine
plateau-first regularization pass on it, per [[feedback_plateau_first_sweep_strategy]]. Neither of
these exists today.

**Decision (user-confirmed 2026-09-26): build the full-label-space dataset generation first,
before scoping any Tier 1 item.** This becomes its own dedicated implementation plan (effectively a
Tier 0 prerequisite) — a `generate_dataset` variant producing the full ancestor closure over every
GO term present in `train_terms.tsv` (no depth cap, no annotation-count threshold on the vocabulary,
extending `generate_dataset.py`'s existing propagation machinery rather than `dataset_raw`'s
no-propagation approach), followed by a plateau-first regularization pass on it per
[[feedback_plateau_first_sweep_strategy]] to produce the actually-tuned baseline every Tier 1/2 item
needs.

**Dataset-generation half DONE (2026-09-27/28), plateau sweep NOT started.**
`generate_dataset_full.py` builds and pushes this variant (`kelmoir/cafa6-full`,
implementation plan: `docs/superpowers/plans/2026-09-27-full-label-space-dataset-and-plateau-sweep.md`).
Two follow-up refinements after the initial build: (1) the sequence-length ceiling
(`generate_dataset()`'s shared `max_length` filter, which excludes rather than truncates
over-length proteins) was raised from the shared 1024 default to 4192, based on a direct
analysis showing only 191/29,386 terms are ever zero-train-positive purely due to a length
cutoff, 81% of which are already recoverable by ~4096 — going further needs proteins up to
35,213 residues for a handful more terms, a bad trade under quadratic self-attention; (2) a
new `select_trainable_terms()` step drops any term with zero examples among train-split
proteins actually usable at that cutoff from the *trainable classifier head* specifically —
not from the label space's definition, which stays the full closure for scoring purposes,
per the scope constraint above — since a zero-example term can't produce gradient signal
under any architecture and only dilutes BCE loss / wastes classifier-head compute otherwise.
**Final vocabulary: 27,191 trainable terms** (down from the initial 29,386; 2,195 genuinely
unlearnable terms dropped), **removed-protein fraction 0.22%** (down from 9.25% at the
original 1024 cutoff). All integrity checks pass.

**Plateau sweep DONE (2026-09-28/10-02), baseline promoted.** 13 trials, logged in
`experiments/full_label_sweep/sweep_log.md`: landed on `esm2_t12_35M_UR50D`,
`unfreeze_last_n=4`, `esm_dropout=attention_dropout=0.1` (resolved the overfitting seen at
dropout=0.0), `weight_decay=0.01` (0.05 tested and rejected), no label smoothing (0.1 collapsed
AUROC to near-chance at this label-space size), `batch_size=2`/`grad_accum_steps=8` (fastest
on the user's hardware at this sequence length), and the full ~52.7k-example train split
(trial-012/013 confirmed more data keeps helping once dropout is in place, unlike the earlier
pre-dropout train_n tests). Trial-013 (30 epochs, full train split) is the sweep's best result —
mean per-label AUROC 0.8731, median 0.9259, best val_f1 0.3741 at epoch 23 — and was promoted via
`experiments/full_label_sweep/promote_trial013.py` to `kelmoir/cafa6-esm@main` as the Tier-0
baseline. **Tier 0 is complete; Tier 1 work can now build on a real, plateau-tuned checkpoint.**

## Tier 1 — highest expected impact, actionable with current infrastructure

### 1. Ensembling / orthogonal evidence fusion
**Source:** report §"Stacked ensembles of heterogeneous evidence are the actual winning recipe" —
the single most durable, cross-generational pattern in CAFA history (GOLabeler→NetGO→GOCurator
placed at or near the top of CAFA3, CAFA4, and CAFA5 by successively fusing new evidence types into
the same stacking framework; CAFA-6's one verified solution, 11th place, combines six components via
an XGBoost learning-to-rank meta-model). This was already the user's own stated next plan before this
research was run.
**Why Tier 1:** best-evidenced technique in the whole report, and the only one with a documented
failure mode that's easy to check for as you go (lift is near-zero when components are redundant —
StarFunc got 98.7% of its ensemble's performance from one component, team Evans got 90–95% from
PLM-alone). That means: before investing in a fusion pipeline, confirm each candidate evidence
source is measurably decorrelated from the current ESM-embedding classifier's errors (e.g. per-term
AUROC agreement, or simply: does a cheap BLAST-KNN or frequency-prior baseline get *different* proteins
right?) — cheap to check, and tells you up front whether a given source is worth the engineering cost.
**Candidate orthogonal sources, cheapest first:** (a) a frequency-prior baseline (near-free, already
partially exists via this session's class-prior estimators), (b) a BLAST-KNN retrieval baseline
against the training set (standard, used by multiple CAFA-5 top-10 teams), (c) the current model's own
predictions under a second, independently-trained checkpoint (cheap decorrelation check), (d) a
non-experimental/electronic-annotation feature (GOA's own IEA-code annotations, used by 3rd place
CAFA-5 and the verified CAFA-6 11th-place solution).

**Candidate (a) ruled out as a decorrelation check (2026-10-02):** a frequency-prior baseline
(`--class-priors-path`'s KM2/Spy estimators -- see `training_run.py:908-1070`) assigns one scalar
score per GO term, identical across every protein. AUROC measures ranking; a constant score is
fully tied, so its AUROC against any validation set is *exactly* 0.5 by construction, not something
an experiment reveals -- it cannot "get different proteins right" since it never varies by protein
in the first place. Its only real role (a calibration floor) is already answered by every
`run_diagnostic()` call's `trivial_baseline_accuracy` field, which trial-013 already beats by a wide
margin (precision/recall/F1@0.5 of 0.60/0.27/0.34 vs. the always-negative floor). **Skip (a); start
the decorrelation check at (b) BLAST-KNN**, the cheapest candidate that actually produces a
per-protein-varying score.

**Candidate (b) probed (2026-10-02), result: weak overall, narrow bright spot — not yet a
Tier 1 build.** No `blastp`/BLAST+ install available locally, so `experiments/ensembling/
blast_knn_probe.py` approximates sequence similarity with amino-acid k-mer (k=3) cosine
similarity + similarity-weighted KNN label transfer (K=10) against the full ~52.6k-example train
split, scored against the identical 2000-example validation slice and evaluable-label filter
`run_diagnostic()` used for trial-013 (confirmed matching: `n_labels_evaluable=9401`, and the
model's own mean/median AUROC reproduced exactly: 0.8731/0.9259). Results: KNN mean AUROC
**0.6186**, median **0.5000** (chance, across roughly half of evaluable terms) -- far below
trial-013's model (0.8731/0.9259). Per-label AUROC correlation between the two scorers: **0.2984**.
Model beats KNN by >=0.05 AUROC on 7,685/9,401 terms (82%); KNN beats the model by >=0.05 on only
378/9,401 (4%).

**Reading this result:** the low correlation is NOT by itself evidence of a good ensembling
candidate here -- median AUROC sitting exactly at chance across half the evaluable terms means
k-mer KNN has no real ranking signal for most of them, so much of the "decorrelation" is KNN being
noise, not a genuinely different valid signal. The one real finding is the narrow 378-term (4%)
minority where KNN clearly beats the model by a wide margin -- plausibly terms driven by literal
close homologs/conserved motifs a learned embedding doesn't capture as sharply as raw sequence
identity does. This crude k-mer approximation (global trigram cosine similarity, no gapped local
alignment, no e-value filtering) is a materially weaker homology detector than real BLAST+, so this
negative-leaning result doesn't rule out a properly-built BLAST/MMseqs2 retrieval baseline --
it mainly rules out *this cheap approximation* as a blanket ensembling source. **Not yet worth
building a full fusion pipeline around; worth a cheap follow-up (install BLAST+/MMseqs2, re-run
the same comparison with real local alignment) before deciding whether to pursue this further or
move to another Tier 1 item.**

**Follow-up with real BLAST+ (2026-10-02): meaningfully stronger, still a minority signal.**
NCBI BLAST+ 2.17.0 installed in an isolated conda env (`./blast_env/`, kept separate from the
training env to avoid a dependency-graph conflict with torch/tensorflow -- the first attempt to
install it directly into the training env stalled conda's classic solver for 30+ minutes and had
to be abandoned). `experiments/ensembling/blast_knn_probe_real.py` reruns the identical comparison
(same validation slice, same evaluable-label filter, same bitscore-weighted KNN transfer, K=10)
with real gapped local alignment instead of k-mer cosine similarity. Results: BLAST mean AUROC
**0.6854** (up from k-mer's 0.6186), median **0.6547** (up from 0.5000 -- real alignment finds
genuine above-chance ranking signal for a majority of terms, unlike the k-mer version). Per-label
AUROC correlation with the model: **0.3462** (up from 0.2984). BLAST beats the model by >=0.05
AUROC on 639/9,401 terms (6.8%, up from 378/9,401=4.0% with k-mer); model still beats BLAST by
>=0.05 on 6,362/9,401 (68%).

**Verdict: real homology signal confirmed, but still clearly secondary to the deep model overall.**
BLAST-KNN is a genuine, non-trivial complementary source -- not noise, unlike what the crude k-mer
approximation suggested -- beating the model outright on a real ~7% minority of terms (plausibly
those with close training-set homologs/conserved motifs). This is exactly the kind of orthogonal
evidence Tier 1 item 1 is looking for, though at this K/weighting scheme it's far from
strong enough on its own to replace or dominate the model. **Worth carrying into an actual
ensembling/stacking build** (e.g. as one input feature per term alongside the model's own
prediction, not a standalone predictor) rather than running further probes on it in isolation --
further tuning (K, bitscore vs. e-value weighting, excluding self-hits) is better spent inside
a real stacking setup than as another isolated decorrelation check.

**Stacking build completed (2026-10-02), result: naive version underperforms, including on its
own target subset.** `experiments/ensembling/stacking_pipeline.py` (plan:
`docs/superpowers/plans/2026-10-02-ensembling-stacking-pipeline.md`, spec:
`docs/superpowers/specs/2026-10-02-ensembling-stacking-pipeline-design.md`) fits a single global
`xgboost.XGBClassifier` over `[esm_prob, blast_score, log1p(term_freq)]` on a new 2000-row
meta-train slice (validation rows 2000:4000, ~54.4M flattened (protein, term) rows), then
evaluates on the original 2000-row meta-eval slice (rows 0:2000, same one every prior probe in
this line used). Results: stacked mean AUROC **0.6179**, median **0.6497** -- *worse* than the
model alone (0.8731/0.9259) and worse than BLAST alone on mean AUROC too (0.6854 vs 0.6179,
though the medians are close: 0.6547 vs 0.6497). More tellingly, on the 639-term bright-spot
subset (where real BLAST already beat the model by >=0.05 -- exactly the subset this build was
meant to help most) the stacked score scores **0.4522** mean AUROC, *below* the model's own
**0.6048** on that same subset.

**Reading this result (corrected 2026-10-02 after a final whole-branch review caught the
original diagnosis below was wrong):** this is a real negative finding about the naive build,
not evidence against stacking as a technique -- but the mechanism is **not** overfitting via
`log1p(term_freq)`. That feature has only **997** distinct values across 27,191 terms (not
"near-unique" as first written here), and an in-sample check (fit and score on the *same* 2000
rows -- the best possible case for overfitting to show up) still collapses to mean AUROC 0.642,
barely better with `term_freq` dropped entirely (0.657). A fit that were truly overfitting
would score well in-sample; this one doesn't even there.

The real mechanism, confirmed directly: per-label **ranking resolution collapses**. Each
evaluable term's `esm_prob` column has a median of ~1991 distinct values across 2000 proteins;
the stacked score has only ~24-33. The pooled `binary:logloss` objective, fit across all
(protein, term) rows at once, spends its splits separating terms by base rate (which pooled
accuracy/std rewards) rather than ranking within each term (which is all per-label AUROC
measures) -- so most of a term's 2000 rows end up tied at the same predicted value, and ties
pull AUROC toward 0.5. `fit_meta_model()`'s degeneracy guard has been strengthened
(`_check_label_resolution()`, commit `f9f14f3`) to catch exactly this directly from training
data, rather than needing a second post-hoc diagnostic to find it -- re-running
`stacking_pipeline.py` now fails loudly at the fit step instead of silently producing the
0.6179 result above.

**Not a dead end for Tier 1 item 1, but the current naive build needs a different fix than
originally written here** -- shallower trees/fewer estimators/L2/dropping `term_freq` would
not help (less resolution, not more; measured directly above). Credible next steps, in
rough order of effort: a ranking objective grouped by term (`rank:pairwise` with `qid=term`,
which matches the spec's own "learning to rank" framing better than pooled binary
classification does), monotone constraints on `esm_prob`/`blast_score`, a larger `max_bin`, or
per-term rank/logit-transformed input features.

**Ranking objective tried (2026-10-03), result: fixes the collapse, but doesn't yet beat a
trivial untrained baseline (corrected 2026-10-03 after a second final-review pass).** Swapped
the pooled `XGBClassifier(binary:logistic)` for `xgboost.XGBRanker(objective="rank:pairwise")`,
grouped by term via `qid` -- the within-term ordering `rank:pairwise` optimizes is exactly what
per-label AUROC measures, unlike the pooled classifier's base-rate-driven loss. One real
dependency behavior confirmed directly, not assumed (`check_xgbranker_qid_ordering.py`): this
xgboost build (3.2.0) requires rows pre-sorted by `qid` -- `fit_meta_model()` sorts internally
so callers don't need to think about it.

The degeneracy guard needed two corrections, not one. The first replacement
(`_check_label_resolution()` → `_check_ranking_signal()`, comparing per-label AUROC against a
fixed `> 0.55` threshold instead of raw distinct-value counts) was itself too weak: the
original collapsed pooled classifier actually reaches in-sample median AUROC ~0.70, comfortably
clearing 0.55 -- so the guard would have let the original failure straight through. The second
pass made the threshold relative to a reference feature's own in-sample AUROC
(`median(pred_auroc) >= median(reference_auroc) - 0.05`) and restored a real-data regression
check (`check_fit_degeneracy.py`) proving the guard actually raises on the real historically-
collapsed classifier, not just on synthetic noise.

Real result on the same meta-train (rows 2000:4000) / meta-eval (rows 0:2000) slices, now also
compared against an untrained **baseline** (50/50 per-term rank average of `esm_prob` and
`blast_score` -- no fitting at all, just rank-transforming each term's column so the two
features' differing scales don't dominate each other):

| | mean AUROC | median AUROC | bright-spot (639 terms) | other 8,762 terms |
|---|---|---|---|---|
| model alone | 0.8731 | 0.9259 | 0.6048 | 0.8927 |
| stacked (XGBRanker) | 0.8732 | 0.9239 | 0.6777 | 0.8874 |
| **baseline (untrained)** | **0.8853** | **0.9421** | **0.6909** | **0.8995** |

The ranking objective genuinely fixed the catastrophic collapse (stacked is no longer far below
both model and BLAST), and it does beat the model alone on the bright-spot subset (+0.073). But
**the trained meta-model currently loses to the free, untrained baseline on every slice** --
including the bright-spot subset it was specifically meant to help -- and it also loses ~0.005
mean AUROC to the model alone on the other 8,762 terms, which an earlier version of this
write-up (now corrected) claimed didn't happen. The earlier "no corresponding loss elsewhere"
claim was wrong: it came from comparing only the aggregate mean (where the bright-spot gain
happens to roughly cancel the broader loss) without ever computing the non-bright-spot subset
directly. `report_results()` now prints the baseline and both subset breakdowns every run, so
this can't recur silently.

**Reading this result:** a pooled tree fit over raw `[esm_prob, blast_score, log1p(term_freq)]`
apparently still can't out-rank a baseline that sidesteps the scale problem entirely via
per-term rank transforms, even with the objective now matching the eval metric. The credible
next step is exactly what was already listed below as untried: feed the meta-model per-term
rank- or logit-transformed features instead of raw scores, so it sees the same scale-normalized
signal the baseline does. Monotone constraints and a larger `max_bin` remain untried and
lower-priority than the rank-transform change, since they don't address the scale mismatch
directly. **Tier 1 item 1 still has no result that beats a trivial baseline** -- the
rank-transform-features attempt is the next thing to try before concluding anything stronger.

**Bright-spot subset tested out-of-sample (2026-10-03), result: does not replicate -- it was
winner's-curse, not a real term-level property.** The 639-term bright-spot subset (where BLAST
beat the model by >=0.05 AUROC) was defined using meta-eval's own 2000 rows and then evaluated
on those same 2000 rows -- a textbook selection-bias setup (any subset selected for "BLAST wins
here" on a small sample will show BLAST winning there, whether or not it's a real property of
those terms). `experiments/ensembling/bright_spot_test_set_probe.py` resolves this directly:
freezes the 639 term *indices* exactly as derived from meta-eval, then scores model/BLAST/
baseline/stacked AUROC on an independent 5000-row slice of the held-out, really-labeled `test`
split (distinct proteins, same label space -- not Kaggle's unlabeled submission test set),
without re-selecting the subset from the new data. Of the 639 frozen terms, 472 had both
classes present in this sample (sparse terms don't always survive subsampling). Result:

| | mean AUROC | median AUROC |
|---|---|---|
| model alone | 0.8577 | 0.8975 |
| BLAST alone | 0.7092 | 0.7040 |
| stacked (XGBRanker) | 0.8669 | 0.9087 |
| baseline (untrained) | 0.8803 | 0.9239 |

On meta-eval these same 639 terms showed BLAST beating the model by **+0.1554** mean AUROC
(0.7735 vs. 0.6048). On the independent test sample the gap **reverses**: the model beats BLAST
by **-0.1484** (0.8577 vs. 0.7092) -- essentially the same pattern as the other 13,140 evaluable
test terms (model 0.8809 vs. BLAST 0.6903), i.e. these aren't special terms at all out-of-sample.
**The bright spot was a sampling artifact of a 2000-row evaluation slice, not a real,
exploitable term-level property.** A better ranking algorithm targeting "terms where BLAST has
genuine complementary signal" has no such target to aim at, at least not this one -- the premise
itself doesn't hold up. This also retroactively explains why the trained stacked model's
bright-spot-subset "win" over the model alone on meta-eval (0.6777 vs. 0.6048, logged above) was
never something to chase: it was measured on exactly the circularly-selected sample where any
BLAST-containing blend was essentially guaranteed to look good.

One consistent thread survives this result, though: the untrained rank-average baseline still
beats both the model alone and the trained stacked ranker on *this* independent test sample too
(0.8803/0.9239, vs. model's 0.8577/0.8975 and stacked's 0.8669/0.9087 on the bright-spot-frozen
terms; 0.8923 vs. 0.8809/0.8793 on the other terms) -- the per-term rank-transform idea stays the
credible next step, just without any bright-spot framing attached to it. Any future ensembling
attempt on this line should drop "bright-spot subset" as a target entirely and just aim to beat
the baseline across the full evaluable label space.

**Rank-transformed features tried (2026-10-03), result: does NOT close the gap -- the
raw-value-scale hypothesis is refuted.** `experiments/ensembling/rank_features_probe.py` fed the
exact same pipeline (same meta-train/meta-eval slices, same `fit_meta_model()`,
`report_results()`) per-term rank-transformed `esm_prob`/`blast_score` (via the new
`build_rank_features()`) instead of raw values, to test the hypothesis that a pointwise tree
scorer underperforms the baseline because it can't infer each term's wildly different raw-value
scale/sparsity on its own (confirmed directly: `blast_score` is nonzero for <1% of rows on
6,061/9,401 evaluable terms, but >50% of rows on others). Result: stacked mean/median AUROC
**0.8688/0.9225** -- *worse* than the raw-feature version (0.8732/0.9239), and still below both
the model alone (0.8731/0.9259) and the baseline (0.8853/0.9421). Rank-transforming the inputs
did not help; if anything it made the headline numbers slightly worse (bright-spot subset
improved marginally, 0.6827 vs. 0.6777, but the other 8,762 terms got slightly worse, 0.8824 vs.
0.8874).

A follow-up in-sample check ruled out both standing hypotheses in one shot: fitting and scoring
on the *identical* 2000 meta-eval rows (the best possible case -- zero generalization gap, and
features already rank-normalized) still only reaches stacked mean AUROC **0.8834**, which is
*still below the baseline's 0.8853*, even here. That means the trained model's shortfall is
**not** overfitting to meta-train (ruled out: even a zero-generalization-gap in-sample fit
can't clear the bar) and **not** raw-feature scale miscalibration (ruled out: the features were
already rank-normalized for this check). The most likely remaining explanation: XGBoost's
default-hyperparameter trees have no built-in bias toward a monotonic function of their inputs,
and a flexible tree ensemble fit via noisy per-term pairwise comparisons (many terms have only a
handful of positives in 2000 rows) can easily introduce small rank-order-violating quirks that a
strictly-monotonic, zero-variance closed-form blend (the baseline) structurally cannot have --
hurting a fine-grained per-label ranking metric like AUROC even when the model has access to
equal or better information. This points directly at **monotone constraints** (`monotone_constraints`
on `rank_esm`/`rank_blast`, already on the original deferred list) as the next, now better-
motivated thing to try -- not a shot in the dark, but a direct test of a mechanism the data
itself just ruled the alternatives out in favor of.

**Monotone constraints tried (2026-10-03), result: closes the gap to parity with the
baseline.** `experiments/ensembling/monotone_constraints_probe.py` adds `monotone_constraints=(1, 1, 0)`
to `fit_meta_model()` (now an optional parameter, `check_monotone_constraints.py` proves it's
actually applied to the fitted booster and that predictions are empirically non-decreasing in
the constrained features) -- forcing the score to be non-decreasing in `rank_esm`/`rank_blast`
while leaving `log1p(term_freq)` unconstrained, on top of the already-tried rank-transformed
features. A quick in-sample check first (fit and score on the same 2000 rows) showed the model
beating the baseline for the first time in this whole investigation: mean AUROC **0.8861** vs.
baseline's 0.8853. Out-of-sample (fit on meta-train, evaluated on meta-eval, the result that
actually matters):

| | mean AUROC | median AUROC |
|---|---|---|
| model alone | 0.8731 | 0.9259 |
| blast alone | 0.6854 | 0.6547 |
| stacked (rank features + monotone constraints) | **0.8854** | 0.9398 |
| baseline (untrained) | 0.8853 | 0.9421 |

Stacked mean AUROC (0.8854) now **ties the baseline** (0.8853 -- a 0.0001 difference, i.e. noise,
not a real win) instead of trailing it by 0.012-0.017 as every previous attempt did. Median AUROC
is still slightly behind (0.9398 vs. 0.9421). On the (now-known-to-be-non-generalizing, see above)
bright-spot subset, stacked (0.6949) edges past the baseline (0.6909) for the first time; on the
other 8,762 terms the two are essentially identical (0.8993 vs. 0.8995). **This confirms the
non-monotonicity hypothesis was the real mechanism**, not a coincidence: going from "clearly
worse than baseline" (every raw-feature and rank-feature-only attempt) to "statistically tied"
required exactly the fix that targets tree non-monotonicity, and nothing else tried did.

**Where this leaves Tier 1 item 1:** the meta-model has gone from a clear loser (0.6179 at the
start of this investigation) to parity with a trivial, parameter-free baseline -- real, honest
progress, but not yet evidence that learned stacking adds value over the free blend. The natural
next increment (untried) is tuning around this configuration -- e.g. a non-zero `weight` other
than 50/50 in the closed-form baseline for comparison, or a wider monotone-constrained
hyperparameter sweep (tree depth, learning rate, `max_bin`) -- rather than another new mechanism,
since the diagnosis (tree non-monotonicity) is now confirmed rather than speculative.

**Hyperparameter search tried (2026-10-04), result: barely, consistently ahead of the baseline
for the first time.** Ran that hyperparameter sweep -- `experiments/ensembling/local_hpo_search.py`,
a 40-config random search (depth, learning rate, `n_estimators`, `min_child_weight`, `subsample`,
`colsample_bytree`, `reg_lambda`, `max_bin`, and the monotone-constraints variant itself) scored
by 4-fold CV split by protein over meta-train, run on the user's local GPU (Colab is retired for
this project -- see the hardware_local_gpu memory note; a sandboxed-session design that defaulted
to a Colab upload/exec workflow was corrected before implementation). Caught and fixed a severe
performance bug before trusting any results: the CV scoring loop recomputed a full boolean mask
over the ~13.6M-element validation array once per term (up to 27,191 times) per fold -- an
O(n_terms x n_val_rows) cost that made a real run take ~17 hours for 40 trials. Fixed via reshape
+ column-slicing plus a fully vectorized rank-sum AUROC (verified to match `sklearn.roc_auc_score`
exactly, including on ties) instead of calling it once per term -- the bug was purely about speed,
not correctness, so the already-completed search's results remained valid throughout.

Winning config (trial 33, CV mean AUROC 0.8913): `max_depth=7, learning_rate=0.1235,
n_estimators=300, min_child_weight=1, subsample=0.6495, colsample_bytree=0.9710,
reg_lambda=0.6240, max_bin=1024, monotone_constraints=(1,1,0)`. Refit on the real meta-train and
evaluated once, honestly, on the never-touched meta-eval slice:

| | mean AUROC | median AUROC |
|---|---|---|
| model alone | 0.8731 | 0.9259 |
| stacked (HPO-tuned) | **0.8857** | **0.9418** |
| baseline (untrained) | 0.8853 | 0.9421 |

For the first time in this entire investigation, the trained model is (very slightly) ahead of
the baseline on *every* reported breakdown, not just tied on the headline number: mean (+0.0004),
the other 8,762 terms (0.8997 vs. 0.8995), and the (known non-generalizing, see above) bright-spot
subset (0.6934 vs. 0.6909). Median is nearly tied (0.9418 vs. 0.9421, closer than the untuned
monotone-constrained config's 0.9398). These margins are tiny -- almost certainly within noise on
a 2,000-row/9,401-term evaluation -- so this should be read as "no longer behind, and directionally
ahead everywhere" rather than a decisive win. **Tier 1 item 1's honest status: the mechanism (rank
objective + monotone constraints) is confirmed and tuned close to its ceiling against this
baseline; there's no demonstrated case yet for shipping learned stacking over the free rank-average
blend**, though it no longer costs anything to use either.

**Decision (2026-10-04): closing out Tier 1 item 1 -- not adopting the stacking meta-model.**
The win margin over the untrained 50/50 rank-average baseline (+0.0004 mean AUROC) is within
noise at this sample size, after real tuning effort (ranking objective, monotone constraints, a
40-config HPO search) closed what started as a large, clear loss. Keeping it simple: there's no
case for shipping the extra complexity (a trained XGBRanker, its HPO config, the qid-sort
plumbing) over the free blend for a result this close. The investigation itself was not wasted --
it produced and fixed two real bugs in the meta-model's own degeneracy guard, confirmed BLAST-KNN
genuinely has no exploitable per-term "bright spot" (tested out-of-sample, not assumed), and
leaves a validated, reusable per-term rank-average baseline on record for any future ensembling
attempt to beat. All of `experiments/ensembling/`'s scripts stay as the historical record (same
convention as the two BLAST probes before them); no further work planned here unless a
fundamentally different signal (not just a retuned combination of the same two inputs) comes
along.

### 2. Partial-Knowledge (PK) conditioning
**Source:** report §"CAFA-6 keeps CAFA-5's evaluation regime, and partial knowledge now dominates the
score" — PK (proteins that already carry *some* GO annotations, need new ones predicted) is 69.7% of
CAFA-5's total scoring mass, with a 47.8–59.7% Fmax drop vs. the No-Knowledge setting, and CAFA-5's own
assessors attribute part of that gap to methods simply not yet being built for this setting.
**Why Tier 1:** if the current pipeline predicts every protein's full label set from sequence alone,
with no way to condition on a protein's already-known annotations, it is optimizing for a minority
(NK/LK) of the real scoring surface. This reframes PK from "predict function for this protein" to
"predict the REST of this protein's function given what's already known" — closer to link prediction
on a partially-observed graph than to sequence classification.
**Prerequisite check (done, 2026-10-04):** current setup is pure NK. `generate_dataset.py`'s
`protein_to_labels` is each protein's *full* true-path-propagated annotation closure — there is no
"known subset held out as input" anywhere. `training_run.py`'s only non-sequence input channel,
`aspect_input`, is a static 3-way C/F/P namespace one-hot (from a precomputed `aspect_labels` column),
not conditioning on any of the protein's own ground-truth labels. Train/val/test is split whole-protein
(cluster-aware CD-HIT), never by partial-annotation masking within a protein. So item 2 is **both** a
data-generation change (need a per-protein known/novel annotation split) and a model-input change
(need a new input channel carrying the "known" subset, analogous to how `aspect_input` already injects
side-information into the classifier head).
**CAFA-5 dataset as PK baseline source — checked and refuted (2026-10-04):** CAFA-5's own Kaggle
release (`train_terms.tsv`, UniProtKB snapshot 2022-11-17 + `testsuperset`) is a **single annotation
snapshot**, structurally identical in shape to CAFA-6's own `train_terms.tsv` — one full annotation
closure per protein, no known/novel split, no ready PK data. The actual NK/LK/PK ground truth is
computed by CAFA's assessors *post-hoc*, outside the Kaggle download, by diffing **two independently-
dated UniProt-GOA experimental-annotation snapshots** (t0 = submission deadline, t1/te = later
accrual): a protein is PK if it already has an experimental annotation in aspect S at t0 and gains a
*novel* one in S by t1; only the novel terms score, with the t0 terms collected into a `known_terms.tsv`
file the official evaluator (`cafaeval`/`protea`, and the `CAFA-Challenge/CAFA_evaluation` GitHub repo)
consumes via a `-known`/`--exclude` flag to suppress credit for repeating prior annotations. That
`known_terms.tsv`/t1-ground-truth pair is not bundled with the Kaggle competition data — the official
eval repo instead points to a separately-hosted FigShare archive for benchmark ground truth, consistent
with organizers holding t1 data back for scoring rather than shipping it to participants. The one
documented community attempt to reconstruct a time-delayed holdout from CAFA-5 data (`wanglab/cafa5` on
Hugging Face, "temporal holdout test set (2023-2025)...with annotation date tracking") only achieves
NK/LK, not full PK, and required going back to dated UniProt-GOA snapshots to build it — confirming
there is no one-click PK download, from CAFA-5 or otherwise. **Conclusion: no shortcut.** Building a
genuine, temporally-faithful PK split costs the same two-snapshot GOA-diffing work regardless of
whether it's keyed to CAFA-5's or CAFA-6's protein set, so there's no reason to prefer CAFA-5's data for
this. The two real options going forward are: (a) a **synthetic/approximate PK split** built directly
from CAFA-6's own `train_terms.tsv` — reveal a random subset of each protein's own true annotation
closure as "known" input, predict the rest — buildable immediately from data already in hand, at the
cost of not capturing genuine temporal/scrutiny-order bias in which terms get annotated first; or (b) a
**real time-delayed split** by fetching two dated UniProt-GOA snapshots and diffing them the way CAFA's
own evaluator does — more faithful, but a standalone data-acquisition task of its own. Recommend (a) as
the Tier-1-scoped first cut, with (b) noted as a possible later refinement.

**Direction revised (2026-10-05): option (b) turns out free, no need to wait for CAFA-7.** This repo
already has a genuinely later UniProt snapshot cached locally — `data/uniprot/uniprot_sprot.dat.gz`
(fetched 2026-08-01 for the unrelated `mf_span` side-project) — against `train_terms.tsv`'s October
2025 vintage, ~9-10 months of real temporal distance with zero new fetching. Built
`experiments/pk_eval/scripts/build_temporal_pk_eval.py` (see
`docs/superpowers/specs/2026-10-05-temporal-pk-eval-design.md` and
`docs/superpowers/plans/2026-10-05-temporal-pk-eval.md`): diffs t0 (`train_terms.tsv`) against t1 (the
cached Swiss-Prot dump), per-aspect PK/LK/NK categorization matching CAFA's own methodology, plus an
`unseen_protein` pool (Swiss-Prot accessions absent from `train_terms.tsv` entirely — proteins never
seen anywhere in the existing pipeline). This is item 2 part (A), an **evaluation harness only** — no
training/architecture change yet; part (B) (an actual "known annotations" model input channel) is a
deliberate follow-up once (A)'s numbers are in hand.

**(A) run (2026-10-05):** 92.1% of t0's ~82.4k proteins matched ≥1 novel experimental annotation in the
cached dump. Balanced 5,242/5,242 `added_annotation`/`unseen_protein` (seed 42) →
`experiments/pk_eval/temporal_pk_validation/` (10,484 rows, full 27,191-term label space). Scored
`trial-013` (`kelmoir/cafa6-esm@main`) against it with real IA-weighted Fmax
(`experiments/pk_eval/scripts/score_temporal_pk_eval.py`, known-terms stripped from predictions before
scoring, cross-checked once against the real `cafaeval` 1.3.0 package):

| aspect | PK (n) | LK (n) | unseen (n) |
|---|---|---|---|
| C | 0.0052 (3834) | 0.5697 (18) | 0.1726 (5242) |
| F | 0.0038 (2947) | 0.5619 (12) | 0.3171 (5242) |
| P | 0.0049 (3848) | 0.3759 (37) | 0.2267 (5242) |

**Interpretation:** the current pure-NK-trained model collapses almost completely on genuine PK
examples (Fmax ~0.004–0.005 — barely above a trivial floor) despite PK being the dominant category by
volume (3834–3848 proteins vs. 12–37 for LK), while it does comparatively well on LK and unseen (novel
proteins it never saw known-annotation context for at all). This is close to the CAFA-5-documented
"PK methods aren't built for this setting" gap the report flagged, and is the strongest evidence yet
*for* building part (B)'s known-annotation input channel: a plausible mechanism (not yet directly
verified — this harness measures the aggregate gap, not per-prediction attribution) is that the model,
having no way to know which terms are already known, spends its confidence on exactly those (likely the
easiest, most sequence-obvious ones) and has little signal left for the novel terms once known ones are
stripped at scoring time — the kind of problem a known-annotation input channel exists to fix.

**Scope caveat (found via the `cafaeval` cross-check, 2026-10-06): these Fmax numbers are not directly
comparable to official CAFA leaderboard scores.** They're computed over `dataset_full`'s 27,191-term
*trainable* vocabulary (the model structurally cannot predict outside it — see Architecture notes), not
the full GO ontology. Confirmed by root-causing an initial cross-check failure: the real `cafaeval`
package propagates both predictions and ground truth through its own complete per-namespace ontology
(25,949/10,130/4,040 terms for P/F/C) before scoring, vs. our vocabulary-scoped 18,263/6,244/2,684 —
feeding `cafaeval`'s own matrices into our hand-rolled formula reproduces its IA-weighted `f_w` within
0.003 for every aspect, proving the *formula* is correct; the gap is purely about which terms get
scored. The relative pattern above (PK collapsing vs. LK/unseen) is real and vocabulary-scope-independent,
but the absolute Fmax values would be somewhat higher under a full-ontology scoring (more generic
ancestor terms, which propagation makes easy credit, would be included). Not something to "fix" here —
a different, deliberate truncation (`select_trainable_terms`) made this same trade-off for the whole
pipeline already, for good reason (zero-train-example terms can't produce gradient signal regardless).

**(B) built and first-trialed (2026-10-09/10):** training-time known-annotation conditioning —
see `docs/superpowers/specs/2026-10-08-known-annotation-conditioning-design.md` and
`docs/superpowers/plans/2026-10-08-known-annotation-conditioning.md`. Each training example's own
t0 annotation closure is split on the fly, per epoch, into a "known" (revealed) subset and a
"novel" (hidden, scored) subset; the known subset is pooled through a learned per-term
`nn.EmbeddingBag` (sum-pooling, `known_embed_dim=64`) and concatenated into the classifier head
alongside the pooled CLS representation (`KnownConditionedEsmForSequenceClassification`), with the
revealed terms excluded from the loss via the existing `loss_mask` mechanism. No new dataset
artifact — the masking lives entirely in `preprocess_batch`, re-randomized every epoch.

First real GPU trial (`known-input-trial-001`, `facebook/esm2_t12_35M_UR50D`, otherwise the Tier-0
plateau sweep's settled hyperparameters, 3 epochs, `kelmoir/cafa6-esm@known-input-trial-001`):
`--run-diagnostic` reported `mean_auroc=0.8483`, `pooled_auroc=0.9755`,
`best_f1=0.4243@threshold=0.16` — in the same ballpark as the Tier-0 baseline's own diagnostic
(`trial-013`: mean AUROC 0.8731), on only 3 epochs with no hyperparameter re-tuning for the new
input channel. This is a first-cut sanity check, not yet a PK-eval-harness comparison: the real
test is whether this checkpoint closes the Fmax gap on genuine PK examples from part (A)'s harness
(`experiments/pk_eval/`), which is part (B)'s remaining step (Task 6: wire real known-terms into
`score_temporal_pk_eval.py`) — a plausible-looking diagnostic AUROC doesn't by itself confirm the
PK mechanism works; the PK-eval harness is the metric that can actually see the effect
([[feedback_check_metric_can_see_the_effect]]).

**Caveat added after the final review (2026-10-10):** the `mean_auroc=0.8483`/`pooled_auroc=0.9755`
figures above were computed by a diagnostic that scored revealed/known cells as prediction targets
even though the model had them as input — not a fair comparison against `trial-013`'s pure-NK
diagnostic, which has no such cells at all. Fixed in `run_diagnostic` (now excludes known cells from
AUROC/F1 the same way `compute_loss` already excludes them from training), but this specific number
was logged before the fix and hasn't been recomputed. Don't treat `0.8483` as a validated
same-ballpark comparison; the PK-eval harness result just below is the number that matters anyway.

**Task 6 PK-eval harness result (2026-10-10): negative on this first trial.** Wired
`score_temporal_pk_eval.py` to feed each PK/LK row's real t0 known terms through
`known-input-trial-001`'s `KnownConditionedEsmForSequenceClassification` (NK/unseen rows get an
empty bag, same as training). Fmax vs. the pure-NK `trial-013` baseline, same harness/vocabulary:

| aspect | PK: baseline → known-input (n) | LK: baseline → known-input (n) | unseen: baseline → known-input (n) |
|---|---|---|---|
| C | 0.0052 → 0.0043 (3834) | 0.5697 → 0.5833 (18) | 0.1726 → 0.1632 (5242) |
| F | 0.0038 → 0.0024 (2947) | 0.5619 → 0.3475 (12) | 0.3171 → 0.2683 (5242) |
| P | 0.0049 → 0.0047 (3848) | 0.3759 → 0.2681 (37) | 0.2267 → 0.1931 (5242) |

**PK did not improve — it's flat-to-slightly-worse on all three aspects**, and LK/unseen are
mostly worse too (C-aspect LK is the one exception, +0.014). The known-annotation input channel
has not yet closed the PK gap this harness exists to measure.

**Confound, not yet controlled for:** `known-input-trial-001` trained for only 3 epochs vs.
`trial-013`'s 23 — the diagnostic AUROC comparison above already flagged this gap, and the
PK-eval result is consistent with "undertrained on everything," not necessarily "the mechanism
doesn't work." The known-term `EmbeddingBag` is newly-initialized, with its own parameters to
learn from scratch alongside the classifier head's adjustment to the wider input — 3 epochs may
simply not be enough for the model to learn to *use* the known-term signal at all, let alone
learn it well enough to outweigh having ~8x less total training. **Before drawing any conclusion
about whether this mechanism works, re-run at a comparable epoch/step budget to `trial-013`** —
this first trial answers "does it break anything" (no) and "is it a free win out of the box" (no),
not yet "does the mechanism work when given a fair chance." an earlier version of this trial's hand-off command included `--push-to-hub`,
which per this repo's own convention unconditionally promotes a run's best checkpoint to
`kelmoir/cafa6-esm@main` — briefly overwriting the Tier-0 baseline with this unvetted trial's
checkpoint. Caught and reverted the same session (see git history around 2026-10-09); the plan
doc was corrected to drop that flag for future trials of this kind.

**Confound resolved (2026-10-11): re-ran at `trial-013`'s own budget — still negative.**
`known-input-trial-002`, identical config to `trial-001` except `--epochs 30
--checkpoint-selection f1` (matching `trial-013` exactly; best epoch 29, `val_f1=0.4282`),
pushed to `kelmoir/cafa6-esm@known-input-trial-002`. The diagnostic (post-review-fix, known
cells masked out) now clearly beats `trial-013`'s: mean AUROC 0.9173 vs 0.8731, median 0.9586
vs 0.9259, pooled AUROC 0.9848 vs 0.9773, best F1 0.4440@0.24 vs 0.4197@0.26 — so "undertrained"
is ruled out as an explanation; given a fair training budget, the model learns to use the
known-term channel and generalizes better overall. The PK-eval harness result, same
baseline/vocabulary as above:

| aspect | PK: baseline → known-input-002 (n) | LK: baseline → known-input-002 (n) | unseen: baseline → known-input-002 (n) |
|---|---|---|---|
| C | 0.0052 → 0.0045 (3834) | 0.5697 → 0.4540 (18) | 0.1726 → 0.1726 (5242) |
| F | 0.0038 → 0.0027 (2947) | 0.5619 → 0.4446 (12) | 0.3171 → 0.3224 (5242) |
| P | 0.0049 → 0.0040 (3848) | 0.3759 → 0.2514 (37) | 0.2267 → 0.2273 (5242) |

**PK is still flat (within noise of `trial-001`'s already-flat numbers), and LK got worse across
all three aspects** despite the general diagnostic improving. Conclusion: this isn't an
undertraining artifact — a model that's unambiguously better on ordinary held-out AUROC/F1 still
does not use the known-term channel to do better on genuinely held-out (PK) proteins, and LK (the
case the mechanism should help most directly, since those proteins' PK-style history is exactly
what's being revealed) gets worse, not better. The mechanism as designed — a single sum-pooled
`EmbeddingBag` of revealed terms concatenated into the classifier head — does not close the PK
gap. Shelving part (B) rather than iterating further on this architecture; the reviewer's
noted second confound (uniform-random reveal vs. ancestor-closed reveal, logged as a ruling above)
is the one unexplored lever, but LK getting worse with more training is a more fundamental signal
against this approach than a sampling-strategy tweak is likely to fix.

### 3. GO-DAG true-path-rule consistency, as a *lightweight* training/inference constraint — NOT the
full graph-embedding plan
**Source:** report §"Structure and ontology awareness pay off mostly at the long tail of rare GO
terms" — DeepGOZero's axiom-embedding losses and TALE's ancestor-constrained loss (parent score ≥
child score) show concentrated, well-evidenced gains specifically on rare/near-zero-shot terms (<100
annotated proteins), which matter most under IA-weighting.
**Important distinction from `experiments/graph_embedding_plan.md`, which stays gated (see
Deprioritized section below):** that plan's mechanism is sharing statistical strength across
structurally-related terms via a GNN, explicitly gated on an OWA/PU-learning fix because it would
"just learn a more sophisticated way to fit the same noise" if the underlying labels are
incomplete-as-negative. A true-path-rule *consistency* constraint (parent score ≥ child score) is a
different, weaker claim — it's a hard logical fact independent of annotation completeness (GO's true
path rule is definitional, not statistical), and the dataset already propagates it into the POSITIVE
labels at generation time (`generate_dataset.py`'s ancestor closure). The missing piece is enforcing
that same consistency on the model's *predicted scores*, which is orthogonal to the OWA problem and
not blocked by the same gate. Don't conflate the two when scoping this — a hierarchy-consistency loss
add is Tier 1; a full graph-embedding architecture is not.

**Result (2026-10-13): mechanism works as a structural fix, roughly neutral on AUROC/F1.**
Implemented per `docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md`: a
`relu(child_logit - parent_logit)` hinge over 47,068 direct `is_a`/`part_of` edges (both
endpoints in-vocab), added to training loss only, plus a diagnostic violation-rate/
rarity-stratified-AUROC backfill so `trial-013` has a real control number it never
originally computed. `hierarchy-loss-trial-001` (`--hierarchy-loss-weight 0.1`, otherwise
identical to `trial-013`'s config — same 30 epochs, `--checkpoint-selection f1`, best
epoch 29 at `val_f1=0.3708`), pushed to `kelmoir/cafa6-esm@hierarchy-loss-trial-001`:

| metric | `trial-013` (backfilled control) | `hierarchy-loss-trial-001` |
|---|---|---|
| mean / median AUROC | 0.8731 / 0.9259 | 0.8744 / 0.9270 |
| pooled AUROC | 0.9773 | 0.9777 |
| best F1 @ threshold | 0.4197 @ 0.26 | 0.4197 @ 0.24 |
| rare-term mean AUROC (n=7014) | 0.8801 | 0.8827 |
| common-term mean AUROC (n=2387) | 0.8526 | 0.8502 |
| **hierarchy violation rate** | **0.1196 (11.96%)** | **0.0004 (0.04%)** |
| mean violation magnitude | 0.000195 | 0.0160 |
| expected violation mass (rate × magnitude) | 2.33e-5 | 6.23e-6 |

**Correction (2026-10-13, final review caught this before merge):** the violation-rate
number alone overstates the effect. Rate collapses ~308x (11.96% → 0.04%), but mean
violation *magnitude* among the pairs that still violate grows ~82x (0.000195 → 0.0160) in
the same comparison — the easy, near-zero-margin violations that dominated the control's
count are gone, but a residual population of harder violations with much bigger margins
remains. The honest single number is **expected violation mass** (rate × mean magnitude,
the quantity the hinge loss actually minimizes in expectation): control 2.33e-5 → trial
6.23e-6, a real but far more modest **~3.75x reduction**, not ~300x. Treat the raw rate
number as misleading on its own; mass is what the loss term is actually driving down.

**The mechanism's direct, mechanical claim holds, at a more modest size than the raw rate
suggests**: training-time expected violation mass drops ~3.75x with a weight of only 0.1,
and the vast majority of (example, edge) pairs that used to violate the true-path rule at
all now don't (rate -308x), even though the few that remain do so by a larger margin.
**The claimed downstream benefit — concentrated AUROC gains on rare terms — did not
materialize**: rare-term AUROC moved +0.0026 (0.8801 → 0.8827) and common-term AUROC moved
-0.0024 (0.8526 → 0.8502) — both well within noise (no formal significance test run; at
2,000 val examples rare-term AUROCs rest on few positives per term), net a wash.
Pooled/mean/median AUROC and best-F1 are likewise flat. Both checkpoints' configs were
confirmed identical on the Hub (`max_length` 4192, epochs 30, f1 selection, unfreeze 4,
bs 2/accum 8, wd 0.01, dropouts 0.1, seed 42, lr 5e-5) except the new flag, so the
comparison is fair.

**Verdict:** this is a cheap, directionally-correct structural fix (real, if partial,
reduction in true-path violations, not harmful) worth keeping on priors, but it is not the
rare-term accuracy lever the DeepGOZero/TALE precedent predicted at
`--hierarchy-loss-weight 0.1`, and the violations that remain are the larger-margin kind a
higher weight would need to address, not the vast population of noise-level ones the rate
number alone suggested were representative. Not escalating to a weight sweep without a
stronger reason to believe a different weight would move the AUROC numbers rather than
just violation mass further — logging as a mild positive (ship the consistency fix)
rather than iterating on this item further for now.

## Tier 2 — cheap, well-evidenced, low implementation risk

### 4. Replace flat BCE with an IA-weighted or ranking loss
**Source:** report §"Protein language model embeddings are table stakes" — two independent CAFA-5
top-6 teams (Evans, Freddolino Lab) found standard BCE a genuine weak point under this label
cardinality/imbalance and switched to IA-weighted soft-F1-style or ranking losses.
**Why Tier 2 not Tier 1:** well-evidenced and cheap (a loss-function swap, no architecture change),
but the expected per-point gain is smaller and less central than items 1–3; do this alongside or
right after Tier 1 work, not before it.

**Result (2026-10-18): `--soft-f1-loss-weight 1.0` makes every metric worse, not better —
diagnosed as a loss-scale mismatch, not a dead end.** Implemented `soft_f1_loss` (protein-level
only, per the explicit scoping decision made before implementation — see git history) as an
additive term on top of BCE (`train_loss = bce + weight * soft_f1_loss`), computed per-protein
with the same IA-weighting convention as `evaluate_cafa_score.py`'s `cafa_weighted_fmax`, excluding
rows with zero true-IA annotation from the mean. Ran a real 30-epoch GPU trial
(`soft-f1-trial-001`) at `trial-013`'s exact config (`esm2_t12_35M_UR50D`, `unfreeze_last_n=4`,
`batch_size=2`/`grad_accum_steps=8`, `max_length=4192`, full ~52.7k train split) plus
`--soft-f1-loss-weight 1.0`, ~17.7 wall-clock hours (63,666s / 30 epochs, ~2,122s/epoch — materially
more expensive to iterate on than item 3/5's trials):

| | mean AUROC | pooled AUROC | best F1 | rare AUROC | common AUROC |
|---|---|---|---|---|---|
| `trial-013` (pure BCE) | **0.8731** | **0.9772** | **0.4186** | **0.8812** | **0.8500** |
| `soft-f1-trial-001` (BCE + 1.0×soft-F1) | 0.7681 | 0.9239 | 0.3990 | 0.7658 | 0.7747 |

Every single metric is worse, not just flat — this is a real regression against the baseline, not
noise.

**Root cause, confirmed by back-calculating the BCE component from the logged fields (not just
guessed): the weight is miscalibrated by roughly 60×, so the model trained almost entirely on
soft-F1 and barely on BCE at all.** `train_soft_f1_loss` and `train_loss` track each other almost
exactly across all 30 epochs (e.g. epoch 1: `train_loss=0.8388`, `train_soft_f1_loss=0.8277`;
epoch 30: `train_loss=0.6100`, `train_soft_f1_loss=0.6009`) — subtracting gives an implied BCE
contribution of only ~0.009–0.011 per epoch, matching `val_loss` (pure BCE, untouched by the new
term) almost exactly throughout (0.0112 → 0.0113). At this task's label cardinality (27,191 terms,
severe positive-rate imbalance), BCE's own natural loss scale is tiny; soft-F1's `1 - score`
formulation starts near its ceiling and stays there, bounded in [0, 1]. A weight of `1.0` treats
the two terms as comparable in scale when they aren't even close — in practice this wasn't an
*additive refinement* of the well-tuned BCE dynamics at all, it was a near-total replacement by an
objective the 35M-parameter model, at this scale/epoch budget, ranks worse under than plain BCE
does.

**Follow-up (2026-10-19): a 5-epoch calibration run at `--soft-f1-loss-weight 0.02` confirms the
scale diagnosis and looks genuinely promising, not just "no longer broken."** `soft-f1-trial-002`
(otherwise identical config, 5 epochs only): `train_loss`/`train_soft_f1_loss` now diverge as
intended (epoch 1: `train_loss=0.0242` vs. `train_soft_f1_loss=0.0171`, implying a BCE component
of ~0.0071 — within the same order of magnitude as `val_loss=0.0068`, not 60× smaller like
`trial-001`), so the term is now a genuine nudge rather than dominating the gradient.

| | mean AUROC | pooled AUROC | best F1 | val_f1 @ epoch 5 |
|---|---|---|---|---|
| `trial-013` @ epoch 5 (pure BCE, 30-epoch run's own 5th epoch) | — (not computed mid-run) | — | — | 0.3177 |
| `soft-f1-trial-002` @ epoch 5 (BCE + 0.02×soft-F1, 5-epoch run) | 0.8361 | 0.9701 | 0.4061 | **0.3780** |
| `trial-013` final (pure BCE, converged @ 30 epochs, for reference) | 0.8731 | 0.9772 | 0.4186 | — |

At the *same* 5-epoch budget, the calibrated soft-F1 run's val_f1 (0.3780) clearly beats
`trial-013`'s own epoch-5 val_f1 (0.3177) — and its AUROC after only 5 epochs (mean 0.8361, pooled
0.9701) is already most of the way to `trial-013`'s fully-converged 30-epoch numbers (mean 0.8731,
pooled 0.9772). Both numbers are measured the same way at both epoch counts, so the comparison is
fair even though `compute_metrics`'s threshold-0.5 val_f1 is a known-noisy metric on its own (see
`compute_metrics`'s docstring/trial-050) — this looks like faster early convergence, not just "not
broken anymore."

**Follow-up (2026-10-20): the first full 30-epoch run (`soft-f1-trial-003`) used a mismatched
checkpoint-selection metric, invalidating its diagnostic — not a finding about the loss term
itself.** Ran `soft-f1-trial-003` (full `trial-013` config + `--soft-f1-loss-weight 0.02`, 30
epochs) without realizing `trial-013`'s own row in `experiments/full_label_sweep/sweep_log.md`
records it was selected by **`--checkpoint-selection f1`** ("best val_f1 0.3741@epoch23"), not
the script's `val_loss` default. `soft-f1-trial-003` was run with the default, so it picked epoch
4 (val_loss=0.0066) as "best" and ran `--run-diagnostic` against that checkpoint — while `val_f1`
kept climbing for the rest of training, peaking at **0.3973 @ epoch 29** (already beating
`trial-013`'s own 0.3741), with no catastrophic-overfitting signal (val_loss only drifts from
0.0066→0.0073 over 26 epochs, val_f1 keeps trending up almost the whole way):

| epoch | val_loss | val_f1 |
|---|---|---|
| 4 (picked by val_loss, used for the diagnostic below) | 0.0066 | 0.3739 |
| 23 (`trial-013`'s own selected epoch) | 0.0071 | 0.3929 |
| 29 (actual val_f1 peak) | 0.0071 | **0.3973** |

Diagnostic run against the wrong (epoch 4, badly under-trained) checkpoint: mean AUROC 0.8361,
median 0.8844 — worse than `trial-013`'s 0.8731/0.9259, but this is comparing a 4-epoch model to
a 23-epoch one, not a fair test of the loss term. The in-memory best-checkpoint tracking only
keeps the epoch selected by whatever `--checkpoint-selection` was actually passed, so epoch 29's
weights are gone — this can't be salvaged from the existing run; it has to be rerun with the
selection metric matched to the baseline.

**Next step: rerun with `--checkpoint-selection f1` added** (`soft-f1-trial-004`, same config
otherwise) to finally get the real apples-to-apples comparison against `trial-013`. Still open,
not yet closed out either way — the per-epoch `val_f1` trend above is a good sign, but the AUROC
diagnostic (the metric this item is actually judged on) hasn't been measured on a fairly-selected
checkpoint yet.

### 5. GO-DAG-consistency post-processing at inference
**Source:** report §"Stacked ensembles..." — multiple CAFA-5 top solutions apply a cheap final step
propagating/clamping predicted scores so parent terms never score below their children, even when the
underlying model has no hierarchy-aware training signal at all.
**Why Tier 2:** nearly free (no retraining required, a post-hoc numpy pass over predicted probabilities),
and current pipeline's flat multi-label classifier has no such step today. Worth adding regardless of
whether item 3's training-time version also gets built — they're complementary, not redundant.

**Result (2026-10-16): a free win on its own, and it's not clear item 3's training-time cost
is still earning its keep once this exists.** `clamp_hierarchy_consistency(probs, edge_index)`
(iteratively raises each parent's probability to at least its direct children's, converging
to a fixed point — same `experiments/scripts/rerun_diagnostic_full.py` backfill re-run, no
retraining) wired into `run_diagnostic` as unconditional `clamped_*` fields whenever
`hierarchy_edge_index` is available:

| | mean AUROC | pooled AUROC | best F1 | violation rate |
|---|---|---|---|---|
| `trial-013` (raw, no hierarchy awareness at all) | 0.8731 | 0.9773 | 0.4197 | 11.96% |
| `trial-013` + post-hoc clamping | 0.8733 | **0.9778** | **0.4217** | **0.00%** |
| `hierarchy-loss-trial-001` (raw, item 3's trained loss) | 0.8733 | 0.9772 | 0.4186 | 0.035% |
| `hierarchy-loss-trial-001` + post-hoc clamping | 0.8733 | 0.9772 | 0.4202 | **0.00%** |

**Clamping alone, on the untouched `trial-013` checkpoint, fixes the violation completely**
(11.96% → exactly 0%, a hard constraint rather than item 3's soft penalty) **at zero
retraining cost, and every quality metric moves up, not down**: mean AUROC +0.0002, pooled
AUROC +0.0005, best F1 +0.0020. No tradeoff to weigh — this is a straightforwardly free win
and should just be left on by default for any checkpoint with a graph path available.

**Clamping composes cleanly with item 3's training-time loss too**: applied on top of
`hierarchy-loss-trial-001`, it mops up the tiny residual violation rate (0.035% → 0%) the
training-time penalty didn't fully reach, with best F1 nudging up slightly (+0.0016).

**Honest complication, logged rather than smoothed over**: comparing the two *clamped* rows
against each other, `trial-013`+clamping ties-or-beats `hierarchy-loss-trial-001`+clamping on
every single metric — mean AUROC tied (0.8733 both), pooled AUROC 0.9778 vs. 0.9772, best F1
0.4217 vs. 0.4202. Once this free post-hoc fix exists, item 3's 30-epoch retrain with
`--hierarchy-loss-weight` doesn't appear to add anything measurable on top of it, and trails
slightly on two of three metrics. This doesn't retroactively make item 3 a mistake — it
genuinely fixed a real structural problem on its own terms, cheaply, before item 5 existed —
but it does mean **future work on this axis should default to post-hoc clamping alone**
rather than paying for a hierarchy-aware retrain, unless a reason surfaces to still want
the training-time signal (e.g. if it ever helps a model learn to use hierarchy-consistent
*reasoning* rather than just outputting hierarchy-consistent *scores* — not something this
comparison tests either way).

### 6. Stop scaling PLM backbone size further; redirect that compute
**Source:** report §"...table stakes, not the winning move" — GOCurator (CAFA-5, self-reported 1st
place) found "little difference" between ESM-1b/ESM-2/ProtT5 once embeddings are one ingredient among
several; a separate transfer-learning benchmark found ESM-2 650M "fell only slightly behind" the 15B
checkpoint on downstream tasks.
**Why Tier 2 (a negative/reallocation finding, not a build item):** this doesn't produce new code, it
changes where future compute should go. If `experiments/ancestor_sweep/` or any successor sweep is
considering scaling toward `esm2_t36_3B` again, this evidence says that compute is better spent on
Tier 1 items (ensembling, PK-conditioning, hierarchy consistency) than on a bigger backbone.

## Tier 3 — real potential, but needs a feasibility spike before it's plan-able

### 7. Temporal / later-GOA-release validation holdouts
**Source:** report + this session's own PU-learning postmortem
([[project_pu_learning_nnpu_km2]]). Multiple CAFA-5 top teams validated against *later* UniProtKB/GOA
releases (checking whether a protein gained annotations after the training snapshot) instead of plain
protein-ID k-fold.
**Why this matters beyond validation methodology:** this is the first genuinely independent evidence
source found (in this research or this session) that could re-open the two-tier PU-learning question
on solid footing — the standing objection to that whole line of work was that "excluded cell" checks
were self-training/circular (validated only against the same baseline model's own ranking). A later
GOA release would let you check whether a flagged "likely hidden positive" cell was actually confirmed
positive later — real ground truth, not the model checking itself.
**Why Tier 3, not Tier 1:** entirely contingent on an unresolved feasibility question — is a
newer UniProt-GOA release than this dataset's training snapshot actually fetchable, and does it cover
enough of the flagged cells to be statistically useful? That's a half-day investigation, not an
implementation task, and should happen before this gets its own plan. If it fails, PU-learning stays
shelved as currently documented; if it succeeds, this becomes the first Tier 1 candidate for
re-opening that line specifically (not the current per-term-calibration approach — see Deprioritized
below).

**Feasibility check DONE (2026-10-07), using the temporal PK/LK/NK harness's own infrastructure —
PU-learning stays PARKED, not reopened.** The feasibility gate is resolved decisively yes: 100%
(54,389/54,395, and separately 27,135/27,137 for the flagged-cell subset) of the two-tier mechanism's
train+validation proteins matched in the cached Swiss-Prot dump. Found and corrected a real
data-quality artifact along the way: 30.5% of the 43,250 "flagged likely-hidden-positive" cells were
already true in `train_terms.tsv`'s full propagated closure — `dataset_raw`'s narrow, unpropagated
668-term view had mislabeled them as unknown; excluded before the real comparison.

On the actual evidence question — does a later GOA release confirm the flagged cells are real, non-
circularly — the first naive pooled-vs-random-baseline check showed a 6.7x elevation, but that didn't
control for the fact that SCAR doesn't hold here (real curation isn't a uniform random sample of true
positives; some terms/proteins are just gaining annotations faster right now, independent of any
mechanism). Redone with two confound controls: **(a) per-term matched** (flagged vs not-flagged
within the same term, not pooled across terms) — pooled result 0.60% vs 0.19%, **3.23x**, holding
directionally for 9/11 terms; **(b) per-protein matched** (same protein's flagged term vs its own
other eligible-but-not-flagged terms) — 0.60% vs 0.18%, **3.30x**, ruling out "just hot proteins get
flagged more." Both independent controls land within rounding of each other — a real, non-circular,
bias-controlled signal that the mechanism finds genuine information, not just self-training noise.

**Decision (user-confirmed 2026-10-07): PU-learning stays PARKED despite this stronger confirmation.**
The controlled 3.2x signal validates that the mechanism *works* as designed, but doesn't move the
actual reason it was shelved (`project_pu_learning_nnpu_km2.md` v2.1) — it still only ever reaches the
same 11 terms that clear the ≥2000-source-positive TIcE calibration floor, and those 11 (nucleus,
cytoplasm, protein binding, membrane, etc.) are the lowest-IA-weight terms under CAFA's own Fmax
scoring. Not resuming now; may be worth reintroducing at a later spot in the training pipeline if it
measurably improves results once other items land, not as a renewed standalone push.

**More promising angle surfaced, not yet scoped:** the temporal t0/t1 infrastructure itself doesn't
need an estimator (TIcE/KM2/spy) at all — a confirmed t0→t1 annotation is a *directly observed*
hidden positive, not a statistically inferred one, removing the ≥2000-positives calibration floor
entirely. That could in principle reach the rare, high-IA terms CAFA actually rewards, at the cost of
only catching positives that happened to get annotated in the ~9-10 months between t0 and the cached
t1 snapshot. Noted here for later; not pursued this session — item 2 part (B) is next instead.

### 8. Label-embedding output setup: predict a functional description, embed it, score against GO-term embeddings
**Source:** your own inference from the deep-research report — "Not directly predicting GO-terms, but
the descriptive text... yields better results" — the ProTranslator/TALE label-embedding pattern
already cited in `experiments/graph_embedding_plan.md`'s "Learning points" section, but keyed off GO
terms' own textual definitions (`go-basic.obo` has these) rather than graph-structural (node2vec/GNN)
embeddings — complementary to that plan, not competing with it.
**Why this is now a prerequisite, not just another Tier 3 candidate (user correction, 2026-09-27):**
item 9 below (auxiliary sequence-based prediction tasks) needs a flexible, non-fixed-classifier-head
output mechanism to be worth building at all — under a flat per-term classification head, every new
auxiliary target needs its own dedicated trained output layer, which doesn't scale past a handful of
hand-picked tasks and cuts directly against the 2026-09-26 full-label-space scope constraint's
~30k+-term vocabulary. A label-embedding architecture (predict a description/representation, embed
it, score compatibility against precomputed target embeddings) sidesteps both problems: a new
auxiliary target, or a new GO term, just needs its own embedding, not a newly trained head.
**Item 9 cannot be properly scoped until this exists — build this one first.**
**Additional relevance:** may also answer the full-label-space vocabulary-scaling question left open
at the end of the Tier 0 discussion above — a compatibility-score head over GO-term text embeddings
doesn't need one classifier weight per term the way a flat sigmoid output layer does. Worth evaluating
as the actual output architecture for the Tier 0 full-label-space baseline itself, not only as an
auxiliary-task enabler — not decided yet, flag for the Tier 0 implementation plan.
**Why Tier 3, not Tier 1:** its own unresolved feasibility questions — what encodes the predicted
description and the GO term definitions into a shared space (reuse ESM's own representations vs. a
separate text encoder), whether GO term definitions are distinctive enough in embedding space to
separate closely related terms, and how compatibility-score training interacts with the existing
BCE-based training loop. Needs its own feasibility spike, same bar as item 7.

### 9. Auxiliary sequence-based prediction tasks (user's own idea: predict positional annotations, etc.)
**Depends on item 8 above (label-embedding setup)** — this item's auxiliary targets are meant to be
implemented as embedding predictions/compatibility scores under that architecture, not as new flat
classifier heads; don't scope this before item 8 has at least a working feasibility result.
**Source:** not directly evidenced in the research (no source reviewed combined multi-task sequence
objectives with GO prediction), but consistent with two things the research does support: (a) the
general principle that orthogonal evidence sources help (item 1), and (b) the finding that structure
specifically delivers only modest gains (item 11 below) — so a sequence-based auxiliary task avoids
paying that cost while still targeting "broader representation learning" the same way structure
was meant to.
**Why Tier 3:** untested idea, no direct precedent found. Per
[[feedback_check_metric_can_see_the_effect]]: before running this, decide up front what metric would
actually show the auxiliary task helped GO prediction specifically (e.g. does adding it measurably
change per-term AUROC on terms it's plausibly related to?), not just whether the auxiliary task itself
trains — that discipline was the exact gap that caused the two-tier PU trial to initially look like a
null result.

### 10. Reopening the shelved two-tier PU-learning mechanism, conditional on a materially stronger base predictor
**Source:** [[project_pu_learning_nnpu_km2]] (this session) + report §"The open-world annotation
problem is acknowledged everywhere but corrected almost nowhere." The shelving decision
(2026-09-24) was driven by two compounding limits: only 11/437 terms ever calibrate usefully with
current estimators (a hard data-availability wall, not a bug), and those 11 ubiquitous terms carry
the *lowest* per-term weight under CAFA's IA-weighted Fmax — so even a working mechanism was a narrow,
low-value win. **User-specified reopening condition (2026-09-25): not permanently closed — if Tier
1/2 work (ensembling, PK-conditioning, hierarchy consistency) produces a materially stronger base
classifier, the two-tier mechanism's excluded-cell effect (already verified real and
validation-generalizing for those 11 terms, see `check_two_tier_excluded_cell_effect.py`'s results)
may become large enough on those terms to be worth harvesting, even though the terms themselves stay
low-value individually.** This mirrors your own reading of the SOTA research: PU-style correction
appears to need a strong base predictor before it pays off, not just a well-calibrated prior.
**Why Tier 3, not Deprioritized:** distinct from item 7 (temporal validation) as a reopening trigger —
this one doesn't need new external data, just re-running the existing, already-built
`check_two_tier_excluded_cell_effect.py` check against whatever improved checkpoint Tier 1/2 work
eventually produces, to see whether the effect size grew. Cheap to recheck opportunistically once a
better baseline exists; not worth deliberately chasing on its own before then. The 11-term/low-IA-weight
scope limit doesn't change — only the mechanism's effect *size* on those terms is in question.

### 13. Lab/database annotator-identity bias — predictability check, then temporal-bias-conditioned PU-learning
**Source:** user's own idea (2026-10-22) — the hypothesis that annotating labs/databases have
persistent preferences (which GO terms/aspects they tend to annotate), driven by research-direction
continuity within a lab across successive generations of researchers, not purely by the biology of
the proteins themselves. If provable, this is evidence that part of what CAFA scores is which labs
happen to be active/publishing in the eval window, not pure protein biology — valuable to know, and
(the user's own words) "very, very valuable and possibly painful" if true.

**Mechanism, in three gated steps:**
1. **Attribution dataset (two chained one-time batch pulls).** GAF format (GO Consortium / EBI
   GOA) carries `Date` (column 14, `YYYYMMDD`), `Assigned_By` (column 15, e.g. `SGD`, `UniProtKB`,
   `FB`), and `DB:Reference` (column 6, a `PMID:...` citation) on every annotation row.
   `goa_uniprot_all_noiea.gaf.gz` (excludes algorithmic/IEA annotations, which carry no
   human-curation bias to find) filtered to CAFA6's protein ID set is pull #1 — a single
   one-shot fetch, no incremental/repeated fetching.
   **Granularity, corrected (2026-10-22, user call): `Assigned_By` alone is too coarse to use as
   the lab key.** It has only ~30-40 distinct values total (per the GO Consortium's own
   contributing-groups list); ~15-20 are genuinely organism-scoped curation teams, but a few
   mega-buckets — chiefly `UniProtKB`'s own internal curation team — span every organism and
   thousands of unrelated papers/labs, i.e. exactly the "almost everything could go there" problem.
   **Real lab identity needs one level deeper:** pull #2 is a single batched NCBI PubMed E-utils
   query over the unique PMIDs collected from `DB:Reference` in pull #1, resolving each paper's
   last/corresponding author — by biology convention, the PI, used as the stable "lab" identity
   key (no need to model the mentor/mentee graph itself; the PI persists while the people under
   them rotate, which already captures the continuity the user expects). **Store both values per
   annotation row** — the raw `Assigned_By` group/database *and* the PubMed-resolved last-author
   "lab" key — so either granularity is available later, including building a **mentor-mentee DAG** on top of the
   stored last-author identities (e.g. a last-author who appears as first/non-last-author under a
   different last-author earlier in their publication history, inferred from the same PubMed
   metadata) as an additional measurement, without needing a second data pull to add it.
2. **Cheap predictability check (go/no-go gate).** Train a cheap classifier (XGBoost) to predict
   which `Assigned_By` lab/database produced a given (protein, term) annotation. If lab identity
   isn't learnably distinct from chance, the premise is wrong and this stops here — cheap to find
   out before committing to step 3.
3. **Temporal-bias-conditioned PU-learning (conditional on step 2 succeeding).** Determine which
   labs/databases are likely to be active annotators within a CAFA-style future delta-t window, take
   their most likely top-N predicted terms, and bias PU-learning toward those protein×term pairs as
   stronger positive candidates — test whether this measurably changes future-prediction accuracy on
   the existing temporal t0/t1 harness (`experiments/pk_eval/`, built for item 2/item 7) rather than
   building new infrastructure.

**Why Tier 3, not Tier 1/2:** entirely gated on step 2's own go/no-go result, same pattern as items 7
and 10 — don't build step 3 before step 2 confirms there's a real, learnable lab-identity signal to
condition on.
**Relationship to other items:** reuses item 2/item 7's cached Swiss-Prot/temporal infrastructure
rather than duplicating it; if it works, it's a second, independent line of evidence (alongside item
7's "later annotation confirms flagged cells") that CAFA's future ground truth carries real
sociological signal, not just biology.

## Deprioritized — evidence says skip, or an existing project gate says not yet

### 11. Structure-based (AlphaFold-predicted structure) features
**Source:** report §"Structure and ontology awareness..." — an independent 14-method benchmark found
pure-sequence, ontology-aware DeepGOZero+ beat the structure-integrating DeepGraphGO (0.463 vs. 0.427
Fmax on biological process); DeepFRI's and GAT-GO's structural gains are characterized as only
"modest"; even PANDA-3D's own feature-importance analysis ranks sequence-derived ESM features above
its structural ones. This validates the user's own instinct (already planning sequence-based, not
structural, auxiliary tasks) rather than contradicting it.
**Status:** not worth the AlphaFold-structure dependency and GCN/GVP training cost given the evidence;
revisit only if a specific benchmark surfaces showing structure helps *on top of* a well-built
sequence+ontology pipeline specifically (the comparisons found so far are against weaker baselines).

### 12. Full GO graph-embedding/GNN approach (`experiments/graph_embedding_plan.md`)
**Status: stays gated**, per that document's own explicit condition — "the Open World Assumption (OWA)
problem must be tackled to at least some extent first" — and the PU-learning line that was the
project's attempt at that prerequisite was just shelved
([[project_pu_learning_nnpu_km2]]) without a working fix. Don't start this because the SOTA research
mentions ontology-aware methods getting good results elsewhere — those results (DeepGOZero, TALE) use
the lighter consistency-constraint mechanism in Tier 1 item 3, not full statistical-strength-sharing
across a noisy label graph, which is the specific mechanism this project's own analysis flagged as
noise-inheriting. Re-open only if item 7 (temporal validation) succeeds and gives a trustworthy way to
tell a shared-embedding model's borrowed signal apart from borrowed noise.

Reopening per-term PU-learning class-prior *calibration* specifically (as opposed to rechecking the
already-built two-tier mechanism's effect size, which is Tier 3 item 10 above) stays out of scope: the
report independently confirms classical PU estimators (Elkan-Noto, KM2, TIcE) assume SCAR, which GO's
"rich-get-richer" annotation bias (Gini coefficient 0.25→0.47, 2001-2017) demonstrably violates, and
that the wall this project hit (needing ≥2000 positives per term to calibrate) is a real field-wide
limit, not a project-specific gap — the one modern PU-learning paper for GO prediction found (PU-GO,
2024) is a near-isolated attempt at this intersection, not a mature technique family with a fix
available. Don't invest in a new calibration estimator without either a hierarchy-borrowing approach
(untested, see the shelving memory) or item 7's temporal-validation approach succeeding first.

## Cross-cutting process rules that apply to whichever item gets picked up next

- **Plateau-first sweep strategy** ([[feedback_plateau_first_sweep_strategy]]): find the cheap-scale
  plateau before scaling any new technique up — don't guess at regularization/hyperparameters for a
  new loss (item 4) or hierarchy constraint (item 3) before locating where it stabilizes at small scale.
- **Check the metric can see the effect** ([[feedback_check_metric_can_see_the_effect]]): before
  reporting a result for any item above, confirm the evaluation actually measures what that item
  targets — this was the exact process gap the two-tier PU trial exposed.
- **No long CPU runs, prefer GPU** ([[feedback_no_long_cpu_runs_prefer_gpu]]): applies to any
  prototyping for the above (embedding extraction, BLAST-KNN index building, etc.) — hand off
  GPU-appropriate or >10-minute CPU jobs via a `!`-prefixed command rather than running them in-sandbox.
