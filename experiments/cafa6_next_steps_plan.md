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

**Result (2026-10-23): `soft-f1-trial-004` (`--checkpoint-selection f1`, matching `trial-013`
exactly) gives the real apples-to-apples comparison — net negative, closing out this item.**
Best epoch 29 (val_f1=0.3975), diagnostic run against that checkpoint:

| | mean AUROC | median AUROC | pooled AUROC | best F1 | rare AUROC | common AUROC | violation rate |
|---|---|---|---|---|---|---|---|
| `trial-013` (pure BCE) | **0.8731** | **0.9259** | **0.9773** | 0.4197 | **0.8812** | **0.8500** | **11.96%** |
| `soft-f1-trial-004` (BCE + 0.02×soft-F1) | 0.8493 | 0.9064 | 0.9647 | **0.4204** | 0.8569 | 0.8268 | 27.55% |

Every AUROC metric (the ranking-quality numbers this whole item is judged on, per the source
report's own framing) is worse — mean -0.0238, median -0.0195, pooled -0.0126, rare -0.0243,
common -0.0232 — and the hierarchy-violation rate more than doubles (11.96%→27.55%, though
free post-hoc clamping from item 5 fixes both checkpoints to 0% anyway). The only things that
improved are threshold-dependent: best F1 (+0.0007, within noise) and val_f1 at the selected
epoch (0.3975 vs 0.3741, +0.0234). The earlier 5-epoch "promising" signal (`soft-f1-trial-002`)
was real but misleading — it only measured early-training speed on a threshold metric, not where
either run actually converges, and the weight=1.0/0.02 exploration never found a setting that
beats plain BCE on the metric that matters.

**Closing this item: soft-F1-weighted loss, at least in this additive/protein-level-only form,
does not improve on plain BCE for this task** — it trades ranking quality for a marginal,
noise-level gain in thresholded F1. Not recommending further weight tuning on this lever; `item
3` (hierarchy loss) and `item 5` (post-hoc clamping) remain the validated Tier 1/2 wins to build
on instead.

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

**Feasibility spike (2026-10-23): question 1 resolved easily, question 2 fails decisively —
not pursuing this item as scoped.** `experiments/label_embedding/feasibility_probe.py`.

**Q1 (encoder choice): resolved.** ESM itself is a non-starter — its tokenizer only knows amino
acids, not English text. A small off-the-shelf sentence-embedding checkpoint
(`sentence-transformers/all-MiniLM-L6-v2`) loads and runs fine through the `transformers`
dependency already in the stack (`AutoModel`/`AutoTokenizer` + manual mean-pooling) — no new
package needed. All 40,122/40,122 `go-basic.obo` terms have a usable `def:` field, so coverage
isn't a concern either.

**Q2 (distinctiveness): fails, for a large and important sub-vocabulary.** Embedded every term
definition with the encoder above and compared cosine similarity across three pair types (500
random pairs, 500 sibling pairs sharing a direct `is_a` parent, and every
`positive_regulation_of_X` / `negative_regulation_of_X` antonym pair found — 2,612 of them):

| pair type | n | mean cosine | median | min | max |
|---|---|---|---|---|---|
| random (unrelated) | 500 | 0.2342 | 0.2207 | -0.1210 | 0.6824 |
| sibling (shares a parent) | 500 | 0.7374 | 0.7738 | 0.1900 | 0.9964 |
| **antonym (pos_reg vs. neg_reg)** | **2,612** | **0.9154** | **0.9138** | 0.4820 | 0.9980 |

A generic sentence encoder rates `positive regulation of X` and `negative regulation of X` as
*more similar to each other* (mean 0.9154) than ordinary siblings are to each other (0.7374) —
these are opposite biological effects the model absolutely needs to tell apart, and the single
most common regulation-direction naming pattern in GO (2,612 such pairs found from this one
pattern alone, pervasive across `biological_process`) collapses to near-identical embeddings.
Concrete examples: `positive regulation of mitochondrial RNA catabolic process` vs. its negative
counterpart scores 0.9476; `positive/negative regulation of type I hypersensitivity` scores
0.8972. This isn't a rare edge case to work around later — it's a systemic blind spot in exactly
the kind of off-the-shelf encoder this architecture needs, baked into the encoder's generic
semantic-similarity training objective (which reasonably treats "X" and "regulation of X" as the
dominant shared signal, direction as a minor modifier).

**Q3 (training-loop integration): moot given Q2, but architecturally sound if revisited.** A
compatibility score (`scale * cosine_sim(protein_proj, term_embedding) + bias`) is a drop-in
replacement for the classifier's logit computation only — `BCEWithLogitsLoss`, the hierarchy-loss
hinge (item 3), and the existing loss-masking mechanism all operate on logits and don't care how
they were produced. Not a blocker, didn't need a code spike to confirm.

**Salvage follow-up (2026-10-23): Q2's failure is fixable, cheaply, for the pattern that
dominates it — reopening item 8.** `experiments/label_embedding/salvage_probe.py`. Tested three
routes to recover antonym-pair distinctiveness:

| fix | random | sibling | antonym |
|---|---|---|---|
| (baseline: definition only) | 0.2342 | 0.7374 | 0.9154 |
| name + definition concatenated | 0.1975 | 0.7278 | 0.9082 |
| name only (no definition) | 0.1553 | 0.6841 | 0.9614 |
| **definition + 1-dim regex direction flag** | 0.1981 | 0.5345 | **-0.6169** |

Text reformatting (rows 2-3) — i.e. "just adjust the descriptions" or "remove the noise" —
**doesn't work**: prepending the name barely moves antonym similarity (0.9154 → 0.9082), and
dropping the definition entirely to keep only the name makes it *worse* (0.9614). The encoder
isn't missing the direction word for lack of visibility; it structurally discounts "positive" /
"negative" as a minor modifier against the shared topical content no matter how the text is
arranged. This refutes the text-engineering hypothesis specifically.

What does work: augmenting the frozen definition embedding with one extra deterministic
dimension — a regex over the term name (`^(positive|negative) regulation of `) emitting
+1/-1/0, scaled and concatenated before re-normalizing. This flips mean antonym similarity to
**-0.6169** (fully separated) while leaving random pairs basically unchanged (0.1981) and
siblings still clearly more similar than random (0.5345 — lower than the 0.7374 baseline,
plausibly because some siblings genuinely are direction-opposite pairs that arguably *should*
end up pushed apart). No training involved — this is a hand-written structural feature, not a
trained fix for route (c) ("train an embedding setup specifically for those"), but it's
suggestive evidence in that direction: once the direction signal is captured at all, the rest of
the embedding geometry is healthy enough to cleanly separate antonym from random/sibling pairs.

Checked how far this generalizes before trusting it as the fix: counted how many GO term names
match other explicit opposite-effect naming conventions besides `positive/negative regulation
of`.

| naming pattern | term count |
|---|---|
| positive/negative regulation of | 5,660 |
| activation of / inhibition of | 34 |
| agonist / antagonist activity | 10 |
| stimulation of / suppression of | 4 |
| up-regulation / down-regulation | 0 |
| increased / decreased | 0 |
| (total terms in `go-basic.obo`) | 40,122 |

**Correction (caught during step 2 below): that table counted one-sided name hits, not
confirmed pairs.** Checked directly whether the three minor patterns ever have *both* sides
present (shared suffix on the positive and negative side) anywhere in `go-basic.obo`: zero
matched pairs for all three (`activation of`: 25 terms vs. `inhibition of`: 9, 0 shared
suffixes; `agonist activity`: 1 vs. `antagonist activity`: 9, 0 shared; `stimulation of`: 0 vs.
`suppression of`: 4, 0 shared). These terms exist, but never as an actual antonym pair that
needs separating — `positive/negative regulation of` isn't just the dominant antonym-collapse
pattern in GO, it's effectively the **only** one expressed as a shared-suffix naming
convention. What it does **not** cover: antonym-like pairs that are semantically opposite
without any shared marker word at all (not counted/quantified here) — those would still need
either more rules or a genuinely trained contrastive encoder (route (c), not attempted).

**Revised verdict: reopening item 8, scoped around the structural-flag fix, not a trained
encoder.** The original "not pursuing" call was based on an incomplete picture — it tested only
the frozen encoder's raw definition embedding and didn't check whether a cheap augmentation could
fix the dominant failure mode. It can. Scoping note for whoever picks this up: ship the compatibility-score
architecture with definition embeddings augmented by a small, explicit set of direction-modifier
regex flags (regulation of as the dominant one, the four smaller patterns above as cheap
additions); treat "semantically opposite but no shared marker word" as a known residual gap, not
yet measured, and re-evaluate Q2's distinctiveness check after the flags are added rather than
assuming the earlier failure numbers still apply.

**Implementation progress (2026-10-23):**

**Step 1 (precompute, done).** `experiments/label_embedding/build_term_embeddings.py`
embeds all 27,191 *trainable* terms' definitions (not just the full 40,122-term ontology),
in the dataset's own `label2id` order, with the direction flag generalized to all 4 known
patterns. Direction flag fires on 3,692/27,191 terms (~13.6%, consistent with the earlier
~14% full-ontology estimate). All 27,191 trainable terms had a usable `def:` field — no
missing-def fallback needed. Output cached to `term_embeddings.pt` (gitignored, ~40MB,
regeneratable).

**Step 2 (revisit Q2 at full trainable-vocabulary scale, done — gate passes).**
`experiments/label_embedding/validate_term_embeddings.py` reran the distinctiveness check
against the actual cached, flag-augmented 27,191-term embeddings, not the 2,612-pair sample
the salvage probe used:

| pair type | n | mean cosine |
|---|---|---|
| random | 500 | 0.1994 |
| sibling | 500 | 0.4887 |
| **antonym (positive/negative regulation of)** | **1,411** | **-0.6175** |
| activation of / inhibition of | 0 | — (no matched pairs, see correction above) |
| agonist / antagonist activity | 0 | — (no matched pairs) |
| stimulation of / suppression of | 0 | — (no matched pairs) |

Matches the salvage probe's numbers closely (-0.6175 vs. -0.6169) at the real scale this will
train against — antonym pairs fully separated, siblings (0.4887) still clearly more similar
than random (0.1994). **Gate passes — proceeding to step 3 (model integration).**

**Step 3 (model integration, done).** `training_run.py`: `LabelEmbeddingClassificationHead` +
`LabelEmbeddingEsmForSequenceClassification`, wired in behind `--use-label-embedding-head`
(mutually exclusive with `--use-lora`/`--use-aspect-input`/`--use-known-input`, same pattern as
those existing head-swap flags) + `--label-embeddings-path`. `create_esm()` validates the
cached tensor's term order matches the dataset's own `label2id` before building the model, and
the per-term prior-bias-init trick (which needs one bias per label) is skipped for this head
with an explanatory log line, since the head has one global scale/bias instead.

**Bug found and fixed during the smoke test, worth flagging for future custom heads:**
`from_pretrained`'s fast meta-device init path only properly initializes module types
`_init_weights` recognizes by `isinstance` (`nn.Linear`, etc) — confirmed `classifier.proj`'s
weight/bias came out correctly initialized this way. But bare `nn.Parameter` scalars not
wrapped in a recognized module type (`classifier.scale`, `classifier.bias`) silently come back
as uninitialized meta-device garbage (~1e-29) instead of the 10.0/0.0 the head's `__init__`
set — and the non-persistent `term_embeddings` buffer is hit even harder, coming back as all
zeros, since it's allocated while `from_pretrained`'s meta-device context is active regardless
of the real tensor passed to `__init__`. First smoke-test run silently trained with scale/bias
≈1e-29 and all-zero term embeddings — loss exploded to ~511,000 (BCE loss on effectively-zero
logits that should have stayed small) instead of crashing, which is the dangerous failure mode
here: it looks like a diverging run, not a wiring bug, unless the loss magnitude is checked
against what a sane logit range should produce. Fixed in `create_esm()` by explicitly
re-filling `scale`/`bias` (fresh start only — a `--resume-from-revision` load has real trained
values in the checkpoint) and re-copying `term_embeddings` from the passed-in tensor
(unconditionally, since it's never in any checkpoint's state dict) right after
`from_pretrained()` returns. Re-verified after the fix: logits land in a sane range (-2.19 to
2.07) and a 20-example/1-epoch CPU smoke test's train_loss drops cleanly step over step (0.59 →
0.24).

**Step 4 (cheap-scale GPU comparison vs. the classifier baseline, done — negative result,
closing out item 8).** 2,000 train / 500 val examples, 5 epochs, `esm2_t12_35M_UR50D`,
identical hyperparameters to the classifier baseline (`unfreeze_last_n=4`, dropout 0.1/0.1,
weight_decay 0.01, batch_size 2/grad_accum 8, `--checkpoint-selection f1`, matching the
baseline's own convention per [[feedback_match_checkpoint_selection_to_baseline]]).

First run (global scalar `bias`): classifier baseline reached `mean_auroc=0.6577`; the
label-embedding head collapsed to `mean_auroc=0.4999` — exact chance — with `val_recall`/`val_f1`
pinned at 0.0 from epoch 2 on despite `train_loss` decreasing smoothly (0.18 → 0.032). Diagnosed
as the global `scale` collapsing toward zero: with no per-term calibration slot, the cheapest way
to fit this task's ~99.8%-negative imbalance is to flatten every term's cosine-similarity signal
at once, destroying ranking everywhere simultaneously.

Fix applied and committed (`afcdf62`): changed `bias` from a scalar to a per-term `[num_labels]`
vector (still `O(num_labels)` parameters, nowhere near the classifier's `O(hidden_size ×
num_labels)` weight matrix), and unified the prior-bias-init logic so both head types get their
bias initialized to each label's own log-odds. A CPU smoke test (20/10 examples, 1 epoch) looked
healthy: loss immediately sane (0.0073 → 0.0028), `val_f1=0.1937` vs. 0.0 pre-fix.

Rerun at the same cheap GPU scale with the fix: `mean_auroc=0.5028` — **unchanged from chance**,
and no better than the pre-fix 0.4999. `val_recall` stayed pinned at ~0.0003-0.0004 across all 5
epochs; `train_loss` barely moved (0.0070 → 0.0066). The per-term bias fix was a real, correct
improvement in isolation (loss is sane instead of exploding, and it closes a genuine global/
per-term capacity gap), but it did not touch the actual root cause — the two cheap-scale runs failed
for different proximate reasons, and neither produced any per-label ranking signal.

**Revised root-cause read, corrected again (user's catch, 2026-10-24): a single-point multi-label
capacity cap, not just a shared-map bottleneck.** `proj(pooled)` produces exactly **one** unit
vector per protein; every one of the 27,191 per-term logits is that single point's cosine
similarity to a fixed term embedding. For multi-label prediction to work, a protein's *entire*
true-label set must be simultaneously close to that one point — unlike the flat classifier, where
each label reads off the full hidden representation through its own independent weight vector, so
labels never compete for a shared direction.

Checked directly whether a protein's true GO terms actually cluster tightly enough in this
embedding space for that to be geometrically plausible (500 sampled training proteins, all
pairwise cosine similarities among each protein's own true labels):

| pair type | n | mean cosine |
|---|---|---|
| random term pairs (Step 2 baseline) | 500 | 0.1994 |
| **a protein's own true-label pairs** | 1,017,024 | **0.3309** |
| sibling term pairs (Step 2 baseline) | 500 | 0.4887 |

Proteins average **43.7 true labels** (median 31, min 3, max 451) — and those true labels sit
*between* random and sibling similarity, only mildly more clustered than two unrelated terms
picked at random. Asking one unit vector to score ~30-40 only-loosely-clustered targets as "high"
while scoring the other ~27,150 as "low" is a real geometric bottleneck independent of training —
it matches the observed flat chance-level AUROC and near-zero training signal far better than the
bias-capacity story. (The shared-projection-map framing in the previous paragraph is still true as
a secondary factor — the map itself is also less expressive than the classifier's per-label
weights — but the single-point multi-label constraint is the more fundamental limit and the one
that actually explains why training produced no signal at all, not just a weaker one.)

**Verdict: closing out item 8 again, for a different reason than the first time.** The salvage
fix (direction-flag augmentation) was real and did what it claimed — it fixed the antonym-collapse
failure in the *text*-embedding geometry (Step 2's gate passed cleanly). But the architecture built
on top of that geometry has a capacity limit the Q2 checks never tested for: a single-vector
compatibility score structurally can't represent an arbitrary ~30-40-term multi-label target set
per protein when those true labels are only loosely clustered (0.33 mean cosine — barely above the
0.1994 random baseline, nowhere near the 0.4887 sibling baseline). That's a limit on this specific
single-point-per-example architecture, not on label-text embeddings in general — a per-aspect head
(one point per GO aspect, where true labels might cluster tighter) or a multi-vector/set-based
scoring scheme could in principle route around it, but either is new architecture, not a tweak.
Escalating further (bigger embed dim, more data/epochs, or chasing another bias-term variant)
would be guessing at symptoms of a problem that's now understood at the structural level — not
pursued further here.

### 9. Auxiliary sequence-based prediction tasks (user's own idea: predict positional annotations, etc.)
**Depends on item 8 above (label-embedding setup)** — this item's auxiliary targets are meant to be
implemented as embedding predictions/compatibility scores under that architecture, not as new flat
classifier heads; don't scope this before item 8 has at least a working feasibility result.
**Status update (2026-10-24):** item 8 is closed out again — the structural-flag fix resolved the
text-embedding geometry (Step 2's gate passed) but two independent cheap-scale GPU trials showed
the resulting architecture still can't learn real per-label ranking (`mean_auroc` stuck at chance,
~0.50, both before and after fixing the global-bias capacity gap). This item stays blocked; it has
no viable embedding-compatibility architecture to build auxiliary targets on top of.
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

### 14. Attention-map validity check on positioned MF spans, then targeted counterfactual "broken-span" negatives
**Source:** user's own idea (2026-10-23), reusing ground-truth assets built for the shelved
mf_span sub-project ([[project_mf_span_shelved]]) for a different purpose. **This does not reopen
that shelved line:** it never trains or scores the dedicated span-prediction model from
`experiments/mf_span/` or relies on its discredited token-level metric (see
`experiments/mf_span/mf_span_postmortem.md` — Fmax 0.18 vs. reported F1 0.82). Only the
*ground-truth* residue-span data that project produced survives into this new use, applied to the
**main** pipeline's classifier (trial-013 lineage) instead.

**Ground truth already in hand:** `experiments/mf_span/positional_mf_links.json` maps each of
82,396 proteins to `{go_id, start, end, source, ...}` localized MF-term spans. **Caveat found
while checking this (2026-10-23):** many entries have `source="domain_whole_protein_fallback"`
with `start=0, end=sequence_length` — i.e. no real localization, just the whole sequence as a
fallback. Only the tighter, genuinely-localized entries (e.g. a `binding_ligand` source with a
narrow span, sometimes a single catalytic/binding residue) are usable here — the fallback rows
must be filtered out, not treated as ground truth spans.

**Step 1 — attention-map validity check (interpretability probe, no training).** For proteins
where the main classifier correctly predicts an MF term with a known, tightly-localized span,
extract the model's attention weights (`output_attentions=True`, standard HF — not yet verified
directly against this repo's `AutoModelForSequenceClassification` wrapping, the first concrete
thing to confirm) and check whether attention mass concentrates on the ground-truth span more
than chance (e.g. mean attention-to-span vs. attention-to-rest-of-sequence, aggregated over
heads/layers). Cheap, no GPU training — answers a yes/no question: does the model's internal
attention already track the biologically relevant residues, or not.

**Step 2 — conditional on step 1 finding a real gap.** If attention is clearly not concentrating
on the relevant span for terms the model is supposedly basing its prediction on, build a targeted
counterfactual augmentation: take a protein with a known tightly-localized span for term X, mutate
the span's residues to ones expected to disrupt that specific function (the same kind of
catalytic/binding-residue knockout real IDA/mutagenesis evidence is built from), and add the
mutated sequence as a new training example with:
- term X labeled a confirmed **negative** (the one deliberately, mechanistically broken
  function) — a genuinely true negative, valuable because explicit correct negatives are
  otherwise essentially absent under this project's OWA/PU-learning problem
  ([[project_pu_learning_nnpu_km2]]);
- every other term **masked out of the loss** (via the existing `--unknown-target-value`/
  `loss_mask` mechanism already in `training_run.py`), since breaking one local motif says
  nothing trustworthy about the protein's other annotations (global stability, other binding
  sites, etc.) — assuming they're unaffected would be its own unverified guess.

**Why Tier 3:** step 1 is a cheap, immediate diagnostic; step 2 is explicitly conditional on step
1 finding a real problem, not built speculatively.

### 15. Mining UniProt's own curated evidence for more trustworthy known-positive/known-negative examples
**Source:** user's own idea (2026-10-23), generalizing item 14's single mechanism (synthesize a
disruptive mutation yourself) into systematically mining evidence UniProt curators have *already*
published, ranked by trust, cheapest/most-direct first:
1. **`FT MUTAGEN` features (primary).** Already-published, experimentally-measured site-directed
   mutagenesis results, with a free-text `/note` describing the effect (e.g. "Loss of kinase
   activity") and usually an experimental evidence code + PubMed reference — a real negative, not
   a guessed one. Parseable today from the already-cached `data/uniprot/uniprot_sprot.dat.gz` via
   `Bio.SwissProt.parse()`'s `record.features`.
2. **Natural inactive paralogs/pseudoenzymes (`CC CAUTION` comments, secondary).** Free natural
   knockouts UniProt already documents in prose (e.g. "lacks the conserved active-site residue...
   probably non-catalytic") — no mutation construction needed, but harder to mine systematically
   since the evidence is unstructured free text, not a feature with coordinates.
3. **Disease-variant loss-of-function alleles (`FT VARIANT`, tertiary, noisiest).** Usable only
   when the variant carries its own experimental-evidence code for a specific molecular
   consequence, not just a clinical/phenotype association — most disease variants don't clear
   that bar.
Same text-pattern/evidence-mapping work `build_positional_mf_links.py` already did for the mf_span
project applies here — mapping a `/note` description to a specific GO term — not a new
data-acquisition problem.

**Deliverable, once mining is done (not yet — logged for later pickup):** a summary of how many
new known-positive and known-negative `(protein, term)` pairs this recovers, broken out by
evidence category and by whether the protein is already in the CAFA6 train/val/test split or new
(some `MUTAGEN`/`CAUTION`-documented proteins may not currently be in `train_terms.tsv` at all —
Swiss-Prot already has their sequences, so adding them is just a `generate_dataset_full.py`-style
inclusion, not a new fetch).

**Why this matters beyond item 14:** that summary is what decides whether there's enough new, real
evidence to justify a broader "mask genuinely-unknown terms instead of treating every unannotated
term as an implicit negative" training strategy — directly relevant to this project's long-running
OWA/PU-learning tension ([[project_pu_learning_nnpu_km2]]). Unconditional on item 14's step 1
result (unlike item 14's step 2) — this is useful regardless of what the attention-map check finds.

### 16. Autoregressive decoder over item 8's term-embedding vocabulary (user's own idea, 2026-10-24)
**Source:** direct follow-up to item 8's closure. Item 8's single-point compatibility-score head
(`proj(pooled)` → one unit vector per protein, scored against every label's frozen term embedding
at once) turned out to have a structural multi-label capacity cap: a protein's true labels (mean
43.7/protein) sit at only 0.33 mean mutual cosine in that embedding space — barely above the 0.1994
random-pair baseline, nowhere near the 0.4887 sibling-pair baseline — too loosely clustered for one
static point to score ~30-40 of them "high" simultaneously while scoring the other ~27,150 "low".
User's proposed fix: emit labels one at a time instead of all at once, so each decode step only has
to match ONE target. This is a known, named approach in the literature (sequence-generation models
for multi-label classification, e.g. Yang et al. 2018's SGM), not a novel mechanism being invented
here — item 9's own auxiliary targets (variable-cardinality positional annotations) need the same
"emit one result, update state, decide whether to continue" loop regardless of whether what's being
emitted is a GO term or a sequence span, which is part of why this is being built now rather than
treating item 9 as separately blocked.

**Why not the cheaper sibling alternative (fixed-K parallel query slots, DETR-style set
prediction):** considered and rejected per the user's own reasoning — item 9's longer prediction
chains need variable, open-ended cardinality (same long tail item 8 already measured: mean 43.7
labels/protein, median 31, **max 451**), and a fixed K would cap representable cardinality by
construction the same way item 8's single point did, just at a less severe choke point. An
autoregressive decoder's output length isn't fixed in advance, so it doesn't reintroduce a capacity
ceiling of the same kind this item exists to fix.

**Design (implemented 2026-10-24, training_run.py + experiments/label_embedding/
train_autoregressive_head.py):**
- `AutoregressiveLabelEmbeddingHead`/`AutoregressiveLabelEmbeddingEsmForSequenceClassification` in
  `training_run.py`, mirroring the existing head-swap class pattern (`AspectConditioned...`,
  `LabelEmbedding...`). A GRUCell decoder: `init_state = Linear(pooled)` seeds the first step;
  each step embeds the previously-emitted label (a `label_embed` table with one extra row for a
  START token, kept separate from the frozen scoring vocabulary so START can never be predicted as
  an output label), updates the GRU state, projects into the same frozen `term_embeddings` space
  item 8 built and validated (`experiments/label_embedding/term_embeddings.pt`, reused byte-for-byte,
  no changes needed), and scores via cosine similarity exactly like item 8's head
  (`scale * sims + per_term_bias`). A separate linear `stop_head` off the same decoder state
  predicts, per step, whether this was the last label (plain BCE target, kept separate from the
  27,191-way softmax rather than diluting it with an extra "STOP" vocabulary entry).
- **Label ordering for teacher forcing:** `compute_label_depth()` (new, next to
  `build_hierarchy_edge_index`) layers the same direct is_a/part_of edges item 3's hierarchy-loss
  work already extracts from `go_dependency_graph.json`, via Kahn's algorithm, taking the MAX depth
  over all in-vocabulary parents — this guarantees every one of a label's in-vocabulary ancestors
  has strictly smaller depth, so sorting a protein's true labels ascending by this depth always
  emits ancestors before descendants. Reuses existing infrastructure rather than inventing a new
  ordering scheme, and happens to bake in the same ancestor-before-descendant property item 3's
  consistency loss targets separately.
- **Max-steps cap (exposure to the long tail):** `--ar-max-decode-steps` (default 64) truncates each
  protein's depth-ordered label sequence, both for training targets and for `generate()`'s decode
  budget. Truncation drops the *deepest/most specific* labels first (ascending-depth order), not a
  random subset — keeping the general terms a protein almost certainly has representable, at the
  cost of the architecture being unable to ever name a protein's longest-tail, most-specific
  annotations at all under this cap. Not yet measured how much true-label mass this truncation
  actually discards at the chosen cap (max 451 labels/protein exists; mean 43.7/median 31 suggests
  most proteins fit, but the tail doesn't) — flagged as a known gap, not yet quantified.
- **Exposure bias:** scheduled sampling (Bengio et al. 2015) — `--ar-scheduled-sampling-final-prob`
  (default 0.25) linearly ramped in after `--ar-scheduled-sampling-warmup-frac` (default 0.3) of
  training, so the decoder is gradually exposed to its own (possibly wrong) predictions during
  training rather than only ever seeing ground-truth history, which is all it would get at
  inference otherwise.
- **Repeat-prevention:** a per-example `emitted_mask` over the 27,191-label vocabulary, updated
  after every step with whichever label was actually fed forward (true or scheduled-sampled, kept
  consistent between training and inference) — a simple deterministic mask, no learned mechanism
  needed.
- **Stopping/inference:** greedy (argmax) decode, step count bounded by `--ar-max-decode-steps`;
  the dedicated `stop_head` output (sigmoid > 0.5) decides when a realistic prediction set actually
  ends. No beam search — greedy is the standard first choice and the simplest decode strategy that
  works; only escalate if greedy measurably underperforms.
- **Meta-device init bug (same class of bug item 8's Step 3 found, pre-empted here):** verified
  directly that `classifier.cell`'s GRUCell parameters (`weight_ih`/`weight_hh`/`bias_ih`/
  `bias_hh`) are NOT isinstance-matched by `EsmPreTrainedModel._init_weights` (only
  `nn.Linear`/`nn.Embedding`/`nn.LayerNorm` are) and so come back as the same kind of meta-device
  garbage `scale`/`bias` did for item 8 — `create_esm()`'s fresh-start-only fix block now also
  calls `model.classifier.cell.reset_parameters()` alongside the existing `scale`/`bias`/
  `term_embeddings` re-fills.
- **Diagnostic adapter, with an explicit caveat:** `run_dense_diagnostic()` in the standalone
  script runs `generate()` for the full `--ar-max-decode-steps` budget regardless of the stop head
  (so every label gets a score), takes the max logit any label received across all steps as its
  AUROC-comparable score, and reuses `_diagnostic_metrics_from_arrays()` directly rather than
  reimplementing the AUROC/F1 math. **Important asymmetry:** ground truth for this diagnostic is
  also truncated to each protein's first `--ar-max-decode-steps` depth-ordered true labels, the
  same truncation training uses — this measures "how well does this architecture do on what it can
  represent at all," not a fair apples-to-apples number against the classifier/item-8 baselines'
  full-27,191-label-set AUROC. Any comparison across architectures needs this caveat stated
  alongside the number, not silently dropped.
- **Deliberately not wired into `training_run.py`'s main() training loop.** That loop's loss/
  metrics/batching all assume a fixed-shape dense multi-hot target; this mechanism's variable-length
  sequential decode doesn't produce one. Lives in its own standalone script
  (`experiments/label_embedding/train_autoregressive_head.py`), reusing `create_esm()`/
  `create_esm_tokenizer()`/`compute_label_depth()`/`build_hierarchy_edge_index()`/
  `_diagnostic_metrics_from_arrays()`/`load_training_data()` from `training_run.py` rather than
  duplicating that logic — same precedent as `experiments/mf_span/scripts/train_mf_span_model.py`
  and `experiments/pk_eval/scripts/*.py` being their own dedicated scripts rather than main()
  branches. Only promote into `training_run.py`'s own CLI if a cheap-scale spike's result
  justifies it — the same validate-cheap-first ordering item 8 itself followed (precompute →
  validate → only then integrate).

**Status (2026-10-24): cheap-scale GPU spike run — real positive signal, not yet proven
competitive.** Same scale as item 8's trials (2,000 train / 500 val, 5 epochs,
`esm2_t12_35M_UR50D`, `unfreeze_last_n=4`, dropout 0.1/0.1, weight_decay 0.01, batch_size
2/grad_accum 8), `--ar-max-decode-steps 64`:

| metric | item 8, single-point head (both trials) | item 16, autoregressive decoder |
|---|---|---|
| mean per-label AUROC | 0.4999 / 0.5028 (chance, flat) | **0.5814** |
| median per-label AUROC | — | 0.5869 |
| rare-term mean AUROC | — | 0.5875 (n=2,248) |
| common-term mean AUROC | — | 0.5745 (n=2,010) |
| val_loss trajectory | flat/stuck by epoch 2 | still decreasing every epoch through epoch 5 (6.79→5.77) |

This is a real, qualitatively different result from item 8's two chance-level failures: ranking
signal above 0.5 for the first time on this label-embedding research line, rare/common terms
roughly balanced (no collapse), and — unlike item 8's trials, which had already flatlined by this
scale — `val_loss` was still improving at epoch 5, suggesting cheap scale is understating this
mechanism's ceiling rather than revealing it.

**Two caveats before trusting 0.5814 as *the* number:** (1) the diagnostic's ground truth is
truncated to each protein's first 64 depth-ordered true labels (the same cap training uses), not
the full true-label set item 8's classifier baseline (0.6577) was scored against — so this isn't
a directly fair comparison yet, just evidence the mechanism ranks real signal rather than sitting
at chance. (2) only 4,258/27,191 labels were evaluable at this val-sample count (both classes
present), same general caveat every cheap-scale AUROC in this project carries.

**Status update (2026-10-24): longer run (20 epochs, same 2,000/500-example scale) — bug hunt
came back clean, and the mechanism keeps improving with more training.** Per the user's explicit
request to use this run to hunt for setup errors first, two diagnostics were added before
scaling epochs: truncation-rate logging and masked top-1 token accuracy (train+val). Neither
turned up a bug:

- **Truncation rate at the real cap** (`--ar-max-decode-steps 64`): 20.10% (10,570/52,579
  training proteins) — far gentler than the 71% seen at the toy cap of 16 used in CPU smoke
  tests, and not alarming at this cap size.
- **Top-1 token accuracy** rises from 6.8%/10.1% (epoch 1, train/val) to 21-26% by epoch 20 on a
  27,191-way vocabulary — thousands of times above the ~0.004% chance rate, confirming the model
  is genuinely discriminating between labels, not just riding the prior-bias-init loss curve
  down.
- **One real (non-bug) finding the diagnostics surfaced:** `train_top1_acc` peaks at epoch 11
  (0.2252) then goes flat/slightly noisy through epoch 20, while `val_top1_acc` keeps climbing
  smoothly the whole time (0.1010→0.2597). This split starts exactly at epoch 7, when scheduled
  sampling begins ramping in — train's forward pass increasingly feeds the model its own
  (sometimes wrong) predictions as decode history, making the train-side metric strictly harder,
  while val always uses full teacher forcing. This is scheduled sampling working as designed, not
  a wiring error.
- **The CE loss component (the ranking-relevant half of the loss) nearly plateaus by ~epoch 13**
  (4.5953→4.5026 over the last 7 epochs, essentially flat), while `val_loss`'s continued decrease
  after that point is almost entirely the stop head getting more confident (0.4362→0.2673) — a
  simpler length-prior signal, not more label-ranking learning. So the real CE plateau at this
  2,000-example scale is around epoch 13-15, not epoch 20.

Despite the CE plateau, the dense diagnostic's AUROC kept improving meaningfully with more
epochs:

| metric | epoch 5 (first trial) | epoch 20 (this run) |
|---|---|---|
| mean per-label AUROC | 0.5814 | **0.6428** |
| median per-label AUROC | 0.5869 | 0.6654 |
| rare-term mean AUROC | 0.5875 (n=2,248) | 0.6541 (n=2,248) |
| common-term mean AUROC | 0.5745 (n=2,010) | 0.6302 (n=2,010) |

A real, repeatable improvement from more training (not noise at the same n=1,000 val sample
count), and now a clearly convincing margin over item 8's two flat 0.4999/0.5028 trials — both
caveats from the first status update (ground-truth truncation to 64 labels; only 4,258/27,191
labels evaluable) still apply unchanged.

**Not closing this out.** The bug hunt came back clean, so the remaining open question is the
same as before: how far this goes with more scale, not whether it works. Candidate next steps,
still not yet decided: a bigger cheap run (more train examples, since epochs alone are now
showing diminishing CE returns past ~epoch 13), a fairer diagnostic against the FULL label set
(not truncated) for a real head-to-head number against item 8's classifier baseline, or
promoting this into `training_run.py`'s main() loop once confidence is higher.

**Status update (2026-10-24): aspect-root labels excluded from targets — real signal survives,
reduced from an inflated baseline.** User's catch: the 3 GO aspect roots
(molecular_function/biological_process/cellular_component) are near-universally true for any
protein — a trivially-correct, free prediction that was inflating both CE loss improvement and
top-1 accuracy without reflecting real functional discrimination. `build_label_order()` now
drops depth-0 labels before building the per-protein decode target sequence, and the dense
diagnostic's AUROC excludes them automatically (their ground-truth column becomes single-class
once they're never emitted as targets). Same 2,000/500-example, 20-epoch scale as the prior run,
for a direct before/after comparison:

| metric | with roots (inflated) | roots excluded (real) |
|---|---|---|
| mean per-label AUROC | 0.6428 | **0.6158** |
| median per-label AUROC | 0.6654 | 0.6287 |
| rare-term mean AUROC | 0.6541 (n=2,248) | 0.6309 (n=2,343) |
| common-term mean AUROC | 0.6302 (n=2,010) | 0.5984 (n=2,030) |
| val_top1_acc (epoch 20) | 0.2597 | 0.2146 |
| train_top1_acc | 0.2105 (peaked 0.2252 @ epoch 11) | 0.1673 (plateaus ~epoch 9-10) |
| truncation rate (cap 64) | 20.10% | 18.62% (3 fewer forced slots/protein) |

Roots were worth roughly 0.027 absolute mean AUROC and ~0.045 absolute top-1 accuracy of "free"
credit — real, but a modest fraction of the total. **The key result: 0.6158 still clears item
8's chance-level trials (0.4999/0.5028) by a wide, convincing margin** — this was not a case
where the apparent signal was mostly trivial root predictions; the decoder genuinely
discriminates between real, non-trivial functional terms. `train_top1_acc` now plateaus earlier
and flatter (~epoch 9-10) than before, while `val_top1_acc` still creeps up through epoch 20 —
the same scheduled-sampling-driven train/val split noted in the prior status update, just more
visible now that the easy 3-root boost no longer smooths over it.

This is now the honest number to carry forward as item 16's result at this cheap scale: **mean
AUROC 0.6158 on non-trivial GO terms only**, both truncation-to-64 and
4,373/27,191-evaluable-labels caveats still apply.

**Status update (2026-10-24): controlled A/B says keep the roots in training after all —
correcting the framing above.** The comparison above conflated two things: removing roots from
training *and* changing which labels got evaluated (truncation freed up slots for different,
harder terms). Added `--ar-include-roots-in-training` to separate them: it controls only
whether roots are training targets; evaluation (`run_eval_pass` and `--run-diagnostic`) now
always scores against the same fixed, root-excluded label set regardless of the flag, via a
dedicated `eval_collate_fn`. Re-ran both configurations at the same scale (2,000/500 examples,
20 epochs) and confirmed `n_labels_evaluable=4373` matched exactly in both logs — a genuine
same-task comparison this time:

| metric | roots excluded from training | roots included in training |
|---|---|---|
| mean AUROC (same 4,373 labels) | 0.6158 | **0.6430** |
| median AUROC | 0.6287 | 0.6657 |
| rare-term mean AUROC | 0.6309 | 0.6530 |
| common-term mean AUROC | 0.5984 | 0.6314 |
| val_top1_acc (epoch 20, same eval) | 0.2146 | 0.1884 |

Training **with** root supervision scores *higher* on the exact same non-trivial terms (+0.027
absolute mean AUROC, +0.037 median) — the opposite of what the previous status update assumed.
Reassuringly, 0.6430 lands almost exactly on the original uncontrolled with-roots run's 0.6428,
while the without-roots condition is consistently lower in both the controlled and uncontrolled
measurements — suggestive of a real effect rather than pure run-to-run noise, though this is
still a single seed per condition, not a replicated result.

One instructive wrinkle: `val_top1_acc` moves the *other* direction (0.2146 without-roots vs.
0.1884 with-roots) — teacher-forced exact-next-token accuracy gets slightly worse with roots in
the sequence, even though `generate()`'s autonomous-decode AUROC ranking gets better. Best
current guess: the root steps act as a short, easy "warm-up" that commits the GRU's hidden state
to an aspect before the harder specific terms follow, which costs a little raw per-token
sharpness but apparently helps downstream ranking quality — plus a few extra supervised steps
per protein that add useful gradient signal at this data-starved cheap scale. Not fully
explained; worth revisiting if this result doesn't hold up at larger scale.

**Revised takeaway: keep `--ar-include-roots-in-training` on by default going forward** (it's
currently opt-in via the flag) — the earlier instinct to exclude roots as "pure free credit with
no informational value" was wrong about the mechanism, even though the underlying observation
(roots are near-trivially true, so their own per-label AUROC is uninteresting) still holds. The
evaluation-side fix (always score against the root-excluded label set) is the part that mattered
and stays.

**Status update (2026-10-24): two pre-big-run checks, before committing more scale.** Added
`run_full_label_diagnostic()` (generate()'s already-full-vocabulary `best_logit_per_label`
scored against each protein's complete, untruncated label set — item 8's classifier baseline
convention, no truncation caveat) and a stop-head calibration print in `run_dense_diagnostic()`
(mean |predicted stop step − true length|). Same scale as the prior A/B (2,000/500 examples, 20
epochs, `--ar-include-roots-in-training`, default seed — this run reproduced the prior one's
epoch curve bit-for-bit, so it's a determinism check, not an independent second seed):

- **Full-label-set AUROC: mean 0.6433, median 0.6613, 6,820/27,191 labels evaluable** — the real,
  fair number against item 8's classifier baseline (`mean_auroc=0.6577`, same 2,000/500-example
  scale, though that baseline was only 5 epochs vs. this run's 20). Landing within ~0.014
  absolute of a baseline that's a proven, competitive architecture at this scale is a genuinely
  good result — the autoregressive decoder isn't just "better than item 8's broken single-point
  head" anymore, it's in the same tier as the classifier baseline.
- **Stop-head calibration is weak: mean |predicted_stop_step − true_length| = 29.58 steps**
  (max_steps=64) — roughly on the order of a typical sequence length itself, meaning the
  dedicated stop head doesn't reliably find the real cutoff point. This doesn't taint any AUROC
  number reported so far (every diagnostic forces the full `max_steps` decode budget,
  sidestepping the stop head's own cutoff decision entirely), but it would matter for any
  realistic variable-length inference/submission use of `generate()`. Flagged as a known gap to
  fix before relying on the stop head operationally — not blocking the ranking-quality
  conclusion above, but on the list before any real submission path.

**Green light for more scale on the ranking question; stop-head calibration is the next thing
to fix, not epoch/data count** — CE loss plateaus by ~epoch 13 at this scale (noted in the
longer-run status update above), so the lever that would move the full-label AUROC further is
likely more training data, not more epochs on the same 2,000 examples.

**Status update (2026-10-24): tried fixing stop-head calibration directly, reverted — the real
cause is content starvation, not architecture.** Two attempts: (1) a prior-bias init for
`stop_head.bias` (same rationale as the label head's bias fix) — genuinely helped overall AUROC
(0.6433 → 0.6580, matching item 8's classifier baseline) by no longer wasting early training on
bias learning, but left calibration itself unchanged (MAE ~30/64 steps, both before and after).
(2) Feeding an explicit `step_frac` (current step / max_steps) into `stop_head` — reverted,
no improvement to calibration and coincided with an AUROC regression (0.6580 → 0.6153, likely
RNG-stream noise from the parameter-count change rather than a causal effect, since `stop_head`'s
calibration has no mathematical path to affect AUROC at all — `generate()`'s AUROC score never
reads `stop_at`). Dataset checks ruled out two other hypotheses: no protein has duplicate label
ids (0/5000 sampled), and 80.4% have an unambiguous single deepest/leaf-like true label (tie
ambiguity only affects ~20% of proteins, nowhere near enough to explain a ~30-step MAE).

Added a qualitative example printer (`print_example_predictions()`) to see what's actually
happening, and it found the real cause: at this cheap scale (2,000 examples), the model predicts
**nearly the same generic sequence for every protein** regardless of true content — a protein
with only 3 true labels and one with 75 true labels (totally different domains: Golgi-apparatus
vs. transport/metabolism) both decode into the same generic "3 roots + nucleus/RNA-processing
cluster" profile past the first few steps. The stop head's weak, nearly-content-independent
timing signal (same slowly-rising probability curve for both examples) is a direct symptom of
this, not an independent architecture bug — a stop head can't learn protein-specific timing when
the model's own belief about "what this protein is" barely varies across genuinely different
proteins. Fixing this needs the model to first learn real content differentiation, which is a
data-scale problem, not a stop-head problem — confirms the "more training data" lever is the
right next move, now with concrete evidence of *why* rather than just the CE-loss-plateau
argument.

**Parked idea, not pursued now:** user raised whether this maps onto Matryoshka representation
learning + RL, given the natural coarse-to-fine (root → general → specific) decode structure.
Agreed assessment: the intuition about coarse-to-fine structure is right, and RL-style
sequence-level reward *would* elegantly unify ranking + stopping into one objective instead of
two competing surrogate losses — but (1) GO is a DAG with multi-parent inheritance, not a clean
single axis Matryoshka nesting needs (already hit this exact issue with depth ties), and (2) RL
methods are more sample-hungry than supervised CE, the wrong direction when the diagnosed problem
is sample starvation. User's own counter-point: **term frequency is a clean, always-computable
total order on GO terms, unlike graph depth** (no multi-parent ambiguity, no ties in practice) —
worth remembering as an alternative decode-ordering scheme (`build_label_order` currently sorts
by `(depth, id)`; frequency could replace or supplement depth) independent of whether RL is ever
revisited. Explicitly deferred: establish the more-data baseline first, revisit alternatives only
if that doesn't resolve the content-differentiation problem.

**Status update (2026-10-25): no plateau at 20 epochs after all — AUROC still climbing through
epoch 40, now beats item 8's classifier baseline for the first time.** User correctly challenged
the earlier "CE plateaus by ~epoch 13 / more data has peaked" framing: a CE plateau under teacher
forcing is not evidence that `generate()`-based AUROC (greedy, self-fed decode) has also
plateaued — these can decouple via exactly the exposure-bias mechanism scheduled sampling exists
to mitigate. Added `--diagnostic-every-n-epochs` to measure the actual metric periodically during
training instead of inferring from the CE proxy. Same 10,000-example run, extended to 40 epochs
(user's own call), dense/truncated mean AUROC at each checkpoint:

| epoch | mean AUROC | 0.5-threshold stop MAE |
|---|---|---|
| 5 | 0.5851 | 30.58 (flat floor, same every prior run) |
| 10 | 0.5934 | 30.58 |
| 15 | 0.6029 | 30.58 |
| 20 | 0.6104 | 30.58 |
| 25 | 0.6119 | 30.38 |
| 30 | 0.6345 | 28.30 |
| 35 | 0.6688 | 26.49 |
| 40 (final) | 0.6807 | 22.70 |

Still climbing at epoch 40, with the largest gains in the *last* 10 epochs — no sign of
flattening. **Full-label-set AUROC at epoch 40: 0.6671, finally exceeding item 8's classifier
baseline (0.6577)** for the first time anywhere in this research line (previous best was 0.6580
at the smaller 2,000-example/20-epoch scale, which trailed the baseline).

**The stop-head calibration improved as a direct side effect, exactly as the content-starvation
diagnosis predicted** — not from any architecture change, just from giving the model enough
training exposure to actually have specific content to decide a stopping point from. The
0.5-threshold MAE had been stuck at a flat 30.58 in *every single prior run* (meaning
`sigmoid(stop_logit)` never once crossed 0.5) — it only starts moving around epoch 25-30, reaching
22.70 by epoch 40 and nearly converging with the argmax-ranking variant (21.89). Qualitatively,
the transport-heavy example (75 true labels, `GO:0005215`/`GO:0022857`) now decodes into a
genuinely coherent, domain-specific cascade (transporter → transmembrane transporter → cation
channel → potassium channel activity) rather than a generic fallback profile.

**Not yet at a true plateau — AUROC is still rising, so the real ceiling at this 10,000-example
scale hasn't been found yet.** Candidate next step: push further (60-80 epochs) using the now-
working periodic-diagnostic instrumentation to actually locate where it tops out, rather than
guess again.

**Status update (2026-10-31): pushed to 80 epochs — still no plateau, stop head has effectively
converged, clear margin over item 8's baseline.** Same 10,000-example run extended to 80 epochs
(user's own call again), full AUROC-vs-epoch curve:

| epoch | mean AUROC (dense) | 0.5-threshold stop MAE | argmax stop MAE |
|---|---|---|---|
| 10 | 0.5923 | 30.58 | 24.16 |
| 20 | 0.5846 | 30.58 | 22.88 |
| 30 | 0.6041 | 28.47 | 20.69 |
| 40 | 0.6114 | 25.26 | 21.48 |
| 50 | 0.6448 | 23.86 | 20.46 |
| 60 | 0.6780 | 21.86 | 20.89 |
| 70 | 0.6905 | 21.42 | 20.97 |
| 80 (final) | 0.7015 | 21.04 | 20.45 |

**Full-label-set AUROC at epoch 80: 0.6877** — a clear, growing margin over item 8's classifier
baseline (0.6577), not just barely crossing it as at epoch 40 (0.6671). Still rising at the very
end (0.6905→0.7015, epoch 70→80), no flattening visible yet even now. (Note: epoch 20 dipped
slightly below epoch 10 — non-monotonic blip, not a trend reversal; sustained climb resumes from
epoch 30 on.)

**The stop head has effectively converged, with zero architecture changes** — the 0.5-threshold
and argmax-ranking MAEs, which differed by ~6-9 steps through epoch 40, are now nearly identical
(21.04 vs 20.45 at epoch 80). `sigmoid(stop_logit)` reliably crosses 0.5 at a sensible point now,
confirming end-to-end that stop-head calibration was always downstream of content
differentiation, never an architecture problem to be fixed directly (consistent with both the
earlier `stop_head.bias`-fix and reverted `step_frac` attempts).

**Real cost: `time` reports 17.5 hours (`real 1052m2s`) for this single run** — roughly linear
with the 40-epoch run's extrapolated ~9-10h, confirming the compute-scaling model held. This is
well outside "cheap-scale exploration" territory in wall-clock terms now, even though the data
scale (10,000/52,579 ≈ 19% of the full training set) is still a fraction of full-dataset
training. Not yet decided: whether to keep pushing epochs at this same 10,000-example scale
(curve still rising, cost roughly doubles each time), pivot to a full-dataset run now that "more
exposure reliably helps" is well-established, or treat 0.6877 as sufficient evidence to promote
this into `training_run.py`'s main() training loop.

**Status update (2026-11-01): pivoted to the full dataset (52,579 examples), 20 epochs — new
best result, still not plateaued.** Before this run, added real checkpoint persistence
(`--checkpoint-dir`/`--resume-from-checkpoint`, `torch.save` of model+optimizer+epoch) to
`train_autoregressive_head.py` — the 10k-scale spikes were ephemeral by design, but a run this
long needs crash recovery. Checkpoints are gitignored (`/experiments/label_embedding/checkpoints/`)
and never committed; a promoted checkpoint would go to the HF Hub instead, same path item 8's
baseline uses. (Separately confirmed the project's GitHub LFS bandwidth complaint was unrelated —
121+27 historical revisions of the shelved mf_span experiment's `positional_mf_links.json`/
`positional_mf_gaps.json`, ~18GB, already on `origin/master` since 2026-09-06 — not this item;
left as-is per the user's call, since it's pre-existing remote state, not new growth.)

Dense/truncated mean AUROC per `--diagnostic-every-n-epochs 2`, same non-monotonic-but-rising
shape as every prior scale:

| epoch | mean AUROC (dense, truncated) |
|---|---|
| 2 | 0.5695 |
| 4 | 0.6016 |
| 6 | 0.5961 |
| 8 | 0.5986 |
| 10 | 0.6188 |
| 12 | 0.6538 |
| 14 | 0.6872 |
| 16 | 0.6831 |
| 18 | 0.6897 |
| 20 (final) | 0.7159 |

**Full-label-set AUROC: 0.6987** (6820/27191 labels evaluable) — the new best result, ahead of
item 8's baseline (0.6577) and the previous best 10k/80-epoch run (0.6877). Still rising through
epoch 20 (0.6897→0.7159, epoch 18→20) with no flattening — the full dataset hasn't even reached
the epoch count the 10k subset needed before *it* showed real convergence signs (stop-head MAE
convergence, slowing AUROC growth), so there's no reason to expect this has converged either.

**Real cost: `real 1484m31s` (~24.7h)** for 20 epochs — matches the pre-run extrapolation
(~23-24.5h) from the 10k-scale timing, confirming the scaling model holds at this data size too.
Per-epoch cost (~74 min) is now the right unit to reason about extending: a `--resume-from-checkpoint`
continuation is cheap incrementally (no restart tax) compared to the first 24.7h. Not yet decided:
extend this run further via resume (more epochs, same full dataset, cost ~74min/epoch) to chase
the still-rising curve, or treat 0.6987 — already a clear margin over item 8's baseline at full
data scale — as sufficient to promote into `training_run.py`'s own `main()` loop now.

**Status update (2026-11-02): promoted into `training_run.py`'s own CLI, then extended the full-
dataset run to 40 epochs via resume — the epoch 20 checkpoint turns out to be the best one, not
epoch 40.** `training_run.py --use-autoregressive-label-head` now short-circuits `main()` into
`run_autoregressive_training()`, factored out of `train_autoregressive_head.py`'s own `main()`
(imported, not duplicated — the autoregressive training loop/loss/batching/diagnostics stay in
that file since they're sequence-shaped and don't fit the rest of `main()`'s fixed-shape dense
multi-hot machinery; the reverse import is deferred inside `training_run.py`'s `main()` to avoid
a circular import). Verified on CPU smoke tests before committing, then verified for real by
using the new path itself to resume the full-dataset run from the epoch-20 checkpoint out to 40
epochs (`--checkpoint-dir`/`--resume-from-checkpoint` both exercised on a real 22h GPU run, not
just the short CPU tests):

| epoch | mean AUROC (dense, truncated) | full-label-set AUROC | val_top1_acc |
|---|---|---|---|
| 20 (prior run) | 0.7159 | **0.6987** | — |
| 24 | — (dense only) | — | 0.3581 |
| 28 | — (dense only) | — | 0.3630 |
| 32 | — (dense only) | — | 0.3639 |
| 36 | 0.7148 | — (dense only) | 0.3627 |
| 40 (final) | 0.7070 | 0.6865 | 0.3630 |

**Extending past epoch 20 did not help — it's a mild regression, not further improvement.**
Full-label-set AUROC dropped from 0.6987 (epoch 20) to 0.6865 (epoch 40), and `val_top1_acc`
plateaued/wobbled in a narrow 0.363-0.370 band from epoch ~28 onward while `train_top1_acc` kept
creeping up (0.4169→0.4257) as `scheduled_sampling_prob` ramped from 0.08 to 0.25 — a train/val
divergence consistent with mild overfitting, not a wiring bug (the same scheduled-sampling-driven
split noted in earlier status updates, just past the point where it still helps). **The epoch-20
checkpoint (`experiments/label_embedding/checkpoints/item16-fulldata/epoch_20.pt`) is the best
result found anywhere in this research line and the one to promote/push to the Hub, not epoch
40** — this reverses the "still rising, not yet plateaued" framing from the 2026-11-01 update;
the plateau was in fact right around epoch 20 on the full dataset, not later.

Not yet done: pushing the epoch-20 checkpoint to the HF Hub (same path item 8's baseline uses).

**Status update (2026-11-02): scored against the real CAFA metric and the real baseline —
the decoder clearly loses, correcting the entire AUROC-based narrative above.** User's
challenge: "the change in quality is so tiny" (re: the AUROC margin over 0.6577), prompting an
actual test-scoring run under the real competition rule (IA-weighted Fmax, DAG-propagated,
threshold-swept — `experiments/label_embedding/evaluate_cafa_fmax.py`, reusing
`compute_ia_weighted_fmax` already cross-checked against the real `cafaeval` package) instead of
AUROC, and against the real Tier-0 baseline (`trial-013`, `kelmoir/cafa6-esm@main`) instead of
item 8's throwaway 2,000-example control run. 2,000-protein test-split subset, same label space,
same proteins, both models DAG-propagated identically:

| | Fmax | precision | recall | @threshold |
|---|---|---|---|---|
| **Baseline** (trial-013, full scale) | **0.3807** | 0.4056 | 0.3586 | 0.25 |
| **Autoregressive decoder** (epoch 20) | 0.2610 | 0.2663 | 0.2559 | 0.98 |

**The decoder loses by a wide margin (−0.12 absolute, −31% relative) on the metric that actually
matters.** This retroactively reframes every "beats the baseline" claim above: those were all
against `mean_auroc=0.6577`, a cheap 2,000-example/5-epoch dense-classifier control run item 8
built for its own ablation, never a real checkpoint — not against `trial-013`, the project's
actual promoted Tier-0 model, which reaches **0.8731 mean AUROC** at full scale (CLAUDE.md). Item
16's decoder (0.6987 full-label AUROC) was never ahead of the real best model on AUROC either; it
only looked competitive against a strawman. The AR decoder's own high confidence threshold
(0.98) at its best Fmax point also suggests its probability outputs are poorly calibrated
relative to the classifier's (best threshold 0.25) — consistent with a model that ranks
real signal (positive AUROC margin over chance) but whose absolute confidence scores don't
translate into good precision/recall trade-offs the way the dense classifier's do.

**Net assessment: item 16 should not be promoted as a replacement for the Tier-0 baseline.** The
promotion work into `training_run.py`'s CLI stays (it's real, reusable infrastructure, and this
scoring script benefited from it), but the autoregressive decoder itself does not currently beat
the thing that matters.

**Root-cause follow-up (2026-11-02): isolated to a training-objective/calibration mismatch, not
decode length.** Two competing hypotheses for the AUROC-vs-Fmax gap: (1) `generate()`'s
`best_logit_per_label` always maxes over the full `--ar-max-decode-steps` budget "regardless of
the stop head" (by design, so every label gets a dense score), but `make_collate_fn`'s
`seq_mask` only supervises steps up to each protein's true label count -- most proteins (mean
43.7/median 31 labels, cap 64) spend a real fraction of that decode in a never-trained,
self-fed-back extrapolation regime, which could be injecting pure noise into every label's
score; or (2) `label_logits = scale * cosine_sim + bias` is trained via per-step **softmax**
cross-entropy (`step_losses`) over a single **shared scalar** `scale` across all 27,191 labels
and every step -- this only has to win a relative ranking within each step, never an absolute,
per-label-calibrated marginal probability the way the baseline's independent BCE-trained logits
are, which is consistent with the pathologically high best-F1 threshold (0.98) seen in both
AUROC and Fmax scoring.

Added `--respect-stop-head` to `evaluate_cafa_fmax.py` (drops each protein's post-stop decode
tail from the max instead of always running the full budget) to separate these. Result: **Fmax
0.2610 -> 0.2688, essentially unchanged (threshold 0.98 -> 0.97)** -- hypothesis (1) is ruled
out, the untrained decode tail is not the dominant cause. This isolates the problem to (2): a
structural training-objective mismatch (per-step softmax CE over a shared scale parameter vs.
the classifier's per-label BCE), not a decode-length or stop-head-calibration bug. Fixing it
would mean either post-hoc calibration (temperature/Platt scaling `best_logit` against held-out
data before thresholding) or a training-time redesign (something that also pressures absolute,
per-label-calibrated scale, not just relative per-step ranking) -- a real architecture change,
not a parameter tweak.

Candidate next steps, not yet decided: attempt post-hoc calibration as a cheap test of whether
this alone closes enough of the Fmax gap to be worth it; abandon this architecture for the
full-label-space GO-term classification task; or treat it as validated infrastructure for item
9's variable-cardinality auxiliary prediction use case (where no fixed-shape baseline exists to
lose against, and relative ranking/sequence structure may matter more than absolute
calibration) rather than as a GO-term classifier replacement.

**Follow-up (2026-11-02): tried the cheapest post-hoc calibration test (`--use-raw-logit`) --
no improvement.** Before attempting a real temperature/Platt fit, checked the cheapest possible
version of hypothesis (2) first: `step()`'s trained `scale` (shared across all 27,191 labels and
every decode step) pushes most `best_logit` magnitudes past ~20, where `sigmoid` saturates to
exactly 0.0/1.0 in float precision -- which would *also* destroy real rank information
regardless of training-objective quality, independent of calibration. Added `--use-raw-logit`
(skips the sigmoid, min-max scales raw `best_logit` into `[0,1]` instead, a monotonic transform
that preserves whatever ranking the saturated sigmoid would have, minus the precision loss).
Result: **Fmax 0.2628 @threshold=0.76** -- statistically indistinguishable from both the default
sigmoid scoring (0.2610 @0.98) and the stop-truncated variant (0.2688 @0.97). All three variants
land in the same 0.26-0.27 band, ~0.12 below the baseline's 0.3807, regardless of how the decode
or the logit-to-probability mapping is handled.

This rules out float-precision saturation as well: it isn't that a real, strong ranking signal
is hiding behind a collapsed sigmoid -- the `best_logit` ranking itself carries less
discriminative power at the precision/recall operating point Fmax cares about than the
baseline's independently-calibrated per-label BCE logits do. Any *global* monotonic rescaling
(sigmoid, min-max raw-logit, stop-truncated raw-logit -- all a single affine map applied
uniformly across every label) cannot manufacture discriminative power that isn't there: it
preserves the same ranking both within and across labels, so it's already fully explored by the
existing threshold sweep (a global Platt/temperature fit -- single `a`,`b` for every label --
would be mathematically redundant with `--use-raw-logit`'s result, not a new data point).

What IS still untested: **per-label** calibration (independent scale/bias fit per label,
against held-out ground truth). Unlike a global rescale, this changes each label's score
*relative to every other label* at a shared global threshold -- exactly the thing a single
shared softmax-CE `scale` never had to get right, since softmax CE only ever compares labels
*within* the same decode step, never calibrates one label's absolute score against another's
across the full 27,191-label space the way independent per-label BCE training does. This is the
next thing worth trying before concluding the architecture itself needs a training-time
redesign or should be retargeted at item 9 instead.

**Follow-up (2026-11-02): tried per-label calibration -- helps, but doesn't close the gap.**
Added `--calibrate` to `evaluate_cafa_fmax.py`: fits independent per-label Platt scaling
(`sigmoid(a[j]*logit + b[j])`, 1-D logistic regression per label via sklearn) against 4,000
held-out **validation**-split proteins (disjoint from the 2,000-protein test split used for
every comparison so far), falling back to one pooled global fit for labels too rare to fit
directly (`--calib-min-pos`/`--calib-min-neg`, default 5 each) -- 3,747/27,191 labels got a
direct fit, the remaining 23,444 (too few positives in 4,000 samples) fell back to the pooled
fit. Applied to the same test-split logits used throughout, then re-scored.

Result: **Fmax 0.2919 @threshold=0.18** (precision=0.3079, recall=0.2775) -- up from 0.2610
uncalibrated, a real **+0.031** absolute gain, not noise (unlike the global-rescale variants,
which all landed within 0.008 of each other). This confirms the per-label cross-label-ordering
hypothesis was onto something real: some of the Fmax loss WAS an artifact of the shared-scale
softmax-CE objective never calibrating one label's absolute score against another's. But it
only recovers about a quarter of the gap to the baseline (0.1197 -> 0.0888 remaining,
~26% closed) -- most of the deficit survives calibration, meaning most of it is a genuine
discriminative-power gap, not a calibration artifact. Per-label Platt scaling with only 4,000
calibration samples also can't meaningfully help the 23,444 labels too rare to fit directly
(the large majority of the label space), so this result likely understates what a better-
resourced calibration (more calibration data, or calibration folded into training itself)
could recover -- but even an generous reading doesn't suggest it would fully close a 0.09 Fmax
gap.

**Follow-up (2026-11-03): tried scoring the decoder's actual generated list -- doesn't help
either.** All of the above still scores the decoder as a dense per-label probability source
(AUROC's native framing), even though item 16's actual purpose (per the class docstring in
`training_run.py`) is to emit a LIST of labels and stop -- `generate()` already returns
`predicted_label_ids` (greedy decode order) and `stop_at` (the step its own stop head first
fires); real usage is `predicted_label_ids[:, :stop_at]`. Added `--binary-list` to
`evaluate_cafa_fmax.py`: every label emitted before stop gets a hard prediction of 1.0,
everything else 0.0 -- no probability, no threshold sweep to game, scored exactly the way the
architecture is meant to be read.

Result: **Fmax 0.2574** (precision=0.2976, recall=0.2267) -- essentially the same as the plain
uncalibrated sigmoid score (0.2610) and *worse* than per-label-calibrated (0.2919). The idea was
sound (sidesteps the probability-calibration question entirely), but it swaps one calibration
problem for another: now the result is gated directly by the stop head's own timing, and that
head's MAE is ~21-22 steps out of a 64-step budget (2026-11-02 status update, same checkpoint) --
far too imprecise to serve as a hard list-length cutoff. Proteins where it fires too early lose
recall outright; proteins where it fires too late pick up extra noise past the real list. This
confirms the stop head itself, not just the per-label score scale, needs real work before this
architecture could be read as "a list generator" in practice -- and that per-label Platt
calibration (0.2919) is still the best result found so far, precisely because it doesn't depend
on the stop head's accuracy at all.

**Follow-up (2026-11-03): epoch 40 beats epoch 20 on calibrated Fmax, despite its weaker AUROC.**
The 2026-11-02 continuation-run finding was that epoch 40's full-label AUROC (0.6865) is slightly
worse than epoch 20's (0.6987), and epoch 20 was picked as "the best checkpoint" on that basis.
Re-ran the per-label-calibrated Fmax comparison against `epoch_40.pt` (same 2,000-protein test
split, same 4,000-protein validation calibration fit) to check whether that AUROC-based pick
still holds on the metric that actually matters.

Result: **Fmax 0.2995 @threshold=0.20** (precision=0.3328, recall=0.2722) -- **better** than
epoch 20's 0.2919, by +0.0076. The AUROC regression does not carry over to Fmax once per-label
calibration is applied: AUROC and Fmax are measuring different things for this model (ranking
quality vs. where the per-label probability actually crosses a shared threshold), so a model that
ranks slightly worse can still calibrate to a better operating point. Gap to the real baseline
narrows slightly (0.3807 - 0.2995 = 0.0812, was 0.0888) but remains substantial. Worth checking
later epochs too if training continues -- AUROC alone is not a reliable proxy for which checkpoint
is actually best on Fmax for this architecture, now that the two metrics have diverged this
noticeably.

**Follow-up (2026-11-03): epoch 32 checked for monotonicity -- flat, not a clean climb.** Same
per-label-calibrated Fmax comparison against `epoch_32.pt`: **Fmax 0.2927** (precision=0.3173,
recall=0.2717) -- essentially unchanged from epoch 20's 0.2919 (+0.0008, noise-level), with all
of the epoch-40 gain showing up between epoch 32 and 40, not gradually across the whole
20-to-40 range:

| Epoch | Fmax (calibrated) |
|---|---|
| 20 | 0.2919 |
| 32 | 0.2927 |
| 36 | 0.2947 |
| 40 | 0.2995 |

(Epoch 36 added after a follow-up run; precision=0.3223, recall=0.2714.) Not enough points to
call this a trend -- could be a genuine late-training uptick, or noise around a plateau that
happened to land slightly higher at 40. The 32->36->40 stretch reads as a small, fairly steady
climb though (+0.0020, then +0.0048), unlike the flat 20->32 stretch -- would need more epochs
(60, 80, ...) to tell a real late-training improvement from noise with any confidence.

**Follow-up (2026-11-03): checked whether the AUROC numbers themselves are a sigmoid-saturation
artifact -- ruled out.** Given the Fmax-side saturation finding (`--use-raw-logit`,
2026-11-02), the same question applies to AUROC: `run_full_label_diagnostic`
(`train_autoregressive_head.py`) computes per-label AUROC via sklearn's `roc_auc_score` on
`sigmoid(best_logit)`, the identical saturating transform. `roc_auc_score` is invariant to any
*monotonic* rescaling of its score argument unless that rescaling introduces new ties -- and a
saturating sigmoid is exactly a tie-introducing transform once logit magnitude exceeds ~20 (many
distinct values collapse to the same float 1.0/0.0). Ties only ever pull AUROC down, never up, so
if this were happening, the "epoch 40 regressed on AUROC" call could have been comparing two
different amounts of a scoring artifact rather than real ranking quality.

Added `check_auroc_saturation.py`: recomputes the identical 1000-example-validation-subset
AUROC (matching `run_full_label_diagnostic`'s own `diagnostic_ds` convention exactly) two ways,
on `sigmoid(best_logit)` and on raw `best_logit` directly, for both epoch 20 and epoch 40.
Result: **zero saturated entries at either checkpoint** (out of 27,191,000 scored entries each),
raw-logit AUROC identical to sigmoid AUROC to four decimal places (0.6996 vs. 0.6996 at epoch 20,
0.6865 vs. 0.6865 at epoch 40, delta exactly +0.0000 both times). The saturation hypothesis is
ruled out for AUROC specifically -- this model's logit magnitudes at these checkpoints never
actually reach the range where sigmoid loses precision, so the reported AUROC regression is
real, not a measurement artifact.

This sharpens rather than overturns the earlier finding: AUROC is correctly measuring a real,
small drop in per-label ranking quality from epoch 20 to 40, while calibrated Fmax genuinely
improved over the same interval. The two metrics haven't diverged because one of them is broken
-- they're measuring genuinely different things (rank order vs. where a calibrated probability
lands relative to one shared threshold), and for this model those two axes have decoupled for
real. AUROC remains a poor proxy for "which checkpoint is best on Fmax" here, but not because
it's buggy.

**Net-net on item 16**: real signal exists (positive AUROC, and calibration recovers real Fmax),
but the architecture as built loses meaningfully to the Tier-0 baseline on the metric that
actually matters, and no cheap post-hoc fix tried so far (decode truncation, raw-logit rescaling,
per-label Platt scaling, hard stop-head-gated list scoring, later checkpoint) closes that gap.
Promoting it as a baseline replacement is not warranted. Worth keeping in mind for item 9's
variable-cardinality use case, where there's no fixed-shape baseline to lose a calibration fight
against; a training-time redesign (replacing or supplementing the shared-scale softmax CE with
something that also pressures absolute, per-label-calibrated scores, and/or giving the stop head
substantially more training signal) is the only remaining path to making this a GO-term classifier
replacement, and hasn't been attempted.

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
