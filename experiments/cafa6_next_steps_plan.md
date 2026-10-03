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
**Prerequisite check, not yet done:** confirm whether `generate_dataset.py`/`training_run.py`'s
current task setup already reflects a PK-style split (partial annotations as input) or is pure
NK-style (predict everything from sequence with zero annotation context) — this determines whether
item 2 is a data-generation change, a model-input change, or both. Scope this as its own investigation
before writing an implementation plan.
**CAFA-5 dataset as PK baseline source (user note, 2026-09-26):** the fastest path to a PK baseline is
likely CAFA-5's own released train/test annotation snapshots, since that competition's evaluation
regime already implements the PK/LK/NK split methodology this item targets (see item 2's report
source above) — rather than inventing a PK split from scratch against CAFA-6 data, check CAFA-5's
data availability/format first and build the baseline against that before deciding whether a
CAFA-6-native PK split is even needed.

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

## Tier 2 — cheap, well-evidenced, low implementation risk

### 4. Replace flat BCE with an IA-weighted or ranking loss
**Source:** report §"Protein language model embeddings are table stakes" — two independent CAFA-5
top-6 teams (Evans, Freddolino Lab) found standard BCE a genuine weak point under this label
cardinality/imbalance and switched to IA-weighted soft-F1-style or ranking losses.
**Why Tier 2 not Tier 1:** well-evidenced and cheap (a loss-function swap, no architecture change),
but the expected per-point gain is smaller and less central than items 1–3; do this alongside or
right after Tier 1 work, not before it.

### 5. GO-DAG-consistency post-processing at inference
**Source:** report §"Stacked ensembles..." — multiple CAFA-5 top solutions apply a cheap final step
propagating/clamping predicted scores so parent terms never score below their children, even when the
underlying model has no hierarchy-aware training signal at all.
**Why Tier 2:** nearly free (no retraining required, a post-hoc numpy pass over predicted probabilities),
and current pipeline's flat multi-label classifier has no such step today. Worth adding regardless of
whether item 3's training-time version also gets built — they're complementary, not redundant.

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
