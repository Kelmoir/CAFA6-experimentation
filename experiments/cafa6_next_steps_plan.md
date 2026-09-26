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
  post-processing) and item 10 (graph-embedding, if reopened) — the GO-DAG-related items are the ones
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
needs. Not started yet.

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

### 8. Auxiliary sequence-based prediction tasks (user's own idea: predict positional annotations, etc.)
**Source:** not directly evidenced in the research (no source reviewed combined multi-task sequence
objectives with GO prediction), but consistent with two things the research does support: (a) the
general principle that orthogonal evidence sources help (item 1), and (b) the finding that structure
specifically delivers only modest gains (item 10 below) — so a sequence-based auxiliary task avoids
paying that cost while still targeting "broader representation learning" the same way structure
was meant to.
**Why Tier 3:** untested idea, no direct precedent found. Per
[[feedback_check_metric_can_see_the_effect]]: before running this, decide up front what metric would
actually show the auxiliary task helped GO prediction specifically (e.g. does adding it measurably
change per-term AUROC on terms it's plausibly related to?), not just whether the auxiliary task itself
trains — that discipline was the exact gap that caused the two-tier PU trial to initially look like a
null result.

### 9. Reopening the shelved two-tier PU-learning mechanism, conditional on a materially stronger base predictor
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

### 9. Structure-based (AlphaFold-predicted structure) features
**Source:** report §"Structure and ontology awareness..." — an independent 14-method benchmark found
pure-sequence, ontology-aware DeepGOZero+ beat the structure-integrating DeepGraphGO (0.463 vs. 0.427
Fmax on biological process); DeepFRI's and GAT-GO's structural gains are characterized as only
"modest"; even PANDA-3D's own feature-importance analysis ranks sequence-derived ESM features above
its structural ones. This validates the user's own instinct (already planning sequence-based, not
structural, auxiliary tasks) rather than contradicting it.
**Status:** not worth the AlphaFold-structure dependency and GCN/GVP training cost given the evidence;
revisit only if a specific benchmark surfaces showing structure helps *on top of* a well-built
sequence+ontology pipeline specifically (the comparisons found so far are against weaker baselines).

### 10. Full GO graph-embedding/GNN approach (`experiments/graph_embedding_plan.md`)
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
already-built two-tier mechanism's effect size, which is Tier 3 item 9 above) stays out of scope: the
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
