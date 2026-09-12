# Positive-Unlabeled (PU) learning for GO label refinement — plan

**Not yet implemented.** Written up 2026-09-13, following `dataset_refinement_pilot_summary.md`'s
closeout: the aggregate-per-run-loss approach to detecting mislabeled GO annotations doesn't work
(pooled val_loss/AUROC can't isolate a single cell's effect from a whole training run's worth of
overrides). This plan reframes the same underlying problem — GO's Open World Assumption (OWA),
`experiments/ancestor_sweep/sweep_log.md` trial-080/081, `dataset_refinement_pilot_summary.md` — as a
**positive-unlabeled learning** problem and proposes computing a *per-cell* signal instead of a
*per-run* one.

## Framing

Every `(protein, GO-term)` cell in the label matrix is one of:
- **P (positive)**: cell = 1. A curator found real experimental evidence. Trustworthy.
- **U (unlabeled)**: cell = 0. Not confirmed absent — could be a true negative, or an undiscovered
  positive nobody has annotated yet. Indistinguishable from the label alone.

This is the textbook PU setup, not standard binary classification with noisy negatives. The raw,
unpropagated label space this work uses (`kelmoir/cafa6-raw`) has a **0.556% observed positive
rate** (42,369 proteins × 668 terms) — the *true* rate among "unknown" cells is unknown by
construction (that's the whole problem), and estimating it is a core PU subproblem in its own
right (class prior / mixture proportion estimation, see references below).

## Two directions, not one

### A. Reliable-negative extraction (primary target)

Because true positives are rare, the *low-confidence* tail of the unlabeled pool is heavily
enriched for true negatives even under an imperfect classifier — a classifier trained naively on
unknowns-as-hard-negative (what `training_run.py` already does) still preserves a usable ranking,
just a biased/uncalibrated one. This is the "easier" side of PU learning and the basis of classic
two-step methods (PEBL, S-EM — see references): extract a large, high-confidence negative subset
first, rather than trying to find positives directly.

**Calibration via spies**: hold out a sample of real, already-confirmed positives per term (as
many as `--min-term-annotations`-style eligibility allows), hide them from training (relabel to
0 — literally `dataset_refinement_pilot.py`'s existing override mechanism, repurposed), train,
then score every unlabeled cell *and* every spy. If a candidate "reliable-negative" threshold
(e.g. bottom 60-80% by predicted probability) contains few or no spies, that's a direct, measured
estimate of how safe that region is — not an assumption.

**Downstream use**: replace today's binary hard-0-for-everything-unknown scheme with a three-tier
one — reliable-negative cells keep a hard 0 (preserves real negative signal density, unlike
trial-081's failed full mask), the residual, genuinely-ambiguous middle keeps today's
`--unknown-target-value` soft target (or gets excluded), and (optionally, see B) a small
reliable-positive tier gets promoted. This is the gentler correction trial-080/081's own closing
note asked for, made surgical instead of uniform.

### B. Reliable-positive / confidence-based promotion (secondary, harder)

The original idea this whole thread started from — "take the top 25% most confident unknowns and
treat them as resolved positives." Structurally harder: at a ~0.5% base rate, even a good
classifier's top-ranked cells are not automatically trustworthy, and this session's own pilot data
showed no clean separation at the population level. Worth revisiting only *after* A is working and
a real class-prior estimate exists — likely usable only at a much more conservative cut (top 1-2%,
probably per-term rather than global) than the original "25%" framing, and should be validated
against spies the same way as A before ever being used to change a training target.

## Concrete pipeline (draft, not yet built)

1. Train one baseline model on the real, unmodified label matrix at a meaningful scale (reuse
   `training_run.py` directly — no need for the pilot's per-run-from-scratch repetition machinery,
   since this only needs one well-trained model, not hundreds of noise-comparison repeats).
2. Per eligible term (same `--min-term-annotations` support-floor idea as the pilot), hold out a
   fraction (e.g. 10-20%) of its real positives as spies; k-fold across terms/positives so no term
   permanently loses its only positive examples.
3. Score every unknown cell and every spy with the trained model.
4. Estimate the calibration constant / class prior from spy scores (Elkan & Noto's `c` = mean
   predicted probability on held-out positives is the simplest starting point; the mixture-proportion
   estimators below are more robust if `c` looks unstable per term).
5. Set the reliable-negative threshold per term so the empirical spy leak rate stays below a fixed
   bound (e.g. <1-2%); optionally set a reliable-positive threshold symmetrically.
6. Re-inject the resulting three-tier targets into `training_run.py` (a new `--label-tier-path` or
   similar, pointing at a precomputed per-cell tier assignment) and re-run `--run-diagnostic`.
7. **Success criterion is downstream model quality, not the label-refinement signal itself** —
   compare `--run-diagnostic`'s AUROC/F1 (or the real CAFA score) against the current
   hard-0-everywhere baseline. Detecting an effect in the labels was never the goal; a better
   classifier is.

## Key risks / open questions

- **SCAR assumption**: most of the methods below assume observed positives are a random sample of
  all true positives ("Selected Completely At Random"). GO annotation is biased toward
  well-studied proteins/organisms, so this likely doesn't hold globally — calibration from spies
  should be read as "trustworthy among well-annotated terms," not a universal guarantee (see
  Bekker & Davis 2020 survey below for the SCAR vs. SAR distinction).
- **Per-term vs. global calibration**: annotation depth varies enormously by term; a single global
  class-prior estimate is probably wrong. Likely needs per-term (or per-term-cluster) thresholds,
  which raises a sample-size problem for rare terms — needs its own floor, similar to
  `--min-term-annotations`.
- **Compute cost**: k-fold spy repeats at real CAFA-6 scale (42k proteins × 668 terms) is a
  different cost profile than the pilot's small repeated fresh-model trains; needs its own budget
  estimate before committing to a full run.

## Relevant papers

**Foundational PU methods**
- Liu, Lee, Yu & Li (2002), *Partially Supervised Classification of Text Documents*, ICML —
  introduces the "spy" technique (S-EM) this plan's calibration step is built on.
- Liu, Dai, Li, Lee & Yu (2003), *Building Text Classifiers Using Positive and Unlabeled
  Examples*, ICDM — PEBL, the canonical two-step reliable-negative-extraction-then-classify
  method direction A follows.
- Elkan & Noto (2008), *Learning Classifiers from Only Positive and Unlabeled Data*, KDD — the
  `c`-calibration correction (`p(y=1|x) = p(s=1|x) / c`) used in step 4 above.
- du Plessis, Niu & Sugiyama (2014, NeurIPS; 2015, ICML), *Analysis of Learning from Positive and
  Unlabeled Data* / *Convex Formulation for Learning from Positive and Unlabeled Data* — unbiased
  PU risk estimation (uPU), a more principled alternative to naive unknowns-as-negative training.
- Kiryo, Niu, du Plessis & Sugiyama (2017), *Positive-Unlabeled Learning with Non-Negative Risk
  Estimator*, NeurIPS — nnPU, fixes uPU's overfitting in practice; the most commonly used modern
  PU loss if this plan moves beyond post-hoc label correction into training-time PU loss directly.

**Class prior / mixture-proportion estimation**
- Ramaswamy, Scott & Tewari (2016), *Mixture Proportion Estimation via Kernel Embeddings of
  Distributions*, ICML — KM1/KM2 estimators, a more robust alternative to the simple Elkan-Noto
  mean if per-term spy counts are small.
- Bekker & Davis (2018), *Estimating the Class Prior in Positive and Unlabeled Data through
  Decision Tree Induction*, AAAI — TIcE, another practical class-prior estimator.

**Survey**
- Bekker & Davis (2020), *Learning from Positive and Unlabeled Data: A Survey*, Machine Learning
  (journal) — best single overview; covers the SCAR vs. SAR (Selected At Random, covariate-
  dependent) distinction directly relevant to the annotation-bias risk noted above.

**Domain precedent (GO / protein function prediction specifically)**
- Youngs, Penfold-Brown, Drew, Shasha & Bonneau (2013), *Parametric Bayesian Priors and Better
  Choice of Negative Examples Improve Protein Function Prediction*, Bioinformatics — directly
  on-topic: proposes better negative-example selection for GO function prediction under the same
  OWA problem this plan addresses. The closest existing precedent for direction A.
- Yang, Li, Mei, Kwoh & Ng (2012), *Positive-Unlabeled Learning for Disease Gene Identification*,
  Bioinformatics — PU learning applied to a comparably sparse, OWA-affected biological annotation
  setting (gene-disease association instead of gene-function); useful as a worked methodological
  analog.
- Gaudet & Dessimoz (2017), *Gene Ontology: Pitfalls, Biases, and Remedies* — already the source of
  this repo's OWA framing (`sweep_log.md` trial-080/081); the "absence of evidence isn't absence of
  function" premise this entire line of work rests on.
