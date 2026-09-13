# Positive-Unlabeled (PU) learning for GO label refinement — plan

**Not yet implemented.** Originally written 2026-09-13 following `dataset_refinement_pilot_summary.md`'s
closeout: the aggregate-per-run-loss approach to detecting mislabeled GO annotations doesn't work
(pooled val_loss/AUROC can't isolate a single cell's effect from a whole training run's worth of
overrides). Reframes the same underlying problem — GO's Open World Assumption (OWA),
`experiments/ancestor_sweep/sweep_log.md` trial-080/081, `dataset_refinement_pilot_summary.md` — as a
**positive-unlabeled learning** problem and proposes computing a *per-cell* signal instead of a
*per-run* one.

**Revised 2026-09-13**, same day, after reading the full paper summaries in `paper_summaries/`: the
plan below now goes directly for the most modern/best-performing PU principles — **nnPU** (Kiryo,
Niu, du Plessis & Sugiyama 2017) as a training-time loss, with **KM2** (Ramaswamy, Scott & Tewari
2016) supplying the per-term class prior — rather than the discrete two-step
reliable-negative/ambiguous/reliable-positive tiering scheme the original draft below centered on.
Higher compute cost is accepted deliberately in exchange for not settling for the cheaper-but-weaker
options (TIcE for speed, Elkan-Noto's simple mean, discrete label tiering). The original two-step
design is kept at the bottom of this file for context, not as the plan going forward.

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

## Chosen approach: nnPU loss with KM2-estimated per-term class priors

Naive training with unknowns-as-hard-negative — what `training_run.py` does today — is not just
"noisier," it's a systematic, theoretically characterized bias (du Plessis, Niu & Sugiyama
2014/2015): a classifier trained this way converges to a shifted decision boundary, not merely a
less-confident one. **nnPU** (Kiryo et al. 2017) is the literature's own fix for this, built
specifically for flexible/deep models (uPU, the unbiased-but-unstable predecessor, is known to
overfit badly on exactly this kind of model — its empirical risk can go arbitrarily negative even
though the true risk can't). Crucially, nnPU corrects the loss **directly** — it needs no discrete
reliable-negative/ambiguous/reliable-positive relabeling step at all. That supersedes both of the
original plan's two directions:

- Direction A (reliable-negative extraction via spies → hard-0 for the "safe" tail) becomes
  unnecessary: nnPU's risk correction already accounts for the fact that U is a mixture of true
  positives and true negatives, without needing to first partition U into discrete buckets.
- Direction B (reliable-positive promotion for top-confidence unknowns) becomes unnecessary for the
  same reason — nnPU doesn't need an explicit high-confidence-promotion step to get credit for the
  true positives hiding in U; it's baked into the risk formula via the class prior.

nnPU needs exactly one externally-supplied ingredient per label: the class prior π (the true
positive rate within U). This plan uses **KM2** ("gradient thresholding" mixture-proportion
estimation, Ramaswamy, Scott & Tewari 2016) to estimate it per term, in preference to:
- **Elkan-Noto's simple mean-`c`**: no proven convergence rate, and per Elkan & Noto's own paper,
  a single global estimator without the per-subset robustness KM2/TIcE offer.
- **TIcE** (Bekker & Davis 2018): explicitly the *cheap* alternative — an order of magnitude faster
  than KM1/KM2, which is exactly why both its own paper and the Bekker & Davis (2020) survey
  recommend it when 668 separate per-term estimates need to be affordable. That's the "not the best
  bet, just the cheap one" tradeoff this plan is deliberately not taking.
- **KM1**: KM2 is reported more robust to its internal threshold hyperparameter ν; KM1 is kept only
  as an internal cross-check if a given term's KM2 QP fails to converge cleanly.

This also has a second, independent justification beyond "most modern": Gaudet & Dessimoz (2017)
document concrete, empirical bias mechanisms in GO annotation (species, authorship, annotator,
propagation bias) that make **SAR** (Selected At Random — labeling probability depends on the
protein, not just a global constant) more realistic than the **SCAR** assumption every method here
relies on, including nnPU/KM2. The Bekker & Davis (2020) survey's own follow-up work found that
**under SAR specifically, KM2 outperforms TIcE in accuracy** — so even setting aside the "accept
the cost" instruction, KM2 is the better bet of the two once SCAR is doubted, not merely the more
expensive one.

**This does not remove the SCAR/SAR risk** — nnPU and KM2 assume SCAR exactly as much as the
discarded alternatives did; going with the theoretically strongest single method is a bet, not a
fix, on that open question (see "Key risks" below). The spy technique from the original plan is
kept, but **repurposed**: not for building reliable-negative/positive tiers, but as an independent,
measured cross-check on each term's KM2 prior — the only empirical signal available, since GO has
essentially no gold-standard negatives to validate against (Gaudet & Dessimoz: <1% of experimental
UniProt-GOA annotations are explicit "NOT" negatives).

## Concrete pipeline (draft, not yet built)

1. Train one baseline model on the real, unmodified label matrix at a meaningful scale (reuse
   `training_run.py` directly — this only needs one well-trained model, not repeated trains).
2. Extract pooled per-protein embeddings from the trained baseline (e.g. mean-pooled final hidden
   state, or the pre-classifier representation) for every protein in train+val, and persist them to
   disk once (e.g. an `.npy`/safetensors file keyed by `EntryID`). This is the shared input every
   per-term KM2 call below reads from — built once regardless of how many of the terms get
   processed, and invalidated only if the baseline checkpoint changes.
3. Per eligible term (same `--min-term-annotations`-style support floor as `training_run.py`
   already uses elsewhere — PU correction is unreliable below a positive-count floor; terms under
   the floor skip PU entirely and keep today's plain BCE-with-unknowns-as-negative unchanged):
   a. Run **KM2** with P = that term's positive-protein embeddings, mixture = the full
      protein-embedding set, to estimate the class prior π_term. **Persist every term's result**
      (π_term, solver diagnostics, the embedding-source checkpoint id) to a cache file (e.g.
      `class_priors_km2.json`), keyed by term. This is a one-time cost per baseline checkpoint,
      reused by every subsequent training run/trial — it must not be recomputed on every
      `training_run.py` invocation.
   b. Cross-check: hold out a spy fraction of that term's positives, score them with the baseline
      model, take the Elkan-Noto e1 mean (mean predicted probability on the spies) as an
      independent second estimate of π_term. Log both numbers side by side per term. Large
      disagreement between the two flags a term whose SCAR assumption is probably violated —
      record it for inspection, don't silently auto-correct it.
   c. Apply Kiryo et al.'s asymmetric-error finding when the two estimates disagree materially:
      bias the chosen π_term slightly **upward**, not toward the lower of the two numbers or a
      simple average. Underestimating π is the failure mode that can collapse nnPU outright
      (du Plessis 2014's π̂ ≤ π/2 result); overestimating degrades gracefully. At a 0.556% observed
      base rate, this asymmetry is not a minor detail — it's the difference between nnPU working at
      all for the rarer terms and it silently failing.
4. Implement a **per-label nnPU risk** (Kiryo et al. 2017 Algorithm 1: non-negative correction with
   mini-batch tolerance β) in `training_run.py`, replacing `BCEWithLogitsLoss` **only for terms
   with a cached, above-floor π_term**; terms without one keep today's behavior unchanged. Each
   label uses its own π_term while all labels still share one backbone — this is a genuine
   per-label generalization of the original (single binary-task) nnPU formulation that the source
   papers don't directly demonstrate. Treat it as new, unreviewed territory: verify on a handful of
   terms first (e.g. that the `max(0, ·)` correction doesn't interact badly across labels sharing
   gradients through the backbone) before running it across all eligible terms at once.
5. Re-run training with the new loss (`--run-diagnostic` enabled, as today) across the same
   eligible terms.
6. **Success criterion is downstream model quality, not the estimation signal itself** — compare
   `--run-diagnostic`'s per-label AUROC/F1 (or the real CAFA score), per term, against the current
   hard-0-everywhere baseline. A plausible-looking π_term is not the goal; a better classifier is.
   Per Bekker, Robberechts & Davis (2019)'s finding that a SCAR-style correction can actively
   underperform doing nothing under real SAR (see "Key risks" below), this comparison is not a
   post-hoc nicety — any term whose nnPU-trained AUROC/F1 doesn't clearly beat its baseline value
   should have its π_term/nnPU treatment reverted to plain BCE for that term, not kept on the
   strength of the theory alone.

## Key risks / open questions

- **SCAR assumption, now fully load-bearing — and empirically shown to be able to backfire, not
  just underperform**: nnPU and KM2 both assume SCAR (observed positives are a random sample of all
  true positives, independent of x). Gaudet & Dessimoz (2017) document concrete GO-specific
  violations (species, authorship, annotator, propagation bias) that make the more general SAR
  (Selected At Random) model more realistic. Bekker & Davis (2020)'s own method-selection guidance
  (§5.6) notes that when separability holds better than a clean global SCAR constant, Two-Step
  (separability-based) methods can be *more* robust than class-prior methods like this one — the
  two-step design this plan discards was, on paper, the more SCAR-robust hedge. Going with
  nnPU+KM2 anyway is a deliberate bet on the theoretically strongest single method per this
  session's explicit instruction, not a resolution of this risk. **This is not just a
  weaker-guarantee concern**: Bekker, Robberechts & Davis (2019) — the direct SAR follow-up paper,
  see its own summary below — empirically show that under genuinely SAR-violated data, a
  SCAR-style class-prior correction (their benchmarks used KM2 and TIcE specifically) can perform
  *worse* than doing nothing at all (the same unknowns-as-negative Naive baseline
  `training_run.py` uses today), because a wrong-subspace correction introduces both new false
  positives and new false negatives, whereas doing nothing only ever contributes false negatives.
  Two mitigations follow from that paper: (1) the spy cross-check (pipeline step 3b) is necessary
  but per this finding **not sufficient on its own** — pipeline step 6's baseline comparison is
  therefore not just good practice but the only real safety net, and any term/run where nnPU+KM2
  doesn't clearly beat the current baseline should be treated as a live SAR signal, not a wash, and
  reverted to plain BCE for that term rather than kept "because the theory says it should help";
  (2) if the spy cross-check keeps flagging material disagreement for a meaningful share of terms,
  the same paper's propensity-weighted ERM / SAR-EM approach (jointly modeling an instance-level
  labeling propensity e(x) from *known* bias covariates — species, evidence-code mix, annotation
  count as a "how well-studied" proxy — instead of one constant π_term) is the documented
  escalation path, at real added engineering/compute cost consistent with this plan's
  accept-the-cost stance. Not built in this revision; noted as the next step if the baseline
  comparison keeps showing regressions.
- **π estimation error is asymmetric and the failure mode is a cliff, not a slope**: du Plessis
  et al. (2014) show PU learning fails completely once π̂ ≤ π/2 (not a graceful degradation); Kiryo
  et al. (2017) confirm empirically that underestimating π hurts more than overestimating it. At a
  0.556% observed base rate, per-term π estimates must be both accurate and, when in doubt, biased
  high rather than split-the-difference.
- **Per-label nnPU on a shared-backbone multi-task network is unreviewed by the source papers**
  (they demonstrate single binary-task nnPU only). Needs its own smoke test — a handful of terms,
  checked for gradient sanity — before trusting a full 668-term run.
- **No ground-truth negatives exist anywhere in the dataset** to validate any of this against
  (Gaudet & Dessimoz: <1% of experimental UniProt-GOA annotations are explicit "NOT" negatives).
  The spy cross-check is the only empirical signal available, regardless of which method is chosen.
- **Compute cost, accepted but must stay a one-time cost**: KM2 is a per-term kernel-embedding QP
  over the full ~42k-protein embedding set — expensive by design compared to TIcE, and that's the
  accepted tradeoff. The one concrete engineering requirement this creates: the per-term
  `class_priors_km2.json` cache (pipeline step 3a) must genuinely be computed once per baseline
  checkpoint and reused, not recomputed per training run/trial — otherwise the accepted cost
  compounds across every future experiment instead of being paid once.

## Relevant papers

**Primary methodology (this plan's chosen path)**
- Kiryo, Niu, du Plessis & Sugiyama (2017), *Positive-Unlabeled Learning with Non-Negative Risk
  Estimator*, NeurIPS — **nnPU**, the training-time loss this plan builds into `training_run.py`.
  Fixes uPU's overfitting on flexible/deep models via a non-negative risk correction
  (Algorithm 1, mini-batch tolerance β); also the source of the "underestimating π hurts more than
  overestimating" asymmetry this plan relies on for calibration safety.
- Ramaswamy, Scott & Tewari (2016), *Mixture Proportion Estimation via Kernel Embeddings of
  Distributions*, ICML — **KM1/KM2**, the class-prior estimator this plan uses (KM2 specifically,
  for its greater robustness to its threshold hyperparameter ν and its better accuracy than TIcE
  under SAR per the Bekker & Davis 2020 follow-up comparison).
- du Plessis, Niu & Sugiyama (2014, NeurIPS; 2015, ICML), *Analysis of Learning from Positive and
  Unlabeled Data* / *Convex Formulation for Learning from Positive and Unlabeled Data* — uPU, the
  unbiased-but-overfitting predecessor nnPU fixes; also the source of the π̂ ≤ π/2 collapse result
  that motivates this plan's "bias high, not low" calibration rule.

**Foundational PU methods (background / superseded design, see below)**
- Liu, Lee, Yu & Li (2002), *Partially Supervised Classification of Text Documents*, ICML —
  introduces the "spy" technique, retained here only as the cross-check mechanism in pipeline
  step 3b (not for label tiering).
- Liu, Dai, Li, Lee & Yu (2003), *Building Text Classifiers Using Positive and Unlabeled
  Examples*, ICDM — PEBL / the canonical two-step framework; superseded by the nnPU approach above
  (kept in the "superseded" section below for context).
- Elkan & Noto (2008), *Learning Classifiers from Only Positive and Unlabeled Data*, KDD — the
  `c`-calibration / e1 spy-mean estimator used as this plan's cross-check signal (step 3b), even
  though KM2 (not Elkan-Noto's mean) is the primary π_term estimator.

**Class prior / mixture-proportion estimation (alternatives considered, not chosen as primary)**
- Bekker & Davis (2018), *Estimating the Class Prior in Positive and Unlabeled Data through
  Decision Tree Induction*, AAAI — **TIcE**, deliberately not used as the primary estimator (see
  "Chosen approach" above) despite being an order of magnitude faster than KM1/KM2; the
  compute-cost tradeoff this plan is explicitly not taking.

**SAR-specific follow-up (mitigations)**
- Bekker, Robberechts & Davis (2019), *Beyond the Selected Completely At Random Assumption for
  Learning from Positive and Unlabeled Data*, ECML/PKDD — the direct SAR follow-up to TIcE; source
  of the finding that a SCAR-style class-prior correction can actively underperform doing nothing
  under real SAR (motivates the per-term revert-on-regression rule in pipeline step 6), and of two
  concrete SAR mitigations beyond the spy cross-check this plan already uses: propensity-weighted
  ERM (reweight by a known, per-instance labeling propensity e(x) instead of one term-level
  constant) and SAR-EM (jointly learn the classifier and a low-dimensional propensity model via
  EM when e(x) is unknown). Not built in this revision — the escalation path if the spy cross-check
  keeps flagging trouble; see its own summary in `paper_summaries/` for the full method.

**Survey**
- Bekker & Davis (2020), *Learning from Positive and Unlabeled Data: A Survey*, Machine Learning
  (journal) — source of the SCAR vs. SAR distinction that motivates this plan's spy cross-check,
  and of the KM2-outperforms-TIcE-under-SAR finding that independently justifies choosing KM2 over
  TIcE beyond just "accept the higher cost."

**Domain precedent (GO / protein function prediction specifically — heuristic alternatives, not
PU-loss-based, kept as background)**
- Youngs, Penfold-Brown, Drew, Shasha & Bonneau (2013), *Parametric Bayesian Priors and Better
  Choice of Negative Examples Improve Protein Function Prediction*, Bioinformatics — ALBias, a
  co-annotation-based continuous negative-example prior; no PU framework, spies, or class-prior
  calibration. Its temporal-holdout evaluation (via the same team's 2014 NoGO Database follow-up)
  is a usable alternative evaluation methodology for pipeline step 6 if `--run-diagnostic`'s
  AUROC/F1 proves insufficient.
- Yang, Li, Mei, Kwoh & Ng (2012), *Positive-Unlabeled Learning for Disease Gene Identification*,
  Bioinformatics — PUDI, a four-tier (RN/LP/LN/WN) heuristic scheme in a comparably sparse
  biological PU setting; superseded here by nnPU's continuous risk correction, which needs no
  discrete tiers at all.
- Gaudet & Dessimoz (2017), *Gene Ontology: Pitfalls, Biases, and Remedies* — source of this repo's
  OWA framing and the concrete species/authorship/annotator/propagation bias evidence behind this
  plan's SCAR-vs-SAR risk assessment above.

## Superseded: original two-step tiering design (kept for context only)

The original draft of this plan (2026-09-13, same day) proposed a discrete two-step design instead
of the nnPU approach above. Not the plan going forward, but kept here since the reasoning
(especially the spy-calibration mechanics now reused as a cross-check) informed the revision:

> **Direction A (reliable-negative extraction)**: hold out a sample of real, confirmed positives
> per term as spies, hide them from training, train, then score every unlabeled cell and every spy.
> If a candidate "reliable-negative" threshold (e.g. bottom 60-80% by predicted probability)
> contains few or no spies, that's a measured estimate of how safe that region is. Downstream:
> replace hard-0-for-everything-unknown with a three-tier scheme — reliable-negative cells keep a
> hard 0, the genuinely-ambiguous middle keeps a soft `--unknown-target-value` target, and a small
> reliable-positive tier (Direction B, "top 25% most confident unknowns → resolved positives," later
> judged likely only viable at a much more conservative per-term cut) gets promoted.

This is the classic Liu/Dai/Li/Lee/Yu (2003) Two-Step framework (Step 1: reliable-negative
extraction via Liu, Lee, Yu & Li (2002)'s spy technique; Step 2: iterative reclassification).
Discarded in favor of nnPU because nnPU corrects the same underlying bias directly in the loss,
without needing either the reliable-negative extraction step or the separate reliable-positive
promotion step — see "Chosen approach" above for the full reasoning, including the one place
(SCAR vs. SAR robustness) where this discarded design was arguably the safer hedge.
