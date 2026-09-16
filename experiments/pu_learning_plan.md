# Positive-Unlabeled (PU) learning for GO label refinement — plan

**v1.2, 2026-09-16: `--nnpu` shelved after a real trial underperformed the plain-BCE baseline.**
Ran the plan's own stated success criterion (pipeline step 6: `--run-diagnostic` AUROC/F1
comparison, `--nnpu` vs. plain BCE, matched seed/data/model/epochs, `dataset_raw`, 16,000-row
train subset, 3 epochs) for the first time. Result: `--nnpu` (spy-mean-only calibration, per v1.1)
**lost on every threshold-independent metric** — mean per-label AUROC 0.773 vs. baseline's 0.806,
pooled AUROC 0.859 vs. 0.900, worst-label AUROC 0.21 (worse than random) vs. baseline's 0.41, and
best-achievable F1 (each method's own optimal threshold) 0.274 vs. baseline's 0.348. The one metric
where `--nnpu` "won" — F1 at the fixed default threshold 0.5 — was a recall/precision inversion
artifact (recall 0.37/precision 0.22 vs. baseline's 0.11/0.65), not better discrimination.

Root cause, verified directly rather than assumed: checked `pi_spy` (the sole surviving estimator
after v1.1) against each term's true empirical prevalence. Median overestimation factor **14.4x**,
90th percentile 46.9x, max 205.8x, with 303/437 terms overestimated by more than 10x. Confirmed
with a synthetic degenerate-case test that this is a calibration problem, not a formula bug:
`nnpu_loss` collapses exactly to plain BCE when `pi=0` (verified numerically, exact match); with a
**correct** `pi`, an oracle model that correctly flags hidden positives scores ~0 loss (vs. 0.25
for plain BCE, which punishes it) — the intended, designed divergence nnPU exists to produce. But
feeding the same oracle model our **actual 14x-overestimated** `pi` pushes its loss to **3.25** —
worse than what a naive all-negative model scores even under a *correctly calibrated* prior (0.25).
That means the miscalibrated loss surface's minimum isn't near the correct answer at all; it's
actively steering away from the best possible model, not just failing to help.

Also confirmed the miscalibration isn't a fixable global bias: the overestimation factor varies
1.9x-205.8x across terms (not a constant multiplier), so no single shrinkage correction on `pi_spy`
can fix every term at once — some would stay badly overestimated, others would flip to
underestimated (the more dangerous failure mode nnPU's `max()`-based asymmetric calibration was
built to avoid in the first place). The real fix is a genuine held-out spy retrain (pipeline step
2's original, not-yet-built design), not a scalar patch.

**Decision (user-confirmed): shelve `--nnpu` for real training until a better-calibrated prior
estimator exists.** The nnPU/KM2/spy machinery stays in the codebase (verified correct against the
paper's math at both boundary conditions above) — this is a data/calibration problem, not an
implementation one — but `--nnpu` should not be used for real training runs until this is
revisited. See [[project_pu_learning_nnpu_km2]] for the memory-level version of this decision.

**v1.1, 2026-09-15: KM2 empirically failed at real scale, dropped from the calibration formula.**
Ran extraction (54,395 proteins, `pretrain-reduced-full-protein-v8` checkpoint) and KM2 estimation
(437/668 eligible terms) at full scale for the first time. `pi_km2` came back within {0.12, 0.14,
0.16} for 435/437 terms regardless of true prevalence (ranging 0.2%-3.7%) — verified this is not a
bug (batched-GPU rewrite matches the original scalar loop to 1e-9 on synthetic data) but a real
failure of the method on this data: D(kappa) decreases monotonically all the way to kappa=1 for
every term checked, in both the pooled-embedding and classifier-probability feature spaces, with
near-identical curves for terms 8x apart in true prevalence. See "v1 implementation notes" below
for the full diagnosis. `training_run.py`'s calibration now uses `pi_spy` alone (dropped `pi_km2`
from the `max()`), a user-confirmed decision, not a silent revert — see
[[project_pu_learning_nnpu_km2]].

**v1 implemented 2026-09-13** (same day as the revision below). All three pipeline pieces exist and
have been smoke-tested end-to-end on CPU (mechanism only — not yet run at real scale, see "v1
implementation notes" below):
- `experiments/scripts/extract_protein_embeddings.py` — pooled per-protein embeddings + classifier
  probabilities from a trained baseline checkpoint (pipeline steps 1-2).
- `experiments/scripts/estimate_class_priors_km2.py` — per-term KM2 class prior + Elkan-Noto spy
  cross-check, cached to `class_priors_km2.json` (pipeline step 3).
- `training_run.py --nnpu --class-priors-path ...` — per-label nnPU risk (new `nnpu_loss()`
  function), replacing BCE only for labels with a cached, above-floor prior; everything else keeps
  today's behavior unchanged (pipeline step 4). Verified the non-`--nnpu` path is byte-for-byte
  unaffected (separate smoke test, same config minus `--nnpu`, matching loss/accuracy scale).

Originally written 2026-09-13 following `dataset_refinement_pilot_summary.md`'s
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

## v1 implementation notes (disclosed deviations from the plan above)

Three things came up building v1 that revise or approximate what's written above — flagged here
explicitly rather than left implicit, since [[project_pu_learning_nnpu_km2]]'s memory entry warns
against silently reversing this plan's decisions:

- **KM2's mixture set is a shared, capped random subsample (default 5000 proteins), not the full
  ~54k-protein population per term.** The earlier note above ("KM2 runs at full protein-set scale
  per term, not a capped subsample") turns out to be computationally infeasible as literally
  stated: an exact per-term QP needs an O(n^2) kernel matrix, and 5000^2 already sits at the edge
  of what's cheap to compute 668 times — 54,395^2 (~2.96 billion entries) per term, 668 times over,
  is many orders of magnitude beyond reach on this CPU or a GPU. The fix that keeps this a
  *disclosed revision within KM2* rather than a silent swap to a cheaper method: the mixture
  sample is the same one draw reused across every term (not resampled per term), so its O(n^2)
  kernel matrix (`K_MM`) is computed exactly once and every term's QP scan reuses it — a shared
  subsample is a standard Monte-Carlo approximation KM2 itself tolerates (larger n -> lower
  variance, not a different assumption), not a different, cheaper estimator the way TIcE would be.
  `--mixture-sample-size` is exposed if more compute should be spent here.
- **kappa\* selection uses a two-segment elbow fit to the D(kappa) curve, not the paper's exact
  statistical slope/threshold test** ("gradient thresholding"). Implementable directly from the
  paper's own theory would need its specific Rademacher-complexity-derived threshold; the elbow fit
  is a practical stand-in with the same qualitative shape (flat near-zero segment, then a rising
  segment past kappa\*) but no statistical guarantee attached to where it draws the line. Worth
  revisiting with real per-term data once a full run is done — see "Key risks" below.
- **The Elkan-Noto spy cross-check (`pi_spy`) is computed from the SAME baseline checkpoint the
  spy positives were trained on** — there's no held-out-per-term retraining in v1 (that's pipeline
  step 2's "per-term spy holdout with k-fold", not built; expensive at 668-term scale, since it
  implies up to 668 separate retrains or a shared multi-fold scheme not yet designed). This makes
  `pi_spy` optimistic and not a rigorous second estimator yet — the code and cache output both flag
  this, and `flagged_scar_violation` (large `pi_km2`/`pi_spy` disagreement) should be read as "worth
  a second look", not as a validated SCAR-violation detector.

- **KM2 empirically fails on this checkpoint's embeddings — verified at full scale, not a bug
  (2026-09-15).** `extract_protein_embeddings.py` and `estimate_class_priors_km2.py` were both run
  for real: 54,395 proteins (`pretrain-reduced-full-protein-v8`, GPU, ~29min) and 437/668 eligible
  terms (`--min-term-positives 100`, GPU, 38s — the QP was rewritten as a batched-across-kappa torch
  op for this, see the script's `frank_wolfe_qp_batched`, since the original per-kappa Python loop
  was CPU-only and estimated at >10min for a full run). Result: `pi_km2` landed on {0.12, 0.14, 0.16}
  for 435/437 terms, with no visible relationship to true per-term prevalence (0.2%-3.7% across
  examples checked). Diagnosis, confirmed with direct evidence before concluding it wasn't a
  refactor bug:
  - Re-ran the original scalar per-kappa Frank-Wolfe loop against the new batched-torch version on
    synthetic data — outputs matched to 1e-9. The rewrite is correct; the estimator itself is not
    finding signal.
  - Extended `--kappa-max` from 0.3 to 1.0 for two real terms (GO:0000122, true prevalence 2.04%;
    GO:0008584, true prevalence 0.24% — 8.5x rarer): D(kappa) decreases **monotonically all the way
    to kappa=1** for both, never plateaus, never rises. KM2 theory expects D(kappa) to stay near a
    noise floor for kappa <= true kappa\* then rise once kappa overshoots what the mixture can
    supply; that shape never appears here.
  - Tried the classifier's own 668-dim probability vectors (already extracted, presumably more
    label-discriminative than raw pooled backbone features) as the KM2 feature space instead of the
    pooled embeddings — same monotonic-decrease-to-1.0 failure.
  - The clinching evidence: the two terms' D(kappa) curves are nearly numerically identical at every
    kappa tested despite an 8.5x true-prevalence gap. If the method were tracking real per-term
    structure, curves for terms that different should look visibly different near their own kappa\*.
    They don't — meaning a term's "positive" protein set has almost the same mean embedding as a
    random subset of the whole population, in both feature spaces tried. KM2's core assumption
    (mean-embedding separability between the positive subpopulation and the general mixture) doesn't
    hold for this checkpoint/embedding combination, at least not via simple RBF kernel mean-matching
    with a median-heuristic bandwidth.
  - **Decision (user-confirmed, not a silent revert)**: `training_run.py`'s calibration now uses
    `entry["pi_spy"]` alone, dropping `pi_km2` from the `max()`. `pi_km2` stays in
    `class_priors_km2.json` for reference/future diagnosis; it's just not consumed by training
    anymore. This makes the Elkan-Noto spy mean's known optimism bias (same checkpoint the spy
    positives were trained on) the sole source of `pi`, no longer cross-checked or floor-raised by a
    second estimator — worth revisiting (different bandwidth, different embedding/pooling strategy,
    or abandoning kernel-mean-based estimation for this checkpoint family entirely) if nnPU training
    with spy-only priors underperforms. **It did — see the v1.2 note at the top of this file.**

**Done, negative result**: pipeline step 6 (the `--run-diagnostic` AUROC/F1 comparison of `--nnpu`
against the plain-BCE baseline, this plan's actual success criterion) has now been run — see the
v1.2 note at the top of this file. `--nnpu` is shelved pending a better-calibrated prior estimator.

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
