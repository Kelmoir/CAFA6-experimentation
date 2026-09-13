# Bekker, Robberechts & Davis (2019) — *Beyond the Selected Completely At Random Assumption for Learning from Positive and Unlabeled Data* (ECML/PKDD)

**Full text available:** Yes (arXiv 1809.03207, read in full).
**Relevance to this project:** The direct follow-on to Bekker & Davis (2018, TIcE) that the plan's
"Key risks" section pointed to as investigating what happens when SCAR is violated for PU learning
— fetched and summarized here specifically to answer "what mitigations exist for SAR" for
`experiments/pu_learning_plan.md`.

## Formalizing SAR: the propensity score

Extends Elkan & Noto's SCAR/MCAR analogy into a full taxonomy of PU labeling mechanisms (mapped
onto the missing-data literature's MCAR/MAR/MNAR categories), using the same single-training-set
triples (x, y, s) as Bekker & Davis (2020):

- **SCAR**: labeling probability is constant, independent of x and of the true class probability.
- **SAR** (this paper's focus): labeling probability depends on x, but — given x — not on the true
  probability of being positive. Formalized via a **propensity score**, borrowed from causal
  inference but conditioned differently:

  **Definition 1**: e(x) = Pr(s=1 | y=1, x) — the probability a truly positive instance with
  attributes x actually gets labeled.

- **SNAR** (Selected Not At Random): the residual, worse case — labeling probability depends on the
  *true* class probability itself, even after conditioning on x (e.g., a borderline-interesting web
  page is labeled inconsistently across near-identical instances). The paper notes all three PU
  mechanisms are technically MNAR in the missing-data sense (since only positives can ever be
  labeled at all), unlike the ordinary MCAR/MAR/MNAR taxonomy.

## Mitigation 1: propensity-weighted empirical risk (when e(x) is known)

Generalizes inverse-propensity-scoring (standard in causal inference/recommender systems) to PU
data, where — unlike those settings — **negative examples always have exactly zero propensity**
(you never observe a negative label at all), so ordinary IPS doesn't directly apply. The fix:
split each observed-positive example's evidence between a positive and a phantom-negative copy:

  R̂(ŷ|e,s) = (1/n) Σᵢ [ sᵢ·( (1/eᵢ)·δ₁(ŷᵢ) + (1 − 1/eᵢ)·δ₀(ŷᵢ) ) + (1−sᵢ)·δ₀(ŷᵢ) ]

Intuition: a labeled example with propensity eᵢ represents 1/eᵢ true positives in expectation, of
which 1/eᵢ − 1 didn't get labeled — so it's weighted 1/eᵢ as a positive and (1 − 1/eᵢ) as a
negative, while every unlabeled example is still weighted fully as negative. **Proven unbiased**
(Definition 2), with a Hoeffding-style concentration bound (Proposition 3) and an ERM
generalization-error bound (Proposition 4) — i.e., if e(x) is known exactly, this is a
theoretically clean drop-in replacement for plain "unknowns-as-negative" training.

**Bias when e is only estimated (Proposition 5)**, the practically relevant case:

  bias(R̂(ŷ|ê,s)) = (1/n) Σᵢ yᵢ·(1 − eᵢ/êᵢ)·(δ₁(ŷᵢ) − δ₀(ŷᵢ))

Two concrete, actionable readings: (1) **propensity accuracy only matters for truly positive
examples** — errors on negatives don't enter the bias term at all; (2) **underestimating e is worse
than overestimating it** — a too-low ê makes the model believe positives are more prevalent than
they are, inflating the bias term. This is the propensity-score analog of the π̂ ≤ π/2 collapse
risk from du Plessis (2014)/Kiryo (2017): the same "bias high, not low" asymmetry this plan already
adopted for π_term should also apply if per-instance propensity weighting is ever used.

**Explicit correctness pitfall flagged by the paper**: naively minimizing the *expected* risk (using
current model estimates in place of the true labels, rather than this unbiased estimator) is
*not* valid ERM — it isn't unbiased, and is trivially minimized by predicting every instance
positive (expected risk of exactly 0). Worth remembering if a "just use the classifier's own
current probabilities as weights" shortcut is ever tempting.

## Mitigation 2: SAR-EM (when e(x) is unknown — the realistic case)

Two options are analyzed for the case that actually matters here (e(x) is never known in advance):

**2a. Reduce SAR to SCAR** by assuming e(x) depends only on a strict *subset* of attributes (the
"propensity attributes" xₑ), i.e. e(x) = e(xₑ). This is necessary for identifiability at all — the
propensity function must depend on fewer/different attributes than the classifier, or there's no
way to distinguish "wasn't labeled because of low propensity" from "wasn't labeled because it's
probably negative." If xₑ is discrete with finitely many configurations, the data can be
stratified — one constant propensity per stratum, solvable with standard SCAR PU techniques per
stratum. **Explicitly called impractical** by the authors: fails for continuous xₑ, the number of
strata explodes combinatorially as |xₑ| grows, information is lost by partitioning, and neither the
classifier nor the propensity function gets to be smooth across strata.

**2b. SAR-EM** (the paper's actual proposal): jointly learn the classifier f̂ and a *lower-dimensional*
propensity model ê via EM, instead of hard-partitioning by xₑ — this relaxes 2a's strict
requirement (only needs xₑ's assumption to hold for *most* configurations; model smoothness covers
the rest). One iteration:

- **E-step**: ŷᵢ = Pr(yᵢ=1 | sᵢ,xᵢ,f̂,ê) = sᵢ + (1−sᵢ)·[f̂(xᵢ)(1−ê(xᵢ))] / [1 − f̂(xᵢ)ê(xᵢ)]
- **M-step**: refit f on every example counted twice (once positive-weighted by ŷᵢ, once
  negative-weighted by 1−ŷᵢ — ordinary soft self-training), and refit e on every example counted
  once (positive if actually labeled, negative if not — weighted by ŷᵢ, since only examples
  believed truly positive are informative about propensity).
- **Initialization**: f starts from the naive balanced model (unlabeled = negative); e starts from
  that model's probability estimates on the unlabeled set.
- **Convergence**: plain log-likelihood convergence isn't sufficient (it can plateau before the
  propensity model has converged) — the paper tracks the trend/slope of e's predictions over the
  last n iterations instead.
- **Known weakness**: works best when the classes are actually separable (it's optimizing log
  loss); if not, the resulting classifier isn't guaranteed optimal for the *learned* propensity
  model, and the paper recommends a follow-up retraining pass using the propensity-weighted risk
  estimator (Mitigation 1 above) as a correction step.

## Empirical results: the critical finding for this plan

8 real-world datasets (text/image/tabular, base rates 0.11–0.55), with **synthetic SAR labeling**
constructed via clustering + per-cluster propensity assignment (4 propensity attributes combined
geometrically, propensities ∈ [0.2, 0.8]). Compared against SCAR-KM2, SCAR-TIcE, and a Naive
(unknowns-as-negative) baseline, all as logistic regression:

- SAR-EM significantly beats all three baselines on ROC-AUC (Friedman + Nemenyi, p<0.01) — the
  expected headline result.
- **The finding that matters most for this plan**: KM2 and TIcE — i.e., exactly the SCAR-style
  class-prior correction this plan's chosen approach relies on — **sometimes performed worse than
  doing nothing at all** (the Naive unknowns-as-negative baseline) on genuinely SAR-violated data.
  The paper's explanation: KM2/TIcE optimize for getting the *overall* positive/negative ratio
  right; when they pick the wrong region of instance space to hit that ratio (because the true
  labeling mechanism isn't the constant they assumed), the correction introduces **both** new false
  positives and false negatives, whereas the conservative Naive baseline only ever contributes false
  negatives. Applying a class-prior correction under real SAR is not merely "less optimal than it
  would be under SCAR" — it can actively make the classifier worse than `training_run.py`'s current,
  uncorrected behavior.
- Propensity-score recovery (SAR-EM) was accurate (MSE < 0.1, often near 0) on 7/8 datasets; the one
  exception, Mushroom, is the same dataset flagged as a weak point for KM1/KM2 in Ramaswamy et al.
  (2016) — both likely tied to that dataset's near-perfect class separability.
- Number of propensity attributes (2 vs. 4) had no measurable effect on either classification or
  propensity-recovery quality.

## Direct relevance to the plan

- **Raises the stakes on the plan's existing "Key risks" SCAR/SAR item**: this isn't just a
  theoretical caveat about a weaker guarantee — the paper's own empirical result shows a KM2-style
  class-prior correction can underperform the current do-nothing baseline under real SAR. This
  makes the plan's step 6 success criterion (compare `--run-diagnostic` against the current
  hard-0-everywhere baseline before adopting anything) not just good practice but the **only**
  safety net standing between this plan and a regression — treat any nnPU+KM2 run that doesn't
  clearly beat the baseline as a signal to suspect SAR, not just a wash.
- **A concrete, more targeted mitigation than the plan's current spy-only cross-check**: if the spy
  cross-check (KM2 estimate vs. Elkan-Noto e1 spy-mean) shows material per-term disagreement,
  the propensity-weighted ERM estimator (Mitigation 1) offers a way to incorporate *known*,
  observable bias covariates directly — e.g. species, evidence-code mix, curating
  database/annotator, or a protein's total annotation count as a proxy for "how well-studied is
  this protein" (the exact mechanisms Gaudet & Dessimoz 2017 document as real GO biases) — rather
  than assuming one constant π_term per term. This only requires that the propensity model's inputs
  be a *different, smaller* set of attributes than the classifier's (identifiability requirement
  from §5.1) — a real design constraint if this is ever built, since the classifier already
  consumes the full ESM protein embedding.
- **SAR-EM itself is a plausible escalation path**, not just a diagnostic: if the spy cross-check
  keeps flagging disagreement for a meaningful fraction of terms, SAR-EM's joint (f, e) training
  could replace the plan's "train once, estimate π via KM2 once" pipeline for those specific terms
  — at real added engineering and compute cost (an EM loop instead of one KM2 QP per term), which
  is consistent with this plan's general "accept the cost, don't take the cheap option" stance, but
  a substantially bigger lift than what's currently scoped (pipeline step 4). Not adopted as the
  default path in this revision — noted here as the next escalation if the spy cross-check /
  baseline comparison keeps showing trouble.
- **The "bias π high, not low" rule already in the plan generalizes**: Proposition 5's bias formula
  shows the same asymmetry for propensity scores as du Plessis/Kiryo showed for the class prior —
  independent confirmation (from a different derivation, in a different paper) that erring toward
  overestimating positivity is the safer direction whenever any of these PU corrections have to
  guess.
