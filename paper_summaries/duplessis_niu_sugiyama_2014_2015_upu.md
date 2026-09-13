# du Plessis, Niu & Sugiyama (2014, NeurIPS) & (2015, ICML) — uPU: Unbiased PU Risk Estimation

**Full text available:** Yes for 2014 (NeurIPS proceedings, read in full); for 2015 only abstract +
secondary citations (ICML proceedings full text not separately fetched, but content well
supported by multiple consistent citations and the direct follow-up paper Kiryo et al. 2017).
**Relevance to this project:** The plan cites this as "unbiased PU risk estimation (uPU), a more
principled alternative to naive unknowns-as-negative training" — the theoretical basis for *why*
pure "unknown = hard negative" (which `training_run.py` allegedly currently does, per the plan)
produces a systematic bias, and how to correct it directly in the loss without reshuffling labels.

## 2014 (NeurIPS): PU classification as cost-sensitive learning
Core idea: if the true class prior (class ratio) π = p(y=1) of the unlabeled distribution is
known, the PU risk can be reformulated exactly from P and U samples alone (never having seen
negative examples):

  R(f) = π·R₁(f) + (1-π)·R₋₁(f) = 2π·R₁(f) + R_X(f) − π

where R_X(f) is the misclassification rate on the *unlabeled* distribution (directly estimable)
and R₁(f) is that on the *positive* examples. This reduces PU learning to **cost-sensitive
classification** between P and U with explicit weights c₁ = 2π/η, c_X = 1/(1-η) derivable from π
(η = fraction of P examples in P∪U) — solvable with standard tools such as weighted SVMs.

**Central, non-trivial result — the convexity trap:** With the usual convex **hinge loss**, this
reformulation introduces a *"superfluous penalty"* term that can force a wrong decision boundary
even on perfectly separable data (in the extreme case: everything gets classified as positive once
π is large enough). The non-convex **ramp loss**, by contrast, is symmetric (ℓ_R(z)+ℓ_R(-z)=const)
and avoids the bias exactly — it reduces to the ordinary supervised risk. Experimentally (USPS
digits) confirmed: ramp loss clearly beats hinge loss, especially at large π.

**Sensitivity to π estimation error:** if π is only estimated (π̂ instead of the true π), the
analysis via the "effective class prior" π_e shows that the resulting error **stays small when π
is large** (dominant positive rate in U) — but **PU learning fails completely when π̂ ≤ π/2**. For
very small true class priors (as here: 0.556% observed positive rate), the method is therefore
sensitive to π estimation errors — exactly the regime the GO/CAFA case is in.

## 2015 (ICML): convex alternative — double-hinge loss
Since non-convex optimization (ramp loss) is hard to train in practice, this follow-up paper
constructs a **convex** loss function (the "double hinge loss": different hinge-like losses for P
and U examples) that still avoids the bias — with theoretical convergence guarantees (optimal
parametric rate), and per the abstract "as accurate as the non-convex method, with a much lower
computational cost." This is the practically usable uPU loss later adopted by Kiryo et al. (2017)
for deep networks.

## Direct relevance to the plan
- Explains **why** the previous document's pilot study (`dataset_refinement_pilot_summary.md`)
  likely ran into trouble: naive training with unknowns-as-hard-negative (hinge-like or
  cross-entropy losses without correction) produces, per this theory, a **systematic, not merely
  noisy** bias — not just "less signal," but a shifted decision boundary.
- A direct alternative to the plan's proposed **discrete three-tier scheme** (reliable-
  negative/ambiguous/reliable-positive as target values): instead of reshuffling label cells,
  `training_run.py` could instead incorporate a uPU/nnPU-style **loss term** that correctly weights
  U as a mixture of a π-fraction positive and a (1-π)-fraction negative — provided a robust
  per-term π estimator is available (this directly connects to Ramaswamy et al. 2016 and Bekker &
  Davis 2018 below).
- The π̂ ≤ π/2 threshold offers a **concrete, checkable warning** for the plan: with an observed
  positive rate of only 0.556%, per-term class-prior estimation must be exceptionally accurate, or
  else the method collapses entirely rather than degrading gracefully — another argument for the
  robustness check the plan already plans against Elkan-Noto's simple `c` estimator.
