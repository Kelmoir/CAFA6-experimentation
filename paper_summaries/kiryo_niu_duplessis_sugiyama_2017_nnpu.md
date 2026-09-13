# Kiryo, Niu, du Plessis & Sugiyama (2017) — *Positive-Unlabeled Learning with Non-Negative Risk Estimator* (NeurIPS/NIPS)

**Full text available:** Yes (arXiv 1703.00593, read in full).
**Relevance to this project:** The plan describes this as "the most commonly used modern PU loss
if this plan moves beyond post-hoc label correction into training-time PU loss directly" — the
most practically relevant reference should Direction A/B move beyond discrete label tiering.

## Starting problem: uPU overfits with flexible models
The unbiased PU (uPU) risk estimator (du Plessis et al. 2014/2015; see own summary) is unbiased in
expectation, but formally decomposes into three partial risks, including a **negative** term
(π_p · R⁻_p(g), the "positive-as-negative" risk) that compensates for the bias caused by treating
unlabeled-as-negative:

  R̂_pu(g) = π_p·R̂⁺_p(g) − π_p·R̂⁻_p(g) + R̂⁻_u(g)

If the model is very flexible (e.g., deep neural networks) and the loss function is not bounded
above, this empirical estimator can become **arbitrarily negative** — even though the true risk
R(g) must be ≥ 0. The paper shows experimentally (MNIST/MLP): uPU's training loss decreases without
bound while the test loss rises again — classic, severe overfitting.

## Solution: non-negative risk estimator (nnPU)
A simple but effective correction: force the potentially-negative partial term to be ≥ 0 via
max(0, ·):

  R̃_pu(g) = π_p·R̂⁺_p(g) + max{0, R̂⁻_u(g) − π_p·R̂⁻_p(g)}

This makes the estimator slightly **biased**, but the bias decays **exponentially** with growing
sample size (Theorem 2), the estimator remains consistent, and under mild conditions the mean
squared error (MSE) even decreases relative to uPU (Theorem 3). For large-scale training, a
mini-batch algorithm with a tolerance parameter β is proposed (Algorithm 1): as long as the
potentially negative part per batch doesn't fall below -β, optimization proceeds normally;
otherwise, a correction step in the opposite direction with reduced step size γ is taken to make
the batch "less overfitted."

## Empirical results
On MNIST, epsilon, 20News, CIFAR-10 (with MLPs/CNNs): uPU consistently overfits, nnPU fixes the
problem and in several cases even outperforms full PN classification (given limited real negative
examples). Also tested: **sensitivity to misspecified class prior π_p** — underestimating π_p
consistently hurts more than overestimating it ("under-misspecification hurt more than
over-misspecification"), and slight overestimation (1.1× π_p) was even optimal in several
datasets, since it pushes nnPU further away from uPU.

## Direct relevance to the plan
- Is the natural **training-loss replacement** if the plan wants to build a PU loss directly into
  `training_run.py` instead of a discrete three-tier label scheme — especially relevant because the
  target model here is presumably deep/flexible (not linear), i.e., exactly the regime in which
  pure uPU fails per this paper.
- The empirical observation "underestimating π hurts more than overestimating" is a concrete,
  actionable detail for the plan's planned class-prior estimation step: given uncertainty about
  the true per-term positive rate, a slightly **too-high** π estimate is the safer error direction
  than too low — relevant when choosing between Elkan-Noto's `c` and the more robust KM1/KM2 or
  TIcE estimators (see below), if these systematically bias differently.
- As with du Plessis 2014/2015, here too π_p is **assumed known** (not co-estimated) — nnPU's
  quality depends entirely on a separately good class-prior estimator. This confirms that Step 4
  of the plan (class-prior/`c` estimation) is the actual prerequisite for any of the loss-based
  methods discussed here, not an optional add-on.
