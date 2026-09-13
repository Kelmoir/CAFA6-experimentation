# Ramaswamy, Scott & Tewari (2016) — *Mixture Proportion Estimation via Kernel Embeddings of Distributions* (ICML)

**Full text available:** No (ICML/PMLR full text not fetched); summary based on the abstract, a
detailed secondary review (Liner/Lacuna), and several consistent citation sources.
**Relevance to this project:** Source of the **KM1/KM2 estimators**, which the plan names as a
more robust alternative to Elkan & Noto's simple `c` mean, "if `c` looks unstable per term."

## Problem statement: Mixture Proportion Estimation (MPE)
A more general problem than PU-specific calibration: given samples from a mixture distribution
F = κ·H + (1-κ)·G (H = a component distribution, known from its own samples; G = "the rest") and
samples from F, estimate the mixture weight κ. PU learning is a special case: the unlabeled
distribution is a mixture of the positive distribution (weight = class prior π) and the negative
distribution — π **is** exactly the κ of this more general problem.

## Contribution
Per the abstract, **no algorithm with a proven convergence rate** existed for MPE in the general,
non-parametric case before this paper. It:
- embeds the distributions into an RKHS (reproducing kernel Hilbert space) via **kernel mean
  embeddings**,
- reduces the estimation to repeatedly solving a **simple convex quadratic program**,
- proves a convergence rate of **O(m^(-1/2))** under certain separability assumptions
  (m = sample size) — the first method with such a guarantee for this problem.
- Two concrete variants: **KM1** ("value thresholding") and **KM2** ("gradient thresholding"); KM2
  is reported in secondary sources to be considerably **more robust to the choice of the internal
  threshold hyperparameter** ν than KM1.

## Experiments
Six standard benchmarks (waveform, mushroom, pageblocks, shuttle, spambase, digits): KM1/KM2
perform comparably or better than existing methods (alphamax, an Elkan-Noto-like "EN", ROC-based
estimators). Weaker on "mushroom" — a secondary source notes that a closer investigation would be
needed for when the underlying separability assumption is violated in practice.

## Direct relevance to the plan
- Provides the **theoretically most rigorous** alternative to Elkan & Noto's simple mean-`c`, with
  an actual convergence guarantee rather than just a plausible heuristic — relevant exactly when
  (as feared by the plan under "Key risks") per-term spy sets are small and a simple mean
  estimator becomes unstable.
- Important caveat for transferring this to the plan: KM1/KM2 are designed for **general,
  continuous feature spaces with kernel similarity** (RKHS embeddings of distributions over x) —
  this requires enough *feature vectors* of the respective protein representation for each GO
  term to estimate a meaningful kernel embedding. For very rare terms (few positive examples),
  KM1/KM2 could hit the same sample-size limit the plan already flags as an open issue — and the
  computational cost (repeated QP per term) for 668 terms separately is non-trivial and should
  feed into the compute-budget estimate the plan already calls for.
- Practically most sensible as a **fallback/robustness check**: try the simple per-term Elkan-Noto
  `c` first, and only fall back to KM1/KM2 or TIcE (Bekker & Davis 2018, see own summary) for
  unstable/noisy estimates (small terms) — exactly as the plan itself suggests.
