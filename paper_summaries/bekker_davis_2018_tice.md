# Bekker & Davis (2018) — *Estimating the Class Prior in Positive and Unlabeled Data through Decision Tree Induction* (AAAI)

**Full text available:** No (AAAI proceedings full text not separately fetched); summary based on
the abstract (confirmed identically via Semantic Scholar, the AAAI page, and ACM citations) and
several secondary citations.
**Relevance to this project:** Source of the **TIcE estimator** ("Tree Induction for c
Estimation"), which the plan names as a second practical alternative to Elkan & Noto's simple `c`
mean.

## Core idea
The goal, as with Elkan & Noto, is to estimate the constant c = p(s=1|y=1) (the probability that a
truly positive example is actually labeled/annotated) — but **without** Elkan & Noto's implicit
assumption that a single global classifier score g(x) is equally well calibrated for all x.
Instead:

- Observation: for **any subset** of the feature space, the ratio (number of labeled examples in
  the subset) / (number of all examples in the subset) gives a **lower bound** on c.
- This lower bound gets tighter (closer to the true c) the higher the fraction of labeled examples
  is within the chosen subset — so one specifically searches for subsets that are particularly
  "pure" (positive-heavy).
- Such subsets can naturally be found via **top-down decision tree induction** (recursively
  splitting the feature space until a locally maximal label ratio is achieved) — hence the name
  TIcE.

## Result
Per the abstract: TIcE produces estimates that are **just as accurate** as the then-state-of-the-
art (including Elkan-Noto-like and KM-like methods), but is **an order of magnitude faster** —
relevant because the plan may need to run such an estimator separately **per term** (668 terms).

## Direct relevance to the plan
- Is the second concrete, practically usable reference (alongside Ramaswamy et al.'s KM1/KM2) for
  the case where Elkan & Noto's simple `c` mean is "unstable" per term — the plan names exactly
  these two as fallback options.
- The speed advantage ("order of magnitude faster") is a direct argument for preferring TIcE over
  KM1/KM2 when c must be estimated separately for all 668 GO terms — this directly addresses the
  point the plan itself raises: "Compute cost... needs its own budget estimate."
- Conceptual closeness to the actual downstream model: TIcE already uses a kind of feature-space
  partitioning that (unlike KM1/KM2's RKHS embeddings) doesn't necessarily require a good kernel
  choice — it could therefore be more robust/simpler to integrate if the project's protein
  representations (from `training_run.py`) don't readily embed into a meaningful RKHS.
- As with all class-prior estimators summarized here, TIcE also implicitly relies on the SCAR
  assumption (see Elkan & Noto, Bekker & Davis 2020) — the authors' own later work ("Beyond the
  Selected Completely At Random Assumption", Bekker, Robberechts & Davis 2019, ECML/PKDD, not part
  of the original list but a direct follow-on) explicitly investigates what happens when this
  assumption — as the plan suspects — is violated for GO annotation. This could be a useful
  pointer for later research in the plan.
