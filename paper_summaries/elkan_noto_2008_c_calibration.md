# Elkan & Noto (2008) — *Learning Classifiers from Only Positive and Unlabeled Data* (KDD)

**Full text available:** Yes (cseweb.ucsd.edu, author PDF).
**Relevance to this project:** Directly supplies the **`c` calibration constant** that Step 4 of
the plan uses ("Elkan & Noto's `c` = mean predicted probability on held-out positives") — including
the application domain (an incomplete specialized biological database), very close to the GO
problem.

## Setting & core assumption (SCAR)
Formalizes the problem with a random variable s ∈ {0,1}: s=1 means "labeled" (⇒ y=1 is certain),
s=0 means "unlabeled" (y can be 0 or 1). Central assumption, here named explicitly for the first
time **"selected completely at random" (SCAR)**:

> p(s=1 | x, y=1) = p(s=1 | y=1) = c

I.e., among all truly positive examples, a fixed constant c, independent of x, is actually labeled
— exactly the SCAR assumption the plan itself flags as questionable for GO annotation under "Key
risks" (see Bekker & Davis 2020 below).

## Core result (Lemma 1)
If you train an ordinary classifier g(x) on "labeled vs. unlabeled" (instead of "positive vs.
negative"), then under SCAR:

  **p(y=1 | x) = p(s=1 | x) / c = g(x) / c**

The classifier trained on P-vs-U thus estimates the true positive probability correctly up to a
**constant, global scaling factor** — a remarkably simple result with large practical
consequences: you don't need to train specially for PU; you only need to divide by c afterward (or
equivalently, set the decision threshold to 0.5·c instead of 0.5).

## Estimating c
On a validation set with known positives P: g(x) for x ∈ P will, on average, estimate exactly c
(because for x ∈ P, p(s=1|x) = p(s=1|y=1) = c). Three estimators are proposed:
- **e1** (recommended): mean of g(x) over x ∈ P — exactly what the plan references as the
  "simplest starting point."
- **e2**: sum of g(x) over P, normalized by the sum of g(x) over the entire validation set.
- **e3**: maximum of g(x) over the validation set (upper bound, high variance, worst choice per
  the paper).
e1 has the lowest variance and is the empirically recommended choice.

## Second method: weighted retraining
Alternative to pure rescaling: each unlabeled example is duplicated — once as positive with weight
w(x) = (1-c)/c · g(x)/(1-g(x)), once as negative with weight 1-w(x) — and a second classifier is
trained on this. This is more rigorously grounded than naive "unlabeled = negative," but more
expensive (two training runs).

## Experiments
Real-world data: proteins from a specialized transporter database (TCDB) as positives, the rest of
SwissProt as unlabeled — structurally almost identical to the GO/CAFA problem (curated, incomplete
functional annotation vs. a large unlabeled remainder). Both new methods beat the then-best
method, **Biased-SVM** (from Liu et al. 2003), both in accuracy and drastically in runtime
(Biased-SVM: 621× slower, since it requires a grid search over weighting hyperparameters; the c
calibration needs no such search).

## Direct relevance to the plan
- Is the explicit source for Step 4 of the plan (`c` as calibration constant).
- Important caveat the plan should carry over: **c here is a single global constant**, estimated
  from a validation set of positives. The plan, however, wants **per-term** calibration — Elkan &
  Noto's e1 estimator can certainly be applied per-term directly (mean of g(x) over that term's
  spies), but the paper itself does not discuss how robust that is with small samples (rare
  terms). This aligns with the open point the plan itself names ("Per-term vs. global
  calibration... raises a sample-size problem for rare terms") and explains why the plan wants to
  fall back on more robust mixture-proportion estimators (Ramaswamy et al., Bekker & Davis TIcE)
  when `c` looks unstable.
- The SCAR assumption of this paper is exactly the assumption that Bekker & Davis (2020) later
  classify as a special case of SAR (Selected At Random) and whose violation for GO is plausible
  (well-studied organisms/proteins get preferentially annotated) — the plan already correctly
  cites this as a risk.
