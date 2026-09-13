# Liu, Lee, Yu & Li (2002) — *Partially Supervised Classification of Text Documents* (ICML)

**Full text available:** Yes (cs.uic.edu, author PDF).
**Relevance to this project:** Source of the "spy" technique — exactly the mechanism the plan
(`pu_learning_plan.md`, Section A "Calibration via spies") adopts directly.

## Problem statement
Given a set **P** (positive documents on a topic) and a large set **M** ("mixed set": contains
more P documents mixed with negatives), with no document in M ever labeled negative. This is
"partially supervised classification" — essentially the same problem as Positive-Unlabeled (PU)
learning, just named before that term was established.

## Theoretical part
Shows formally (PAC-style bounds via VC-dimension) that in the *noiseless* case (a target
function exists that correctly classifies P), it suffices to choose a function that (a) correctly
classifies all P documents as positive and (b) minimizes the number of M documents classified as
positive — this yields, with high probability, small error if |P| and |M| are large enough. For
the *noisy* case (label noise, or the target function not contained in the function class), the
problem becomes: minimize positively-classified M documents subject to a recall target r on P.

This theory is **not directly practically solvable** (the constrained optimization problem is
hard) — hence the heuristic part.

## The practical algorithm: S-EM ("Spy EM")
Two steps, built on Naive Bayes + EM:

**Step 1 — reinitialization with spies:**
1. Randomly draw a fraction s (in the paper: 10%) of the P documents as "spies" and mix them into
   M. The spies behave (by assumption) identically to the real but unknown positive documents in
   M — they provide a **measurable, not assumed**, signal for how real positives behave in the
   unlabeled set.
2. Run I-EM (iterative EM with Naive Bayes) on P (all positive) and M (all initially negative) —
   2 iterations suffice per the paper.
3. Score every document in M, including the spies, with the resulting classifier.
4. Determine a threshold t from the spies' score distribution (not: the minimum spy score, since
   that is noise-sensitive — instead a chosen "noise level" l, such that l% of the spies fall
   below t; the paper tests l ∈ {0.1, 0.15, 0.2, 0.3} and finds the choice robust).
5. M documents below t → "likely negative" (RN), the rest → remain unlabeled (U).

**Step 2 — final classifier:** put the spies back into P; run EM again on P (fixed positive), RN
(initially negative, but changeable) and U (initially unlabeled, gets probabilistically labeled by
EM) — about 4 more iterations.

**Model selection across EM iterations:** since increasing likelihood in EM does not monotonically
correlate with decreasing classification error, a heuristic error estimate (based on the positive
rate in the unlabeled set) is used to pick the best iteration, rather than simply running to
convergence.

## Results
30 text datasets (20 Newsgroups, WebKB). S-EM clearly beats both naive NB (treating all of M as
negative) and I-EM without spies, especially on F-score; it even outperforms the "omniscient" best
I-EM iteration (which would not be selectable in practice, since it would require true negative
labels). Computational cost is linear in |P|+|M|, only 6 data scans total.

## Direct relevance to the plan
- The plan's "calibration via spies" idea (holding out real positives per term, relabeling them
  to 0, scoring them) **is** this spy technique, transposed from documents/words to
  (protein, GO-term) cells.
- However, the plan uses spies differently from here: not for EM reinitialization, but for
  **threshold calibration for "reliable negative"** directly on an already-trained model (no
  iterative EM) — conceptually closer to Step 1 of this paper, but without Step 2 (no
  reinitialization/retraining with RN as hard negative in the same round; instead a downstream
  label correction for the next training run).
- The noise-level trick described here for robust threshold selection (using a percentile of spy
  scores rather than the minimum) is directly relevant to Step 5 of the plan ("Set the
  reliable-negative threshold per term so the empirical spy leak rate stays below a fixed
  bound").
