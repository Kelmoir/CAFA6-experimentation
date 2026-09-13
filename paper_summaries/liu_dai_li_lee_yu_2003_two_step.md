# Liu, Dai, Li, Lee & Yu (2003) — *Building Text Classifiers Using Positive and Unlabeled Examples* (ICDM)

**Full text available:** Only partially (the IEEE conference volume itself is paywalled; summary
based on author-preprint excerpts, abstract, and several secondary citations/reviews that
consistently reproduce the core algorithm and results).
**Relevance to this project:** Formalizes the **"Two-Step" framework**, which the plan explicitly
calls "the canonical two-step reliable-negative-extraction-then-classify method direction A
follows" — this paper *is* the canonical systematization of that framework.

## Core contribution
The paper recognizes that three independently developed PU methods (S-EM, PEBL, Roc-SVM) up to
that point followed the same pattern, and explicitly decomposes it into two interchangeable steps:

- **Step 1 — reliable-negative extraction:** identify a subset RN of the unlabeled set U that is
  negative with high confidence.
  - S-EM: spy technique (see Liu et al. 2002)
  - PEBL: 1-DNF (identify "strong positive features"; documents lacking these features are
    marked as RN)
  - Roc-SVM: Rocchio algorithm (prototype vectors for P and U, similarity comparison)
  - This paper itself additionally introduces a **Naive Bayes variant** for step 1.
- **Step 2 — iterative classifier + model selection:** repeatedly train and refine a classifier
  from P and RN (plus the remaining U).
  - S-EM: EM with Naive Bayes
  - PEBL/Roc-SVM: iterative SVM (RN as negative, P as positive; in each round, newly classified
    U elements move to RN, until U stops changing or an iteration limit is reached)
  - PEBL simply takes the last converged classifier (per this paper, often a poor choice);
    S-EM/Roc-SVM have explicit selection criteria.

The paper systematically tests all 2×2(+) combinations of Step-1 and Step-2 methods ("LPU"
benchmark) and shows that the choice of combination strongly affects performance.

## Own contribution: Biased-SVM
Instead of the Two-Step pattern, the authors propose a more direct, "more principled" approach: an
SVM formulation that treats all of U as (noisily) negative, but assigns **different
misclassification costs/penalty weights** for P vs. U (asymmetric slack-variable weighting,
similar to cost-sensitive SVM). This avoids the need to explicitly and discretely extract RN — the
"hardness" of the negative assumption is instead controlled continuously via a weight. According
to the abstract and citing overview papers, Biased-SVM experimentally outperforms the tested
Two-Step combinations.

## Direct relevance to the plan
- Supplies the terminology ("Step 1: RN extraction", "Step 2: iterative classification") that the
  plan adopts wholesale for Direction A — including the idea of treating RN as hard negative
  while leaving the rest (U) marked as uncertain/soft.
- Important nuance for the plan: **the choice of Step-1 method is not neutral** — this paper shows
  empirically that different RN-extraction procedures (spy vs. 1-DNF vs. Rocchio vs. NB) lead to
  noticeably different downstream quality. This supports point 7 of the plan ("Success criterion
  is downstream model quality") as the right test, not just the spy leak rate itself.
- The **Biased-SVM alternative** hints at a possible alternate path for the plan: instead of
  discrete three-class target values (reliable-negative/ambiguous/reliable-positive), one could
  instead use continuous, calibrated example weights in the loss — conceptually closer to the
  later uPU/nnPU loss approaches (Kiryo et al. 2017, du Plessis et al. 2014/2015) than to the
  discrete tiering the plan currently envisions.
