# Yang, Li, Mei, Kwoh & Ng (2012) — *Positive-Unlabeled Learning for Disease Gene Identification* (Bioinformatics)

**Full text available:** No (full text not directly retrieved); summary based on the abstract
(confirmed identically across OUP/Bioinformatics, PubMed, ACM) and several secondary sources that
consistently describe the PUDI algorithm.
**Relevance to this project:** The plan describes this as "PU learning applied to a comparably
sparse, OWA-affected biological annotation setting... useful as a worked methodological analog" —
i.e., the second direct domain precedent (after Youngs et al.) for a PU framework on biological
annotation data.

## Problem
Disease gene identification: known disease genes form P, all other genes had previously been
naively treated as N (negative) — even though this necessarily still contains undiscovered disease
genes. Structurally identical to the GO problem: a curated but incomplete positive set, with a huge
"remainder" that is in truth mixed.

## Method: PUDI (PU learning for Disease gene Identification)
Core idea: **U is not decomposed into a binary split, but into four subsets** before the final
classifier is trained:
- **RN (reliable negative)** — high-confidence negative
- **LP (likely positive)** — likely positive
- **LN (likely negative)** — likely negative
- **WN (weak negative)** — weakly negative (residual category, least certain)

These four subsets are fed together with P into a **weighted multi-level SVM scheme** — per the
abstract, each subset receives its own weight/treatment in the SVM optimization, rather than (as
in classic Two-Step methods) distinguishing only two classes (RN vs. the rest).

## Result
Per the abstract, PUDI identifies disease genes more accurately because "the unknown" is treated
in a more differentiated (four-level) rather than binary manner — direct confirmation that a finer
partitioning of the unlabeled space (more than just "reliable negative vs. the rest") does bring
measurable benefits in a related, low-positive-rate bioinformatics domain.

## Direct relevance to the plan
- Is the **most direct existing precedent for a multi-tier scheme** (here: four levels) in a PU
  bioinformatics application — the plan itself currently proposes **three** tiers
  (reliable-negative / ambiguous / reliable-positive). PUDI suggests that finer granularity (e.g.,
  an additional intermediate tier between "ambiguous" and "reliable-negative") could bring
  measurable added value, though this should be weighed against added complexity (more thresholds,
  more calibration effort per term).
- Unlike the plan, PUDI uses **no spies and no class-prior calibration** — the four subsets are
  presumably formed via other confidence measures (similarity/score ranking). This means: PUDI
  itself does not validate its subsets with the "measured, not assumed" spy leak rate the plan
  intends — a methodological point where the plan explicitly improves on this precedent.
- A follow-up work by the same author group (Yang, Li, Chua, Kwoh & Ng 2014, "Ensemble Positive
  Unlabeled Learning for Disease Gene Identification," PLoS ONE) extends PUDI into an ensemble
  approach — if the plan later wants to improve robustness against individual misclassified tier
  assignments, this would be a natural follow-on reference.
