# Youngs, Penfold-Brown, Drew, Shasha & Bonneau (2013) — *Parametric Bayesian Priors and Better Choice of Negative Examples Improve Protein Function Prediction* (Bioinformatics)

**Full text available:** Partially (open access per PMC/PubMed, but automated retrieval was
blocked by a reCAPTCHA; summary based on the abstract, the freely accessible OUP-Bioinformatics
excerpt page with method sections, and several secondary citations that consistently describe the
algorithm — "ALBias").
**Relevance to this project:** The plan calls this "the closest existing precedent for direction
A" — i.e., the most direct prior work on the actual GO/protein function prediction problem under
exactly the same OWA problem the plan addresses.

## Problem
GO annotation databases contain almost no explicit negative examples (proteins known *not* to have
a given function) — but most machine learning methods for function prediction (particularly
GeneMANIA) require a negative set. Prior practice: coarse heuristics (e.g., "everything unannotated
counts as negative," or using sibling terms in the ontology tree as a source of negatives — the
latter is itself error-prone per secondary sources, since a gene can well carry several sibling
terms simultaneously once more experimental evidence accumulates).

## Method: ALBias (parameterizable Bayesian prior)
Instead of a hard heuristic, a **continuous "label bias"** value in [-1, 1] is computed for every
gene-function pair: -1 = certainly negative, +1 = certainly positive, values in between = prior
belief about the likelihood of the gene having the function in question. This bias is computed
**from the conditional probability with which an annotation a is observed jointly with another
annotation m on the same gene** — across all three GO branches (BP/MF/CC), not just within one
branch or ontology neighborhood. Genes that have no annotation in any of the three branches from
which a prior could be derived remain neutral (white/no bias).
- **Negative examples** are then concretely defined as: genes with an annotation in the same GO
  branch as the target term, whose computed prior score for the target term is 0 — capped at a
  maximum of 50% of the non-positive genes for that function (prevents overweighting).
- The resulting bias vector is fed directly into GeneMANIA's network combination optimization (as
  a prior/regularization term in the discriminant-vector optimization problem), rather than
  serving merely as an upstream label filter.
- Additionally: new parameter-tuning procedures for GeneMANIA's own hyperparameters that exploit
  the structure of the optimization problem.

## Results
Tested on yeast and mouse proteomes; per the abstract, consistently better accuracy than the
then-top methods for function prediction, across all tested metrics.

## Direct relevance to the plan
- Is essentially a **domain-specific reliable-negative heuristic**, conceptually related to
  Direction A of the plan, but **without** an explicit PU framework, spies, or class-prior
  calibration — ALBias instead uses co-annotation patterns (which GO terms are observed jointly on
  the same gene) as a proxy signal for "likely negative," not a trained classifier ranking as the
  plan envisions.
- Important methodological contrast: Youngs et al. produce a **continuous bias** fed directly into
  the predictor (similar to a soft target/prior term), whereas the plan proposes a **discrete
  three-tier scheme** (hard negative / ambiguous / hard positive). The ALBias result suggests that
  a continuous, calibrated bias term may work just as well or better than a hard tier boundary — an
  alternative design the plan could consider.
- The same team's follow-up work (Youngs, Penfold-Brown, Bonneau & Shasha 2014, "NoGO Database,"
  PLoS Comput Biol) explicitly compares ALBias against adapted text-classification PU methods
  (including the Two-Step framework of Liu et al. 2003) under a **temporal holdout** (annotations
  that appeared later serve as ground truth for "was actually positive") — this is a concrete
  **evaluation methodology**, already proven in the GO context, that the plan could directly adopt
  for its own Step 7 ("Success criterion is downstream model quality") instead of devising a new
  evaluation strategy.
