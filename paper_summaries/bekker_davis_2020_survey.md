# Bekker & Davis (2020) — *Learning from Positive and Unlabeled Data: A Survey* (Machine Learning journal)

**Full text available:** Yes (arXiv 1811.04820, identical to the journal version, read in full).
**Relevance to this project:** The plan calls this the "best single overview" and explicitly
invokes the **SCAR-vs-SAR distinction** from this paper to justify the annotation-bias risk in GO.
This survey is the central map into which all other papers summarized here fit.

## Formalization
PU data as triples (x, y, s): y = true label (unobserved except when s=1), s = "was labeled." Two
scenarios are distinguished:
- **Single-training-set**: one i.i.d. dataset, a fraction of the positives get labeled. (This is
  the scenario relevant to the plan: the GO matrix is a single dataset.)
- **Case-control**: P and U come from two independently drawn datasets (e.g., Elkan & Noto's
  TCDB-vs-SwissProt experiment). Important: c and the class prior α convert **differently**
  between the two scenarios (Equations 5–7 of the paper) — when transferring formulas from other
  papers (Elkan & Noto, KM1/KM2, etc.), one must always check which scenario underlies them.

## Central taxonomy of labeling-mechanism assumptions
- **SCAR** (Selected Completely At Random): e(x) = Pr(s=1|x,y=1) = c, constant for all x. Basis of
  almost all methods summarized here (Elkan & Noto, du Plessis et al., Kiryo et al., TIcE,
  KM1/KM2).
- **SAR** (Selected At Random): e(x) depends on x — a more general, more realistic assumption.
  Motivated by labeling bias in real applications (e.g., the position of a search-ad link
  affecting its click probability).
- **Probabilistic Gap (PGPU)**: intermediate stage — examples that "resemble both positive and
  negative examples" (small probabilistic gap Δ Pr(x) = Pr(y=1|x) − Pr(y=0|x)) are labeled less
  often. Allows formally proven extraction of **reliable** positives/negatives purely from the
  observed gap (a theorem-style derivation in the paper), **without** the full SCAR assumption.

For GO annotation, SAR (not SCAR) is the more realistic model: well-studied organisms/proteins are
systematically annotated more often — exactly the risk the plan itself names under "Key risks,"
now with an exact name and formalism.

## Method landscape (3 main categories)
1. **Two-Step techniques** (Separability + Smoothness assumption): Step 1 extract reliable
   negatives (Spy/S-EM, 1-DNF/PEBL, Rocchio/Roc-SVM, k-means, kNN, DILCA, PGPU, and many more — the
   survey lists more than ten variants in a table), Step 2 (semi-)supervised classification, Step 3
   optional model selection (ΔE, ΔF, "FNR>5%", Vote, Last).
2. **Biased learning** (SCAR assumption, U = noisy negative): cost-sensitive SVMs (Biased-SVM,
   Bagging-SVM, Robust-Ensemble-SVM), weighted logistic regression, etc.
3. **Class-prior incorporation** (SCAR assumption, c/α used directly): postprocessing (Elkan-Noto
   scaling), preprocessing (rebalancing, Elkan-Noto weighting, **empirical-risk-minimization
   reweighting** — here it's explicitly shown that du Plessis et al.'s uPU approach is a special
   case of this more general ERM-reweighting family), method modification (e.g., directly adjusting
   Naive Bayes counts).

**Important comparison rule from the paper** (Section 5.6, directly actionable): if separability
holds → prefer Two-Step; if SCAR holds → prefer biased learning/class-prior methods; if both hold,
the choice depends on how clearly separated the classes are (for very closely neighboring classes,
Two-Step methods are more error-prone, exploiting SCAR is more robust; for clearly separated
classes, the reverse).

## Class prior estimation (Section 6, directly relevant to plan Step 4)
A systematic overview of exactly the methods the plan cites:
- **Non-traditional classifier** (Elkan & Noto: mean of g(x) on validation positives).
- **Rank Pruning** (Northcutt et al.) as a more robust variant — uses only "confident positive"
  examples instead of all.
- **Partial matching** (du Plessis & Sugiyama 2014/2015): density estimation, class prior as a
  scaling factor between the estimated positive and total distributions — with a **penalized
  divergence** variant for overlapping classes (relevant, since pure separability is unrealistic
  in the GO domain).
- **TIcE** (Bekker & Davis 2018, see own summary) — decision-tree-based lower bound on subdomains.
- **KM1/KM2** (Ramaswamy et al. 2016, see own summary) — weakest assumption ("positive function"
  instead of "positive subdomain"), strongest theoretical guarantee.

**The survey reports a direct empirical comparison** (Bekker & Davis themselves, 11 datasets,
various SCAR settings): **KM2 and TIcE deliver the most accurate estimates**, with TIcE being
markedly faster (KM2 was only feasible on small subsets in some cases). Under **SAR** (not SCAR),
KM2 per a follow-up work (Bekker, Robberechts & Davis 2019) even outperforms TIcE in accuracy — a
direct hint at which of the two estimators the plan should choose if the SCAR assumption is indeed
violated.

## Direct relevance to the plan
- Provides the **precise definitions** of SCAR and SAR that the plan already (correctly) invokes —
  including the warning that SCAR and SAR are *both* actually "MNAR" (Missing Not At Random) in
  missing-data terminology, not "MCAR", which is often misunderstood.
- The empirical KM2-vs-TIcE comparison gives a **concrete recommendation** for plan Step 4: TIcE
  first for speed (668 terms!), KM2 as a quality check on a sample of terms, especially if SAR
  (rather than SCAR) is suspected.
- Confirms that the plan's chosen basic architecture — Two-Step-style reliable-negative extraction
  via spies (Direction A) — is one of the three established main categories, and provides via the
  comparison rule in 5.6 a criterion for when this category (rather than biased learning or
  class-prior incorporation) is the right choice: when separability rather than a robust global
  SCAR assumption is more plausible — which seems more likely for per-term-heterogeneous GO
  annotation than a single global calibration constant.
- Names **Rank Pruning** (Northcutt et al. 2017) as a more robust alternative to Elkan & Noto's
  simple classifier mean — not yet mentioned in the plan, but potentially relevant as a third
  alternative alongside TIcE/KM.
