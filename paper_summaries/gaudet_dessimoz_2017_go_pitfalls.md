# Gaudet & Dessimoz (2017) — *Gene Ontology: Pitfalls, Biases, and Remedies* (GO Handbook, Methods Mol. Biol. 1446)

**Full text available:** Yes (arXiv 1602.01875, identical to the book-chapter version, read in
full).
**Relevance to this project:** The plan cites this as "already the source of this repo's OWA
framing" — this chapter is the basis for the OWA terminology the plan uses throughout, and
additionally supplies several **concrete, empirically documented bias mechanisms** that go beyond
the plain OWA claim and are directly relevant to the SCAR-vs-SAR question.

## Core claim: Open World Assumption (OWA)
GO annotations are necessarily incomplete: **absence of an annotation does not mean absence of the
function.** Formally framed as the opposite of the closed-world assumption. The chapter even shows
the edge case in a footnote: proteins with no characterized function at all are explicitly
annotated with "ND" (No biological Data available) at the ontology root — a formal distinction
between "nothing is known" and "nothing was annotated" that may be relevant for interpreting U
cells in the plan.

## Concrete bias mechanisms (Section 3, with empirical evidence)
Several effects documented here directly justify why the **SCAR assumption** (from Elkan & Noto,
du Plessis et al., etc.) is questionable for GO data — exactly the point the plan already
anticipates correctly under "Key risks," here spelled out with specifics:

- **Species differences**: annotation density and type differ strongly across species (zebrafish
  heavily annotated for developmental biology, rat for toxicology) — simply using the whole
  database as background leads to spurious enrichments.
- **Authorship bias**: annotations from the same article are, on average, markedly more similar
  than annotations from different articles — an earlier comparison of orthologs vs. paralogs
  (Nehrt et al.) found an apparent difference in functional conservation that completely
  disappeared (and even reversed) once controlling for authorship. Directly relevant, since GO
  annotations for closely related proteins often come from the same publications/labs — a
  systematic, non-random labeling mechanism.
- **Annotator/database bias**: different curating groups (e.g., MGI vs. UniProt for mouse) can be
  shown to preferentially annotate different GO terms, even with comparable evidence-code
  proportions — a concrete numeric example in the chapter (e.g., "in utero embryonic development":
  1170 MGI annotations vs. 4 for UniProt).
- **Propagation bias**: electronic (automatically transferred) annotations between homologous
  sequences are systematically more similar than experimental ones — an artifact of the transfer
  method, not of biology.
- **Positive/negative imbalance**: **fewer than 1% of all experimental annotations in
  UniProt-GOA are negative ("NOT") annotations** — a very concrete figure that can be related to
  the "0.556% observed positive rate" cited in the plan (from the raw label space): the plan's
  positive rate concerns *all* unknown cells, while this <1% figure targets explicit NOT
  annotations specifically — together, the two figures illustrate how extremely rare *any*
  explicit signal (positive or negative) is relative to the full matrix.
- Directly cites Rider et al. (2013) — the same observation that missing negative annotations
  **systematically distort** standard ML metrics (ROC-AUC, PR-AUC), with the warning that the
  effect is "likely even greater" under true OWA (rather than the closed-world simplification
  assumed in that study).

## Other structural pitfalls (less directly PU-relevant, but important for data preparation)
- **The NOT qualifier propagates to child terms, not to parent terms** (opposite to positive
  annotations) — relevant if the plan ever incorporates explicit negative GO annotations.
- **Transitivity**: "is a"/"part of" are transitive, "regulates"/"has part" are not — important if
  the plan uses label propagation over the ontology hierarchy (as in the referenced
  `ancestor_sweep` experiment).
- **Evidence-code quality varies widely** (IDA most reliable, IEP/IGI/high-throughput annotations
  least certain) — practically relevant to the question of which cells the plan should even treat
  as "P (positive)," or whether positives should be weighted by evidence code, which the plan
  currently does not consider.

## Direct relevance to the plan
- Provides the **empirical grounding** for the risk the plan only asserts under "Key risks" ("GO
  annotation is biased toward well-studied proteins/organisms") — with concrete, citable figures
  and mechanisms (authorship, annotator, species, propagation), not just a general suspicion.
- The authorship- and annotator-bias findings suggest that the SCAR assumption underlying
  virtually all calibration methods summarized in this project (Elkan-Noto, TIcE, KM1/KM2) is
  **very likely violated** — further supporting the plan's intent to empirically validate
  spy-based calibration (Direction A) rather than rely purely on modeling assumptions, and
  suggesting the spy leak rate be checked **per term and possibly per species/data source**
  separately rather than only globally.
- The <1% rate of explicit NOT annotations offers a second, independent data point that "true"
  negatives are extremely rarely stated explicitly in the GO matrix — the plan must therefore rely
  almost entirely on indirectly inferred (reliable-negative) cells; there is no meaningful pool of
  "gold-standard" negatives available to validate the spy thresholds against.
