# Positional-MF review: GO:0000156

Name: phosphorelay response regulator activity
Definition: Responds to a phosphorelay sensor to initiate a change in cell state or activity. The activity of the response regulator is regulated by transfer of a phosphate from a histidine residue in the sensor, to an aspartate residue in the response regulator. Many but not all response regulators act as transcriptional regulators to elicit a response.

Every protein below carries the term above as a real, asserted annotation (from UniProt's GO cross-references) but no positional feature has yet been confirmed to explain *why*. For your assigned protein (given separately in your task instructions), judge which annotated feature, if any, plausibly accounts for this GO term.

Reconstructed 2026-08-19 from the original per-protein prompt files (P0AE88.md/P0AFB8.md/P23620.md,
commit b2d0b9c), which prepare_term_review_files.py's later restructure into a single combined
review.md (commit eb90059) overwrote with a freshly-resampled, non-overlapping set of 10 proteins --
findings.md/parsed_findings.json were correctly left untouched at that point (the merge itself was
only re-run, not the review), but the result was this file no longer showing what the 3 merged
witnesses below were actually shown. Content below is byte-for-byte the original prompt text for
those 3 proteins, not a new sample.

---

## Protein P0AE88
Sequence length: 232 aa

Annotated features (position, type, qualifiers):
- DOMAIN 2-115
    note: Response regulatory
    evidence: ECO:0000255|PROSITE-ProRule:PRU00169
- DNA_BIND 130-230
    note: OmpR/PhoB-type
    evidence: ECO:0000255|PROSITE-ProRule:PRU01091

Domain/family cross-references (no position, but may help identify a domain):
- InterPro:IPR011006 -- CheY-like_superfamily
- InterPro:IPR058122 -- CpxR
- InterPro:IPR058124 -- CpxR-like_REC
- InterPro:IPR001867 -- OmpR/PhoB-type_DNA-bd
- InterPro:IPR001789 -- Sig_transdc_resp-reg_receiver
- InterPro:IPR039420 -- WalR-like
- InterPro:IPR036388 -- WH-like_DNA-bd_sf
- Pfam:PF00072 -- Response_reg
- Pfam:PF00486 -- Trans_reg_C
- PROSITE:PS51755 -- OMPR_PHOB
- PROSITE:PS50110 -- RESPONSE_REGULATORY

---

## Protein P0AFB8
Sequence length: 469 aa

Annotated features (position, type, qualifiers):
- DOMAIN 4-119
    note: Response regulatory
    evidence: ECO:0000255|PROSITE-ProRule:PRU00169
- DOMAIN 139-369
    note: Sigma-54 factor interaction
    evidence: ECO:0000255|PROSITE-ProRule:PRU00193
- DOMAIN 419-469
    note: HTH NtrC-type
    evidence: ECO:0000255|PROSITE-ProRule:PRU01435
- DNA_BIND 444-468
    note: H-T-H motif
    evidence: ECO:0000255|PROSITE-ProRule:PRU01435
- BINDING 167-175
    ligand: ATP
    ligand_id: ChEBI:CHEBI:30616
    evidence: ECO:0000255|PROSITE-ProRule:PRU00193
- BINDING 230-240
    ligand: ATP
    ligand_id: ChEBI:CHEBI:30616
    evidence: ECO:0000255|PROSITE-ProRule:PRU00193

Domain/family cross-references (no position, but may help identify a domain):
- InterPro:IPR003593 -- AAA+_ATPase
- InterPro:IPR058031 -- AAA_lid_NorR
- InterPro:IPR011006 -- CheY-like_superfamily
- InterPro:IPR009057 -- Homeodomain-like_sf
- InterPro:IPR002197 -- HTH_Fis
- InterPro:IPR027417 -- P-loop_NTPase
- InterPro:IPR001789 -- Sig_transdc_resp-reg_receiver
- InterPro:IPR002078 -- Sigma_54_int
- InterPro:IPR025662 -- Sigma_54_int_dom_ATP-bd_1
- InterPro:IPR025943 -- Sigma_54_int_dom_ATP-bd_2
- InterPro:IPR025944 -- Sigma_54_int_dom_CS
- InterPro:IPR010114 -- Transcript_reg_NtrC
- Pfam:PF25601 -- AAA_lid_14
- Pfam:PF02954 -- HTH_8
- Pfam:PF00072 -- Response_reg
- Pfam:PF00158 -- Sigma54_activat
- PROSITE:PS52089 -- HTH_NTRC
- PROSITE:PS50110 -- RESPONSE_REGULATORY
- PROSITE:PS00675 -- SIGMA54_INTERACT_1
- PROSITE:PS00676 -- SIGMA54_INTERACT_2
- PROSITE:PS00688 -- SIGMA54_INTERACT_3
- PROSITE:PS50045 -- SIGMA54_INTERACT_4

---

## Protein P23620
Sequence length: 229 aa

Annotated features (position, type, qualifiers):
- DOMAIN 4-121
    note: Response regulatory
    evidence: ECO:0000255|PROSITE-ProRule:PRU00169
- DNA_BIND 129-227
    note: OmpR/PhoB-type
    evidence: ECO:0000255|PROSITE-ProRule:PRU01091

Domain/family cross-references (no position, but may help identify a domain):
- InterPro:IPR011006 -- CheY-like_superfamily
- InterPro:IPR001867 -- OmpR/PhoB-type_DNA-bd
- InterPro:IPR016032 -- Sig_transdc_resp-reg_C-effctor
- InterPro:IPR011879 -- Sig_transdc_resp-reg_PhoB
- InterPro:IPR001789 -- Sig_transdc_resp-reg_receiver
- InterPro:IPR039420 -- WalR-like
- InterPro:IPR036388 -- WH-like_DNA-bd_sf
- Pfam:PF00072 -- Response_reg
- Pfam:PF00486 -- Trans_reg_C
- PROSITE:PS51755 -- OMPR_PHOB
- PROSITE:PS50110 -- RESPONSE_REGULATORY

---
