"""Item 15, tier 1 (experiments/cafa6_next_steps_plan.md): mine real, already-published `FT MUTAGEN`
evidence for trustworthy known-negative (protein, term) examples -- the generalized, real-evidence
version of item 14 step 2's synthetic broken-span mechanism (no mutation construction needed, a
curator/researcher already made and characterized one).

For every MUTAGEN feature whose free-text `/note` describes a clear loss of function, check
whether the described activity matches (by content-word overlap, same token-overlap idea
build_positional_mf_links.py already uses for domain-description bridging) one of the protein's
own real molecular_function (aspect F) GO annotations -- read directly from each record's own
`cross_references` GO entries (e.g. ('GO', 'GO:0004672', 'F:protein kinase activity', ...)), no
go-basic.obo lookup needed. A confirmed match is real evidence that the WILD-TYPE protein's
documented function is lost in the specific point-mutant UniProt describes at that position --
the raw material for a later known-negative training example (constructing and adding the mutant
sequence itself is not done here, see the plan's "deliverable, once mining is done" note).

Single streaming pass over the already-cached data/uniprot/uniprot_sprot.dat.gz dump (same pattern
as experiments/mf_span/scripts/fetch_uniprot_positional_features.py and
experiments/ensembling/goa_iea_probe.py) -- never holds the full dump in memory.
"""
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from Bio import SwissProt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
UNIPROT_DUMP = REPO_ROOT / "data/uniprot/uniprot_sprot.dat.gz"
TRAIN_TERMS_PATH = REPO_ROOT / "data/cafa-6-protein-function-prediction/Train/train_terms.tsv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "mutagen_negatives.json"

# "Definite loss" language only -- conservative on purpose, this is meant to be trustworthy
# evidence, not a guess. Excludes "reduced"/"impaired"/"decreased" (partial effects, noisier) and
# "normal"/"no effect" (not a loss at all).
LOSS_PATTERN = re.compile(
    r"\b(loss of|abolish|ablat|eliminat|no detectable|completely (lose|loses|lost|abolished))\b",
    re.IGNORECASE,
)
STOPWORDS = {"activity", "binding", "function", "of", "the", "to", "a", "an", "and", "protein",
             "process", "regulation"}


def tokenize(text: str) -> set:
    return set(re.split(r"[\s\-/,.();]+", text.lower())) - STOPWORDS - {""}


def real_mf_terms(record) -> list:
    """[(go_id, name)] from the record's own GO cross-references, aspect F only."""
    out = []
    for xref in record.cross_references:
        if xref[0] == "GO" and xref[2].startswith("F:"):
            out.append((xref[1], xref[2][2:]))
    return out


def main():
    cafa6_proteins = set()
    with open(TRAIN_TERMS_PATH) as f:
        next(f)
        for line in f:
            cafa6_proteins.add(line.split("\t", 1)[0])
    print(f"[INFO] {len(cafa6_proteins)} unique CAFA6 training proteins")

    n_proteins_with_mutagen = 0
    n_mutagen_total = 0
    n_loss_pattern = 0
    matches = []

    print(f"[INFO] Streaming {UNIPROT_DUMP.name}...")
    with gzip.open(UNIPROT_DUMP, "rt") as fh:
        for record in SwissProt.parse(fh):
            mutagens = [f for f in record.features if f.type == "MUTAGEN"]
            if not mutagens:
                continue
            n_proteins_with_mutagen += 1
            n_mutagen_total += len(mutagens)
            accession = record.accessions[0]
            mf_terms = None  # computed lazily, only if a loss-pattern note shows up

            for feat in mutagens:
                note = feat.qualifiers.get("note", "")
                if not LOSS_PATTERN.search(note):
                    continue
                n_loss_pattern += 1
                if mf_terms is None:
                    mf_terms = real_mf_terms(record)
                note_tokens = tokenize(note)
                for go_id, name in mf_terms:
                    if note_tokens & tokenize(name):
                        matches.append({
                            "accession": accession,
                            "go_id": go_id,
                            "go_name": name,
                            "start": int(feat.location.start),
                            "end": int(feat.location.end),
                            "note": note,
                            "evidence": feat.qualifiers.get("evidence", ""),
                            "in_cafa6": accession in cafa6_proteins,
                        })

            if n_proteins_with_mutagen % 20000 == 0:
                print(f"[INFO] {n_proteins_with_mutagen} MUTAGEN-bearing proteins scanned, "
                      f"{len(matches)} confirmed matches so far", flush=True)

    OUTPUT_PATH.write_text(json.dumps(matches, indent=2))

    n_confirmed_proteins = len({m["accession"] for m in matches})
    n_in_cafa6 = sum(1 for m in matches if m["in_cafa6"])
    term_counts = Counter(m["go_id"] for m in matches)

    print(f"[RESULT] proteins with >=1 MUTAGEN feature: {n_proteins_with_mutagen}")
    print(f"[RESULT] total MUTAGEN features: {n_mutagen_total}")
    print(f"[RESULT] MUTAGEN notes matching the loss-of-function pattern: {n_loss_pattern}")
    print(f"[RESULT] confirmed (protein, term) negative candidates: {len(matches)} "
          f"across {n_confirmed_proteins} proteins")
    print(f"[RESULT] of those, already in CAFA6's train_terms.tsv protein set: {n_in_cafa6} "
          f"({100 * n_in_cafa6 / max(len(matches), 1):.1f}%)")
    print(f"[RESULT] top 10 matched GO terms by frequency: {term_counts.most_common(10)}")
    print(f"[INFO] wrote {len(matches)} records -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
