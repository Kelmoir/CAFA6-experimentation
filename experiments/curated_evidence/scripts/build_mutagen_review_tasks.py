"""Item 15, tier 1, prep stage for the LLM-review follow-up to mine_mutagen_evidence.py.

mine_mutagen_evidence.py's own term-matching step (plain content-word token overlap between the
note and a candidate GO term's name) has two real failure modes found by direct inspection: it
can't tell a negated clause ("does not abolish X") from a real loss, and it can't bridge a
synonym/paraphrase the note uses instead of the term's literal name -- the latter almost certainly
undercounts real losses, since it's never been able to match wording it doesn't share verbatim.

This script redoes just the funnel/candidate-assembly half (same LOSS_PATTERN, same per-protein
real-molecular-function-term lookup, both imported unchanged from mine_mutagen_evidence.py) but
skips the token-overlap matching step entirely -- every loss-language-flagged MUTAGEN note becomes
one review task, carrying the FULL list of the protein's own real MF terms (id, name, and official
definition straight from go-basic.obo, via prepare_term_review_files.py's load_go_term_info,
imported unchanged) as the closed candidate set. The actual loss-vs-term matching, including
negation and conditional-mutation handling, is left to local_llm_review_mutagen.py -- a local LLM
reading the full note in context, not a bag-of-words heuristic.
"""
import gzip
import json
import sys
from pathlib import Path

from Bio import SwissProt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_mutagen_evidence import LOSS_PATTERN, UNIPROT_DUMP, real_mf_terms  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mf_span/scripts"))
from prepare_term_review_files import load_go_term_info  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "mutagen_review_tasks.json"


def main():
    print("[INFO] Loading go-basic.obo term name/definition table...")
    go_info = load_go_term_info()
    print(f"[INFO] {len(go_info)} GO terms with name/definition")

    tasks = []
    n_proteins_with_mutagen = 0
    print(f"[INFO] Streaming {UNIPROT_DUMP.name}...")
    with gzip.open(UNIPROT_DUMP, "rt") as fh:
        for record in SwissProt.parse(fh):
            mutagens = [f for f in record.features if f.type == "MUTAGEN"]
            if not mutagens:
                continue
            n_proteins_with_mutagen += 1
            accession = record.accessions[0]
            mf_terms = None  # computed lazily, only if a loss-pattern note shows up

            for feat in mutagens:
                note = feat.qualifiers.get("note", "")
                if not LOSS_PATTERN.search(note):
                    continue
                if mf_terms is None:
                    mf_terms = real_mf_terms(record)
                    candidates = [
                        {"go_id": go_id, "name": name, "def": go_info.get(go_id, {}).get("def", "")}
                        for go_id, name in mf_terms
                    ]
                if not candidates:
                    continue  # no real MF term on record at all -- nothing for an LLM to pick from
                tasks.append({
                    "id": len(tasks),
                    "accession": accession,
                    "start": int(feat.location.start),
                    "end": int(feat.location.end),
                    "note": note,
                    "evidence": feat.qualifiers.get("evidence", ""),
                    "candidates": candidates,
                })

            if n_proteins_with_mutagen % 20000 == 0:
                print(f"[INFO] {n_proteins_with_mutagen} MUTAGEN-bearing proteins scanned, "
                      f"{len(tasks)} review tasks so far", flush=True)

    OUTPUT_PATH.write_text(json.dumps(tasks, indent=2))
    print(f"[RESULT] {len(tasks)} review tasks (loss-pattern note + >=1 real MF-term candidate)")
    print(f"[INFO] wrote -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
