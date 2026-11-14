"""Item 15, tier 1, prep stage for the LLM-review follow-up to mine_mutagen_evidence.py.

mine_mutagen_evidence.py's own term-matching step (plain content-word token overlap between the
note and a candidate GO term's name) has two real failure modes found by direct inspection: it
can't tell a negated clause ("does not abolish X") from a real loss, and it can't bridge a
synonym/paraphrase the note uses instead of the term's literal name -- the latter almost certainly
undercounts real losses, since it's never been able to match wording it doesn't share verbatim.

This script redoes just the funnel/candidate-assembly half (same LOSS_PATTERN, same per-protein
real-molecular-function-term lookup, both imported unchanged from mine_mutagen_evidence.py) but
skips the token-overlap matching step entirely -- every loss-language-flagged MUTAGEN note becomes
one review task, carrying the protein's real MF terms (id, name, and official definition straight
from go-basic.obo, via prepare_term_review_files.py's load_go_term_info, imported unchanged) as the
closed candidate set. The actual loss-vs-term matching, including negation and conditional-mutation
handling, is left to local_llm_review_mutagen.py -- a local LLM reading the full note in context,
not a bag-of-words heuristic.

Candidate set includes each asserted term's full is_a/part_of ANCESTOR closure too, not just the
directly-asserted leaf terms -- found necessary directly (not a hypothetical): a note can describe
something that maps cleanly to a BROADER term that's only implicitly true under GO's true-path
rule (every ancestor of a true term is also true) but was never spelled out as its own UniProt
cross-reference row, so without this the candidate list can be missing the one term that should
have matched. Same ancestor-propagation mechanism used throughout this project (restrict_to_true_
path, imported from generate_dataset.py unchanged) -- only the bare MF root (GO:0003674,
"molecular_function" itself) is excluded, since it's definitionally too generic to ever be a
meaningful match; every other ancestor is left in, trusting local_llm_review_mutagen.py's own
generic-activity/sole-candidate instructions to decline an ancestor that's still too broad to be
confident about. Hierarchy-consistency clean-up of the resulting matches (e.g. the same protein
confirming both a term and one of its own ancestors as independently "lost" from different notes)
is left for later, not handled here.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import networkx as nx
import obonet
from Bio import SwissProt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_mutagen_evidence import LOSS_PATTERN, UNIPROT_DUMP, real_mf_terms  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mf_span/scripts"))
from prepare_term_review_files import load_go_term_info  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from generate_dataset import restrict_to_true_path  # noqa: E402

GO_OBO_PATH = (Path(__file__).resolve().parent.parent.parent.parent
               / "data/cafa-6-protein-function-prediction/Train/go-basic.obo")
MF_ROOT = "GO:0003674"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "mutagen_review_tasks.json"


def build_ancestor_lookup():
    graph = restrict_to_true_path(obonet.read_obo(str(GO_OBO_PATH)))
    cache = {}

    def ancestors_of(term):
        if term not in cache:
            cache[term] = nx.descendants(graph, term) if term in graph else set()
        return cache[term]

    return ancestors_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH,
                     help="Override the output path -- use a distinct path (e.g. "
                          "mutagen_review_tasks_leaf.json) so a --leaf-only build doesn't overwrite "
                          "the existing ancestor-expanded tasks.json, which is still needed for "
                          "benchmarking against it.")
    ap.add_argument("--leaf-only", action="store_true",
                     help="Candidates are ONLY this protein's own real, directly-asserted MF terms "
                          "(real_mf_terms(record), aspect F) -- no is_a/part_of ancestor closure at "
                          "all. For testing whether the review task can be posed as a much smaller "
                          "closed yes/no question per already-known leaf term, instead of picking "
                          "among the full ancestor-expanded candidate set.")
    args = ap.parse_args()

    print("[INFO] Loading go-basic.obo term name/definition table...")
    go_info = load_go_term_info()
    print(f"[INFO] {len(go_info)} GO terms with name/definition")

    ancestors_of = None
    if not args.leaf_only:
        print("[INFO] Building is_a/part_of ancestor lookup from go-basic.obo...")
        ancestors_of = build_ancestor_lookup()

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
                    candidate_ids = {go_id for go_id, _ in mf_terms}
                    if not args.leaf_only:
                        for go_id, _ in mf_terms:
                            candidate_ids |= ancestors_of(go_id)
                    candidate_ids.discard(MF_ROOT)
                    candidates = [
                        {"go_id": go_id, "name": go_info[go_id].get("name", ""),
                         "def": go_info[go_id].get("def", "")}
                        for go_id in sorted(candidate_ids) if go_id in go_info
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

    args.output.write_text(json.dumps(tasks, indent=2))
    print(f"[RESULT] {len(tasks)} review tasks (loss-pattern note + >=1 real MF-term candidate)")
    print(f"[INFO] wrote -> {args.output}")


if __name__ == "__main__":
    main()
