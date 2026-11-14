"""Second ground-truth variant: filters mutagen_review_results_claude.jsonl's verdicts down to
ONLY this protein's real, directly-asserted molecular-function GO terms (what would actually show
up as a UniProt/GOA cross-reference, i.e. what CAFA's own train_terms.tsv is built from) --
dropping any verdict that's only in the candidate list because of build_mutagen_review_tasks.py's
ancestor-closure expansion (e.g. the bare "catalytic activity" root).

Why this variant matters: the point of this whole pipeline is new CAFA training labels, and CAFA
only scores real, UniProt-curated leaf terms, never a generic ancestor like "catalytic activity" --
a verdict that's just an ancestor carries no new information for that purpose even when it's a
correct statement. If most of the real signal in a MUTAGEN note turns out to land on one of this
protein's own already-known leaf terms rather than requiring ancestor/specificity reasoning, the
review task can be posed as a much smaller, closed yes/no question per real leaf term instead of
the current "which of N ancestor-expanded candidates" judgment call -- smaller prompt, no DAG
reasoning needed, faster per-task inference. This file exists to measure how much of the 100-task
ground truth survives that narrower framing before committing to redesigning the task around it.

Reuses real_mf_terms/UNIPROT_DUMP from mine_mutagen_evidence.py unchanged -- same source of truth
(the record's own GO cross-references, aspect F) already used to build the candidate lists.
"""
import gzip
import json
import sys
from pathlib import Path

from Bio import SwissProt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_mutagen_evidence import UNIPROT_DUMP, real_mf_terms  # noqa: E402

SOURCE = Path(__file__).resolve().parent.parent / "mutagen_review_results_claude.jsonl"
OUTPUT = Path(__file__).resolve().parent.parent / "mutagen_review_results_claude_leaf.jsonl"


def main():
    records = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    wanted_accessions = {r["accession"] for r in records}

    print(f"[INFO] Looking up real leaf MF terms for {len(wanted_accessions)} accessions in "
          f"{UNIPROT_DUMP.name}...")
    real_terms_by_accession = {}
    with gzip.open(UNIPROT_DUMP, "rt") as fh:
        for rec in SwissProt.parse(fh):
            acc = rec.accessions[0]
            if acc in wanted_accessions:
                real_terms_by_accession[acc] = {go_id for go_id, _ in real_mf_terms(rec)}
                if len(real_terms_by_accession) == len(wanted_accessions):
                    break
    missing = wanted_accessions - set(real_terms_by_accession)
    if missing:
        print(f"[WARN] {len(missing)} accessions not found in the dump: {sorted(missing)}")

    n_dropped_verdicts = 0
    n_emptied = 0
    for rec in records:
        real_ids = real_terms_by_accession.get(rec["accession"], set())
        kept = [v for v in rec["verdicts"] if v["go_id"] in real_ids]
        n_dropped_verdicts += len(rec["verdicts"]) - len(kept)
        if rec["verdicts"] and not kept:
            n_emptied += 1
            rec["status"] = "NONE"
        rec["verdicts"] = kept

    OUTPUT.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    n_with_verdict = sum(1 for r in records if r["verdicts"])
    print(f"[RESULT] {len(records)} tasks, {n_with_verdict} keep >=1 real leaf verdict, "
          f"{n_emptied} dropped to NONE (their only verdict(s) were ancestor-only), "
          f"{n_dropped_verdicts} individual ancestor-only verdicts removed total -> {OUTPUT}")


if __name__ == "__main__":
    main()
