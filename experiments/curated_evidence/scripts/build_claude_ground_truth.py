"""Builds a 100-task ground-truth file for benchmarking candidate local models, by applying the
corrections an Opus review (see experiments/cafa6_next_steps_plan.md, item 15) found against
mutagen_review_results_5.jsonl's 100 tasks: 3 false positives, 4 false negatives. Everything else
(including the "too narrow"/"too specific" calls Opus flagged as informative-only, not actual
problems) is kept as-is from the round-5 run.

A second round of corrections (ROUND_2_CORRECTIONS) was added after the is_a-indexed-candidate-list
prompt redesign: several of the round-5/Opus-era answers under-climbed a real, single is_a chain
(e.g. stopping at bare "catalytic activity" instead of this protein's own specific leaf annotation)
or missed a genuine shared-more-specific-ancestor among real but unrelated sibling leaf terms.
Every one of these was verified directly against the actual go-basic.obo true-path-restricted
graph (not assumed from term names) before being applied here -- see the chat history around
2026-11-13/14 for the specific ancestor checks run.
"""
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "mutagen_review_results_5.jsonl"
OUTPUT = Path(__file__).resolve().parent.parent / "mutagen_review_results_claude.jsonl"

# id -> (new status, new verdicts, new reasoning). Only the 7 ids Opus flagged as real errors.
CORRECTIONS = {
    # false positives
    30: ("NONE", [], "Note hedges with \"Probable loss\" -- the source itself is unsure, not a "
                      "clear unconditional loss."),
    54: ("OK", [{"go_id": "GO:0016887", "conditional": False}],
         "ATPase loss (GO:0016887) stands, but the triphosphatase loss is explicitly reversed when "
         "manganese replaces magnesium -- not a real unconditional loss, dropped rather than kept "
         "(conditional) per the false-negative-over-false-positive preference."),
    67: ("NONE", [], "Loss is specific to one nAChR subtype (alpha-1-beta-1-delta-epsilon) while "
                      "the other subtype (alpha-3-beta-2) is still inhibited -- generalizing to the "
                      "receptor-class term violates the same subtype-specificity rule as the KcsA "
                      "case; task 66's identical note structure correctly answered NONE."),
    # false negatives
    68: ("OK", [{"go_id": "GO:0003824", "conditional": False}],
         "\"Complete loss of catalytic activity\" with two unrelated specific siblings (mannosidase "
         "vs xylosidase) on record -- keep the general match per the sibling-ambiguity rule, as "
         "task 69's identical note correctly did."),
    71: ("OK", [{"go_id": "GO:0008190", "conditional": False}],
         "\"Loss of interaction with EIF4E\" matches GO:0008190 (eukaryotic initiation factor 4E "
         "binding) literally by name -- the partner-exclusion rule's exception, as task 70's "
         "identical note correctly applied."),
    73: ("OK", [{"go_id": "GO:0003824", "conditional": False}],
         "\"Loss of activity\" with two unrelated specific dehydrogenase siblings on record -- keep "
         "the general match per the sibling-ambiguity rule, as tasks 74/75's identical note "
         "correctly did."),
    93: ("OK", [{"go_id": "GO:0003824", "conditional": False}],
         "\"Loss of activity\" with two unrelated specific siblings on record (pyruvate carboxylase "
         "vs methylmalonyl-CoA carboxytransferase) -- keep the general match per the "
         "sibling-ambiguity rule."),
}

# Round 2: verified against the real go-basic.obo graph after the is_a-indexed prompt redesign.
ROUND_2_CORRECTIONS = {
    15: ("OK", [{"go_id": "GO:0003824", "conditional": False}],
         "Bare \"loss of function\" directly names catalytic activity; the transaminase/lyase/"
         "vitamin-binding branches share no ancestor more specific than the bare root (verified: "
         "none of them is an ancestor of another), so there's no rescue -- keep the general match."),
    16: ("OK", [{"go_id": "GO:0003824", "conditional": False}], "Same protein/candidate set as 15."),
    36: ("OK", [{"go_id": "GO:0000334", "conditional": False}],
         "Verified single is_a chain, catalytic activity -> ... -> 3-hydroxyanthranilate "
         "3,4-dioxygenase activity -- climb to the leaf, same as task 35's identical note."),
    38: ("OK", [{"go_id": "GO:0030550", "conditional": False}],
         "Verified GO:0030547 (signaling receptor inhibitor activity) is a real ancestor of "
         "GO:0030550 (acetylcholine receptor inhibitor activity) -- single chain, climb to the leaf."),
    57: ("OK", [{"go_id": "GO:0050355", "conditional": False}],
         "Verified single is_a chain, hydrolase activity -> ... -> inorganic triphosphate "
         "phosphatase activity -- climb to the leaf."),
    68: ("OK", [{"go_id": "GO:0004553", "conditional": False}],
         "Verified GO:0004553 (hydrolase activity, hydrolyzing O-glycosyl compounds) is a real "
         "shared ancestor of BOTH the mannosidase and xylosidase branches -- more informative than "
         "the bare root and still unambiguous; supersedes the round-1 GO:0003824 correction."),
    69: ("OK", [{"go_id": "GO:0004553", "conditional": False}], "Same protein/candidate set as 68."),
    73: ("OK", [{"go_id": "GO:0016616", "conditional": False}],
         "Verified GO:0016616 is a real shared ancestor of BOTH the alcohol-dehydrogenase chain and "
         "the unrelated 4-hydroxybutyrate dehydrogenase activity -- supersedes the round-1 "
         "GO:0003824 correction."),
    74: ("OK", [{"go_id": "GO:0016616", "conditional": False}], "Same protein/candidate set as 73."),
    75: ("OK", [{"go_id": "GO:0016616", "conditional": False}], "Same protein/candidate set as 73."),
    77: ("OK", [{"go_id": "GO:0047580", "conditional": False}],
         "Verified GO:0047661 (amino-acid racemase activity) is a real ANCESTOR of GO:0047580 -- "
         "a single chain, not unrelated siblings as earlier assumed; climb to the leaf."),
    78: ("OK", [{"go_id": "GO:0047580", "conditional": False}], "Same protein/candidate set as 77."),
    80: ("OK", [{"go_id": "GO:0047787", "conditional": False}],
         "Verified single is_a chain all the way to Delta4-3-oxosteroid 5beta-reductase activity -- "
         "climb to the leaf."),
    87: ("OK", [{"go_id": "GO:0008253", "conditional": False}, {"go_id": "GO:0000215", "conditional": False}],
         "Two separate, unconditional losses, each the leaf of a verified single is_a chain: "
         "nucleotidase -> 5'-nucleotidase (GO:0008253), phosphotransferase -> tRNA 2'-phosphotransferase "
         "(GO:0000215, verified descendant of GO:0016773 and GO:0140101, not a sibling of either)."),
    88: ("OK", [{"go_id": "GO:0008253", "conditional": False}, {"go_id": "GO:0000215", "conditional": False}],
         "Same protein/chains as 87 (\"almost complete\" is NEAR-TOTAL, same as unconditional)."),
    90: ("OK", [{"go_id": "GO:0000215", "conditional": False}],
         "Nucleotidase clause is explicitly negated (\"no effect\"); only the phosphotransferase "
         "clause is a genuine loss, climbing the same verified chain to GO:0000215."),
    91: ("OK", [{"go_id": "GO:0008253", "conditional": False}, {"go_id": "GO:0000215", "conditional": False}],
         "Same protein/chains as 87/88."),
    99: ("OK", [{"go_id": "GO:0016758", "conditional": False}],
         "Verified this protein has TWO real, unrelated leaf annotations (GO:0035251 and GO:0120514, "
         "neither an ancestor of the other) that share GO:0016758 (hexosyltransferase activity) as "
         "their deepest common ancestor -- more complete than the round-1 single-leaf guess "
         "(GO:0120514), which silently dropped the other true annotation."),
}


def main():
    records = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    n_fixed = 0
    for rec in records:
        fix = ROUND_2_CORRECTIONS.get(rec["id"]) or CORRECTIONS.get(rec["id"])
        if fix:
            rec["status"], rec["verdicts"], rec["reasoning"] = fix
            n_fixed += 1
    records.sort(key=lambda r: r["id"])
    OUTPUT.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    n_flagged = len(set(CORRECTIONS) | set(ROUND_2_CORRECTIONS))
    print(f"[INFO] {len(records)} tasks, {n_fixed} corrected (of {n_flagged} flagged) -> {OUTPUT}")


if __name__ == "__main__":
    main()
