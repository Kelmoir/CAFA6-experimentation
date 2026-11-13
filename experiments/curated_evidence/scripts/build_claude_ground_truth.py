"""Builds a 100-task ground-truth file for benchmarking candidate local models, by applying the
corrections an Opus review (see experiments/cafa6_next_steps_plan.md, item 15) found against
mutagen_review_results_5.jsonl's 100 tasks: 3 false positives, 4 false negatives. Everything else
(including the "too narrow"/"too specific" calls Opus flagged as informative-only, not actual
problems) is kept as-is from the round-5 run.
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


def main():
    records = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    n_fixed = 0
    for rec in records:
        fix = CORRECTIONS.get(rec["id"])
        if fix:
            rec["status"], rec["verdicts"], rec["reasoning"] = fix
            n_fixed += 1
    records.sort(key=lambda r: r["id"])
    OUTPUT.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"[INFO] {len(records)} tasks, {n_fixed} corrected (of {len(CORRECTIONS)} flagged) -> {OUTPUT}")


if __name__ == "__main__":
    main()
