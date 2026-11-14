"""Diff two local_llm_review_mutagen.py result files by task id, for comparing a prompt/flag
change (e.g. the bounded-scoring --chain-of-thought fix) against a prior run's results on the
same tasks -- built for the specific before/after workflow of renaming an existing
mutagen_review_results.jsonl to _old.jsonl and regenerating a fresh batch to diff against it.

Only prints tasks present in both files whose result actually changed (status, or the verdict set
-- a result is a set of {go_id, conditional} pairs now that multiple verdicts per task are
allowed) -- an unchanged result isn't interesting for this comparison.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_llm_review_mutagen import load_done  # noqa: E402

TASKS_PATH = Path(__file__).resolve().parent.parent / "mutagen_review_tasks.json"
DEFAULT_OLD = Path(__file__).resolve().parent.parent / "mutagen_review_results_old.jsonl"
DEFAULT_NEW = Path(__file__).resolve().parent.parent / "mutagen_review_results.jsonl"


def normalize(records: dict) -> dict:
    """Upgrades legacy single-verdict records ({"verdict", "conditional"}, pre multi-verdict
    schema) to the current {"status", "verdicts"} shape in place, so an old archived results file
    can still be diffed against a current run -- this project's own convention is to archive
    rather than delete prior exploration rounds, so comparisons spanning a schema change are
    expected to come up, not a one-off."""
    for rec in records.values():
        if "verdicts" in rec:
            continue
        verdict = rec.pop("verdict")
        conditional = rec.pop("conditional", False)
        if verdict == "NONE":
            rec["status"], rec["verdicts"] = "NONE", []
        elif verdict in ("UNPARSED", "INVALID_VERDICT"):
            rec["status"], rec["verdicts"] = verdict, []
        else:
            rec["status"] = "OK"
            rec["verdicts"] = [{"go_id": verdict, "conditional": conditional}]
    return records


def verdict_set(rec):
    return frozenset((v["go_id"], v["conditional"]) for v in rec.get("verdicts", []))


def format_verdicts(rec):
    vs = rec.get("verdicts", [])
    if not vs:
        return rec["status"]
    return ", ".join(f"{v['go_id']}{' (conditional)' if v['conditional'] else ''}" for v in vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, default=DEFAULT_OLD)
    ap.add_argument("--new", type=Path, default=DEFAULT_NEW)
    ap.add_argument("--tasks-path", type=Path, default=TASKS_PATH,
                     help="Override the tasks file used for context (note/candidates) printing -- "
                          "e.g. mutagen_review_tasks_leaf.json when diffing a --leaf-only run. "
                          "Defaults to the regular ancestor-expanded tasks.json, which prints the "
                          "WRONG candidate list for a leaf-only run's task ids.")
    args = ap.parse_args()

    tasks = {t["id"]: t for t in json.loads(args.tasks_path.read_text())}
    old = normalize(load_done(args.old))
    new = normalize(load_done(args.new))
    common_ids = sorted(set(old) & set(new))
    print(f"[INFO] {len(old)} old results, {len(new)} new results, {len(common_ids)} ids in both\n")

    changed = [i for i in common_ids
               if (old[i]["status"], verdict_set(old[i])) != (new[i]["status"], verdict_set(new[i]))]
    print(f"[INFO] {len(changed)}/{len(common_ids)} changed result\n")

    for i in changed:
        task = tasks.get(i)
        print("=" * 80)
        if task:
            print(f"id {i}  accession {task['accession']}  span {task['start']}-{task['end']}")
            print(f"note: {task['note']}")
            print("candidates: " + ", ".join(f"{c['go_id']} ({c['name']})" for c in task["candidates"]))
        else:
            print(f"id {i}  (task not found in {TASKS_PATH.name})")
        o, n = old[i], new[i]
        print(f"OLD: {format_verdicts(o)}  reasoning={o.get('reasoning', '')!r}")
        print(f"NEW: {format_verdicts(n)}  reasoning={n.get('reasoning', '')!r}")

    # Quick categorical breakdown -- which direction did changes mostly go.
    to_none = sum(1 for i in changed if verdict_set(old[i]) and not verdict_set(new[i]))
    from_none = sum(1 for i in changed if not verdict_set(old[i]) and verdict_set(new[i]))
    swapped = sum(1 for i in changed if verdict_set(old[i]) and verdict_set(new[i])
                   and verdict_set(old[i]) != verdict_set(new[i]))
    print("\n" + "=" * 80)
    print(f"[SUMMARY] gained an empty result: {to_none}  |  gained a non-empty result: {from_none}  "
          f"|  verdict set changed (swap/add/drop): {swapped}  |  other (parse-status changes "
          f"etc.): {len(changed) - to_none - from_none - swapped}")


if __name__ == "__main__":
    main()
