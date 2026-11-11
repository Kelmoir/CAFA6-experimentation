"""Diff two local_llm_review_mutagen.py result files by task id, for comparing a prompt/flag
change (e.g. the bounded-scoring --chain-of-thought fix) against a prior run's results on the
same tasks -- built for the specific before/after workflow of renaming an existing
mutagen_review_results.jsonl to _old.jsonl and regenerating a fresh batch to diff against it.

Only prints tasks present in both files whose verdict actually changed (including either side
being UNPARSED/INVALID_VERDICT) -- an unchanged verdict isn't interesting for this comparison.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, default=DEFAULT_OLD)
    ap.add_argument("--new", type=Path, default=DEFAULT_NEW)
    args = ap.parse_args()

    tasks = {t["id"]: t for t in json.loads(TASKS_PATH.read_text())}
    old = load_done(args.old)
    new = load_done(args.new)
    common_ids = sorted(set(old) & set(new))
    print(f"[INFO] {len(old)} old results, {len(new)} new results, {len(common_ids)} ids in both\n")

    changed = [i for i in common_ids
               if (old[i]["verdict"], old[i].get("conditional")) != (new[i]["verdict"], new[i].get("conditional"))]
    print(f"[INFO] {len(changed)}/{len(common_ids)} changed verdict or conditional flag\n")

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
        print(f"OLD: verdict={o['verdict']}  conditional={o.get('conditional')}  "
              f"reasoning={o.get('reasoning', '')!r}")
        print(f"NEW: verdict={n['verdict']}  conditional={n.get('conditional')}  "
              f"reasoning={n.get('reasoning', '')!r}")

    # Quick categorical breakdown -- which direction did changes mostly go.
    to_none = sum(1 for i in changed if old[i]["verdict"] != "NONE" and new[i]["verdict"] == "NONE")
    from_none = sum(1 for i in changed if old[i]["verdict"] == "NONE" and new[i]["verdict"] != "NONE")
    other_go_id = sum(1 for i in changed if old[i]["verdict"] not in ("NONE", "UNPARSED", "INVALID_VERDICT")
                       and new[i]["verdict"] not in ("NONE", "UNPARSED", "INVALID_VERDICT")
                       and old[i]["verdict"] != new[i]["verdict"])
    print("\n" + "=" * 80)
    print(f"[SUMMARY] changed to NONE: {to_none}  |  changed away from NONE: {from_none}  |  "
          f"swapped to a different GO id: {other_go_id}  |  other (parse-status changes etc.): "
          f"{len(changed) - to_none - from_none - other_go_id}")


if __name__ == "__main__":
    main()
