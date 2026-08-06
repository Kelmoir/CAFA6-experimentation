"""Plan one round of the LLM term-review pipeline: 2 fresh agents per brand-new term, +1 more
agent for any not-yet-resolved term from a prior round -- up to 10 terms per round total.

A term is:
  - resolved:   positional_mf_llm_review_log.json[go_id]["merged"] is non-empty (>=2 subagents
                already agreed on something, and it made it into the dictionary). Done, skipped.
  - exhausted:  not resolved, and every sampled protein (up to MAX_PROTEINS=10) has already been
                assigned to a subagent. No more evidence available to try for agreement -- stamped
                "exhausted": true in the review log so this is queryable without re-deriving it,
                and permanently excluded from future rounds rather than retried forever.
  - unresolved: not resolved, and at least one sampled-but-unreviewed protein remains. Carried
                into this round for exactly one more agent (not a fresh 2, since findings.md
                already has prior rounds' blocks and only needs one more witness to try to reach
                the 2-agreement threshold -- parse_llm_findings.py/merge_llm_findings.py always
                re-derive from the full accumulated file, so a partial top-up is enough).

Fills up to 10 term slots per round: unresolved carry-overs first (1 new protein each), then
brand-new never-touched no_signal terms (2 fresh proteins each, via prepare_term_review_files.py)
for whatever slots remain. Total subagents per round is therefore n_carried*1 + n_new*2, which
lands in [10, 20] depending entirely on how many terms resolved or exhausted out of the prior
round's 10.

Writes experiments/llm_term_review/_round_manifest.json (this round's plan: which protein goes to
which term) and updates each term's _meta.json assigned_accessions in place. Does not spawn
subagents itself -- only prints the manifest for the orchestrating session to act on.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("experiments/llm_term_review")
VISITED_PATH = Path("experiments/positional_mf_visited_terms.json")
LOG_PATH = Path("experiments/positional_mf_llm_review_log.json")
TERMS_PER_ROUND = 10


def classify_existing_terms():
  log = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else {}
  resolved, exhausted, unresolved = [], [], []
  for d in sorted(ROOT.iterdir()):
    if not d.is_dir():
      continue
    go_id = d.name.replace("_", ":", 1)
    meta = json.loads((d / "_meta.json").read_text())
    n_sampled, n_assigned = len(meta["sampled_accessions"]), len(meta["assigned_accessions"])
    merged = log.get(go_id, {}).get("merged", [])
    if merged:
      resolved.append(go_id)
    elif n_assigned >= n_sampled:
      exhausted.append(go_id)
    else:
      unresolved.append(go_id)
  return resolved, exhausted, unresolved, log


def stamp_exhausted(log, exhausted_ids):
  changed = False
  for go_id in exhausted_ids:
    entry = log.setdefault(go_id, {"merged": [], "unmerged_agreements": [],
                                    "unconfirmed_singletons": [], "uncertain": [], "issues": []})
    if not entry.get("exhausted"):
      entry["exhausted"] = True
      changed = True
  if changed:
    LOG_PATH.write_text(json.dumps(dict(sorted(log.items())), indent=1))


def next_new_terms(n, existing_dirs):
  visited = json.loads(VISITED_PATH.read_text())
  no_signal = sorted(t for t, d in visited.items() if d.get("status") == "no_signal")
  picked = []
  for t in no_signal:
    if len(picked) >= n:
      break
    if (ROOT / t.replace(":", "_")).exists() or t in existing_dirs:
      continue
    picked.append(t)
  return picked


def main():
  resolved, exhausted, unresolved, log = classify_existing_terms()
  stamp_exhausted(log, exhausted)

  print(f"resolved: {len(resolved)}, exhausted: {len(exhausted)}, unresolved (carry-over): {len(unresolved)}")

  carried = sorted(unresolved)[:TERMS_PER_ROUND]
  n_new_needed = TERMS_PER_ROUND - len(carried)
  new_terms = next_new_terms(n_new_needed, set(resolved) | set(exhausted) | set(unresolved))

  manifest = {"carried_over": {}, "new_terms": {}}

  for go_id in carried:
    term_dir = ROOT / go_id.replace(":", "_")
    meta_path = term_dir / "_meta.json"
    meta = json.loads(meta_path.read_text())
    remaining = [a for a in meta["sampled_accessions"] if a not in meta["assigned_accessions"]]
    pick = remaining[0]
    meta["assigned_accessions"].append(pick)
    meta_path.write_text(json.dumps(meta, indent=2))
    manifest["carried_over"][go_id] = [pick]

  for go_id in new_terms:
    subprocess.run([sys.executable, "experiments/scripts/prepare_term_review_files.py", go_id], check=True)
    term_dir = ROOT / go_id.replace(":", "_")
    meta_path = term_dir / "_meta.json"
    meta = json.loads(meta_path.read_text())
    pick = meta["sampled_accessions"][:2]
    meta["assigned_accessions"] = pick
    meta_path.write_text(json.dumps(meta, indent=2))
    manifest["new_terms"][go_id] = pick

  total_agents = sum(len(v) for v in manifest["carried_over"].values()) + \
                 sum(len(v) for v in manifest["new_terms"].values())
  manifest["total_agents"] = total_agents

  manifest_path = ROOT / "_round_manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2))

  print(f"\nRound plan: {len(carried)} carried-over term(s), {len(new_terms)} new term(s), "
        f"{total_agents} subagent(s) total")
  for go_id, accs in manifest["carried_over"].items():
    print(f"  carry-over {go_id}: {accs}")
  for go_id, accs in manifest["new_terms"].items():
    print(f"  new         {go_id}: {accs}")
  print(f"-> {manifest_path}")


if __name__ == "__main__":
  main()
