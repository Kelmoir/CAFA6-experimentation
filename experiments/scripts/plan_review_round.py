"""Plan one round of the LLM term-review pipeline: fixed budget, one pass per term.

Each round picks TERMS_PER_ROUND brand-new (never-touched) no_signal terms, samples up to
MAX_PROTEINS (3, see prepare_term_review_files.py) proteins for each, and assigns the whole sample
at once -- one subagent per protein, min(3, num carriers) per term. There is no retry/carry-over:
once a term has a directory under experiments/llm_term_review/, plan_review_round.py's own
next_new_terms() skips it forever, regardless of whether that single pass reached merge agreement.
A term that comes back NONE/no-consensus after 3 independent reviews is treated as genuinely hard
rather than under-sampled -- an earlier version of this script kept feeding one more witness per
unresolved term per round (up to 10 sampled proteins total), but most retries were just re-hitting
the same NONE verdicts or the same already-rejected generic signal (see merge_llm_findings.py's
genericity gate), not converging.

Round size is therefore deterministic given the term pool: TERMS_PER_ROUND terms * up to 3 agents
each, fewer only when a term has fewer than 3 total carriers dataset-wide.

Writes experiments/llm_term_review/_round_manifest.json (this round's plan) and each new term's
_meta.json assigned_accessions (== its full sample, since the whole thing is reviewed at once now).
Does not spawn subagents itself -- only prints the manifest for the orchestrating session to act on.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("experiments/llm_term_review")
VISITED_PATH = Path("experiments/positional_mf_visited_terms.json")
TERMS_PER_ROUND = 10


def already_started():
  return {d.name.replace("_", ":", 1) for d in ROOT.iterdir() if d.is_dir()}


def next_new_terms(n, started):
  visited = json.loads(VISITED_PATH.read_text())
  no_signal = sorted(t for t, d in visited.items() if d.get("status") == "no_signal")
  picked = []
  for t in no_signal:
    if len(picked) >= n:
      break
    if t in started:
      continue
    picked.append(t)
  return picked


def main():
  started = already_started()
  print(f"already-searched terms (permanently out of scope): {len(started)}")

  new_terms = next_new_terms(TERMS_PER_ROUND, started)
  manifest = {"new_terms": {}}

  for go_id in new_terms:
    subprocess.run([sys.executable, "experiments/scripts/prepare_term_review_files.py", go_id], check=True)
    term_dir = ROOT / go_id.replace(":", "_")
    meta_path = term_dir / "_meta.json"
    meta = json.loads(meta_path.read_text())
    sample = meta["sampled_accessions"]
    meta["assigned_accessions"] = sample
    meta_path.write_text(json.dumps(meta, indent=2))
    manifest["new_terms"][go_id] = sample

  total_agents = sum(len(v) for v in manifest["new_terms"].values())
  manifest["total_agents"] = total_agents

  manifest_path = ROOT / "_round_manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2))

  print(f"\nRound plan: {len(new_terms)} term(s), {total_agents} subagent(s) total")
  for go_id, accs in manifest["new_terms"].items():
    print(f"  {go_id}: {accs}")
  print(f"-> {manifest_path}")


if __name__ == "__main__":
  main()
