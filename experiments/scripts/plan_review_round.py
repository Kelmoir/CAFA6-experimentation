"""Plan one round of the LLM term-review pipeline: fixed budget, one pass per term, capped at
MAX_AGENTS_PER_ROUND subagents total.

Each round fills an agent budget (not a term count) with brand-new (never-touched) no_signal
terms: for each candidate term, sample up to MAX_PROTEINS (5, see prepare_term_review_files.py)
proteins, and only commit to the term if its sample fits in whatever budget remains. A term that
doesn't fit is left untouched (no directory created) and reconsidered fresh next round -- terms
are never partially reviewed. There is no retry/carry-over: once a term has a directory under
experiments/llm_term_review/, next_new_terms() skips it forever, regardless of whether that single
pass reached merge agreement (see the module history for why -- most retries were just re-hitting
the same NONE verdicts or the same already-rejected generic signal, not converging).

Writes experiments/llm_term_review/_round_manifest.json (this round's plan) and each new term's
_meta.json assigned_accessions (== its full sample, since the whole thing is reviewed at once now).
Does not spawn subagents itself -- only prints the manifest for the orchestrating session to act on.
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("experiments/llm_term_review")
VISITED_PATH = Path("experiments/positional_mf_visited_terms.json")
RAW_FEATURES_PATH = Path("experiments/positional_mf_raw_features.json")
MAX_AGENTS_PER_ROUND = 10
MAX_PROTEINS_PER_TERM = 5


def already_started():
  return {d.name.replace("_", ":", 1) for d in ROOT.iterdir() if d.is_dir()}


def count_carriers():
  """go_id -> count of proteins carrying it as a real aspect=F annotation, one pass over
  raw_features.json rather than re-scanning per candidate term."""
  raw_features = json.loads(RAW_FEATURES_PATH.read_text())
  counts = defaultdict(int)
  for rec in raw_features.values():
    for g in rec["go_terms"]:
      if g["aspect"] == "F":
        counts[g["go_id"]] += 1
  return counts


def pick_round_terms(budget, started, carrier_counts):
  visited = json.loads(VISITED_PATH.read_text())
  no_signal = sorted(t for t, d in visited.items() if d.get("status") == "no_signal")
  picked = []
  for t in no_signal:
    if budget <= 0:
      break
    if t in started:
      continue
    n = min(MAX_PROTEINS_PER_TERM, carrier_counts.get(t, 0))
    if n == 0 or n > budget:
      continue
    picked.append(t)
    budget -= n
  return picked


def main():
  started = already_started()
  print(f"already-searched terms (permanently out of scope): {len(started)}")

  carrier_counts = count_carriers()
  round_terms = pick_round_terms(MAX_AGENTS_PER_ROUND, started, carrier_counts)
  manifest = {"new_terms": {}}

  for go_id in round_terms:
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

  print(f"\nRound plan: {len(round_terms)} term(s), {total_agents} subagent(s) total (cap {MAX_AGENTS_PER_ROUND})")
  for go_id, accs in manifest["new_terms"].items():
    print(f"  {go_id}: {accs}")
  print(f"-> {manifest_path}")


if __name__ == "__main__":
  main()
