"""Standalone driver for the LLM term-review pipeline, powered by a local model (LM Studio's
OpenAI-compatible server) instead of Claude subagents.

Why this exists: the review loop up to this point ran as Claude Code subagents spawned one call
per protein, which is fast and reliable but burns through session usage quickly at the rate this
loop consumes them (up to 10 subagents/round). This script reuses every piece of the existing
pipeline unchanged (plan_review_round.py for term/protein selection, parse_llm_findings.py +
merge_llm_findings.py for the agreement-gated merge, build_positional_mf_links.py for coverage) --
only the "ask an LLM to judge one protein" step is swapped for an HTTP call to a local model
instead of an Agent-tool subagent. No network egress needed beyond localhost/LAN to the LM Studio
host, so it runs entirely unattended, as slowly as the local model allows.

Run this ON THE MACHINE THAT HOSTS LM STUDIO (or anywhere with network access to it), NOT inside
the Claude Code sandbox -- that sandbox's network namespace has no route to the host's LAN, which
is exactly why this script exists as something the user runs directly rather than something Claude
Code calls out to.

Usage:
  python3 experiments/scripts/local_llm_review_loop.py --host http://192.168.1.122:1234 --rounds 10

Each round: plan (agent-budget-capped term/protein selection) -> one HTTP completion per assigned
protein -> parse + merge every touched term -> rebuild links -> git commit -> short report (which
GO terms merged). Safe to Ctrl-C between rounds; each round's git commit is a checkpoint, and
plan_review_round.py never revisits an already-started term, so re-running this script later just
continues where it left off.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("experiments/llm_term_review")
SCRIPTS = Path("experiments/scripts")

GENERIC_CAUTION = (
    "General caution -- generic signals: many proteins carry real but non-diagnostic biochemical "
    "machinery: common cofactor-binding sites (ATP/GTP/ADP/GDP/AMP/NAD(P)H/heme/common metal ions), "
    "broad structural folds or catalytic domains shared across dozens of unrelated protein families, "
    "generic catalytic-residue labels (e.g. a bare \"Nucleophile\"/\"Proton acceptor\" note used across "
    "countless unrelated enzyme families), or disordered regions. These are real features but they "
    "explain routine biochemistry, not this SPECIFIC molecular function. Before marking LIKELY, ask: "
    "is this feature actually distinctive to this GO term, or would many unrelated proteins/functions "
    "also have it? A downstream script independently rejects overly generic signals (common cofactors, "
    "domains shared by >15 proteins dataset-wide, single generic words) even if you mark them LIKELY -- "
    "so if the only candidate is generic, it's more useful and honest to mark UNCERTAIN (or NONE) and "
    "say so, rather than claim confidence in something that won't hold up anyway."
)

SYSTEM_PROMPT = (
    "You are reviewing structural/functional annotation evidence for a molecular-function GO term "
    "prediction task (CAFA6 protein function prediction challenge). You will be shown a Markdown "
    "digest for one GO term (its definition, followed by a section per protein that really carries "
    "the term) and told which ONE protein you are responsible for. Decide which ONE annotated "
    "feature (by type + exact coordinates, copied verbatim) most plausibly explains why that protein "
    "carries the term -- i.e. which feature is the structural/functional basis for that molecular "
    "function. If nothing plausibly explains it, say so honestly; do not force a match.\n\n"
    + GENERIC_CAUTION
    + "\n\nRespond with EXACTLY three lines, nothing else, no markdown formatting, no extra "
    "commentary before or after:\n"
    "VERDICT: LIKELY or UNCERTAIN or NONE\n"
    "FEATURE: <TYPE> <START>-<END>  (use NONE 0-0 if verdict is NONE)\n"
    "REASONING: <one sentence, plain text>\n\n"
    "The (TYPE, START, END) you report MUST exactly match one of the feature lines in your assigned "
    "protein's section verbatim (identical type string, identical integers) -- it will be looked up "
    "by exact match and discarded if it doesn't match, so don't paraphrase or round coordinates."
)


def call_local_llm(host, model, prompt, timeout=120):
  url = host.rstrip("/") + "/v1/chat/completions"
  payload = {
      "model": model,
      "messages": [
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": prompt},
      ],
      "temperature": 0.2,
      "max_tokens": 300,
  }
  req = urllib.request.Request(
      url, data=json.dumps(payload).encode(), method="POST",
      headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"})
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    body = json.loads(resp.read().decode())
  return body["choices"][0]["message"]["content"]


def discover_model(host, timeout=10):
  url = host.rstrip("/") + "/v1/models"
  req = urllib.request.Request(url, headers={"Authorization": "Bearer lm-studio"})
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    body = json.loads(resp.read().decode())
  models = body.get("data", [])
  if not models:
    raise RuntimeError(f"No models reported loaded at {url} -- load a model in LM Studio first.")
  return models[0]["id"]


def parse_response(text):
  """Lenient extraction -- local models don't always follow formatting instructions exactly."""
  v = re.search(r"VERDICT:\s*(LIKELY|UNCERTAIN|NONE)", text, re.I)
  f = re.search(r"FEATURE:\s*(\S+)\s+(\d+)\s*-\s*(\d+)", text)
  r = re.search(r"REASONING:\s*(.+)", text)
  verdict = v.group(1).upper() if v else "NONE"
  if f:
    ftype, start, end = f.group(1), f.group(2), f.group(3)
  else:
    ftype, start, end = "NONE", "0", "0"
  reasoning = r.group(1).strip() if r else "(no reasoning extracted from model response)"
  reasoning = reasoning.split("\n")[0][:400]  # keep it single-line, bounded
  return verdict, ftype, start, end, reasoning


def build_prompt(go_id, accession, review_text):
  return (
      f"Here is the review file for {go_id}:\n\n"
      f"--- BEGIN REVIEW FILE ---\n{review_text}\n--- END REVIEW FILE ---\n\n"
      f"Your assigned protein is {accession}. Find its \"## Protein {accession}\" section above "
      f"and focus only on that section (ignore the other proteins' sections)."
  )


def run(cmd):
  subprocess.run(cmd, check=True)


def plan_round():
  run([sys.executable, str(SCRIPTS / "plan_review_round.py")])
  manifest_path = ROOT / "_round_manifest.json"
  return json.loads(manifest_path.read_text())


def process_round(host, model, manifest, request_timeout, delay):
  new_terms = manifest.get("new_terms", {})
  touched = []
  for go_id, accessions in new_terms.items():
    term_dir = ROOT / go_id.replace(":", "_")
    review_text = (term_dir / "review.md").read_text()
    findings_path = term_dir / "findings.md"
    for accession in accessions:
      prompt = build_prompt(go_id, accession, review_text)
      try:
        raw = call_local_llm(host, model, prompt, timeout=request_timeout)
      except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  ! {go_id} {accession}: request failed ({e}), logging as NONE and continuing")
        raw = "VERDICT: NONE\nFEATURE: NONE 0-0\nREASONING: local model request failed"
      verdict, ftype, start, end, reasoning = parse_response(raw)
      block = (
          f"## {accession} {go_id}\n"
          f"verdict: {verdict}\n"
          f"feature: {ftype} {start}-{end}\n"
          f"reasoning: {reasoning}\n"
          f"---\n"
      )
      with open(findings_path, "a") as fh:
        fh.write(block)
      print(f"  {go_id} {accession}: {verdict} ({ftype} {start}-{end})")
      if delay:
        time.sleep(delay)
    touched.append(go_id)
  return touched


def merge_round(touched):
  merged_terms = []
  for go_id in touched:
    run([sys.executable, str(SCRIPTS / "parse_llm_findings.py"), go_id])
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "merge_llm_findings.py"), go_id],
        check=True, capture_output=True, text=True)
    print(result.stdout, end="")
    if re.match(rf"^{re.escape(go_id)}: merged [1-9]", result.stdout):
      merged_terms.append(go_id)
  return merged_terms


def commit_round(round_n, merged_terms, total_agents):
  run(["git", "add", "experiments/llm_term_review/", "experiments/positional_mf_gaps.json",
       "experiments/positional_mf_links.json", "experiments/positional_mf_llm_review_log.json",
       "experiments/positional_mf_localization_dictionary.json",
       "experiments/positional_mf_spotcheck_sample.json"])
  status = subprocess.run(["git", "diff", "--cached", "--quiet"])
  if status.returncode == 0:
    print("  (nothing to commit this round)")
    return
  merged_line = ", ".join(merged_terms) if merged_terms else "none"
  msg = (
      f"feat: local-model LLM review round {round_n} -- {total_agents} proteins, "
      f"merged: {merged_line}\n\n"
      f"Generated by local_llm_review_loop.py against a local LM Studio model, unattended.\n"
      f"Merged this round: {merged_line}."
  )
  run(["git", "commit", "-m", msg])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--host", required=True, help="LM Studio server, e.g. http://192.168.1.122:1234")
  ap.add_argument("--model", default=None, help="Model id; auto-detected from /v1/models if omitted")
  ap.add_argument("--rounds", type=int, default=10)
  ap.add_argument("--request-timeout", type=int, default=180, help="Seconds per completion call")
  ap.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between calls")
  args = ap.parse_args()

  model = args.model or discover_model(args.host)
  print(f"Using model: {model} @ {args.host}")

  for round_n in range(1, args.rounds + 1):
    print(f"\n=== Round {round_n}/{args.rounds} ===")
    manifest = plan_round()
    total_agents = manifest.get("total_agents", 0)
    if total_agents == 0:
      print("No more terms fit the budget (backlog may be exhausted). Stopping.")
      break
    touched = process_round(args.host, model, manifest, args.request_timeout, args.delay)
    merged_terms = merge_round(touched)
    run([sys.executable, str(SCRIPTS / "build_positional_mf_links.py")])
    commit_round(round_n, merged_terms, total_agents)
    print(f"Round {round_n} done. Merged: {', '.join(merged_terms) if merged_terms else 'none'}")


if __name__ == "__main__":
  main()
