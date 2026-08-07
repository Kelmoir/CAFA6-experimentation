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

TASK_DESCRIPTION = (
    "You are reviewing structural/functional annotation evidence for a molecular-function GO term "
    "prediction task (CAFA6 protein function prediction challenge). You will be shown one GO term's "
    "definition and the annotated features for ONE specific protein already known to carry that "
    "term. Decide which ONE annotated feature (by type + exact coordinates, copied verbatim) most "
    "plausibly explains why that protein carries the term -- i.e. which feature is the "
    "structural/functional basis for that molecular function. If nothing plausibly explains it, say "
    "so honestly; do not force a match."
)

ANSWER_FORMAT = (
    "VERDICT: LIKELY or UNCERTAIN or NONE\n"
    "FEATURE: <TYPE> <START>-<END>  (use NONE 0-0 if verdict is NONE)\n"
    "REASONING: <one sentence, plain text>\n\n"
    "The (TYPE, START, END) you report MUST exactly match one of the feature lines in the protein's "
    "section verbatim (identical type string, identical integers) -- it will be looked up by exact "
    "match and discarded if it doesn't match, so don't paraphrase or round coordinates."
)

COT_ANSWER_FORMAT = (
    "THINKING: <work through it here -- consider each candidate feature against the GO term's "
    "definition and the genericity caution above, compare it to the alternatives, and only then "
    "settle on an answer. Can span as many lines as you need; there is no length limit.>\n"
    "VERDICT: LIKELY or UNCERTAIN or NONE\n"
    "FEATURE: <TYPE> <START>-<END>  (use NONE 0-0 if verdict is NONE)\n"
    "REASONING: <one sentence, plain text, summarizing the conclusion your THINKING reached>\n\n"
    "The (TYPE, START, END) you report MUST exactly match one of the feature lines in the protein's "
    "section verbatim (identical type string, identical integers) -- it will be looked up by exact "
    "match and discarded if it doesn't match, so don't paraphrase or round coordinates."
)


def build_system_prompt(chain_of_thought):
  """Two genuinely different instruction shapes, not one prompt with a bolted-on CoT phrase --
  demanding "exactly three lines, no commentary" while also asking a model to "think step by step"
  is self-contradicting, and forcing a weaker/non-reasoning model straight into a terse structured
  answer with no scratch space produces worse judgments than letting it reason in free text first.
  With chain_of_thought set: reasoning is scaffolded directly into the answer format itself, via a
  leading THINKING field, rather than left to a free-form "reason before you answer" instruction --
  a prior version of this prompt used the free-form instruction and testing showed models tend to
  skip straight to the terse answer anyway unless the output schema itself demands the reasoning
  field. VERDICT/FEATURE/REASONING still come after THINKING, in that order, as the LAST thing in
  the response (parse_response finds the last match of each field, so however long THINKING runs,
  the real answer is still found correctly). Without chain_of_thought: same terse contract as
  before, for models that already reason internally (e.g. via their own <think> block) or where
  speed matters more than a visible reasoning trace."""
  if chain_of_thought:
    return (
        TASK_DESCRIPTION + "\n\n" + GENERIC_CAUTION + "\n\n"
        + chain_of_thought.rstrip(". ") + " Respond using EXACTLY this format, in this order, "
        "starting with THINKING (nothing before it) and ending with REASONING (nothing after it):\n"
        + COT_ANSWER_FORMAT
    )
  return (
      TASK_DESCRIPTION + "\n\n" + GENERIC_CAUTION
      + "\n\nRespond with EXACTLY three lines, nothing else, no markdown formatting, no extra "
      "commentary before or after:\n" + ANSWER_FORMAT
  )


def call_local_llm(host, model, system_prompt, prompt, max_tokens, timeout=120):
  """Returns (content, finish_reason). finish_reason is the OpenAI-compatible API's own account of
  why generation stopped ("stop" = model concluded normally, "length" = it hit max_tokens mid-
  generation) -- authoritative, unlike guessing from the absence of expected fields in the text,
  which can't tell a genuine token-budget cutoff apart from a model that finished normally but
  didn't follow the requested format."""
  url = host.rstrip("/") + "/v1/chat/completions"
  payload = {
      "model": model,
      "messages": [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt},
      ],
      "temperature": 0.2,
      "max_tokens": max_tokens,
  }
  req = urllib.request.Request(
      url, data=json.dumps(payload).encode(), method="POST",
      headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"})
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    body = json.loads(resp.read().decode())
  choice = body["choices"][0]
  return choice["message"]["content"], choice.get("finish_reason")


def discover_model(host, timeout=10):
  url = host.rstrip("/") + "/v1/models"
  req = urllib.request.Request(url, headers={"Authorization": "Bearer lm-studio"})
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    body = json.loads(resp.read().decode())
  models = body.get("data", [])
  if not models:
    raise RuntimeError(f"No models reported loaded at {url} -- load a model in LM Studio first.")
  return models[0]["id"]


def _last_match(pattern, text, flags=0):
  matches = list(re.finditer(pattern, text, flags))
  return matches[-1] if matches else None


def parse_response(text):
  """Lenient extraction -- local models don't always follow formatting instructions exactly.
  Uses the LAST match of each field, not the first: with chain-of-thought enabled the model's
  reasoning text can easily mention "verdict" or reference the answer format conceptually before
  it actually commits to one, so searching from the front risks grabbing a stray earlier mention
  instead of the real final answer.
  Returns a 6th element, `unparsed`, so callers can warn: True means neither VERDICT nor FEATURE
  appeared anywhere in the text. This function has no access to the API's finish_reason, so it
  can't tell WHY on its own (genuine max_tokens cutoff vs. a normally-finished response that just
  didn't follow the requested format) -- callers that have finish_reason should use it to give a
  precise diagnosis instead of guessing; this function's own fallback reasoning text is
  deliberately neutral about the cause."""
  v = _last_match(r"VERDICT:\s*(LIKELY|UNCERTAIN|NONE)", text, re.I)
  f = _last_match(r"FEATURE:\s*(\S+)\s+(\d+)\s*-\s*(\d+)", text)
  r = _last_match(r"REASONING:\s*(.+)", text)
  unparsed = v is None and f is None
  verdict = v.group(1).upper() if v else "NONE"
  if f:
    ftype, start, end = f.group(1), f.group(2), f.group(3)
  else:
    ftype, start, end = "NONE", "0", "0"
  if unparsed:
    reasoning = "no VERDICT or FEATURE found anywhere in the response text"
  else:
    reasoning = r.group(1).strip() if r else "(no reasoning extracted from model response)"
  reasoning = reasoning.split("\n")[0][:400]  # keep it single-line, bounded
  return verdict, ftype, start, end, reasoning, unparsed


def extract_protein_section(review_text, accession):
  """Pull just the term header + this one protein's section out of a combined multi-protein
  review.md, instead of embedding the whole file. The combined-file format exists so Claude
  subagents reading the same term share one file, but a local model gets a fresh, isolated HTTP
  call per protein with no cross-call caching -- so every OTHER protein's section in the prompt is
  pure clutter: wasted context, slower generation, and (per the user's own testing motivation for
  trying a smaller non-reasoning model) a real risk of the model latching onto the wrong protein's
  features instead of the assigned one."""
  header_match = re.search(r"^(.*?)\n---\n", review_text, re.DOTALL)
  header = (header_match.group(1) if header_match else review_text.split("## Protein")[0]).strip()
  section_match = re.search(
      rf"## Protein {re.escape(accession)}\n(.*?)(?=\n## Protein |\Z)", review_text, re.DOTALL)
  if section_match:
    section = f"## Protein {accession}\n{section_match.group(1).strip()}"
  else:
    section = f"(protein {accession} section not found in review.md -- this is a bug, please report)"
  return f"{header}\n\n{section}"


def build_prompt(go_id, accession, review_text):
  """The chain-of-thought instruction (if any) lives entirely in the system prompt now -- no need
  to repeat it here, and repeating it in both places risks the two copies drifting or conflicting."""
  trimmed = extract_protein_section(review_text, accession)
  return (
      f"GO term and protein evidence for {go_id}, protein {accession}:\n\n"
      f"--- BEGIN EVIDENCE ---\n{trimmed}\n--- END EVIDENCE ---"
  )


def run(cmd):
  subprocess.run(cmd, check=True)


def plan_round():
  run([sys.executable, str(SCRIPTS / "plan_review_round.py")])
  manifest_path = ROOT / "_round_manifest.json"
  return json.loads(manifest_path.read_text())


def process_round(host, model, manifest, request_timeout, delay, max_tokens, system_prompt):
  new_terms = manifest.get("new_terms", {})
  touched = []
  for go_id, accessions in new_terms.items():
    term_dir = ROOT / go_id.replace(":", "_")
    review_text = (term_dir / "review.md").read_text()
    findings_path = term_dir / "findings.md"
    for accession in accessions:
      prompt = build_prompt(go_id, accession, review_text)
      finish_reason = None
      try:
        raw, finish_reason = call_local_llm(
            host, model, system_prompt, prompt, max_tokens, timeout=request_timeout)
      except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  ! {go_id} {accession}: request failed ({e}), logging as NONE and continuing")
        raw = "VERDICT: NONE\nFEATURE: NONE 0-0\nREASONING: local model request failed"
      verdict, ftype, start, end, reasoning, unparsed = parse_response(raw)
      if unparsed:
        debug_path = term_dir / "raw_debug.log"
        with open(debug_path, "a") as dbg:
          dbg.write(f"=== {accession} {go_id} (finish_reason={finish_reason}) ===\n{raw}\n\n")
        if finish_reason == "length":
          reasoning = (f"hit max_tokens ({max_tokens}) before VERDICT/FEATURE appeared -- "
                       f"raw text saved to {debug_path}")
        else:
          reasoning = (f"response ended (finish_reason={finish_reason}) without VERDICT/FEATURE -- "
                       f"raw text saved to {debug_path}, inspect for format drift")
        print(f"  ! {go_id} {accession}: unparsed response (finish_reason={finish_reason}, "
              f"{len(raw)} chars) -- raw text saved to {debug_path}")
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
  ap.add_argument("--chain-of-thought", type=str, default="",
                   help="Enables free-text reasoning before the answer -- recommended for "
                        "non-reasoning/smaller models, which judge more accurately with room to "
                        "think than when forced straight into the terse VERDICT/FEATURE/REASONING "
                        "format. Leave unset for models that already reason internally (e.g. via "
                        "their own <think> block) or when speed matters more than accuracy. "
                        "Example: --chain-of-thought \"Let's think step by step.\"")
  ap.add_argument("--max-tokens", type=int, default=1536,
                   help="Generation budget per protein. Raise this (not an LM Studio setting -- "
                        "it's a per-request parameter this script sends) if responses come back "
                        "truncated. Needs real headroom whenever --chain-of-thought is set: a full "
                        "reasoning trace plus the final answer easily runs past 1000 tokens, and a "
                        "reasoning-style model's own <think> preamble adds even more on top.")
  args = ap.parse_args()

  model = args.model or discover_model(args.host)
  system_prompt = build_system_prompt(args.chain_of_thought)
  print(f"Using model: {model} @ {args.host}"
        + (" (chain-of-thought enabled)" if args.chain_of_thought else ""))

  for round_n in range(1, args.rounds + 1):
    print(f"\n=== Round {round_n}/{args.rounds} ===")
    manifest = plan_round()
    total_agents = manifest.get("total_agents", 0)
    if total_agents == 0:
      print("No more terms fit the budget (backlog may be exhausted). Stopping.")
      break
    touched = process_round(args.host, model, manifest, args.request_timeout, args.delay,
                             args.max_tokens, system_prompt)
    merged_terms = merge_round(touched)
    run([sys.executable, str(SCRIPTS / "build_positional_mf_links.py")])
    commit_round(round_n, merged_terms, total_agents)
    print(f"Round {round_n} done. Merged: {', '.join(merged_terms) if merged_terms else 'none'}")


if __name__ == "__main__":
  main()
