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
protein -> parse each touched term -> retry any accession whose response was genuinely invalid
(unresolved pointer, malformed verdict, wrong go_id -- not a decisive NONE, that's a real answer)
once, re-parse if anything was retried -> merge -> rebuild links -> git commit -> short report
(which GO terms merged). Safe to Ctrl-C between rounds; each round's git commit is a checkpoint,
and plan_review_round.py never revisits an already-started term, so re-running this script later
just continues where it left off.
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

THINKING_INSTRUCTIONS = (
    "In the THINKING block, follow these steps in order:\n"
    "- Restate the GO term's id, name, and Definition in your own words.\n"
    "- List every candidate annotated feature for this protein -- not just binding sites: DOMAIN, "
    "ACT_SITE, DNA_BIND, MOTIF, REGION, and any other annotated type are all candidates, exactly as "
    "diverse as the feature types actually listed under \"Annotated features\" for this protein.\n"
    "- For each candidate feature, in turn:\n"
    "  - Compare its type and qualifier text against the GO term's Definition, anchoring on the "
    "Definition's PRIMARY stated reaction or mechanism (see the definition-focus note below -- a "
    "candidate that only echoes a word from a secondary/qualifying clause does not count as a "
    "strong match).\n"
    "  - Rate how specifically it matches that primary reaction, on a scale of 1 (generic, could "
    "belong to almost any protein) to 10 (names or is mechanistically specific to this exact GO "
    "term).\n"
    "- When every candidate has been rated, use the highest score to decide: 7-10 -> LIKELY, a "
    "genuine near-tie between two comparably-plausible candidates -> UNCERTAIN, everything else (no "
    "candidates, or all candidates generic/low-scoring) -> NONE. Then proceed to VERDICT, FEATURE, "
    "REASONING below.\n"
)


DEFINITION_FOCUS_WARNING = (
    "Definition focus -- some GO definitions contain a secondary or qualifying clause (e.g. \"many "
    "but not all X act as Y\", \"which may also...\") describing a downstream consequence that "
    "applies to only SOME proteins carrying the term, not the defining chemistry itself. Anchor "
    "your judgment on the definition's PRIMARY stated reaction or mechanism (usually the main "
    "clause -- what binds what, what is transferred to what, what bond is formed or broken), not "
    "on whichever feature happens to echo a word from a secondary clause. Concretely: a GO term "
    "about receiving a phosphoryl transfer is about the receiving domain, even if the definition "
    "also notes that some carriers of the term go on to regulate transcription -- a separate DNA-"
    "binding domain in the same protein explains that secondary, optional consequence, not the "
    "phosphorelay-receiving activity the term actually defines. When a protein has multiple "
    "candidate domains, prefer whichever one implements the definition's primary reaction over one "
    "that merely resembles a word from a secondary clause."
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

FEATURE_SOURCE_WARNING = (
    "FEATURE must be copied verbatim from a line under \"Annotated features (position, type, "
    "qualifiers)\" -- the exact TYPE string and the exact two integers of one of those lines. Never "
    "make up a span or reuse a number from anywhere else, including any example number written "
    "elsewhere in this prompt -- an example is illustrating the FORMAT, not a real coordinate for "
    "this protein. It must NEVER come from the separate \"Domain/family cross-references (no "
    "position...)\" list further down: those are InterPro/Pfam/PROSITE accessions (like "
    "\"InterPro:IPR000276\" or \"Pfam:PF00001\") that carry no coordinates at all -- they exist only "
    "as background context to help you identify what a domain is, not as something you can report "
    "as a FEATURE. This holds even when that cross-reference list strongly suggests a specific, "
    "well-known domain family (e.g. \"MFS_dom\" for a sugar transporter) and the protein's own "
    "Annotated features section has nothing that actually matches it (empty, or only a generic/"
    "irrelevant type like a Disordered REGION): don't invent a plausible-looking span for a domain "
    "family just because the cross-reference list names one. If the only evidence you find "
    "convincing lives in that cross-reference list, or the Annotated features section is empty, "
    "that means no *positional* feature explains this term -- say so (UNCERTAIN or NONE)."
)

ANSWER_FORMAT = (
    "VERDICT: LIKELY or UNCERTAIN or NONE\n"
    "FEATURE: <TYPE> <START>-<END>  (use NONE 0-0 if verdict is NONE)\n"
    "REASONING: <one sentence, plain text>\n\n"
    + FEATURE_SOURCE_WARNING + "\n\n"
    "The (TYPE, START, END) you report MUST exactly match one of the feature lines in the protein's "
    "section verbatim (identical type string, identical integers) -- it will be looked up by exact "
    "match and discarded if it doesn't match, so don't paraphrase or round coordinates."
)

COT_ANSWER_FORMAT = (
    "THINKING: <follow the numbered steps given above -- list and rate every candidate feature in "
    "turn, then state your best candidate. End THINKING with one explicit sentence naming your "
    "single best candidate feature (or stating there is none) and saying plainly whether it "
    "specifically matches the GO term's own chemistry -- only THEN settle on an answer below. Can "
    "span as many lines as you need; there is no length limit.>\n"
    "VERDICT: LIKELY or UNCERTAIN or NONE\n"
    "FEATURE: <TYPE> <START>-<END>  (use NONE 0-0 if verdict is NONE)\n"
    "REASONING: <one sentence, plain text, summarizing the conclusion your THINKING reached>\n\n"
    + FEATURE_SOURCE_WARNING + "\n\n"
    "The (TYPE, START, END) you report MUST exactly match one of the feature lines in the protein's "
    "section verbatim (identical type string, identical integers) -- it will be looked up by exact "
    "match and discarded if it doesn't match, so don't paraphrase or round coordinates.\n"
    "Once again, all of the four output qualifiers THINKING: VERDICT: FEATURE: REASONING: "
    "must be used in this order!"
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
        TASK_DESCRIPTION + "\n\n" + THINKING_INSTRUCTIONS + "\n\n"
        + DEFINITION_FOCUS_WARNING + "\n\n"
        + chain_of_thought.rstrip(". ") + ". Respond using EXACTLY this format, in this order, "
        "starting with THINKING (nothing before it) and ending with REASONING (nothing after it):\n"
        + COT_ANSWER_FORMAT
    )
  return (
      TASK_DESCRIPTION + "\n\n"
      + DEFINITION_FOCUS_WARNING
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
  All three field labels are matched case-insensitively (re.I) -- an eval run against bonsai-8b
  showed VERDICT/FEATURE landing fine but REASONING coming back empty on 3 of 6 responses, and
  VERDICT was the only one of the three with re.I originally; a model that's inconsistent about
  label casing (e.g. "Reasoning:" instead of "REASONING:") would silently lose just that one field
  under the old case-sensitive regex, without ever showing up as a fully `unparsed` response since
  VERDICT/FEATURE still matched.
  Returns a 6th element, `unparsed`, so callers can warn: True means neither VERDICT nor FEATURE
  appeared anywhere in the text. This function has no access to the API's finish_reason, so it
  can't tell WHY on its own (genuine max_tokens cutoff vs. a normally-finished response that just
  didn't follow the requested format) -- callers that have finish_reason should use it to give a
  precise diagnosis instead of guessing; this function's own fallback reasoning text is
  deliberately neutral about the cause.
  Whenever no valid VERDICT: line was found at all (regardless of whether FEATURE happened to
  parse), the verdict is reported as the sentinel "UNPARSED", NOT "NONE" -- these look the same to
  a human skimming findings.md but are not the same thing downstream: parse_llm_findings.py buckets
  a genuine "NONE" as a decisive, real answer (issue reason none_verdict, deliberately excluded
  from RETRYABLE_ISSUE_REASONS -- retrying a real answer would be wrong), whereas any other
  unrecognized verdict string buckets as invalid_verdict, which IS retryable. Using "NONE" as the
  no-VERDICT fallback (an earlier version of this function did) meant a response that never
  produced a valid VERDICT silently masqueraded as a deliberate NONE and could never be retried,
  even though its raw text was already being saved to raw_debug.log the whole time -- logged for
  human diagnosis, but invisible to the automatic retry pass."""
  v = _last_match(r"VERDICT:\s*(LIKELY|UNCERTAIN|NONE)", text, re.I)
  f = _last_match(r"FEATURE:\s*(\S+)\s+(\d+)\s*-\s*(\d+)", text, re.I)
  r = _last_match(r"REASONING:\s*(.+)", text, re.I)
  unparsed = v is None and f is None
  verdict = v.group(1).upper() if v else "UNPARSED"
  if f:
    ftype, start, end = f.group(1), f.group(2), f.group(3)
  else:
    ftype, start, end = "NONE", "0", "0"
  reasoning_missing = not unparsed and r is None
  if unparsed:
    reasoning = "no VERDICT or FEATURE found anywhere in the response text"
  else:
    reasoning = r.group(1).strip() if r else "(no reasoning extracted from model response)"
  reasoning = reasoning.split("\n")[0][:400]  # keep it single-line, bounded
  return verdict, ftype, start, end, reasoning, unparsed, reasoning_missing


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


NO_FEATURES_MARKER = "(none -- this protein has no MF-relevant positional features on record)"

CROSSREF_FEATURE_RE = r"FEATURE:\s*(InterPro|Pfam|PROSITE)\s*:?\s*([A-Za-z0-9_.]+)"


def _tokens(text):
  """Mirrors merge_llm_findings.py's normalize() -- same token-overlap test as that script's
  resolve_domain_accessions, just run in the opposite direction below (cross-reference description
  -> DOMAIN note, instead of DOMAIN note -> cross-reference description)."""
  return set(re.split(r"[\s\-_,]+", text.lower())) - {""}


def resolve_crossref_feature(protein_section, raw_text):
  """Despite FEATURE_SOURCE_WARNING, the model sometimes still names a cross-reference accession
  (e.g. "FEATURE: InterPro:IPR009081") instead of a real position -- exactly the failure mode that
  warning exists to prevent. Rather than discard that finding as unresolved, try to recover the
  real DOMAIN feature it's actually pointing at: look the accession up in this protein's own
  "Domain/family cross-references" list to get its short description, then token-overlap that
  description against every DOMAIN feature's note. Only resolves when EXACTLY ONE DOMAIN feature
  matches -- an accession that's ambiguous (matches 0 or >1 DOMAIN notes) falls through to the
  normal unresolved-pointer/retry path instead of guessing. Returns (ftype, start, end) or None."""
  m = _last_match(CROSSREF_FEATURE_RE, raw_text, re.I)
  if not m:
    return None
  db, acc = m.group(1), m.group(2)
  xref = re.search(rf"-\s*{re.escape(db)}:{re.escape(acc)}\s*--\s*(.+)", protein_section, re.I)
  if not xref:
    return None
  desc_tokens = _tokens(xref.group(1))
  if not desc_tokens:
    return None
  candidates = []
  for feat in re.finditer(r"-\s*DOMAIN\s+(\d+)-(\d+)\n((?:    .+\n)*)", protein_section):
    start, end, quals = feat.groups()
    note = re.search(r"note:\s*(.+)", quals, re.I)
    if note and desc_tokens & _tokens(note.group(1)):
      candidates.append(("DOMAIN", start, end))
  return candidates[0] if len(candidates) == 1 else None


def build_prompt(go_id, accession, protein_section):
  """The chain-of-thought instruction (if any) lives entirely in the system prompt now -- no need
  to repeat it here, and repeating it in both places risks the two copies drifting or conflicting."""
  return (
      f"GO term and protein evidence for {go_id}, protein {accession}:\n\n"
      f"--- BEGIN EVIDENCE ---\n{protein_section}\n--- END EVIDENCE ---"
  )


def run(cmd):
  subprocess.run(cmd, check=True)


def plan_round():
  run([sys.executable, str(SCRIPTS / "plan_review_round.py")])
  manifest_path = ROOT / "_round_manifest.json"
  return json.loads(manifest_path.read_text())


RETRYABLE_ISSUE_REASONS = {"unresolved_pointer", "invalid_verdict", "go_id_mismatch"}


def review_protein(host, model, system_prompt, go_id, accession, review_text, term_dir,
                    findings_path, request_timeout, max_tokens):
  """One LLM call for one (go_id, accession), appended to findings.md as a block. Shared by
  process_round (first pass) and retry_issues (second-chance pass on invalid responses) so both
  go through the exact same call/parse/debug-log/append logic. Returns (verdict, ftype, start,
  end, reasoning, unparsed) too, for callers (e.g. an eval harness comparing against known-good
  ground truth) that need the parsed result without re-reading findings.md back.
  Also runs resolve_crossref_feature() as a fallback whenever the model committed to a verdict but
  FEATURE didn't parse to a real position -- see that function's docstring. This happens here,
  not in parse_response, because resolution needs the protein's own review-text section (to look
  up the cross-reference's description and the candidate DOMAIN notes), which parse_response
  deliberately doesn't have access to.
  Short-circuits to a deterministic NONE, with no LLM call at all, when the protein has zero
  annotated positional features on record (NO_FEATURES_MARKER). Found necessary against real
  output: a protein with no real feature but a suggestive cross-reference-only domain family (e.g.
  "MFS_dom") got LIKELY verdicts hallucinating a "DOMAIN 19-119" span that doesn't exist anywhere
  on that protein's record -- and did it for multiple different proteins reviewed under the same
  term, which looked like independent corroborating agreement but was really the same prompt
  eliciting the same fabrication (traced to FEATURE_SOURCE_WARNING's own now-fixed illustrative
  example, which used that exact "19-119" span). Skipping the call entirely when there is
  categorically nothing to point at removes this failure mode by construction rather than hoping
  the prompt is followed, and saves a request besides."""
  protein_section = extract_protein_section(review_text, accession)
  if NO_FEATURES_MARKER in protein_section:
    verdict, ftype, start, end = "NONE", "NONE", "0", "0"
    reasoning = "protein has zero annotated positional features on record -- nothing to point at"
    block = (
        f"## {accession} {go_id}\n"
        f"verdict: {verdict}\n"
        f"feature: {ftype} {start}-{end}\n"
        f"reasoning: {reasoning}\n"
        f"---\n"
    )
    with open(findings_path, "a") as fh:
      fh.write(block)
    print(f"  {go_id} {accession}: NONE (no annotated features on record, skipped LLM call)")
    return verdict, ftype, start, end, reasoning, False
  prompt = build_prompt(go_id, accession, protein_section)
  finish_reason = None
  try:
    raw, finish_reason = call_local_llm(
        host, model, system_prompt, prompt, max_tokens, timeout=request_timeout)
  except (urllib.error.URLError, TimeoutError, OSError) as e:
    print(f"  ! {go_id} {accession}: request failed ({e}), logging as NONE and continuing")
    raw = "VERDICT: NONE\nFEATURE: NONE 0-0\nREASONING: local model request failed"
  verdict, ftype, start, end, reasoning, unparsed, reasoning_missing = parse_response(raw)
  debug_path = term_dir / "raw_debug.log"
  auto_resolved = False
  if verdict in ("LIKELY", "UNCERTAIN") and ftype == "NONE" and start == "0" and end == "0":
    resolved = resolve_crossref_feature(protein_section, raw)
    if resolved:
      ftype, start, end = resolved
      auto_resolved = True
      reasoning = f"[auto-resolved from cross-reference accession] {reasoning}"
      print(f"  {go_id} {accession}: auto-resolved cross-reference FEATURE to {ftype} {start}-{end}")
  if verdict == "UNPARSED" or reasoning_missing or auto_resolved:
    with open(debug_path, "a") as dbg:
      if auto_resolved:
        tag = "auto-resolved cross-ref"
      elif unparsed:
        tag = f"finish_reason={finish_reason}"
      elif verdict == "UNPARSED":
        tag = "VERDICT missing"
      else:
        tag = "reasoning missing"
      dbg.write(f"=== {accession} {go_id} ({tag}) ===\n{raw}\n\n")
  if auto_resolved:
    pass
  elif unparsed:
    if finish_reason == "length":
      reasoning = (f"hit max_tokens ({max_tokens}) before VERDICT/FEATURE appeared -- "
                   f"raw text saved to {debug_path}")
    else:
      reasoning = (f"response ended (finish_reason={finish_reason}) without VERDICT/FEATURE -- "
                   f"raw text saved to {debug_path}, inspect for format drift")
    print(f"  ! {go_id} {accession}: unparsed response (finish_reason={finish_reason}, "
          f"{len(raw)} chars) -- raw text saved to {debug_path}")
  elif verdict == "UNPARSED":
    reasoning = f"FEATURE parsed but no valid VERDICT found -- raw text saved to {debug_path}"
    print(f"  ! {go_id} {accession}: VERDICT missing from an otherwise-parsed response -- "
          f"raw text saved to {debug_path}")
  elif reasoning_missing:
    reasoning = f"VERDICT/FEATURE parsed but no REASONING found -- raw text saved to {debug_path}"
    print(f"  ! {go_id} {accession}: REASONING missing from an otherwise-parsed response -- "
          f"raw text saved to {debug_path}")
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
  return verdict, ftype, start, end, reasoning, unparsed


def process_round(host, model, manifest, request_timeout, delay, max_tokens, system_prompt):
  new_terms = manifest.get("new_terms", {})
  touched = []
  for go_id, accessions in new_terms.items():
    term_dir = ROOT / go_id.replace(":", "_")
    review_text = (term_dir / "review.md").read_text()
    findings_path = term_dir / "findings.md"
    for accession in accessions:
      review_protein(host, model, system_prompt, go_id, accession, review_text, term_dir,
                      findings_path, request_timeout, max_tokens)
      if delay:
        time.sleep(delay)
    touched.append(go_id)
  return touched


def retry_issues(host, model, system_prompt, go_id, request_timeout, max_tokens):
  """Give a term's genuinely invalid responses (not decisive NONE verdicts -- those are real
  answers, not garbled ones) one extra shot. Reads the issues parse_llm_findings.py just logged,
  strips each retryable accession's block out of findings.md so it isn't double-counted, and
  re-queries it fresh. Returns how many accessions were retried, so the caller knows whether a
  re-parse is worthwhile."""
  term_dir = ROOT / go_id.replace(":", "_")
  parsed_path = term_dir / "parsed_findings.json"
  if not parsed_path.exists():
    return 0
  parsed = json.loads(parsed_path.read_text())
  to_retry = sorted({i["accession"] for i in parsed.get("issues", [])
                      if i.get("reason") in RETRYABLE_ISSUE_REASONS})
  if not to_retry:
    return 0

  findings_path = term_dir / "findings.md"
  text = findings_path.read_text()
  for acc in to_retry:
    text = re.sub(
        rf"## {re.escape(acc)} {re.escape(go_id)}\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)
  findings_path.write_text(text)

  review_text = (term_dir / "review.md").read_text()
  for acc in to_retry:
    print(f"  retrying {go_id} {acc} (previously invalid)")
    review_protein(host, model, system_prompt, go_id, acc, review_text, term_dir,
                    findings_path, request_timeout, max_tokens)
  return len(to_retry)


def merge_round(touched, host, model, system_prompt, request_timeout, max_tokens):
  merged_terms = []
  for go_id in touched:
    run([sys.executable, str(SCRIPTS / "parse_llm_findings.py"), go_id])
    n_retried = retry_issues(host, model, system_prompt, go_id, request_timeout, max_tokens)
    if n_retried:
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
    merged_terms = merge_round(touched, args.host, model, system_prompt, args.request_timeout,
                                args.max_tokens)
    run([sys.executable, str(SCRIPTS / "build_positional_mf_links.py")])
    commit_round(round_n, merged_terms, total_agents)
    print(f"Round {round_n} done. Merged: {', '.join(merged_terms) if merged_terms else 'none'}")


if __name__ == "__main__":
  main()
