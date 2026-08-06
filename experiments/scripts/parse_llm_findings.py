"""Procedurally parse an LLM-review findings.md into structured, re-derived binding signals.

Subagents judge *which* annotated feature (by type + position) explains a GO term for a given
protein -- that's the part heuristics can't do. This script does the part that shouldn't be left
to an LLM: given a subagent's (accession, feature_type, start, end) pointer, it looks the feature
back up in positional_mf_raw_features.json and pulls the exact, verifiable qualifier value (ligand
name or free-text note) straight from the source record. The LLM never gets to invent a signal
value by typing it out -- it can only point at a real feature, and the value used downstream is
whatever is actually on record for that feature.

Every block in findings.md is captured, not just clean LIKELY ones -- a subagent that says NONE,
hedges with UNCERTAIN, points at coordinates that don't exist on the named protein, or reports the
wrong GO id is as informative to the merge step as a clean hit, and none of that should silently
vanish.

Expects findings.md entries in the fixed block format specified in the subagent prompt:

    ## <accession> <go_id>
    verdict: LIKELY | UNCERTAIN | NONE
    feature: <TYPE> <start>-<end>
    reasoning: <text>
    ---

Writes <term_dir>/parsed_findings.json:
  resolved  -- verdict=LIKELY, pointer resolved to a real feature. Candidates for merging once
              >=2 independently-reviewed proteins agree on the same underlying signal.
  uncertain -- verdict=UNCERTAIN, pointer resolved. Never counted toward merge agreement, but
              logged so a human/future round can see what almost matched.
  issues    -- anything that didn't produce a usable signal: NONE verdicts, go_id mismatches,
              malformed verdicts, or a pointer that doesn't resolve to a real feature on that
              protein (wrong coordinates, hallucinated feature).
"""
import json
import re
import sys
from pathlib import Path

RAW_FEATURES_PATH = Path("experiments/positional_mf_raw_features.json")
BLOCK_RE = re.compile(
    r"^##\s+(\S+)\s+(GO:\d+)\s*\n"
    r"verdict:\s*(\w+)\s*\n"
    r"feature:\s*(\S+)\s+(\d+)-(\d+)\s*\n"
    r"reasoning:\s*(.*?)\n"
    r"---",
    re.MULTILINE | re.DOTALL,
)


def extract_signal_value(feature):
  """Pull the one qualifier value that would generalize to other proteins: ligand > note.
  DOMAIN/MOTIF/REGION features with neither carry no per-feature generalizable value here --
  their signal (if any) lives in domain_xrefs, resolved separately by the merge step."""
  q = feature["qualifiers"]
  if "ligand" in q:
    return "ligand", q["ligand"]
  if "note" in q:
    return "text", q["note"]
  return "feature_type_only", None


def resolve_pointer(raw_features, acc, ftype, start, end):
  rec = raw_features.get(acc)
  if not rec:
    return None, "unknown accession"
  match = next((f for f in rec["features"]
                if f["type"] == ftype and f["start"] == start and f["end"] == end), None)
  if not match:
    return None, f"no matching {ftype} {start}-{end} on record for {acc} -- pointer didn't resolve"
  return match, None


def main():
  if len(sys.argv) < 2:
    print("usage: parse_llm_findings.py <GO_ID>")
    return
  go_id = sys.argv[1]
  term_dir = Path("experiments/llm_term_review") / go_id.replace(":", "_")
  findings_path = term_dir / "findings.md"
  if not findings_path.exists():
    print(f"no findings.md in {term_dir}")
    return

  raw_features = json.load(open(RAW_FEATURES_PATH))
  text = findings_path.read_text()
  blocks = BLOCK_RE.findall(text)
  print(f"parsed {len(blocks)} findings blocks")

  resolved, uncertain, issues = [], [], []

  for acc, block_go_id, verdict, ftype, start, end, reasoning in blocks:
    verdict_u = verdict.upper()
    reasoning = reasoning.strip()
    if block_go_id != go_id:
      issues.append({"accession": acc, "reason": "go_id_mismatch",
                      "detail": f"block reported {block_go_id}, expected {go_id}"})
      continue
    if verdict_u == "NONE":
      issues.append({"accession": acc, "reason": "none_verdict", "reasoning": reasoning})
      continue
    if verdict_u not in ("LIKELY", "UNCERTAIN"):
      issues.append({"accession": acc, "reason": "invalid_verdict", "detail": verdict})
      continue

    start_i, end_i = int(start), int(end)
    match, err = resolve_pointer(raw_features, acc, ftype, start_i, end_i)
    if err:
      issues.append({"accession": acc, "reason": "unresolved_pointer", "detail": err,
                      "reported_feature": f"{ftype} {start_i}-{end_i}", "reasoning": reasoning})
      continue

    kind, value = extract_signal_value(match)
    entry = {
        "accession": acc, "go_id": go_id, "feature_type": ftype,
        "start": start_i, "end": end_i,
        "signal_kind": kind, "signal_value": value, "reasoning": reasoning,
    }
    (resolved if verdict_u == "LIKELY" else uncertain).append(entry)

  out = {"resolved": resolved, "uncertain": uncertain, "issues": issues}
  out_path = term_dir / "parsed_findings.json"
  out_path.write_text(json.dumps(out, indent=2))
  print(f"resolved (LIKELY): {len(resolved)}, uncertain: {len(uncertain)}, issues: {len(issues)}")
  for i in issues:
    print(f"  issue: {i['accession']} -- {i['reason']}")
  print(f"-> {out_path}")


if __name__ == "__main__":
  main()
