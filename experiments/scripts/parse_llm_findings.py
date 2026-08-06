"""Procedurally parse an LLM-review findings.md into structured, re-derived binding signals.

Subagents judge *which* annotated feature (by type + position) explains a GO term for a given
protein -- that's the part heuristics can't do. This script does the part that shouldn't be left
to an LLM: given the subagent's (accession, feature_type, start, end) pointer, it looks the feature
back up in positional_mf_raw_features.json and pulls the exact, verifiable qualifier value (domain
accession, ligand name, or text note) straight from the source record. The LLM never gets to invent
a signal value by typing it out -- it can only point at a real feature, and the value used downstream
is whatever is actually on record for that feature. A pointer that doesn't resolve to a real feature
on the named protein (wrong coordinates, hallucinated feature) is dropped, not guessed at.

Expects findings.md entries in the fixed block format specified in the subagent prompt:

    ## <accession> <go_id>
    verdict: LIKELY | UNCERTAIN | NONE
    feature: <TYPE> <start>-<end>
    reasoning: <text>
    ---

Writes <term_dir>/parsed_findings.json: a list of resolved signals for verdict=LIKELY blocks only,
each carrying the underlying qualifier value plus enough provenance to hand-check it.
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
  """Pull the one qualifier value that would generalize to other proteins, same priority as
  term_gap_autolink.py's protein_signal(): ligand > note. DOMAIN/MOTIF/REGION features carry no
  useful qualifier here (their generalizable signal is the domain_xrefs list, not a per-feature
  qualifier) so they resolve via ('feature_type', None) and get bridged separately if needed."""
  q = feature["qualifiers"]
  if "ligand" in q:
    return "ligand", q["ligand"]
  if "note" in q:
    return "text", q["note"]
  return "feature_type_only", None


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

  resolved = []
  skipped = []
  for acc, block_go_id, verdict, ftype, start, end, reasoning in blocks:
    if block_go_id != go_id:
      skipped.append((acc, "go_id mismatch"))
      continue
    if verdict.upper() != "LIKELY":
      skipped.append((acc, f"verdict={verdict}"))
      continue
    rec = raw_features.get(acc)
    if not rec:
      skipped.append((acc, "unknown accession"))
      continue
    start, end = int(start), int(end)
    match = next((f for f in rec["features"]
                  if f["type"] == ftype and f["start"] == start and f["end"] == end), None)
    if not match:
      skipped.append((acc, f"no matching {ftype} {start}-{end} on record -- pointer didn't resolve"))
      continue
    kind, value = extract_signal_value(match)
    resolved.append({
        "accession": acc,
        "go_id": go_id,
        "feature_type": ftype,
        "start": start,
        "end": end,
        "signal_kind": kind,
        "signal_value": value,
        "reasoning": reasoning.strip(),
    })

  out_path = term_dir / "parsed_findings.json"
  out_path.write_text(json.dumps(resolved, indent=2))
  print(f"resolved: {len(resolved)}, skipped: {len(skipped)}")
  for acc, reason in skipped:
    print(f"  skipped {acc}: {reason}")
  print(f"-> {out_path}")


if __name__ == "__main__":
  main()
