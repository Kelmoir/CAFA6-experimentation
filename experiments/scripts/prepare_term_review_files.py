"""Prepare a single per-term digest file for an LLM-driven pass over the positional-MF gap.

The procedural sweep (term_gap_autolink.py) has visited every never-localized MF term with
signal-recurrence heuristics alone; 1769 of those came back "no_signal" -- no domain/ligand/text
value recurred cleanly enough across the sample to trust. This script is the first stage of a
higher-effort follow-up: instead of counting string recurrence, an LLM subagent reads a protein's
full annotated-feature record and the target GO term's own definition and judges which feature(s)
plausibly explain the term, the way the manual review rounds did before automation.

One combined file per term (not one file per protein): the GO term header (id, name, definition)
is written once, followed by a section per sampled protein. Cheaper to generate (the target term's
definition text can be long) and simpler to hand to subagents -- each subagent still reviews one
assigned protein, but all of them read the same file, so there's a single source of truth per term
rather than N near-duplicate files differing only in which protein section matters.

Output: experiments/llm_term_review/<go_id>/review.md plus a companion _meta.json manifest
recording the term, its definition, the full sampled accession list, and which accessions have
been assigned to a review run so far.
"""
import json
import random
import re
import sys
from pathlib import Path

RAW_FEATURES_PATH = Path("experiments/positional_mf_raw_features.json")
VISITED_PATH = Path("experiments/positional_mf_visited_terms.json")
GO_OBO_PATH = Path("data/cafa-6-protein-function-prediction/Train/go-basic.obo")
OUT_ROOT = Path("experiments/llm_term_review")
SEED = 42
MAX_PROTEINS = 10


def load_go_term_info():
  """id -> {"name":..., "def":...}, parsed straight from go-basic.obo."""
  info = {}
  cur_id = None
  cur = {}
  with open(GO_OBO_PATH) as f:
    for line in f:
      line = line.rstrip("\n")
      if line == "[Term]":
        if cur_id:
          info[cur_id] = cur
        cur_id = None
        cur = {}
      elif line.startswith("id: GO:"):
        cur_id = line[4:]
      elif line.startswith("name: ") and cur_id:
        cur["name"] = line[6:]
      elif line.startswith("def: ") and cur_id:
        m = re.match(r'def: "(.*)" \[', line)
        cur["def"] = m.group(1) if m else line[5:]
    if cur_id:
      info[cur_id] = cur
  return info


def pick_next_term(visited):
  """First (sorted) no_signal term without an existing review directory."""
  no_signal = sorted(t for t, d in visited.items() if d.get("status") == "no_signal")
  for t in no_signal:
    if not (OUT_ROOT / t.replace(":", "_")).exists():
      return t
  return None


def render_protein_section(accession, rec):
  lines = [
      f"## Protein {accession}",
      f"Sequence length: {rec['sequence_length']} aa",
      "",
      "Annotated features (position, type, qualifiers):",
  ]
  if not rec["features"]:
    lines.append("(none -- this protein has no MF-relevant positional features on record)")
  for f in rec["features"]:
    lines.append(f"- {f['type']} {f['start']}-{f['end']}")
    for k, v in f["qualifiers"].items():
      lines.append(f"    {k}: {v}")
  lines.append("")
  lines.append("Domain/family cross-references (no position, but may help identify a domain):")
  if not rec.get("domain_xrefs"):
    lines.append("(none)")
  for d in rec.get("domain_xrefs", []):
    lines.append(f"- {d['db']}:{d['accession']} -- {d['description']}")
  return "\n".join(lines) + "\n"


def render_review_file(go_id, go_info, samples, raw_features):
  lines = [
      f"# Positional-MF review: {go_id}",
      "",
      f"Name: {go_info.get('name', '?')}",
      f"Definition: {go_info.get('def', '(no definition available)')}",
      "",
      "Every protein below carries the term above as a real, asserted annotation (from UniProt's "
      "GO cross-references) but no positional feature has yet been confirmed to explain *why*. "
      "For your assigned protein (given separately in your task instructions), judge which "
      "annotated feature, if any, plausibly accounts for this GO term.",
      "",
      "---",
      "",
  ]
  for acc in samples:
    lines.append(render_protein_section(acc, raw_features[acc]))
    lines.append("---")
    lines.append("")
  return "\n".join(lines)


def main():
  go_id = sys.argv[1] if len(sys.argv) > 1 else None
  raw_features = json.load(open(RAW_FEATURES_PATH))
  visited = json.load(open(VISITED_PATH))
  go_info = load_go_term_info()

  if go_id is None:
    go_id = pick_next_term(visited)
    if go_id is None:
      print("No unreviewed no_signal terms remain.")
      return
    print(f"Auto-selected term: {go_id}")

  term_dir = OUT_ROOT / go_id.replace(":", "_")
  term_dir.mkdir(parents=True, exist_ok=True)

  accs = sorted(acc for acc, rec in raw_features.items()
                if any(g["go_id"] == go_id and g["aspect"] == "F" for g in rec["go_terms"]))
  if not accs:
    print(f"No proteins in raw_features carry {go_id} as a real MF annotation.")
    return

  r = random.Random(SEED + (hash(go_id) % 1000000))
  sample = sorted(r.sample(accs, min(MAX_PROTEINS, len(accs))))

  review_path = term_dir / "review.md"
  review_path.write_text(render_review_file(go_id, go_info.get(go_id, {}), sample, raw_features))

  meta = {
      "go_id": go_id,
      "name": go_info.get(go_id, {}).get("name", "?"),
      "def": go_info.get(go_id, {}).get("def", ""),
      "n_proteins_total": len(accs),
      "sampled_accessions": sample,
      "assigned_accessions": [],
  }
  (term_dir / "_meta.json").write_text(json.dumps(meta, indent=2))

  print(f"{go_id} ({go_info.get(go_id, {}).get('name', '?')})")
  print(f"  {len(accs)} proteins carry this term; sampled {len(sample)}")
  print(f"  wrote combined review file ({len(sample)} protein sections) -> {review_path}")


if __name__ == "__main__":
  main()
