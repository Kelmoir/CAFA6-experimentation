"""Visualizer for the positional_mf linking pipeline.

Enter a UniProt accession and see every positional annotation the databanks (UniProt
BINDING/DNA_BIND/DOMAIN features, plus their PROSITE/Pfam/InterPro cross-references) offered
as a candidate GO term, alongside which of those candidates build_positional_mf_links.py
actually confirmed against the protein's real GO annotations vs. left unconfirmed/unlocalized.

Reuses build_positional_mf_links.link_protein(..., verbose=True) rather than reimplementing
candidate generation, so this can never drift from what the real pipeline run in
experiments/positional_mf_links.json actually computed.

Usage:
  python experiments/scripts/visualize_protein.py                  # interactive prompt loop
  python experiments/scripts/visualize_protein.py Q13444 P04637    # print markdown for each
  python experiments/scripts/visualize_protein.py Q13444 --html experiments/Q13444.html
  python experiments/scripts/visualize_protein.py Q13444 P04637 --out-dir experiments/reports
"""
import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_positional_mf_links as bml

GO_OBO_PATH = Path("data/cafa-6-protein-function-prediction/Train/go-basic.obo")
CONSUMED_FEATURE_TYPES = {"BINDING", "DNA_BIND", "DOMAIN"}


def load_go_names(obo_path):
  """GO:id -> human-readable name, parsed from go-basic.obo's [Term] stanzas."""
  names = {}
  current_id = None
  for line in obo_path.read_text().splitlines():
    if line == "[Term]":
      current_id = None
    elif line.startswith("id: "):
      current_id = line[4:].strip()
    elif line.startswith("name: ") and current_id and current_id not in names:
      names[current_id] = line[6:].strip()
  return names


def build_report(accession, record, mappings, ancestors_with_depth, go_names):
  result = bml.link_protein(accession, record, mappings, ancestors_with_depth, verbose=True)
  candidates = result["all_candidates"]
  real_terms = {t["go_id"]: t for t in record["go_terms"] if t["aspect"] == "F"}
  unlocalized = set(result["unlocalized_real_mf_terms"])

  def label(go_id):
    name = go_names.get(go_id)
    return f"{go_id} ({name})" if name else go_id

  def match_text(match):
    if match is None:
      return "not confirmed"
    real_id, hops = match
    kind = "exact match" if hops == 0 else f"ancestor match, {hops} hop(s)"
    return f"confirmed -> {label(real_id)} [{kind}]"

  real_rows = [
      (go_id, t["name"], t["evidence"], "unlocalized" if go_id in unlocalized else "localized")
      for go_id, t in sorted(real_terms.items())
  ]

  binding_rows = []
  for f in record["features"]:
    if f["type"] != "BINDING":
      continue
    ligand = f["qualifiers"].get("ligand") or f["qualifiers"].get("ligand_id") or "?"
    go_id = bml.LIGAND_TO_GO.get(ligand)
    if go_id is None:
      binding_rows.append((f"{f['start']}-{f['end']}", ligand, "-", "no dictionary entry for this ligand"))
      continue
    cand = next((c for c in candidates if c["source"] == "binding_ligand"
                 and c["start"] == f["start"] and c["end"] == f["end"] and c["go_id"] == go_id), None)
    binding_rows.append((f"{f['start']}-{f['end']}", ligand, label(go_id),
                          match_text(cand["match"]) if cand else "not confirmed"))

  dna_rows = []
  dna_go_id = bml.FEATURE_TYPE_TO_GO.get("DNA_BIND")
  for f in record["features"]:
    if f["type"] != "DNA_BIND":
      continue
    cand = next((c for c in candidates if c["source"] == "feature_type"
                 and c["start"] == f["start"] and c["end"] == f["end"]), None)
    dna_rows.append((f"{f['start']}-{f['end']}", label(dna_go_id),
                      match_text(cand["match"]) if cand else "not confirmed"))

  domain_feature_rows = [
      (f"{f['start']}-{f['end']}", f["qualifiers"].get("note", ""))
      for f in record["features"] if f["type"] == "DOMAIN"
  ]

  domain_xrefs = record.get("domain_xrefs", [])
  acc_go = bml.collect_domain_accession_candidates(domain_xrefs, mappings)
  domain_xref_rows = []
  for xref in domain_xrefs:
    db, acc, desc = xref["db"], xref["accession"], xref["description"]
    go_ids = acc_go.get((db, acc, desc))
    if not go_ids:
      domain_xref_rows.append((db, acc, desc, "-", "no *2go mapping", ""))
      continue
    xref_key = f"{db}:{acc}"
    matching = [c for c in candidates if c["xref"] == xref_key]
    if any(c["source"] == "domain_bridge" for c in matching):
      bridge = "bridged to a DOMAIN note"
    elif any(c["source"] == "domain_fallback_union" for c in matching):
      bridge = "fallback union (no note bridged)"
    else:
      bridge = "no candidate generated"
    go_labels = ", ".join(label(g) for g in sorted(go_ids))
    statuses = "; ".join(f"{label(c['go_id'])}: {match_text(c['match'])}" for c in matching)
    domain_xref_rows.append((db, acc, desc, go_labels, bridge, statuses))

  other_rows = [
      (f["type"], f"{f['start']}-{f['end']}", f["qualifiers"].get("note", "") or json.dumps(f["qualifiers"]))
      for f in record["features"] if f["type"] not in CONSUMED_FEATURE_TYPES
  ]

  localized_rows = [
      (label(l["go_id"]), f"{l['start']}-{l['end']}", l["source"], l["table_agreement"] or "-",
       label(l["matched_real_term"]), "exact" if l["match_hops"] == 0 else f"ancestor +{l['match_hops']}")
      for l in result["localized"]
  ]
  unlocalized_rows = [label(go_id) for go_id in sorted(unlocalized)]

  n_real = len(real_terms)
  n_covered = n_real - len(unlocalized)

  return {
      "accession": accession,
      "sequence_length": record["sequence_length"],
      "n_real": n_real,
      "n_covered": n_covered,
      "real_rows": real_rows,
      "binding_rows": binding_rows,
      "dna_rows": dna_rows,
      "domain_feature_rows": domain_feature_rows,
      "domain_xref_rows": domain_xref_rows,
      "other_rows": other_rows,
      "localized_rows": localized_rows,
      "unlocalized_rows": unlocalized_rows,
  }


def to_markdown(d):
  lines = [f"# Protein `{d['accession']}`\n"]
  pct = 100 * d["n_covered"] / d["n_real"] if d["n_real"] else 0
  lines.append(f"- Sequence length: {d['sequence_length']} aa")
  lines.append(f"- Real molecular_function annotations (UniProt): {d['n_real']}")
  lines.append(f"- Confirmed localized to a span: {d['n_covered']}/{d['n_real']} ({pct:.0f}%)\n")

  def table(headers, rows, empty="_None._"):
    if not rows:
      lines.append(empty + "\n")
      return
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
      lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")

  lines.append("## Real molecular_function annotations (from UniProt)\n")
  table(["GO ID", "Name", "Evidence", "Status"], d["real_rows"])

  lines.append("## Positional annotations offered by the databanks\n")
  lines.append(f"### BINDING (n={len(d['binding_rows'])})\n")
  table(["Span", "Ligand", "Candidate GO", "Status"], d["binding_rows"])

  lines.append(f"### DNA_BIND (n={len(d['dna_rows'])})\n")
  table(["Span", "Candidate GO", "Status"], d["dna_rows"])

  lines.append(f"### DOMAIN (n={len(d['domain_feature_rows'])})\n")
  table(["Span", "Note"], d["domain_feature_rows"])
  lines.append(f"Cross-references offered for DOMAIN bridging (n={len(d['domain_xref_rows'])}):\n")
  table(["DB", "Accession", "Description", "Candidate GO ids", "Bridge status", "Per-candidate outcome"],
        d["domain_xref_rows"])

  lines.append(f"### Other feature types not yet linked by the pipeline (n={len(d['other_rows'])})\n")
  lines.append("_ACT_SITE / SITE / METAL / CA_BIND / NP_BIND / ZN_FING / MOTIF / REGION are extracted but not "
               "consumed by any tier yet (a known escalation gap, not a bug)._\n")
  table(["Type", "Span", "Note/qualifiers"], d["other_rows"])

  lines.append("## Confirmed localized links (what got selected)\n")
  table(["GO ID", "Span", "Source", "Table agreement", "Matched real term", "Match"], d["localized_rows"],
        empty="_None -- no candidate confirmed for this protein._")

  lines.append("## Unlocalized real MF terms (no confirmed span)\n")
  if d["unlocalized_rows"]:
    lines.extend(f"- {r}" for r in d["unlocalized_rows"])
  else:
    lines.append("_None -- every real MF annotation got localized._")
  lines.append("")

  return "\n".join(lines)


def to_html(d):
  def esc(s):
    return html.escape(str(s))

  def table(headers, rows, empty="None."):
    if not rows:
      return f"<p><em>{esc(empty)}</em></p>"
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

  pct = 100 * d["n_covered"] / d["n_real"] if d["n_real"] else 0
  parts = [
      f"<h1>Protein <code>{esc(d['accession'])}</code></h1>",
      "<ul class='summary'>",
      f"<li>Sequence length: {d['sequence_length']} aa</li>",
      f"<li>Real molecular_function annotations (UniProt): {d['n_real']}</li>",
      f"<li>Confirmed localized to a span: {d['n_covered']}/{d['n_real']} ({pct:.0f}%)</li>",
      "</ul>",
      "<h2>Real molecular_function annotations (from UniProt)</h2>",
      table(["GO ID", "Name", "Evidence", "Status"], d["real_rows"]),
      "<h2>Positional annotations offered by the databanks</h2>",
      f"<h3>BINDING (n={len(d['binding_rows'])})</h3>",
      table(["Span", "Ligand", "Candidate GO", "Status"], d["binding_rows"]),
      f"<h3>DNA_BIND (n={len(d['dna_rows'])})</h3>",
      table(["Span", "Candidate GO", "Status"], d["dna_rows"]),
      f"<h3>DOMAIN (n={len(d['domain_feature_rows'])})</h3>",
      table(["Span", "Note"], d["domain_feature_rows"]),
      f"<p>Cross-references offered for DOMAIN bridging (n={len(d['domain_xref_rows'])}):</p>",
      table(["DB", "Accession", "Description", "Candidate GO ids", "Bridge status", "Per-candidate outcome"],
            d["domain_xref_rows"]),
      f"<h3>Other feature types not yet linked by the pipeline (n={len(d['other_rows'])})</h3>",
      "<p><em>ACT_SITE / SITE / METAL / CA_BIND / NP_BIND / ZN_FING / MOTIF / REGION are extracted but not "
      "consumed by any tier yet (a known escalation gap, not a bug).</em></p>",
      table(["Type", "Span", "Note/qualifiers"], d["other_rows"]),
      "<h2>Confirmed localized links (what got selected)</h2>",
      table(["GO ID", "Span", "Source", "Table agreement", "Matched real term", "Match"], d["localized_rows"],
            empty="None -- no candidate confirmed for this protein."),
      "<h2>Unlocalized real MF terms (no confirmed span)</h2>",
  ]
  if d["unlocalized_rows"]:
    parts.append("<ul>" + "".join(f"<li>{esc(r)}</li>" for r in d["unlocalized_rows"]) + "</ul>")
  else:
    parts.append("<p><em>None -- every real MF annotation got localized.</em></p>")

  body = "\n".join(parts)
  return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{esc(d['accession'])} — positional MF annotations</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem;
       line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.5rem; font-size: 0.9rem; }}
th, td {{ border: 1px solid #8888; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }}
th {{ background: #8881; }}
code {{ background: #8881; padding: 0.1rem 0.3rem; border-radius: 3px; }}
h2 {{ border-bottom: 1px solid #8884; padding-bottom: 0.2rem; margin-top: 2rem; }}
.summary {{ background: #8881; border-radius: 6px; padding: 0.8rem 1.5rem; }}
</style></head>
<body>
{body}
</body></html>"""


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("accessions", nargs="*", help="UniProt accession(s), e.g. Q13444. If omitted, "
                       "starts an interactive prompt loop.")
  parser.add_argument("--html", metavar="FILE", help="Write one self-contained HTML report "
                       "(requires exactly one accession) instead of printing markdown.")
  parser.add_argument("--out-dir", metavar="DIR", help="Write one <ACCESSION>.md file per given "
                       "accession into this directory instead of printing to stdout.")
  args = parser.parse_args()

  if args.html and len(args.accessions) != 1:
    parser.error("--html requires exactly one accession")
  if args.out_dir and not args.accessions:
    parser.error("--out-dir requires at least one accession")

  print("Loading raw features, GO mappings, ancestor graph, and GO term names...", file=sys.stderr)
  raw_features = json.loads(bml.RAW_FEATURES_PATH.read_text())
  mappings = bml.load_mappings()
  ancestors_with_depth = bml.build_ancestor_lookup(bml.GO_GRAPH_PATH)
  go_names = load_go_names(GO_OBO_PATH)
  print(f"Ready: {len(raw_features)} proteins available.\n", file=sys.stderr)

  def render(accession):
    resolved = accession.strip()
    if resolved not in raw_features and resolved.upper() in raw_features:
      resolved = resolved.upper()
    if resolved not in raw_features:
      return None, f"not found in the {len(raw_features)}-protein working set (experiments/positional_mf_raw_features.json)"
    data = build_report(resolved, raw_features[resolved], mappings, ancestors_with_depth, go_names)
    return data, None

  if args.html:
    data, err = render(args.accessions[0])
    if err:
      sys.exit(f"{args.accessions[0]}: {err}")
    Path(args.html).write_text(to_html(data))
    print(f"Wrote {args.html}")
    return

  if args.out_dir:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for acc in args.accessions:
      data, err = render(acc)
      if err:
        print(f"[skip] {acc}: {err}", file=sys.stderr)
        continue
      (out_dir / f"{acc}.md").write_text(to_markdown(data))
      written += 1
    print(f"Wrote {written} report(s) -> {out_dir}/")
    return

  if args.accessions:
    for acc in args.accessions:
      data, err = render(acc)
      print(to_markdown(data) if data else f"# {acc}\n\n_{err}_\n")
      print("\n---\n")
    return

  print("Enter a UniProt accession (e.g. Q13444), blank line or Ctrl-D to quit.")
  while True:
    try:
      query = input("accession> ").strip()
    except EOFError:
      print()
      break
    if not query:
      break
    data, err = render(query)
    print(to_markdown(data) if data else f"_{err}_\n")


if __name__ == "__main__":
  main()
