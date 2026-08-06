"""Merge an LLM-reviewed term's parsed_findings.json into the localization dictionary, gated on
independent agreement -- mirrors the procedural sweep's own cnt>=2 corroboration rule (see
term_gap_autolink.py), just fed by subagent judgment instead of blind string-recurrence counting.

A signal (kind, normalized value) merges only when it was independently reported by >=2 different
reviewed proteins for the term. A single witness isn't corroboration -- it's logged as an
unconfirmed singleton so a later review round can pick it up if a second protein agrees. Every
non-merge outcome (disagreement, low confidence, a pointer that never resolved, an outright NONE
verdict) is logged too, in experiments/positional_mf_llm_review_log.json -- nothing from a review
round is silently dropped.

Where a signal lands in the dictionary depends on the feature type it came from, because
build_positional_mf_links.py's matching tiers are feature-type-specific (see that file's module
docstring):
  - signal_kind == "ligand"                       -> dictionary[go_id]["ligands"]
  - signal_kind == "text", feature_type != DOMAIN  -> dictionary[go_id]["text_patterns"]
    (ACT_SITE/SITE/MOTIF/REGION notes are matched against text_patterns by link_protein directly)
  - signal_kind == "text", feature_type == DOMAIN  -> dictionary[go_id]["domain_accessions"]
    DOMAIN feature notes are *never* checked against text_patterns (link_protein's text tier
    explicitly excludes DOMAIN, which has its own accession-bridging tier instead) -- writing a
    DOMAIN note there would be a silent no-op. Instead this re-derives which PROSITE/Pfam/InterPro
    accession the note actually corresponds to, the same token-overlap test link_protein itself
    uses (normalize(note) & normalize(description)), against each corroborating protein's own
    domain_xrefs, and adds the resolved accession(s) as domain_accessions overrides. This is the
    existing "accessions the bulk *2go tables don't cover" escape hatch already documented in
    build_positional_mf_links.py -- confirmed necessary here, not just theoretically: GO:0000156's
    receiver-domain accessions (Pfam:PF00072, PROSITE:PS50110) map in the bulk tables to
    GO:0000160 (a biological_process term), never to the correct GO:0000156 molecular_function
    term, so the bulk-table bridge alone would never have found this regardless of dictionary size.
  - signal_kind == "feature_type_only" (no ligand/note qualifier at all) -> logged as unmergeable;
    there's no generalizable value to add, whatever the feature type.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

RAW_FEATURES_PATH = Path("experiments/positional_mf_raw_features.json")
DICT_PATH = Path("experiments/positional_mf_localization_dictionary.json")
LOG_PATH = Path("experiments/positional_mf_llm_review_log.json")
AGREEMENT_THRESHOLD = 2


def normalize(text):
  if not text:
    return set()
  return set(re.split(r"[\s\-_,]+", text.lower())) - {""}


def resolve_domain_accessions(raw_features, accessions, note):
  """For a DOMAIN feature's note, find which of each corroborating protein's domain_xrefs it
  plausibly bridges to (same test build_positional_mf_links.py's link_protein uses), across all
  proteins in the agreeing group -- union of whatever resolves, since different proteins in the
  same family may carry different subsets of PROSITE/Pfam/InterPro cross-references."""
  note_tokens = normalize(note)
  resolved = set()
  for acc in accessions:
    rec = raw_features.get(acc, {})
    for d in rec.get("domain_xrefs", []):
      if note_tokens & normalize(d["description"]):
        resolved.add(f"{d['db']}:{d['accession']}")
  return sorted(resolved)


def main():
  if len(sys.argv) < 2:
    print("usage: merge_llm_findings.py <GO_ID>")
    return
  go_id = sys.argv[1]
  term_dir = Path("experiments/llm_term_review") / go_id.replace(":", "_")
  findings_path = term_dir / "parsed_findings.json"
  if not findings_path.exists():
    print(f"no parsed_findings.json in {term_dir} -- run parse_llm_findings.py first")
    return

  findings = json.loads(findings_path.read_text())
  raw_features = json.load(open(RAW_FEATURES_PATH))
  dictionary = json.load(open(DICT_PATH))
  log = json.load(open(LOG_PATH)) if LOG_PATH.exists() else {}
  meta_path = term_dir / "_meta.json"
  go_name = json.loads(meta_path.read_text())["name"] if meta_path.exists() else "?"

  resolved = findings.get("resolved", [])
  groups = defaultdict(list)  # (kind, feature_type, normalized_value) -> [entry, ...]
  for e in resolved:
    key = (e["signal_kind"], e["feature_type"], (e["signal_value"] or "").lower().strip())
    groups[key].append(e)

  merged, unmerged_agreements, unconfirmed_singletons = [], [], []
  entry = dictionary.get(go_id, {"name": go_name, "ligands": [],
                                  "feature_types": [], "domain_accessions": [], "text_patterns": []})
  for (kind, ftype, norm_value), entries in groups.items():
    accessions = sorted({e["accession"] for e in entries})
    if kind == "feature_type_only":
      unmerged_agreements.append({
          "kind": kind, "feature_type": ftype, "accessions": accessions,
          "reason": "no ligand/note qualifier to generalize"})
      continue
    if len(accessions) < AGREEMENT_THRESHOLD:
      unconfirmed_singletons.append({
          "kind": kind, "feature_type": ftype, "value": entries[0]["signal_value"],
          "accessions": accessions})
      continue

    original_value = entries[0]["signal_value"]
    if kind == "ligand":
      if original_value not in entry["ligands"]:
        entry["ligands"].append(original_value)
      merged.append({"field": "ligands", "value": original_value, "accessions": accessions})
    elif kind == "text" and ftype != "DOMAIN":
      if norm_value not in entry["text_patterns"]:
        entry["text_patterns"].append(norm_value)
      merged.append({"field": "text_patterns", "value": norm_value, "accessions": accessions})
    elif kind == "text" and ftype == "DOMAIN":
      bridged = resolve_domain_accessions(raw_features, accessions, original_value)
      if not bridged:
        unmerged_agreements.append({
            "kind": kind, "feature_type": ftype, "value": original_value, "accessions": accessions,
            "reason": "agreed but the note text didn't token-overlap any corroborating protein's "
                      "domain_xrefs descriptions -- couldn't resolve to a specific accession"})
        continue
      added = [a for a in bridged if a not in entry["domain_accessions"]]
      entry["domain_accessions"].extend(added)
      merged.append({"field": "domain_accessions", "value": bridged, "accessions": accessions,
                      "note": f"resolved from DOMAIN note {original_value!r}"})

  if merged:
    dictionary[go_id] = entry
    DICT_PATH.write_text(json.dumps(dict(sorted(dictionary.items())), indent=1))

  log[go_id] = {
      "n_resolved": len(resolved),
      "n_uncertain": len(findings.get("uncertain", [])),
      "n_issues": len(findings.get("issues", [])),
      "merged": merged,
      "unmerged_agreements": unmerged_agreements,
      "unconfirmed_singletons": unconfirmed_singletons,
      "uncertain": findings.get("uncertain", []),
      "issues": findings.get("issues", []),
  }
  LOG_PATH.write_text(json.dumps(dict(sorted(log.items())), indent=1))

  print(f"{go_id}: merged {len(merged)} signal group(s), "
        f"{len(unmerged_agreements)} agreed-but-unmergeable, "
        f"{len(unconfirmed_singletons)} unconfirmed singleton(s)")
  for m in merged:
    print(f"  + {m['field']}: {m['value']} (from {m['accessions']})")
  for u in unmerged_agreements:
    print(f"  ! unmergeable: {u}")
  print(f"-> {DICT_PATH} (if merged), {LOG_PATH}")


if __name__ == "__main__":
  main()
