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

Two safety filters from term_gap_autolink.py apply here too, and are just as necessary for
LLM-sourced signals as for string-recurrence-sourced ones -- subagent agreement establishes that a
signal is *real*, not that it's *specific enough to generalize*. Caught concretely, not just in
theory: round 3 had 2/2 agents agree a DOMAIN noted "Nudix hydrolase" explains GO:0000298
(endopolyphosphatase activity) -- true as far as it goes, but NUDIX hydrolase is a broad catalytic
superfamily (8-oxo-dGTPases, mRNA decapping enzymes, ADP-ribose pyrophosphatases, and more, not
just polyphosphatases), and its bridged accessions turned out to sit on 68-129 proteins dataset-wide
-- far past term_gap_autolink's own NARROW_DOMAIN_THRESHOLD=15 cutoff for "specific enough to trust
as this term's own marker" rather than "generic fold this term's family happens to share":
  - domain_accessions: only accessions with dataset-wide frequency <= NARROW_DOMAIN_THRESHOLD are
    kept from a bridged set; if every bridged accession is too generic, the group logs as
    unmergeable rather than silently degrading to whatever's narrowest.
  - ligands: GENERIC_LIGANDS (heme, ATP, common ions, ...) are excluded regardless of agreement --
    also caught concretely, not just in theory: round 3 also had 2/2 agents agree on "heme" for
    GO:0000249 (a real cofactor for that P450 enzyme), but heme already triggers GO:0020037 (heme
    binding) elsewhere in the dictionary -- ligand_to_go in build_positional_mf_links.py is a plain
    dict keyed by ligand name, so a second GO id claiming the same ligand string doesn't add a
    second trigger, it silently overwrites the first in whichever order the dictionary happens to
    sort. A cross-dictionary collision check backs this up independent of the generic-ligand list,
    in case a future non-generic ligand still collides with a different GO term's existing claim.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAW_FEATURES_PATH = Path("experiments/positional_mf_raw_features.json")
DICT_PATH = Path("experiments/positional_mf_localization_dictionary.json")
LOG_PATH = Path("experiments/positional_mf_llm_review_log.json")
AGREEMENT_THRESHOLD = 2
NARROW_DOMAIN_THRESHOLD = 15
# Text-pattern analogue of NARROW_DOMAIN_THRESHOLD: how many proteins dataset-wide have some
# TEXT_TIER_FEATURE_TYPES note containing the candidate pattern as a substring (the same test
# link_protein itself applies). Set higher than the domain threshold because substring matching
# against free-text notes naturally has a wider match surface than an exact accession id, but an
# Opus audit of the first 48 LLM merges found this cap was needed regardless: catalytic-residue
# labels like "proton donor" (2724 proteins) and "proton acceptor" (5219 proteins) passed the old
# length-only is_specific_text check and ended up reused as the text_pattern for over 100 unrelated
# dictionary entries each -- agreement between 2 reviewed proteins never meant the phrase was
# specific to their shared GO term, just that the phrase was real.
NARROW_TEXT_THRESHOLD = 30
TEXT_TIER_FEATURE_TYPES = {"ACT_SITE", "SITE", "MOTIF", "REGION", "ZN_FING"}
GENERIC_LIGANDS = {
    "atp", "gtp", "adp", "gdp", "amp", "substrate", "product", "water", "nadh", "nad+", "nadph",
    "nadp+", "fad", "fmn", "coenzyme a", "pyridoxal 5'-phosphate", "heme", "heme b", "dna", "rna",
}
GENERIC_LIGAND_PREFIXES = ("a divalent", "an ", "a ")
# "proton donor"/"proton acceptor" are the two catalytic-residue labels the Opus audit caught
# concretely (see NARROW_TEXT_THRESHOLD above); listed explicitly, not just left to the frequency
# check, since they're common enough phrases worth rejecting on sight without a raw_features scan.
GENERIC_TEXT_NOTES = {
    "disordered", "cleavage", "n-acetylmethionine", "phosphoserine", "phosphothreonine",
    "proton donor", "proton acceptor",
}


def normalize(text):
  if not text:
    return set()
  return set(re.split(r"[\s\-_,]+", text.lower())) - {""}


def text_pattern_frequency(raw_features, pattern):
  """How many distinct proteins dataset-wide have some TEXT_TIER_FEATURE_TYPES note containing
  pattern as a (lowercase) substring -- the exact test link_protein applies. Counts proteins, not
  occurrences, since link_protein only needs one match per feature to add a candidate."""
  count = 0
  for rec in raw_features.values():
    for f in rec.get("features", []):
      if f["type"] in TEXT_TIER_FEATURE_TYPES:
        note = (f.get("qualifiers", {}).get("note") or "").lower()
        if pattern in note:
          count += 1
          break
  return count


def is_specific_text(raw_features, norm_value):
  """Mirrors term_gap_autolink.py's is_specific_text: a text_pattern needs to be a real phrase
  (>=10 chars, contains a space or hyphen), not a single bare word -- otherwise it risks becoming
  a global substring trigger (link_protein matches text_patterns via `pattern in note`, checked
  against every ACT_SITE/SITE/MOTIF/REGION/ZN_FING note in the dataset). Missing from this script
  until a real near-miss surfaced it: two unrelated pseudouridine synthases both had an ACT_SITE
  note that was just the bare word "Nucleophile" -- a catalytic-residue label generic enough to
  appear on countless unrelated hydrolase/transferase active sites, not specific to either protein's
  actual GO term. 2/2 agreement alone doesn't make a single generic word safe to generalize.

  The length/space-or-hyphen check alone still let real generic phrases through -- "proton donor"
  and "proton acceptor" both pass it easily. NARROW_TEXT_THRESHOLD (dataset-wide substring-match
  frequency, the same idea as NARROW_DOMAIN_THRESHOLD for accessions) catches those, and any future
  phrase shaped the same way, without needing another name added to GENERIC_TEXT_NOTES by hand."""
  if norm_value in GENERIC_TEXT_NOTES:
    return False
  if not (len(norm_value) >= 10 and (" " in norm_value or "-" in norm_value)):
    return False
  return text_pattern_frequency(raw_features, norm_value) <= NARROW_TEXT_THRESHOLD


def is_generic_ligand(lig):
  low = lig.lower().strip()
  if low in GENERIC_LIGANDS:
    return True
  if low.endswith(")") and "(" in low and len(low) <= 10:
    return True  # bare ion, e.g. "Zn(2+)"
  if low.startswith(GENERIC_LIGAND_PREFIXES):
    return True
  return False


def get_evidence(raw_features, acc, ftype, start, end):
  rec = raw_features.get(acc, {})
  for f in rec.get("features", []):
    if f["type"] == ftype and f["start"] == start and f["end"] == end:
      return f.get("qualifiers", {}).get("evidence", "")
  return ""


def independent_accessions(raw_features, entries):
  """Accessions in an agreeing group whose evidence isn't itself just an inference from another
  member of the SAME group -- e.g. two ortholog records where one's feature is annotated
  'ECO:0000250|UniProtKB:<the other accession>' (curated by similarity, not independently observed
  in that protein). Caught concretely by an Opus audit of round-1 Haiku merges: GO:0000268,
  GO:0001067, GO:0001096, GO:0001155, and GO:0001648 all rested on exactly 2 contributing
  accessions where one's annotation cited the other's UniProtKB accession as its evidence source --
  the AGREEMENT_THRESHOLD=2 gate was satisfied by one real witness counted twice, not two
  independent ones. A 3+-accession group survives fine as long as enough OTHER members are
  independently evidenced; this only screens out groups that don't actually have 2 real sources.

  A second, commoner variant of the same loophole slipped through the original version of this
  check: N accessions each citing the SAME THIRD-PARTY accession outside the group (e.g. 3 rodent
  ion-channel orthologs whose ACT_SITE/BINDING features are all independently annotated
  'ECO:0000250|UniProtKB:P63142', a single curated source none of them shares with each other by
  name) -- the old check only looked for citations of other members of the SAME group, so this
  passed as "3 independent witnesses" when it's really one curated record propagated to 3 orthologs
  and counted 3 times. Caught concretely by an Opus audit of the first 48 merges: GO:0005250,
  GO:0008948, GO:0018505, GO:0031858, GO:0047184, and GO:0160125 all rested on exactly this pattern.
  Every entry is now attributed to a "source" -- the external UniProtKB accession its evidence cites
  by similarity, or itself if its evidence is a direct/primary observation -- and entries sharing a
  source collapse to one representative, since they're not independent of each other either."""
  accs = {e["accession"] for e in entries}
  sources = {}
  for e in entries:
    ev = get_evidence(raw_features, e["accession"], e["feature_type"], e["start"], e["end"])
    cited = re.findall(r"UniProtKB:(\w+)", ev)
    if any(c in accs and c != e["accession"] for c in cited):
      continue  # derivative from another member of THIS group -- excluded outright, as before
    sources[e["accession"]] = cited[0] if cited else e["accession"]
  kept, seen_sources = set(), set()
  for acc, src in sources.items():
    if src not in seen_sources:
      seen_sources.add(src)
      kept.add(acc)
  return kept


def build_domain_freq(raw_features):
  freq = Counter()
  for rec in raw_features.values():
    seen = {f"{d['db']}:{d['accession']}" for d in rec.get("domain_xrefs", [])}
    freq.update(seen)
  return freq


def resolve_domain_accessions(raw_features, accessions, note):
  """For a DOMAIN feature's note, find which of each corroborating protein's domain_xrefs it
  plausibly bridges to (same test build_positional_mf_links.py's link_protein uses), across all
  proteins in the agreeing group -- union of whatever resolves, since different proteins in the
  same family may carry different subsets of PROSITE/Pfam/InterPro cross-references (e.g. a
  heterodimeric transporter's two witnesses legitimately each carry only their own subunit's
  accession, and both belong in the union -- GO:0015439's CydC/CydD merge is a real, correct
  instance of this). So this deliberately does NOT require an accession to be carried by >=2
  witnesses to be included -- a per-accession witness count is returned alongside instead, purely
  for visibility (recorded in the merge log's note field), since an Opus audit of the first 48
  merges found the union sometimes let ONE witness's own idiosyncratic cross-reference dominate a
  merge that read as "N witnesses agree" (13 of 28 domain merges had a resolved accession carried
  by only 1 of N witnesses) -- not always wrong (as the CydC/CydD case shows) but always worth a
  human being able to see at a glance."""
  note_tokens = normalize(note)
  witnesses = defaultdict(set)
  for acc in accessions:
    rec = raw_features.get(acc, {})
    for d in rec.get("domain_xrefs", []):
      if note_tokens & normalize(d["description"]):
        witnesses[f"{d['db']}:{d['accession']}"].add(acc)
  resolved = sorted(witnesses)
  witness_counts = {a: sorted(witnesses[a]) for a in resolved}
  return resolved, witness_counts


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
  domain_freq = build_domain_freq(raw_features)
  existing_ligands = {lig: g for g, e in dictionary.items() for lig in e.get("ligands", [])}

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
    indep = independent_accessions(raw_features, entries)
    if len(indep) < AGREEMENT_THRESHOLD:
      unmerged_agreements.append({
          "kind": kind, "feature_type": ftype, "value": entries[0]["signal_value"],
          "accessions": accessions,
          "reason": f"circular agreement -- only {len(indep)} of {len(accessions)} contributing "
                    "accession(s) have evidence independent of another accession in this same "
                    "group (e.g. an ortholog pair where one's annotation cites "
                    "'ECO:0000250|UniProtKB:<the other>' -- inferred by similarity from it, not a "
                    "second independent witness)"})
      continue

    original_value = entries[0]["signal_value"]
    if kind == "ligand":
      if is_generic_ligand(original_value):
        unmerged_agreements.append({
            "kind": kind, "feature_type": ftype, "value": original_value, "accessions": accessions,
            "reason": "generic ligand -- shared cofactor across too many unrelated functions to "
                      "trust as this term's own marker"})
        continue
      claimed_by = existing_ligands.get(original_value)
      if claimed_by and claimed_by != go_id:
        unmerged_agreements.append({
            "kind": kind, "feature_type": ftype, "value": original_value, "accessions": accessions,
            "reason": f"ligand already claimed by {claimed_by} elsewhere in the dictionary -- "
                      "ligand_to_go is a plain dict, a second claim would silently overwrite it"})
        continue
      if original_value not in entry["ligands"]:
        entry["ligands"].append(original_value)
        existing_ligands[original_value] = go_id
      merged.append({"field": "ligands", "value": original_value, "accessions": accessions})
    elif kind == "text" and ftype != "DOMAIN":
      if not is_specific_text(raw_features, norm_value):
        unmerged_agreements.append({
            "kind": kind, "feature_type": ftype, "value": norm_value, "accessions": accessions,
            "reason": "agreed, but the note is too short/generic a phrase (or a known-generic "
                      "catalytic-residue label, or matches too many proteins dataset-wide -- see "
                      "NARROW_TEXT_THRESHOLD) to trust as a global substring trigger for this term"})
        continue
      if norm_value not in entry["text_patterns"]:
        entry["text_patterns"].append(norm_value)
      merged.append({"field": "text_patterns", "value": norm_value, "accessions": accessions})
    elif kind == "text" and ftype == "DOMAIN":
      bridged, witness_counts = resolve_domain_accessions(raw_features, accessions, original_value)
      narrow = [a for a in bridged if domain_freq.get(a, 999999) <= NARROW_DOMAIN_THRESHOLD]
      if not bridged:
        unmerged_agreements.append({
            "kind": kind, "feature_type": ftype, "value": original_value, "accessions": accessions,
            "reason": "agreed but the note text didn't token-overlap any corroborating protein's "
                      "domain_xrefs descriptions -- couldn't resolve to a specific accession"})
        continue
      if not narrow:
        unmerged_agreements.append({
            "kind": kind, "feature_type": ftype, "value": original_value, "accessions": accessions,
            "reason": f"resolved to {bridged}, but every accession has dataset-wide frequency > "
                      f"{NARROW_DOMAIN_THRESHOLD} -- too generic a fold to trust as this term's "
                      "own marker (frequencies: "
                      f"{ {a: domain_freq.get(a, 0) for a in bridged} })"})
        continue
      added = [a for a in narrow if a not in entry["domain_accessions"]]
      entry["domain_accessions"].extend(added)
      merged.append({"field": "domain_accessions", "value": narrow, "accessions": accessions,
                      "witness_counts": {a: witness_counts[a] for a in narrow},
                      "note": f"resolved from DOMAIN note {original_value!r}"
                              + (f" ({len(bridged) - len(narrow)} bridged accession(s) dropped "
                                 "for being too generic)" if len(narrow) < len(bridged) else "")})

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
