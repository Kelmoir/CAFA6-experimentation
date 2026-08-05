"""Link UniProt positional features to a protein's real molecular_function GO annotations.

Design (see the positional-MF design discussion for the full reasoning): a tiered
candidate-generation pass over experiments/positional_mf_raw_features.json, followed by an
automatic verification pass that only keeps a candidate if it agrees - under GO's true-path
ancestor closure, not exact-ID equality - with a GO term the protein is *already* annotated
with. Nothing here invents a GO term a protein isn't already labeled with; it only tries to
find *where in the sequence* an already-real annotation comes from. This is also what makes it
safe to grow the trigger dictionary (below) somewhat liberally: a bad/overbroad trigger just
fails to confirm on proteins that don't already carry the real annotation, rather than
introducing a false one.

All trigger data - ligand names, feature types, domain accession overrides, and free-text
note patterns - lives in experiments/positional_mf_localization_dictionary.json, keyed by GO
id (see load_localization_dictionary), not hardcoded here. That file is meant to be grown over
time: run this script, inspect experiments/positional_mf_gaps.json (written by main() below -
per protein, whatever raw annotations produced no candidate at all, plus whatever real MF terms
still have no confirmed span), spot-check a sample of gap proteins by hand, add new trigger
entries for whatever you can confirm, and re-run. The goal is for that loop to do almost all of
the annotation work algorithmically, with manual review only occasionally topping up the
dictionary rather than annotating proteins one at a time.

Tiers:
  - BINDING features: ligand name -> GO term via dictionary["*"]["ligands"]. No bridging needed
    - the feature already carries its own span.
  - Structural feature types (currently just DNA_BIND -> GO:0003677 DNA binding): feature type
    directly implies a GO term via dictionary["*"]["feature_types"]. No lookup table.
  - Text-tier features (ACT_SITE/SITE/MOTIF/REGION): free-text `note` qualifier matched by
    substring against dictionary["*"]["text_patterns"] (e.g. a MOTIF noted "SH3-binding" directly
    implies GO:0017124 SH3 domain binding). Starts empty; populated via the gap-review loop.
  - DOMAIN features: bridged to a PROSITE/Pfam/InterPro accession (captured separately in
    domain_xrefs, since UniProt doesn't attach these directly to the DOMAIN feature) via
    note-text/description token overlap, then looked up in the GO Consortium's prosite2go/
    pfam2go/interpro2go tables, plus any dictionary["*"]["domain_accessions"] overrides for
    accessions the bulk tables don't cover. Verified directly against Q13444/ADAM17: the bridge
    is not perfect (it can miss the exact accession that has the real mapping, e.g. it missed
    Pfam:PF01421 for the "Peptidase M12B" domain note) - so any accession's GO candidates that
    never bridge to a specific domain fall back to being assigned to the union of all DOMAIN
    spans in that protein (domain_fallback_union), rather than silently discarded or guessed at.

A candidate is "confirmed" only if it survives the ancestor-closure check against the protein's
real (aspect=F) GO annotations. Real MF terms that never get a confirmed positional match are
kept in unlocalized_real_mf_terms - a weak whole-protein positive, not dropped. Raw annotations
(a ligand, domain accession, or note) that never even produced a candidate - no dictionary entry
covers them at all - are tracked separately as uncovered_raw_annotations, the raw material for
the gap-review loop.
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

RAW_FEATURES_PATH = Path("experiments/positional_mf_raw_features.json")
GO_GRAPH_PATH = Path("data/cafa-6-protein-function-prediction/dataset/go_dependency_graph.json")
GO_MAPPINGS_DIR = Path("data/go_mappings")
LOCALIZATION_DICT_PATH = Path("experiments/positional_mf_localization_dictionary.json")
OUTPUT_PATH = Path("experiments/positional_mf_links.json")
GAPS_PATH = Path("experiments/positional_mf_gaps.json")
SPOTCHECK_PATH = Path("experiments/positional_mf_spotcheck_sample.json")

TRUE_PATH_RELATIONSHIPS = {"is_a", "part_of"}
SEED = 42

# Feature types whose free-text `note` qualifier is matched against the dictionary's
# text_patterns (Tier 3). Excludes BINDING/DNA_BIND/DOMAIN, which have their own tiers above.
TEXT_TIER_FEATURE_TYPES = {"ACT_SITE", "SITE", "MOTIF", "REGION"}

MAPPING_FILES = {
    "PROSITE": GO_MAPPINGS_DIR / "prosite2go.txt",
    "Pfam": GO_MAPPINGS_DIR / "pfam2go.txt",
    "InterPro": GO_MAPPINGS_DIR / "interpro2go.txt",
}


def load_localization_dictionary(path):
  """Loads the GO-id-keyed trigger dictionary and flattens it into the four lookup structures
  link_protein actually needs: ligand name -> GO id, feature type -> GO id, "DB:ACCESSION" ->
  set(GO id) (domain overrides), and a list of (lowercased text pattern, GO id) pairs.
  """
  raw = json.loads(path.read_text())
  ligand_to_go = {}
  feature_type_to_go = {}
  domain_accession_to_go = defaultdict(set)
  text_patterns = []
  for go_id, entry in raw.items():
    for ligand in entry.get("ligands", []):
      ligand_to_go[ligand] = go_id
    for ftype in entry.get("feature_types", []):
      feature_type_to_go[ftype] = go_id
    for acc in entry.get("domain_accessions", []):
      domain_accession_to_go[acc].add(go_id)
    for pattern in entry.get("text_patterns", []):
      text_patterns.append((pattern.lower(), go_id))
  return {
      "ligand_to_go": ligand_to_go,
      "feature_type_to_go": feature_type_to_go,
      "domain_accession_to_go": dict(domain_accession_to_go),
      "text_patterns": text_patterns,
  }


def normalize(text):
  if not text:
    return set()
  return set(re.split(r"[\s\-_,]+", text.lower())) - {""}


def parse_2go_file(path, db_prefix):
  """accession -> set(GO ids), from a line format like:
  'PROSITE:PS00011 GLA_1 > GO:calcium ion binding ; GO:0005509'
  """
  mapping = defaultdict(set)
  pattern = re.compile(rf"^{re.escape(db_prefix)}:(\S+)\b.*;\s*(GO:\d+)\s*$")
  for line in path.read_text().splitlines():
    if not line or line.startswith("!"):
      continue
    m = pattern.match(line)
    if not m:
      continue
    acc, go_id = m.groups()
    mapping[acc].add(go_id)
  return mapping


def load_mappings():
  return {db: parse_2go_file(path, db) for db, path in MAPPING_FILES.items()}


def build_ancestor_lookup(graph_path):
  """Returns ancestors(term) -> set of proper ancestors, and depth(term, ancestor) -> hop count.

  Restricted to is_a/part_of edges only (GO's true-path-rule-safe relations), matching
  generate_dataset.py's own propagation scope.
  """
  data = json.loads(graph_path.read_text())
  parents = defaultdict(set)
  for link in data["graph"]["links"]:
    if link["relationship"] in TRUE_PATH_RELATIONSHIPS:
      parents[link["source"]].add(link["target"])

  ancestor_cache = {}
  depth_cache = {}

  def ancestors_with_depth(term):
    if term in ancestor_cache:
      return ancestor_cache[term], depth_cache[term]
    depths = {}
    frontier = [term]
    hop = 0
    while frontier:
      hop += 1
      nxt = []
      for cur in frontier:
        for p in parents.get(cur, ()):
          if p not in depths:
            depths[p] = hop
            nxt.append(p)
      frontier = nxt
    ancestor_cache[term] = set(depths)
    depth_cache[term] = depths
    return ancestor_cache[term], depth_cache[term]

  return ancestors_with_depth


def match_against_real(go_id, real_terms, ancestors_with_depth):
  """Returns (matched_real_term, hop_count) for the closest agreeing real annotation, or None.

  Agreement is bidirectional: go_id may be a more specific descendant of a generic real
  annotation (e.g. candidate GO:0008270 zinc ion binding vs. real GO:0046872 metal ion binding),
  or a more generic ancestor of a specific real annotation. Exact match is hop 0.
  """
  if go_id in real_terms:
    return go_id, 0
  cand_ancestors, cand_depths = ancestors_with_depth(go_id)
  best = None
  # real_terms is a set - iterate in a fixed order so a hop-distance tie always resolves to the
  # same real_id regardless of Python's per-process hash randomization (verified directly: two
  # back-to-back runs with an unchanged dictionary produced different aggregate coverage counts
  # before this fix, purely from set-iteration order flipping which tied real_id won).
  for real_id in sorted(real_terms):
    if real_id in cand_ancestors:
      hop = cand_depths[real_id]
      if best is None or hop < best[1]:
        best = (real_id, hop)
    else:
      real_ancestors, real_depths = ancestors_with_depth(real_id)
      if go_id in real_ancestors:
        hop = real_depths[go_id]
        if best is None or hop < best[1]:
          best = (real_id, hop)
  return best


def collect_domain_accession_candidates(domain_xrefs, mappings, overrides=None):
  """(db, accession, description) -> set(GO ids), for every cross-ref with a *2go mapping
  and/or a dictionary domain_accessions override (merged - either source alone is enough)."""
  overrides = overrides or {}
  acc_go = {}
  for xref in domain_xrefs:
    db, acc, desc = xref["db"], xref["accession"], xref["description"]
    go_ids = set(mappings.get(db, {}).get(acc, ())) | set(overrides.get(f"{db}:{acc}", ()))
    if go_ids:
      acc_go[(db, acc, desc)] = go_ids
  return acc_go


def link_protein(accession, record, mappings, ancestors_with_depth, localization_dict, verbose=False):
  """verbose=True additionally returns every raw candidate (confirmed or not) as
  all_candidates, for the visualizer (experiments/scripts/visualize_protein.py) to show what
  the databanks offered vs. what got selected. Bulk pipeline runs (main(), below) never pass
  verbose=True, so their output is unaffected.

  uncovered_raw_annotations (always populated, not just under verbose) lists every raw
  annotation - a BINDING ligand, a DOMAIN accession, or a text-tier feature's note - that
  produced *no* candidate at all because no dictionary entry covers it. This is the raw
  material for experiments/positional_mf_gaps.json (see main()).
  """
  real_mf_terms = {t["go_id"] for t in record["go_terms"] if t["aspect"] == "F"}
  features = record["features"]
  domain_features = [f for f in features if f["type"] == "DOMAIN"]

  raw_candidates = []  # (start, end, go_id, source, xref_key or None)
  uncovered_raw = []

  for f in features:
    if f["type"] == "BINDING":
      ligand = f["qualifiers"].get("ligand") or f["qualifiers"].get("ligand_id")
      go_id = localization_dict["ligand_to_go"].get(ligand)
      if go_id:
        raw_candidates.append((f["start"], f["end"], go_id, "binding_ligand", None))
      else:
        uncovered_raw.append({"kind": "ligand", "value": ligand, "start": f["start"], "end": f["end"]})
    elif f["type"] in localization_dict["feature_type_to_go"]:
      go_id = localization_dict["feature_type_to_go"][f["type"]]
      raw_candidates.append((f["start"], f["end"], go_id, "feature_type", None))
    elif f["type"] in TEXT_TIER_FEATURE_TYPES:
      note = (f["qualifiers"].get("note") or "").lower()
      matched = False
      for pattern, go_id in localization_dict["text_patterns"]:
        if pattern in note:
          raw_candidates.append((f["start"], f["end"], go_id, "text_pattern", pattern))
          matched = True
      if not matched and note:
        uncovered_raw.append({"kind": "text", "feature_type": f["type"],
                               "note": f["qualifiers"].get("note"), "start": f["start"], "end": f["end"]})

  acc_go = collect_domain_accession_candidates(
      record.get("domain_xrefs", []), mappings, localization_dict["domain_accession_to_go"])
  for xref in record.get("domain_xrefs", []):
    key = (xref["db"], xref["accession"], xref["description"])
    if key not in acc_go:
      uncovered_raw.append({"kind": "domain_accession", "db": xref["db"], "accession": xref["accession"],
                             "description": xref["description"]})
  if domain_features and acc_go:
    bridged_keys = set()
    for f in domain_features:
      note_tokens = normalize(f["qualifiers"].get("note", ""))
      if not note_tokens:
        continue
      for key, go_ids in acc_go.items():
        _, acc, desc = key
        if note_tokens & normalize(desc):
          bridged_keys.add(key)
          for go_id in go_ids:
            raw_candidates.append((f["start"], f["end"], go_id, "domain_bridge", f"{key[0]}:{acc}"))
    # Any accession's GO candidates that never bridged to a specific domain get assigned to
    # every DOMAIN span in the protein (coarse but not silently dropped or guessed).
    for key, go_ids in acc_go.items():
      if key in bridged_keys:
        continue
      _, acc, desc = key
      for go_id in go_ids:
        for f in domain_features:
          raw_candidates.append((f["start"], f["end"], go_id, "domain_fallback_union", f"{key[0]}:{acc}"))
  elif acc_go:
    # No curated DOMAIN feature span exists at all - common for proteins that only got an
    # automatic InterPro/Pfam family assignment without a UniProt-curated domain boundary
    # (e.g. short single-domain proteins like actin or FGF family members). Without this,
    # every acc_go candidate here - including any dictionary domain_accessions override -
    # would have nowhere to attach and would silently vanish. The whole-protein span is the
    # coarsest possible placement, but it's still more honest than dropping the candidate.
    for key, go_ids in acc_go.items():
      _, acc, desc = key
      for go_id in go_ids:
        raw_candidates.append((0, record["sequence_length"], go_id, "domain_whole_protein_fallback", f"{key[0]}:{acc}"))

  # Table-agreement count: how many of the 3 tables independently produced this GO id anywhere
  # in this protein's domain cross-references (protein-level signal, not per-span).
  term_to_dbs = defaultdict(set)
  for (db, acc, desc), go_ids in acc_go.items():
    for go_id in go_ids:
      term_to_dbs[go_id].add(db)

  localized = []
  seen = set()
  for start, end, go_id, source, xref_key in raw_candidates:
    match = match_against_real(go_id, real_mf_terms, ancestors_with_depth)
    if match is None:
      continue
    matched_real_term, hops = match
    dedup_key = (start, end, go_id)
    if dedup_key in seen:
      continue
    seen.add(dedup_key)
    localized.append({
        "go_id": go_id,
        "start": start,
        "end": end,
        "source": source,
        "xref": xref_key,
        "table_agreement": len(term_to_dbs.get(go_id, ())) or None,
        "matched_real_term": matched_real_term,
        "match_hops": hops,
    })

  covered_real_terms = {l["matched_real_term"] for l in localized}
  unlocalized = sorted(real_mf_terms - covered_real_terms)

  result = {
      "localized": localized,
      "unlocalized_real_mf_terms": unlocalized,
      "uncovered_raw_annotations": uncovered_raw,
  }
  if verbose:
    result["all_candidates"] = [
        {
            "go_id": go_id,
            "start": start,
            "end": end,
            "source": source,
            "xref": xref_key,
            "table_agreement": len(term_to_dbs.get(go_id, ())) or None,
            "match": match_against_real(go_id, real_mf_terms, ancestors_with_depth),
        }
        for start, end, go_id, source, xref_key in raw_candidates
    ]
  return result


def print_aggregate_stats(results):
  n_proteins = len(results)
  n_with_localized = sum(1 for r in results.values() if r["localized"])
  total_localized = sum(len(r["localized"]) for r in results.values())
  total_real_mf_covered = sum(len({l["matched_real_term"] for l in r["localized"]}) for r in results.values())
  total_unlocalized = sum(len(r["unlocalized_real_mf_terms"]) for r in results.values())
  total_real_mf = total_real_mf_covered + total_unlocalized

  source_counts = Counter()
  agreement_counts = Counter()
  hop_counts = Counter()
  for r in results.values():
    for l in r["localized"]:
      source_counts[l["source"]] += 1
      agreement_counts[l["table_agreement"]] += 1
      hop_counts["exact" if l["match_hops"] == 0 else "ancestor"] += 1

  print(f"\n=== Aggregate stats over {n_proteins} proteins ===")
  print(f"Proteins with >=1 confirmed localized MF span: {n_with_localized} ({100*n_with_localized/n_proteins:.1f}%)")
  print(f"Total confirmed localized (span, GO id) links: {total_localized}")
  print(f"Real MF term-instances covered by >=1 localization: {total_real_mf_covered}/{total_real_mf} "
        f"({100*total_real_mf_covered/total_real_mf:.1f}%)" if total_real_mf else "No real MF term-instances found")
  print(f"By source: {dict(source_counts)}")
  # Non-None can appear on a non-domain source too, when a binding/feature-type GO id happens
  # to also be produced by a domain cross-reference elsewhere in the same protein.
  print(f"By table agreement count (None=no domain-table corroboration): {dict(agreement_counts)}")
  print(f"By match type: {dict(hop_counts)}")


def write_spotcheck_sample(results, raw_features, n_per_source=5):
  rng = random.Random(SEED)
  by_source = defaultdict(list)
  for accession, r in results.items():
    sources_present = {l["source"] for l in r["localized"]}
    for source in sources_present:
      by_source[source].append(accession)

  sample = {}
  for source, accessions in by_source.items():
    rng.shuffle(accessions)
    picked = accessions[:n_per_source]
    sample[source] = [
        {
            "accession": acc,
            "localized": [l for l in results[acc]["localized"] if l["source"] == source],
            "real_mf_terms": sorted({t["go_id"] for t in raw_features[acc]["go_terms"] if t["aspect"] == "F"}),
        }
        for acc in picked
    ]
  SPOTCHECK_PATH.write_text(json.dumps(sample, indent=2))
  print(f"Wrote stratified spot-check sample ({sum(len(v) for v in sample.values())} entries) -> {SPOTCHECK_PATH}")


def main():
  raw_features = json.loads(RAW_FEATURES_PATH.read_text())
  mappings = load_mappings()
  ancestors_with_depth = build_ancestor_lookup(GO_GRAPH_PATH)
  localization_dict = load_localization_dictionary(LOCALIZATION_DICT_PATH)

  results = {}
  gaps = {}
  for accession, record in raw_features.items():
    r = link_protein(accession, record, mappings, ancestors_with_depth, localization_dict)
    results[accession] = {"localized": r["localized"], "unlocalized_real_mf_terms": r["unlocalized_real_mf_terms"]}
    # Only proteins with *something* left to review go in the gap report - most proteins are
    # either fully covered or have no candidate-worthy raw material at all, and including those
    # would just be noise for the manual gap-review loop this file is written for.
    if r["uncovered_raw_annotations"] or r["unlocalized_real_mf_terms"]:
      gaps[accession] = {
          "uncovered_raw_annotations": r["uncovered_raw_annotations"],
          "unlocalized_real_mf_terms": r["unlocalized_real_mf_terms"],
      }

  OUTPUT_PATH.write_text(json.dumps(results, indent=2))
  print(f"Wrote {len(results)} protein link records -> {OUTPUT_PATH}")

  GAPS_PATH.write_text(json.dumps(gaps, indent=2))
  print(f"Wrote gap report ({len(gaps)}/{len(results)} proteins with something unresolved) -> {GAPS_PATH}")

  print_aggregate_stats(results)
  write_spotcheck_sample(results, raw_features)


if __name__ == "__main__":
  main()
