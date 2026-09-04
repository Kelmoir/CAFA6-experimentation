"""Build the "cafa6_mf_placed" dataset: proteins whose real molecular_function GO-term
annotations have a confirmed positional localization (a specific residue span, domain, or -- for
proteins with no curated DOMAIN feature to anchor to -- the whole protein) of *high quality*
evidence, i.e. not resting on homology/automatic inference.

Task framing (see the 2026-08-20 conversation): unlike the earlier "fully placed" framing (every
real MF term on a protein localized), the actual prediction task here is the location span itself
-- given a protein's sequence and its known MF GO terms, predict where in the sequence each one is
placed. That means:
  - No per-protein completeness requirement. A protein with 3 real MF terms where only 1 has a
    confirmed high-quality placement still contributes that 1 placement as a training example; the
    other 2 terms are simply absent from mf_localizations, not represented as negatives.
  - domain_fallback_union / domain_whole_protein_fallback sources (a whole-domain or whole-protein
    span, not a tight residue range) are kept -- they're still real, useful supervision for "which
    known term goes with which domain/the protein as a whole", not excluded as imprecise.
  - "High quality" is enforced instead via a two-axis evidence filter (see HOMOLOG_AUTO_TERM_EVIDENCE
    / HOMOLOG_AUTO_FEATURE_EVIDENCE below), not via requiring full coverage.

Evidence filter (both axes must pass; a localized entry with no applicable evidence value on an
axis passes that axis by default -- "no evidence" isn't proof of homology, this is a
"confirmed-homolog gets excluded" filter, not a "confirmed-non-homolog required" one):
  1. Term evidence: the matched_real_term's own GO evidence code (aspect=F annotation in
     positional_mf_raw_features.json) must not be IEA/IBA/ISO/ISS/ISA/ISM -- electronic, PAINT
     phylogenetic, or sequence/structural-similarity inference, all homology-derived or fully
     automatic.
  2. Feature evidence: the underlying UniProt feature the span comes from (BINDING/DOMAIN/
     ACT_SITE/etc, looked up by exact (start, end) match in the protein's raw features -- link_protein
     itself doesn't carry this through to positional_mf_links.json, so it's re-derived here) must
     not be ECO:0000250 (by similarity to another protein -- homology at the feature level) or
     ECO:0000256 (automatic rule/SAM match). ECO:0000255 (PROSITE-ProRule / profile match, the
     normal basis for curated domain boundaries) is deliberately NOT filtered -- excluding it would
     gut domain_bridge/domain_fallback_union almost entirely, contrary to the point above.
     domain_whole_protein_fallback has no anchoring feature (fires only when a protein has domain
     cross-references but no curated DOMAIN feature at all), so it's gated on term evidence alone.

Output: one row per protein (sequence stored once, not repeated per placement) with a nested
mf_localizations list, pushed to the HF Hub as kelmoir/cafa6_mf_placed. Protein-level train/
validation/test split is inherited from experiments/mf_span/positional_mf_protein_split.json (the same
by-EntryID, seed=42 partition generate_dataset.py uses for kelmoir/cafa6), so this dataset never
splits a protein across sets differently from the main one.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import datasets
from Bio import SeqIO

DATA_PATH = Path("./data/cafa-6-protein-function-prediction")
RAW_FEATURES_PATH = Path("experiments/mf_span/positional_mf_raw_features.json")
LINKS_PATH = Path("experiments/mf_span/positional_mf_links.json")
SPLIT_PATH = Path("experiments/mf_span/positional_mf_protein_split.json")
GO_GRAPH_PATH = DATA_PATH / "dataset" / "go_dependency_graph.json"

HOMOLOG_AUTO_TERM_EVIDENCE = {"IEA", "IBA", "ISO", "ISS", "ISA", "ISM"}
HOMOLOG_AUTO_FEATURE_EVIDENCE = {"ECO:0000250", "ECO:0000256"}

# Source categories whose span comes from a specific real feature we can look evidence up for,
# restricted to the feature type(s) that source is actually derived from in
# build_positional_mf_links.py's link_protein() -- keeps the (start, end) lookup unambiguous even
# when two different-typed features happen to share a span.
SOURCE_FEATURE_TYPES = {
    "binding_ligand": {"BINDING"},
    "domain_bridge": {"DOMAIN"},
    "domain_fallback_union": {"DOMAIN"},
    "text_pattern": {"ACT_SITE", "SITE", "MOTIF", "REGION", "ZN_FING"},
    "feature_type": {"DNA_BIND"},
    # domain_whole_protein_fallback has no anchoring feature -- absent from this map on purpose.
}

MAX_LENGTH = 1024  # matches generate_dataset.py's default ESM-2 input cap


def load_sequences(fasta_path):
  sequences = {}
  for record in SeqIO.parse(fasta_path, "fasta"):
    accession = record.id.split("|")[1]
    sequences[accession] = str(record.seq).upper().replace(r"[UZOB]", "X")
  return sequences


def build_term_evidence_lookup(raw_features):
  """(accession, go_id) -> evidence code prefix, for aspect=F GO-term annotations."""
  lookup = {}
  for accession, record in raw_features.items():
    for term in record["go_terms"]:
      if term["aspect"] == "F":
        lookup[(accession, term["go_id"])] = term["evidence"].split(":")[0]
  return lookup


def build_feature_evidence_index(raw_features):
  """accession -> (start, end, type) -> evidence string (raw ECO qualifier, may be None)."""
  index = defaultdict(dict)
  for accession, record in raw_features.items():
    for f in record["features"]:
      index[accession][(f["start"], f["end"], f["type"])] = f["qualifiers"].get("evidence")
  return index


def build_go_name_lookup(go_graph_path):
  graph = json.loads(go_graph_path.read_text())
  return {node["id"]: node.get("name") for node in graph["graph"]["nodes"]}


def feature_evidence_for(feature_index, accession, source, start, end):
  """Returns the evidence string for the feature backing this localized entry, or None if the
  source has no anchoring feature (domain_whole_protein_fallback) or none is found.

  Iterates candidate types in a fixed (sorted) order rather than raw set order: if a span
  coincidentally matches more than one feature type with differing evidence, unordered set
  iteration would flip which one wins across runs purely from PYTHONHASHSEED, the same class of
  nondeterminism merge_llm_findings.py hit and documented (see match_against_real there) --
  verified directly here too, via back-to-back runs producing different kept-entry counts
  (109136/109151/109137) before this fix.
  """
  types = SOURCE_FEATURE_TYPES.get(source)
  if types is None:
    return None
  by_span = feature_index.get(accession, {})
  for ftype in sorted(types):
    ev = by_span.get((start, end, ftype))
    if ev is not None:
      return ev
  return None


def passes_quality_filter(term_evidence, feature_evidence):
  if term_evidence in HOMOLOG_AUTO_TERM_EVIDENCE:
    return False
  if feature_evidence:
    # A feature can carry multiple comma-separated evidence entries, each "ECO:XXXXXXX|source"
    # (e.g. "ECO:0000250|UniProtKB:Q94IP1"); take just the bare ECO code before "|" per entry.
    # Reject only if *every* listed code is homology/automatic -- one high-quality corroborating
    # code is enough to keep it.
    codes = {c.strip().split("|")[0] for c in feature_evidence.split(",")}
    if codes and codes <= HOMOLOG_AUTO_FEATURE_EVIDENCE:
      return False
  return True


def build_rows(raw_features, links, sequences, term_evidence_lookup, feature_index, go_names):
  rows = []
  n_skipped_no_sequence = 0
  n_skipped_too_long = 0
  n_localized_total = 0
  n_localized_kept = 0

  for accession, link_record in links.items():
    sequence = sequences.get(accession)
    if sequence is None:
      n_skipped_no_sequence += 1
      continue
    if len(sequence) > MAX_LENGTH:
      n_skipped_too_long += 1
      continue

    # datasets.Sequence(Features({...})) stores each row's value in columnar form (a dict whose
    # values are same-length lists), not as a list of per-item dicts -- passing the latter fails
    # to encode (AttributeError: 'list' object has no attribute 'get'), a well-known HF gotcha.
    localizations = defaultdict(list)
    n_kept_this_protein = 0
    for entry in link_record["localized"]:
      n_localized_total += 1
      matched_real_term = entry["matched_real_term"]
      term_evidence = term_evidence_lookup.get((accession, matched_real_term))
      feature_evidence = feature_evidence_for(
          feature_index, accession, entry["source"], entry["start"], entry["end"])
      if not passes_quality_filter(term_evidence, feature_evidence):
        continue
      n_localized_kept += 1
      n_kept_this_protein += 1
      localizations["go_id"].append(matched_real_term)
      localizations["name"].append(go_names.get(matched_real_term))
      localizations["start"].append(entry["start"])
      localizations["end"].append(entry["end"])
      localizations["trigger_go_id"].append(entry["go_id"])
      localizations["trigger_name"].append(go_names.get(entry["go_id"]))
      localizations["match_hops"].append(entry["match_hops"])
      localizations["source"].append(entry["source"])
      localizations["table_agreement"].append(entry["table_agreement"])
      localizations["term_evidence"].append(term_evidence)
      localizations["feature_evidence"].append(feature_evidence)

    if not n_kept_this_protein:
      continue

    placed_mf_terms = ", ".join(sorted(set(localizations["go_id"])))

    rows.append({
        "protein": accession,
        "sequence": sequence,
        "mf_localizations": dict(localizations),
        "placed_mf_terms": placed_mf_terms,
    })

  print(f"[INFO] {n_skipped_no_sequence} proteins skipped (no sequence found in fasta)")
  print(f"[INFO] {n_skipped_too_long} proteins skipped (sequence > {MAX_LENGTH} aa)")
  print(f"[INFO] Localized entries: {n_localized_total} total, {n_localized_kept} kept "
        f"after quality filter ({100 * n_localized_kept / n_localized_total:.1f}%)")
  print(f"[INFO] {len(rows)} proteins retained with >=1 high-quality localization")
  return rows


def rows_to_dataset(rows):
  features = datasets.Features({
      "protein": datasets.Value("string"),
      "sequence": datasets.Value("string"),
      "placed_mf_terms": datasets.Value("string"),
      "mf_localizations": datasets.Sequence(datasets.Features({
          "go_id": datasets.Value("string"),
          "name": datasets.Value("string"),
          "start": datasets.Value("int64"),
          "end": datasets.Value("int64"),
          "trigger_go_id": datasets.Value("string"),
          "trigger_name": datasets.Value("string"),
          "match_hops": datasets.Value("int64"),
          "source": datasets.Value("string"),
          "table_agreement": datasets.Value("int64"),
          "term_evidence": datasets.Value("string"),
          "feature_evidence": datasets.Value("string"),
      })),
  })
  info = datasets.DatasetInfo(
      description=(
          "CAFA-6 protein sequences with molecular_function GO-term annotations positionally "
          "localized to a residue span, domain, or whole protein -- derived from UniProt "
          "Swiss-Prot feature/cross-reference data via experiments/mf_span/scripts/build_positional_mf_links.py "
          "and filtered to high-quality (non-homology, non-automatic) evidence only. The prediction "
          "task this supports: given a protein sequence and one of its known MF GO terms, locate the "
          "span responsible for it."
      ),
      citation="Iddo Friedberg, Predrag Radivojac, Paul D Thomas, An Phan, M. Clara De Paolis Kaluza, "
               "Damiano Piovesan, Parnal Joshi, Chris Mungall, Martyna Plomecka, Walter Reade, and "
               "María Cruz. CAFA 6 Protein Function Prediction. "
               "https://kaggle.com/competitions/cafa-6-protein-function-prediction, 2025. Kaggle.",
      features=features,
  )
  return datasets.Dataset.from_list(rows, features=features, info=info)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--push-to-hub", action="store_true",
                   help="Push the built dataset to kelmoir/cafa6_mf_placed. Off by default -- "
                        "build and save locally first, inspect, then re-run with this flag.")
  ap.add_argument("--hf-repo-id", default="kelmoir/cafa6_mf_placed")
  ap.add_argument("--out-dir", default=str(DATA_PATH / "dataset_mf_placed"))
  args = ap.parse_args()

  print("[INFO] Loading raw features, links, split, sequences, GO names...")
  raw_features = json.loads(RAW_FEATURES_PATH.read_text())
  links = json.loads(LINKS_PATH.read_text())
  split = json.loads(SPLIT_PATH.read_text())["by_split"]
  sequences = load_sequences(DATA_PATH / "Train" / "train_sequences.fasta")
  go_names = build_go_name_lookup(GO_GRAPH_PATH)

  term_evidence_lookup = build_term_evidence_lookup(raw_features)
  feature_index = build_feature_evidence_index(raw_features)

  rows = build_rows(raw_features, links, sequences, term_evidence_lookup, feature_index, go_names)
  rows_by_accession = {r["protein"]: r for r in rows}

  dataset_dict = {}
  for split_name, accessions in split.items():
    split_rows = [rows_by_accession[acc] for acc in accessions if acc in rows_by_accession]
    dataset_dict[split_name] = rows_to_dataset(split_rows)
    print(f"[INFO] {split_name}: {len(split_rows)} proteins")

  dataset = datasets.DatasetDict(dataset_dict)

  out_dir = Path(args.out_dir)
  dataset.save_to_disk(out_dir)
  print(f"[INFO] Saved dataset to {out_dir}")

  if args.push_to_hub:
    dataset.push_to_hub(args.hf_repo_id, private=False)
    print(f"[INFO] Pushed to hub: {args.hf_repo_id}")
  else:
    print("[INFO] --push-to-hub not set; skipping Hub push.")


if __name__ == "__main__":
  main()
