"""Fetch UniProt Swiss-Prot flat-file records for the positional_mf working set.

Data acquisition design (see experiments/sweep_log.md for the full write-up):
one bulk download of the complete Swiss-Prot flat file from UniProt's own FTP mirror
(https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/
uniprot_sprot.dat.gz) - this is UniProt's documented, sanctioned bulk-access mechanism,
not the rate-limited REST API, so a single ~700MB download satisfies "one request" rather
than thousands of per-protein queries. Cached locally and idempotent (re-running skips the
download if already present). The dump is then streamed record-by-record via
Bio.SwissProt.parse() directly from the gzip stream - the ~3-4GB decompressed content is
never held in memory at once, only the ~5000 target proteins' records are kept, everything
else is discarded as it streams past. Confirmed reachable (HTTP 200, Content-Length
699031150 bytes) and confirmed the flat-file feature/cross_references shape matches what's
needed via a single small test record (P04637), not assumed from docs.

NOT run automatically - triggers a real ~700MB network transfer, left for explicit approval.
"""
import gzip
import json
from pathlib import Path

from Bio import SwissProt

UNIPROT_DUMP_URL = ("https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
                     "knowledgebase/complete/uniprot_sprot.dat.gz")
LOCAL_DUMP_PATH = Path("data/uniprot/uniprot_sprot.dat.gz")
SELECTION_PATH = Path("experiments/positional_mf_well_known_proteins.json")
OUTPUT_PATH = Path("experiments/positional_mf_raw_features.json")

# First-pass candidate list of feature types that plausibly localize a molecular_function
# (catalytic/binding/nucleic-acid-interaction) rather than describing cellular_component
# topology (TRANSMEM/SIGNAL/TRANSIT), structure (HELIX/STRAND/TURN/COILED), PTMs (MOD_RES/
# CROSSLNK/LIPID/CARBOHYD), or variants (VARIANT/VAR_SEQ/MUTAGEN). REGION is included but
# flagged: its span sizes vary enormously (a few residues to most of the protein), so it
# may need a length filter or separate handling once we see real coverage - not resolved yet.
MF_RELEVANT_FEATURE_TYPES = {
    "ACT_SITE", "BINDING", "DNA_BIND", "METAL", "SITE", "CA_BIND", "NP_BIND", "ZN_FING",
    "DOMAIN", "MOTIF", "REGION",
}


def download_dump(dest: Path = LOCAL_DUMP_PATH, url: str = UNIPROT_DUMP_URL) -> Path:
  """One-time bulk download. Idempotent: skips if already cached locally."""
  if dest.exists():
    print(f"[cache hit] {dest} already exists ({dest.stat().st_size / 1e6:.0f} MB), skipping download")
    return dest
  dest.parent.mkdir(parents=True, exist_ok=True)
  import urllib.request
  print(f"Downloading {url} -> {dest} (one-time, ~700MB)...")
  urllib.request.urlretrieve(url, dest)
  print(f"Done: {dest.stat().st_size / 1e6:.0f} MB")
  return dest


def extract_for_proteins(dump_path: Path, target_accessions: set) -> dict:
  """Stream-parse the flat file, keeping only records for target_accessions.

  Bio.SwissProt.parse() yields one record at a time from the gzip stream; non-matching
  records are discarded immediately, so memory use stays bounded by the match set, not
  the full dump size.
  """
  found = {}
  remaining = set(target_accessions)
  with gzip.open(dump_path, "rt") as fh:
    for record in SwissProt.parse(fh):
      if not remaining:
        break
      accession = record.accessions[0]
      if accession not in remaining:
        continue
      remaining.discard(accession)

      features = [
          {"type": f.type, "start": int(f.location.start), "end": int(f.location.end),
           "qualifiers": dict(f.qualifiers)}
          for f in record.features
          if f.type in MF_RELEVANT_FEATURE_TYPES
      ]
      go_terms = [
          {"go_id": xref[1], "aspect": xref[2].split(":")[0], "name": xref[2].split(":", 1)[1],
           "evidence": xref[3]}
          for xref in record.cross_references
          if xref[0] == "GO"
      ]
      # PROSITE/Pfam/InterPro accessions, needed to bridge DOMAIN features to prosite2go/
      # pfam2go/interpro2go entries - these don't carry a position themselves, unlike GO xrefs.
      domain_xrefs = [
          {"db": xref[0], "accession": xref[1], "description": xref[2]}
          for xref in record.cross_references
          if xref[0] in ("PROSITE", "Pfam", "InterPro")
      ]
      found[accession] = {
          "sequence_length": len(record.sequence),
          "features": features,
          "go_terms": go_terms,
          "domain_xrefs": domain_xrefs,
      }
  if remaining:
    print(f"[warning] {len(remaining)} target accessions not found in the dump "
          f"(likely obsolete/merged accessions)")
  return found


def main():
  selection = json.loads(SELECTION_PATH.read_text())
  target_accessions = {pid for pids in selection["by_split"].values() for pid in pids}
  print(f"Target: {len(target_accessions)} accessions")

  dump_path = download_dump()
  extracted = extract_for_proteins(dump_path, target_accessions)

  OUTPUT_PATH.write_text(json.dumps(extracted, indent=2))
  print(f"Extracted {len(extracted)}/{len(target_accessions)} proteins -> {OUTPUT_PATH}")

  n_with_mf_features = sum(1 for v in extracted.values() if v["features"])
  print(f"Proteins with >=1 candidate MF-relevant positional feature: "
        f"{n_with_mf_features}/{len(extracted)} ({100 * n_with_mf_features / len(extracted):.1f}%)")


if __name__ == "__main__":
  main()
