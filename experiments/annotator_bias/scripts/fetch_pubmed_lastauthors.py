"""Item 13 step 1, pull #2 (experiments/cafa6_next_steps_plan.md): resolve each PMID collected
from pull #1 (fetch_goa_attributions.py) to its paper's last/corresponding author -- used as the
stable "lab" identity key (the PI persists across lab members rotating, which already captures
the continuity the plan's hypothesis is about, with no need to model the mentor/mentee graph
directly).

NCBI's ESummary endpoint returns a ready-made `lastauthor` field per PubMed record (confirmed
directly against a real PMID, not assumed from docs) -- no need to parse the full author list or
guess which position is "last" ourselves. Also keeps `sortfirstauthor`/`pubdate`/`source` per
record: not needed for step 2's predictability check, but the plan explicitly wants to be able to
build a mentor-mentee DAG later from this same metadata without a second pull.

Only PMIDs from NON-IEA annotation rows are resolved -- IEA rows carry no human-curation bias to
find in the first place (same rationale pull #1's docstring already applies), so their references
are irrelevant to this item.

Respects NCBI's unauthenticated E-utils rate limit (3 req/s) via REQUEST_DELAY_S; no API key, no
personal email passed (this is a public bulk-metadata-only lookup, not a registered tool).
Resumable: one JSON chunk per batch under pubmed_lastauthors_chunks/, skips batches already done.
"""
import json
import time
from pathlib import Path

import requests

GOA_FILE = Path(__file__).resolve().parents[1] / "goa_attributions.tsv"
CHUNK_DIR = Path(__file__).resolve().parents[1] / "pubmed_lastauthors_chunks"
MERGED_OUT = Path(__file__).resolve().parents[1] / "pubmed_lastauthors.tsv"
BATCH_SIZE = 200
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
REQUEST_DELAY_S = 0.4


def load_pmids(goa_file: Path) -> list:
    pmids = set()
    with open(goa_file) as f:
        header = f.readline().rstrip("\n").split("\t")
        evidence_i = header.index("GO EVIDENCE CODE")
        ref_i = header.index("REFERENCE")
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if cols[evidence_i] == "IEA":
                continue
            ref = cols[ref_i]
            if ref.startswith("PMID:"):
                pmid = ref[len("PMID:"):]
                if pmid.isdigit():
                    pmids.add(pmid)
    return sorted(pmids, key=int)


def fetch_batch(pmids: list, max_retries: int = 4) -> dict:
    for attempt in range(max_retries):
        try:
            resp = requests.get(ESUMMARY_URL, params={
                "db": "pubmed", "id": ",".join(pmids), "retmode": "json",
            }, timeout=60)
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[WARN] batch request failed ({exc}), retrying in {wait}s "
                  f"({attempt + 1}/{max_retries})", flush=True)
            time.sleep(wait)
    result = resp.json()["result"]
    out = {}
    for uid in result.get("uids", []):
        rec = result[uid]
        out[uid] = {
            "lastauthor": rec.get("lastauthor", ""),
            "sortfirstauthor": rec.get("sortfirstauthor", ""),
            "pubdate": rec.get("pubdate", ""),
            "source": rec.get("source", ""),
        }
    return out


def merge_chunks(chunk_dir: Path, out_path: Path) -> None:
    n = 0
    with open(out_path, "w") as out_f:
        out_f.write("pmid\tlastauthor\tsortfirstauthor\tpubdate\tsource\n")
        for cf in sorted(chunk_dir.glob("batch_*.json")):
            data = json.loads(cf.read_text())
            for pmid, rec in data.items():
                out_f.write(f"{pmid}\t{rec['lastauthor']}\t{rec['sortfirstauthor']}\t"
                            f"{rec['pubdate']}\t{rec['source']}\n")
                n += 1
    print(f"[INFO] Merged -> {out_path} ({n} resolved PMIDs)")


def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    pmids = load_pmids(GOA_FILE)
    batches = [pmids[i:i + BATCH_SIZE] for i in range(0, len(pmids), BATCH_SIZE)]
    print(f"[INFO] {len(pmids)} unique non-IEA PMIDs -> {len(batches)} batches of <= {BATCH_SIZE}")

    for i, batch in enumerate(batches):
        out_path = CHUNK_DIR / f"batch_{i:04d}.json"
        if out_path.exists():
            continue
        records = fetch_batch(batch)
        out_path.write_text(json.dumps(records))
        if (i + 1) % 20 == 0 or (i + 1) == len(batches):
            print(f"[INFO] fetched {i + 1}/{len(batches)} batches", flush=True)
        time.sleep(REQUEST_DELAY_S)

    print("[INFO] All batches fetched. Merging chunks...")
    merge_chunks(CHUNK_DIR, MERGED_OUT)


if __name__ == "__main__":
    main()
