"""Item 13 step 1, pull #1 (experiments/cafa6_next_steps_plan.md): per-annotation attribution
data (date, assigned-by lab/database, PMID reference, evidence code) for every CAFA6 training
protein, via EBI QuickGO's bulk TSV download endpoint.

The plan's originally-named convenience file (goa_uniprot_all_noiea.gaf.gz) no longer exists at
EBI's GOA FTP -- the current full dump (goa_uniprot_all.gaf.gz) is 28.4GB, and CAFA6's ~82k
proteins are a tiny fraction of the ~250M+ annotations it holds. QuickGO's REST API
(/annotation/downloadSearch) supports a batched, comma-separated geneProductId query returning
exactly the needed columns (GENE PRODUCT ID, GO TERM, GO EVIDENCE CODE, REFERENCE, ASSIGNED BY,
DATE) as a flat TSV with no row-count pagination cap (confirmed directly against a real 200-ID
batch, not assumed from docs) -- so this fetches only what's needed instead of the full dump.

Resumable: writes one TSV chunk per batch to goa_attributions_chunks/, skips batches already
fetched on a re-run. IEA (electronic, no-human-curation) rows are NOT filtered out here -- kept
so filtering can be redone without a second fetch; downstream consumers should drop
`GO EVIDENCE CODE` == "IEA" per the plan's own "no human-curation bias to find" rationale.
"""
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_TERMS = REPO_ROOT / "data/cafa-6-protein-function-prediction/Train/train_terms.tsv"
OUT_DIR = Path(__file__).resolve().parents[1] / "goa_attributions_chunks"
MERGED_OUT = Path(__file__).resolve().parents[1] / "goa_attributions.tsv"
BATCH_SIZE = 200
QUICKGO_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch"
REQUEST_DELAY_S = 0.3


def load_protein_ids() -> list:
    ids = set()
    with open(TRAIN_TERMS) as f:
        next(f)
        for line in f:
            ids.add(line.split("\t", 1)[0])
    return sorted(ids)


def fetch_batch(protein_ids: list) -> str:
    resp = requests.get(QUICKGO_URL, params={"geneProductId": ",".join(protein_ids)},
                         headers={"Accept": "text/tsv"}, timeout=60)
    resp.raise_for_status()
    return resp.text


def merge_chunks(chunk_dir: Path, out_path: Path) -> None:
    chunk_files = sorted(chunk_dir.glob("batch_*.tsv"))
    n_rows = 0
    with open(out_path, "w") as out_f:
        header_written = False
        for cf in chunk_files:
            lines = cf.read_text().splitlines()
            if not lines:
                continue
            if not header_written:
                out_f.write(lines[0] + "\n")
                header_written = True
            for line in lines[1:]:
                out_f.write(line + "\n")
                n_rows += 1
    print(f"[INFO] Merged {len(chunk_files)} chunks -> {out_path} ({n_rows} annotation rows)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protein_ids = load_protein_ids()
    batches = [protein_ids[i:i + BATCH_SIZE] for i in range(0, len(protein_ids), BATCH_SIZE)]
    print(f"[INFO] {len(protein_ids)} unique CAFA6 proteins -> {len(batches)} batches of <= {BATCH_SIZE}")

    for i, batch in enumerate(batches):
        out_path = OUT_DIR / f"batch_{i:04d}.tsv"
        if out_path.exists():
            continue
        text = fetch_batch(batch)
        out_path.write_text(text)
        if (i + 1) % 20 == 0 or (i + 1) == len(batches):
            print(f"[INFO] fetched {i + 1}/{len(batches)} batches", flush=True)
        time.sleep(REQUEST_DELAY_S)

    print("[INFO] All batches fetched. Merging chunks...")
    merge_chunks(OUT_DIR, MERGED_OUT)


if __name__ == "__main__":
    main()
