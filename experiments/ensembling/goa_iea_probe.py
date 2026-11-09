"""Candidate (d) from Tier 1 item 1's original list (experiments/cafa6_next_steps_plan.md):
GOA's own non-experimental/electronic-annotation evidence, used by 3rd place CAFA-5 and the
verified CAFA-6 11th-place solution. Never actually tried -- (a)/(b)/(c) were, and the
BLAST-KNN + ESM stacking line that came out of (b) was closed out 2026-10-04 (tied with a
trivial baseline, not adopted).

Approximated here via UniProt/Swiss-Prot's own Pfam/InterPro/PROSITE domain cross-references
(already present in the locally-cached data/uniprot/uniprot_sprot.dat.gz dump, verified via a
direct zgrep check), mapped to GO terms through the GO Consortium's *2go cross-reference tables
(data/go_mappings/, already fetched for the mf_span side experiment) -- this IS the actual
mechanism UniProt's automatic/IEA-evidence-code GO annotation pipeline uses internally, not an
approximation of something else. Each protein's directly-mapped terms are propagated to their
full is_a/part_of ancestor closure (reusing generate_dataset.py's restrict_to_true_path) before
scoring, since a true label's ancestors are also true under the true-path rule.

Same validation slice, same evaluable-label filter, same checkpoint, same decorrelation-check
methodology as blast_knn_probe_real.py -- the cheap check item 1's own process rule calls for
before investing in any stacking build.
"""
import gzip
import sys
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import obonet
import torch
from Bio import SwissProt
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from training_run import data_collate_function, preprocess_batch  # noqa: E402
from generate_dataset import restrict_to_true_path  # noqa: E402
from blast_knn_probe_real import labels_to_multihot  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "data/cafa-6-protein-function-prediction/dataset_full"
CHECKPOINT_DIR = REPO_ROOT / "models/full-label-trial013-trainNfull/best_checkpoint"
UNIPROT_DUMP = REPO_ROOT / "data/uniprot/uniprot_sprot.dat.gz"
GO_MAPPINGS_DIR = REPO_ROOT / "data/go_mappings"
GO_OBO_PATH = REPO_ROOT / "data/cafa-6-protein-function-prediction/Train/go-basic.obo"

MAX_VAL_SAMPLES = 2000


def parse_2go_file(path: Path) -> dict:
    """{bare_domain_accession: set(GO_ID)} from one GO Consortium `*2go` table. Each line looks
    like 'Pfam:PF00001 7tm_1 > GO:G protein-coupled receptor activity ; GO:0004930' -- the domain
    accession is right after the DB prefix, the GO id is the last field."""
    mapping = {}
    for line in path.read_text().splitlines():
        if line.startswith("!") or not line.strip():
            continue
        left, go_id = line.rsplit("; ", 1)
        db_acc = left.split(">")[0].strip().split()[0]
        bare_acc = db_acc.split(":", 1)[1]
        mapping.setdefault(bare_acc, set()).add(go_id.strip())
    return mapping


def build_domain_to_go(mappings_dir: Path) -> dict:
    merged = {}
    for fname in ("pfam2go.txt", "interpro2go.txt", "prosite2go.txt"):
        for acc, go_ids in parse_2go_file(mappings_dir / fname).items():
            merged.setdefault(acc, set()).update(go_ids)
    return merged


def fetch_cross_refs(dump_path: Path, accessions: set) -> dict:
    """{accession: set(domain_accession)}, Pfam/InterPro/PROSITE only, via a single streaming
    pass over the Swiss-Prot dump (Bio.SwissProt.parse() -- never holds the full ~4GB
    decompressed dump in memory, same pattern as experiments/mf_span's fetch script). Matches on
    each record's PRIMARY accession only, same convention as that script."""
    found = {}
    remaining = set(accessions)
    with gzip.open(dump_path, "rt") as fh:
        for record in SwissProt.parse(fh):
            if not remaining:
                break
            acc = record.accessions[0]
            if acc not in remaining:
                continue
            remaining.discard(acc)
            found[acc] = {xref[1] for xref in record.cross_references
                          if xref[0] in ("Pfam", "InterPro", "PROSITE")}
    if remaining:
        print(f"[WARN] {len(remaining)}/{len(accessions)} accessions not found in Swiss-Prot dump "
              f"(scored all-zero -- expected for TrEMBL-only proteins, Swiss-Prot is reviewed-only)")
    return found


def main():
    print("[INFO] Loading dataset_full...")
    dataset = load_from_disk(str(DATA_PATH))
    val_split = dataset["validation"]
    label_names = [str(name) for name in val_split.features["labels"].feature.names]
    label2id = {name: i for i, name in enumerate(label_names)}

    n_val = min(MAX_VAL_SAMPLES, len(val_split))
    val_subset = val_split.select(range(n_val))
    accessions = val_subset["protein"]
    val_labels = labels_to_multihot(val_subset["labels"], label_names, label2id)
    print(f"[INFO] val_subset={n_val} labels={len(label_names)}")

    print("[INFO] Parsing *2go mapping tables...")
    domain_to_go = build_domain_to_go(GO_MAPPINGS_DIR)
    print(f"[INFO] {len(domain_to_go)} domain accessions map to >=1 GO term")

    print("[INFO] Loading go-basic.obo for ancestor propagation...")
    obo_graph = restrict_to_true_path(obonet.read_obo(str(GO_OBO_PATH)))
    ancestor_cache = {}

    def ancestors_of(term):
        if term not in ancestor_cache:
            ancestor_cache[term] = nx.descendants(obo_graph, term) if term in obo_graph else set()
        return ancestor_cache[term]

    print(f"[INFO] Streaming Swiss-Prot dump for {len(set(accessions))} target accessions "
          f"(single pass over {UNIPROT_DUMP.name})...")
    cross_refs = fetch_cross_refs(UNIPROT_DUMP, set(accessions))

    print("[INFO] Scoring IEA-transfer candidate terms per protein...")
    iea_scores = np.zeros((n_val, len(label_names)), dtype=np.float32)
    n_with_hits = 0
    for i, acc in enumerate(accessions):
        domain_accs = cross_refs.get(acc, set())
        if not domain_accs:
            continue
        direct_terms = set()
        for d_acc in domain_accs:
            direct_terms |= domain_to_go.get(d_acc, set())
        if not direct_terms:
            continue
        closure = set(direct_terms)
        for term in direct_terms:
            closure |= ancestors_of(term)
        hit = False
        for term in closure:
            j = label2id.get(term)
            if j is not None:
                iea_scores[i, j] = 1.0
                hit = True
        if hit:
            n_with_hits += 1
    print(f"[INFO] {n_with_hits}/{n_val} validation proteins got >=1 evaluable IEA-transferred term")

    cached_probs_path = Path(__file__).resolve().parent / "blast_work" / "model_probs_partial.npy"
    cached_n_done_path = cached_probs_path.with_name("model_probs_n_done.txt")
    if (cached_probs_path.exists() and cached_n_done_path.exists()
            and int(cached_n_done_path.read_text()) >= n_val):
        print(f"[INFO] Reusing cached trial-013 model probs from {cached_probs_path} "
              f"(same val_subset blast_knn_probe_real.py already scored)")
        model_probs = np.load(cached_probs_path)[:n_val]
    else:
        print("[INFO] Loading trial-013 checkpoint for comparison (CPU)...")
        model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
        model.eval()

        preprocess_partial = partial(preprocess_batch, tokenizer=tokenizer, label2id=label2id,
                                      unique_items=label_names, max_length=4192)
        val_subset_proc = val_subset.with_transform(transform=preprocess_partial)
        batch_size = 4
        loader = torch.utils.data.DataLoader(
            val_subset_proc, batch_size=batch_size, shuffle=False,
            collate_fn=partial(data_collate_function, tokenizer=tokenizer))

        model_probs = np.zeros((n_val, len(label_names)), dtype=np.float32)
        print("[INFO] Running model inference on CPU...", flush=True)
        with torch.inference_mode():
            for batch_i, batch in enumerate(loader):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                probs = torch.sigmoid(logits).numpy()
                start = batch_i * batch_size
                model_probs[start:start + len(probs)] = probs
                if batch_i % 50 == 0:
                    print(f"[INFO] inference progress: {start}/{n_val}", flush=True)

    print("[INFO] Computing per-label AUROC for both scorers...")
    evaluable = [j for j in range(len(label_names)) if val_labels[:, j].min() != val_labels[:, j].max()]
    model_auroc = np.array([roc_auc_score(val_labels[:, j], model_probs[:, j]) for j in evaluable])
    iea_auroc = np.array([roc_auc_score(val_labels[:, j], iea_scores[:, j]) for j in evaluable])

    corr = np.corrcoef(model_auroc, iea_auroc)[0, 1]
    diff = iea_auroc - model_auroc

    print(f"[RESULT] n_labels_evaluable={len(evaluable)}")
    print(f"[RESULT] model mean_auroc={model_auroc.mean():.4f} median={np.median(model_auroc):.4f}")
    print(f"[RESULT] IEA-transfer mean_auroc={iea_auroc.mean():.4f} median={np.median(iea_auroc):.4f}")
    print(f"[RESULT] per-label AUROC correlation (model vs IEA-transfer) = {corr:.4f}")
    print(f"[RESULT] labels where IEA-transfer beats model by >=0.05 AUROC: "
          f"{(diff >= 0.05).sum()} / {len(evaluable)}")
    print(f"[RESULT] labels where model beats IEA-transfer by >=0.05 AUROC: "
          f"{(diff <= -0.05).sum()} / {len(evaluable)}")


if __name__ == "__main__":
    main()
