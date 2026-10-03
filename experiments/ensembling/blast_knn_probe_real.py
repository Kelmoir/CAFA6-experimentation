"""Real-BLAST follow-up to blast_knn_probe.py: the k-mer-cosine approximation scored
weak overall (mean AUROC 0.6186/median 0.5000 vs. trial-013's 0.8731/0.9259). Before
writing off BLAST-KNN as an ensembling candidate, redo the identical comparison with
real gapped local alignment (NCBI BLAST+ 2.17.0, installed in the isolated ./blast_env/
conda env to avoid conflicting with the training env's dependency graph) instead of
raw trigram cosine similarity.

Same validation slice, same evaluable-label filter, same model checkpoint as the k-mer
version, so the two probes are directly comparable.
"""
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from training_run import data_collate_function, preprocess_batch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "data/cafa-6-protein-function-prediction/dataset_full"
CHECKPOINT_DIR = REPO_ROOT / "models/full-label-trial013-trainNfull/best_checkpoint"
BLAST_BIN_DIR = REPO_ROOT / "blast_env/bin"
WORK_DIR = Path(__file__).resolve().parent / "blast_work"

MAX_VAL_SAMPLES = 2000
K_NEIGHBORS = 10
NUM_THREADS = 8


def write_fasta(path: Path, protein_ids, sequences) -> None:
    with open(path, "w") as f:
        for pid, seq in zip(protein_ids, sequences):
            f.write(f">{pid}\n{seq}\n")


def labels_to_multihot(raw_labels_list, label_names, label2id):
    mat = np.zeros((len(raw_labels_list), len(label_names)), dtype=np.float32)
    for i, raw in enumerate(raw_labels_list):
        for item in raw:
            name = item if isinstance(item, str) else label_names[item]
            if name in label2id:
                mat[i, label2id[name]] = 1.0
    return mat


def transfer_labels_via_blast(blast_out_path, query_ids, train_id_to_idx, train_labels):
    """Parses a BLAST+ outfmt-6 hits file and bitscore-weight-transfers training labels
    to each query. Shared by this probe and stacking_pipeline.py's two slices -- was
    copy-pasted in three places before (final-review minor finding, 2026-10-02)."""
    hits_by_query = {qid: [] for qid in query_ids}
    with open(blast_out_path) as f:
        for line in f:
            qseqid, sseqid, bitscore, pident = line.strip().split("\t")
            hits_by_query[qseqid].append((sseqid, float(bitscore)))

    scores = np.zeros((len(query_ids), train_labels.shape[1]), dtype=np.float32)
    for i, qid in enumerate(query_ids):
        hits = hits_by_query.get(qid, [])
        if not hits:
            continue
        weights = np.array([b for _, b in hits], dtype=np.float32)
        neighbor_idx = [train_id_to_idx[sseqid] for sseqid, _ in hits]
        scores[i] = (train_labels[neighbor_idx] * weights[:, None]).sum(axis=0) / weights.sum()
    return scores


def main():
    WORK_DIR.mkdir(exist_ok=True)

    print("[INFO] Loading dataset_full...")
    dataset = load_from_disk(str(DATA_PATH))
    train_split, val_split = dataset["train"], dataset["validation"]

    label_names = [str(name) for name in train_split.features["labels"].feature.names]
    label2id = {name: i for i, name in enumerate(label_names)}

    n_val = min(MAX_VAL_SAMPLES, len(val_split))
    val_subset = val_split.select(range(n_val))
    print(f"[INFO] train={len(train_split)} val_subset={n_val} labels={len(label_names)}")

    train_ids = [f"train_{i}" for i in range(len(train_split))]
    val_ids = [f"val_{i}" for i in range(n_val)]

    train_fasta = WORK_DIR / "train.fasta"
    val_fasta = WORK_DIR / "val.fasta"
    blast_db = WORK_DIR / "train_blastdb"
    blast_out = WORK_DIR / "blast_results.tsv"

    if blast_out.exists():
        print(f"[INFO] Reusing cached BLAST results at {blast_out}")
    else:
        print("[INFO] Writing FASTA files...")
        write_fasta(train_fasta, train_ids, train_split["sequence"])
        write_fasta(val_fasta, val_ids, val_subset["sequence"])

        print("[INFO] Building BLAST database (makeblastdb)...")
        subprocess.run(
            [str(BLAST_BIN_DIR / "makeblastdb"), "-in", str(train_fasta),
             "-dbtype", "prot", "-out", str(blast_db)],
            check=True, capture_output=True, text=True,
        )

        print("[INFO] Running blastp (this is the slow step)...")
        subprocess.run(
            [str(BLAST_BIN_DIR / "blastp"), "-query", str(val_fasta), "-db", str(blast_db),
             "-out", str(blast_out), "-outfmt", "6 qseqid sseqid bitscore pident",
             "-max_target_seqs", str(K_NEIGHBORS), "-num_threads", str(NUM_THREADS)],
            check=True, capture_output=True, text=True,
        )

    train_id_to_idx = {tid: i for i, tid in enumerate(train_ids)}

    print("[INFO] Building label matrices...")
    train_labels = labels_to_multihot(train_split["labels"], label_names, label2id)
    val_labels = labels_to_multihot(val_subset["labels"], label_names, label2id)

    print("[INFO] Parsing BLAST hits and transferring labels via bitscore weighting...")
    blast_scores = transfer_labels_via_blast(blast_out, val_ids, train_id_to_idx, train_labels)
    # Approximates "zero BLAST hits" via an all-zero score row; also counts the
    # practically-negligible case of a protein with hits whose neighbors carry no
    # labels at all (final-review minor finding, 2026-10-03).
    n_no_hits = int((blast_scores.sum(axis=1) == 0).sum())
    print(f"[INFO] {n_no_hits}/{n_val} validation proteins had zero BLAST hits (scored all-zero)")

    print("[INFO] Loading trial-013 checkpoint for comparison (CPU)...")
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model.eval()

    preprocess_partial = partial(
        preprocess_batch, tokenizer=tokenizer, label2id=label2id,
        unique_items=label_names, max_length=4192,
    )
    val_subset_proc = val_subset.with_transform(transform=preprocess_partial)
    batch_size = 4
    loader = torch.utils.data.DataLoader(
        val_subset_proc, batch_size=batch_size, shuffle=False,
        collate_fn=partial(data_collate_function, tokenizer=tokenizer),
    )

    # Resumable: CPU inference over 2000 long sequences can exceed a single background
    # task's 30-minute cap, so progress is checkpointed to disk every few batches and a
    # restart picks up where the last one left off (DataLoader order is deterministic:
    # shuffle=False, no randomness in preprocessing).
    probs_ckpt = WORK_DIR / "model_probs_partial.npy"
    n_done_ckpt = WORK_DIR / "model_probs_n_done.txt"
    if probs_ckpt.exists() and n_done_ckpt.exists():
        model_probs_arr = np.load(probs_ckpt)
        n_done = int(n_done_ckpt.read_text())
        print(f"[INFO] Resuming model inference from checkpoint: {n_done}/{n_val} done")
    else:
        model_probs_arr = np.zeros((n_val, len(label_names)), dtype=np.float32)
        n_done = 0

    print("[INFO] Running model inference on CPU...", flush=True)
    n_batches_done = n_done // batch_size
    with torch.inference_mode():
        for batch_i, batch in enumerate(loader):
            if batch_i < n_batches_done:
                continue
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            probs = torch.sigmoid(logits).numpy()
            start = batch_i * batch_size
            model_probs_arr[start:start + len(probs)] = probs
            n_done = start + len(probs)
            if batch_i % 20 == 0:
                np.save(probs_ckpt, model_probs_arr)
                n_done_ckpt.write_text(str(n_done))
                print(f"[INFO] inference progress: {n_done}/{n_val}", flush=True)
    np.save(probs_ckpt, model_probs_arr)
    n_done_ckpt.write_text(str(n_done))
    model_probs = model_probs_arr

    print("[INFO] Computing per-label AUROC for both scorers...")
    evaluable = [j for j in range(len(label_names)) if val_labels[:, j].min() != val_labels[:, j].max()]
    model_auroc = np.array([roc_auc_score(val_labels[:, j], model_probs[:, j]) for j in evaluable])
    blast_auroc = np.array([roc_auc_score(val_labels[:, j], blast_scores[:, j]) for j in evaluable])

    corr = np.corrcoef(model_auroc, blast_auroc)[0, 1]
    diff = blast_auroc - model_auroc

    print(f"[RESULT] n_labels_evaluable={len(evaluable)}")
    print(f"[RESULT] model mean_auroc={model_auroc.mean():.4f} median={np.median(model_auroc):.4f}")
    print(f"[RESULT] blast mean_auroc={blast_auroc.mean():.4f} median={np.median(blast_auroc):.4f}")
    print(f"[RESULT] per-label AUROC correlation (model vs blast) = {corr:.4f}")
    print(f"[RESULT] labels where BLAST beats model by >=0.05 AUROC: {(diff >= 0.05).sum()} / {len(evaluable)}")
    print(f"[RESULT] labels where model beats BLAST by >=0.05 AUROC: {(diff <= -0.05).sum()} / {len(evaluable)}")


if __name__ == "__main__":
    main()
