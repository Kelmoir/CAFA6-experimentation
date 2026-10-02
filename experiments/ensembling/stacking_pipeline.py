"""Fuses trial-013's ESM-2 predictions with the BLAST-KNN retrieval score via a global
XGBoost meta-model, per experiments/cafa6_next_steps_plan.md Tier 1 item 1 and
docs/superpowers/specs/2026-10-02-ensembling-stacking-pipeline-design.md.
"""
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blast_knn_probe_real import (  # noqa: E402
    BLAST_BIN_DIR, CHECKPOINT_DIR, DATA_PATH, WORK_DIR,
    labels_to_multihot, write_fasta,
)

MAX_VAL_SAMPLES = 2000  # meta-eval slice: validation rows 0:2000 (matches run_diagnostic())
META_TRAIN_START = 2000
META_TRAIN_END = 4000


def load_cached_meta_eval():
    """Re-derives meta-eval's (protein x term) score matrices from the cache files
    blast_knn_probe_real.py already wrote -- no new BLAST/inference, just parsing."""
    dataset = load_from_disk(str(DATA_PATH))
    train_split, val_split = dataset["train"], dataset["validation"]
    label_names = [str(name) for name in train_split.features["labels"].feature.names]
    label2id = {name: i for i, name in enumerate(label_names)}

    val_subset = val_split.select(range(MAX_VAL_SAMPLES))
    val_ids = [f"val_{i}" for i in range(MAX_VAL_SAMPLES)]
    train_ids = [f"train_{i}" for i in range(len(train_split))]
    train_id_to_idx = {tid: i for i, tid in enumerate(train_ids)}

    train_labels = labels_to_multihot(train_split["labels"], label_names, label2id)
    val_labels = labels_to_multihot(val_subset["labels"], label_names, label2id)

    blast_out = WORK_DIR / "blast_results.tsv"
    hits_by_query = {qid: [] for qid in val_ids}
    with open(blast_out) as f:
        for line in f:
            qseqid, sseqid, bitscore, pident = line.strip().split("\t")
            hits_by_query[qseqid].append((sseqid, float(bitscore)))

    blast_score = np.zeros_like(val_labels)
    for i, qid in enumerate(val_ids):
        hits = hits_by_query.get(qid, [])
        if not hits:
            continue
        weights = np.array([b for _, b in hits], dtype=np.float32)
        neighbor_idx = [train_id_to_idx[sseqid] for sseqid, _ in hits]
        blast_score[i] = (train_labels[neighbor_idx] * weights[:, None]).sum(axis=0) / weights.sum()

    esm_prob = np.load(WORK_DIR / "model_probs_partial.npy")
    assert esm_prob.shape == val_labels.shape, (
        f"cached model_probs_partial.npy shape {esm_prob.shape} != val_labels shape {val_labels.shape}"
    )

    return {
        "esm_prob": esm_prob, "blast_score": blast_score, "val_labels": val_labels,
        "train_labels": train_labels, "label_names": label_names,
    }
