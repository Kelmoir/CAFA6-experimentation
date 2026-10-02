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


def build_features(slice_dict):
    """Flattens a (protein x term) slice dict (from load_cached_meta_eval() or
    compute_meta_train_slice()) into row-per-(protein, term) features:
    [esm_prob, blast_score, log1p(term_freq)]."""
    term_freq = np.log1p(slice_dict["train_labels"].sum(axis=0)).astype(np.float32)
    n_rows, n_labels = slice_dict["val_labels"].shape
    X = np.stack([
        slice_dict["esm_prob"].ravel(),
        slice_dict["blast_score"].ravel(),
        np.tile(term_freq, n_rows),
    ], axis=1).astype(np.float32)
    y = slice_dict["val_labels"].ravel().astype(np.float32)
    assert np.isfinite(X).all(), "non-finite value in constructed features"
    return X, y


def assert_no_overlap():
    assert META_TRAIN_START >= MAX_VAL_SAMPLES, (
        f"meta-train range [{META_TRAIN_START}:{META_TRAIN_END}) overlaps meta-eval's "
        f"[0:{MAX_VAL_SAMPLES}) rows -- would leak eval rows into training"
    )


def fit_meta_model(X_train, y_train):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist",
        eval_metric="auc", random_state=42,
    )
    model.fit(X_train, y_train)

    train_preds = model.predict_proba(X_train)[:, 1]
    assert train_preds.std() > 1e-6, (
        "meta-model predictions collapsed to a near-constant value on its own training "
        "data -- likely needs negative subsampling (see the design spec's Meta-model "
        "risk note) before being trusted on the real meta-train set"
    )
    return model


def compute_meta_train_slice(start, end):
    """Builds (protein x term) score matrices for validation rows [start:end), reusing
    the already-built BLAST database and the resumable checkpoint-every-20-batches
    inference pattern blast_knn_probe_real.py validated. Cache filenames are keyed by
    (start, end) so a smoke-test range never collides with the real 2000:4000 run."""
    import subprocess
    from functools import partial

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from training_run import data_collate_function, preprocess_batch  # noqa: E402

    dataset = load_from_disk(str(DATA_PATH))
    train_split, val_split = dataset["train"], dataset["validation"]
    label_names = [str(name) for name in train_split.features["labels"].feature.names]
    label2id = {name: i for i, name in enumerate(label_names)}

    val_subset = val_split.select(range(start, end))
    n_rows = end - start
    val_ids = [f"metatrain_{i}" for i in range(n_rows)]
    train_ids = [f"train_{i}" for i in range(len(train_split))]
    train_id_to_idx = {tid: i for i, tid in enumerate(train_ids)}

    train_labels = labels_to_multihot(train_split["labels"], label_names, label2id)
    val_labels = labels_to_multihot(val_subset["labels"], label_names, label2id)

    tag = f"{start}_{end}"
    metatrain_fasta = WORK_DIR / f"metatrain_{tag}.fasta"
    blast_db = WORK_DIR / "train_blastdb"
    blast_out = WORK_DIR / f"blast_results_metatrain_{tag}.tsv"

    if blast_out.exists():
        print(f"[INFO] Reusing cached meta-train BLAST results at {blast_out}")
    else:
        write_fasta(metatrain_fasta, val_ids, val_subset["sequence"])
        print(f"[INFO] Running blastp for rows [{start}:{end}) against the existing train_blastdb...")
        subprocess.run(
            [str(BLAST_BIN_DIR / "blastp"), "-query", str(metatrain_fasta), "-db", str(blast_db),
             "-out", str(blast_out), "-outfmt", "6 qseqid sseqid bitscore pident",
             "-max_target_seqs", "10", "-num_threads", "8"],
            check=True, capture_output=True, text=True,
        )

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

    probs_ckpt = WORK_DIR / f"model_probs_metatrain_{tag}_partial.npy"
    n_done_ckpt = WORK_DIR / f"model_probs_metatrain_{tag}_n_done.txt"
    expected_shape = (n_rows, len(label_names))
    if probs_ckpt.exists() and n_done_ckpt.exists():
        cached = np.load(probs_ckpt)
        if cached.shape == expected_shape:
            esm_prob = cached
            n_done = int(n_done_ckpt.read_text())
            print(f"[INFO] Resuming meta-train inference from checkpoint: {n_done}/{n_rows} done")
        else:
            print(
                f"[WARN] cached checkpoint shape {cached.shape} != expected {expected_shape} "
                "-- discarding stale checkpoint, starting fresh"
            )
            esm_prob = np.zeros(expected_shape, dtype=np.float32)
            n_done = 0
    else:
        esm_prob = np.zeros(expected_shape, dtype=np.float32)
        n_done = 0

    n_batches_done = n_done // batch_size
    with torch.inference_mode():
        for batch_i, batch in enumerate(loader):
            if batch_i < n_batches_done:
                continue
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            probs = torch.sigmoid(logits).numpy()
            batch_start = batch_i * batch_size
            esm_prob[batch_start:batch_start + len(probs)] = probs
            n_done = batch_start + len(probs)
            if batch_i % 20 == 0:
                np.save(probs_ckpt, esm_prob)
                n_done_ckpt.write_text(str(n_done))
                print(f"[INFO] meta-train inference progress: {n_done}/{n_rows}", flush=True)
    np.save(probs_ckpt, esm_prob)
    n_done_ckpt.write_text(str(n_done))

    return {
        "esm_prob": esm_prob, "blast_score": blast_score, "val_labels": val_labels,
        "train_labels": train_labels, "label_names": label_names,
    }
