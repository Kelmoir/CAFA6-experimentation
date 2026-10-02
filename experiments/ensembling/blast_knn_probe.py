"""Spike (experiments/cafa6_next_steps_plan.md Tier 1 item 1): is a sequence-similarity
KNN baseline decorrelated enough from trial-013's ESM-2 classifier to be worth building
into an ensemble? No BLAST+ install available locally, so this approximates sequence
similarity with amino-acid k-mer cosine similarity + weighted KNN label transfer --
cheap, dependency-free (numpy/scipy/sklearn, already in env/), and good enough to answer
the decorrelation question without a real BLAST index.

Mirrors run_diagnostic()'s exact validation slice (first 2000 rows of the validation
split, same as trial-013's --run-diagnostic call) so the two scorers are compared on
identical examples and an identical evaluable-label filter.
"""
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from training_run import data_collate_function, preprocess_batch  # noqa: E402

DATA_PATH = Path("data/cafa-6-protein-function-prediction/dataset_full")
CHECKPOINT_DIR = Path("models/full-label-trial013-trainNfull/best_checkpoint")
MAX_VAL_SAMPLES = 2000
K_NEIGHBORS = 10
KMER_K = 3
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYXUZOBJ"  # 20 standard + ambiguous/rare codes seen in UniProt


def kmer_vectorize(sequences, k=KMER_K):
    aa_index = {c: i for i, c in enumerate(AMINO_ACIDS)}
    base = len(AMINO_ACIDS)
    rows, cols, data = [], [], []
    for row_i, seq in enumerate(sequences):
        counts = {}
        for i in range(len(seq) - k + 1):
            idx = 0
            valid = True
            for ch in seq[i:i + k]:
                if ch not in aa_index:
                    valid = False
                    break
                idx = idx * base + aa_index[ch]
            if valid:
                counts[idx] = counts.get(idx, 0) + 1
        for idx, c in counts.items():
            rows.append(row_i)
            cols.append(idx)
            data.append(c)
    mat = csr_matrix((data, (rows, cols)), shape=(len(sequences), base ** k), dtype=np.float32)
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return mat.multiply(1.0 / norms[:, None]).tocsr()


def labels_to_multihot(raw_labels_list, label_names, label2id):
    mat = np.zeros((len(raw_labels_list), len(label_names)), dtype=np.float32)
    for i, raw in enumerate(raw_labels_list):
        for item in raw:
            name = item if isinstance(item, str) else label_names[item]
            if name in label2id:
                mat[i, label2id[name]] = 1.0
    return mat


def main():
    print("[INFO] Loading dataset_full...")
    dataset = load_from_disk(str(DATA_PATH))
    train_split, val_split = dataset["train"], dataset["validation"]

    label_names = [str(name) for name in train_split.features["labels"].feature.names]
    label2id = {name: i for i, name in enumerate(label_names)}

    n_val = min(MAX_VAL_SAMPLES, len(val_split))
    val_subset = val_split.select(range(n_val))
    print(f"[INFO] train={len(train_split)} val_subset={n_val} labels={len(label_names)}")

    print("[INFO] Building label matrices...")
    train_labels = labels_to_multihot(train_split["labels"], label_names, label2id)
    val_labels = labels_to_multihot(val_subset["labels"], label_names, label2id)

    print("[INFO] Building k-mer vectors (train)...")
    train_vecs = kmer_vectorize(train_split["sequence"])
    print("[INFO] Building k-mer vectors (val)...")
    val_vecs = kmer_vectorize(val_subset["sequence"])

    print("[INFO] Fitting KNN index and querying...")
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="cosine", algorithm="brute", n_jobs=-1)
    knn.fit(train_vecs)
    distances, indices = knn.kneighbors(val_vecs)
    weights = np.clip(1.0 - distances, 0.0, None)

    print("[INFO] Transferring labels via similarity-weighted KNN vote...")
    knn_scores = np.zeros_like(val_labels)
    for i in range(n_val):
        w = weights[i]
        if w.sum() == 0:
            w = np.ones_like(w)
        knn_scores[i] = (train_labels[indices[i]] * w[:, None]).sum(axis=0) / w.sum()

    print("[INFO] Loading trial-013 checkpoint for comparison (CPU)...")
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model.eval()

    preprocess_partial = partial(
        preprocess_batch, tokenizer=tokenizer, label2id=label2id,
        unique_items=label_names, max_length=4192,
    )
    val_subset_proc = val_subset.with_transform(transform=preprocess_partial)
    loader = torch.utils.data.DataLoader(
        val_subset_proc, batch_size=4, shuffle=False,
        collate_fn=partial(data_collate_function, tokenizer=tokenizer),
    )

    print("[INFO] Running model inference on CPU...")
    all_probs = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            all_probs.append(torch.sigmoid(logits).numpy())
    model_probs = np.concatenate(all_probs, axis=0)

    print("[INFO] Computing per-label AUROC for both scorers...")
    evaluable = [j for j in range(len(label_names)) if val_labels[:, j].min() != val_labels[:, j].max()]
    model_auroc = np.array([roc_auc_score(val_labels[:, j], model_probs[:, j]) for j in evaluable])
    knn_auroc = np.array([roc_auc_score(val_labels[:, j], knn_scores[:, j]) for j in evaluable])

    corr = np.corrcoef(model_auroc, knn_auroc)[0, 1]
    diff = knn_auroc - model_auroc

    print(f"[RESULT] n_labels_evaluable={len(evaluable)}")
    print(f"[RESULT] model mean_auroc={model_auroc.mean():.4f} median={np.median(model_auroc):.4f}")
    print(f"[RESULT] knn   mean_auroc={knn_auroc.mean():.4f} median={np.median(knn_auroc):.4f}")
    print(f"[RESULT] per-label AUROC correlation (model vs knn) = {corr:.4f}")
    print(f"[RESULT] labels where KNN beats model by >=0.05 AUROC: {(diff >= 0.05).sum()} / {len(evaluable)}")
    print(f"[RESULT] labels where model beats KNN by >=0.05 AUROC: {(diff <= -0.05).sum()} / {len(evaluable)}")


if __name__ == "__main__":
    main()
