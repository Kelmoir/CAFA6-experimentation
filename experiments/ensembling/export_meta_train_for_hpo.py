"""Exports meta-train's (rows 2000:4000) raw score matrices to a compact, compressed
.npz for uploading to a GPU-backed Colab session, so colab_hpo_search.py can run a
hyperparameter search there without needing this repo, the HF dataset, BLAST, or the
ESM-2 checkpoint -- just numpy/scipy/xgboost and this one file.

Exports RAW (not rank-transformed) matrices -- colab_hpo_search.py does the rank
transform itself, matching build_rank_features() exactly. Raw esm_prob/blast_score
compress far better than their rank-transformed counterparts (both are heavily
sparse/repetitive; a rank transform turns every value into a near-unique float,
which barely compresses). float16 for the two score matrices (histogram-based tree
splitting doesn't need more precision than that) and uint8 for the binary labels
keep the file small without losing anything the search needs.

Run via: env/bin/python experiments/ensembling/export_meta_train_for_hpo.py
"""
from pathlib import Path

import numpy as np

from stacking_pipeline import META_TRAIN_END, META_TRAIN_START, compute_meta_train_slice

OUT_PATH = Path(__file__).resolve().parent / "blast_work" / "meta_train_for_hpo.npz"

print(f"[INFO] Loading cached meta-train slice (rows {META_TRAIN_START}:{META_TRAIN_END})...")
meta_train = compute_meta_train_slice(META_TRAIN_START, META_TRAIN_END)

term_freq = np.log1p(meta_train["train_labels"].sum(axis=0)).astype(np.float32)

np.savez_compressed(
    OUT_PATH,
    esm_prob=meta_train["esm_prob"].astype(np.float16),
    blast_score=meta_train["blast_score"].astype(np.float16),
    val_labels=meta_train["val_labels"].astype(np.uint8),
    term_freq=term_freq,
)

size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
print(f"[DONE] wrote {OUT_PATH} ({size_mb:.1f} MB)")
