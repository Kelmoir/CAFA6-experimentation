"""Standalone correctness check (no pytest in this repo -- see CLAUDE.md): confirms
load_cached_meta_eval() reproduces the exact numbers already logged in
experiments/cafa6_next_steps_plan.md for the real-BLAST probe, before any new BLAST/
inference compute is spent on the meta-train slice (Task 2+).

Run via: env/bin/python experiments/ensembling/check_meta_eval_regression.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from stacking_pipeline import load_cached_meta_eval

data = load_cached_meta_eval()
val_labels, esm_prob, blast_score = data["val_labels"], data["esm_prob"], data["blast_score"]

evaluable = [j for j in range(val_labels.shape[1]) if val_labels[:, j].min() != val_labels[:, j].max()]
assert len(evaluable) == 9401, f"n_labels_evaluable={len(evaluable)}, expected 9401"

model_auroc = np.array([roc_auc_score(val_labels[:, j], esm_prob[:, j]) for j in evaluable])
blast_auroc = np.array([roc_auc_score(val_labels[:, j], blast_score[:, j]) for j in evaluable])

assert abs(model_auroc.mean() - 0.8731) < 0.001, f"model mean AUROC {model_auroc.mean():.4f}, expected ~0.8731"
assert abs(blast_auroc.mean() - 0.6854) < 0.001, f"blast mean AUROC {blast_auroc.mean():.4f}, expected ~0.6854"

print("[PASS] meta-eval regression check: n_labels_evaluable, model/blast mean AUROC all match logged values")
