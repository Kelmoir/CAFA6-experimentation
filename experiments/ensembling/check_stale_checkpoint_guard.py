"""Standalone check (no pytest in this repo): confirms compute_meta_train_slice()
discards a stale (wrong-shaped) model_probs checkpoint instead of silently using it --
the only guard this behavior had before was the implementation itself, with no check
exercising the discard path (final-review minor finding, 2026-10-02). Run via:
    env/bin/python experiments/ensembling/check_stale_checkpoint_guard.py
"""
import contextlib
import io

import numpy as np

from stacking_pipeline import WORK_DIR, compute_meta_train_slice

start, end = 2010, 2020  # a range check_metatrain_smoke.py doesn't also use
tag = f"{start}_{end}"
for name in (
    f"metatrain_{tag}.fasta", f"blast_results_metatrain_{tag}.tsv",
    f"model_probs_metatrain_{tag}_partial.npy", f"model_probs_metatrain_{tag}_n_done.txt",
):
    (WORK_DIR / name).unlink(missing_ok=True)

compute_meta_train_slice(start, end)  # builds a real, correctly-shaped checkpoint

# Corrupt it with a wrong shape, as if an earlier run used a different n_labels/n_rows.
probs_ckpt = WORK_DIR / f"model_probs_metatrain_{tag}_partial.npy"
n_done_ckpt = WORK_DIR / f"model_probs_metatrain_{tag}_n_done.txt"
np.save(probs_ckpt, np.zeros((3, 5), dtype=np.float32))
n_done_ckpt.write_text("3")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    result = compute_meta_train_slice(start, end)
output = buf.getvalue()

assert "discarding stale checkpoint" in output, f"expected a stale-checkpoint warning, got:\n{output}"
n_rows = end - start
n_labels = len(result["label_names"])
assert result["esm_prob"].shape == (n_rows, n_labels), result["esm_prob"].shape
assert np.isfinite(result["esm_prob"]).all(), "non-finite value after recovering from a stale checkpoint"

print("[PASS] stale checkpoint check: mismatched-shape checkpoint was discarded and recomputed correctly")
