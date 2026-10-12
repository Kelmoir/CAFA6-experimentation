"""Standalone check that run_diagnostic's new hierarchy_edge_index/label_is_rare
parameters are accepted and threaded through correctly (Task 4 of
docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md). Uses a tiny real ESM
model (facebook/esm2_t6_8M_UR50D) on CPU with a fabricated in-memory dataset -- small
enough to run quickly without GPU, exercising the real forward pass rather than mocking
it. Run directly:
    env/bin/python experiments/scripts/check_run_diagnostic_hierarchy_fields.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import datasets
import numpy as np
import torch

import training_run

LABEL_NAMES = [f"GO:{i:07d}" for i in range(6)]
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


def _fabricate_val_dataset():
    return datasets.Dataset.from_dict({
        "sequence": ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "MSEQKLISEEDLNS", "MKVLAT"],
        "labels": [[0, 1, 2], [1, 3], [2, 4, 5]],
    })


def check_run_diagnostic_accepts_and_reports_hierarchy_fields():
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type="facebook/esm2_t6_8M_UR50D",
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    tokenizer = training_run.create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    val_ds = _fabricate_val_dataset().with_transform(
        transform=lambda ex: training_run.preprocess_batch(
            ex, tokenizer=tokenizer, unique_items=LABEL_NAMES, label2id=LABEL2ID))
    device = torch.device("cpu")
    edge_index = np.array([[0, 1], [2, 3]])  # fabricated, doesn't need to be "real" GO edges
    label_is_rare = np.array([True, True, False, False, True, False])

    result = training_run.run_diagnostic(
        model, tokenizer, val_ds, LABEL_NAMES, device,
        max_val_samples=10, batch_size=2,
        hierarchy_edge_index=edge_index, label_is_rare=label_is_rare,
    )
    assert "violation_rate" in result, result
    assert "mean_violation_magnitude" in result, result
    assert result["n_edges"] == 2, result
    assert "rare_n_labels_evaluable" in result, result
    print(f"[OK] run_diagnostic reports hierarchy fields "
          f"(violation_rate={result['violation_rate']:.4f}, "
          f"rare_n={result['rare_n_labels_evaluable']})")


def check_run_diagnostic_without_hierarchy_args_is_unchanged():
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type="facebook/esm2_t6_8M_UR50D",
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    tokenizer = training_run.create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    val_ds = _fabricate_val_dataset().with_transform(
        transform=lambda ex: training_run.preprocess_batch(
            ex, tokenizer=tokenizer, unique_items=LABEL_NAMES, label2id=LABEL2ID))
    device = torch.device("cpu")
    result = training_run.run_diagnostic(
        model, tokenizer, val_ds, LABEL_NAMES, device, max_val_samples=10, batch_size=2,
    )
    assert "violation_rate" not in result, result
    assert "rare_mean_auroc" not in result, result
    print("[OK] omitting the new kwargs leaves run_diagnostic's output unchanged")


if __name__ == "__main__":
    check_run_diagnostic_accepts_and_reports_hierarchy_fields()
    check_run_diagnostic_without_hierarchy_args_is_unchanged()
    print("[OK] all checks passed")
