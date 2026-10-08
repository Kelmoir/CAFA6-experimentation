"""Standalone check for data_collate_function's known_indices/known_offsets batching
(Task 2 of docs/superpowers/plans/2026-10-08-known-annotation-conditioning.md). Run:
    env/bin/python experiments/scripts/check_known_input_collate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from training_run import create_esm_tokenizer, data_collate_function


def check_variable_length_bags():
    tokenizer = create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    preprocessed_batch = [
        {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]),
         "known_term_ids": [0, 2, 5]},
        {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]),
         "known_term_ids": []},
        {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]),
         "known_term_ids": [7]},
    ]
    collated = data_collate_function(preprocessed_batch, tokenizer=tokenizer)
    assert torch.equal(collated["known_indices"], torch.tensor([0, 2, 5, 7], dtype=torch.long))
    assert torch.equal(collated["known_offsets"], torch.tensor([0, 3, 3], dtype=torch.long))

    # EmbeddingBag must consume this exactly (verified directly: an empty bag sums to
    # the zero vector, not NaN or an error).
    eb = torch.nn.EmbeddingBag(10, 4, mode="sum")
    out = eb(collated["known_indices"], collated["known_offsets"])
    assert out.shape == (3, 4)
    assert torch.allclose(out[1], torch.zeros(4)), "example 1's empty bag must be the zero vector"
    print("[OK] variable-length known-term bags batch and pool correctly")


def check_all_empty_batch():
    tokenizer = create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    preprocessed_batch = [
        {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]), "known_term_ids": []},
        {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]), "known_term_ids": []},
    ]
    collated = data_collate_function(preprocessed_batch, tokenizer=tokenizer)
    assert collated["known_indices"].numel() == 0
    assert torch.equal(collated["known_offsets"], torch.tensor([0, 0], dtype=torch.long))
    eb = torch.nn.EmbeddingBag(10, 4, mode="sum")
    out = eb(collated["known_indices"], collated["known_offsets"])
    assert torch.allclose(out, torch.zeros(2, 4)), "an all-empty batch must not crash EmbeddingBag"
    print("[OK] an all-zero-known batch collates to an empty indices tensor without crashing")


def check_absent_when_no_known_term_ids():
    tokenizer = create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    preprocessed_batch = [
        {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1])},
    ]
    collated = data_collate_function(preprocessed_batch, tokenizer=tokenizer)
    assert "known_indices" not in collated and "known_offsets" not in collated
    print("[OK] known_indices/known_offsets stay absent when no sample carries known_term_ids")


if __name__ == "__main__":
    check_variable_length_bags()
    check_all_empty_batch()
    check_absent_when_no_known_term_ids()
    print("[OK] all checks passed")
