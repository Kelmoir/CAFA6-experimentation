"""Standalone check for KnownAwareClassificationHead (Task 3 of
docs/superpowers/plans/2026-10-08-known-annotation-conditioning.md). Run:
    env/bin/python experiments/scripts/check_known_aware_head.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from training_run import KnownAwareClassificationHead


class _FakeConfig:
    hidden_size = 8
    hidden_dropout_prob = 0.0
    num_labels = 5
    initializer_range = 0.02


def check_output_shape_and_empty_bag_equivalence():
    torch.manual_seed(0)
    head = KnownAwareClassificationHead(_FakeConfig(), known_embed_dim=4)
    head.eval()

    batch, seq_len = 3, 6
    features = torch.randn(batch, seq_len, _FakeConfig.hidden_size)

    # example 0: empty known set; example 1: known={2, 4}; example 2: empty known set
    known_indices = torch.tensor([2, 4], dtype=torch.long)
    known_offsets = torch.tensor([0, 0, 2], dtype=torch.long)

    logits = head(features, known_indices=known_indices, known_offsets=known_offsets)
    assert logits.shape == (batch, _FakeConfig.num_labels), logits.shape

    # Manual cross-check for example 0: an empty EmbeddingBag bag must contribute
    # exactly the zero vector, verified independently against torch's own EmbeddingBag
    # semantics -- this checks the WIRING (forward()'s concatenation), not EmbeddingBag
    # itself.
    cls0 = features[0, 0, :]
    x0 = torch.cat([cls0, torch.zeros(4)])
    x0 = head.dense(x0)
    x0 = torch.tanh(x0)
    x0 = head.out_proj(x0)
    assert torch.allclose(logits[0], x0, atol=1e-6), (logits[0], x0)
    print("[OK] output shape correct, empty-bag example matches manual zero-vector computation")


def check_missing_known_input_raises():
    head = KnownAwareClassificationHead(_FakeConfig(), known_embed_dim=4)
    features = torch.randn(1, 3, _FakeConfig.hidden_size)
    try:
        head(features)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("[OK] missing known_indices/known_offsets raises ValueError")


if __name__ == "__main__":
    check_output_shape_and_empty_bag_equivalence()
    check_missing_known_input_raises()
    print("[OK] all checks passed")
