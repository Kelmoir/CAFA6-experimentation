"""Standalone check for preprocess_batch's known/novel masking (Task 1 of
docs/superpowers/plans/2026-10-08-known-annotation-conditioning.md). Run directly:
    env/bin/python experiments/scripts/check_known_input_masking.py
"""
import random
import sys
from pathlib import Path

# training_run.py lives at the repo root; this script lives two levels down
# (experiments/scripts/), so the repo root needs to go on sys.path explicitly -
# same ruling as experiments/pk_eval/scripts/*.py (Python puts a script's own
# directory on sys.path, not the repo root, when run directly).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from training_run import create_esm_tokenizer, preprocess_batch

UNIQUE_ITEMS = ["GO:A", "GO:B", "GO:C"]
LABEL2ID = {name: i for i, name in enumerate(UNIQUE_ITEMS)}


def _run(examples, **kwargs):
    tokenizer = create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    return preprocess_batch(
        examples, tokenizer=tokenizer, label2id=LABEL2ID, unique_items=UNIQUE_ITEMS,
        **kwargs,
    )


def check_forced_zero_known():
    examples = {"sequence": ["MKT", "MKTG"], "labels": [[0, 2], [1]]}
    out = _run(examples, use_known_input=True, known_mix_zero_frac=1.0)
    assert out["known_term_ids"] == [[], []], out["known_term_ids"]
    for mask in out["loss_mask"]:
        assert torch.all(mask == 1.0), mask
    print("[OK] forced known_mix_zero_frac=1.0 -> every example has zero known terms")


def check_partial_reveal_properties():
    # known_mix_zero_frac=0.0 -> never a forced-zero draw; known_reveal_max=1.0 ->
    # reveal fraction can be anywhere in [0, 1). Assert structural properties rather
    # than exact sampled ids, since the output is genuinely randomized by design.
    examples = {"sequence": ["MKT"], "labels": [[0, 1, 2]]}
    out = _run(examples, use_known_input=True, known_mix_zero_frac=0.0,
               known_reveal_max=1.0, rng=random.Random(42))
    known_ids = out["known_term_ids"][0]
    positive_ids = {0, 1, 2}
    assert set(known_ids).issubset(positive_ids), known_ids
    assert len(known_ids) == len(set(known_ids)), "known ids must not repeat"
    mask = out["loss_mask"][0]
    for idx in range(len(UNIQUE_ITEMS)):
        expected = 0.0 if idx in known_ids else 1.0
        assert mask[idx].item() == expected, (idx, mask)
    print(f"[OK] partial reveal: known_ids={known_ids}, loss_mask matches exactly")


def check_few_positives_edge_case():
    # Exactly 1 positive term + known_reveal_max < 1.0 -> floor(reveal_frac * 1) == 0
    # for every possible reveal_frac in [0, known_reveal_max) < 1.0 -> always degrades
    # to zero known terms, deterministically, regardless of rng state.
    examples = {"sequence": ["MKT"], "labels": [[1]]}
    for seed in range(5):
        out = _run(examples, use_known_input=True, known_mix_zero_frac=0.0,
                   known_reveal_max=0.5, rng=random.Random(seed))
        assert out["known_term_ids"] == [[]], out["known_term_ids"]
    print("[OK] single-positive protein always degrades to zero known terms, no crash")


def check_known_terms_from_dataset():
    # No "labels" column at all -- matches experiments/pk_eval's validation dataset
    # shape exactly (it has known_labels/novel_labels, never a generic labels column).
    examples = {"sequence": ["MKT", "MKTG"], "known_labels": [[0, 2], []]}
    out = _run(examples, use_known_input=True, known_terms_from_dataset=True)
    assert out["known_term_ids"] == [[0, 2], []], out["known_term_ids"]
    assert torch.equal(out["loss_mask"][0], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(out["loss_mask"][1], torch.tensor([1.0, 1.0, 1.0]))
    print("[OK] known_terms_from_dataset reads known_labels directly, no randomization")


def check_missing_labels_raises():
    examples = {"sequence": ["MKT"]}  # no "labels", no "known_labels"
    try:
        _run(examples, use_known_input=True, known_mix_zero_frac=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("[OK] use_known_input without a labels column raises ValueError")


if __name__ == "__main__":
    check_forced_zero_known()
    check_partial_reveal_properties()
    check_few_positives_edge_case()
    check_known_terms_from_dataset()
    check_missing_labels_raises()
    print("[OK] all checks passed")
