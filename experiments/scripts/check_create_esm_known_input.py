"""Standalone check for create_esm's --use-known-input wiring (Task 4 of
docs/superpowers/plans/2026-10-08-known-annotation-conditioning.md). Loads a real tiny
ESM backbone (facebook/esm2_t6_8M_UR50D, ~8M params) -- network access required once
(cached afterward), same as this repo's own CPU smoke-test convention. Run:
    env/bin/python experiments/scripts/check_create_esm_known_input.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training_run import KnownConditionedEsmForSequenceClassification, create_esm

ID2LABEL = {0: "GO:A", 1: "GO:B", 2: "GO:C"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
TINY_MODEL = "facebook/esm2_t6_8M_UR50D"


def check_use_known_input_selects_the_right_class():
    model = create_esm(
        unfreeze_last_n=0, model_type=TINY_MODEL, id2label=ID2LABEL, label2id=LABEL2ID,
        use_known_input=True, known_embed_dim=8,
    )
    assert isinstance(model, KnownConditionedEsmForSequenceClassification), type(model)
    assert model.classifier.known_embed_dim == 8
    print("[OK] use_known_input=True selects KnownConditionedEsmForSequenceClassification")


def check_use_known_input_plus_use_lora_raises():
    try:
        create_esm(
            model_type=TINY_MODEL, id2label=ID2LABEL, label2id=LABEL2ID,
            use_known_input=True, use_lora=True,
        )
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
    print("[OK] use_known_input + use_lora raises NotImplementedError")


def check_use_known_input_plus_use_aspect_input_raises():
    try:
        create_esm(
            model_type=TINY_MODEL, id2label=ID2LABEL, label2id=LABEL2ID,
            use_known_input=True, use_aspect_input=True,
        )
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
    print("[OK] use_known_input + use_aspect_input raises NotImplementedError")


if __name__ == "__main__":
    check_use_known_input_selects_the_right_class()
    check_use_known_input_plus_use_lora_raises()
    check_use_known_input_plus_use_aspect_input_raises()
    print("[OK] all checks passed")
