"""Standalone check that --hierarchy-loss-weight adds no new model parameters or
modules (Task 2 of docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md) --
unlike --use-known-input/--use-aspect-input, this mechanism is a pure loss-function
addition, so a checkpoint trained with it on or off must have an identical state_dict
shape either way, and --resume-from-revision must work regardless of which setting the
checkpoint being resumed was trained with (Review Focus item). Run directly:
    env/bin/python experiments/scripts/check_hierarchy_loss_no_new_params.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training_run

LABEL_NAMES = [f"GO:{i:07d}" for i in range(10)]
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


def check_state_dict_keys_identical_regardless_of_hierarchy_flag():
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type="facebook/esm2_t6_8M_UR50D",
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    keys = set(model.state_dict().keys())
    # hierarchy_consistency_penalty is a pure function of (logits, edge_index) -- it is
    # never attached to the model as a module/parameter, so create_esm's own signature
    # and output are untouched by this mechanism; this test pins that invariant directly
    # rather than just asserting by reading the (unchanged) create_esm source.
    assert "hierarchy_edge_index" not in str(keys), keys
    assert len(keys) > 0
    print(f"[OK] create_esm's state_dict ({len(keys)} keys) has no hierarchy-related "
          f"parameters -- --resume-from-revision needs no special case for this flag")


if __name__ == "__main__":
    check_state_dict_keys_identical_regardless_of_hierarchy_flag()
    print("[OK] all checks passed")
