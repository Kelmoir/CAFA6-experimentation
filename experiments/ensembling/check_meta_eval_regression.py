"""Standalone correctness check (no pytest in this repo -- see CLAUDE.md): confirms
load_cached_meta_eval() reproduces the exact numbers already logged in
experiments/cafa6_next_steps_plan.md for the real-BLAST probe, before any new BLAST/
inference compute is spent on the meta-train slice. main() now runs this same
assertion itself before computing meta-train (final-review minor finding, 2026-10-02) --
this script is the standalone entry point for the identical check.

Run via: env/bin/python experiments/ensembling/check_meta_eval_regression.py
"""
from stacking_pipeline import assert_meta_eval_matches_logged, load_cached_meta_eval

assert_meta_eval_matches_logged(load_cached_meta_eval())
print("[PASS] meta-eval regression check: n_labels_evaluable, model/blast mean AUROC all match logged values")
