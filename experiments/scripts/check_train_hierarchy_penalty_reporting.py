"""Standalone check for separating the hierarchy-penalty contribution out of the reported
train_loss (final-review finding on docs/superpowers/plans/2026-10-11-go-dag-hierarchy-
consistency.md: with --hierarchy-loss-weight > 0, train_loss silently became BCE+penalty
combined, making it misleading for anyone judging train/val curves -- a new
train_hierarchy_penalty field reports the penalty's own contribution separately, so
train_loss's historical meaning/comparability is at least auditable). Runs
training_run.py as a subprocess (tiny CPU configs) to check real printed/JSON output.
Run directly:
    env/bin/python experiments/scripts/check_train_hierarchy_penalty_reporting.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = "data/cafa-6-protein-function-prediction/dataset_full"


def _run(extra_args):
    return subprocess.run(
        [sys.executable, "training_run.py", "--data-path", DATA_PATH,
         "--max-train-samples", "20", "--max-val-samples", "10",
         "--epochs", "1", "--device", "cpu", *extra_args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )


def check_hierarchy_weight_on_reports_penalty_field_and_its_positive():
    result = _run(["--hierarchy-loss-weight", "0.1"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "train_hierarchy_penalty" in result.stdout, result.stdout
    summary_line = next(line for line in result.stdout.splitlines() if line.startswith("[SUMMARY]"))
    summary = json.loads(summary_line[len("[SUMMARY] "):])
    assert "train_hierarchy_penalty" in summary["best_metrics"], summary
    assert summary["best_metrics"]["train_hierarchy_penalty"] > 0, summary
    print("[OK] --hierarchy-loss-weight > 0 reports a positive train_hierarchy_penalty field")


def check_hierarchy_weight_off_adds_no_new_key():
    result = _run([])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "train_hierarchy_penalty" not in result.stdout, result.stdout
    summary_line = next(line for line in result.stdout.splitlines() if line.startswith("[SUMMARY]"))
    summary = json.loads(summary_line[len("[SUMMARY] "):])
    assert "train_hierarchy_penalty" not in summary["best_metrics"], summary
    print("[OK] default --hierarchy-loss-weight=0.0 adds no train_hierarchy_penalty key (byte-compat)")


if __name__ == "__main__":
    check_hierarchy_weight_on_reports_penalty_field_and_its_positive()
    check_hierarchy_weight_off_adds_no_new_key()
    print("[OK] all checks passed")
