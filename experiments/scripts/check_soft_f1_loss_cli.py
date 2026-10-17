"""Standalone check for --soft-f1-loss-weight's CLI wiring (Tier 2 item 4): the default
must be a complete no-op, a missing IA file must fail fast when the flag is actually
requested, and a real tiny CPU run with the flag on must train without error and report
train_soft_f1_loss. Runs training_run.py as a subprocess (tiny CPU smoke configs), mirroring
check_hierarchy_loss_cli.py.

Run directly:
    env/bin/python experiments/scripts/check_soft_f1_loss_cli.py
"""
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


def check_missing_ia_file_fails_fast_when_weight_set():
    result = _run(["--soft-f1-loss-weight", "0.5", "--ia-file", "/nonexistent/IA.tsv"])
    assert result.returncode != 0, result.stdout + result.stderr
    assert "ia-file" in (result.stdout + result.stderr), result.stderr
    print("[OK] --soft-f1-loss-weight > 0 with a missing IA file fails fast, doesn't train silently")


def check_default_weight_zero_never_loads_ia_weights():
    result = _run([])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Loaded IA weights" not in result.stdout, result.stdout
    assert "train_soft_f1_loss" not in result.stdout, result.stdout
    print("[OK] default --soft-f1-loss-weight=0.0 never loads IA weights (no-op)")


def check_real_tiny_run_with_weight_set_trains_and_reports_soft_f1():
    result = _run(["--soft-f1-loss-weight", "0.5"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Loaded IA weights" in result.stdout, result.stdout
    assert "train_soft_f1_loss" in result.stdout, result.stdout
    print("[OK] a real tiny run with --soft-f1-loss-weight set trains cleanly and reports train_soft_f1_loss")


if __name__ == "__main__":
    check_missing_ia_file_fails_fast_when_weight_set()
    check_default_weight_zero_never_loads_ia_weights()
    check_real_tiny_run_with_weight_set_trains_and_reports_soft_f1()
    print("[OK] all checks passed")
