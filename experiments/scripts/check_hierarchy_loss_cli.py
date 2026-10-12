"""Standalone check for --hierarchy-loss-weight's CLI wiring (Task 2 of
docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md): the default must be
a complete no-op, and a missing graph file must fail fast when the flag is actually
requested. Runs training_run.py as a subprocess (no GPU, tiny CPU smoke configs) rather
than importing main() directly, since main() has no return value to assert against --
this checks real process exit codes and stdout, as a user running the CLI would see.
Run directly:
    env/bin/python experiments/scripts/check_hierarchy_loss_cli.py
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


def check_missing_graph_file_fails_fast_when_weight_set():
    result = _run(["--hierarchy-loss-weight", "0.1",
                    "--go-dependency-graph-path", "/nonexistent/go_dependency_graph.json"])
    assert result.returncode != 0, result.stdout + result.stderr
    assert "go-dependency-graph-path" in (result.stdout + result.stderr), result.stderr
    print("[OK] --hierarchy-loss-weight > 0 with a missing graph file fails fast, doesn't train silently")


def check_default_weight_zero_builds_no_edge_index():
    result = _run([])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Built hierarchy edge index" not in result.stdout, result.stdout
    print("[OK] default --hierarchy-loss-weight=0.0 never builds the edge index (no-op)")


if __name__ == "__main__":
    check_missing_graph_file_fails_fast_when_weight_set()
    check_default_weight_zero_builds_no_edge_index()
    print("[OK] all checks passed")
