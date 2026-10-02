"""Standalone check (no pytest in this repo): unit-tests _blastp_atomic() with a fake
failing command, fast and deterministic -- the TDD proof for the final-review fix to
compute_meta_train_slice()'s BLAST-output caching. The original code wrote blastp's
output directly to its final cache path; a `timeout`-induced kill mid-run (the plan's
own documented retry pattern) could leave a partial file there, which the next
invocation's `blast_out.exists()` check would then silently treat as a complete cache
hit. Run via:
    env/bin/python experiments/ensembling/check_blastp_atomic.py
"""
import subprocess
from pathlib import Path

from stacking_pipeline import WORK_DIR, _blastp_atomic

out_path = WORK_DIR / "check_blastp_atomic_fake_out.tsv"
tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
for p in (out_path, tmp_path):
    if p.exists():
        p.unlink()


def fake_failing_cmd(tmp):
    # Simulates a killed/failing blastp: writes partial output, then exits nonzero.
    return ["bash", "-c", f'echo partial_line > "{tmp}"; exit 1']


try:
    _blastp_atomic(fake_failing_cmd, out_path)
    raise SystemExit("[FAIL] expected CalledProcessError to propagate from the failing command")
except subprocess.CalledProcessError:
    pass

assert not out_path.exists(), (
    "a failed/killed run must never leave a file at the final cache path -- "
    "otherwise the next invocation would silently treat a partial file as a cache hit"
)
print("[PASS] blastp_atomic check: a failed run leaves no file at the final cache path")

if tmp_path.exists():
    tmp_path.unlink()
