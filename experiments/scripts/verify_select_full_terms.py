"""Standalone verification for generate_dataset_full.select_full_terms.

This repo has no pytest/test framework (see CLAUDE.md's "no test suite, linter, or CI"
note) -- this follows the project's own established convention (generate_dataset_raw.py's
select_raw_terms, verified directly against a fixture/real data and asserted with plain
`assert`) instead of introducing a new test-framework dependency.
"""
import sys
from pathlib import Path

# Add the project root to sys.path so we can import generate_dataset_full
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate_dataset_full import select_full_terms


def test_no_threshold_and_full_union():
  propagated_terms = {
      "P1": {"GO:0000001", "GO:0000002"},
      "P2": {"GO:0000002"},
      "P3": {"GO:0000003"},  # singleton term -- must still be included, no threshold
  }
  depths = {"GO:0000001": 1, "GO:0000002": 2}  # GO:0000003 deliberately missing

  result = select_full_terms(propagated_terms, depths)

  assert set(result.keys()) == {"GO:0000001", "GO:0000002", "GO:0000003"}, result.keys()
  assert result["GO:0000001"] == {"depth": 1, "count": 1}, result["GO:0000001"]
  assert result["GO:0000002"] == {"depth": 2, "count": 2}, result["GO:0000002"]
  assert result["GO:0000003"] == {"depth": None, "count": 1}, (
      "a term missing from the depths dict (graph-external/obsolete) must still be "
      f"included, with depth=None, not dropped: {result.get('GO:0000003')}")
  print("[OK] test_no_threshold_and_full_union")


if __name__ == "__main__":
  test_no_threshold_and_full_union()
  print("[OK] all verifications passed")
