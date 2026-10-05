"""One-time cross-check of compute_ia_weighted_fmax against the real cafaeval 1.3.0
package (installed via `pip install cafaeval` - verified directly during planning: its
CLI is `python -m cafaeval <obo> <pred_dir> <gt_file> -ia <ia_file> -out_dir <dir>`, its
ground_truth.tsv is 2-column protein_id/term with no known-exclusion flag, so known-term
stripping happens here before cafaeval ever sees the prediction file).
"""
from pathlib import Path

import numpy as np


def write_prediction_tsv(probs, known, protein_ids, label_names, out_path, threshold_floor=0.01) -> None:
  lines = []
  for row_idx, protein_id in enumerate(protein_ids):
    for col_idx, term in enumerate(label_names):
      if known[row_idx, col_idx]:
        continue  # stripped, same as compute_ia_weighted_fmax's known-exclusion
      score = probs[row_idx, col_idx]
      if score >= threshold_floor:
        lines.append(f"{protein_id}\t{term}\t{score:.3f}")
  out_path.write_text("\n".join(lines) + "\n" if lines else "")


def _self_test():
  import tempfile

  probs = np.array([[0.9, 0.02], [0.6, 0.6]])
  known = np.array([[1, 0], [0, 0]], dtype=bool)
  protein_ids = ["T0001", "T0002"]
  label_names = ["GO:AAA", "GO:BBB"]
  with tempfile.TemporaryDirectory() as tmp:
    out_path = Path(tmp) / "pred.tsv"
    write_prediction_tsv(probs, known, protein_ids, label_names, out_path)
    lines = out_path.read_text().strip().split("\n")
    # T0001/GO:AAA has prob 0.9 but is known -> must be stripped, not written.
    assert not any(line.startswith("T0001\tGO:AAA") for line in lines), lines
    assert any(line.startswith("T0001\tGO:BBB") for line in lines), lines  # 0.02 >= floor
    assert any(line.startswith("T0002\tGO:AAA") for line in lines), lines
  print("[OK] crosscheck_cafaeval self-test passed")


if __name__ == "__main__":
  _self_test()
