"""Score a checkpoint against the temporal PK/LK/NK validation set (Task 7's output),
breaking Fmax down by aspect x category. See
docs/superpowers/specs/2026-10-05-temporal-pk-eval-design.md.
"""
import json
from pathlib import Path

import numpy as np

from pk_eval_common import compute_ia_weighted_fmax

HF_REPO_ID = "kelmoir/cafa6-esm"
DEFAULT_REVISION = "main"
DATASET_FULL_PATH = Path("data/cafa-6-protein-function-prediction/dataset_full")
VALIDATION_PATH = Path("experiments/pk_eval/temporal_pk_validation")
IA_PATH = Path("data/cafa-6-protein-function-prediction/IA.tsv")


def densify_term_lists(encoded_lists: list, n_labels: int) -> np.ndarray:
  dense = np.zeros((len(encoded_lists), n_labels), dtype=bool)
  for row_idx, indices in enumerate(encoded_lists):
    for idx in indices:
      dense[row_idx, idx] = True
  return dense


def score_breakdown(probs, known, novel, category_columns, ia_weights, thresholds) -> dict:
  results = {}
  for aspect, categories in category_columns.items():
    categories = np.array(categories, dtype=object)
    results[aspect] = {}
    for category in set(categories) - {None}:
      row_mask = categories == category
      if not row_mask.any():
        continue
      fmax, best_t, pr, rc = compute_ia_weighted_fmax(
          probs[row_mask], known[row_mask], novel[row_mask], ia_weights, thresholds)
      results[aspect][category] = {
          "fmax": fmax, "best_threshold": best_t, "precision": pr, "recall": rc,
          "n_proteins": int(row_mask.sum()),
      }
  return results


def _self_test():
  # --- densify_term_lists ---
  dense = densify_term_lists([[0, 2], [1]], n_labels=3)
  assert dense.shape == (2, 3)
  assert dense.tolist() == [[1, 0, 1], [0, 1, 0]]

  # --- score_breakdown: 2 proteins, both category "PK" on aspect "F", using the same
  # hand-verified toy numbers as pk_eval_common's compute_ia_weighted_fmax test ---
  probs = np.array([[0.9, 0.8, 0.1], [0.6, 0.6, 0.4]])
  known = np.array([[1, 0, 0], [0, 0, 0]])
  novel = np.array([[0, 1, 0], [1, 0, 1]])
  ia_weights = np.array([1.0, 1.0, 1.0])
  category_columns = {"F": ["PK", "PK"]}  # both toy proteins tagged PK on aspect F
  result = score_breakdown(probs, known, novel, category_columns, ia_weights,
                           thresholds=np.array([0.5]))
  assert abs(result["F"]["PK"]["fmax"] - 0.75) < 1e-9, result

  print("[OK] score_temporal_pk_eval self-test passed")


if __name__ == "__main__":
  _self_test()
