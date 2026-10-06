"""One-time cross-check of compute_ia_weighted_fmax against the real cafaeval 1.3.0
package (installed via `pip install cafaeval` - verified directly during planning: its
CLI is `python -m cafaeval <obo> <pred_dir> <gt_file> -ia <ia_file> -out_dir <dir>`, its
ground_truth.tsv is 2-column protein_id/term with no known-exclusion flag, so known-term
stripping happens here before cafaeval ever sees the prediction file).
"""
import os
import sys
from pathlib import Path

# Same two rulings as score_temporal_pk_eval.py (Task 9): sys.path fix for
# `import training_run` from the repo root, and HF_HUB_DISABLE_XET set before
# huggingface_hub's first import anywhere in the process (documented CLAUDE.md gotcha).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

from pk_eval_common import compute_ia_weighted_fmax


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


def main():
  import csv
  import subprocess

  import datasets as hf_datasets
  import pandas as pd

  import score_temporal_pk_eval as scorer

  dataset_full = hf_datasets.load_from_disk(str(scorer.DATASET_FULL_PATH))
  label_names = dataset_full["train"].features["labels"].feature.names
  n_labels = len(label_names)

  validation = hf_datasets.load_from_disk(str(scorer.VALIDATION_PATH))
  protein_ids = list(validation["protein_id"])
  known = scorer.densify_term_lists(validation["known_labels"], n_labels)

  # Model-loading below mirrors score_temporal_pk_eval.main()'s own checkpoint-loading
  # block verbatim (trial_metrics.json -> create_esm_tokenizer/create_esm) rather than
  # calling that main() directly, since this script only needs the resulting probs
  # matrix, not score_temporal_pk_eval's own aspect x category report.
  import json

  import torch
  from huggingface_hub import hf_hub_download

  import training_run

  trial_metrics_path = hf_hub_download(
      repo_id=scorer.HF_REPO_ID, filename="trial_metrics.json", revision=scorer.DEFAULT_REVISION)
  trial_config = json.loads(Path(trial_metrics_path).read_text())["config"]
  model_name = trial_config["model_name"]
  label2id = {name: i for i, name in enumerate(label_names)}
  id2label = {i: name for name, i in label2id.items()}
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = training_run.create_esm_tokenizer(model_name)
  model = training_run.create_esm(
      unfreeze_last_n=1, model_type=scorer.HF_REPO_ID, revision=scorer.DEFAULT_REVISION,
      id2label=id2label, label2id=label2id,
  ).to(device)
  model.eval()

  from functools import partial
  preprocess = partial(training_run.preprocess_batch, tokenizer=tokenizer,
                       unique_items=label_names, label2id=label2id)
  probs = scorer.run_inference(model, tokenizer, validation.with_transform(transform=preprocess), device)

  flat_dir = Path("experiments/pk_eval/temporal_pk_validation_flat")
  pred_dir = flat_dir / "predictions"
  pred_dir.mkdir(parents=True, exist_ok=True)
  write_prediction_tsv(probs, known, protein_ids, label_names, pred_dir / "model.tsv")

  cross_out_dir = Path("experiments/pk_eval/cafaeval_crosscheck")
  subprocess.run([
      "env/bin/python", "-m", "cafaeval",
      "data/cafa-6-protein-function-prediction/Train/go-basic.obo",
      str(pred_dir),
      str(flat_dir / "ground_truth.tsv"),
      "-ia", "data/cafa-6-protein-function-prediction/IA.tsv",
      "-out_dir", str(cross_out_dir),
  ], check=True)

  # NOTE on methodology (found via systematic-debugging, 2026-10-06): cafaeval computes
  # TWO separate metric sets - unweighted (evaluation_best_f.tsv's "f") and IA-weighted
  # (evaluation_best_f_w.tsv's "f_w", only populated when -ia is passed). The weighted
  # one is the right comparison for compute_ia_weighted_fmax, not "f".
  #
  # Even against f_w, comparing cafaeval's end-to-end CLI run (full ~26k-term
  # per-namespace ontology from go-basic.obo, with its own internal ancestor-max
  # propagation of predictions AND ground truth) against OUR end-to-end pipeline
  # (scoped to dataset_full's 27,191-term *trainable* vocabulary - a real, deliberate,
  # already-documented truncation the model structurally cannot predict outside of) is
  # an apples-to-oranges comparison of two different scopes, not a test of the formula -
  # confirmed directly: feeding cafaeval's OWN propagated/scoped matrices (via its
  # parser functions) into our own compute_ia_weighted_fmax reproduces its f_w number
  # within 0.003 for all three aspects. That is the actual test of the formula; running
  # the CLI above is kept only to produce the real package's human-readable output files
  # as a persisted artifact, not for the pass/fail assertion below.
  from cafaeval.parser import obo_parser, gt_parser, pred_parser

  ontologies = obo_parser(
      "data/cafa-6-protein-function-prediction/Train/go-basic.obo", ("is_a", "part_of"),
      "data/cafa-6-protein-function-prediction/IA.tsv", True)
  gt = gt_parser(str(flat_dir / "ground_truth.tsv"), ontologies)
  cafaeval_pred = pred_parser(str(pred_dir / "model.tsv"), ontologies, gt, "max", None)

  thresholds = np.arange(0.01, 1.0, 0.01)
  namespace_to_letter = {"biological_process": "P", "molecular_function": "F", "cellular_component": "C"}
  print(f"{'aspect':<8}{'our formula on cafaeval matrices':<36}")
  for ns, letter in namespace_to_letter.items():
    ont = ontologies[ns]
    toi_ia = ont.toi_ia
    gt_w = gt[ns].matrix[:, toi_ia]
    pred_w = cafaeval_pred[ns].matrix[:, toi_ia]
    ia_w = ont.ia[toi_ia]
    fmax, _, _, _ = compute_ia_weighted_fmax(
        pred_w, np.zeros_like(gt_w, dtype=bool), gt_w, ia_w, thresholds)
    cafaeval_f_w_path = cross_out_dir / "evaluation_best_f_w.tsv"
    f_w_rows = pd.read_csv(cafaeval_f_w_path, sep="\t")
    cafaeval_f_w = f_w_rows.loc[
        (f_w_rows["filename"] == "model.tsv") & (f_w_rows["ns"] == ns), "f_w"
    ].iloc[0]
    print(f"{letter:<8}{fmax:<16.4f} (cafaeval f_w: {cafaeval_f_w:.4f})")
    assert abs(fmax - cafaeval_f_w) < 0.02, (
        f"aspect {letter}: our formula on cafaeval's own matrices {fmax} vs "
        f"cafaeval's own f_w {cafaeval_f_w} - outside tolerance (this WOULD be a real "
        f"formula bug, since both sides use the identical scope/propagation here)")
  print("[OK] cafaeval cross-check passed within tolerance")


if __name__ == "__main__":
  import sys
  if "--self-test" in sys.argv:
    _self_test()
  else:
    main()
