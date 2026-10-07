"""Score a checkpoint against the temporal PK/LK/NK validation set (Task 7's output),
breaking Fmax down by aspect x category. See
docs/superpowers/specs/2026-10-05-temporal-pk-eval-design.md.
"""
import json
import os
import sys
from pathlib import Path

# Same sys.path fix as pk_eval_common.py/build_temporal_pk_eval.py (Task 1 ruling) -
# running this script directly puts experiments/pk_eval/scripts/ on sys.path, not the
# repo root, and this script needs `import training_run` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# CLAUDE.md's documented env gotcha: huggingface_hub/hf_xet version mismatch raises
# "cannot import name 'XetSession'" / an Xet-storage OSError unless this is set BEFORE
# huggingface_hub is first imported anywhere in the process - training_run.py sets this
# itself, but main() below imports huggingface_hub before `import training_run`, so it
# must also be set here, at module level, ahead of any import.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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


def run_inference(model, tokenizer, dataset, device, batch_size: int = 16) -> np.ndarray:
  import torch
  from functools import partial

  import training_run

  loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=False,
      collate_fn=partial(training_run.data_collate_function, tokenizer=tokenizer))
  was_training = model.training
  model.eval()
  all_probs = []
  with torch.inference_mode():
    for batch in loader:
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
      all_probs.append(torch.sigmoid(logits.float()).cpu())
  if was_training:
    model.train()
  return torch.cat(all_probs).numpy()


def main():
  import argparse

  import datasets as hf_datasets
  import pandas as pd
  import torch
  from huggingface_hub import hf_hub_download

  import obonet
  import training_run
  from pk_eval_common import build_term_to_aspect

  parser = argparse.ArgumentParser()
  parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
  parser.add_argument("--revision", default=DEFAULT_REVISION)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--th-step", type=float, default=0.01)
  args = parser.parse_args()

  dataset_full = hf_datasets.load_from_disk(str(DATASET_FULL_PATH))
  label_names = dataset_full["train"].features["labels"].feature.names
  label2id = {name: i for i, name in enumerate(label_names)}
  n_labels = len(label_names)
  id2label = {i: name for name, i in label2id.items()}

  ia_df = pd.read_csv(IA_PATH, sep="\t", header=None, names=["term", "weight"])
  ia_map = dict(zip(ia_df["term"], ia_df["weight"]))
  ia_weights = np.array([ia_map.get(name, 0.0) for name in label_names])

  # Same checkpoint-config-recovery pattern diagnostic_eval_stage2.py already
  # established: read trial_metrics.json rather than re-deriving model_name/use_lora by
  # hand, so this script works unmodified for a future LoRA checkpoint too.
  trial_metrics_path = hf_hub_download(
      repo_id=args.hf_repo_id, filename="trial_metrics.json", revision=args.revision)
  trial_config = json.loads(Path(trial_metrics_path).read_text())["config"]
  model_name = trial_config["model_name"]
  use_lora = trial_config.get("use_lora", False)

  device = torch.device(args.device)
  tokenizer = training_run.create_esm_tokenizer(model_name)
  if use_lora:
    model = training_run.create_esm(
        model_type=model_name, use_lora=True, lora_resume_repo_id=args.hf_repo_id,
        revision=args.revision, id2label=id2label, label2id=label2id,
    ).to(device)
  else:
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type=args.hf_repo_id, revision=args.revision,
        id2label=id2label, label2id=label2id,
    ).to(device)
  model.eval()
  print(f"[INFO] Loaded {args.hf_repo_id}@{args.revision} ({model_name}, use_lora={use_lora})")

  validation = hf_datasets.load_from_disk(str(VALIDATION_PATH))
  from functools import partial
  preprocess = partial(training_run.preprocess_batch, tokenizer=tokenizer,
                       unique_items=label_names, label2id=label2id)
  tokenized = validation.with_transform(transform=preprocess)

  probs = run_inference(model, tokenizer, tokenized, device, batch_size=args.batch_size)
  known = densify_term_lists(validation["known_labels"], n_labels)
  novel = densify_term_lists(validation["novel_labels"], n_labels)

  obonet_graph = obonet.read_obo(
      "data/cafa-6-protein-function-prediction/Train/go-basic.obo")
  term_to_aspect = build_term_to_aspect(obonet_graph)
  aspect_vector = np.array([term_to_aspect.get(name) for name in label_names])

  thresholds = np.arange(args.th_step, 1.0, args.th_step)
  is_unseen = np.array(list(validation["pool"])) == "unseen_protein"
  full_report = {}
  for aspect in ("C", "F", "P"):
    aspect_mask = aspect_vector == aspect
    raw_categories = list(validation[f"category_{aspect}"])
    # unseen_protein rows are stored with category "NK" in category_C/F/P (true - they
    # are NK-shaped: zero known in every aspect), but they are a distinct pool from
    # added_annotation's genuine "returning" NK rows (a protein the model trained on in
    # OTHER aspects, gaining a first annotation in this one) - conflating the two would
    # dilute exactly the distinction this harness exists to measure. Null out unseen
    # rows before the PK/LK/NK pass, then score them separately as their own "unseen"
    # bucket below.
    # write_outputs (Task 6) stores the string "none" for a no-category row (HF Datasets
    # can't hold Python None in a string column) - normalize it back to None here so
    # score_breakdown's `- {None}` filter actually excludes these rows, instead of
    # treating "none" as a real, scorable category (it isn't: these rows have zero
    # novel terms in this aspect by definition, so "scoring" them is undefined/0).
    pk_lk_nk_categories = {aspect: [None if (is_unseen[i] or c == "none") else c
                                    for i, c in enumerate(raw_categories)]}
    breakdown = score_breakdown(
        probs[:, aspect_mask], known[:, aspect_mask], novel[:, aspect_mask],
        pk_lk_nk_categories, ia_weights[aspect_mask], thresholds)

    unseen_only_categories = {aspect: ["unseen" if is_unseen[i] else None
                                       for i in range(len(raw_categories))]}
    unseen_breakdown = score_breakdown(
        probs[:, aspect_mask], known[:, aspect_mask], novel[:, aspect_mask],
        unseen_only_categories, ia_weights[aspect_mask], thresholds)
    full_report[aspect] = {**breakdown.get(aspect, {}), **unseen_breakdown.get(aspect, {})}

  print(f"[RESULT] {json.dumps(full_report, indent=2)}")
  return full_report


if __name__ == "__main__":
  import sys
  if "--self-test" in sys.argv:
    _self_test()
  else:
    main()
