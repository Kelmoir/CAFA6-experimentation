"""Checks whether item 16's reported full-label AUROC numbers (e.g. epoch 20's 0.6987 vs.
epoch 40's 0.6865, experiments/cafa6_next_steps_plan.md) are depressed by the same sigmoid-
saturation issue diagnosed for Fmax (--use-raw-logit, 2026-11-02 status update).

run_full_label_diagnostic (train_autoregressive_head.py) computes AUROC via sklearn's
roc_auc_score on probs = sigmoid(best_logit), the exact same saturating transform. roc_auc_score
is invariant to any monotonic transform of its score argument UNLESS that transform introduces
new ties -- and sigmoid at this model's large trained `scale` collapses many distinct large-
magnitude logits to the identical float 1.0/0.0, which is exactly a tie-introducing transform.
Ties only ever pull AUROC down (each tied pair contributes 0.5 instead of 1.0/0.0), so if ties
are common here, the reported AUROC is a pessimistic lower bound on the model's real ranking
quality -- and if different checkpoints saturate by different amounts, comparing their reported
AUROC numbers directly (as the epoch 20 vs. 40 "regression" call did) could be comparing
different amounts of this same artifact rather than real ranking quality.

Recomputes per-label AUROC two ways, on the SAME 1000-example validation subset
(dataset["validation"].select(range(1000)), matching run_full_label_diagnostic's own
diagnostic_ds convention exactly) for a given checkpoint:
  (a) on sigmoid(best_logit) -- replicates the existing reported numbers
  (b) on raw best_logit directly -- no saturation, no precision-collapsed ties

Usage:
  env/bin/python experiments/label_embedding/check_auroc_saturation.py --ar-checkpoint .../epoch_20.pt
  env/bin/python experiments/label_embedding/check_auroc_saturation.py --ar-checkpoint .../epoch_40.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "pk_eval" / "scripts"))

import training_run  # noqa: E402
import evaluate_cafa_score as ecs  # noqa: E402
from evaluate_cafa_fmax import load_ar_model, run_ar_inference_raw  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "cafa-6-protein-function-prediction"
DEFAULT_DATA_PATH = DATA_DIR / "dataset_full"
DEFAULT_LABEL_EMBEDDINGS_PATH = REPO_ROOT / "experiments" / "label_embedding" / "term_embeddings.pt"


def mean_auroc(scores: np.ndarray, labels: np.ndarray) -> dict:
    aurocs = []
    for j in range(labels.shape[1]):
        y = labels[:, j]
        if y.min() == y.max():
            continue  # not evaluable: single-class column, same skip run_full_label_diagnostic uses
        aurocs.append(roc_auc_score(y, scores[:, j]))
    aurocs = np.array(aurocs)
    return {"mean_auroc": float(aurocs.mean()), "median_auroc": float(np.median(aurocs)),
            "n_labels_evaluable": int(aurocs.size)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--dataset-repo-id", default="kelmoir/cafa6-full")
    ap.add_argument("--val-samples", type=int, default=1000,
                     help="Matches run_full_label_diagnostic's diagnostic_ds convention exactly "
                          "(dataset['validation'].select(range(N))) for an apples-to-apples check.")
    ap.add_argument("--ar-checkpoint", required=True)
    ap.add_argument("--ar-model-name", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--ar-decoder-hidden-dim", type=int, default=256)
    ap.add_argument("--ar-max-decode-steps", type=int, default=64)
    ap.add_argument("--ar-batch-size", type=int, default=8)
    ap.add_argument("--label-embeddings-path", default=str(DEFAULT_LABEL_EMBEDDINGS_PATH))
    ap.add_argument("--max-length", type=int, default=training_run.MAX_LENGTH)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device)

    dataset = training_run.load_training_data(data_path=Path(args.data_path), repo_id=args.dataset_repo_id)
    label_names = [str(n) for n in dataset["train"].features["labels"].feature.names]

    val_split = dataset["validation"].select(range(min(args.val_samples, len(dataset["validation"]))))
    sequences = list(val_split["sequence"])
    labels_matrix = ecs.extract_labels_matrix(val_split, len(label_names))
    print(f"[INFO] {len(sequences)} validation proteins, {len(label_names)} labels")

    model = load_ar_model(Path(args.ar_checkpoint), Path(args.label_embeddings_path),
                           args.ar_model_name, args.ar_decoder_hidden_dim, label_names, device)
    tokenizer = training_run.create_esm_tokenizer(args.ar_model_name)
    raw_logits = run_ar_inference_raw(model, tokenizer, sequences, device, args.ar_batch_size,
                                       args.max_length, args.ar_max_decode_steps)
    sigmoid_probs = 1.0 / (1.0 + np.exp(-raw_logits))

    n_saturated_hi = int((sigmoid_probs >= 1.0 - 1e-7).sum())
    n_saturated_lo = int((sigmoid_probs <= 1e-7).sum())
    print(f"[INFO] {n_saturated_hi} entries saturated to ~1.0, {n_saturated_lo} to ~0.0 "
          f"out of {sigmoid_probs.size} total")

    sig_result = mean_auroc(sigmoid_probs, labels_matrix)
    raw_result = mean_auroc(raw_logits, labels_matrix)
    print(f"[SIGMOID]    mean_auroc={sig_result['mean_auroc']:.4f} "
          f"median_auroc={sig_result['median_auroc']:.4f} n_evaluable={sig_result['n_labels_evaluable']}")
    print(f"[RAW LOGIT]  mean_auroc={raw_result['mean_auroc']:.4f} "
          f"median_auroc={raw_result['median_auroc']:.4f} n_evaluable={raw_result['n_labels_evaluable']}")
    print(f"[DELTA] raw - sigmoid mean_auroc = {raw_result['mean_auroc'] - sig_result['mean_auroc']:+.4f}")


if __name__ == "__main__":
    main()
