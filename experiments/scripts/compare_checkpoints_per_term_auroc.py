"""Per-term AUROC comparison between two checkpoints -- built specifically to check whether --nnpu
with the new eligibility-floored, held-out-spy-retrain priors (class_priors_spy_heldout.json +
--class-priors-min-source-positives, see project_pu_learning_nnpu_km2.md v1.3) actually helps on the
handful of terms it applies to.

Why not just read training_run.py --run-diagnostic's own summary: with the floor set (default 2000
source positives), only ~11/437 terms get nnPU treatment -- the other ~650+ labels get identical
plain BCE in both the baseline and nnPU runs. run_diagnostic()'s aggregate mean/median/pooled AUROC
averages over all ~667 evaluable labels, so any real effect on the 11 targeted terms would be
diluted into noise. This script scores exactly the same validation rows with both checkpoints and
reports AUROC per targeted term directly.

Usage (after both checkpoints exist locally):
  env/bin/python experiments/scripts/compare_checkpoints_per_term_auroc.py \\
      --checkpoint-a models/pu-baseline-trial001/best_checkpoint \\
      --checkpoint-b models/pu-nnpu-trial002-floor2000/best_checkpoint \\
      --terms-from-cache experiments/pu_cache/class_priors_spy_heldout.json \\
      --min-source-positives 2000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, EsmTokenizer


def score_checkpoint(checkpoint_path: str, data_path: str, max_val_samples: int, batch_size: int,
                      device: torch.device, max_length: int):
    tokenizer = EsmTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)
    model.eval()
    label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    dataset = load_from_disk(data_path)
    val = dataset["validation"]
    if max_val_samples is not None and len(val) > max_val_samples:
        val = val.select(range(max_val_samples))
    n = len(val)

    hard_labels = np.zeros((n, len(label_names)), dtype=np.float32)
    probs_chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = val[start:end]
        tokenized = tokenizer(batch["sequence"], max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits)
        probs_chunks.append(probs.cpu().numpy().astype(np.float32))
        for i, labs in enumerate(batch["labels"]):
            for t in labs:
                hard_labels[start + i, int(t)] = 1.0

    return np.concatenate(probs_chunks, axis=0), hard_labels, label_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-a", required=True, help="Baseline (plain BCE) checkpoint")
    parser.add_argument("--checkpoint-b", required=True, help="nnPU checkpoint")
    parser.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset_raw")
    parser.add_argument("--terms", default=None, help="Comma-separated GO term ids to compare; overrides --terms-from-cache")
    parser.add_argument("--terms-from-cache", default=None,
                         help="Read target terms from a class-priors cache's 'n_positives' field instead of an explicit list")
    parser.add_argument("--min-source-positives", type=int, default=2000,
                         help="Only used with --terms-from-cache: matches --class-priors-min-source-positives")
    parser.add_argument("--max-val-samples", type=int, default=4000, help="Must match the trial's --max-val-samples for a fair comparison")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.terms:
        target_terms = args.terms.split(",")
    else:
        if not args.terms_from_cache:
            raise ValueError("Provide either --terms or --terms-from-cache")
        cache = json.loads(Path(args.terms_from_cache).read_text())["priors"]
        target_terms = [t for t, v in cache.items() if v.get("n_positives", 0) >= args.min_source_positives]
    print(f"[INFO] Comparing {len(target_terms)} terms: {target_terms}")

    print(f"[INFO] Scoring checkpoint A: {args.checkpoint_a}")
    probs_a, labels_a, label_names = score_checkpoint(args.checkpoint_a, args.data_path, args.max_val_samples,
                                                        args.batch_size, device, args.max_length)
    print(f"[INFO] Scoring checkpoint B: {args.checkpoint_b}")
    probs_b, labels_b, label_names_b = score_checkpoint(args.checkpoint_b, args.data_path, args.max_val_samples,
                                                          args.batch_size, device, args.max_length)
    assert label_names == label_names_b, "label spaces differ between checkpoints -- not a fair comparison"
    assert np.array_equal(labels_a, labels_b), "validation subset/labels differ between checkpoints -- not a fair comparison"

    label2id = {name: i for i, name in enumerate(label_names)}
    print(f"{'term':14s} {'n_pos(val)':>10s} {'AUROC_a(base)':>14s} {'AUROC_b(nnpu)':>14s} {'delta':>8s}")
    deltas = []
    for term in target_terms:
        idx = label2id[term]
        y = labels_a[:, idx]
        if y.sum() == 0 or y.sum() == len(y):
            print(f"{term:14s} {'skipped: single class in this validation subset'}")
            continue
        auc_a = roc_auc_score(y, probs_a[:, idx])
        auc_b = roc_auc_score(y, probs_b[:, idx])
        deltas.append(auc_b - auc_a)
        print(f"{term:14s} {int(y.sum()):10d} {auc_a:14.4f} {auc_b:14.4f} {auc_b - auc_a:+8.4f}")

    deltas = np.array(deltas)
    if len(deltas) > 0:
        print(f"[SUMMARY] mean delta (nnpu - baseline) = {deltas.mean():+.4f}, "
              f"{int((deltas > 0).sum())}/{len(deltas)} terms improved, "
              f"{int((deltas < 0).sum())}/{len(deltas)} terms worsened")
    else:
        print("[SUMMARY] no terms had both classes present in this validation subset")


if __name__ == "__main__":
    main()
