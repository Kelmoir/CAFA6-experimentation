"""Extract pooled per-protein embeddings + classifier probabilities from a trained baseline
checkpoint, for `experiments/pu_learning_plan.md`'s nnPU + KM2 pipeline (pipeline step 1-2).

One forward pass per protein gives both:
  - a pooled embedding (mean-pooled last_hidden_state over real, non-padding tokens) -- the
    per-term KM2 class-prior estimator's input (experiments/scripts/estimate_class_priors_km2.py).
  - the classifier's predicted probability per label (sigmoid of the logits already computed by
    the same forward pass) -- used as the Elkan-Noto spy cross-check signal there. NOTE: these
    come from the SAME checkpoint the spy positives were trained on (no held-out-spy retraining
    in this v1), so they're optimistic/overfit for that purpose -- see the caveat in
    estimate_class_priors_km2.py and pu_learning_plan.md's "Key risks" section.

Output: three files under --output-dir (default experiments/pu_cache/), reused as a one-time
cost by every subsequent estimate_class_priors_km2.py run against the same checkpoint:
  - protein_embeddings.npy   (n_proteins x hidden_dim, float32)
  - protein_probs.npy        (n_proteins x n_labels, float32)
  - protein_meta.json        {"protein_ids": [...], "label_names": [...], "checkpoint_path": ...,
                               "data_path": ..., "splits": [...], "pooling": "mean_unmasked"}

Usage:
  # Smoke test
  env/bin/python experiments/scripts/extract_protein_embeddings.py --max-samples 64 \\
      --output-dir experiments/pu_cache/smoke

  # Full extraction (train + validation, ~54k proteins) -- long-running on CPU, run via nohup:
  nohup env/bin/python experiments/scripts/extract_protein_embeddings.py \\
      > experiments/logs/extract_protein_embeddings.out 2>&1 &
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, EsmTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", default="models/pretrain-reduced-full-protein-v1/best_checkpoint",
                         help="Local checkpoint dir (AutoModelForSequenceClassification.from_pretrained-compatible). "
                              "Must match --data-path's label space (668-term raw space by default).")
    parser.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset_raw")
    parser.add_argument("--splits", default="train,validation", help="Comma-separated dataset splits to extract")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap per split, for a quick smoke test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads override (CPU throttling)")
    parser.add_argument("--output-dir", default="experiments/pu_cache")
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading checkpoint from {args.checkpoint_path}")
    tokenizer = EsmTokenizer.from_pretrained(args.checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path).to(device)
    model.eval()
    label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    hidden_dim = model.config.hidden_size

    dataset = load_from_disk(args.data_path)
    splits = args.splits.split(",")

    protein_ids = []
    embeddings_chunks = []
    probs_chunks = []

    start = time.time()
    n_done = 0
    for split in splits:
        rows = dataset[split]
        if args.max_samples is not None:
            rows = rows.select(range(min(args.max_samples, len(rows))))
        n = len(rows)
        print(f"[INFO] Extracting split={split} n={n}")
        for batch_start in range(0, n, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n)
            batch = rows[batch_start:batch_end]
            sequences = batch["sequence"]
            proteins = batch["protein"]
            tokenized = tokenizer(sequences, max_length=args.max_length, truncation=True, padding=True, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            with torch.inference_mode():
                # One forward pass gives both: the backbone's last_hidden_state (mean-pooled
                # below into the KM2 embedding) and, via the model's own classifier head, the
                # per-label logits (turned into the Elkan-Noto spy probabilities) -- no need to
                # call the backbone and the full model separately.
                outputs = model.esm(input_ids=input_ids, attention_mask=attention_mask)
                pooled = mean_pool(outputs.last_hidden_state, attention_mask)
                logits = model.classifier(outputs.last_hidden_state)
                probs = torch.sigmoid(logits)

            protein_ids.extend(proteins)
            embeddings_chunks.append(pooled.cpu().numpy().astype(np.float32))
            probs_chunks.append(probs.cpu().numpy().astype(np.float32))
            n_done += (batch_end - batch_start)

            if (batch_start // args.batch_size) % 20 == 0:
                elapsed = time.time() - start
                rate = n_done / max(elapsed, 1e-9)
                print(f"[STEP] split={split} {batch_end}/{n} total_done={n_done} rate={rate:.1f} proteins/s elapsed={elapsed:.0f}s")

    embeddings = np.concatenate(embeddings_chunks, axis=0)
    probs = np.concatenate(probs_chunks, axis=0)
    assert embeddings.shape == (len(protein_ids), hidden_dim)
    assert probs.shape == (len(protein_ids), len(label_names))

    np.save(output_dir / "protein_embeddings.npy", embeddings)
    np.save(output_dir / "protein_probs.npy", probs)
    meta = {
        "protein_ids": protein_ids,
        "label_names": label_names,
        "checkpoint_path": str(args.checkpoint_path),
        "data_path": str(args.data_path),
        "splits": splits,
        "pooling": "mean_unmasked",
        "n_proteins": len(protein_ids),
        "hidden_dim": hidden_dim,
    }
    with open(output_dir / "protein_meta.json", "w") as f:
        json.dump(meta, f)

    print(f"[DONE] Extracted {len(protein_ids)} proteins in {time.time() - start:.0f}s -> {output_dir}")


if __name__ == "__main__":
    main()
