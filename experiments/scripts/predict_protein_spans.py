"""Run a trained span-localization checkpoint (see train_mf_span_model.py) on one protein and
print predicted vs. ground-truth MF-term spans side by side.

The protein's sequence and ground-truth spans are looked up in kelmoir/cafa6_mf_placed (local
dataset_mf_placed if present, else the Hub) -- this is a qualitative held-out-label check, not
just raw model output, so --protein only needs to be a valid EntryID from that dataset. By
default every ground-truth MF term on the protein is evaluated (excluding whole_protein-scope
entries, same exclusion train_mf_span_model.py's build_pair_examples applies -- that source's
span is always the full sequence by construction and carries no real positional signal); pass
--go-id to restrict to specific term(s), including ones NOT annotated on this protein (a
prediction-only negative-case check, ground truth column reads "-").

Example:
  python experiments/scripts/predict_protein_spans.py \\
      --checkpoint models/mf-span-esm2_t12_35M_UR50D-epochs10-bs8/best_checkpoint \\
      --protein A0A0B4J1F2
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import EsmTokenizer

from train_mf_span_model import (
    MAX_LENGTH, WHOLE_PROTEIN_SOURCE, LabelTextEncoder, load_dataset_dict, load_go_term_texts,
    load_span_model, token_level_metrics,
)


def find_protein_row(dataset_dict, protein_id):
  for split_name, split in dataset_dict.items():
    for row in split:
      if row["protein"] == protein_id:
        return split_name, row
  return None, None


def ground_truth_spans(row):
  """go_id -> sorted list of (start, end), excluding whole_protein-scope entries."""
  loc = row["mf_localizations"]
  spans = defaultdict(set)
  for go_id, start, end, source in zip(loc["go_id"], loc["start"], loc["end"], loc["source"]):
    if source == WHOLE_PROTEIN_SOURCE:
      continue
    spans[go_id].add((start, end))
  return {go_id: sorted(s) for go_id, s in spans.items()}


def spans_to_target(spans, token_len):
  """Same +1 <cls>-offset convention as train_mf_span_model.py's collate function."""
  target = torch.zeros(token_len)
  for start, end in spans:
    lo, hi = start + 1, min(end + 1, token_len)
    target[lo:hi] = 1.0
  return target


def predict_spans(logits, valid_mask, threshold):
  """Per-token sigmoid probs -> contiguous positive token runs -> residue (start, end) spans
  (0-indexed, end-exclusive, same convention the dataset stores). Token index = residue index +
  1 (<cls> at position 0), so a positive token run [tok_start, tok_end) maps directly to residue
  span [tok_start - 1, tok_end - 1)."""
  probs = torch.sigmoid(logits)
  positive = ((probs >= threshold) & valid_mask.bool()).tolist()
  spans = []
  start = None
  for i, is_pos in enumerate(positive):
    if is_pos and start is None:
      start = i
    elif not is_pos and start is not None:
      spans.append((start - 1, i - 1))
      start = None
  if start is not None:
    spans.append((start - 1, len(positive) - 1))
  return spans


def format_spans(spans):
  return ", ".join(f"{s}-{e}" for s, e in spans) if spans else "-"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True,
                   help="Local checkpoint dir (e.g. models/<run_name>/best_checkpoint) or a Hub "
                        "repo id (e.g. kelmoir/cafa6-mf-span)")
  ap.add_argument("--revision", default=None,
                   help="Hub branch/revision, if --checkpoint is a Hub repo id")
  ap.add_argument("--protein", required=True, help="EntryID accession to run predictions on")
  ap.add_argument("--go-id", nargs="*", default=None,
                   help="Restrict to these GO term(s) instead of every ground-truth term on the "
                        "protein; a term not annotated on the protein is a prediction-only check "
                        "(ground truth column reads '-')")
  ap.add_argument("--threshold", type=float, default=0.5)
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = ap.parse_args()

  print(f"[INFO] Loading checkpoint from {args.checkpoint}"
        f"{'@' + args.revision if args.revision else ''}...")
  model = load_span_model(args.checkpoint, device=args.device, revision=args.revision)
  model.eval()
  tokenizer = EsmTokenizer.from_pretrained(args.checkpoint, revision=args.revision)

  print("[INFO] Loading kelmoir/cafa6_mf_placed to look up "
        f"{args.protein}'s sequence + ground truth...")
  dataset_dict = load_dataset_dict()
  split_name, row = find_protein_row(dataset_dict, args.protein)
  if row is None:
    raise SystemExit(f"Protein {args.protein!r} not found in any split of cafa6_mf_placed.")
  print(f"[INFO] Found {args.protein} in the '{split_name}' split ({len(row['sequence'])} residues).")

  gt_spans = ground_truth_spans(row)
  go_ids = args.go_id if args.go_id else sorted(gt_spans)
  if not go_ids:
    raise SystemExit(
        f"{args.protein} has no positionally-grounded ground-truth MF terms "
        f"(placed_mf_terms={row['placed_mf_terms']!r} were all whole-protein-scope, or none "
        f"placed) -- pass --go-id explicitly to run a prediction-only check.")

  texts = load_go_term_texts(go_ids)
  encoder = LabelTextEncoder(device=args.device)

  enc = tokenizer(row["sequence"], truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
  input_ids = enc["input_ids"].to(args.device)
  attention_mask = enc["attention_mask"].to(args.device)
  valid_mask = attention_mask.clone().float()
  valid_mask[:, 0] = 0.0  # <cls>
  seq_len = int(attention_mask.sum(dim=1).item())
  valid_mask[:, seq_len - 1] = 0.0  # <eos>
  valid_mask = valid_mask.to(args.device)

  results = []
  with torch.no_grad():
    for go_id in go_ids:
      label_embedding = torch.from_numpy(encoder.encode([texts[go_id]])).float().to(args.device)
      logits = model(input_ids, attention_mask, label_embedding)[0]

      truth = gt_spans.get(go_id, [])
      target = spans_to_target(truth, input_ids.shape[1]).unsqueeze(0)
      metrics = token_level_metrics(logits.unsqueeze(0).cpu(), target, valid_mask.cpu(),
                                     threshold=args.threshold)
      predicted = predict_spans(logits.cpu(), valid_mask[0].cpu(), args.threshold)

      results.append({
          "go_id": go_id,
          "name": texts[go_id].split(":", 1)[0],
          "ground_truth": format_spans(truth),
          "predicted": format_spans(predicted),
          "precision": round(metrics["precision"], 3),
          "recall": round(metrics["recall"], 3),
          "f1": round(metrics["f1"], 3),
      })

  df = pd.DataFrame(results)
  print(f"\nProtein: {args.protein}  (split={split_name}, {len(row['sequence'])} residues, "
        f"threshold={args.threshold})\n")
  print(df.to_string(index=False))


if __name__ == "__main__":
  main()
