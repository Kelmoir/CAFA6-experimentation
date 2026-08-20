"""Train a term-conditioned span-localization model on kelmoir/cafa6_mf_placed: given a protein
sequence and one of its known molecular_function GO terms, predict which residues are the
evidence for that term.

Design (see the 2026-08-20 conversation for the full reasoning, backed by concrete stats pulled
from the dataset before committing to any of this):

  - Training unit is a (protein, GO-term) pair, not a row-per-localization-event. Built by
    grouping cafa6_mf_placed's mf_localizations by (protein, go_id), unioning every span that
    term has on that protein. whole_protein-scope entries (source=domain_whole_protein_fallback)
    are EXCLUDED from this grouping: their span is always the full sequence by construction (fires
    only when a protein has domain cross-references but no curated DOMAIN feature to anchor to at
    all), so it carries no real positional signal -- that "this term is real, no location" fact is
    already exactly what training_run.py's multi-label classifier predicts.

  - Label conditioning is a semantic text embedding (name + definition, via a small pretrained
    sentence encoder), not a learned per-GO-id embedding table. The label vocabulary here is
    4,950 distinct GO ids with a brutal long tail (61.8% have <=5 examples, 21.7% exactly 1) --
    a lookup-by-ID table would leave most rows untrained and has no way to handle a term outside
    the fixed vocabulary at inference. A semantic encoder generalizes via shared meaning instead.

  - Output is a per-token binary logit (sigmoid), not a SQuAD-style single (start, end) softmax
    pair. Verified directly: 49.1% of (protein, term) pairs have MORE THAN ONE distinct span (up
    to 10+ in some cases) -- a single-span extractive-QA head structurally cannot represent that.
    A contiguous run of positive-predicted tokens *is* a span; multiple disjoint runs *are*
    multiple spans; no fixed-cardinality assumption anywhere.

  - Training includes sampled negative pairs (a GO term NOT annotated on that protein, paired
    with an all-zero token target) so the model doesn't degrade to "always highlight something" --
    it has to learn what "this term doesn't apply here" looks like too, not just localize terms
    it's handed that are guaranteed correct.

This script builds the pair-level dataset, precomputes (and caches) label text embeddings for
every distinct GO id encountered, defines the FiLM-conditioned ESM-2 span model, and runs a plain
PyTorch training loop (same manual-loop style as training_run.py, no Trainer abstraction). No
Hub-push/trackio wiring yet -- this is the "get a first working model" step; that infra can be
ported over from training_run.py once there's a config worth promoting.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import obonet
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer

DATA_PATH = Path("./data/cafa-6-protein-function-prediction")
LOCAL_DATASET_PATH = DATA_PATH / "dataset_mf_placed"
HUB_DATASET_ID = "kelmoir/cafa6_mf_placed"
OBO_PATH = DATA_PATH / "Train" / "go-basic.obo"
GO_TEXT_CACHE_PATH = Path("experiments/positional_mf_go_term_texts.json")

WHOLE_PROTEIN_SOURCE = "domain_whole_protein_fallback"
LABEL_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
MAX_LENGTH = 1024
SEED = 42

DEF_PATTERN = re.compile(r'^"(.*)"')


# --------------------------------------------------------------------------------------------
# GO term text (name + definition), cached to disk -- obonet parses the full ~31MB go-basic.obo,
# not worth re-running per invocation once the ~5k terms we actually need are known.
# --------------------------------------------------------------------------------------------

def load_go_term_texts(go_ids):
  if GO_TEXT_CACHE_PATH.exists():
    cached = json.loads(GO_TEXT_CACHE_PATH.read_text())
    if set(go_ids) <= set(cached):
      return {go_id: cached[go_id] for go_id in go_ids}

  print(f"[INFO] Parsing {OBO_PATH} for GO term text ({len(go_ids)} ids needed)...")
  graph = obonet.read_obo(OBO_PATH)
  texts = {}
  for go_id in go_ids:
    node = graph.nodes.get(go_id, {})
    name = node.get("name", "")
    raw_def = node.get("def", "")
    m = DEF_PATTERN.match(raw_def) if raw_def else None
    definition = m.group(1) if m else ""
    texts[go_id] = f"{name}: {definition}" if definition else name

  GO_TEXT_CACHE_PATH.write_text(json.dumps(texts, indent=1))
  print(f"[INFO] Wrote GO term text cache -> {GO_TEXT_CACHE_PATH}")
  return texts


# --------------------------------------------------------------------------------------------
# Label text encoder -- plain transformers AutoModel + mean pooling (this is exactly what
# sentence-transformers does internally for this checkpoint; loading it via plain `transformers`
# avoids adding a new dependency for one pooling operation). Run once per distinct GO id, frozen,
# results cached as a plain tensor lookup -- there's no reason to re-embed static label text on
# every training step.
# --------------------------------------------------------------------------------------------

class LabelTextEncoder:
  def __init__(self, model_name=LABEL_TEXT_MODEL, device="cpu"):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name).to(device)
    self.model.eval()
    self.device = device
    self.hidden_size = self.model.config.hidden_size

  @torch.no_grad()
  def encode(self, texts, batch_size=64):
    """texts: list[str] -> np.ndarray [len(texts), hidden_size], mean-pooled + L2-normalized."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
      batch = texts[i:i + batch_size]
      enc = self.tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
      enc = {k: v.to(self.device) for k, v in enc.items()}
      out = self.model(**enc).last_hidden_state  # [B, L, H]
      mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
      pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # [B, H]
      pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
      all_embeddings.append(pooled.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def build_label_embeddings(go_ids, device="cpu"):
  go_ids = sorted(set(go_ids))
  texts = load_go_term_texts(go_ids)
  encoder = LabelTextEncoder(device=device)
  print(f"[INFO] Encoding {len(go_ids)} distinct GO term texts with {LABEL_TEXT_MODEL}...")
  embeddings = encoder.encode([texts[g] for g in go_ids])
  return {go_id: embeddings[i] for i, go_id in enumerate(go_ids)}, encoder.hidden_size


# --------------------------------------------------------------------------------------------
# Pair-level example construction: group mf_localizations by (protein, go_id), excluding
# whole_protein-scope entries, then add sampled negatives per protein.
# --------------------------------------------------------------------------------------------

def build_pair_examples(hf_dataset, n_negatives_per_protein, seed=SEED, max_length=MAX_LENGTH):
  """Returns a list of examples: {protein, sequence, go_id, spans: [(start,end),...], is_positive}.
  spans is empty for negatives. Sequences longer than max_length are skipped (mf_placed was
  already built with this cap, but kept here defensively in case a differently-built dataset is
  passed in)."""
  rng = random.Random(seed)

  rows = []
  for ex in hf_dataset:
    if len(ex["sequence"]) > max_length:
      continue
    rows.append(ex)

  # Global pool for negative sampling: every distinct go_id placed anywhere in this split.
  vocab = sorted({go_id for ex in rows for go_id in ex["mf_localizations"]["go_id"]})

  examples = []
  for ex in rows:
    protein, sequence = ex["protein"], ex["sequence"]
    loc = ex["mf_localizations"]
    per_term_spans = defaultdict(set)
    for go_id, start, end, source in zip(loc["go_id"], loc["start"], loc["end"], loc["source"]):
      if source == WHOLE_PROTEIN_SOURCE:
        continue
      per_term_spans[go_id].add((start, end))

    for go_id, spans in per_term_spans.items():
      examples.append({
          "protein": protein, "sequence": sequence, "go_id": go_id,
          "spans": sorted(spans), "is_positive": True,
      })

    if not per_term_spans:
      continue  # nothing positive to contribute; also skip negatives for this protein rather
      # than sampling them against an empty positive set (would need placed_mf_terms instead --
      # simpler to just skip, these proteins' only placements were whole_protein-scope anyway).

    placed_here = set(per_term_spans)
    candidates = [g for g in vocab if g not in placed_here]
    if candidates:
      n = min(n_negatives_per_protein, len(candidates))
      for go_id in rng.sample(candidates, n):
        examples.append({
            "protein": protein, "sequence": sequence, "go_id": go_id,
            "spans": [], "is_positive": False,
        })

  rng.shuffle(examples)
  return examples


# --------------------------------------------------------------------------------------------
# torch Dataset + collate: tokenize sequences and build per-token binary targets at batch time.
# --------------------------------------------------------------------------------------------

class SpanPairDataset(Dataset):
  def __init__(self, examples, label_embeddings):
    self.examples = examples
    self.label_embeddings = label_embeddings

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx):
    return self.examples[idx]


def make_collate_fn(tokenizer, label_embeddings, max_length=MAX_LENGTH):
  def collate(batch):
    sequences = [ex["sequence"] for ex in batch]
    enc = tokenizer(sequences, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    batch_size, padded_len = input_ids.shape

    # Token index = residue index + 1 (index 0 is <cls>; verified directly against EsmTokenizer:
    # tokenizing "MKV" gives [<cls>, M, K, V, <eos>]).
    targets = torch.zeros(batch_size, padded_len, dtype=torch.float32)
    valid_mask = attention_mask.clone().float()
    valid_mask[:, 0] = 0.0  # <cls> is never a real residue
    seq_lens = attention_mask.sum(dim=1)
    valid_mask[torch.arange(batch_size), seq_lens - 1] = 0.0  # <eos>

    for i, ex in enumerate(batch):
      for start, end in ex["spans"]:
        lo, hi = start + 1, min(end + 1, padded_len)  # +1 for <cls> offset
        targets[i, lo:hi] = 1.0

    label_emb = np.stack([label_embeddings[ex["go_id"]] for ex in batch])
    label_emb = torch.from_numpy(label_emb).float()

    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "targets": targets, "valid_mask": valid_mask, "label_embedding": label_emb,
        "go_id": [ex["go_id"] for ex in batch], "protein": [ex["protein"] for ex in batch],
    }
  return collate


# --------------------------------------------------------------------------------------------
# Model: ESM-2 backbone (same freeze-all-but-last-N convention as training_run.py's create_esm)
# + FiLM conditioning on the label embedding + a per-token binary head.
# --------------------------------------------------------------------------------------------

class SpanLocalizationModel(nn.Module):
  def __init__(self, esm_model_name, label_dim, unfreeze_last_n=2, output_dropout=0.2):
    super().__init__()
    self.esm = EsmModel.from_pretrained(esm_model_name, add_pooling_layer=False)
    hidden_size = self.esm.config.hidden_size

    for param in self.esm.parameters():
      param.requires_grad = False
    num_layers = self.esm.config.num_hidden_layers
    unfreeze_last_n = min(unfreeze_last_n, num_layers)
    if unfreeze_last_n > 0:
      for i in range(num_layers - unfreeze_last_n, num_layers):
        for param in self.esm.encoder.layer[i].parameters():
          param.requires_grad = True
    print(f"[INFO] ESM backbone: {num_layers} layers, last {unfreeze_last_n} unfrozen")

    self.film_scale = nn.Linear(label_dim, hidden_size)
    self.film_shift = nn.Linear(label_dim, hidden_size)
    # Start FiLM as a near-identity transform (scale~0 -> multiplier~1, shift~0) so early
    # training doesn't immediately scramble the pretrained ESM representations.
    nn.init.zeros_(self.film_scale.weight)
    nn.init.zeros_(self.film_scale.bias)
    nn.init.zeros_(self.film_shift.weight)
    nn.init.zeros_(self.film_shift.bias)

    self.span_head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.GELU(),
        nn.Dropout(output_dropout),
        nn.Linear(hidden_size // 2, 1),
    )

  def forward(self, input_ids, attention_mask, label_embedding):
    hidden = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    scale = self.film_scale(label_embedding).unsqueeze(1)  # [B, 1, H]
    shift = self.film_shift(label_embedding).unsqueeze(1)
    hidden = hidden * (1.0 + scale) + shift
    logits = self.span_head(hidden).squeeze(-1)  # [B, L]
    return logits


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------

def token_level_metrics(logits, targets, valid_mask, threshold=0.5):
  probs = torch.sigmoid(logits)
  preds = (probs >= threshold).float()
  valid = valid_mask.bool()
  preds_v, targets_v = preds[valid], targets[valid]

  tp = ((preds_v == 1) & (targets_v == 1)).sum().item()
  fp = ((preds_v == 1) & (targets_v == 0)).sum().item()
  fn = ((preds_v == 0) & (targets_v == 1)).sum().item()

  precision = tp / (tp + fp) if (tp + fp) else 0.0
  recall = tp / (tp + fn) if (tp + fn) else 0.0
  f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
  return {"precision": precision, "recall": recall, "f1": f1}


def compute_pos_weight(examples):
  total_valid = 0
  total_positive = 0
  for ex in examples:
    n = len(ex["sequence"])
    total_valid += n
    total_positive += sum(min(end, n) - start for start, end in ex["spans"])
  total_negative = total_valid - total_positive
  return total_negative / max(total_positive, 1)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def load_dataset_dict():
  import datasets
  if LOCAL_DATASET_PATH.exists():
    print(f"[INFO] Loading dataset from {LOCAL_DATASET_PATH}")
    return datasets.DatasetDict.load_from_disk(LOCAL_DATASET_PATH)
  print(f"[INFO] Local dataset not found, loading {HUB_DATASET_ID} from the Hub")
  return datasets.load_dataset(HUB_DATASET_ID)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model-name", default=DEFAULT_ESM_MODEL)
  ap.add_argument("--epochs", type=int, default=3)
  ap.add_argument("--batch-size", type=int, default=8)
  ap.add_argument("--learning-rate", type=float, default=5e-5)
  ap.add_argument("--unfreeze-last-n", type=int, default=2)
  ap.add_argument("--output-dropout", type=float, default=0.2)
  ap.add_argument("--weight-decay", type=float, default=0.01)
  ap.add_argument("--grad-clip-norm", type=float, default=1.0)
  ap.add_argument("--n-negatives-per-protein", type=int, default=2)
  ap.add_argument("--pos-weight", type=float, default=None,
                   help="Override the auto-computed BCE pos_weight (negative/positive token ratio).")
  ap.add_argument("--max-train-pairs", type=int, default=None, help="Subset for smoke tests.")
  ap.add_argument("--max-val-pairs", type=int, default=None, help="Subset for smoke tests.")
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--seed", type=int, default=SEED)
  args = ap.parse_args()

  random.seed(args.seed)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  config = vars(args)
  print(f"[CONFIG] {json.dumps(config)}")

  dataset_dict = load_dataset_dict()

  print("[INFO] Building (protein, GO-term) pair examples...")
  train_examples = build_pair_examples(dataset_dict["train"], args.n_negatives_per_protein, args.seed)
  val_examples = build_pair_examples(dataset_dict["validation"], args.n_negatives_per_protein, args.seed)
  if args.max_train_pairs:
    train_examples = train_examples[:args.max_train_pairs]
  if args.max_val_pairs:
    val_examples = val_examples[:args.max_val_pairs]
  print(f"[INFO] train pairs: {len(train_examples)} "
        f"({sum(e['is_positive'] for e in train_examples)} positive)")
  print(f"[INFO] val pairs: {len(val_examples)} "
        f"({sum(e['is_positive'] for e in val_examples)} positive)")

  all_go_ids = {ex["go_id"] for ex in train_examples} | {ex["go_id"] for ex in val_examples}
  label_embeddings, label_dim = build_label_embeddings(all_go_ids, device=args.device)

  pos_weight_value = args.pos_weight or compute_pos_weight([e for e in train_examples if e["is_positive"]])
  print(f"[INFO] BCE pos_weight: {pos_weight_value:.2f}")

  tokenizer = EsmTokenizer.from_pretrained(args.model_name)
  collate_fn = make_collate_fn(tokenizer, label_embeddings)

  train_loader = DataLoader(SpanPairDataset(train_examples, label_embeddings),
                             batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
  val_loader = DataLoader(SpanPairDataset(val_examples, label_embeddings),
                           batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

  model = SpanLocalizationModel(args.model_name, label_dim, args.unfreeze_last_n,
                                 args.output_dropout).to(args.device)

  trainable_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
  loss_fn = nn.BCEWithLogitsLoss(reduction="none")
  pos_weight = torch.tensor(pos_weight_value, device=args.device)

  for epoch in range(args.epochs):
    model.train()
    train_loss_total, train_n_batches = 0.0, 0
    for batch in train_loader:
      input_ids = batch["input_ids"].to(args.device)
      attention_mask = batch["attention_mask"].to(args.device)
      targets = batch["targets"].to(args.device)
      valid_mask = batch["valid_mask"].to(args.device)
      label_embedding = batch["label_embedding"].to(args.device)

      logits = model(input_ids, attention_mask, label_embedding)
      weight = torch.where(targets == 1, pos_weight, torch.tensor(1.0, device=args.device))
      per_token_loss = nn.functional.binary_cross_entropy_with_logits(
          logits, targets, weight=weight, reduction="none")
      loss = (per_token_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip_norm)
      optimizer.step()

      train_loss_total += loss.item()
      train_n_batches += 1

    model.eval()
    val_loss_total, val_n_batches = 0.0, 0
    all_logits, all_targets, all_valid = [], [], []
    with torch.no_grad():
      for batch in val_loader:
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        targets = batch["targets"].to(args.device)
        valid_mask = batch["valid_mask"].to(args.device)
        label_embedding = batch["label_embedding"].to(args.device)

        logits = model(input_ids, attention_mask, label_embedding)
        weight = torch.where(targets == 1, pos_weight, torch.tensor(1.0, device=args.device))
        per_token_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=weight, reduction="none")
        loss = (per_token_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        val_loss_total += loss.item()
        val_n_batches += 1

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())
        all_valid.append(valid_mask.cpu())

    # Pad variable-length batches to the max width before concatenating for aggregate metrics.
    max_len = max(t.shape[1] for t in all_logits)
    def pad(t):
      return nn.functional.pad(t, (0, max_len - t.shape[1]))
    metrics = token_level_metrics(
        torch.cat([pad(t) for t in all_logits]),
        torch.cat([pad(t) for t in all_targets]),
        torch.cat([pad(t) for t in all_valid]),
    )

    train_loss = train_loss_total / max(train_n_batches, 1)
    val_loss = val_loss_total / max(val_n_batches, 1)
    print(f"[EPOCH {epoch + 1}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
          f"val_precision={metrics['precision']:.4f} val_recall={metrics['recall']:.4f} "
          f"val_f1={metrics['f1']:.4f}")

  print(f"[SUMMARY] {json.dumps({'final_val_loss': val_loss, **metrics})}")


if __name__ == "__main__":
  main()
