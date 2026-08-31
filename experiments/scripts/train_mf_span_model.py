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
PyTorch training loop (same manual-loop style as training_run.py, no Trainer abstraction). The
best (lowest val_loss, or highest F1 with --checkpoint-selection f1) checkpoint is saved locally
every epoch it improves and can optionally be pushed to the Hub via --push-to-hub (see
save_checkpoint/push_checkpoint_to_hub below) -- this model isn't a transformers PreTrainedModel
(it's a plain nn.Module: ESM backbone + FiLM layers + span head), so checkpointing is state_dict +
a small span_model_config.json rather than save_pretrained/push_to_hub, and load_span_model()
reconstructs it on the other end. Separately, a *latest_checkpoint* (model weights +
optimizer/scheduler/RNG state, via save_training_state) is written unconditionally every epoch,
regardless of improvement, so --resume-from-checkpoint can continue a crashed run losing at most
one epoch of work instead of restarting the whole trial -- see experiments/mf_span_sweep_log.md's
trial 16/18 crashes (Xid 13 GPU exceptions, not a code bug) for why this exists, and
experiments/scripts/run_mf_span_trial.sh for a thin wrapper that does this automatically. Also logs
per-epoch metrics to trackio if installed, local-only (no Hub Space) since training now runs on
local hardware -- see init_trackio_run below. Regularization knobs mirror training_run.py's
established surface (same flag names/semantics, duplicated rather than imported -- see the
self-containment note above save_checkpoint): --esm-dropout/--attention-dropout on the unfrozen
backbone layers (in addition to the existing --output-dropout on the span head), an
LR warmup + --lr-schedule {constant,linear_decay,cosine} via get_lr_schedule, and an opt-in
--decouple-bias-wd to exclude 1-D (bias/LayerNorm) params from weight decay.
"""
import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import obonet
import torch
import torch.nn as nn
from huggingface_hub import HfApi, get_token, hf_hub_download
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer

try:
  import trackio
except ImportError:  # pragma: no cover - optional dependency
  trackio = None

DATA_PATH = Path("./data/cafa-6-protein-function-prediction")
LOCAL_DATASET_PATH = DATA_PATH / "dataset_mf_placed"
HUB_DATASET_ID = "kelmoir/cafa6_mf_placed"
OBO_PATH = DATA_PATH / "Train" / "go-basic.obo"
GO_TEXT_CACHE_PATH = Path("experiments/positional_mf_go_term_texts.json")
HF_REPO_ID = os.getenv("HF_MF_SPAN_REPO_ID", "kelmoir/cafa6-mf-span")

WHOLE_PROTEIN_SOURCE = "domain_whole_protein_fallback"
LABEL_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
MAX_LENGTH = 1024
SEED = 42

DEF_PATTERN = re.compile(r'^"(.*)"')


# --------------------------------------------------------------------------------------------
# LR schedule -- same warmup + {constant,linear_decay,cosine} shape as training_run.py's
# get_lr_schedule, duplicated here for the same self-containment reason as get_hf_token below.
# --------------------------------------------------------------------------------------------

def get_lr_schedule(optimizer, num_warmup_steps, num_training_steps, schedule_type="constant"):
  def lr_lambda(step):
    if step < num_warmup_steps:
      return (step + 1) / max(1, num_warmup_steps)
    if schedule_type == "constant":
      return 1.0
    progress = min(1.0, (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps))
    if schedule_type == "linear_decay":
      return max(0.0, 1.0 - progress)
    if schedule_type == "cosine":
      return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown schedule_type: {schedule_type}")
  return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
  def __init__(self, esm_model_name, label_dim, unfreeze_last_n=2, output_dropout=0.2,
               esm_dropout=0.0, attention_dropout=0.0):
    super().__init__()
    self.esm = EsmModel.from_pretrained(esm_model_name, add_pooling_layer=False)
    hidden_size = self.esm.config.hidden_size

    for param in self.esm.parameters():
      param.requires_grad = False
    num_layers = self.esm.config.num_hidden_layers
    unfreeze_last_n = min(unfreeze_last_n, num_layers)
    if unfreeze_last_n > 0:
      # esm_dropout/attention_dropout only set on the unfrozen layers -- same convention as
      # training_run.py's create_esm: dropout on a frozen layer has no effect anyway (no
      # backward pass touches it), so there's nothing to gain applying it more broadly.
      for i in range(num_layers - unfreeze_last_n, num_layers):
        for param in self.esm.encoder.layer[i].parameters():
          param.requires_grad = True
        self.esm.encoder.layer[i].output.dropout.p = esm_dropout
        self.esm.encoder.layer[i].attention.output.dropout.p = attention_dropout
    print(f"[INFO] ESM backbone: {num_layers} layers, last {unfreeze_last_n} unfrozen "
          f"(esm_dropout={esm_dropout}, attention_dropout={attention_dropout})")

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
# Experiment tracking -- local trackio only (no space_id): training now runs on local hardware
# (a single ~15GB-usable-VRAM GPU, not Colab), and trackio's Hub-Space persistence isn't a fit
# for that -- see training_run.py's init_trackio_run for the Space-backed variant this project
# used with Colab. trackio.init() with no space_id writes to a local sqlite db and serves its
# own local dashboard (`trackio show`), which is exactly what's wanted here.
# --------------------------------------------------------------------------------------------

def init_trackio_run(run_name, config=None):
  if trackio is None:
    print("[INFO] trackio is not installed; skipping experiment tracking.")
    return None
  try:
    trackio.init(project="cafa6-mf-span", name=run_name, config=config)
    return trackio
  except Exception as exc:  # pragma: no cover - runtime dependency
    print(f"[WARN] Could not initialize trackio: {exc}")
    return None


# --------------------------------------------------------------------------------------------
# Checkpointing -- this model is a plain nn.Module (ESM backbone + FiLM layers + span head), not
# a transformers PreTrainedModel, so there's no save_pretrained/push_to_hub to lean on. Saves the
# full state dict (safetensors) plus a small span_model_config.json recording the constructor
# args needed to rebuild the architecture before loading weights back in -- same
# local-dir-then-upload_folder pattern training_run.py's push_checkpoint uses. Kept self-contained
# here rather than imported from training_run.py: colab exec -f only ever transmits a single
# local file's source to the remote VM, no sibling-module imports (see CLAUDE.md).
# --------------------------------------------------------------------------------------------

def get_hf_token(hf_token_file=None):
  try:
    from google.colab import userdata  # type: ignore
    token = userdata.get("HF_TOKEN")
    if token:
      print("[INFO] HF token resolved via Colab Secrets")
      return token
  except Exception:
    pass

  if hf_token_file:
    try:
      token = Path(hf_token_file).read_text().strip()
      os.remove(hf_token_file)
      if token:
        print("[INFO] HF token resolved via --hf-token-file")
        return token
    except Exception as exc:
      print(f"[WARN] Could not read --hf-token-file {hf_token_file}: {exc}")

  env_token = os.getenv("HF_TOKEN")
  if env_token:
    print("[INFO] HF token resolved via HF_TOKEN env var")
    return env_token

  cached_token = get_token()
  if cached_token:
    print("[INFO] HF token resolved via huggingface_hub cache")
    return cached_token

  print("[WARN] No HF token found from any source")
  return None


def save_checkpoint(model, tokenizer, local_dir, esm_model_name, label_dim, unfreeze_last_n,
                     output_dropout, esm_dropout, attention_dropout, epoch, metrics, config):
  local_dir.mkdir(parents=True, exist_ok=True)
  state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
  save_file(state_dict, local_dir / "model.safetensors")
  tokenizer.save_pretrained(local_dir)
  span_config = {
      "esm_model_name": esm_model_name, "label_dim": label_dim,
      "unfreeze_last_n": unfreeze_last_n, "output_dropout": output_dropout,
      "esm_dropout": esm_dropout, "attention_dropout": attention_dropout,
  }
  (local_dir / "span_model_config.json").write_text(json.dumps(span_config, indent=1))
  (local_dir / "trial_metrics.json").write_text(
      json.dumps({"epoch": epoch, **metrics, "config": config}, default=str))


def save_training_state(local_dir, optimizer, scheduler, epoch, best_score, best_val_loss,
                         best_epoch, best_metrics, epochs_without_improvement):
  """Saves everything save_checkpoint() doesn't -- optimizer/scheduler state, epoch progress, the
  best-so-far bookkeeping, and RNG state -- so --resume-from-checkpoint can continue training
  exactly where a crash interrupted it (same optimizer momentum, same LR schedule position, same
  data shuffling order via the restored RNG state) instead of resuming with the right weights but
  a cold-started optimizer. Written alongside save_checkpoint()'s model.safetensors in the same
  directory (see latest_checkpoint_dir in main()), unconditionally every epoch -- not just
  improving ones like best_checkpoint_dir -- so a crash mid-plateau costs at most one epoch of
  work, not everything back to the last improving epoch. torch.save (not safetensors): the
  optimizer state_dict and RNG states aren't a flat tensor dict, and this file is never pushed to
  the Hub or loaded by anything other than this script's own --resume-from-checkpoint path, so
  there's no interop reason to avoid pickling here the way save_checkpoint avoids it for weights."""
  state = {
      "epoch": epoch, "best_score": best_score, "best_val_loss": best_val_loss,
      "best_epoch": best_epoch, "best_metrics": best_metrics,
      "epochs_without_improvement": epochs_without_improvement,
      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
      "torch_rng_state": torch.get_rng_state(),
      "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
  }
  if torch.cuda.is_available():
    state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
  torch.save(state, local_dir / "training_state.pt")


def load_training_state(local_dir, device):
  # weights_only=False: this file carries RNG-state tuples and other non-tensor Python objects
  # (see save_training_state), which torch>=2.6's default weights_only=True unpickling rejects.
  # Safe here since it's a file this script wrote itself, never a downloaded/untrusted one.
  return torch.load(local_dir / "training_state.pt", map_location=device, weights_only=False)


def push_checkpoint_to_hub(local_dir, repo_id, branch, token, epoch, metrics):
  api = HfApi()
  api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)
  api.create_branch(repo_id, branch=branch, exist_ok=True, token=token)
  api.upload_folder(
      repo_id=repo_id, folder_path=str(local_dir), revision=branch, token=token,
      commit_message=f"epoch {epoch}: val_loss={metrics.get('val_loss', float('nan')):.4f}",
  )


def load_span_model(repo_id_or_path, device="cpu", revision=None):
  """Reconstruct a SpanLocalizationModel from a checkpoint written by save_checkpoint() --
  repo_id_or_path may be a local directory or a Hub repo id (optionally at a branch/revision)."""
  path = Path(repo_id_or_path)
  if path.exists():
    config_path = path / "span_model_config.json"
    weights_path = path / "model.safetensors"
  else:
    config_path = Path(hf_hub_download(repo_id_or_path, "span_model_config.json", revision=revision))
    weights_path = Path(hf_hub_download(repo_id_or_path, "model.safetensors", revision=revision))

  span_config = json.loads(config_path.read_text())
  model = SpanLocalizationModel(
      span_config["esm_model_name"], span_config["label_dim"],
      span_config["unfreeze_last_n"], span_config["output_dropout"],
      # .get(): checkpoints saved before esm_dropout/attention_dropout existed default to 0.0,
      # which is also what they trained with (the args didn't exist yet).
      span_config.get("esm_dropout", 0.0), span_config.get("attention_dropout", 0.0),
  )
  model.load_state_dict(load_file(weights_path))
  return model.to(device)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def iter_grad_accum_windows(loader, accum_steps):
  """Groups a DataLoader's micro-batches into windows of up to accum_steps batches each (the
  final window of an epoch may be shorter if len(loader) isn't evenly divisible). Batches stay
  as CPU tensors here -- only moved to args.device inside the per-micro-batch training step --
  so grouping them costs nothing beyond a Python list."""
  window = []
  for batch in loader:
    window.append(batch)
    if len(window) == accum_steps:
      yield window
      window = []
  if window:
    yield window


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
  ap.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Accumulate gradients over this many micro-batches (each of size "
                        "--batch-size) before every optimizer step, simulating a "
                        "--batch-size * --grad-accum-steps effective batch without the memory "
                        "cost of holding that many sequences' activations at once -- only the "
                        "accumulated gradients are cheap to keep around; forward/backward still "
                        "run one micro-batch at a time. Normalizes each micro-batch's loss by the "
                        "*whole window's* total valid-token count (computed up front from "
                        "valid_mask, before any forward pass -- free) rather than its own "
                        "micro-batch's count -- the fix for the well-known 2024 Unsloth/HF "
                        "grad-accum bug, where naively averaging each micro-batch's own "
                        "already-token-normalized loss is NOT equivalent to computing loss over "
                        "the true combined token count whenever micro-batches have unequal valid "
                        "token counts (guaranteed here, since protein sequences vary in length). "
                        "Default 1 reduces to the exact prior per-micro-batch behavior.")
  ap.add_argument("--learning-rate", type=float, default=5e-5)
  ap.add_argument("--unfreeze-last-n", type=int, default=2)
  ap.add_argument("--output-dropout", type=float, default=0.2, help="Dropout inside the span head")
  ap.add_argument("--esm-dropout", type=float, default=0.0,
                   help="Dropout on unfrozen ESM encoder layers' output sublayer")
  ap.add_argument("--attention-dropout", type=float, default=0.0,
                   help="Attention dropout on unfrozen ESM encoder layers")
  ap.add_argument("--weight-decay", type=float, default=0.01)
  ap.add_argument("--decouple-bias-wd", action="store_true",
                   help="Exclude 1-D params (biases, LayerNorm weights) from weight decay "
                        "(standard practice; opt-in to match training_run.py's default)")
  ap.add_argument("--grad-clip-norm", type=float, default=1.0)
  ap.add_argument("--compile", action="store_true",
                   help="Wrap the model with torch.compile() for PyTorch 2 graph-compilation "
                        "speedups. Optional/opt-in: first-batch compile overhead can dominate on "
                        "short smoke tests, and checkpointing always saves from the uncompiled "
                        "module reference regardless (see checkpoint_model below), so this only "
                        "affects training speed, not checkpoint compatibility.")
  ap.add_argument("--lr-schedule", choices=["constant", "linear_decay", "cosine"], default="constant",
                   help="Shape of the post-warmup LR schedule")
  ap.add_argument("--warmup-steps", type=int, default=None,
                   help="Override the default warmup step count (default: min(100, steps_per_epoch // 10))")
  ap.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Soften training-time BCE targets toward 0.5 by this amount (0=off, e.g. "
                        "0.1 caps target confidence at 0.95/0.05 instead of 1.0/0.0). Applied only "
                        "to the training loss, not validation -- val_loss stays computed against "
                        "hard 0/1 targets so it remains comparable to every past trial and to "
                        "--checkpoint-selection val_loss. Targets the overconfidence-driven "
                        "val_loss/F1 divergence documented in experiments/mf_span_sweep_log.md "
                        "trials 10-12 (train_loss falling toward ~0 while val_loss rises, even as "
                        "F1 keeps improving).")
  ap.add_argument("--early-stop-patience", type=int, default=None,
                   help="Stop training if --checkpoint-selection's monitored metric hasn't improved "
                        "for this many consecutive epochs (default: disabled, train the full "
                        "--epochs budget). Reuses the exact same improvement check checkpoint "
                        "selection uses, so it reflects what's actually being saved -- e.g. with "
                        "--checkpoint-selection f1, a stall in F1 stops the run even if val_loss is "
                        "still moving.")
  ap.add_argument("--n-negatives-per-protein", type=int, default=2)
  ap.add_argument("--pos-weight", type=float, default=None,
                   help="Override the auto-computed BCE pos_weight (negative/positive token ratio).")
  ap.add_argument("--max-train-pairs", type=int, default=None, help="Subset for smoke tests.")
  ap.add_argument("--max-val-pairs", type=int, default=None, help="Subset for smoke tests.")
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--seed", type=int, default=SEED)
  ap.add_argument("--run-name", default=None,
                   help="Local checkpoint dir name / trial_metrics.json label; auto-derived from "
                        "model/epochs/batch-size if omitted.")
  ap.add_argument("--log-file", default=None,
                   help="Per-epoch JSONL log path, flushed+fsynced after every epoch so an "
                        "accidental crash mid-run doesn't lose completed epochs' metrics "
                        "(default: experiments/logs/mf_span_<run-name>.jsonl)")
  ap.add_argument("--checkpoint-selection", choices=["val_loss", "f1"], default="val_loss",
                   help="Metric used to decide which epoch's checkpoint is 'best' (saved/pushed). "
                        "val_loss and F1 can diverge once the model gets confident under this "
                        "task's class imbalance -- see experiments/mf_span_sweep_log.md trials "
                        "5/10/11/12, where the val_loss-selected checkpoint trailed the best "
                        "observed F1 by as much as 0.05. Default kept as val_loss to match every "
                        "past trial's behavior.")
  ap.add_argument("--push-to-hub", action="store_true",
                   help="Upload the best checkpoint (by --checkpoint-selection) to the Hub after training.")
  ap.add_argument("--hf-repo-id", default=HF_REPO_ID,
                   help="Hugging Face Hub repository id, e.g. kelmoir/cafa6-mf-span")
  ap.add_argument("--hub-branch", default="main", help="Branch to push the checkpoint to")
  ap.add_argument("--hf-token-file", default=None,
                   help="Path to a file containing the HF token (e.g. uploaded separately on "
                        "Colab); read once then deleted")
  ap.add_argument("--resume-from-checkpoint", default=None,
                   help="Path to a checkpoint directory previously written as this run's "
                        "'latest_checkpoint' (model.safetensors + training_state.pt, saved "
                        "unconditionally every epoch -- see save_training_state). Resumes the "
                        "optimizer/scheduler/RNG state and epoch count from there instead of "
                        "restarting the trial from scratch -- meant to recover from a mid-run "
                        "crash (see experiments/mf_span_sweep_log.md trial 16/18) without losing "
                        "more than one epoch of work. All other args should match the original "
                        "run's; this isn't validated, just assumed, same as load_span_model(). "
                        "See also experiments/scripts/run_mf_span_trial.sh, a thin wrapper that "
                        "does this automatically on a crash.")
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
                                 args.output_dropout, args.esm_dropout,
                                 args.attention_dropout).to(args.device)

  trainable_params = [p for p in model.parameters() if p.requires_grad]
  # Keep a reference to the uncompiled module for checkpointing: torch.compile's wrapper can
  # prefix state_dict keys ("_orig_mod.") depending on PyTorch version, and parameters are shared
  # (not copied) between the wrapper and the original module either way, so saving from this
  # reference is always safe and produces a checkpoint load_span_model() can read regardless of
  # whether --compile was used to train it.
  checkpoint_model = model
  if args.compile:
    print("[INFO] Compiling model with torch.compile() (--compile)...")
    model = torch.compile(model)
  if args.decouple_bias_wd:
    decay_params = [p for p in trainable_params if p.ndim != 1]
    no_decay_params = [p for p in trainable_params if p.ndim == 1]
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
  else:
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
  loss_fn = nn.BCEWithLogitsLoss(reduction="none")
  pos_weight = torch.tensor(pos_weight_value, device=args.device)

  # Optimizer steps per epoch = ceil(micro-batches / grad_accum_steps), not raw micro-batch
  # count -- with grad_accum_steps > 1, scheduler.step() only fires once per accumulation window.
  steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
  warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, min(100, steps_per_epoch // 10))
  scheduler = get_lr_schedule(
      optimizer, num_warmup_steps=warmup_steps,
      num_training_steps=max(1, steps_per_epoch * args.epochs), schedule_type=args.lr_schedule,
  )

  run_name = args.run_name or (
      f"mf-span-{args.model_name.split('/')[-1]}-epochs{args.epochs}-bs{args.batch_size}")
  best_checkpoint_dir = Path("models") / run_name / "best_checkpoint"
  # Defaults to this run's own latest_checkpoint dir even when NOT resuming, so the very first
  # epoch already has somewhere to write to -- if --resume-from-checkpoint is set, saves continue
  # into that same directory going forward (it's both the source and the ongoing destination),
  # rather than resuming from one location and then silently starting a second one.
  latest_checkpoint_dir = (Path(args.resume_from_checkpoint) if args.resume_from_checkpoint
                            else Path("models") / run_name / "latest_checkpoint")

  start_epoch = 0
  best_val_loss = float("inf")
  best_score = float("inf") if args.checkpoint_selection == "val_loss" else float("-inf")
  best_epoch = None
  best_metrics = None
  epochs_without_improvement = 0

  if args.resume_from_checkpoint:
    resume_dir = Path(args.resume_from_checkpoint)
    print(f"[INFO] Resuming from checkpoint: {resume_dir}")
    checkpoint_model.load_state_dict(load_file(resume_dir / "model.safetensors", device=str(args.device)))
    state = load_training_state(resume_dir, args.device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    start_epoch = state["epoch"]
    best_score = state["best_score"]
    best_val_loss = state["best_val_loss"]
    best_epoch = state["best_epoch"]
    best_metrics = state["best_metrics"]
    epochs_without_improvement = state["epochs_without_improvement"]
    # Restoring RNG state (not just model/optimizer state) means the train_loader's shuffle order
    # for the resumed epoch continues exactly where the crashed run's would have gone, rather than
    # replaying the pre-crash epochs' order over again from a fresh seed.
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
      torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])
    print(f"[INFO] Resumed at epoch {start_epoch} (best so far: "
          f"{args.checkpoint_selection}={best_score:.4f} @ epoch {best_epoch})")

  tracker = init_trackio_run(run_name, config)

  log_file_path = Path(args.log_file) if args.log_file else Path("experiments/logs") / f"mf_span_{run_name}.jsonl"
  log_file_path.parent.mkdir(parents=True, exist_ok=True)
  if args.resume_from_checkpoint and log_file_path.exists():
    # Append rather than overwrite: the pre-crash epochs' lines are still valid and worth keeping
    # in one contiguous log rather than losing them to a fresh "w".
    log_file = open(log_file_path, "a")
    log_file.write(json.dumps({"resumed_from_epoch": start_epoch}) + "\n")
  else:
    log_file = open(log_file_path, "w")
    log_file.write(json.dumps({"config": config}) + "\n")
  log_file.flush()
  os.fsync(log_file.fileno())
  print(f"[INFO] Per-epoch metrics log -> {log_file_path}")

  # Wall-clock from this process's own start, not the crashed run's -- after a resume this
  # undercounts the trial's true total time (see [SUMMARY]'s resumed_from_epoch flag as the signal
  # to add the prior segment's total_time_s back in by hand if the true total matters).
  training_start = time.perf_counter()
  for epoch in range(start_epoch, args.epochs):
    epoch_start = time.perf_counter()
    model.train()
    # Accumulate the raw token-weighted loss numerator and the true valid-token denominator
    # separately across the WHOLE epoch, dividing once at the end -- averaging each window's
    # already-normalized mean (the old train_loss_total / train_n_batches) has the same bias as
    # the val_loss bug fixed below, just usually smaller in practice since --grad-accum-steps
    # windows tend to be more uniformly composed than raw --batch-size validation batches are.
    train_loss_sum, train_valid_tokens = 0.0, 0.0
    for window in iter_grad_accum_windows(train_loader, args.grad_accum_steps):
      # Precompute the whole window's total valid-token count up front -- valid_mask comes
      # straight from tokenization/collation, not the model, so this costs nothing extra and
      # doesn't require a forward pass. Normalizing every micro-batch's loss by this shared total
      # (instead of its own valid_mask sum) makes accumulation exactly equivalent to running the
      # whole window as one big batch -- see --grad-accum-steps help for why the naive per-
      # micro-batch normalization (the 2024 Unsloth/HF grad-accum bug) is wrong here.
      total_valid_tokens = sum(b["valid_mask"].sum().item() for b in window)

      optimizer.zero_grad()
      for batch in window:
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        targets = batch["targets"].to(args.device)
        valid_mask = batch["valid_mask"].to(args.device)
        label_embedding = batch["label_embedding"].to(args.device)

        logits = model(input_ids, attention_mask, label_embedding)
        weight = torch.where(targets == 1, pos_weight, torch.tensor(1.0, device=args.device))
        # Weighting uses the hard targets (a token's true positive/negative status); only the BCE
        # target itself is softened, so pos_weight still applies to the right tokens.
        loss_targets = targets * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
        per_token_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, loss_targets, weight=weight, reduction="none")
        numerator = (per_token_loss * valid_mask).sum()
        loss = numerator / max(total_valid_tokens, 1)
        loss.backward()
        train_loss_sum += numerator.item()
        train_valid_tokens += valid_mask.sum().item()

      torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip_norm)
      optimizer.step()
      scheduler.step()

    model.eval()
    # Same fix as training: accumulate the raw numerator/token-count and divide once, rather than
    # averaging each batch's own already-normalized mean. val_loader's batch size follows
    # --batch-size directly (no accumulation concept in eval), so this is what makes val_loss
    # comparable across runs using different --batch-size -- previously, a smaller --batch-size
    # (e.g. to enable more --grad-accum-steps) silently biased val_loss upward, since sequence
    # length varies and per-batch normalization is uneven across unequal-token-count batches --
    # more so the smaller/more numerous the batches, exactly the pattern that motivated this fix.
    val_loss_sum, val_valid_tokens = 0.0, 0.0
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
        val_loss_sum += (per_token_loss * valid_mask).sum().item()
        val_valid_tokens += valid_mask.sum().item()

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

    train_loss = train_loss_sum / max(train_valid_tokens, 1)
    val_loss = val_loss_sum / max(val_valid_tokens, 1)
    epoch_time = time.perf_counter() - epoch_start
    print(f"[EPOCH {epoch + 1}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
          f"val_precision={metrics['precision']:.4f} val_recall={metrics['recall']:.4f} "
          f"val_f1={metrics['f1']:.4f} epoch_time={epoch_time:.1f}s")

    latest_metrics = {"train_loss": train_loss, "val_loss": val_loss,
                       "epoch_time_s": round(epoch_time, 2), **metrics}
    log_file.write(json.dumps({"epoch": epoch + 1, "lr": scheduler.get_last_lr()[0], **latest_metrics}) + "\n")
    log_file.flush()
    os.fsync(log_file.fileno())
    if tracker is not None:
      tracker.log({"epoch": epoch + 1, "lr": scheduler.get_last_lr()[0], **latest_metrics})
    current_score = val_loss if args.checkpoint_selection == "val_loss" else metrics["f1"]
    improved = (current_score < best_score if args.checkpoint_selection == "val_loss"
                else current_score > best_score)
    if improved:
      best_score = current_score
      best_val_loss = val_loss
      best_epoch = epoch + 1
      best_metrics = latest_metrics
      epochs_without_improvement = 0
      save_checkpoint(checkpoint_model, tokenizer, best_checkpoint_dir, args.model_name, label_dim,
                       args.unfreeze_last_n, args.output_dropout, args.esm_dropout,
                       args.attention_dropout, epoch + 1, latest_metrics, config)
      print(f"[INFO] New best checkpoint ({args.checkpoint_selection}={current_score:.4f}) "
            f"saved -> {best_checkpoint_dir}")
    else:
      epochs_without_improvement += 1

    # Unconditional every-epoch checkpoint for --resume-from-checkpoint, separate from
    # best_checkpoint_dir above (which only updates on an improving epoch, and by itself carries
    # no optimizer/scheduler/RNG state -- not enough to resume training from, only to re-evaluate
    # or deploy the model). Written after the improvement check so best_score/best_epoch/etc.
    # reflect this epoch's result, and before the early-stop check below so even the final epoch
    # before an early stop remains resumable. Costs a full model write every epoch rather than
    # only on improving ones -- negligible at this script's current 35M-param scale (a few seconds
    # against a ~700s epoch), but worth reconsidering if this script is ever pointed at a
    # substantially larger backbone.
    save_checkpoint(checkpoint_model, tokenizer, latest_checkpoint_dir, args.model_name, label_dim,
                     args.unfreeze_last_n, args.output_dropout, args.esm_dropout,
                     args.attention_dropout, epoch + 1, latest_metrics, config)
    save_training_state(latest_checkpoint_dir, optimizer, scheduler, epoch + 1, best_score,
                         best_val_loss, best_epoch, best_metrics, epochs_without_improvement)

    if (args.early_stop_patience is not None
        and epochs_without_improvement >= args.early_stop_patience):
      print(f"[INFO] Early stopping: {args.checkpoint_selection} hasn't improved for "
            f"{epochs_without_improvement} epochs (--early-stop-patience="
            f"{args.early_stop_patience}); stopping after epoch {epoch + 1}/{args.epochs}.")
      break

  total_time = time.perf_counter() - training_start
  log_file.close()
  summary = {'final_val_loss': val_loss, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
             'total_time_s': round(total_time, 2), **(best_metrics or metrics)}
  if args.resume_from_checkpoint:
    # total_time_s above is only this process's own wall-clock -- add the prior segment's
    # total_time_s by hand (from its own [SUMMARY]/crash-time log) if the trial's true total matters.
    summary['resumed_from_epoch'] = start_epoch
  print(f"[SUMMARY] {json.dumps(summary)}")

  if args.push_to_hub:
    if best_epoch is None:
      print("[WARN] --push-to-hub requested but no checkpoint was ever saved "
            "(val_loss never improved); skipping upload.")
    else:
      token = get_hf_token(args.hf_token_file)
      if token:
        print(f"[INFO] Uploading best checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f}) "
              f"to {args.hf_repo_id}@{args.hub_branch}")
        try:
          push_checkpoint_to_hub(best_checkpoint_dir, args.hf_repo_id, args.hub_branch, token,
                                  best_epoch, best_metrics)
          print(f"[INFO] Pushed to {args.hf_repo_id}@{args.hub_branch}")
        except Exception as exc:  # pragma: no cover - runtime dependency
          print(f"[WARN] Hub push failed: {exc}")
      else:
        print("[WARN] --push-to-hub requested but no HF token was found; skipping upload.")


if __name__ == "__main__":
  main()
