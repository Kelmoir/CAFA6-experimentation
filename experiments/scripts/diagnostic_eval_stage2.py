"""Diagnostic eval: trial-050 (383 coarse GO terms) showed val_loss improving
(0.0803 -> 0.0772 over 10 epochs) while recall stayed pinned near 3% the whole run and
accuracy sat right on top of the trivial "always predict negative" baseline (~96.4%,
given the task's 3.58% overall positive rate). That's the textbook signature of
unweighted BCE collapsing to near-constant low-probability predictions under severe
class imbalance -- but it could also mean there's real per-label ranking signal that
just never crosses the 0.5 threshold. This checks which, via AUROC (threshold-
independent) computed per label, then summarized as a distribution (not printed
383-wide, that's unreadable) - if most labels show AUROC well above 0.5, the model
learned something real and just needs a rebalanced loss/threshold; if AUROC clusters
near 0.5 across the board, training genuinely isn't learning per-protein distinctions
for most terms.

Runs entirely locally on CPU (no GPU needed for a 2000-example inference pass on a
35M-param model) - meant to run directly via `python3 experiments/scripts/diagnostic_eval_stage2.py`,
not through Colab, since it doesn't need training_run.py's Colab-secrets HF token path
(local huggingface_hub cache already has one).
"""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_run

CHECKPOINT_REVISION = sys.argv[1] if len(sys.argv) > 1 else "trial-050"
HF_REPO_ID = "kelmoir/cafa6-esm"
MAX_VAL_SAMPLES = 2000

# push_checkpoint() (training_run.py) writes trial_metrics.json alongside every
# checkpoint with the full CLI config for that trial (model_name, use_lora, etc.) - read
# it back instead of hardcoding model_name/use_lora here, so this script works
# unmodified for any future trial (e.g. a --use-lora esm2_t36_3B run), not just the
# esm2_t12_35M non-LoRA trials it was originally written against.
trial_metrics_path = hf_hub_download(repo_id=HF_REPO_ID, filename="trial_metrics.json", revision=CHECKPOINT_REVISION)
trial_config = json.loads(Path(trial_metrics_path).read_text())["config"]
MODEL_NAME = trial_config["model_name"]
USE_LORA = trial_config.get("use_lora", False)

dataset = training_run.load_training_data(data_path=Path("data/cafa-6-protein-function-prediction/dataset"))
label_names = [str(name) for name in dataset["train"].features["labels"].feature.names]
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in id2label.items()}

tokenizer = training_run.create_esm_tokenizer(MODEL_NAME)
preprocess = partial(
    training_run.preprocess_batch,
    tokenizer=tokenizer,
    max_length=training_run.MAX_LENGTH,
    unique_items=label_names,
    label2id=label2id,
)

val_ds = dataset["validation"].select(range(min(MAX_VAL_SAMPLES, len(dataset["validation"]))))
val_ds = val_ds.with_transform(transform=preprocess)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if USE_LORA:
    # A LoRA checkpoint on HF_REPO_ID is adapter-only, not a full standalone model - the
    # base architecture must load fresh from MODEL_NAME, with the adapter weights
    # attached separately via lora_resume_repo_id. PeftModel.from_pretrained reads the
    # adapter's own saved adapter_config.json (rank/alpha/target_modules) rather than
    # anything passed here, so lora_r/lora_alpha/etc. don't need to be reconstructed.
    model = training_run.create_esm(
        model_type=MODEL_NAME,
        use_lora=True,
        lora_resume_repo_id=HF_REPO_ID,
        revision=CHECKPOINT_REVISION,
        id2label=id2label,
        label2id=label2id,
    ).to(device)
else:
    model = training_run.create_esm(
        unfreeze_last_n=1,
        model_type=HF_REPO_ID,
        revision=CHECKPOINT_REVISION,
        id2label=id2label,
        label2id=label2id,
    ).to(device)
model.eval()
print(f"[INFO] Loaded {HF_REPO_ID}@{CHECKPOINT_REVISION} ({MODEL_NAME}, use_lora={USE_LORA}) for eval, device={device}")

loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False,
    collate_fn=partial(training_run.data_collate_function, tokenizer=tokenizer),
)

all_probs, all_labels = [], []
with torch.no_grad():
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(batch["labels"])
        if (i + 1) % 20 == 0:
            print(f"[INFO] processed {(i + 1) * 16}/{len(val_ds)} examples")

probs = torch.cat(all_probs).numpy()
labels = torch.cat(all_labels).numpy()
print(f"[INFO] Ran inference on {len(labels)} validation examples, {probs.shape[1]} labels")

preds_at_half = (probs >= 0.5).astype(float)

# per-label AUROC, skipping labels with no positive (or no negative) examples in this
# val slice - roc_auc_score is undefined for a single-class label
per_label_auroc = []
n_skipped = 0
for j in range(labels.shape[1]):
    y_true = labels[:, j]
    if y_true.min() == y_true.max():
        n_skipped += 1
        continue
    per_label_auroc.append(roc_auc_score(y_true, probs[:, j]))
per_label_auroc = np.array(per_label_auroc)

print("\n" + "=" * 70)
print(f"STAGE-2 DIAGNOSTIC: trial-050 (383 coarse GO terms) vs trivial baseline")
print("=" * 70)

print(f"\nEvaluable labels (had both classes in this val slice): {len(per_label_auroc)}/{probs.shape[1]} "
      f"({n_skipped} skipped, single-class in this sample)")
print(f"Per-label AUROC distribution:")
print(f"  mean={per_label_auroc.mean():.4f}  median={np.median(per_label_auroc):.4f}  "
      f"min={per_label_auroc.min():.4f}  max={per_label_auroc.max():.4f}")
for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
    frac = (per_label_auroc > thresh).mean()
    print(f"  labels with AUROC > {thresh}: {(per_label_auroc > thresh).sum()}/{len(per_label_auroc)} ({frac * 100:.1f}%)")

# pooled (micro) metrics at threshold 0.5, for continuity with training_run.py's own accuracy/recall
prior = labels.mean()
tp = ((preds_at_half == 1) & (labels == 1)).sum()
fp = ((preds_at_half == 1) & (labels == 0)).sum()
fn = ((preds_at_half == 0) & (labels == 1)).sum()
accuracy = (preds_at_half == labels).mean()
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
pooled_auroc = roc_auc_score(labels.ravel(), probs.ravel())

print(f"\nPooled (micro, all labels flattened) @ threshold 0.5, prior={prior:.4f}:")
print(f"  accuracy={accuracy:.4f} (trivial always-negative baseline: {1 - prior:.4f})")
print(f"  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
print(f"  pooled AUROC={pooled_auroc:.4f}  (0.5 = no ranking power at all)")

# best-F1-over-threshold-sweep, pooled - shows how much is recoverable by just moving
# the classification threshold rather than retraining
best_f1, best_thresh = 0.0, 0.5
for t in np.arange(0.02, 0.51, 0.02):
    f1_t = f1_score(labels.ravel(), (probs.ravel() >= t).astype(float), zero_division=0)
    if f1_t > best_f1:
        best_f1, best_thresh = f1_t, t
print(f"\nBest pooled F1 over a threshold sweep (0.02-0.50): {best_f1:.4f} @ threshold={best_thresh:.2f}")
print("(0.5 was tuned for the old ~70%-positive-rate task; a 3.58%-positive task's optimal "
      "operating point is expected to sit much lower)")

print("\n[SUMMARY] If per-label AUROC clusters near 0.5, training genuinely isn't learning")
print("per-protein distinctions for most terms (a loss-function/imbalance problem, likely")
print("needs pos_weight or focal loss). If AUROC is well above 0.5 for most labels but recall")
print("at 0.5 is near zero, the model learned real ranking signal that the classification")
print("threshold is just badly miscalibrated for - primarily a threshold problem.")
