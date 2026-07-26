"""Diagnostic eval: does the aspect-level model (trial-038 checkpoint) learn real
per-protein signal, or does it just reproduce the label prior?

Runs a no-grad inference pass over the validation split (same 2000-row slice used
during training, deterministic since validation isn't shuffled) and reports, per
aspect and macro-averaged: accuracy, precision, recall, F1 (at threshold 0.5), best-F1
over a threshold sweep, and AUROC. Prints the trivial "always predict positive"
baseline alongside every number for direct comparison -- that baseline's accuracy and
recall are exactly what training_run.py's own compute_metrics() has been reporting all
sweep, so this is the check for whether the model beats it on the metrics that
actually distinguish "learned something" from "learned the prior."

Assumes training_run.py has already been %%writefile'd and imported as a module.
"""

import os

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader
from functools import partial

import training_run

CHECKPOINT_REVISION = "trial-038"
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MAX_VAL_SAMPLES = 2000

token = training_run.get_hf_token()
if token:
    # create_esm's from_pretrained call has no explicit token= param; it relies on
    # ambient auth via huggingface_hub's ("HF_TOKEN" env var / cached CLI login)
    # implicit resolution, which get_hf_token() finding a Colab Secret doesn't set
    # by itself -- do it here so loading the private kelmoir/cafa6-esm repo works.
    os.environ["HF_TOKEN"] = token
else:
    print("[WARN] No HF token resolved -- loading a private Hub checkpoint will likely fail.")

dataset = training_run.load_training_data()
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
model = training_run.create_esm(
    unfreeze_last_n=1,
    model_type="kelmoir/cafa6-esm",
    revision=CHECKPOINT_REVISION,
    id2label=id2label,
    label2id=label2id,
).to(device)
model.eval()
print(f"[INFO] Loaded kelmoir/cafa6-esm@{CHECKPOINT_REVISION} ({MODEL_NAME}) for eval")

loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    collate_fn=partial(training_run.data_collate_function, tokenizer=tokenizer),
)

all_probs, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(batch["labels"])

probs = torch.cat(all_probs).numpy()
labels = torch.cat(all_labels).numpy()
print(f"[INFO] Ran inference on {len(labels)} validation examples")

preds = (probs >= 0.5).astype(float)


def report(name, y_true, y_prob, y_pred):
    prior = y_true.mean()
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    accuracy = (y_pred == y_true).mean()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")  # only one class present in y_true

    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.05):
        f1_t = f1_score(y_true, (y_prob >= t).astype(float), zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, t

    trivial_precision = prior
    trivial_recall = 1.0
    trivial_f1 = 2 * trivial_precision * trivial_recall / (trivial_precision + trivial_recall + 1e-8)

    print(f"\n[{name}] prior={prior:.4f}  n={len(y_true)}")
    print(f"  {'metric':<12}{'model @0.5':>12}{'trivial baseline':>18}")
    print(f"  {'accuracy':<12}{accuracy:>12.4f}{prior:>18.4f}")
    print(f"  {'precision':<12}{precision:>12.4f}{trivial_precision:>18.4f}")
    print(f"  {'recall':<12}{recall:>12.4f}{trivial_recall:>18.4f}")
    print(f"  {'f1':<12}{f1:>12.4f}{trivial_f1:>18.4f}")
    print(f"  {'auroc':<12}{auroc:>12.4f}{0.5:>18.4f}   (0.5 = no ranking power at all, i.e. the trivial case)")
    print(f"  best f1 over threshold sweep: {best_f1:.4f} @ threshold={best_thresh:.2f}")


print("=" * 70)
print("STAGE-1 DIAGNOSTIC: trial-038 (esm2_t12_35M, epoch 10) vs trivial baseline")
print("=" * 70)

for i, aspect in enumerate(label_names):
    report(aspect, labels[:, i], probs[:, i], preds[:, i])

report("MACRO (all aspects pooled)", labels.ravel(), probs.ravel(), preds.ravel())

print("\n[SUMMARY] If AUROC is close to 0.5 across aspects, the model has ~no real")
print("per-protein discriminative signal beyond the label prior. If AUROC is clearly")
print("above 0.5 (e.g. >0.65-0.7), it has learned something real even if 0.5-threshold")
print("accuracy looks unimpressive next to the trivial baseline.")
