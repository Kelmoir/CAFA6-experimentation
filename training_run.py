import argparse
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import get_token
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, EsmTokenizer

if "google.colab" in sys.modules:
    required_packages = ["torch", "transformers", "datasets", "trackio", "huggingface_hub"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import trackio
except ImportError:  # pragma: no cover - optional dependency
    trackio = None

MAX_LENGTH = 1024
UNIQUE_ASPECTS = ["C", "F", "P"]
ESM_MODEL_NAMES = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
    "facebook/esm2_t48_15B_UR50D",
]
SELECTED_MODEL = "facebook/esm2_t12_35M_UR50D"
HF_REPO_ID = os.getenv("HF_REPO_ID", "kelmoir/cafa6-esm")


def get_linear_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int):
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, num_warmup_steps)) if step < num_warmup_steps else 1.0,
    )
    return warmup


def create_esm_tokenizer(model_name: str = "facebook/esm2_t30_150M_UR50D") -> EsmTokenizer:
    return EsmTokenizer.from_pretrained(model_name)


def create_esm(
    unfreeze_last_n: int = 2,
    model_type: str = "facebook/esm2_t30_150M_UR50D",
    output_dropout: float = 0.0,
    attention_dropout: float = 0.0,
    esm_dropout: float = 0.0,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
):
    if id2label is None or label2id is None:
        raise ValueError("id2label and label2id must be provided")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    for param in model.esm.parameters():
        param.requires_grad = False

    if unfreeze_last_n > 0:
        num_layers = model.esm.config.num_hidden_layers
        unfreeze_last_n = min(unfreeze_last_n, num_layers)
        print(f"Model layers: {num_layers}, request unfreeze layers: {unfreeze_last_n}")
        for i in range(num_layers - unfreeze_last_n, num_layers):
            for param in model.esm.encoder.layer[i].parameters():
                param.requires_grad = True
            model.esm.encoder.layer[i].output.dropout.p = esm_dropout
            model.esm.encoder.layer[i].attention.output.dropout.p = attention_dropout

    model.classifier.dropout.p = output_dropout
    return model


def preprocess_batch(
    examples,
    tokenizer: EsmTokenizer,
    label2id: Dict[str, int],
    unique_items: List[str],
    max_length: int = MAX_LENGTH,
):
    seq_key = "sequence" if "sequence" in examples else "input_ids"
    if seq_key not in examples:
        return examples

    sequences = examples[seq_key]
    if not isinstance(sequences, list):
        sequences = [sequences]

    tokenized_output = tokenizer(sequences, max_length=max_length, truncation=True)

    input_ids = []
    attention_masks = []
    for item in tokenized_output["input_ids"]:
        if hasattr(item, "tolist"):
            input_ids.append(torch.tensor(item.tolist(), dtype=torch.long))
        else:
            input_ids.append(torch.tensor(item, dtype=torch.long))

    for item in tokenized_output["attention_mask"]:
        if hasattr(item, "tolist"):
            attention_masks.append(torch.tensor(item.tolist(), dtype=torch.long))
        else:
            attention_masks.append(torch.tensor(item, dtype=torch.long))

    output_dict: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }

    if "protein" in examples:
        output_dict["protein"] = examples["protein"]

    if "labels" in examples:
        raw_labels = examples["labels"]
        if not isinstance(raw_labels, list):
            raw_labels = [raw_labels]

        labels_list = []
        for labels_value in raw_labels:
            labels = torch.zeros((len(unique_items),), dtype=torch.float)
            if isinstance(labels_value, str):
                label_terms = labels_value.split()
            elif isinstance(labels_value, (list, tuple)):
                label_terms = [str(item) for item in labels_value]
            else:
                label_terms = []

            for item in label_terms:
                if item in label2id:
                    labels[label2id[item]] = 1.0
            labels_list.append(labels)
        output_dict["labels"] = labels_list

    return output_dict


def data_collate_function(preprocessed_batch: List[Dict[str, Any]], tokenizer: EsmTokenizer) -> Dict[str, Any]:
    collated_data: Dict[str, Any] = {}

    input_ids = [sample["input_ids"] for sample in preprocessed_batch]
    attention_mask = [sample["attention_mask"] for sample in preprocessed_batch]

    collated_data["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=int(tokenizer.pad_token_id)
    )
    collated_data["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    labels = [sample["labels"] for sample in preprocessed_batch if "labels" in sample]
    if labels:
        collated_data["labels"] = torch.stack(labels)

    return collated_data


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    accuracy = (preds == labels).float().mean().item()
    tp = ((preds == 1) & (labels == 1)).sum(dim=0)
    fn = ((preds == 0) & (labels == 1)).sum(dim=0)
    recall = (tp / (tp + fn + 1e-8)).mean().item()
    return {"accuracy": accuracy, "recall": recall}


def init_trackio_run(run_name: str):
    if trackio is None:
        print("[INFO] trackio is not installed; skipping experiment tracking.")
        return None

    try:
        trackio.init(project="cafa6", name=run_name)
        return trackio
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[WARN] Could not initialize trackio: {exc}")
        return None


def load_training_data(data_path: Optional[Path] = None):
    if data_path is not None and data_path.exists():
        return load_from_disk(str(data_path))

    try:
        return load_dataset("kelmoir/CAFA6")
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(f"Could not load the dataset. Provide a local dataset at {data_path} or ensure the Hugging Face dataset is available. Original error: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight ESM-based CAFA6 classifier")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--model-name", default=SELECTED_MODEL)
    parser.add_argument("--unfreeze-last-n", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the finished model to Hugging Face Hub")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID, help="Hugging Face Hub repository id, e.g. kelmoir/cafa6-esm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    data_path = Path("data/cafa-6-protein-function-prediction/dataset")
    dataset = load_training_data(data_path=data_path)

    dataset_train = dataset["train"]
    label_names = [str(name) for name in dataset_train.features["labels"].feature.names]
    id2label = {index: item for index, item in enumerate(label_names)}
    label2id = {value: key for key, value in id2label.items()}

    tokenizer = create_esm_tokenizer(args.model_name)
    preprocess_batch_partial = partial(
        preprocess_batch,
        tokenizer=tokenizer,
        max_length=args.max_length,
        unique_items=UNIQUE_ASPECTS,
        label2id=label2id,
    )

    processed_dataset = {
        "train": dataset["train"].with_transform(transform=preprocess_batch_partial),
        "validation": dataset["validation"].with_transform(transform=preprocess_batch_partial),
        "test": dataset["test"].with_transform(transform=preprocess_batch_partial),
    }

    if args.max_train_samples is not None:
        processed_dataset["train"] = processed_dataset["train"].select(range(min(args.max_train_samples, len(processed_dataset["train"]))))
    if args.max_val_samples is not None:
        processed_dataset["validation"] = processed_dataset["validation"].select(range(min(args.max_val_samples, len(processed_dataset["validation"]))))

    model_esm = create_esm(
        unfreeze_last_n=args.unfreeze_last_n,
        model_type=args.model_name,
        output_dropout=0.2,
        esm_dropout=0.0,
        attention_dropout=0.0,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    train_dataloader = torch.utils.data.DataLoader(
        processed_dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(data_collate_function, tokenizer=tokenizer),
    )
    validation_dataloader = torch.utils.data.DataLoader(
        processed_dataset["validation"],
        batch_size=args.batch_size,
        collate_fn=partial(data_collate_function, tokenizer=tokenizer),
    )

    optimizer = AdamW(model_esm.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, min(100, len(train_dataloader) // 10)),
        num_training_steps=max(1, len(train_dataloader) * args.epochs),
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    run_name = f"{args.model_name.split('/')[-1]}-epochs{args.epochs}-bs{args.batch_size}-device{device.type}"
    tracker = init_trackio_run(run_name)

    for epoch in range(args.epochs):
        model_esm.train()
        train_loss = 0.0
        train_metrics = {"accuracy": 0.0, "recall": 0.0}

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model_esm(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model_esm.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            batch_metrics = compute_metrics(logits, labels)
            train_metrics["accuracy"] += batch_metrics["accuracy"]
            train_metrics["recall"] += batch_metrics["recall"]
            train_loss += loss.item()

            if tracker is not None:
                tracker.log({"epoch": epoch, "step": len(train_dataloader) * epoch + 1, "train_loss": loss.item(), **batch_metrics})

        train_loss /= max(1, len(train_dataloader))
        train_metrics["accuracy"] /= max(1, len(train_dataloader))
        train_metrics["recall"] /= max(1, len(train_dataloader))

        validation_loss = 0.0
        validation_metrics = {"accuracy": 0.0, "recall": 0.0}
        model_esm.eval()
        with torch.inference_mode():
            for batch in validation_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model_esm(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                validation_loss += loss.item()

                batch_metrics = compute_metrics(outputs.logits, labels)
                validation_metrics["accuracy"] += batch_metrics["accuracy"]
                validation_metrics["recall"] += batch_metrics["recall"]

        validation_loss /= max(1, len(validation_dataloader))
        validation_metrics["accuracy"] /= max(1, len(validation_dataloader))
        validation_metrics["recall"] /= max(1, len(validation_dataloader))

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={validation_loss:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | val_acc={validation_metrics['accuracy']:.4f} | "
            f"train_recall={train_metrics['recall']:.4f} | val_recall={validation_metrics['recall']:.4f}"
        )

        if tracker is not None:
            tracker.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": validation_loss,
                    "train_accuracy": train_metrics["accuracy"],
                    "val_accuracy": validation_metrics["accuracy"],
                    "train_recall": train_metrics["recall"],
                    "val_recall": validation_metrics["recall"],
                }
            )

    if args.push_to_hub:
        token = os.getenv("HF_TOKEN") or get_token()
        if token:
            print(f"[INFO] Uploading model to Hugging Face Hub as {args.hf_repo_id}")
            model_esm.push_to_hub(args.hf_repo_id, token=token, commit_message="CAFA6 ESM training run")
            tokenizer.push_to_hub(args.hf_repo_id, token=token, commit_message="CAFA6 ESM training run")
        else:
            print("[WARN] Hugging Face Hub upload requested but no token was found; skipping upload.")


if __name__ == "__main__":
    main()


