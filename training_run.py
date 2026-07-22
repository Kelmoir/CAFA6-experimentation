import argparse
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi, get_token
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
    revision: Optional[str] = None,
):
    if id2label is None or label2id is None:
        raise ValueError("id2label and label2id must be provided")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        revision=revision,
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
                for item in labels_value.split():
                    if item in label2id:
                        labels[label2id[item]] = 1.0
            elif isinstance(labels_value, (list, tuple)):
                # datasets.Sequence(ClassLabel(...)) yields already-encoded integer
                # class ids here, not the string names label2id is keyed on.
                for item in labels_value:
                    if isinstance(item, str):
                        if item in label2id:
                            labels[label2id[item]] = 1.0
                    else:
                        idx = int(item)
                        if 0 <= idx < len(unique_items):
                            labels[idx] = 1.0
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


def init_trackio_run(run_name: str, space_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    if trackio is None:
        print("[INFO] trackio is not installed; skipping experiment tracking.")
        return None

    try:
        trackio.init(project="cafa6", name=run_name, space_id=space_id, config=config)
        return trackio
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[WARN] Could not initialize trackio: {exc}")
        return None


def get_hf_token(hf_token_file: Optional[str] = None) -> Optional[str]:
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


def push_checkpoint(
    model,
    tokenizer,
    repo_id: str,
    branch: str,
    token: str,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    local_dir: Path,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    (local_dir / "trial_metrics.json").write_text(
        json.dumps({"epoch": epoch, **metrics, "config": config}, default=str)
    )

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)
    api.create_branch(repo_id, branch=branch, exist_ok=True, token=token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        revision=branch,
        token=token,
        commit_message=f"epoch {epoch}: val_loss={metrics.get('val_loss', float('nan')):.4f}",
    )


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
    parser.add_argument("--output-dropout", type=float, default=0.2, help="Dropout on the classifier head")
    parser.add_argument("--esm-dropout", type=float, default=0.0, help="Dropout on unfrozen ESM encoder layers")
    parser.add_argument("--attention-dropout", type=float, default=0.0, help="Attention dropout on unfrozen ESM encoder layers")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--max-minutes", type=float, default=None, help="Wall-clock safety cutoff; stop after the current epoch once exceeded")
    parser.add_argument("--run-name", default=None, help="Explicit run/trial id; used for trackio run name and checkpoint branch name")
    parser.add_argument("--trackio-space-id", default=None, help="HF Space id to persist trackio runs to, e.g. kelmoir/cafa6-trackio")
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Push a per-epoch checkpoint to a per-trial Hub branch")
    parser.add_argument("--checkpoint-branch", default=None, help="Overrides the auto-derived (run-name) branch name for per-epoch checkpoints")
    parser.add_argument("--hf-token-file", default=None, help="Path to a file containing only an HF token; read once, then deleted")
    parser.add_argument("--resume-from-revision", default=None, help="Load model weights from this revision (branch) of --hf-repo-id instead of the base pretrained model; used to continue a multi-epoch trial across separate colab exec invocations after a connection drop")
    parser.add_argument("--start-epoch", type=int, default=0, help="Count of already-completed epochs when resuming; offsets epoch numbering in logs/checkpoints (0 for a fresh run)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = vars(args)
    print(f"[CONFIG] {json.dumps(config, default=str)}")

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
        model_type=args.hf_repo_id if args.resume_from_revision else args.model_name,
        output_dropout=args.output_dropout,
        esm_dropout=args.esm_dropout,
        attention_dropout=args.attention_dropout,
        id2label=id2label,
        label2id=label2id,
        revision=args.resume_from_revision,
    ).to(device)
    if args.resume_from_revision:
        print(f"[INFO] Resumed model weights from {args.hf_repo_id}@{args.resume_from_revision}")
    else:
        # BCE with an imbalanced label prior otherwise wastes early gradient budget just
        # learning the bias (saturating logits toward the prior), which shrinks per-example
        # gradient magnitude and starves the weight matrix of signal to actually discriminate.
        raw_train_labels = dataset_train["labels"]
        if args.max_train_samples is not None:
            raw_train_labels = raw_train_labels[: min(args.max_train_samples, len(raw_train_labels))]
        label_counts = torch.zeros(len(UNIQUE_ASPECTS))
        for labs in raw_train_labels:
            for idx in labs:
                label_counts[int(idx)] += 1
        label_priors = (label_counts / max(1, len(raw_train_labels))).clamp(1e-4, 1 - 1e-4)
        prior_bias = torch.log(label_priors / (1 - label_priors)).to(device)
        with torch.no_grad():
            model_esm.classifier.out_proj.bias.copy_(prior_bias)
        print(f"[INFO] Initialized classifier bias to label priors: {dict(zip(label_names, label_priors.tolist()))}")

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

    optimizer = AdamW(model_esm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, min(100, len(train_dataloader) // 10)),
        num_training_steps=max(1, len(train_dataloader) * args.epochs),
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    run_name = args.run_name or f"{args.model_name.split('/')[-1]}-epochs{args.epochs}-bs{args.batch_size}-device{device.type}"
    tracker = init_trackio_run(run_name, space_id=args.trackio_space_id, config=config)
    token = get_hf_token(args.hf_token_file)
    checkpoint_branch = args.checkpoint_branch or run_name
    checkpoint_dir = Path("models") / run_name / "checkpoint"
    start_time = time.time()

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        epoch_start = time.time()
        model_esm.train()
        train_loss = 0.0
        train_metrics = {"accuracy": 0.0, "recall": 0.0}

        log_every = max(1, len(train_dataloader) // 25)
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model_esm(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model_esm.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            if (step + 1) % log_every == 0:
                # Periodic output keeps the colab exec websocket carrying traffic during
                # the otherwise-silent multi-minute training loop; a long silent stretch
                # has been observed to trigger "Connection was lost" mid-epoch.
                print(f"[STEP] epoch={epoch + 1} step={step + 1}/{len(train_dataloader)} loss={loss.item():.4f}")

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
            f"Epoch {epoch + 1}/{args.start_epoch + args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={validation_loss:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | val_acc={validation_metrics['accuracy']:.4f} | "
            f"train_recall={train_metrics['recall']:.4f} | val_recall={validation_metrics['recall']:.4f}"
        )
        print(f"[TIMING] epoch {epoch + 1} elapsed={time.time() - epoch_start:.1f}s")

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

        if args.checkpoint and token:
            try:
                push_checkpoint(
                    model_esm,
                    tokenizer,
                    repo_id=args.hf_repo_id,
                    branch=checkpoint_branch,
                    token=token,
                    epoch=epoch + 1,
                    metrics={
                        "train_loss": train_loss,
                        "val_loss": validation_loss,
                        "train_accuracy": train_metrics["accuracy"],
                        "val_accuracy": validation_metrics["accuracy"],
                        "train_recall": train_metrics["recall"],
                        "val_recall": validation_metrics["recall"],
                    },
                    config=config,
                    local_dir=checkpoint_dir,
                )
                print(f"[INFO] Pushed checkpoint for epoch {epoch + 1} to {args.hf_repo_id}@{checkpoint_branch}")
            except Exception as exc:  # pragma: no cover - runtime dependency
                print(f"[WARN] Checkpoint push failed for epoch {epoch + 1}: {exc}")
        elif args.checkpoint and not token:
            print("[WARN] --checkpoint requested but no HF token was found; skipping checkpoint push.")

        if args.max_minutes and (time.time() - start_time) / 60 >= args.max_minutes:
            print(f"[INFO] Reached --max-minutes={args.max_minutes} after epoch {epoch + 1}; stopping early.")
            break

    summary = {
        "run_name": run_name,
        "epochs_completed": epoch + 1,
        "final_train_loss": train_loss,
        "final_val_loss": validation_loss,
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": validation_metrics["accuracy"],
        "train_recall": train_metrics["recall"],
        "val_recall": validation_metrics["recall"],
    }
    print(f"[SUMMARY] {json.dumps(summary, default=str)}")

    if args.push_to_hub:
        if token:
            print(f"[INFO] Uploading model to Hugging Face Hub as {args.hf_repo_id}")
            model_esm.push_to_hub(args.hf_repo_id, token=token, commit_message="CAFA6 ESM training run")
            tokenizer.push_to_hub(args.hf_repo_id, token=token, commit_message="CAFA6 ESM training run")
        else:
            print("[WARN] Hugging Face Hub upload requested but no token was found; skipping upload.")


if __name__ == "__main__":
    main()


