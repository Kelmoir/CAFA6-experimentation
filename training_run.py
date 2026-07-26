import argparse
import json
import math
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
    required_packages = ["torch", "transformers", "datasets", "huggingface_hub"]
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


def get_lr_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_type: str = "constant",
):
    def lr_lambda(step: int) -> float:
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


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    """Per-batch correct/tp/fn counts, meant to be summed across an epoch and turned
    into accuracy/recall via finalize_metrics() rather than averaged as per-batch
    ratios. Averaging per-batch recall directly (the previous approach) systematically
    underestimates it once label count is high and positive rate is low relative to
    batch size (e.g. 383 labels at ~3.6% positive with batch_size=8: most (label,
    batch) pairs then have zero true positives, so tp/(tp+fn+eps) is spuriously 0 for
    the overwhelming majority of a batch's labels regardless of true model quality,
    dragging every batch's - and therefore the epoch's - averaged recall toward 0).
    Confirmed directly on trial-050: training reported val_recall=3.55% at epoch 10,
    while a proper pooled-over-the-full-validation-set recall was 45.8% for the same
    checkpoint.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    correct = (preds == labels).float().sum().item()
    total = labels.numel()
    tp = ((preds == 1) & (labels == 1)).sum(dim=0)
    fn = ((preds == 0) & (labels == 1)).sum(dim=0)
    return {"correct": correct, "total": total, "tp": tp, "fn": fn}


def finalize_metrics(accumulated: Dict[str, Any]) -> Dict[str, float]:
    accuracy = accumulated["correct"] / max(1, accumulated["total"])
    recall = (accumulated["tp"] / (accumulated["tp"] + accumulated["fn"] + 1e-8)).mean().item()
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
    parser.add_argument("--checkpoint-mode", choices=["none", "final", "every-epoch"], default="final", help="'final' pushes one commit (the last completed epoch) to the trial's Hub branch; 'every-epoch' pushes after each epoch (needed for --resume-from-revision retries); 'none' disables Hub checkpointing entirely")
    parser.add_argument("--checkpoint-branch", default=None, help="Overrides the auto-derived (run-name) branch name for per-epoch checkpoints")
    parser.add_argument("--hf-token-file", default=None, help="Path to a file containing only an HF token; read once, then deleted")
    parser.add_argument("--resume-from-revision", default=None, help="Load model weights from this revision (branch) of --hf-repo-id instead of the base pretrained model; used to continue a multi-epoch trial across separate colab exec invocations after a connection drop")
    parser.add_argument("--start-epoch", type=int, default=0, help="Count of already-completed epochs when resuming; offsets epoch numbering in logs/checkpoints (0 for a fresh run)")
    parser.add_argument("--lr-schedule", choices=["constant", "linear_decay", "cosine"], default="constant", help="Shape of the post-warmup LR schedule")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override the default warmup step count (default: min(100, steps_per_epoch // 10))")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Smooth 0/1 BCE targets toward 0.5 by this amount (0 disables)")
    parser.add_argument("--decouple-bias-wd", action=argparse.BooleanOptionalAction, default=False, help="Exclude bias and LayerNorm params from weight decay (standard AdamW refinement)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Accumulate gradients over this many steps before each optimizer step (raises effective batch size)")
    parser.add_argument("--pos-weight", action=argparse.BooleanOptionalAction, default=False, help="Weight positive examples in BCE loss by (n_negative/n_positive)**--pos-weight-exponent per label, computed from the training split - helps severely imbalanced multi-label tasks (e.g. the coarse GO-term label space, ~3.6% average positive rate) at the cost of one more axis to tune")
    parser.add_argument("--pos-weight-exponent", type=float, default=0.5, help="Compression exponent for --pos-weight: pos_weight = (n_negative/n_positive)**exponent per label. 0.5=sqrt (default; trial-051 used this), 1.0=raw inverse frequency (not recommended - spans ~12,600x on the coarse-GO-term dataset), lower=gentler correction")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False, help="Suppress per-step [STEP] prints (only needed to keep a colab-cli exec websocket alive; noisy for interactive/notebook use)")
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
        unique_items=label_names,
        label2id=label2id,
    )

    processed_dataset = {
        "train": dataset["train"].with_transform(transform=preprocess_batch_partial),
        "validation": dataset["validation"].with_transform(transform=preprocess_batch_partial),
        "test": dataset["test"].with_transform(transform=preprocess_batch_partial),
    }

    # train stays unsliced here (unlike validation) so each epoch can draw a different
    # --max-train-samples-sized subset below, instead of memorizing the same fixed rows
    # every epoch of a multi-epoch run.
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
    # Computed unconditionally (not just on a fresh start) since --pos-weight needs it
    # regardless of whether we're resuming from a checkpoint.
    raw_train_labels = dataset_train["labels"]
    label_counts = torch.zeros(len(label_names))
    for labs in raw_train_labels:
        for idx in labs:
            label_counts[int(idx)] += 1

    if args.resume_from_revision:
        print(f"[INFO] Resumed model weights from {args.hf_repo_id}@{args.resume_from_revision}")
    else:
        # BCE with an imbalanced label prior otherwise wastes early gradient budget just
        # learning the bias (saturating logits toward the prior), which shrinks per-example
        # gradient magnitude and starves the weight matrix of signal to actually discriminate.
        label_priors = (label_counts / max(1, len(raw_train_labels))).clamp(1e-4, 1 - 1e-4)
        prior_bias = torch.log(label_priors / (1 - label_priors)).to(device)
        with torch.no_grad():
            model_esm.classifier.out_proj.bias.copy_(prior_bias)
        print(f"[INFO] Initialized classifier bias to label priors: {dict(zip(label_names, label_priors.tolist()))}")

    def build_train_dataloader(epoch: int) -> torch.utils.data.DataLoader:
        train_subset = processed_dataset["train"]
        if args.max_train_samples is not None:
            n = min(args.max_train_samples, len(train_subset))
            train_subset = train_subset.shuffle(seed=args.seed + epoch).select(range(n))
        return torch.utils.data.DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(data_collate_function, tokenizer=tokenizer),
        )

    train_dataloader = build_train_dataloader(args.start_epoch)
    validation_dataloader = torch.utils.data.DataLoader(
        processed_dataset["validation"],
        batch_size=args.batch_size,
        collate_fn=partial(data_collate_function, tokenizer=tokenizer),
    )

    if args.decouple_bias_wd:
        decay_params, no_decay_params = [], []
        for name, param in model_esm.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay_params if param.ndim == 1 else decay_params).append(param)
        optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=args.learning_rate,
        )
    else:
        optimizer = AdamW(model_esm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, min(100, len(train_dataloader) // 10))
    scheduler = get_lr_schedule(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, len(train_dataloader) * args.epochs),
        schedule_type=args.lr_schedule,
    )
    pos_weight = None
    if args.pos_weight:
        # Compression exponent, not raw n_negative/n_positive: raw inverse frequency on
        # this dataset spans ~0.4-4800x across labels (the 3 majority-positive aspect
        # roots vs. the rarest depth<=2 terms), which would blow up gradients on the rare
        # occasions a rare-term positive appears in a batch. ratio**exponent compresses
        # that range - trial-051 (exponent=0.5, i.e. sqrt) gave a small AUROC gain but a
        # net-worse best-achievable F1 (0.5727 -> 0.5007), overcorrecting precision away;
        # a lower exponent is a gentler correction, worth testing directly.
        n_negative = len(raw_train_labels) - label_counts
        ratio = n_negative / label_counts.clamp(min=1)
        pos_weight = ratio.pow(args.pos_weight_exponent).to(device)
        print(f"[INFO] Using --pos-weight (exponent={args.pos_weight_exponent}): "
              f"min={pos_weight.min():.2f} median={pos_weight.median():.2f} max={pos_weight.max():.2f}")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    run_name = args.run_name or f"{args.model_name.split('/')[-1]}-epochs{args.epochs}-bs{args.batch_size}-device{device.type}"
    tracker = init_trackio_run(run_name, space_id=args.trackio_space_id, config=config)
    token = get_hf_token(args.hf_token_file)
    checkpoint_branch = args.checkpoint_branch or run_name
    checkpoint_dir = Path("models") / run_name / "checkpoint"
    start_time = time.time()

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        if epoch != args.start_epoch:
            train_dataloader = build_train_dataloader(epoch)
        epoch_start = time.time()
        model_esm.train()
        train_loss = 0.0
        train_metrics_acc = {"correct": 0.0, "total": 0, "tp": torch.zeros(len(label_names), device=device), "fn": torch.zeros(len(label_names), device=device)}

        log_every = max(1, len(train_dataloader) // 25)
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hard_labels = batch["labels"].to(device)
            labels = hard_labels
            if args.label_smoothing > 0:
                labels = labels * (1 - args.label_smoothing) + 0.5 * args.label_smoothing

            outputs = model_esm(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            (loss / args.grad_accum_steps).backward()

            is_accum_boundary = (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_dataloader)
            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(model_esm.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not args.quiet and (step + 1) % log_every == 0:
                # Periodic output keeps the colab exec websocket carrying traffic during
                # the otherwise-silent multi-minute training loop; a long silent stretch
                # has been observed to trigger "Connection was lost" mid-epoch.
                print(f"[STEP] epoch={epoch + 1} step={step + 1}/{len(train_dataloader)} loss={loss.item():.4f}")

            batch_metrics = compute_metrics(logits, hard_labels)
            train_metrics_acc["correct"] += batch_metrics["correct"]
            train_metrics_acc["total"] += batch_metrics["total"]
            train_metrics_acc["tp"] += batch_metrics["tp"]
            train_metrics_acc["fn"] += batch_metrics["fn"]
            train_loss += loss.item()

            if tracker is not None:
                tracker.log({"epoch": epoch, "step": len(train_dataloader) * epoch + 1, "train_loss": loss.item(), **finalize_metrics(batch_metrics)})

        train_loss /= max(1, len(train_dataloader))
        train_metrics = finalize_metrics(train_metrics_acc)

        validation_loss = 0.0
        validation_metrics_acc = {"correct": 0.0, "total": 0, "tp": torch.zeros(len(label_names), device=device), "fn": torch.zeros(len(label_names), device=device)}
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
                validation_metrics_acc["correct"] += batch_metrics["correct"]
                validation_metrics_acc["total"] += batch_metrics["total"]
                validation_metrics_acc["tp"] += batch_metrics["tp"]
                validation_metrics_acc["fn"] += batch_metrics["fn"]

        validation_loss /= max(1, len(validation_dataloader))
        validation_metrics = finalize_metrics(validation_metrics_acc)

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

        latest_metrics = {
            "train_loss": train_loss,
            "val_loss": validation_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": validation_metrics["accuracy"],
            "train_recall": train_metrics["recall"],
            "val_recall": validation_metrics["recall"],
        }

        if args.checkpoint_mode == "every-epoch" and token:
            try:
                push_checkpoint(
                    model_esm,
                    tokenizer,
                    repo_id=args.hf_repo_id,
                    branch=checkpoint_branch,
                    token=token,
                    epoch=epoch + 1,
                    metrics=latest_metrics,
                    config=config,
                    local_dir=checkpoint_dir,
                )
                print(f"[INFO] Pushed checkpoint for epoch {epoch + 1} to {args.hf_repo_id}@{checkpoint_branch}")
            except Exception as exc:  # pragma: no cover - runtime dependency
                print(f"[WARN] Checkpoint push failed for epoch {epoch + 1}: {exc}")
        elif args.checkpoint_mode == "every-epoch" and not token:
            print("[WARN] --checkpoint-mode every-epoch requested but no HF token was found; skipping checkpoint push.")

        if args.max_minutes and (time.time() - start_time) / 60 >= args.max_minutes:
            print(f"[INFO] Reached --max-minutes={args.max_minutes} after epoch {epoch + 1}; stopping early.")
            break

    if args.checkpoint_mode == "final":
        if token:
            try:
                push_checkpoint(
                    model_esm,
                    tokenizer,
                    repo_id=args.hf_repo_id,
                    branch=checkpoint_branch,
                    token=token,
                    epoch=epoch + 1,
                    metrics=latest_metrics,
                    config=config,
                    local_dir=checkpoint_dir,
                )
                print(f"[INFO] Pushed final-epoch checkpoint (epoch {epoch + 1}) to {args.hf_repo_id}@{checkpoint_branch}")
            except Exception as exc:  # pragma: no cover - runtime dependency
                print(f"[WARN] Final checkpoint push failed: {exc}")
        else:
            print("[WARN] --checkpoint-mode final requested but no HF token was found; skipping checkpoint push.")

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


