"""One-off: push trial-013's already-saved best checkpoint (epoch 23, local disk) to
kelmoir/cafa6-esm@main as the Tier-0 baseline. No retraining -- reuses training_run.py's
own push_checkpoint() against the checkpoint it already wrote during the trial-013 run.
"""
import sys
from pathlib import Path
import os

# Known env gotcha (see CLAUDE.md): huggingface_hub/hf_xet version mismatches on this host
# can raise "ImportError: cannot import name 'XetSession' from 'hf_xet'" on any push/upload.
# huggingface_hub.constants reads this env var at IMPORT time (not lazily at push time), so
# it must be set before `import huggingface_hub` runs, not just before the push call itself.
# setdefault (not a direct assignment) leaves an explicit user override alone.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training_run import get_hf_token, push_checkpoint

LOCAL_DIR = Path("models/full-label-trial013-trainNfull/best_checkpoint")

CONFIG = {
    "device": "cuda", "epochs": 30, "batch_size": 2, "learning_rate": 5e-05,
    "max_length": 4192, "model_name": "facebook/esm2_t12_35M_UR50D", "unfreeze_last_n": 4,
    "max_train_samples": None, "max_val_samples": 4000,
    "data_path": "data/cafa-6-protein-function-prediction/dataset_full",
    "hf_repo_id": "kelmoir/cafa6-esm", "dataset_repo_id": "kelmoir/cafa6-full", "seed": 42,
    "output_dropout": 0.2, "esm_dropout": 0.1, "attention_dropout": 0.1,
    "grad_clip_norm": 1.0, "weight_decay": 0.01, "run_name": "full-label-trial013-trainNfull",
    "checkpoint_mode": "final", "lr_schedule": "constant", "label_smoothing": 0.0,
    "grad_accum_steps": 8, "use_lora": False, "use_aspect_input": False,
    "mixed_precision": "no", "gradient_checkpointing": False,
    "checkpoint_selection": "f1", "run_diagnostic": True,
}

BEST_METRICS = {
    "train_loss": 0.004179470722132089, "val_loss": 0.006141055117550423,
    "train_accuracy": 0.9987776926363957, "val_accuracy": 0.9984442370637343,
    "train_recall": 0.01577221415936947, "val_recall": 0.009526001289486885,
    "train_precision": 0.7434542775154114, "val_precision": 0.5827198028564453,
    "train_f1": 0.47326497173446747, "val_f1": 0.37413404119070737,
}

BEST_EPOCH = 23

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    token = get_hf_token()
    if not token:
        raise SystemExit("No HF token found (hf auth token) -- aborting push.")

    push_checkpoint(
        model, tokenizer,
        repo_id="kelmoir/cafa6-esm", branch="main", token=token,
        epoch=BEST_EPOCH, metrics=BEST_METRICS, config=CONFIG, local_dir=LOCAL_DIR,
    )
    print(f"[INFO] Pushed trial-013 best checkpoint (epoch {BEST_EPOCH}) to kelmoir/cafa6-esm@main")
