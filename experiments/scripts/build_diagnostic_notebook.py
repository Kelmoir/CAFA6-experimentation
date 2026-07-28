"""Generate a standalone notebook that loads a specific Hub checkpoint and runs
training_run.run_diagnostic() against it - for checkpoints trained before
--run-diagnostic existed (trial-071/trial-072), or any post-hoc recheck.

Deliberately NOT run on the same GPU tier training used: this is inference-only (no
backward pass, no optimizer state, no training-batch-size memory budget), so a 3B LoRA
checkpoint needs only ~6-10GB VRAM to evaluate (per the smoke test's own numbers) - a
cheap T4/15GB-class card is plenty, no reason to pay A100 rates for this step.

Usage: build_diagnostic_notebook.py <revision> <output.ipynb>
"""

import json
import sys
from pathlib import Path

TRAINING_SCRIPT = Path("training_run.py")


def code_cell(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def markdown_cell(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def main() -> None:
    revision, out_path = sys.argv[1], sys.argv[2]
    training_source = TRAINING_SCRIPT.read_text()

    intro = f"""# CAFA6 diagnostic-only run — {revision}

Loads the `kelmoir/cafa6-esm@{revision}` checkpoint and runs the AUROC/F1 diagnostic
against it (no training). **Use the cheapest GPU available (T4/15GB-class is plenty)
— this is inference-only, not the training-grade card that checkpoint was trained on.**

Run **Runtime > Run all**. Add an `HF_TOKEN` secret (key icon, left sidebar) with read
access to `kelmoir/cafa6-esm` first. The last cell releases the runtime automatically
when done.
"""

    module_load_cell = """!pip install --upgrade torchao

try:
    import torch
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi, get_token
    from transformers import AutoModelForSequenceClassification, EsmTokenizer
    import peft
    import sklearn
except ModuleNotFoundError:
    !pip install datasets
    !pip install huggingface_hub
    !pip install transformers
    !pip install peft
    !pip install scikit-learn
"""

    writefile_cell = "%%writefile training_run.py\n" + training_source

    runner_source = f"""import json
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

import training_run

HF_REPO_ID = "kelmoir/cafa6-esm"
CHECKPOINT_REVISION = {revision!r}

trial_metrics_path = hf_hub_download(repo_id=HF_REPO_ID, filename="trial_metrics.json", revision=CHECKPOINT_REVISION)
trial_config = json.loads(Path(trial_metrics_path).read_text())["config"]
MODEL_NAME = trial_config["model_name"]
USE_LORA = trial_config.get("use_lora", False)
MIXED_PRECISION = trial_config.get("mixed_precision", "no")
AUTOCAST_DTYPE = {{"fp16": torch.float16, "bf16": torch.bfloat16}}.get(MIXED_PRECISION)

dataset = training_run.load_training_data(data_path=Path("data/cafa-6-protein-function-prediction/dataset"))
label_names = [str(name) for name in dataset["train"].features["labels"].feature.names]
id2label = {{i: name for i, name in enumerate(label_names)}}
label2id = {{name: i for i, name in id2label.items()}}

tokenizer = training_run.create_esm_tokenizer(MODEL_NAME)
preprocess = partial(
    training_run.preprocess_batch,
    tokenizer=tokenizer,
    max_length=training_run.MAX_LENGTH,
    unique_items=label_names,
    label2id=label2id,
)
val_ds = dataset["validation"].with_transform(transform=preprocess)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = AUTOCAST_DTYPE is not None and device.type == "cuda"

if USE_LORA:
    model = training_run.create_esm(
        model_type=MODEL_NAME,
        use_lora=True,
        lora_resume_repo_id=HF_REPO_ID,
        revision=CHECKPOINT_REVISION,
        id2label=id2label,
        label2id=label2id,
        lora_base_dtype=AUTOCAST_DTYPE,
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
print(f"[INFO] Loaded {{HF_REPO_ID}}@{{CHECKPOINT_REVISION}} ({{MODEL_NAME}}, use_lora={{USE_LORA}}) for eval, device={{device}}")

training_run.run_diagnostic(
    model,
    tokenizer,
    val_ds,
    label_names,
    device,
    autocast_dtype=AUTOCAST_DTYPE,
    use_amp=use_amp,
    max_val_samples=2000,
    batch_size=16,
)
"""

    runtime_release_cell = """try:
    from google.colab import runtime
    print("[INFO] Diagnostic done - releasing the Colab runtime now.")
    runtime.unassign()
except ImportError:
    print("[INFO] Not running in Colab (or google.colab.runtime unavailable) - skipping runtime release.")
"""

    notebook = {
        "cells": [
            markdown_cell(intro),
            code_cell(module_load_cell),
            code_cell(writefile_cell),
            code_cell(runner_source),
            code_cell(runtime_release_cell),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    Path(out_path).write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {out_path} (diagnostic for {revision})")


if __name__ == "__main__":
    main()
