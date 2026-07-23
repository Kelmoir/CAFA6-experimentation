"""Generate a standalone Jupyter notebook that runs a batch of trials manually in the
Colab UI (workaround for the unreliable CLI-driven exec connection). Embeds the full
current training_run.py source via %%writefile, then runs each trial in-process by
setting sys.argv and calling main() directly -- same argv convention as the CLI wrapper
scripts, so output format (and thus copy-paste-ability back into this chat) is identical.

Usage: build_notebook.py <manifest.json> <output.ipynb>
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
    manifest_path, out_path = sys.argv[1], sys.argv[2]
    experiments = json.loads(Path(manifest_path).read_text())
    training_source = TRAINING_SCRIPT.read_text()

    intro = f"""# CAFA6 ESM sweep — manual batch run

Generated from `{manifest_path}`. Run cells top to bottom in a Colab GPU runtime
(**Runtime > Change runtime type > T4 GPU**), then copy the full output of the last
cell back into the chat.

**Before running**: add an `HF_TOKEN` secret (key icon in the left sidebar) with a
Hugging Face token that has write access to `kelmoir/cafa6-esm`, and toggle
"Notebook access" on for it. Checkpoints get pushed to `kelmoir/cafa6-esm@<trial-id>`
after every epoch, same as the CLI-driven runs.

This notebook runs {len(experiments)} experiment(s) sequentially: {', '.join(e['trial_id'] for e in experiments)}.
"""

    module_load_cell = """try:
    import torch
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi, get_token
    from transformers import AutoModelForSequenceClassification, EsmTokenizer
except ModuleNotFoundError:
    !pip install datasets
    !pip install huggingface_hub
    !pip install transformers
"""

    writefile_cell = "%%writefile training_run.py\n" + training_source

    runner_lines = [
        "import sys",
        "import time",
        "",
        "import training_run",
        "training_run.trackio = None  # pure text output, no trackio dashboard noise",
        "",
        "experiments = " + json.dumps(experiments, indent=4),
        "",
        "results = {}",
        "for exp in experiments:",
        "    trial_id = exp['trial_id']",
        "    args = exp['args'] + ['--run-name', trial_id, '--checkpoint-branch', trial_id, '--quiet']",
        "    print(f\"\\n{'=' * 20} RUNNING {trial_id} ({exp.get('axis', '')}) {'=' * 20}\\n\")",
        "    print(f\"note: {exp.get('note', '')}\")",
        "    sys.argv = ['training_run.py'] + args",
        "    start = time.time()",
        "    try:",
        "        training_run.main()",
        "        results[trial_id] = 'completed'",
        "    except Exception as exc:",
        "        print(f\"[ERROR] {trial_id} raised: {exc!r}\")",
        "        results[trial_id] = f'FAILED: {exc!r}'",
        "    print(f\"[TIMING] {trial_id} total wall clock: {time.time() - start:.1f}s\")",
        "",
        "print(f\"\\n{'=' * 20} BATCH DONE {'=' * 20}\")",
        "for trial_id, status in results.items():",
        "    print(f\"{trial_id}: {status}\")",
        "",
    ]
    runner_source = "\n".join(runner_lines)

    notebook = {
        "cells": [
            markdown_cell(intro),
            code_cell(module_load_cell),
            code_cell(writefile_cell),
            code_cell(runner_source),
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
    print(f"Wrote {out_path} ({len(experiments)} experiment(s))")


if __name__ == "__main__":
    main()
