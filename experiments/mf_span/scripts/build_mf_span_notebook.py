"""Generate a standalone Jupyter notebook that runs a batch of train_mf_span_model.py trials
manually in the Colab UI -- same rationale and shape as build_notebook.py (training_run.py's
notebook generator), adapted for the mf-span script:

  - Embeds the full current train_mf_span_model.py source via %%writefile (colab exec/run's
    single-file-transmission limitation applies here too -- no sibling-module imports, see
    CLAUDE.md's "self-containment" note on train_mf_span_model.py).
  - Also embeds experiments/mf_span/positional_mf_go_term_texts.json (the cached GO id -> "name: def"
    text lookup used by load_go_term_texts). Without this file present, the script falls back to
    parsing data/cafa-6-protein-function-prediction/Train/go-basic.obo, which isn't available on
    a fresh Colab VM (the raw Kaggle data dir is git-ignored and never uploaded there). The cache
    already covers every GO id any --max-train-pairs/--max-val-pairs prefix of the seed=42,
    n_negatives_per_protein=2 shuffled example list touches (verified against the sweep's own
    larger local runs -- see the 2026-08-27 conversation), so embedding it is enough; no OBO file
    needed for this notebook's trials.
  - kelmoir/cafa6_mf_placed is a public dataset, so no HF_TOKEN secret is needed to load it. A
    token is only required if a trial sets --push-to-hub (off by default here, since this
    notebook's purpose is a wall-clock speed comparison against the local-GPU runs already logged
    in experiments/mf_span/logs/mf_span_mf-span-speed-*.jsonl, not producing a real checkpoint).

Usage: build_mf_span_notebook.py <manifest.json> <output.ipynb>
"""

import json
import sys
from pathlib import Path

TRAINING_SCRIPT = Path("experiments/mf_span/scripts/train_mf_span_model.py")
GO_TEXT_CACHE = Path("experiments/mf_span/positional_mf_go_term_texts.json")


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
    go_text_cache_source = GO_TEXT_CACHE.read_text()

    any_push = any(exp["args"].count("--push-to-hub") for exp in experiments)

    intro = f"""# CAFA6 mf-span Colab speed comparison — manual batch run

Generated from `{manifest_path}`. Run **Runtime > Run all** in a Colab GPU runtime
(**Runtime > Change runtime type > T4 GPU**, to match the local-GPU class these configs were
already timed on), then copy the full output of the training cell back into the chat.

This notebook runs {len(experiments)} trial(s) sequentially, each the exact same config already
timed locally (see `experiments/mf_span/logs/mf_span_mf-span-speed-*.jsonl`), just prefixed `colab-` and
run here instead: {', '.join(e['trial_id'] for e in experiments)}. Compare each trial's per-epoch
`epoch_time=` and final `[SUMMARY]` `total_time_s` against the matching local log to see the
Colab-vs-local wall-clock difference for that batch-size/grad-accum/compile combination.
{"" if not any_push else '''

**Before running**: add an `HF_TOKEN` secret (key icon in the left sidebar) with a Hugging Face
token that has write access to `kelmoir/cafa6-mf-span`, and toggle "Notebook access" on for it --
at least one trial below sets `--push-to-hub`.'''}

**The last cell releases the runtime automatically** once the batch finishes (success or
failure) — use "Run all" rather than stepping through cells by hand, so the disconnect isn't
waiting on someone noticing training is done. Rented GPUs charge for idle time same as busy time.
"""

    module_load_cell = """try:
    import torch
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi, get_token, hf_hub_download
    from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer
    from safetensors.torch import load_file, save_file
    import obonet
except ModuleNotFoundError:
    !pip install datasets
    !pip install huggingface_hub
    !pip install transformers
    !pip install safetensors
    !pip install obonet
"""

    mkdir_cell = """import os
os.makedirs("experiments", exist_ok=True)
"""

    writefile_script_cell = f"%%writefile {TRAINING_SCRIPT.name}\n" + training_source
    writefile_cache_cell = f"%%writefile {GO_TEXT_CACHE}\n" + go_text_cache_source

    runner_lines = [
        "import sys",
        "import time",
        "",
        f"import {TRAINING_SCRIPT.stem} as mf_span",
        "mf_span.trackio = None  # pure text output, no trackio dashboard noise",
        "",
        "experiments = " + json.dumps(experiments, indent=4),
        "",
        "results = {}",
        "for exp in experiments:",
        "    trial_id = exp['trial_id']",
        "    args = exp['args'] + ['--run-name', trial_id]",
        "    print(f\"\\n{'=' * 20} RUNNING {trial_id} ({exp.get('axis', '')}) {'=' * 20}\\n\")",
        "    print(f\"note: {exp.get('note', '')}\")",
        f"    sys.argv = ['{TRAINING_SCRIPT.name}'] + args",
        "    start = time.time()",
        "    try:",
        "        mf_span.main()",
        "        results[trial_id] = 'completed'",
        "    except Exception as exc:",
        "        print(f\"[ERROR] {trial_id} raised: {exc!r}\")",
        "        results[trial_id] = f'FAILED: {exc!r}'",
        "    print(f\"[TIMING] {trial_id} total wall clock (incl. dataset/label-embedding setup): {time.time() - start:.1f}s\")",
        "",
        "print(f\"\\n{'=' * 20} BATCH DONE {'=' * 20}\")",
        "for trial_id, status in results.items():",
        "    print(f\"{trial_id}: {status}\")",
        "",
    ]
    runner_source = "\n".join(runner_lines)

    runtime_release_cell = """try:
    from google.colab import runtime
    print("[INFO] Batch finished - releasing the Colab runtime now.")
    runtime.unassign()
except ImportError:
    print("[INFO] Not running in Colab (or google.colab.runtime unavailable) - skipping runtime release.")
"""

    notebook = {
        "cells": [
            markdown_cell(intro),
            code_cell(module_load_cell),
            code_cell(mkdir_cell),
            code_cell(writefile_script_cell),
            code_cell(writefile_cache_cell),
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
    print(f"Wrote {out_path} ({len(experiments)} experiment(s))")


if __name__ == "__main__":
    main()
