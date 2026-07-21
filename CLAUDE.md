# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a solution for the Kaggle CAFA 6 (Critical Assessment of Function Annotation) protein function
prediction challenge. The goal is to predict Gene Ontology (GO) terms for proteins from their amino
acid sequences. The active pipeline has two stages, each with its own entry point:

1. **Dataset generation** (`generate_dataset.py`) — reads raw Kaggle data (FASTA sequences, GO term
   annotations, the `go-basic.obo` ontology graph) and produces a Hugging Face `DatasetDict`
   (train/validation/test) pushed to `kelmoir/cafa6` on the HF Hub.
2. **Training** (`training_run.py`) — fine-tunes an ESM-2 protein language model
   (`facebook/esm2_*`) as a multi-label sequence classifier over GO aspects, optionally pushing the
   trained model to `kelmoir/cafa6-esm` on the HF Hub. Currently driven by an automated regularization
   sweep on Colab — see "Colab regularization sweep" below.

Training is intended to run on Google Colab (for GPU access). Colab is driven via the real `colab`
CLI (`google-colab-cli` package) run **inside the Podman sandbox container** (`podman exec cafa6-sandbox
colab ...`) — see "Colab CLI" below. (An earlier version of this doc said to run `colab` directly on
the host instead; that's no longer the case — host-side OAuth is a separate, unrelated credential
from the container's, and the operative auth/session state now lives in the container's bind-mounted
`.podman/colab-config`.)

There is no `claude_cafa6.py` in this repo — an earlier session created a same-named file directly
on a Colab VM as an unmodified copy of `training_run.py`. It's superseded: `training_run.py` is
already self-contained and Colab-aware (see the `google.colab` bootstrap in Architecture notes
below), so every Colab invocation targets it directly via `colab run`/`colab exec`. Don't recreate
`claude_cafa6.py` — a second copy can only drift out of sync, since `colab run`/`colab exec -f` only
ever transmit a single local file's source to the remote VM (no sibling-module imports).

The `.ipynb` notebooks (`CAFA6_exploration.ipynb`, `CAFA6_model_building.ipynb`,
`Biopython learning course.ipynb`) are legacy — early EDA/prototyping that predates `generate_dataset.py`
and `training_run.py`. Don't treat them as reflecting the current pipeline or data flow; the two
scripts above are the source of truth.

## Environment setup

A local virtualenv lives in `env/` (not committed). Activate it before running anything locally:

```bash
source env/bin/activate
```

Two requirements files exist for different purposes:
- `requirements-minimal.txt` — the actual ML stack needed to run `training_run.py` / `generate_dataset.py`
  locally (torch, transformers, datasets, biopython, huggingface-hub, trackio, accelerate, ...).
- `requirements.txt` — minimal deps for the Podman container, which only needs the `colab` CLI and
  `huggingface-hub` (heavy ML packages are installed inside Colab itself, not in the container).

## Common commands

```bash
# Generate the train/val/test HF dataset from raw Kaggle data (data/cafa-6-protein-function-prediction/)
python generate_dataset.py

# Train the ESM classifier (defaults: facebook/esm2_t12_35M_UR50D, 1 epoch, batch size 2)
python training_run.py --epochs 3 --batch-size 8 --learning-rate 5e-5

# Train on a subset for a quick smoke test
python training_run.py --max-train-samples 100 --max-val-samples 50 --device cpu

# Train and push the resulting model to the HF Hub
python training_run.py --push-to-hub --hf-repo-id kelmoir/cafa6-esm

# Launch a ~1h regularization-sweep trial on a Colab T4 (see experiments/sweep_log.md).
# Colab Secrets are NOT reachable from CLI-provisioned sessions (confirmed via
# experiments/smoke_test.py: "Secrets can only be fetched when running from the Colab UI"),
# so `colab run`'s one-shot form can't be used directly — the HF token has to be uploaded as
# a file to a persistent session instead. Run from inside the container
# (podman exec cafa6-sandbox sh -c '...'):
colab new -s trial-NNN --gpu T4
colab upload /secrets/huggingface_token /content/.hf_token -s trial-NNN
colab upload training_run.py /content/training_run.py -s trial-NNN
colab exec -s trial-NNN --timeout 4200 -f wrapper.py   # sets sys.argv incl.
  # --run-name trial-NNN --max-minutes 55 --hf-token-file /content/.hf_token
  # [other hyperparameter flags...], then execs /content/training_run.py
colab stop -s trial-NNN

# Hugging Face auth (needed for dataset/model push, or to read gated resources)
hf auth token

# Upload the raw Kaggle dataset directory to the HF Hub as a dataset repo
hf upload --repo-type=dataset kelmoir/cafa6 ./data/cafa-6-protein-function-prediction/dataset
```

There is no test suite, linter, or CI configured in this repo.

## Data layout

- `data/cafa-6-protein-function-prediction/` — raw Kaggle competition data (`Train/`, `Test/`,
  `IA.tsv`, `sample_submission.tsv`); git-ignored, expected to be downloaded/unzipped locally.
  - `Train/train_sequences.fasta` — protein sequences, parsed with Biopython (`SeqIO`).
  - `Train/train_terms.tsv` — `EntryID` → GO `term` → `aspect` (C/F/P) annotations.
  - `Train/go-basic.obo` — GO ontology graph, loaded with `obonet`.
- `data/cafa-6-protein-function-prediction/dataset/` — output of `generate_dataset.py`
  (`DatasetDict.save_to_disk`), also the default local path `training_run.py` looks for before
  falling back to `load_dataset("kelmoir/CAFA6")` from the Hub.
- `data/exports/` — ad hoc exports produced from the notebooks (git-ignored).
- `models/` — used transiently by `push_checkpoint()` as a local staging dir (`models/<run_name>/checkpoint`)
  before uploading to the Hub; not git-ignored in `.gitignore` today despite being treated as such —
  on a Colab VM this never touches the local sandbox host anyway, so it doesn't matter in practice.

## Architecture notes

- **Label space**: `training_run.py` currently classifies proteins into the three top-level GO
  aspects (`C`ellular component, `F`unction, `P`rocess) as a multi-label problem, not individual GO
  terms — `id2label`/`label2id` are derived from `dataset["train"].features["labels"].feature.names`.
- **Model**: ESM-2 (`facebook/esm2_t*`) loaded via `AutoModelForSequenceClassification` with
  `problem_type="multi_label_classification"`. Only the classifier head plus the last
  `--unfreeze-last-n` encoder layers are trainable; the rest of the ESM backbone is frozen
  (`create_esm` in `training_run.py`).
- **Preprocessing**: `preprocess_batch` tokenizes with `EsmTokenizer` and multi-hot encodes labels;
  it's applied lazily via `dataset.with_transform(...)` rather than a full `.map()` pass. Sequences
  longer than `max_length` (1024 by default) are truncated at generation time in
  `generate_dataset.py` by filtering, not at train time.
- **Training loop**: plain PyTorch (no `Trainer`/`accelerate` loop abstraction) — manual epoch loop,
  `AdamW`, linear warmup schedule, `BCEWithLogitsLoss`, gradient clipping. Metrics
  (`compute_metrics`) are simple threshold-0.5 accuracy/recall, not the official CAFA metric.
- **Experiment tracking**: optional `trackio` integration (`init_trackio_run`) exists but is currently
  **unused** — trackio's Hub-Space persistence isn't viable right now (HF changed free-Space policy),
  so sweep trials don't pass `--trackio-space-id`. The script degrades gracefully (prints a warning,
  continues without tracking) if `trackio` isn't installed, isn't passed a space, or fails to init.
  Per-trial `experiments/logs/trial-NNN.log` plus the `trial_metrics.json` pushed alongside each
  Hub checkpoint are the actual source of truth for the sweep (see "Colab regularization sweep").
- **Colab bootstrapping**: `training_run.py` detects `google.colab` in `sys.modules` and pip-installs
  missing heavy dependencies at import time — this is intentional for the Colab-first workflow, don't
  "fix" it into a hard requirements dependency.

## Colab CLI

**`PODMAN_SANDBOX.md` and `COLAB_CLI_GUIDE.md` describe a `colab notebook ...` command that does not
exist** — they were written against a mistaken understanding of the `google-colab-cli` tool and are
stale. Don't follow them. The real CLI (verified via `colab --help` / `colab skill`) has no
`notebook` subcommand; the ones that matter here are `new`, `exec`, `run`, `install`, `log`,
`status`, `sessions`, `stop`. Auth defaults to `--auth=oauth2` (not `adc`).

Commands that matter here:
- `colab exec -s <name> -f script.py --timeout <seconds>` — runs a local script's *content* on an
  existing session (no argv passthrough; default timeout is only 30s, always override for real work).
- `colab run --session <name> --gpu T4 --timeout <seconds> script.py [args...]` — provisions a fresh
  VM, runs the script with real `sys.argv`/`__name__=="__main__"` semantics, then tears the VM down
  automatically, even on error (unless `--keep`). Fine for smoke tests, but **not used for real sweep
  trials** — it has no window to pre-upload a file, and Colab Secrets are unreachable from
  CLI-provisioned sessions (see "Colab regularization sweep"), so trials need the token uploaded as a
  file via `colab new`/`colab upload`/`colab exec`/`colab stop` instead. Note the flag spelling
  difference: `run` takes `--session NAME`; `new`/`exec`/`stop`/`upload` take `-s NAME`.

**Run all `colab` commands from inside the Podman sandbox container**
(`podman exec cafa6-sandbox colab ...`, or `podman exec -it cafa6-sandbox bash` for an interactive
shell) — not on the host. The container's `.podman/colab-config` is bind-mounted to
`/home/sandboxuser/.config/colab-cli` and genuinely persists across `podman-start.sh`/`podman-stop.sh`
cycles (verified directly: stopping and restarting the container does not lose this state). Host-side
`colab` auth is a **separate, unrelated OAuth grant** from the container's — don't conflate the two,
and don't try to share credentials between them (an earlier attempt at this, via a `COLAB_CLI_CONFIG`
env-var trick, did not work).

If `colab` commands start demanding a fresh interactive login on every invocation even though
`.podman/colab-config/token.json` exists and looks current, check
`.podman/colab-config/colab.log` for `colab_cli.auth - WARNING - Failed to refresh token:
invalid_scope` — that means the cached refresh token is stale/broken (not a persistence problem; the
bind mount is fine), and the fix is one fresh interactive login (`podman exec -it cafa6-sandbox colab
status`, click through the browser prompt) after moving the broken `token.json` aside. `.podman/colab-config/colab.log` has previously logged plaintext OAuth tokens at DEBUG level —
don't reintroduce that log level, and never `cat`/print this log's contents into a shared or logged
context without filtering.

Claude Code itself needs `/home/sandboxuser/.local/bin` on `PATH` inside the container (fixed via an
`ENV PATH=...` in the `Dockerfile`) — a native install already lives there (bind-mounted via
`.podman/colab-local`), it just wasn't reachable by non-interactive shells until that fix.

## Colab regularization sweep

An agent-driven loop (see `experiments/sweep_log.md`) launches ~1h training trials on a Colab T4 (via
the `colab new`/`upload`/`exec`/`stop` sequence described in "Colab CLI" above, run from inside the
Podman sandbox container), judges each trial's train/val curves against a fixed rubric, and iterates
toward regularization/unfreezing settings that train reliably (manual attempts had consistently
either overfit or failed to learn at all). `experiments/sweep_log.md` is the source of truth: it pins
the judgment rubric and search strategy at the top, then logs one section per trial (config, per-epoch
metrics, verdict, next-config decision, Hub checkpoint branch, commit sha). `experiments/logs/`
holds each trial's captured stdout/stderr; `experiments/smoke_test.py` is a throwaway infra check
(GPU availability, Colab Secrets reachability) run once before trusting the pipeline for real trials —
it confirmed GPU access works but **Colab Secrets are not reachable from CLI-provisioned sessions**
(`"Secrets can only be fetched when running from the Colab UI"`), which is why every trial uploads the
HF token as a file rather than relying on `google.colab.userdata`. trackio-Space tracking
(`--trackio-space-id`) is currently **not used** — HF's free-Space policy changes made it a non-option
for now — so `experiments/logs/trial-NNN.log` and each checkpoint's `trial_metrics.json` are the sole
record of a trial's curves. Every trial pushes per-epoch checkpoints to its own branch
(`kelmoir/cafa6-esm@trial-NNN`) so progress survives the VM teardown; only a config confirmed HEALTHY
at target scale gets promoted via `--push-to-hub` to `kelmoir/cafa6-esm@main`. Each loop iteration
ends in a git commit (no push — no remote is configured on this repo).

## Data science conventions in this repo

- Kaggle competition citation/description text (used in `DatasetInfo`) should be preserved verbatim
  when editing `generate_dataset.py` — it's the official competition citation.
- Random seeds are fixed at `42` throughout (dataset shuffling/splitting, `torch.manual_seed`) for
  reproducibility — keep new randomized code seeded consistently.
- Train/val/test protein ID splits are done by unique `EntryID` (not by row) before dataset
  generation, to keep all annotations for a given protein in the same split.
