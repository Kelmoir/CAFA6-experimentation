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

A local **conda** environment lives at `env/` (not committed) — created with something like
`conda create --prefix ./env`, not a plain `venv`/`virtualenv` and not managed by `uv` (confirmed via
`conda info --envs`, which lists this path). `source env/bin/activate` happens to work for it (conda
environments created with `--prefix` get an `activate` script too), but prefer running scripts via the
env's own interpreter directly (`env/bin/python script.py`) when a full shell activation isn't needed —
and if the env ever needs recreating, use `conda create --prefix ./env`/`conda install`, not `venv`/`uv`.

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

# Launch a ~1h regularization-sweep trial on a Colab T4 (see experiments/ancestor_sweep/sweep_log.md).
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

**Known env gotcha**: `datasets`/`huggingface_hub`'s push/upload calls (`push_to_hub`, `HfApi.upload_file`)
can fail with `ImportError: cannot import name 'XetSession' from 'hf_xet'` on this host — a version
mismatch between the installed `huggingface_hub` and `hf_xet` packages, not a code bug. Fix: set
`HF_HUB_DISABLE_XET=1` in the environment before running the push (falls back to the older, non-xet
upload path). `training_run.py` sets this itself now (`os.environ.setdefault("HF_HUB_DISABLE_XET", "1")`,
placed before `import huggingface_hub` since the constant is read at import time, not lazily at push
time) — its own checkpoint pushes don't need this set manually anymore. `generate_dataset*.py`'s dataset
pushes still do (they don't import `training_run.py`), e.g. `HF_HUB_DISABLE_XET=1 env/bin/python
generate_dataset_full.py`.

## Experiments directory layout

`experiments/` was reorganized 2026-09-04 to separate two large, closed-out bodies of work from
active/current material, since both had accumulated hundreds of scripts/logs/notebooks at the
top level:

- `experiments/ancestor_sweep/` — the beam-search regularization sweep (`sweep_log.md`, trial
  logs/notebooks/manifests, `beam_search_state.json`, `results_index.csv`) and its dedicated
  `scripts/`/`logs/` subfolders. This is the sweep that trained against the ancestor-propagated
  coarse (383-term, depth≤2) label space — see the Architecture notes' label-space caveat below.
  Still referenced as historical context/precedent (e.g. `training_run.py`'s
  `--unknown-target-value` flag cites its trial-080/081 finding), just no longer active work.
- `experiments/mf_span/` — the shelved positional-molecular-function side experiment (see
  `experiments/mf_span/mf_span_postmortem.md` for why it was shelved), covering both its data-prep
  phase (`positional_mf_*.json`, `llm_term_review/`) and training phase (`mf_span_*.md`), plus its
  dedicated `scripts/`/`logs/` subfolders.
- Everything else directly under `experiments/` (`dataset_refinement_pilot.py`,
  `analyze_go_term_pareto.py`, `sequential_cascade_plan.md`, `graph_embedding_plan.md`,
  `check_label_encoding.py`, `smoke_test.py`, and `experiments/logs/`) is current/active material,
  working directly with the real human-annotated CAFA-6 label space rather than the old
  ancestor-propagated coarse one.

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
- `data/uniprot/uniprot_sprot.dat.gz` — bulk Swiss-Prot flat-file dump (git-ignored, ~700MB),
  fetched once via `experiments/mf_span/scripts/fetch_uniprot_positional_features.py` for the positional
  molecular_function sub-project (localizing GO MF terms to sequence spans within the
  `experiments/mf_span/positional_mf_*` working set — a side experiment, not part of the main
  dataset/training pipeline above). Streamed record-by-record with `Bio.SwissProt.parse()`, never
  fully decompressed to disk/memory at once, re-parsed locally (no re-download) whenever the
  extraction script's captured fields change.
- `data/go_mappings/{prosite2go,pfam2go,interpro2go}.txt` — GO Consortium's bulk `*2go` cross-
  reference tables (git-ignored, one `curl` each from
  `https://current.geneontology.org/ontology/external2go/<name>`; unreachable via `WebFetch`,
  403s there for unclear reasons — plain `curl` works). Used by
  `experiments/mf_span/scripts/build_positional_mf_links.py` to map a DOMAIN feature's bridged
  PROSITE/Pfam/InterPro accession to a candidate GO term. Note the accession-system mismatch this
  cost real debugging time to find: `prosite2go` keys on `PS#####` pattern/profile accessions
  only, never the `PRU#####` ProRule IDs UniProt's DOMAIN `evidence` qualifier carries — the
  correct `PS#####` accession lives in the separate `record.cross_references` entries instead
  (captured as `domain_xrefs` by the fetch script).
- `models/` — used transiently by `push_checkpoint()` as a local staging dir (`models/<run_name>/checkpoint`)
  before uploading to the Hub; not git-ignored in `.gitignore` today despite being treated as such —
  on a Colab VM this never touches the local sandbox host anyway, so it doesn't matter in practice.

## Architecture notes

- **Label space**: `training_run.py` classifies proteins over a set of **coarse GO terms** (not just
  the 3 top-level aspects) as a multi-label problem — `id2label`/`label2id` are derived from
  `dataset["train"].features["labels"].feature.names`, so the script itself is label-space-agnostic;
  what the label space actually contains is decided entirely in `generate_dataset.py`. That script
  loads `go-basic.obo` with `obonet`, restricts propagation to `is_a`/`part_of` edges (the two
  relationship types GO's true-path rule supports — `regulates`/`positively_regulates`/
  `negatively_regulates` are excluded from label propagation, though preserved in the exported
  dependency graph below), and propagates every protein's asserted `train_terms.tsv` annotations to
  their full ancestor closure (necessary: ~98% of a protein's implied ancestor annotations aren't
  spelled out as explicit rows in the raw file). The label vocabulary is the 3 aspect roots (depth 0,
  informationally equivalent to the old aspect-only task) plus every term within `COARSE_MAX_DEPTH`
  (2) hops of its root with at least `COARSE_MIN_ANNOTATIONS` (50) proteins after propagation — 383
  terms as of the last regeneration. `generate_dataset.py` also writes
  `data/cafa-6-protein-function-prediction/dataset/go_dependency_graph.json`: the full GO graph (all
  5 relationship types, not just the propagation-safe 2) plus the coarse-term selection, so a future
  curriculum stage can look up any coarse term's descendants directly without recomputing the graph.
  **Note**: the beam-search sweep documented in `experiments/ancestor_sweep/sweep_log.md` (trial-001 through
  trial-049) predates this change and was run against the old 3-aspect-only label space — its
  hyperparameter conclusions (regularization, batch size, etc.) don't automatically transfer to the
  richer label space, per the project's own "re-run from scratch at a new scale" precedent.
- **Model**: ESM-2 (`facebook/esm2_t*`) loaded via `AutoModelForSequenceClassification` with
  `problem_type="multi_label_classification"`. Default mode: only the classifier head plus the
  last `--unfreeze-last-n` encoder layers are trainable; the rest of the ESM backbone is frozen
  (`create_esm` in `training_run.py`). **`--use-lora`** (added for the scale-up phase, see
  "Colab regularization sweep" below) switches to LoRA fine-tuning via `peft` instead: the entire
  backbone is frozen and small rank-`--lora-r` adapters are injected into `query`/`value`
  attention projections (`--lora-target-modules` to override), with `modules_to_save=["classifier"]`
  keeping the (newly-initialized, not pretrained) classification head fully trainable alongside
  the adapters. This is what makes a much bigger model (e.g. `esm2_t36_3B_UR50D`) fit in limited
  VRAM — only the adapters + classifier need gradient/optimizer memory, not the frozen backbone.
  Two non-obvious things verified directly against a real `peft` install (not just assumed from
  docs, since the exact internal structure is version-dependent): (1) `modules_to_save` wraps the
  classifier in a `ModulesToSaveWrapper` that changes its attribute path
  (`classifier.modules_to_save.default.out_proj.bias`, with a frozen `classifier.original_module...`
  sibling) — `get_classifier_bias_param()`/`set_classifier_dropout()` locate it by
  parameter/module-name suffix instead of a hardcoded attribute chain, so they work whether or not
  `--use-lora` is set; (2) when the frozen base loads in reduced precision via `--mixed-precision`
  (`torch_dtype=` at `from_pretrained` time), `peft` automatically upcasts LoRA adapter weights to
  fp32 for training stability, but `modules_to_save`'s classifier copy has no such special-casing
  and silently inherits the reduced dtype — `create_esm` explicitly upcasts it back to fp32 after
  wrapping (negligible VRAM cost, avoids AdamW updates getting rounded away on a bf16/fp16
  classifier with no fp32 master copy). A `--use-lora` checkpoint pushed to the Hub is adapter-only
  (a few MB), not a full standalone model — `--resume-from-revision` and
  `experiments/ancestor_sweep/scripts/diagnostic_eval_stage2.py` both branch on this (loading the base architecture
  fresh from `--model-name`/the trial's recorded `model_name`, then attaching the adapter via
  `PeftModel.from_pretrained`), never treating the checkpoint repo itself as a loadable full model
  the way the non-LoRA path does. `--mixed-precision {fp16,bf16}` (autocast + `GradScaler` for fp16)
  and `--gradient-checkpointing` (with the peft-required `model.enable_input_require_grads()`) are
  independent flags, useful with or without `--use-lora`, but especially necessary to fit a 3B-scale
  model in a capped VRAM budget at all. **Important cost caveat found via trial-067**: LoRA only
  removes the frozen backbone's optimizer/gradient *memory* cost, not its forward/backward *compute*
  cost — per-example wall-clock at `esm2_t36_3B_UR50D` scale was ~138x slower than the
  `esm2_t12_35M_coarse_go_terms` phase's trials (far more than the ~81x parameter-count ratio alone
  predicts), so a similar per-trial data budget (12k examples/epoch) isn't affordable at 3B scale.
  `--lora-layers-to-transform N` (restrict adapters to the last N layers instead of every layer) is a
  real, verified lever for this: peft's `layers_to_transform` combined with standard PyTorch autograd
  behavior means the untouched earlier layers' output activations have `requires_grad=False` (verified
  directly against peft 0.19.1, not assumed), so autograd builds no backward graph for them at all —
  forward compute still runs for every layer, but backward, typically the more expensive half, only
  runs for the last N. Trades adaptation breadth for real wall-clock savings.
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
  Per-trial `experiments/ancestor_sweep/logs/trial-NNN.log` plus the `trial_metrics.json` pushed alongside each
  Hub checkpoint are the actual source of truth for the sweep (see "Colab regularization sweep").
- **Colab bootstrapping**: `training_run.py` detects `google.colab` in `sys.modules` and pip-installs
  missing heavy dependencies at import time — this is intentional for the Colab-first workflow, don't
  "fix" it into a hard requirements dependency.
- **Post-training diagnostic (`--run-diagnostic`)**: `compute_metrics`'s threshold-0.5 accuracy/recall
  can look like a collapsed, non-learning model under this task's severe class imbalance (3.58%
  average positive rate across 383 labels) even when the model has learned real per-label ranking
  signal — trial-050's false alarm, where `val_recall` stayed pinned near 3% despite a real,
  well-ranking model underneath (found via a one-off post-hoc AUROC check). `run_diagnostic()` makes
  that check a built-in, reusable part of `training_run.py` instead: after training finishes, it
  computes per-label AUROC (threshold-independent — the aggregate is reported as a distribution,
  mean/median/min/max, not printed 383-wide) plus pooled precision/recall/F1 at threshold 0.5 and a
  best-F1 threshold sweep, all against the model **already resident in memory** (no Hub
  re-download/reload). `--diagnostic-max-val-samples` (default 2000) is independent of
  `--max-val-samples`, which governs the training loop's own per-epoch validation pass and may be set
  far smaller for a quick calibration trial. `build_notebook.py` appends `--run-diagnostic`
  automatically to every trial's args, so every future manual-notebook trial gets a `[DIAGNOSTIC]`
  line for free, on whatever GPU it trained on. `experiments/ancestor_sweep/scripts/diagnostic_eval_stage2.py` still
  exists as a standalone script, now refactored to call the shared `run_diagnostic()` rather than
  duplicating the inference/metrics logic — it's for post-hoc re-diagnosis of a checkpoint trained
  *before* `--run-diagnostic` existed (e.g. trial-071/trial-072), where the model has to be
  re-downloaded from the Hub. It has no GPU access in this sandbox, so it runs on CPU only — verified
  directly to be dramatically slow for anything above ~100M params (a single trial-071 diagnostic run,
  650M params, took most of a working session to finish on CPU) — another reason `--run-diagnostic`
  running on the GPU that just trained the model is the preferred path going forward.

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

An agent-driven loop (see `experiments/ancestor_sweep/sweep_log.md`) launches ~1h training trials on a Colab T4 (via
the `colab new`/`upload`/`exec`/`stop` sequence described in "Colab CLI" above, run from inside the
Podman sandbox container), judges each trial's train/val curves against a fixed rubric, and iterates
toward regularization/unfreezing settings that train reliably (manual attempts had consistently
either overfit or failed to learn at all). `experiments/ancestor_sweep/sweep_log.md` is the source of truth: it pins
the judgment rubric and search strategy at the top, then logs one section per trial (config, per-epoch
metrics, verdict, next-config decision, Hub checkpoint branch, commit sha). `experiments/ancestor_sweep/logs/`
holds each trial's captured stdout/stderr; `experiments/smoke_test.py` is a throwaway infra check
(GPU availability, Colab Secrets reachability) run once before trusting the pipeline for real trials —
it confirmed GPU access works but **Colab Secrets are not reachable from CLI-provisioned sessions**
(`"Secrets can only be fetched when running from the Colab UI"`), which is why every trial uploads the
HF token as a file rather than relying on `google.colab.userdata`. trackio-Space tracking
(`--trackio-space-id`) is currently **not used** — HF's free-Space policy changes made it a non-option
for now — so `experiments/ancestor_sweep/logs/trial-NNN.log` and each checkpoint's `trial_metrics.json` are the sole
record of a trial's curves. `training_run.py`'s `--checkpoint-mode` defaults to `final`: exactly one
commit gets pushed per trial, to that trial's own branch (`kelmoir/cafa6-esm@trial-NNN`), after the
last completed epoch — cheap re-runs (~40min) don't justify littering the Hub with a commit per
epoch, and since there's only ever one commit on the branch, the branch name alone identifies the
checkpoint (no SHA needed). The one exception is the CLI self-healing retry path
(`run_trial.sh`/`make_wrapper.py`), which forces `--checkpoint-mode every-epoch` because its
`--resume-from-revision` retry logic needs a checkpoint to resume from after a connection drop
mid-trial. Only a config confirmed HEALTHY at target scale gets promoted via `--push-to-hub` to
`kelmoir/cafa6-esm@main`. Each loop iteration ends in a git commit (no push — no remote is configured
on this repo).

## Data science conventions in this repo

- Kaggle competition citation/description text (used in `DatasetInfo`) should be preserved verbatim
  when editing `generate_dataset.py` — it's the official competition citation.
- Random seeds are fixed at `42` throughout (dataset shuffling/splitting, `torch.manual_seed`) for
  reproducibility — keep new randomized code seeded consistently.
- Train/val/test protein ID splits are done by unique `EntryID` (not by row) before dataset
  generation, to keep all annotations for a given protein in the same split.
