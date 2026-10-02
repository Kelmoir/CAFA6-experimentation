# Ensembling Stacking Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single global XGBoost meta-model that fuses trial-013's ESM-2
per-(protein, term) probability with the BLAST-KNN retrieval score (plus a
term-frequency feature), and confirm whether the fused score beats trial-013 alone on
held-out per-label AUROC.

**Architecture:** One new script, `experiments/ensembling/stacking_pipeline.py`. It
(1) re-derives the already-cached meta-eval score matrices (validation rows 0:2000,
reusing `blast_knn_probe_real.py`'s cache files — no recompute), (2) computes a new
meta-train slice (validation rows 2000:4000: new BLAST+model-inference run, same
resumable-checkpoint pattern), (3) flattens both into row-per-(protein, term) feature
vectors `[esm_prob, blast_score, log1p(term_freq)]`, (4) fits one global
`xgboost.XGBClassifier` on meta-train, (5) evaluates on meta-eval using the exact
per-label-AUROC methodology every prior probe in this line of work used.

**Tech Stack:** Python, numpy, `xgboost` (3.2.0, already in `env/`), scikit-learn
(`roc_auc_score`, already in `env/`), the existing BLAST+ 2.17.0 install in
`./blast_env/`, `training_run.py`'s inference helpers (reused via
`blast_knn_probe_real.py`, not reimplemented). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-02-ensembling-stacking-pipeline-design.md`

## Global Constraints

- No pytest/test framework in this repo (CLAUDE.md). Verification is standalone
  assertion scripts run via `env/bin/python <script>.py`, red (fails for the right
  reason) before green, per the spec's Testing section.
- Seed 42 throughout (`random_state=42` on the XGBoost model), per this repo's
  established convention.
- `xgboost` 3.2.0 is already installed in `env/` — confirmed directly
  (`env/bin/python -c "import xgboost; print(xgboost.__version__)"` → `3.2.0`). Do not
  add it to any requirements file as a new dependency; it's already there.
- BLAST+ binaries live in `./blast_env/bin/`, not `./env/bin/` — always reference them
  via `BLAST_BIN_DIR` (imported from `blast_knn_probe_real.py`), never assume they're on
  `PATH`.
- Meta-eval = validation rows **0:2000** (already fully cached by
  `blast_knn_probe_real.py`; never recomputed, never used for fitting). Meta-train =
  validation rows **2000:4000** (new, computed by this plan's Task 2). These ranges
  must never overlap.
- The test split is out of scope entirely — every step in this plan stays inside the
  validation split.
- No GPU passthrough in this sandbox. BLAST and XGBoost are CPU-native; the one
  GPU-shaped step (ESM-2 forward inference) reuses the already-proven CPU inference
  path. Any step expected to run more than a couple of minutes runs in the
  **foreground** wrapped in `timeout`, never `run_in_background` (confirmed this
  session: background execution was severely throttled relative to foreground for this
  exact kind of CPU inference).
- `experiments/ensembling/blast_work/*` is gitignored — new cache files created there
  are never committed; only `stacking_pipeline.py` itself and the plan-doc log entry
  (Task 6) get committed.

## Review Focus

- **A validation protein with zero BLAST hits** (already possible — both existing
  probes leave such rows all-zero) must not produce NaN/inf once folded into the
  feature matrix fed to XGBoost. *(Task 3: explicit `np.isfinite` assertion on the
  constructed feature matrix.)*
- **A stale or shape-mismatched inference checkpoint** (e.g. left over from a run with
  a different row range or label vocabulary) must not be silently resumed as if it were
  valid — the existing `blast_knn_probe_real.py` pattern only checks file *existence*,
  not shape. *(Task 2: checkpoint shape is validated before resuming; mismatched
  checkpoints are discarded with a `[WARN]`, not trusted.)*
- **Meta-train and meta-eval row ranges overlapping** would leak evaluation rows into
  training and silently invalidate every reported number. *(Task 3: explicit assertion
  that `META_TRAIN_START >= MAX_VAL_SAMPLES` before any fitting happens.)*
- **The XGBoost fit collapsing to a near-constant predictor** under meta-train's heavy
  class imbalance (consistent with trial-013's low overall positive rate) would silently
  produce a stacked score with no real signal, indistinguishable from a working fit
  without an explicit check. *(Task 4: assert nonzero prediction variance on meta-train
  immediately after fitting; documented fallback is negative subsampling, not a
  different model.)*
- **The targeted "bright-spot" subset check reusing a different label filter than the
  headline numbers** would make the comparison not apples-to-apples (e.g. including a
  term that's degenerate in meta-eval but happened to show a ≥0.05 BLAST/model gap for
  spurious reasons). *(Task 5: the bright-spot mask is computed only over the same
  `evaluable` index set used for every other reported number, not recomputed
  independently.)*

---

## Task 1: Cached meta-eval loader + regression check

**Files:**
- Create: `experiments/ensembling/stacking_pipeline.py`
- Create: `experiments/ensembling/check_meta_eval_regression.py`

**Interfaces:**
- Consumes: `experiments/ensembling/blast_work/blast_results.tsv` and
  `model_probs_partial.npy` (already on disk from the earlier real-BLAST probe); the
  dataset at `data/cafa-6-protein-function-prediction/dataset_full`; `BLAST_BIN_DIR`,
  `CHECKPOINT_DIR`, `DATA_PATH`, `WORK_DIR`, `write_fasta`, `labels_to_multihot` imported
  from `blast_knn_probe_real.py` (same directory).
- Produces: `load_cached_meta_eval() -> dict` with keys `esm_prob` (shape
  `(2000, n_labels)`), `blast_score` (shape `(2000, n_labels)`), `val_labels` (shape
  `(2000, n_labels)`), `train_labels` (shape `(n_train, n_labels)`), `label_names`
  (`list[str]`, length `n_labels`). Later tasks import this function from
  `stacking_pipeline`.

- [ ] **Step 1: Write the failing regression check**

```python
# experiments/ensembling/check_meta_eval_regression.py
"""Standalone correctness check (no pytest in this repo -- see CLAUDE.md): confirms
load_cached_meta_eval() reproduces the exact numbers already logged in
experiments/cafa6_next_steps_plan.md for the real-BLAST probe, before any new BLAST/
inference compute is spent on the meta-train slice (Task 2+).

Run via: env/bin/python experiments/ensembling/check_meta_eval_regression.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from stacking_pipeline import load_cached_meta_eval

data = load_cached_meta_eval()
val_labels, esm_prob, blast_score = data["val_labels"], data["esm_prob"], data["blast_score"]

evaluable = [j for j in range(val_labels.shape[1]) if val_labels[:, j].min() != val_labels[:, j].max()]
assert len(evaluable) == 9401, f"n_labels_evaluable={len(evaluable)}, expected 9401"

model_auroc = np.array([roc_auc_score(val_labels[:, j], esm_prob[:, j]) for j in evaluable])
blast_auroc = np.array([roc_auc_score(val_labels[:, j], blast_score[:, j]) for j in evaluable])

assert abs(model_auroc.mean() - 0.8731) < 0.001, f"model mean AUROC {model_auroc.mean():.4f}, expected ~0.8731"
assert abs(blast_auroc.mean() - 0.6854) < 0.001, f"blast mean AUROC {blast_auroc.mean():.4f}, expected ~0.6854"

print("[PASS] meta-eval regression check: n_labels_evaluable, model/blast mean AUROC all match logged values")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd experiments/ensembling && ../../env/bin/python check_meta_eval_regression.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'stacking_pipeline'` (the file
doesn't exist yet).

- [ ] **Step 3: Write `load_cached_meta_eval()`**

```python
# experiments/ensembling/stacking_pipeline.py
"""Fuses trial-013's ESM-2 predictions with the BLAST-KNN retrieval score via a global
XGBoost meta-model, per experiments/cafa6_next_steps_plan.md Tier 1 item 1 and
docs/superpowers/specs/2026-10-02-ensembling-stacking-pipeline-design.md.
"""
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blast_knn_probe_real import (  # noqa: E402
    BLAST_BIN_DIR, CHECKPOINT_DIR, DATA_PATH, WORK_DIR,
    labels_to_multihot, write_fasta,
)

MAX_VAL_SAMPLES = 2000  # meta-eval slice: validation rows 0:2000 (matches run_diagnostic())
META_TRAIN_START = 2000
META_TRAIN_END = 4000


def load_cached_meta_eval():
    """Re-derives meta-eval's (protein x term) score matrices from the cache files
    blast_knn_probe_real.py already wrote -- no new BLAST/inference, just parsing."""
    dataset = load_from_disk(str(DATA_PATH))
    train_split, val_split = dataset["train"], dataset["validation"]
    label_names = [str(name) for name in train_split.features["labels"].feature.names]
    label2id = {name: i for i, name in enumerate(label_names)}

    val_subset = val_split.select(range(MAX_VAL_SAMPLES))
    val_ids = [f"val_{i}" for i in range(MAX_VAL_SAMPLES)]
    train_ids = [f"train_{i}" for i in range(len(train_split))]
    train_id_to_idx = {tid: i for i, tid in enumerate(train_ids)}

    train_labels = labels_to_multihot(train_split["labels"], label_names, label2id)
    val_labels = labels_to_multihot(val_subset["labels"], label_names, label2id)

    blast_out = WORK_DIR / "blast_results.tsv"
    hits_by_query = {qid: [] for qid in val_ids}
    with open(blast_out) as f:
        for line in f:
            qseqid, sseqid, bitscore, pident = line.strip().split("\t")
            hits_by_query[qseqid].append((sseqid, float(bitscore)))

    blast_score = np.zeros_like(val_labels)
    for i, qid in enumerate(val_ids):
        hits = hits_by_query.get(qid, [])
        if not hits:
            continue
        weights = np.array([b for _, b in hits], dtype=np.float32)
        neighbor_idx = [train_id_to_idx[sseqid] for sseqid, _ in hits]
        blast_score[i] = (train_labels[neighbor_idx] * weights[:, None]).sum(axis=0) / weights.sum()

    esm_prob = np.load(WORK_DIR / "model_probs_partial.npy")
    assert esm_prob.shape == val_labels.shape, (
        f"cached model_probs_partial.npy shape {esm_prob.shape} != val_labels shape {val_labels.shape}"
    )

    return {
        "esm_prob": esm_prob, "blast_score": blast_score, "val_labels": val_labels,
        "train_labels": train_labels, "label_names": label_names,
    }
```

- [ ] **Step 4: Run the check again to verify it passes**

Run: `cd experiments/ensembling && ../../env/bin/python check_meta_eval_regression.py`
Expected: `[PASS] meta-eval regression check: n_labels_evaluable, model/blast mean AUROC all match logged values`

- [ ] **Step 5: Commit**

```bash
git add experiments/ensembling/stacking_pipeline.py experiments/ensembling/check_meta_eval_regression.py
git commit -m "feat: add cached meta-eval loader for the stacking pipeline

Re-derives the real-BLAST probe's (protein x term) score matrices from its
already-written cache files, with a regression check confirming the numbers
match what's already logged in cafa6_next_steps_plan.md before any new
BLAST/inference compute is spent.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Task 2: Meta-train slice collection (new BLAST run + resumable model inference)

**Files:**
- Modify: `experiments/ensembling/stacking_pipeline.py`
- Create: `experiments/ensembling/check_metatrain_smoke.py`

**Interfaces:**
- Consumes: `load_from_disk`, `labels_to_multihot`, `write_fasta`, `BLAST_BIN_DIR`,
  `CHECKPOINT_DIR`, `DATA_PATH`, `WORK_DIR` (all already in `stacking_pipeline.py` from
  Task 1); `training_run.preprocess_batch`/`data_collate_function` (imported the same
  way `blast_knn_probe_real.py` imports them).
- Produces: `compute_meta_train_slice(start, end) -> dict` with the same four keys as
  `load_cached_meta_eval()` (`esm_prob`, `blast_score`, `val_labels`, `train_labels`,
  `label_names`), but for validation rows `[start:end)`. Task 3 calls this with
  `(META_TRAIN_START, META_TRAIN_END)` for the real run.

- [ ] **Step 1: Write the failing smoke test (tiny 10-row range, fast)**

```python
# experiments/ensembling/check_metatrain_smoke.py
"""Standalone smoke check (no pytest in this repo): runs compute_meta_train_slice() on
a tiny 10-row range so BLAST+inference finish in seconds, to validate the function's
wiring before spending real wall-clock on the full 2000-row meta-train range (Task 3's
real run). Run via:
    env/bin/python experiments/ensembling/check_metatrain_smoke.py
"""
import numpy as np

from stacking_pipeline import MAX_VAL_SAMPLES, compute_meta_train_slice

start, end = MAX_VAL_SAMPLES, MAX_VAL_SAMPLES + 10
result = compute_meta_train_slice(start, end)

n_rows = end - start
n_labels = len(result["label_names"])
assert result["esm_prob"].shape == (n_rows, n_labels), result["esm_prob"].shape
assert result["blast_score"].shape == (n_rows, n_labels), result["blast_score"].shape
assert result["val_labels"].shape == (n_rows, n_labels), result["val_labels"].shape
assert np.isfinite(result["esm_prob"]).all(), "non-finite value in smoke esm_prob"
assert np.isfinite(result["blast_score"]).all(), "non-finite value in smoke blast_score"

print(f"[PASS] metatrain smoke check: shapes and finiteness OK for rows [{start}:{end})")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd experiments/ensembling && ../../env/bin/python check_metatrain_smoke.py`
Expected: FAIL with `ImportError: cannot import name 'compute_meta_train_slice'`.

- [ ] **Step 3: Write `compute_meta_train_slice()`**

```python
# append to experiments/ensembling/stacking_pipeline.py
def compute_meta_train_slice(start, end):
    """Builds (protein x term) score matrices for validation rows [start:end), reusing
    the already-built BLAST database and the resumable checkpoint-every-20-batches
    inference pattern blast_knn_probe_real.py validated. Cache filenames are keyed by
    (start, end) so a smoke-test range never collides with the real 2000:4000 run."""
    import subprocess
    from functools import partial

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from training_run import data_collate_function, preprocess_batch  # noqa: E402

    dataset = load_from_disk(str(DATA_PATH))
    train_split, val_split = dataset["train"], dataset["validation"]
    label_names = [str(name) for name in train_split.features["labels"].feature.names]
    label2id = {name: i for i, name in enumerate(label_names)}

    val_subset = val_split.select(range(start, end))
    n_rows = end - start
    val_ids = [f"metatrain_{i}" for i in range(n_rows)]
    train_ids = [f"train_{i}" for i in range(len(train_split))]
    train_id_to_idx = {tid: i for i, tid in enumerate(train_ids)}

    train_labels = labels_to_multihot(train_split["labels"], label_names, label2id)
    val_labels = labels_to_multihot(val_subset["labels"], label_names, label2id)

    tag = f"{start}_{end}"
    metatrain_fasta = WORK_DIR / f"metatrain_{tag}.fasta"
    blast_db = WORK_DIR / "train_blastdb"
    blast_out = WORK_DIR / f"blast_results_metatrain_{tag}.tsv"

    if blast_out.exists():
        print(f"[INFO] Reusing cached meta-train BLAST results at {blast_out}")
    else:
        write_fasta(metatrain_fasta, val_ids, val_subset["sequence"])
        print(f"[INFO] Running blastp for rows [{start}:{end}) against the existing train_blastdb...")
        subprocess.run(
            [str(BLAST_BIN_DIR / "blastp"), "-query", str(metatrain_fasta), "-db", str(blast_db),
             "-out", str(blast_out), "-outfmt", "6 qseqid sseqid bitscore pident",
             "-max_target_seqs", "10", "-num_threads", "8"],
            check=True, capture_output=True, text=True,
        )

    hits_by_query = {qid: [] for qid in val_ids}
    with open(blast_out) as f:
        for line in f:
            qseqid, sseqid, bitscore, pident = line.strip().split("\t")
            hits_by_query[qseqid].append((sseqid, float(bitscore)))

    blast_score = np.zeros_like(val_labels)
    for i, qid in enumerate(val_ids):
        hits = hits_by_query.get(qid, [])
        if not hits:
            continue
        weights = np.array([b for _, b in hits], dtype=np.float32)
        neighbor_idx = [train_id_to_idx[sseqid] for sseqid, _ in hits]
        blast_score[i] = (train_labels[neighbor_idx] * weights[:, None]).sum(axis=0) / weights.sum()

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model.eval()

    preprocess_partial = partial(
        preprocess_batch, tokenizer=tokenizer, label2id=label2id,
        unique_items=label_names, max_length=4192,
    )
    val_subset_proc = val_subset.with_transform(transform=preprocess_partial)
    batch_size = 4
    loader = torch.utils.data.DataLoader(
        val_subset_proc, batch_size=batch_size, shuffle=False,
        collate_fn=partial(data_collate_function, tokenizer=tokenizer),
    )

    probs_ckpt = WORK_DIR / f"model_probs_metatrain_{tag}_partial.npy"
    n_done_ckpt = WORK_DIR / f"model_probs_metatrain_{tag}_n_done.txt"
    expected_shape = (n_rows, len(label_names))
    if probs_ckpt.exists() and n_done_ckpt.exists():
        cached = np.load(probs_ckpt)
        if cached.shape == expected_shape:
            esm_prob = cached
            n_done = int(n_done_ckpt.read_text())
            print(f"[INFO] Resuming meta-train inference from checkpoint: {n_done}/{n_rows} done")
        else:
            print(
                f"[WARN] cached checkpoint shape {cached.shape} != expected {expected_shape} "
                "-- discarding stale checkpoint, starting fresh"
            )
            esm_prob = np.zeros(expected_shape, dtype=np.float32)
            n_done = 0
    else:
        esm_prob = np.zeros(expected_shape, dtype=np.float32)
        n_done = 0

    n_batches_done = n_done // batch_size
    with torch.inference_mode():
        for batch_i, batch in enumerate(loader):
            if batch_i < n_batches_done:
                continue
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            probs = torch.sigmoid(logits).numpy()
            batch_start = batch_i * batch_size
            esm_prob[batch_start:batch_start + len(probs)] = probs
            n_done = batch_start + len(probs)
            if batch_i % 20 == 0:
                np.save(probs_ckpt, esm_prob)
                n_done_ckpt.write_text(str(n_done))
                print(f"[INFO] meta-train inference progress: {n_done}/{n_rows}", flush=True)
    np.save(probs_ckpt, esm_prob)
    n_done_ckpt.write_text(str(n_done))

    return {
        "esm_prob": esm_prob, "blast_score": blast_score, "val_labels": val_labels,
        "train_labels": train_labels, "label_names": label_names,
    }
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `cd experiments/ensembling && ../../env/bin/python check_metatrain_smoke.py`
Expected: `[PASS] metatrain smoke check: shapes and finiteness OK for rows [2000:2010)`
(takes well under a minute — 10 queries/sequences).

- [ ] **Step 5: Run the real meta-train collection (2000 rows) in the foreground**

This is the expensive step the smoke test de-risked — run it directly, not as another
test iteration. Follow this session's validated pattern: foreground, `timeout`-wrapped,
resumable, re-invoked until it finishes (the checkpoint picks up where the last
invocation left off).

Run: `cd experiments/ensembling && timeout 550 ../../env/bin/python -c "from stacking_pipeline import compute_meta_train_slice, META_TRAIN_START, META_TRAIN_END; compute_meta_train_slice(META_TRAIN_START, META_TRAIN_END)"`

Expected: prints `[INFO] meta-train inference progress: N/2000` lines; re-run the exact
same command if it hits the timeout, until it completes without printing progress lines
anymore (i.e. the function returns). The `blastp` sub-step (first invocation only) may
itself take a few minutes before any progress lines appear — that's expected, matching
the original real-BLAST probe's behavior.

- [ ] **Step 6: Commit**

```bash
git add experiments/ensembling/stacking_pipeline.py experiments/ensembling/check_metatrain_smoke.py
git commit -m "feat: add meta-train slice collection (new BLAST run + resumable inference)

compute_meta_train_slice(start, end) builds the (protein x term) score
matrices for a given validation row range, reusing the existing BLAST
database and the checkpoint-every-20-batches inference pattern. Checkpoint
shape is validated before resuming so a stale/mismatched cache file can't
silently corrupt results.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Task 3: Feature flattening

**Files:**
- Modify: `experiments/ensembling/stacking_pipeline.py`
- Create: `experiments/ensembling/check_build_features.py`

**Interfaces:**
- Consumes: the `dict` shape produced by both `load_cached_meta_eval()` and
  `compute_meta_train_slice()` (Tasks 1–2) — any caller can pass either's return value.
- Produces: `build_features(slice_dict) -> (X, y)` where `X` is `(n_rows * n_labels, 3)`
  float32 (`[esm_prob, blast_score, log1p(term_freq)]` per row) and `y` is
  `(n_rows * n_labels,)` float32. Task 4 consumes `X`/`y` directly for fitting; Task 5
  consumes them for evaluation.

- [ ] **Step 1: Write the failing feature-construction check**

```python
# experiments/ensembling/check_build_features.py
"""Standalone check (no pytest in this repo): validates build_features() on the cheap
meta-eval slice (already cached, no new compute) before it's trusted on meta-train's
~54M-row flatten. Run via:
    env/bin/python experiments/ensembling/check_build_features.py
"""
import numpy as np

from stacking_pipeline import build_features, load_cached_meta_eval

meta_eval = load_cached_meta_eval()
X, y = build_features(meta_eval)

n_rows, n_labels = meta_eval["val_labels"].shape
assert X.shape == (n_rows * n_labels, 3), X.shape
assert y.shape == (n_rows * n_labels,), y.shape
assert np.isfinite(X).all(), "non-finite value in constructed features"
assert set(np.unique(y).tolist()) <= {0.0, 1.0}, "y should be binary 0/1"

print(f"[PASS] build_features check: X.shape={X.shape}, y.shape={y.shape}, all finite, y binary")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd experiments/ensembling && ../../env/bin/python check_build_features.py`
Expected: FAIL with `ImportError: cannot import name 'build_features'`.

- [ ] **Step 3: Write `build_features()` and the non-overlap guard**

```python
# append to experiments/ensembling/stacking_pipeline.py
def build_features(slice_dict):
    """Flattens a (protein x term) slice dict (from load_cached_meta_eval() or
    compute_meta_train_slice()) into row-per-(protein, term) features:
    [esm_prob, blast_score, log1p(term_freq)]."""
    term_freq = np.log1p(slice_dict["train_labels"].sum(axis=0)).astype(np.float32)
    n_rows, n_labels = slice_dict["val_labels"].shape
    X = np.stack([
        slice_dict["esm_prob"].ravel(),
        slice_dict["blast_score"].ravel(),
        np.tile(term_freq, n_rows),
    ], axis=1).astype(np.float32)
    y = slice_dict["val_labels"].ravel().astype(np.float32)
    assert np.isfinite(X).all(), "non-finite value in constructed features"
    return X, y


def assert_no_overlap():
    assert META_TRAIN_START >= MAX_VAL_SAMPLES, (
        f"meta-train range [{META_TRAIN_START}:{META_TRAIN_END}) overlaps meta-eval's "
        f"[0:{MAX_VAL_SAMPLES}) rows -- would leak eval rows into training"
    )
```

- [ ] **Step 4: Run the check again to verify it passes**

Run: `cd experiments/ensembling && ../../env/bin/python check_build_features.py`
Expected: `[PASS] build_features check: X.shape=(54382000, 3), y.shape=(54382000,), all finite, y binary`
(exact first number is `2000 * n_labels`; n_labels is 27,191 per the plan doc, giving
54,382,000 — confirm the printed number matches `2000 * len(meta_eval["label_names"])`
rather than hardcoding it as a pass/fail criterion).

- [ ] **Step 5: Commit**

```bash
git add experiments/ensembling/stacking_pipeline.py experiments/ensembling/check_build_features.py
git commit -m "feat: add feature flattening for the stacking meta-model

build_features() turns a (protein x term) score-matrix slice into
row-per-(protein, term) [esm_prob, blast_score, log1p(term_freq)] features,
with an explicit finiteness check and a non-overlap guard between the
meta-train and meta-eval row ranges.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Task 4: Fit the XGBoost meta-model

**Files:**
- Modify: `experiments/ensembling/stacking_pipeline.py`
- Create: `experiments/ensembling/check_fit_degeneracy.py`

**Interfaces:**
- Consumes: `X, y` from `build_features()` (Task 3).
- Produces: `fit_meta_model(X_train, y_train) -> xgboost.XGBClassifier`. Task 5 calls
  `.predict_proba(X_eval)` on the returned model.

- [ ] **Step 1: Write the failing degeneracy check (on a cheap slice, not real meta-train)**

```python
# experiments/ensembling/check_fit_degeneracy.py
"""Standalone check (no pytest in this repo): fits the meta-model on the cheap,
already-cached meta-eval slice (not real meta-train -- this is a wiring/degeneracy
check, not the real fit) and confirms predictions aren't collapsed to a constant.
Run via:
    env/bin/python experiments/ensembling/check_fit_degeneracy.py
"""
from stacking_pipeline import build_features, fit_meta_model, load_cached_meta_eval

meta_eval = load_cached_meta_eval()
X, y = build_features(meta_eval)

model = fit_meta_model(X, y)
preds = model.predict_proba(X)[:, 1]
print(f"[PASS] fit_meta_model check: predictions std={preds.std():.4f} (nonzero, not collapsed)")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd experiments/ensembling && ../../env/bin/python check_fit_degeneracy.py`
Expected: FAIL with `ImportError: cannot import name 'fit_meta_model'`.

- [ ] **Step 3: Write `fit_meta_model()`**

```python
# append to experiments/ensembling/stacking_pipeline.py
def fit_meta_model(X_train, y_train):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist",
        eval_metric="auc", random_state=42,
    )
    model.fit(X_train, y_train)

    train_preds = model.predict_proba(X_train)[:, 1]
    assert train_preds.std() > 1e-6, (
        "meta-model predictions collapsed to a near-constant value on its own training "
        "data -- likely needs negative subsampling (see the design spec's Meta-model "
        "risk note) before being trusted on the real meta-train set"
    )
    return model
```

- [ ] **Step 4: Run the check again to verify it passes**

Run: `cd experiments/ensembling && ../../env/bin/python check_fit_degeneracy.py`
Expected: `[PASS] fit_meta_model check: predictions std=<some nonzero number> (nonzero, not collapsed)`

- [ ] **Step 5: Fit on the real meta-train set and watch for the same degeneracy risk at scale**

This runs against the real ~54M-row meta-train set from Task 2 — if `fit_meta_model`
raises its `AssertionError` here (unlike on the small check above), apply the documented
fallback before re-running: downsample negative rows (keep all `y_train == 1` rows,
randomly sample negatives at a fixed ratio e.g. 20:1) and re-fit, rather than switching
model families.

- [ ] **Step 6: Commit**

```bash
git add experiments/ensembling/stacking_pipeline.py experiments/ensembling/check_fit_degeneracy.py
git commit -m "feat: add XGBoost meta-model fitting with a degeneracy guard

fit_meta_model() fits a global XGBClassifier over [esm_prob, blast_score,
log1p(term_freq)] and asserts its own training predictions aren't collapsed
to a near-constant value, so a degenerate fit under meta-train's class
imbalance surfaces immediately instead of silently producing a meaningless
stacked score.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Task 5: Evaluate on meta-eval, targeted subset check, and wire up `main()`

**Files:**
- Modify: `experiments/ensembling/stacking_pipeline.py`

**Interfaces:**
- Consumes: `fit_meta_model`'s returned model, `build_features()`'s `X_eval`,
  `load_cached_meta_eval()`'s `meta_eval` dict (for `val_labels`/`esm_prob`/
  `blast_score`, needed to recompute the per-label AUROC baselines alongside the
  stacked score).
- Produces: `report_results(model, X_eval, meta_eval)` (prints `[RESULT]` lines, no
  return value needed by anything else) and a `main()` that runs the full pipeline
  end-to-end, matching this script's role as the final deliverable (there's no further
  task consuming its output programmatically).

- [ ] **Step 1: Write `report_results()` and `main()`**

No separate failing-test step here — this task's correctness is pinned by the Review
Focus item about reusing the same `evaluable` filter for the bright-spot subset, which
the assertion below enforces directly (reusing the identical list, not a second
independently-computed one).

```python
# append to experiments/ensembling/stacking_pipeline.py
def report_results(model, X_eval, meta_eval):
    from sklearn.metrics import roc_auc_score

    val_labels = meta_eval["val_labels"]
    n_rows, n_labels = val_labels.shape
    stacked_scores = model.predict_proba(X_eval)[:, 1].reshape(n_rows, n_labels)

    evaluable = [j for j in range(n_labels) if val_labels[:, j].min() != val_labels[:, j].max()]
    assert len(evaluable) == 9401, f"n_labels_evaluable={len(evaluable)}, expected 9401"

    model_auroc = np.array([roc_auc_score(val_labels[:, j], meta_eval["esm_prob"][:, j]) for j in evaluable])
    blast_auroc = np.array([roc_auc_score(val_labels[:, j], meta_eval["blast_score"][:, j]) for j in evaluable])
    stacked_auroc = np.array([roc_auc_score(val_labels[:, j], stacked_scores[:, j]) for j in evaluable])

    print(f"[RESULT] n_labels_evaluable={len(evaluable)}")
    print(f"[RESULT] model   mean_auroc={model_auroc.mean():.4f} median={np.median(model_auroc):.4f}")
    print(f"[RESULT] blast   mean_auroc={blast_auroc.mean():.4f} median={np.median(blast_auroc):.4f}")
    print(f"[RESULT] stacked mean_auroc={stacked_auroc.mean():.4f} median={np.median(stacked_auroc):.4f}")

    # Bright-spot subset: same `evaluable` index set, not a separately recomputed one.
    diff_blast_model = blast_auroc - model_auroc
    bright_spot_mask = diff_blast_model >= 0.05
    print(f"[RESULT] bright-spot subset (BLAST beat model by >=0.05 on meta-eval): {bright_spot_mask.sum()} terms")
    print(
        f"[RESULT] on bright-spot subset: model mean_auroc={model_auroc[bright_spot_mask].mean():.4f} "
        f"stacked mean_auroc={stacked_auroc[bright_spot_mask].mean():.4f}"
    )


def main():
    assert_no_overlap()

    print("[INFO] Loading cached meta-eval slice (rows 0:2000)...")
    meta_eval = load_cached_meta_eval()

    print(f"[INFO] Computing/loading meta-train slice (rows {META_TRAIN_START}:{META_TRAIN_END})...")
    meta_train = compute_meta_train_slice(META_TRAIN_START, META_TRAIN_END)

    print("[INFO] Building flattened features...")
    X_train, y_train = build_features(meta_train)
    X_eval, _ = build_features(meta_eval)

    print("[INFO] Fitting XGBoost meta-model on meta-train...")
    model = fit_meta_model(X_train, y_train)

    print("[INFO] Evaluating on meta-eval...")
    report_results(model, X_eval, meta_eval)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full pipeline end-to-end**

Run: `cd experiments/ensembling && timeout 580 ../../env/bin/python stacking_pipeline.py`
(Task 2's Step 5 should already have the meta-train BLAST/inference cache fully
populated, so this run reuses those cache files and only pays the XGBoost fit + eval
cost. If it still exceeds the timeout — e.g. the fit itself is slow at ~54M rows — re-run
the same command; nothing above is lost, since every expensive sub-step checkpoints to
disk.)
Expected: five or more `[RESULT]` lines ending in the bright-spot subset comparison.

- [ ] **Step 3: Commit**

```bash
git add experiments/ensembling/stacking_pipeline.py
git commit -m "feat: wire up stacking pipeline evaluation and main() entry point

report_results() computes the stacked score's per-label AUROC against the
same evaluable-label filter and bright-spot subset used throughout this
line of probes, so the final numbers are directly comparable to the
already-logged model-alone and BLAST-alone results.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Task 6: Log the result and close out the plan-doc entry

**Files:**
- Modify: `experiments/cafa6_next_steps_plan.md`

**Interfaces:** None — this task only adds prose to the plan doc, following the exact
style of the two prior probe entries under Tier 1 item 1 (a `**Result (date):**`
paragraph with the actual printed numbers from Task 5's run).

- [ ] **Step 1: Append the result**

Add a new paragraph directly after the existing "Verdict: real homology signal
confirmed..." paragraph under Tier 1 item 1 in `experiments/cafa6_next_steps_plan.md`,
reporting the actual `[RESULT]` numbers Task 5's run printed (mean/median AUROC for
model/blast/stacked, and the bright-spot-subset comparison), with a verdict on whether
the stacked score beats trial-013 alone overall and/or specifically on the bright-spot
subset — written once the real numbers are in hand, not before.

- [ ] **Step 2: Commit**

```bash
git add experiments/cafa6_next_steps_plan.md
git commit -m "docs: log stacking pipeline result (ESM-2 + BLAST-KNN meta-model)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
