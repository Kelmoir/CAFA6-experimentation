"""Fuses trial-013's ESM-2 predictions with the BLAST-KNN retrieval score via a global
XGBoost meta-model, per experiments/cafa6_next_steps_plan.md Tier 1 item 1 and
docs/superpowers/specs/2026-10-02-ensembling-stacking-pipeline-design.md.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blast_knn_probe_real import (  # noqa: E402
    BLAST_BIN_DIR, CHECKPOINT_DIR, DATA_PATH, WORK_DIR,
    labels_to_multihot, transfer_labels_via_blast, write_fasta,
)

MAX_VAL_SAMPLES = 2000  # meta-eval slice: validation rows 0:2000 (matches run_diagnostic())
META_TRAIN_START = 2000
META_TRAIN_END = 4000


def load_cached_meta_eval(dataset=None):
    """Re-derives meta-eval's (protein x term) score matrices from the cache files
    blast_knn_probe_real.py already wrote -- no new BLAST/inference, just parsing.
    Pass `dataset` (an already-loaded DatasetDict) to avoid reloading it when the
    caller also needs it for compute_meta_train_slice()."""
    if dataset is None:
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
    blast_score = transfer_labels_via_blast(blast_out, val_ids, train_id_to_idx, train_labels)

    esm_prob = np.load(WORK_DIR / "model_probs_partial.npy")
    assert esm_prob.shape == val_labels.shape, (
        f"cached model_probs_partial.npy shape {esm_prob.shape} != val_labels shape {val_labels.shape}"
    )

    return {
        "esm_prob": esm_prob, "blast_score": blast_score, "val_labels": val_labels,
        "train_labels": train_labels, "label_names": label_names,
    }


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


def assert_meta_eval_matches_logged(meta_eval):
    """Confirms load_cached_meta_eval() reproduces the exact numbers already logged in
    experiments/cafa6_next_steps_plan.md for the real-BLAST probe -- called by main()
    before the expensive meta-train BLAST/inference step is spent, so a broken cache
    fails fast instead of only being caught by the standalone regression check
    (final-review minor finding, 2026-10-02)."""
    from sklearn.metrics import roc_auc_score

    val_labels, esm_prob, blast_score = (
        meta_eval["val_labels"], meta_eval["esm_prob"], meta_eval["blast_score"]
    )
    evaluable = [j for j in range(val_labels.shape[1]) if val_labels[:, j].min() != val_labels[:, j].max()]
    assert len(evaluable) == 9401, f"n_labels_evaluable={len(evaluable)}, expected 9401"

    model_auroc = np.array([roc_auc_score(val_labels[:, j], esm_prob[:, j]) for j in evaluable])
    blast_auroc = np.array([roc_auc_score(val_labels[:, j], blast_score[:, j]) for j in evaluable])
    assert abs(model_auroc.mean() - 0.8731) < 0.001, f"model mean AUROC {model_auroc.mean():.4f}, expected ~0.8731"
    assert abs(blast_auroc.mean() - 0.6854) < 0.001, f"blast mean AUROC {blast_auroc.mean():.4f}, expected ~0.6854"


def _check_label_resolution(preds, reference_feature, n_labels):
    """Raises if `preds` has far less per-label distinct-value resolution than
    `reference_feature` (a known-good input, e.g. esm_prob). Pooled std cannot see
    this: a fit can separate terms by base rate (plenty of pooled spread) while
    producing almost no within-term ranking signal at all, which is exactly what
    per-label AUROC measures and pooled std does not (final-review finding, 2026-10-02
    -- see cafa6_next_steps_plan.md's corrected write-up)."""
    n_rows = len(preds) // n_labels
    preds_by_label = preds.reshape(n_rows, n_labels)
    ref_by_label = reference_feature.reshape(n_rows, n_labels)
    median_preds = np.median([len(np.unique(preds_by_label[:, j])) for j in range(n_labels)])
    median_ref = np.median([len(np.unique(ref_by_label[:, j])) for j in range(n_labels)])
    assert median_preds >= 0.1 * median_ref, (
        f"per-label prediction resolution collapsed: median {median_preds:.0f} distinct "
        f"values/label vs the reference feature's {median_ref:.0f} -- the fit is likely "
        "separating terms by base rate rather than ranking within each term, which pooled "
        "std can't detect"
    )


def fit_meta_model(X_train, y_train, n_labels):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", random_state=42,
    )
    model.fit(X_train, y_train)

    train_preds = model.predict_proba(X_train)[:, 1]
    assert train_preds.std() > 1e-6, (
        "meta-model predictions collapsed to a near-constant value on its own training "
        "data -- likely needs negative subsampling (see the design spec's Meta-model "
        "risk note) before being trusted on the real meta-train set"
    )
    _check_label_resolution(train_preds, X_train[:, 0], n_labels)
    return model


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

    # Same correlation/">=0.05 beats" breakdown the two prior probes used, for direct
    # comparability (spec's Evaluation section; final-review minor finding, 2026-10-02).
    diff_blast_model = blast_auroc - model_auroc
    print(f"[RESULT] per-label AUROC correlation (model vs blast) = {np.corrcoef(model_auroc, blast_auroc)[0, 1]:.4f}")
    print(f"[RESULT] per-label AUROC correlation (stacked vs model) = {np.corrcoef(stacked_auroc, model_auroc)[0, 1]:.4f}")
    print(f"[RESULT] labels where BLAST beats model by >=0.05 AUROC: {(diff_blast_model >= 0.05).sum()} / {len(evaluable)}")
    print(f"[RESULT] labels where model beats BLAST by >=0.05 AUROC: {(diff_blast_model <= -0.05).sum()} / {len(evaluable)}")
    bright_spot_mask = diff_blast_model >= 0.05
    print(f"[RESULT] bright-spot subset (BLAST beat model by >=0.05 on meta-eval): {bright_spot_mask.sum()} terms")
    print(
        f"[RESULT] on bright-spot subset: model mean_auroc={model_auroc[bright_spot_mask].mean():.4f} "
        f"stacked mean_auroc={stacked_auroc[bright_spot_mask].mean():.4f}"
    )


def main():
    assert_no_overlap()
    dataset = load_from_disk(str(DATA_PATH))

    print("[INFO] Loading cached meta-eval slice (rows 0:2000)...")
    meta_eval = load_cached_meta_eval(dataset=dataset)
    assert_meta_eval_matches_logged(meta_eval)

    print(f"[INFO] Computing/loading meta-train slice (rows {META_TRAIN_START}:{META_TRAIN_END})...")
    meta_train = compute_meta_train_slice(META_TRAIN_START, META_TRAIN_END, dataset=dataset)

    print("[INFO] Building flattened features...")
    X_train, y_train = build_features(meta_train)
    X_eval, _ = build_features(meta_eval)

    print("[INFO] Fitting XGBoost meta-model on meta-train...")
    model = fit_meta_model(X_train, y_train, len(meta_train["label_names"]))

    print("[INFO] Evaluating on meta-eval...")
    report_results(model, X_eval, meta_eval)


def _blastp_atomic(argv_builder, out_path):
    """Runs a blastp-style command atomically: writes to a temp file first and renames
    into place only after the subprocess succeeds, so a `timeout`-induced kill mid-run
    (the plan's own documented retry pattern) can never leave a partial file at
    out_path -- which the next invocation's `out_path.exists()` cache check would
    otherwise silently treat as a complete result (final-review finding, 2026-10-02)."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    subprocess.run(argv_builder(tmp_path), check=True, capture_output=True, text=True)
    os.replace(tmp_path, out_path)


def compute_meta_train_slice(start, end, dataset=None):
    """Builds (protein x term) score matrices for validation rows [start:end), reusing
    the already-built BLAST database and the resumable checkpoint-every-20-batches
    inference pattern blast_knn_probe_real.py validated. Cache filenames are keyed by
    (start, end) so a smoke-test range never collides with the real 2000:4000 run.
    Pass `dataset` (an already-loaded DatasetDict) to avoid reloading it when the
    caller also needs it for load_cached_meta_eval()."""
    from functools import partial

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from training_run import data_collate_function, preprocess_batch  # noqa: E402

    if dataset is None:
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
        _blastp_atomic(
            lambda tmp: [
                str(BLAST_BIN_DIR / "blastp"), "-query", str(metatrain_fasta), "-db", str(blast_db),
                "-out", str(tmp), "-outfmt", "6 qseqid sseqid bitscore pident",
                "-max_target_seqs", "10", "-num_threads", "8",
            ],
            blast_out,
        )

    blast_score = transfer_labels_via_blast(blast_out, val_ids, train_id_to_idx, train_labels)

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


if __name__ == "__main__":
    main()
