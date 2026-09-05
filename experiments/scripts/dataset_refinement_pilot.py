"""Dataset-refinement pilot: does resolving a handful of "unknown" GO labels leave a detectable
trace in validation loss?

`training_run.py` treats every non-asserted (protein, GO-term) cell as a hard negative. Per
`experiments/ancestor_sweep/sweep_log.md`'s trial-080/081 investigation, that's the field's own Open World
Assumption problem -- absence of annotation isn't confirmed absence of function, and trial-081's
own closing note recommended trying a soft-target correction instead of a hard mask (which made
things measurably worse) or a hard 0. This script is that follow-up, run as a small, fast,
controlled pilot before any commitment to a real sweep.

It runs a base configuration plus two experimental conditions (see CONDITIONS below), each
repeated many times with a fresh random sample of "resolved" cells, and asks: does the resulting
validation loss differ from the base run in a way that survives averaging over repeats -- and if
not at this item count/data scale, the charts this script produces are meant to make that (lack
of) signal directly visible, motivating what scale to try next.

(An earlier version also had a false_positive condition -- one currently-unknown cell mistakenly
forced to 1 instead of the faithful 0 -- modeling a curation mistake in the other direction. It's
been dropped: the pilot's real focus is the false_negative blind spot, see the paragraph below, and
the matched AUROC comparison this script now runs is clean/interpretable with fewer groups.)

Pooled val_loss turns out to have a real blind spot for exactly the error type this pilot cares
about most (false_negative): under this task's severe class imbalance, suppressing a term's only
positive training signal nudges the model toward predicting closer to 0 for that term everywhere,
which *lowers* pooled BCE loss (fewer, larger negative-class predictions dominate the sum) even
though the model got worse at the one thing that matters -- ranking that term's real positives
above its negatives. Per-term AUROC is threshold-independent and normalized per term, so it isn't
subject to the same imbalance-driven flattery (see --compute-eligible-term-auroc below) -- this is
the "does the injected error leave a detectable trace" instrument pooled val_loss can't be trusted
to be.

Every run in this pilot is trained on the EXACT SAME fixed --max-train-samples/--max-val-samples
protein subset and the EXACT SAME model initialization + data order (both pinned to --seed) -- the
only thing that ever differs between runs is which cells got resolved and how. Validation loss is
always computed against the real, untouched ground truth on the (separate) validation subset --
neither --unknown-target-value nor any run's overrides ever touch it.

Override-cell sampling is restricted to terms with >= --min-term-annotations (default 100) REAL,
unpropagated annotations in the raw CAFA-6 train_terms.tsv -- the best-supported part of the label
space, as a baseline: if the model can't detect a resolved cell there, that's not yet a fair test
of whether resolution is detectable at all (rarer terms have less signal to begin with). Pass
--min-term-annotations 0 to sample from the full label space as before.

Defaults to kelmoir/cafa6-raw (generate_dataset_raw.py) -- the fully raw/asserted, unpropagated
668-term label space, not the older ancestor-propagated 383-term kelmoir/CAFA6/coarse space. That
dataset's own vocabulary is already selected at the same >=100-raw-annotation bar, so
--min-term-annotations is a no-op there by construction (every label-space term already clears
it) -- it only starts doing real filtering work if --data-path/--hf-dataset-repo point at a
richer or differently-thresholded label space instead.

Usage:
  # Smoke test (~a couple of minutes)
  python experiments/scripts/dataset_refinement_pilot.py --run-tag smoke \\
      --repeats-per-condition 2 --items-per-trial 5 --epochs 1 \\
      --max-train-samples 200 --max-val-samples 100

  # Full pilot (all defaults: 4000/1000 items, 100/100 repeats, 3 epochs, smallest model) --
  # long-running, recommended via nohup/background:
  nohup env/bin/python experiments/scripts/dataset_refinement_pilot.py --run-tag pilot1 \\
      > experiments/logs/dataset_refinement_pilot1.out 2>&1 &

  # Regenerate charts from an existing log without retraining
  python experiments/scripts/dataset_refinement_pilot.py --run-tag pilot1 --plot-only

  # Resume a crashed/killed run -- skips run_ids already present in the log
  python experiments/scripts/dataset_refinement_pilot.py --run-tag pilot1 --resume
"""

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import evaluate_cafa_score  # reuse extract_labels_matrix
import training_run

CONDITIONS = ["faithful", "false_negative"]
CONDITION_OFFSET = {"faithful": 1, "false_negative": 3}  # for deterministic seeding (unchanged
# numeric values, so existing per-repeat seeds/logs stay reproducible after dropping false_positive)


# --------------------------------------------------------------------------------------------
# Deterministic item selection -- every RNG used anywhere in this script is explicitly
# constructed from --seed via plain integer arithmetic (never hash()/tuple-seeding, which would
# be silently unstable across processes under Python's default randomized string hashing).
# --------------------------------------------------------------------------------------------

def item_rng(seed: int, condition: str, repeat_index: int) -> random.Random:
    derived = seed * 100_000 + CONDITION_OFFSET[condition] * 10_000 + repeat_index
    return random.Random(derived)


def load_raw_annotation_counts(train_terms_tsv: Path) -> Dict[str, int]:
    """Real, unpropagated annotation count per GO term, straight from the raw CAFA-6
    train_terms.tsv (EntryID, term, aspect) -- NOT the ancestor-propagated counts implicit in
    the coarse label space's own labels_matrix (a term's propagated positive count includes
    every descendant's annotations, which is a very different, much larger number). Used to
    restrict override-cell sampling to terms with real, directly-observed human support -- see
    --min-term-annotations."""
    counts: Counter = Counter()
    with train_terms_tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            counts[row["term"]] += 1
    return dict(counts)


def pick_cell(labels_matrix: np.ndarray, protein_ids: List[str], rng: random.Random,
              value: float, exclude: set, eligible_term_idx: np.ndarray) -> Tuple[str, int]:
    """A random (protein, term) cell whose real hard label == value, protein not in exclude,
    term restricted to eligible_term_idx (see --min-term-annotations)."""
    order = list(range(labels_matrix.shape[0]))
    rng.shuffle(order)
    for row in order:
        if protein_ids[row] in exclude:
            continue
        eligible_vals = labels_matrix[row, eligible_term_idx]
        cols = np.nonzero(eligible_vals == value)[0]
        if len(cols) == 0:
            continue
        term_idx = int(eligible_term_idx[cols[rng.randrange(len(cols))]])
        return protein_ids[row], term_idx
    raise RuntimeError(f"Could not find a protein with an eligible-term label == {value} "
                        f"(excluding {len(exclude)} already used, {len(eligible_term_idx)} eligible terms).")


def pick_distinct_cells(labels_matrix: np.ndarray, protein_ids: List[str], k: int,
                         rng: random.Random, eligible_term_idx: np.ndarray, value: float = 0.0,
                         exclude: Optional[set] = None) -> List[Tuple[str, int]]:
    exclude = set(exclude or ())
    chosen = []
    for _ in range(k):
        pid, term_idx = pick_cell(labels_matrix, protein_ids, rng, value, exclude, eligible_term_idx)
        chosen.append((pid, term_idx))
        exclude.add(pid)  # 1 experimental annotation per protein per run
    return chosen


def build_overrides(condition: str, labels_matrix: np.ndarray, protein_ids: List[str],
                     items_per_trial: int, rng: random.Random,
                     eligible_term_idx: np.ndarray) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """{protein_id: {label_idx: (forced_value, original_value)}}. original_value is carried
    through purely for logging -- it's what lets a later look at the JSONL tell the
    false_negative condition's single injected false-negative apart from its 99 "faithful"
    siblings in the same run, since forced_value alone doesn't distinguish them (both look like
    plain 0.0 overrides). See this script's module docstring / the plan's "user-confirmed, exact"
    condition definitions.

    eligible_term_idx restricts which term columns can be chosen at all (see
    --min-term-annotations): starting from the best-supported part of the label space as a
    baseline, on the notion that if the model can't detect resolution there, it's not a fair
    test of whether resolution is detectable at all -- rarer terms (which the model has less
    signal on to begin with) are a harder, separate question for a later pilot scale."""
    overrides: Dict[str, Dict[int, Tuple[float, float]]] = {}

    if condition == "faithful":
        for pid, idx in pick_distinct_cells(labels_matrix, protein_ids, items_per_trial, rng,
                                             eligible_term_idx, value=0.0):
            overrides[pid] = {idx: (0.0, 0.0)}

    elif condition == "false_negative":
        cells = pick_distinct_cells(labels_matrix, protein_ids, items_per_trial - 1, rng,
                                     eligible_term_idx, value=0.0)
        for pid, idx in cells:
            overrides[pid] = {idx: (0.0, 0.0)}
        used = {pid for pid, _ in cells}
        pos_pid, pos_idx = pick_cell(labels_matrix, protein_ids, rng, value=1.0, exclude=used,
                                      eligible_term_idx=eligible_term_idx)
        overrides[pos_pid] = {pos_idx: (0.0, 1.0)}  # the injected false negative

    else:
        raise ValueError(f"Unknown condition {condition!r}")

    return overrides


# --------------------------------------------------------------------------------------------
# Model caching -- create_esm() re-downloads/re-parses the pretrained checkpoint from disk and
# rebuilds the whole HF module graph every time it's called, which used to happen once per run
# (301 times). Since every run is seeded identically anyway (see build_base_model), a from-scratch
# rebuild was always producing a bit-identical model to the one already sitting in memory -- pure
# waste. Now built exactly ONCE, kept resident on --device (GPU included, per the whole point of
# caching it there), and copy.deepcopy'd per run instead: a plain tensor copy already on-device,
# no disk I/O, no from_pretrained bookkeeping, no shape/config validation.
# --------------------------------------------------------------------------------------------

def build_base_model(args, model_name: str, label_names: List[str], device: torch.device):
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    model = training_run.create_esm(
        unfreeze_last_n=args.unfreeze_last_n, model_type=model_name,
        id2label=id2label, label2id=label2id,
    ).to(device)
    model.eval()  # the cached reference itself is never trained directly
    return model


# --------------------------------------------------------------------------------------------
# Training / evaluation for one run
# --------------------------------------------------------------------------------------------

def train_one_run(args, base_model, train_ds, val_loader, label_names: List[str],
                   overrides: Dict[str, Dict[int, float]], device: torch.device,
                   tokenizer) -> Tuple[np.ndarray, np.ndarray]:
    """Trains one run and returns the raw (logits, labels) arrays from the final validation
    pass -- val_loss and any per-term AUROC diagnostics are derived from this single pass by the
    caller, rather than each paying for their own forward pass over the validation set."""
    # A fresh, independent copy of the cached base model -- bit-identical starting weights to
    # every other run (base_model was built once, under the fixed seed), without re-reading
    # anything from disk.
    model = copy.deepcopy(base_model)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=generator,
        collate_fn=partial(training_run.data_collate_function, tokenizer=tokenizer),
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(args.epochs):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hard_labels = batch["labels"].to(device)
            labels = hard_labels.clone()

            if args.unknown_target_value is not None:
                labels = torch.where(hard_labels == 0,
                                      torch.full_like(labels, args.unknown_target_value), labels)
            if overrides:
                for row, pid in enumerate(batch["protein"]):
                    cell_overrides = overrides.get(pid)
                    if cell_overrides:
                        for term_idx, (forced_value, _original_value) in cell_overrides.items():
                            labels[row, term_idx] = forced_value

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()

    return evaluate_val_predictions(model, val_loader, device)


def evaluate_val_predictions(model, val_loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Single forward pass over the whole validation set, against the real, unmodified ground
    truth (never sees --unknown-target-value or any run's overrides). Returns (logits, labels) as
    (n_val, n_terms) numpy arrays -- val_loss and per-term AUROC are both derived from this one
    pass rather than each paying for their own."""
    was_training = model.training
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(all_logits), np.concatenate(all_labels)


def pooled_bce_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Pooled mean BCE-with-logits over every element -- the same sum-then-divide-once quantity
    the previous torch-based evaluate_val_loss computed (mathematically identical to
    reduction="mean" over the full concatenated array), just computed from the (logits, labels)
    evaluate_val_predictions already produced. Uses the standard numerically-stable formula
    (max(x,0) - x*y + log1p(exp(-|x|)), the same one torch.nn.BCEWithLogitsLoss uses internally)."""
    loss = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    return float(loss.mean())


def per_term_auroc(probs: np.ndarray, labels: np.ndarray, term_idx: int) -> Optional[float]:
    """AUROC for one term column against the real validation labels -- threshold-independent and
    normalized per term, so (unlike pooled val_loss) it isn't dominated by this task's severe
    class imbalance. None if that column is single-class in this validation slice (roc_auc_score
    undefined there -- same check training_run.py's run_diagnostic() uses)."""
    y_true = labels[:, term_idx]
    if y_true.min() == y_true.max():
        return None
    return float(roc_auc_score(y_true, probs[:, term_idx]))


def compute_term_auroc_dict(probs: np.ndarray, labels: np.ndarray, term_indices,
                             label_names: List[str]) -> Dict[str, Optional[float]]:
    return {label_names[int(idx)]: per_term_auroc(probs, labels, int(idx)) for idx in term_indices}


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------

def build_run_specs(args) -> List[Dict]:
    specs = [{"run_id": f"base-{i:03d}" if args.base_repeats > 1 else "base",
              "condition": "base", "repeat_index": i} for i in range(args.base_repeats)]
    for condition in CONDITIONS:
        for i in range(args.repeats_per_condition):
            specs.append({"run_id": f"{condition}-{i:03d}", "condition": condition, "repeat_index": i})
    return specs


def load_completed_run_ids(log_path: Path) -> set:
    if not log_path.exists():
        return set()
    completed = set()
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            completed.add(json.loads(line)["run_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def append_record(log_path: Path, record: Dict) -> None:
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_pilot(args) -> None:
    device = torch.device(args.device)
    log_path = Path(args.log_dir) / f"dataset_refinement_pilot_{args.run_tag}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = training_run.load_training_data(data_path=Path(args.data_path), repo_id=args.hf_dataset_repo)
    label_names = [str(name) for name in dataset["train"].features["labels"].feature.names]

    # Fixed once, for every one of the 301 runs -- see module docstring.
    train_raw = dataset["train"].shuffle(seed=args.seed).select(
        range(min(args.max_train_samples, len(dataset["train"]))))
    val_raw = dataset["validation"].shuffle(seed=args.seed).select(
        range(min(args.max_val_samples, len(dataset["validation"]))))
    print(f"[INFO] Fixed subsets: {len(train_raw)} train / {len(val_raw)} val proteins "
          f"(seed={args.seed}), {len(label_names)} label terms")

    labels_matrix = evaluate_cafa_score.extract_labels_matrix(train_raw, len(label_names))
    train_protein_ids = list(train_raw["protein"])

    raw_counts = load_raw_annotation_counts(Path(args.train_terms_tsv))
    val_labels_matrix = evaluate_cafa_score.extract_labels_matrix(val_raw, len(label_names))
    val_positive_counts = val_labels_matrix.sum(axis=0)
    eligible_term_idx = np.array([i for i, name in enumerate(label_names)
                                   if raw_counts.get(name, 0) >= args.min_term_annotations
                                   and val_positive_counts[i] >= args.min_val_positives])
    if len(eligible_term_idx) == 0:
        raise RuntimeError(f"No terms in the {len(label_names)}-term label space have "
                            f">= {args.min_term_annotations} real annotations in "
                            f"{args.train_terms_tsv} AND >= {args.min_val_positives} positives in "
                            f"the fixed {len(val_raw)}-protein validation subset -- lower "
                            f"--min-term-annotations/--min-val-positives or raise --max-val-samples.")
    print(f"[INFO] {len(eligible_term_idx)}/{len(label_names)} label terms have >= "
          f"{args.min_term_annotations} real (unpropagated) annotations in {args.train_terms_tsv} "
          f"AND >= {args.min_val_positives} positives in the fixed validation subset -- override "
          f"sampling restricted to those (best-supported terms first, per this pilot's baseline "
          f"scope; the validation-positives bar exists so a chosen term's AUROC is actually "
          f"computable -- see --compute-eligible-term-auroc)")

    tokenizer = training_run.create_esm_tokenizer(args.model_name)
    label2id = {name: i for i, name in enumerate(label_names)}
    preprocess = partial(training_run.preprocess_batch, tokenizer=tokenizer,
                          max_length=training_run.MAX_LENGTH, unique_items=label_names, label2id=label2id)
    train_ds = train_raw.with_transform(transform=preprocess)
    val_ds = val_raw.with_transform(transform=preprocess)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=partial(training_run.data_collate_function, tokenizer=tokenizer),
    )

    base_model = build_base_model(args, args.model_name, label_names, device)
    print(f"[INFO] Base model cached on {device} -- every run copies from this instead of "
          f"rebuilding from disk")

    specs = build_run_specs(args)
    completed = load_completed_run_ids(log_path) if args.resume else set()
    if completed:
        print(f"[INFO] --resume: {len(completed)} run(s) already in {log_path}, skipping those")

    for spec in specs:
        if spec["run_id"] in completed:
            continue

        if spec["condition"] == "base":
            overrides = {}
        else:
            rng = item_rng(args.seed, spec["condition"], spec["repeat_index"])
            overrides = build_overrides(spec["condition"], labels_matrix, train_protein_ids,
                                         args.items_per_trial, rng, eligible_term_idx)

        start = time.time()
        val_logits, val_labels_arr = train_one_run(args, base_model, train_ds, val_loader, label_names,
                                                    overrides, device, tokenizer)
        val_loss = pooled_bce_loss(val_logits, val_labels_arr)
        elapsed = time.time() - start

        record = {
            "run_id": spec["run_id"], "condition": spec["condition"],
            "repeat_index": spec["repeat_index"], "val_loss": val_loss,
            "train_time_s": elapsed,
            "config": {"model_name": args.model_name, "max_train_samples": args.max_train_samples,
                       "max_val_samples": args.max_val_samples,
                       "unknown_target_value": args.unknown_target_value, "epochs": args.epochs,
                       "items_per_trial": args.items_per_trial, "seed": args.seed,
                       "min_term_annotations": args.min_term_annotations,
                       "min_val_positives": args.min_val_positives,
                       "n_eligible_terms": len(eligible_term_idx)},
        }

        # AUROC diagnostics -- see module docstring for why pooled val_loss alone can't be
        # trusted to catch a false_negative-type injected error under this task's class
        # imbalance. eligible_term_auroc/touched_term_auroc are cheap: every override candidate
        # comes from eligible_term_idx by construction (build_overrides), so a single full pass
        # over eligible_term_idx covers both the global aggregate and every touched term at once
        # -- no separate computation needed for the "touched" subset.
        if args.compute_eligible_term_auroc:
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            eligible_auroc = compute_term_auroc_dict(val_probs, val_labels_arr, eligible_term_idx, label_names)
            valid_eligible = [v for v in eligible_auroc.values() if v is not None]
            record["eligible_terms_mean_auroc"] = float(np.mean(valid_eligible)) if valid_eligible else None
            record["n_eligible_terms_evaluable"] = len(valid_eligible)

            if spec["condition"] == "base":
                # Full per-term vector, kept only for base runs -- this is what make_charts uses
                # as the "no corruption happened" baseline to match every other run's touched
                # term(s) against.
                record["eligible_term_auroc"] = eligible_auroc
            else:
                touched_idx = sorted({idx for cell_map in overrides.values() for idx in cell_map})
                touched_auroc = {label_names[idx]: eligible_auroc[label_names[idx]] for idx in touched_idx}
                injected_error_term = None
                for cell_map in overrides.values():
                    for idx, (forced_val, orig_val) in cell_map.items():
                        if forced_val != orig_val:
                            injected_error_term = label_names[idx]
                valid_touched = [v for v in touched_auroc.values() if v is not None]
                record["touched_term_auroc"] = touched_auroc
                record["injected_error_term"] = injected_error_term
                record["injected_error_term_auroc"] = (
                    touched_auroc.get(injected_error_term) if injected_error_term else None)
                record["overridden_terms_mean_auroc"] = float(np.mean(valid_touched)) if valid_touched else None
        if args.overridden_cells_log_level == 2:
            record["overridden_cells"] = [
                {"protein": pid, "term": label_names[idx], "forced_value": forced_val,
                 "original_value": orig_val, "is_injected_error": forced_val != orig_val}
                for pid, cell_map in overrides.items()
                for idx, (forced_val, orig_val) in cell_map.items()
            ]
        elif args.overridden_cells_log_level == 1:
            record["overridden_cells"] = [
                {"protein": pid, "term": label_names[idx],
                 "is_injected_error": forced_val != orig_val}
                for pid, cell_map in overrides.items()
                for idx, (forced_val, orig_val) in cell_map.items()
            ]
        # level 0: omit "overridden_cells" from the record entirely -- these lists are a
        # substantial share of the JSONL's size at full pilot scale (300 runs x 100 cells each).
        append_record(log_path, record)
        auroc_note = ""
        if args.compute_eligible_term_auroc and spec["condition"] != "base" and record.get("injected_error_term"):
            ia = record["injected_error_term_auroc"]
            auroc_note = f" injected_term_auroc={ia:.4f}" if ia is not None else " injected_term_auroc=N/A"
        print(f"[RUN {spec['run_id']}] val_loss={val_loss:.6f}{auroc_note} ({elapsed:.1f}s)")

    print(f"[INFO] Pilot complete -> {log_path}")


# --------------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------------

def make_charts(log_path: Path, out_prefix: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]

    # A run's val_loss can occasionally come back nan/inf -- observed directly at tiny
    # (--max-train-samples ~200) smoke scale, non-reproducible run-to-run despite the fixed
    # seed (PyTorch's CPU ops aren't guaranteed bit-identical across runs without explicitly
    # enabling deterministic algorithms, which this script doesn't). Dropped here rather than
    # left to reach np.mean/histogram_bin_edges, which would otherwise crash and lose every
    # other run's results in the same log file -- a real risk at full pilot scale (300 runs,
    # hours of compute) if even one run hits this.
    nan_run_ids = [r["run_id"] for r in records if not np.isfinite(r["val_loss"])]
    if nan_run_ids:
        print(f"[WARN] {len(nan_run_ids)} run(s) had a non-finite val_loss (nan/inf) -- excluded "
              f"from charts/summary: {nan_run_ids}")
    records = [r for r in records if np.isfinite(r["val_loss"])]

    base_losses = [r["val_loss"] for r in records if r["condition"] == "base"]
    if not base_losses:
        raise RuntimeError(f"No finite 'base' run found in {log_path} -- nothing to compare against.")
    base_mean = float(np.mean(base_losses))
    by_condition = {c: [r["val_loss"] for r in records if r["condition"] == c] for c in CONDITIONS}
    for c in CONDITIONS:
        if not by_condition[c]:
            print(f"[WARN] No completed runs for condition {c!r} yet -- charts will omit it.")

    # 1. Loss after training, by condition
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["base"] + CONDITIONS
    means = [base_mean] + [float(np.mean(by_condition[c])) if by_condition[c] else float("nan") for c in CONDITIONS]
    stds = [float(np.std(base_losses))] + [float(np.std(by_condition[c])) if by_condition[c] else 0.0 for c in CONDITIONS]
    ax.bar(names, means, yerr=stds, capsize=4, color=["gray", "tab:blue", "tab:red"])
    ax.set_ylabel("Validation BCE loss")
    ax.set_title(f"Validation loss after training (n_base={len(base_losses)}, "
                 f"n_condition={len(by_condition[CONDITIONS[0]])})")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_loss_after_training.png", dpi=150)
    plt.close(fig)

    # 2. Mean loss difference vs. base
    fig, ax = plt.subplots(figsize=(8, 5))
    diffs = [float(np.mean(by_condition[c])) - base_mean if by_condition[c] else float("nan") for c in CONDITIONS]
    diff_stds = [float(np.std(by_condition[c])) if by_condition[c] else 0.0 for c in CONDITIONS]
    ax.bar(CONDITIONS, diffs, yerr=diff_stds, capsize=4, color=["tab:blue", "tab:red"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean val loss difference vs. base run")
    ax.set_title("Mean loss difference vs. base, by condition")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_loss_diff_vs_base.png", dpi=150)
    plt.close(fig)

    # 3. Distribution of every individual repeat's own difference from base, as a histogram --
    # overlaid per condition (shared bin edges) rather than one bar per run, so the shape of each
    # condition's spread (and any overlap between conditions) is directly visible.
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"faithful": "tab:blue", "false_negative": "tab:red"}
    all_diffs = [v - base_mean for c in CONDITIONS for v in by_condition[c]]
    bins = np.histogram_bin_edges(all_diffs, bins=30) if all_diffs else 30
    for c in CONDITIONS:
        diffs = [v - base_mean for v in by_condition[c]]
        if diffs:
            ax.hist(diffs, bins=bins, color=colors[c], alpha=0.5, edgecolor="black",
                     linewidth=0.3, label=f"{c} (n={len(diffs)})")
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", label="base")
    ax.set_xlabel("Val loss difference vs. base run")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of per-run loss difference vs. base, by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_loss_diff_histogram.png", dpi=150)
    plt.close(fig)

    print(f"[INFO] Wrote charts -> {out_prefix}_loss_after_training.png, "
          f"{out_prefix}_loss_diff_vs_base.png, {out_prefix}_loss_diff_histogram.png")
    print(f"[SUMMARY] base: mean={base_mean:.6f} std={np.std(base_losses):.6f} n={len(base_losses)}")
    for c in CONDITIONS:
        if by_condition[c]:
            print(f"[SUMMARY] {c}: mean={np.mean(by_condition[c]):.6f} "
                  f"std={np.std(by_condition[c]):.6f} diff_vs_base={np.mean(by_condition[c]) - base_mean:.6f} "
                  f"n={len(by_condition[c])}")

    make_auroc_charts(records, out_prefix)


def make_auroc_charts(records: List[Dict], out_prefix: Path) -> None:
    """Two AUROC-based charts, meant to directly test the pooled-val_loss blind-spot hypothesis
    (see module docstring): (4) a global, imbalance-resistant aggregate (mean AUROC across every
    eligible term) to confirm one corrupted cell among hundreds of terms doesn't move a broad
    average either way -- pooled val_loss and this should agree it's a null result; (5) the
    targeted, per-term-matched comparison that pooled val_loss can't do at all: for every
    non-base run, each touched term's post-training AUROC minus that SAME term's own AUROC in the
    base runs, split into the one deliberately-corrupted term ("injected") vs. the other,
    correctly-resolved touched terms in the same run ("control"). If the false_negative
    mechanism is real, "injected" should show a visible AUROC drop while every "control" group
    (including all of faithful, which never corrupts anything) hovers near zero.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_records = [r for r in records if r["condition"] == "base"]
    if not any("eligible_term_auroc" in r or "eligible_terms_mean_auroc" in r for r in base_records):
        print("[WARN] No AUROC data in this log (predates --compute-eligible-term-auroc, or it "
              "was disabled) -- skipping AUROC charts.")
        return

    # 4. Global aggregate AUROC (mean over all eligible terms), base vs. each condition.
    global_by_condition = {c: [r["eligible_terms_mean_auroc"] for r in records
                                if r["condition"] == c and r.get("eligible_terms_mean_auroc") is not None]
                            for c in CONDITIONS}
    global_base = [r["eligible_terms_mean_auroc"] for r in base_records
                   if r.get("eligible_terms_mean_auroc") is not None]
    if global_base:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = ["base"] + CONDITIONS
        means = [float(np.mean(global_base))] + [
            float(np.mean(global_by_condition[c])) if global_by_condition[c] else float("nan") for c in CONDITIONS]
        stds = [float(np.std(global_base))] + [
            float(np.std(global_by_condition[c])) if global_by_condition[c] else 0.0 for c in CONDITIONS]
        ax.bar(names, means, yerr=stds, capsize=4, color=["gray", "tab:blue", "tab:red"])
        ax.set_ylabel("Mean AUROC across all eligible terms")
        ax.set_title("Global (imbalance-resistant) aggregate AUROC, by condition")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_global_auroc.png", dpi=150)
        plt.close(fig)
        print(f"[SUMMARY] global eligible-terms mean AUROC: base={np.mean(global_base):.4f} " +
              " ".join(f"{c}={np.mean(global_by_condition[c]):.4f}" if global_by_condition[c] else f"{c}=N/A"
                        for c in CONDITIONS))

    # 5. The targeted comparison: matched per-term AUROC delta vs. base, injected term vs. control.
    base_term_lists: Dict[str, List[float]] = {}
    for r in base_records:
        for term, val in r.get("eligible_term_auroc", {}).items():
            if val is not None:
                base_term_lists.setdefault(term, []).append(val)
    base_term_auroc_mean = {t: float(np.mean(v)) for t, v in base_term_lists.items()}
    if not base_term_auroc_mean:
        print("[WARN] No base run logged a full eligible_term_auroc vector -- skipping the "
              "matched per-term AUROC delta chart (need at least one base run with "
              "--compute-eligible-term-auroc on).")
        return

    def matched_diffs(condition: str, want_injected: bool) -> List[float]:
        diffs = []
        for r in records:
            if r["condition"] != condition:
                continue
            touched = r.get("touched_term_auroc")
            if not touched:
                continue
            injected_term = r.get("injected_error_term")
            for term, val in touched.items():
                if val is None:
                    continue
                is_injected_term = (term == injected_term)
                if is_injected_term != want_injected:
                    continue
                base_val = base_term_auroc_mean.get(term)
                if base_val is None:
                    continue
                diffs.append(val - base_val)
        return diffs

    groups = [
        ("faithful\n(control)", matched_diffs("faithful", False)),
        ("false_negative\n(control)", matched_diffs("false_negative", False)),
        ("false_negative\n(injected)", matched_diffs("false_negative", True)),
    ]
    if not any(diffs for _, diffs in groups):
        print("[WARN] No matched per-term AUROC deltas available -- skipping that chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [name for name, diffs in groups]
    means = [float(np.mean(diffs)) if diffs else float("nan") for _, diffs in groups]
    stds = [float(np.std(diffs)) if diffs else 0.0 for _, diffs in groups]
    colors = ["tab:blue", "tab:red", "tab:red"]
    alphas = [0.5, 0.5, 1.0]  # "injected" bars drawn solid, "control" bars faded
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Val AUROC (this run) - Val AUROC (base, same term)")
    ax.set_title("Matched per-term AUROC delta vs. base: injected-error term vs. control terms")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_matched_term_auroc_delta.png", dpi=150)
    plt.close(fig)

    print(f"[INFO] Wrote charts -> {out_prefix}_global_auroc.png, "
          f"{out_prefix}_matched_term_auroc_delta.png")
    for name, diffs in groups:
        if diffs:
            print(f"[SUMMARY] matched AUROC delta {name.replace(chr(10), ' ')}: "
                  f"mean={np.mean(diffs):.4f} std={np.std(diffs):.4f} n={len(diffs)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset_raw",
                     help="Local on-disk dataset (generate_dataset_raw.py's output by default -- "
                          "the raw/unpropagated 668-term label space, not the older ancestor-"
                          "propagated 383-term kelmoir/CAFA6/coarse one)")
    ap.add_argument("--hf-dataset-repo", default="kelmoir/cafa6-raw",
                     help="Hub fallback if --data-path doesn't exist locally")
    ap.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D",
                     help="Smallest ESM2 variant by default, per this pilot's starting point")
    ap.add_argument("--max-train-samples", type=int, default=4000)
    ap.add_argument("--max-val-samples", type=int, default=1000)
    ap.add_argument("--unknown-target-value", type=float, default=0.5,
                     help="Applied to every currently-0 cell in every run, base included -- pass "
                          "0.0 to disable and fall back to today's hard-negative behavior")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-terms-tsv", default="data/cafa-6-protein-function-prediction/Train/train_terms.tsv",
                     help="Raw (unpropagated) CAFA-6 annotation table, used only to compute each "
                          "term's real annotation count for --min-term-annotations filtering")
    ap.add_argument("--min-term-annotations", type=int, default=100,
                     help="Only label-space terms with at least this many REAL (asserted, "
                          "unpropagated) CAFA-6 annotations are eligible for override-cell "
                          "sampling. Rationale: the best-supported terms should be easiest for "
                          "the model to have learned already, so this is the baseline to check "
                          "detectability against first -- resolving a term the model barely has "
                          "signal on to begin with isn't a fair test. 0 disables the filter "
                          "(every label-space term eligible, today's prior behavior).")
    ap.add_argument("--min-val-positives", type=int, default=3,
                     help="Additional override-eligibility filter, applied together with "
                          "--min-term-annotations: only terms with at least this many REAL "
                          "positives in the fixed validation subset are eligible for override "
                          "sampling. Needed because roc_auc_score is undefined for a term with "
                          "zero validation positives -- a term clearing >=100 annotations across "
                          "the whole dataset can still land very few, or zero, in a 1000-protein "
                          "validation slice, which would make --injected-error-term-auroc come "
                          "back None for that repeat.")
    ap.add_argument("--compute-eligible-term-auroc", action=argparse.BooleanOptionalAction, default=True,
                     help="Per-term AUROC diagnostics (this pilot's answer to pooled val_loss's "
                          "class-imbalance blind spot -- see module docstring): every run logs "
                          "its global eligible_terms_mean_auroc; base runs additionally log the "
                          "full per-term eligible_term_auroc vector (the baseline every other "
                          "run's touched term(s) get matched against in make_charts); every "
                          "other run logs touched_term_auroc plus which one term (if any) was the "
                          "actual injected error and that term's own AUROC. Cheap (reuses the "
                          "validation logits already computed for val_loss, no extra forward "
                          "pass) -- disable only if you specifically want the original, "
                          "loss-only pilot.")
    ap.add_argument("--items-per-trial", type=int, default=100)
    ap.add_argument("--repeats-per-condition", type=int, default=100)
    ap.add_argument("--overridden-cells-log-level", type=int, default=2, choices=[0, 1, 2],
                     help="Detail logged per run for the overridden cells (a substantial share of "
                          "the JSONL's size at full pilot scale): 2 = full (protein, term, "
                          "forced_value, original_value, is_injected_error); 1 = protein, term, "
                          "is_injected_error only; 0 = omit overridden_cells entirely")
    ap.add_argument("--base-repeats", type=int, default=4,
                     help="Repeat the base run and average, to smooth out training noise -- a "
                          "fixed-seed base run still showed a ~0.00003 val-loss noise floor "
                          "between repeats")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4,
                     help="4 was fastest on the speed-test set among {2,4,8}")
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--unfreeze-last-n", type=int, default=2)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--run-tag", default="default", help="Names the log/chart files for this pilot scale")
    ap.add_argument("--log-dir", default="experiments/logs")
    ap.add_argument("--resume", action="store_true", help="Skip run_ids already present in the log file")
    ap.add_argument("--plot-only", action="store_true", help="Skip training; regenerate charts from the existing log")
    args = ap.parse_args()

    log_path = Path(args.log_dir) / f"dataset_refinement_pilot_{args.run_tag}.jsonl"
    if args.plot_only:
        make_charts(log_path, Path(args.log_dir) / f"dataset_refinement_pilot_{args.run_tag}")
        return

    run_pilot(args)
    make_charts(log_path, Path(args.log_dir) / f"dataset_refinement_pilot_{args.run_tag}")


if __name__ == "__main__":
    main()
