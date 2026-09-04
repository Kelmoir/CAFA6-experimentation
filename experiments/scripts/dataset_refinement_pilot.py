"""Dataset-refinement pilot: does resolving a handful of "unknown" GO labels leave a detectable
trace in validation loss?

`training_run.py` treats every non-asserted (protein, GO-term) cell as a hard negative. Per
`experiments/sweep_log.md`'s trial-080/081 investigation, that's the field's own Open World
Assumption problem -- absence of annotation isn't confirmed absence of function, and trial-081's
own closing note recommended trying a soft-target correction instead of a hard mask (which made
things measurably worse) or a hard 0. This script is that follow-up, run as a small, fast,
controlled pilot before any commitment to a real sweep.

It runs a base configuration plus three experimental conditions (see CONDITIONS below), each
repeated many times with a fresh random sample of "resolved" cells, and asks: does the resulting
validation loss differ from the base run in a way that survives averaging over repeats -- and if
not at this item count/data scale, the three charts this script produces (two bar charts plus a
per-condition histogram) are meant to make that (lack of) signal directly visible, motivating what
scale to try next.

Every run in this pilot is trained on the EXACT SAME fixed --max-train-samples/--max-val-samples
protein subset and the EXACT SAME model initialization + data order (both pinned to --seed) -- the
only thing that ever differs between runs is which cells got resolved and how. Validation loss is
always computed against the real, untouched ground truth on the (separate) validation subset --
neither --unknown-target-value nor any run's overrides ever touch it.

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
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import evaluate_cafa_score  # reuse extract_labels_matrix
import training_run

CONDITIONS = ["faithful", "false_positive", "false_negative"]
CONDITION_OFFSET = {"faithful": 1, "false_positive": 2, "false_negative": 3}  # for deterministic seeding


# --------------------------------------------------------------------------------------------
# Deterministic item selection -- every RNG used anywhere in this script is explicitly
# constructed from --seed via plain integer arithmetic (never hash()/tuple-seeding, which would
# be silently unstable across processes under Python's default randomized string hashing).
# --------------------------------------------------------------------------------------------

def item_rng(seed: int, condition: str, repeat_index: int) -> random.Random:
    derived = seed * 100_000 + CONDITION_OFFSET[condition] * 10_000 + repeat_index
    return random.Random(derived)


def pick_cell(labels_matrix: np.ndarray, protein_ids: List[str], rng: random.Random,
              value: float, exclude: set) -> Tuple[str, int]:
    """A random (protein, term) cell whose real hard label == value, protein not in exclude."""
    order = list(range(labels_matrix.shape[0]))
    rng.shuffle(order)
    for row in order:
        if protein_ids[row] in exclude:
            continue
        cols = np.nonzero(labels_matrix[row] == value)[0]
        if len(cols) == 0:
            continue
        term_idx = int(cols[rng.randrange(len(cols))])
        return protein_ids[row], term_idx
    raise RuntimeError(f"Could not find a protein with a real label == {value} (excluding {len(exclude)} already used).")


def pick_distinct_cells(labels_matrix: np.ndarray, protein_ids: List[str], k: int,
                         rng: random.Random, value: float = 0.0, exclude: Optional[set] = None) -> List[Tuple[str, int]]:
    exclude = set(exclude or ())
    chosen = []
    for _ in range(k):
        pid, term_idx = pick_cell(labels_matrix, protein_ids, rng, value, exclude)
        chosen.append((pid, term_idx))
        exclude.add(pid)  # 1 experimental annotation per protein per run
    return chosen


def build_overrides(condition: str, labels_matrix: np.ndarray, protein_ids: List[str],
                     items_per_trial: int, rng: random.Random) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """{protein_id: {label_idx: (forced_value, original_value)}}. original_value is carried
    through purely for logging -- it's what lets a later look at the JSONL tell condition 2's
    single injected fake-positive, or condition 3's single injected false-negative, apart from
    its 99 "faithful" siblings in the same run, since forced_value alone doesn't distinguish them
    (both look like plain 0.0 overrides). See this script's module docstring / the plan's
    "user-confirmed, exact" condition definitions."""
    overrides: Dict[str, Dict[int, Tuple[float, float]]] = {}

    if condition == "faithful":
        for pid, idx in pick_distinct_cells(labels_matrix, protein_ids, items_per_trial, rng, value=0.0):
            overrides[pid] = {idx: (0.0, 0.0)}

    elif condition == "false_positive":
        cells = pick_distinct_cells(labels_matrix, protein_ids, items_per_trial, rng, value=0.0)
        flip = rng.randrange(items_per_trial)
        for i, (pid, idx) in enumerate(cells):
            overrides[pid] = {idx: (1.0, 0.0) if i == flip else (0.0, 0.0)}

    elif condition == "false_negative":
        cells = pick_distinct_cells(labels_matrix, protein_ids, items_per_trial - 1, rng, value=0.0)
        for pid, idx in cells:
            overrides[pid] = {idx: (0.0, 0.0)}
        used = {pid for pid, _ in cells}
        pos_pid, pos_idx = pick_cell(labels_matrix, protein_ids, rng, value=1.0, exclude=used)
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
                   overrides: Dict[str, Dict[int, float]], device: torch.device, tokenizer) -> float:
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

    return evaluate_val_loss(model, val_loader, device)


def evaluate_val_loss(model, val_loader, device: torch.device) -> float:
    """Pooled mean BCE over every element in the validation set (sum-then-divide, not an average
    of per-batch means -- avoids the batch-size-dependent bias training_run.py's own
    compute_metrics docstring flags for the analogous recall computation), against the real,
    unmodified ground truth. Never sees --unknown-target-value or any run's overrides."""
    was_training = model.training
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, total_elements = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            total_loss += criterion(logits, labels).item()
            total_elements += labels.numel()
    if was_training:
        model.train()
    return total_loss / total_elements


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
                                         args.items_per_trial, rng)

        start = time.time()
        val_loss = train_one_run(args, base_model, train_ds, val_loader, label_names,
                                  overrides, device, tokenizer)
        elapsed = time.time() - start

        record = {
            "run_id": spec["run_id"], "condition": spec["condition"],
            "repeat_index": spec["repeat_index"], "val_loss": val_loss,
            "train_time_s": elapsed,
            "config": {"model_name": args.model_name, "max_train_samples": args.max_train_samples,
                       "max_val_samples": args.max_val_samples,
                       "unknown_target_value": args.unknown_target_value, "epochs": args.epochs,
                       "items_per_trial": args.items_per_trial, "seed": args.seed},
        }
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
        print(f"[RUN {spec['run_id']}] val_loss={val_loss:.6f} ({elapsed:.1f}s)")

    print(f"[INFO] Pilot complete -> {log_path}")


# --------------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------------

def make_charts(log_path: Path, out_prefix: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    base_losses = [r["val_loss"] for r in records if r["condition"] == "base"]
    if not base_losses:
        raise RuntimeError(f"No 'base' run found in {log_path} -- nothing to compare against.")
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
    ax.bar(names, means, yerr=stds, capsize=4, color=["gray", "tab:blue", "tab:orange", "tab:red"])
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
    ax.bar(CONDITIONS, diffs, yerr=diff_stds, capsize=4, color=["tab:blue", "tab:orange", "tab:red"])
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
    colors = {"faithful": "tab:blue", "false_positive": "tab:orange", "false_negative": "tab:red"}
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset")
    ap.add_argument("--hf-dataset-repo", default="kelmoir/CAFA6")
    ap.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D",
                     help="Smallest ESM2 variant by default, per this pilot's starting point")
    ap.add_argument("--max-train-samples", type=int, default=4000)
    ap.add_argument("--max-val-samples", type=int, default=1000)
    ap.add_argument("--unknown-target-value", type=float, default=0.5,
                     help="Applied to every currently-0 cell in every run, base included -- pass "
                          "0.0 to disable and fall back to today's hard-negative behavior")
    ap.add_argument("--epochs", type=int, default=3)
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
