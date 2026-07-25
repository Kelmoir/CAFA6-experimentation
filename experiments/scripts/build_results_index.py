"""Rebuild experiments/results_index.csv from the raw trial logs.

Parses every experiments/logs/trial-*.log (including *_epochN.log continuation
fragments from the old per-epoch-chained runs), groups fragments belonging to the
same trial id, and picks each trial's best (lowest val_loss) epoch across all its
fragments. Safe to re-run any time — it's a full rebuild, not an incremental append,
so the CSV always reflects exactly what's in experiments/logs/.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path("experiments/logs")
OUT_PATH = Path("experiments/results_index.csv")

# Manually curated from sweep_log.md's per-trial verdicts (not reliably derivable from the
# logs alone). Update this alongside sweep_log.md whenever a trial gets a verdict.
VERDICTS = {
    "trial-001": "VOID (label bug)",
    "trial-002": "INCOMPLETE (connection lost)",
    "trial-003": "DEAD",
    "trial-004": "DEAD",
    "trial-005": "WORSE (collapse toward all-positive)",
    "trial-006": "DEAD",
    "trial-007": "DEAD",
    "trial-008": "OVERFITTING",
    "trial-009": "OVERFITTING",
    "trial-010": "OVERFITTING",
    "trial-011": "OVERFITTING",
    "trial-012": "OVERFITTING",
    "trial-013": "DEAD (re-approached via excess wd)",
    "trial-014": "MIXED (epochs 1-4 pre-fix, 5-6 post per-epoch-variation fix)",
    "trial-015": "PLATEAU (stable, beam search round-0 seed)",
    "trial-016": "PLATEAU (matches trial-015, unfreeze bump had no effect)",
    "trial-018": "PLATEAU (round-1 winner: cosine lr_schedule, best val_loss of round)",
    "trial-019": "PLATEAU (round-1: warmup_steps=300, no improvement over seed)",
    "trial-020": "PLATEAU (round-1 reg: label_smoothing=0.05, no improvement; train_acc/recall=0 is a metric-computation artifact, not the real training signal - fixed post-hoc, see training_run.py)",
    "trial-021": "PLATEAU (round-1 reg: decouple_bias_wd, roughly ties seed)",
    "trial-022": "PLATEAU (round-1 runner-up: grad_accum_steps=2, ~seed-level)",
    "trial-023": "PLATEAU (round-1: unfreeze_last_n=3, noisier + worse than seed, slower per-epoch)",
    "trial-024": "PLATEAU (round-2: learning_rate=3e-4 off trial-018, no improvement over parent)",
    "trial-025": "PLATEAU (round-2 winner: weight_decay=0.2 off trial-018, ties trial-018 within noise)",
    "trial-026": "PLATEAU (round-2 runner-up: unfreeze_last_n=2 off trial-018, ties trial-018 within noise, slower per-epoch)",
    "trial-027": "PLATEAU (round-2: cosine+grad_accum=2 stacked off trial-022, no improvement over trial-018)",
    "trial-028": "PLATEAU (round-2 reg: output_dropout=0.1 off trial-022, worst of round 2)",
    "trial-029": "PLATEAU (round-2: grad_accum_steps=4 off trial-022, no improvement over trial-022)",
    "trial-030": "MILD OVERFITTING (35M-scale seed, 6 epochs: best val_loss 0.5946 at epoch 4, then val_loss rises 0.5946->0.5959 while train_loss keeps falling - same shape as trial-031, much weaker; per user's val_loss>0.595 noise-floor call, this is the only 8M/35M-comparable result besides trial-031 to clear it)",
    "trial-031": "OVERFITTING (35M-scale, zero regularization: best val_loss 0.5918 at epoch 4 - then val_loss rises while train_loss keeps falling through epoch 6; first clear overfitting signal in the whole sweep)",
    "trial-032": "MILD OVERFITTING (batch_size=32 off trial-031: best val_loss 0.5916 at epoch 4, ties trial-031; slower per-epoch than batch=8 despite fewer steps, likely padding-waste from random per-batch length padding)",
    "trial-033": "MILD OVERFITTING (batch_size=128 off trial-031: best val_loss 0.5902 at epoch 5 - new best of the entire sweep, despite ~94 opt steps/epoch (16x fewer than trial-031); user reports this trial nearly OOM'd the T4)",
    "trial-034": "STILL CONVERGING (batch_size=128 + grad_accum_steps=4 off trial-031: best val_loss 0.5960 at epoch 6, its last epoch and still falling - above the 0.595 noise floor, so not yet a real result; too few optimizer steps (~23/epoch) to have converged in 6 epochs)",
    "trial-035": "MILD OVERFITTING (light regularization off trial-030, dropout=0.1/wd=0.1: best val_loss 0.5914 at epoch 5 - 2nd best of the sweep, beats both trial-030's heavy reg and roughly ties trial-031's zero reg)",
    "trial-036": "MILD OVERFITTING (wd=0.0 with dropout=0.2 off trial-030: best val_loss 0.5937 at epoch 4 - better than trial-030 but worse than trial-035's lighter-both combo)",
    "trial-037": "MILD OVERFITTING (unfreeze_last_n=2 off trial-030: best val_loss 0.5936 at epoch 4, ~15% slower per epoch than unfreeze=1)",
    "trial-038": "NOISY (batch_size=128 off trial-033, epochs=10 rerun: best val_loss 0.5891 at epoch 7 - new best of the entire sweep - but val_loss oscillates rather than cleanly rising after its epoch-5 local min (0.5902->0.5922->0.5891->0.5923...), consistent with few (~94) optimizer steps/epoch adding noise, not a clean overfitting curve)",
    "trial-039": "NOISY (batch=128, dropout=0.1/wd=0.1 off trial-033, epochs=10 rerun: best val_loss 0.5900 at epoch 7, same oscillating pattern as trial-038)",
    "trial-040": "NOISY (batch_size=64 off trial-033: best val_loss 0.5905 at epoch 7, same oscillating pattern, milder)",
    "trial-041": "OVERFITTING (rerun, complete: dropout=0.1/wd=0.1, batch=8 - best val_loss 0.5914 at epoch 5, then rises clearly through epoch 10 (0.6088) while train_loss keeps falling - the harsh late-epoch overfit the user expected, now visible at the full 10 epochs)",
    "trial-042": "OVERFITTING (dropout=0.05/wd=0.05 off trial-035, batch=8: best val_loss 0.5929 at epoch 4 - worse than trial-035/041's 0.1/0.1, so 0.1/0.1 is closer to the regularization sweet spot than going lighter still; also overfits clearly by epoch 10 (0.6046))",
    "trial-043": "OVERFITTING (unfreeze_last_n=2 + dropout=0.1/wd=0.1 off trial-035: best val_loss 0.5923 at epoch 3, worse than trial-041's unfreeze=1 (0.5914) and the most severe late-epoch overfit of the round - val_loss jumps to 0.6166/0.6249 at epochs 9-10)",
}

CONFIG_RE = re.compile(r"^\[CONFIG\] (\{.*\})$")
EPOCH_RE = re.compile(
    r"^Epoch (\d+)/(\d+) \| train_loss=([\d.]+) \| val_loss=([\d.]+) \| "
    r"train_acc=([\d.]+) \| val_acc=([\d.]+) \| train_recall=([\d.]+) \| val_recall=([\d.]+)"
)

FIELDS = [
    "trial_id", "verdict", "model_name", "batch_size", "unfreeze_last_n", "learning_rate",
    "output_dropout", "esm_dropout", "attention_dropout", "weight_decay",
    "lr_schedule", "warmup_steps", "label_smoothing", "decouple_bias_wd", "grad_accum_steps",
    "max_train_samples", "max_val_samples", "epochs_completed",
    "best_epoch", "best_val_loss", "best_val_acc", "best_val_recall",
    "best_train_loss", "best_train_acc", "n_log_fragments",
]


def trial_id_from_filename(name: str) -> str:
    # trial-012_epoch5.log -> trial-012 ; trial-014_epochs5-14.log -> trial-014 ; trial-015.log -> trial-015
    stem = name[:-4] if name.endswith(".log") else name
    return re.sub(r"_epochs?\d+(-\d+)?$", "", stem)


def parse_log(path: Path):
    config = None
    epochs = []
    for line in path.read_text(errors="replace").splitlines():
        m = CONFIG_RE.match(line)
        if m:
            try:
                config = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
            continue
        m = EPOCH_RE.match(line)
        if m:
            epoch_num, _total, train_loss, val_loss, train_acc, val_acc, train_recall, val_recall = m.groups()
            epochs.append({
                "epoch": int(epoch_num),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "train_recall": float(train_recall),
                "val_recall": float(val_recall),
            })
    return config, epochs


def main() -> None:
    by_trial = defaultdict(lambda: {"configs": [], "epochs": [], "n_fragments": 0})

    for log_path in sorted(LOG_DIR.glob("trial-*.log")):
        trial_id = trial_id_from_filename(log_path.name)
        config, epochs = parse_log(log_path)
        entry = by_trial[trial_id]
        entry["n_fragments"] += 1
        if config is not None:
            entry["configs"].append(config)
        entry["epochs"].extend(epochs)

    rows = []
    for trial_id in sorted(by_trial, key=lambda t: (len(t), t)):
        data = by_trial[trial_id]
        if not data["epochs"] or not data["configs"]:
            continue
        config = data["configs"][0]
        best = min(data["epochs"], key=lambda e: e["val_loss"])
        rows.append({
            "trial_id": trial_id,
            "verdict": VERDICTS.get(trial_id, "UNVERIFIED - add to VERDICTS in this script"),
            "model_name": config.get("model_name"),
            "batch_size": config.get("batch_size"),
            "unfreeze_last_n": config.get("unfreeze_last_n"),
            "learning_rate": config.get("learning_rate"),
            "output_dropout": config.get("output_dropout"),
            "esm_dropout": config.get("esm_dropout"),
            "attention_dropout": config.get("attention_dropout"),
            "weight_decay": config.get("weight_decay"),
            "lr_schedule": config.get("lr_schedule", "constant"),
            "warmup_steps": config.get("warmup_steps"),
            "label_smoothing": config.get("label_smoothing", 0.0),
            "decouple_bias_wd": config.get("decouple_bias_wd", False),
            "grad_accum_steps": config.get("grad_accum_steps", 1),
            "max_train_samples": config.get("max_train_samples"),
            "max_val_samples": config.get("max_val_samples"),
            "epochs_completed": max(e["epoch"] for e in data["epochs"]),
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "best_val_acc": best["val_acc"],
            "best_val_recall": best["val_recall"],
            "best_train_loss": best["train_loss"],
            "best_train_acc": best["train_acc"],
            "n_log_fragments": data["n_fragments"],
        })

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} trial(s) to {OUT_PATH}")


if __name__ == "__main__":
    main()
