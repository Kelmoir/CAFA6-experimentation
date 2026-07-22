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
    "trial-015": "PLATEAU (stable, best trajectory so far)",
    "trial-016": "PLATEAU (matches trial-015, unfreeze bump had no effect)",
}

CONFIG_RE = re.compile(r"^\[CONFIG\] (\{.*\})$")
EPOCH_RE = re.compile(
    r"^Epoch (\d+)/(\d+) \| train_loss=([\d.]+) \| val_loss=([\d.]+) \| "
    r"train_acc=([\d.]+) \| val_acc=([\d.]+) \| train_recall=([\d.]+) \| val_recall=([\d.]+)"
)

FIELDS = [
    "trial_id", "verdict", "model_name", "unfreeze_last_n", "learning_rate",
    "output_dropout", "esm_dropout", "attention_dropout", "weight_decay",
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
            "unfreeze_last_n": config.get("unfreeze_last_n"),
            "learning_rate": config.get("learning_rate"),
            "output_dropout": config.get("output_dropout"),
            "esm_dropout": config.get("esm_dropout"),
            "attention_dropout": config.get("attention_dropout"),
            "weight_decay": config.get("weight_decay"),
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
