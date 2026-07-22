"""Verify the CAFA6 dataset's `labels` column is genuinely integer-encoded (not corrupted by the
stringify bug that was found in training_run.py's old preprocess_batch). Checks both the HF Hub
copy and a local on-disk copy if present. This is a read-only diagnostic; it does not modify or
regenerate anything.
"""

import sys
from pathlib import Path

from datasets import load_dataset, load_from_disk


def check(name: str, dataset_train) -> None:
    print(f"=== {name} ===")
    label_feature = dataset_train.features["labels"].feature
    names = label_feature.names
    print(f"ClassLabel names: {names}")

    raw_labels = dataset_train["labels"][:2000]
    n_int = 0
    n_str = 0
    n_other = 0
    all_zero = True
    for labs in raw_labels:
        if len(labs) > 0:
            all_zero = False
        for item in labs:
            if isinstance(item, bool):
                n_other += 1
            elif isinstance(item, int):
                n_int += 1
            elif isinstance(item, str):
                n_str += 1
            else:
                n_other += 1

    print(f"sampled {len(raw_labels)} rows")
    print(f"label items: int={n_int} str={n_str} other={n_other}")
    print(f"any row with a non-empty label list: {not all_zero}")
    if n_int:
        idxs = {item for labs in raw_labels for item in labs if isinstance(item, int)}
        out_of_range = {i for i in idxs if not (0 <= i < len(names))}
        print(f"distinct int label ids seen: {sorted(idxs)}")
        print(f"out-of-range ids (would indicate real corruption): {sorted(out_of_range) or 'none'}")
    print()


def main() -> None:
    print("Checking Hugging Face Hub dataset: kelmoir/cafa6\n")
    hub_ds = load_dataset("kelmoir/cafa6")
    check("HF Hub (kelmoir/cafa6)", hub_ds["train"])

    local_path = Path("data/cafa-6-protein-function-prediction/dataset")
    if local_path.exists():
        print(f"Checking local dataset at {local_path}\n")
        local_ds = load_from_disk(str(local_path))
        check(f"local ({local_path})", local_ds["train"])
    else:
        print(f"No local dataset found at {local_path} — skipping local check.")


if __name__ == "__main__":
    sys.exit(main())
