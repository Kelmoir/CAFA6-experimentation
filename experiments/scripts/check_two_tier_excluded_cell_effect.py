"""Direct check of whether the two-tier mask actually changed model belief on the cells it
targeted -- not just whether overall AUROC moved.

Why this is a different (and more relevant) question than the AUROC comparison
(compare_checkpoints_per_term_auroc.py) already ran: run_diagnostic() and that script both score
AUROC against dataset_raw's ORIGINAL, unmodified hard labels -- the SAME incomplete annotations the
whole two-tier exercise is built to route around. The cells build_two_tier_dataset.py excluded from
the loss (TIcE-and-classifier-flagged as likely-hidden-positive) are STILL scored as ground-truth
negative in that evaluation. So if the mechanism worked -- the two-tier model became less confident
those specific cells are negative, precisely because training stopped punishing it for that -- the
existing AUROC metric would register that as a FALSE POSITIVE against the old incomplete labels,
not a win. Near-zero aggregate AUROC movement is therefore uninformative about whether the
mechanism did anything at all; it could mean "no effect" or "a real effect exactly offset by being
scored as wrong by the same incomplete ground truth it was designed to question."

This script checks the mechanism directly instead: for each targeted term, split VALIDATION rows
(never used in any gradient update, for either checkpoint) into three groups --
  - known positive (label == 1)
  - confirmed negative, NOT excluded (label == 0, not flagged by build_two_tier_dataset.py)
  - excluded / likely-hidden-positive (label == 0, flagged and excluded from the TRAIN split's
    loss for this term -- note: a row can be excluded in train without being excluded in
    validation, since the mask was computed independently per split; this script only looks at
    validation's OWN excluded flag, so any shift found here is the model generalizing something
    learned from train, not memorizing these exact rows)
and reports each group's mean predicted probability under the baseline vs. two-tier checkpoint. If
the mechanism is doing what it's supposed to, the "excluded" group's probability should rise more
(or at least differently) under the two-tier model than the "confirmed negative" group's does.

Also relevant to the self-training/bootstrapping concern this whole design carries: the exclusion
mask was built from the BASELINE checkpoint's own predicted probabilities (see
build_two_tier_dataset.py). A shift found here confirms the two-tier model moved in the direction
its own training target pushed it -- expected by construction, not independent proof the excluded
cells are truly positive. It does NOT resolve whether the baseline's original ranking was trustworthy
in the first place; that would need genuinely independent evidence (e.g. literature-confirmed
annotations added after this dataset's snapshot), which isn't available here.

Usage:
  env/bin/python experiments/scripts/check_two_tier_excluded_cell_effect.py \\
      --checkpoint-a models/pu-baseline-trial001/best_checkpoint \\
      --checkpoint-b models/pu-two-tier-trial001/best_checkpoint
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent))
from compare_checkpoints_per_term_auroc import score_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-a", default="models/pu-baseline-trial001/best_checkpoint")
    parser.add_argument("--checkpoint-b", default="models/pu-two-tier-trial001/best_checkpoint")
    parser.add_argument("--two-tier-data-path", default="data/cafa-6-protein-function-prediction/dataset_raw_two_tier")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()
    import torch
    device = torch.device(args.device)

    print(f"[INFO] Scoring checkpoint A (baseline): {args.checkpoint_a}")
    probs_a, hard_labels, label_names = score_checkpoint(args.checkpoint_a, args.two_tier_data_path, None,
                                                          args.batch_size, device, args.max_length)
    print(f"[INFO] Scoring checkpoint B (two-tier): {args.checkpoint_b}")
    probs_b, hard_labels_b, _ = score_checkpoint(args.checkpoint_b, args.two_tier_data_path, None,
                                                  args.batch_size, device, args.max_length)
    assert np.array_equal(hard_labels, hard_labels_b), "validation labels differ between scoring runs"

    dataset = load_from_disk(args.two_tier_data_path)
    val = dataset["validation"]
    excluded_col = val["excluded_labels"]
    label2id = {n: i for i, n in enumerate(label_names)}

    print(f"{'term':14s} {'group':28s} {'n':>6s} {'prob_a(base)':>13s} {'prob_b(2tier)':>14s} {'shift':>8s}")
    for term in sorted({label_names[t] for row in excluded_col for t in row}):
        idx = label2id[term]
        excluded_mask = np.array([idx in row for row in excluded_col])
        y = hard_labels[:, idx]
        groups = {
            "known_positive (label=1)": (y == 1),
            "confirmed_negative (label=0, kept)": (y == 0) & (~excluded_mask),
            "excluded (label=0, flagged hidden-pos)": (y == 0) & excluded_mask,
        }
        for gname, mask in groups.items():
            n = int(mask.sum())
            if n == 0:
                continue
            pa = float(probs_a[mask, idx].mean())
            pb = float(probs_b[mask, idx].mean())
            print(f"{term:14s} {gname:28s} {n:6d} {pa:13.4f} {pb:14.4f} {pb - pa:+8.4f}")
        print("-" * 100)


if __name__ == "__main__":
    main()
