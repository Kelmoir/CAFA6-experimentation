"""Diagnose whether mean-pooling washes out short, localized molecular_function (MF) signal.

Context: KM2 (experiments/scripts/estimate_class_priors_km2.py) failed empirically -- D(kappa)
never plateaus/rises for any term checked, in either mean-pooled backbone embeddings or classifier
probabilities (see pu_learning_plan.md's v1.1 note). One live hypothesis for *why* mean-pooling in
particular might lack per-term structure: many MF terms are caused by a short motif within the
sequence (a zinc-finger, a P-loop/Walker A ATP-binding motif, a G-domain), not a property of the
whole protein -- averaging that signal over up to 1024 residues could dilute it into noise. Other
MF terms (DNA-binding domains, a ribosomal structural role) plausibly involve more of the sequence
and shouldn't suffer the same dilution.

This script tests that directly: for a handful of MF terms split into "short-motif" and
"broader-role" groups, it computes BOTH mean-pooled and MAX-pooled (per-dimension max over real,
non-padding tokens -- the standard "does ANY single position show a strong signal" alternative to
averaging) per-protein embeddings from the same baseline checkpoint, then checks two independent
separability signals for each (term, pooling) pair:

  1. Nearest-centroid cosine-similarity AUROC: split each term's positives in half, build a
     centroid from one half, rank the other half + a random background sample by cosine similarity
     to that centroid, report AUROC. Directly answers "does this embedding space cluster this
     term's positives at all" without KM2's kernel-QP machinery in the way.
  2. The same KM2 D(kappa) curve check used to diagnose the original failure (imported from
     estimate_class_priors_km2.py, kappa grid extended to 1.0): monotonic decrease all the way to
     kappa=1 is the failure signature already established; an interior minimum with a rise after it
     is the "real signal" signature the paper's theory predicts.

Does NOT touch the production pipeline (extract_protein_embeddings.py / estimate_class_priors_km2.py
/ training_run.py) -- this is a standalone diagnostic to decide whether pooling strategy is worth
revisiting before spending more effort on PU learning at all.

Usage (GPU strongly recommended -- see feedback_no_long_cpu_runs_prefer_gpu memory):
  env/bin/python experiments/scripts/diagnose_mf_pooling.py --device cuda
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, EsmTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from estimate_class_priors_km2 import (  # noqa: E402
    rbf_kernel_matrix_torch,
    frank_wolfe_qp_batched,
    median_heuristic_bandwidth,
)

# (term, hypothesis group, human label) -- picked from the 437 KM2-eligible terms, spanning the
# "how localized is the causal motif" spectrum. Short-motif candidates are classic examples of a
# function carried by a small, well-characterized structural element; broader-role candidates
# plausibly depend on more of the sequence (a full DNA-binding domain, or a whole-protein
# structural/assembly role), so should suffer less from mean-pooling dilution if the hypothesis
# is right.
DEFAULT_TERMS = [
    ("GO:0008270", "short_motif", "zinc ion binding (zinc-finger motif)"),
    ("GO:0005524", "short_motif", "ATP binding (P-loop/Walker A motif)"),
    ("GO:0003924", "short_motif", "GTPase activity (G-domain)"),
    ("GO:0003677", "broader_role", "DNA binding"),
    ("GO:0003735", "broader_role", "structural constituent of ribosome"),
    ("GO:0004674", "broader_role", "protein serine/threonine kinase activity"),
]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def max_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).bool()
    masked = last_hidden_state.masked_fill(~mask, float("-inf"))
    pooled = masked.max(dim=1).values
    # Guard fully-masked rows (shouldn't happen with real sequences, but avoid -inf leaking out).
    pooled = torch.nan_to_num(pooled, neginf=0.0)
    return pooled


def nearest_centroid_auroc(pos_embeddings: np.ndarray, bg_embeddings: np.ndarray, seed: int) -> float:
    """Split positives in half; build a centroid from half A; rank half B + bg by cosine
    similarity to that centroid; report AUROC (half-B positives should rank above background)."""
    rng = np.random.default_rng(seed)
    n_pos = pos_embeddings.shape[0]
    if n_pos < 4:
        return float("nan")
    idx = rng.permutation(n_pos)
    half_a, half_b = idx[: n_pos // 2], idx[n_pos // 2:]
    centroid = pos_embeddings[half_a].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    def cos_sim(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return (x / norms) @ centroid

    scores_pos = cos_sim(pos_embeddings[half_b])
    scores_bg = cos_sim(bg_embeddings)
    labels = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_bg)])
    scores = np.concatenate([scores_pos, scores_bg])
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    n_p, n_n = labels_sorted.sum(), len(labels_sorted) - labels_sorted.sum()
    if n_p == 0 or n_n == 0:
        return float("nan")
    ranks = np.arange(1, len(labels_sorted) + 1)
    sum_ranks_pos = ranks[labels_sorted == 1].sum()
    # Mann-Whitney U / AUROC from rank-sum (labels_sorted is by descending score, so "rank 1" is
    # the highest score -- flip to the usual ascending-rank convention before applying the formula).
    ranks_ascending = len(labels_sorted) + 1 - ranks
    sum_ranks_pos = ranks_ascending[labels_sorted == 1].sum()
    auroc = (sum_ranks_pos - n_p * (n_p + 1) / 2) / (n_p * n_n)
    return float(auroc)


def km2_shape_summary(pos_embeddings: torch.Tensor, mixture_t: torch.Tensor, k_mm: torch.Tensor,
                       bandwidth: float, kappa_grid_t: torch.Tensor, fw_iters: int) -> dict:
    k_mp_mean = rbf_kernel_matrix_torch(mixture_t, pos_embeddings, bandwidth).mean(dim=1)
    k_pp_mean = rbf_kernel_matrix_torch(pos_embeddings, pos_embeddings, bandwidth).mean()
    d_curve = (frank_wolfe_qp_batched(k_mm, k_mp_mean, kappa_grid_t, max_iters=fw_iters)
               + kappa_grid_t ** 2 * k_pp_mean)
    d_curve = torch.clamp(d_curve, min=0).cpu().numpy()
    argmin_idx = int(np.argmin(d_curve))
    monotonic_decreasing = bool(np.all(np.diff(d_curve) <= 1e-9))
    return {
        "d_curve": d_curve.tolist(),
        "argmin_kappa": float(kappa_grid_t.cpu().numpy()[argmin_idx]),
        "argmin_at_edge": argmin_idx == len(d_curve) - 1,
        "monotonic_decreasing": monotonic_decreasing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", default="models/pretrain-reduced-full-protein-v8/best_checkpoint")
    parser.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset_raw")
    parser.add_argument("--mixture-sample-size", type=int, default=5000,
                         help="Same draw (seed-reproduced) as estimate_class_priors_km2.py's default run, "
                              "so this is an apples-to-apples comparison against the already-diagnosed failure.")
    parser.add_argument("--max-positives-per-term", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--kappa-steps", type=int, default=26)
    parser.add_argument("--fw-iters", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output-dir", default="experiments/pu_cache/mf_pooling_diag")
    parser.add_argument("--terms", default=None,
                         help="Override: comma-separated GO:ids from DEFAULT_TERMS' pool; default uses all 6.")
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    terms = DEFAULT_TERMS
    if args.terms:
        wanted = set(args.terms.split(","))
        terms = [t for t in DEFAULT_TERMS if t[0] in wanted]

    print(f"[INFO] Loading checkpoint from {args.checkpoint_path}")
    tokenizer = EsmTokenizer.from_pretrained(args.checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path).to(device)
    model.eval()
    label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    label2id = {name: i for i, name in enumerate(label_names)}

    dataset = load_from_disk(args.data_path)
    splits = ["train", "validation"]
    protein_ids, hard_label_idx, sequences_all = [], [], []
    for split in splits:
        rows = dataset[split]
        protein_ids.extend(rows["protein"])
        hard_label_idx.extend(rows["labels"])
        sequences_all.extend(rows["sequence"])
    n = len(protein_ids)
    print(f"[INFO] {n} proteins across {splits}")

    # Reproduce estimate_class_priors_km2.py's exact mixture draw (same seed, same size, same
    # underlying protein order) so this diagnostic is directly comparable to the original failure.
    rng = np.random.default_rng(args.seed)
    mixture_idx = rng.choice(n, size=min(args.mixture_sample_size, n), replace=False)
    mixture_id_set = set(mixture_idx.tolist())

    term_positive_rows = {}
    for term_name, _, _ in terms:
        tid = label2id[term_name]
        rows = [i for i, labels in enumerate(hard_label_idx) if tid in labels]
        if len(rows) > args.max_positives_per_term:
            rows = [int(r) for r in rng.choice(rows, size=args.max_positives_per_term, replace=False)]
        term_positive_rows[term_name] = rows
        print(f"[INFO] {term_name}: {len(rows)} positive rows selected")

    needed_rows = sorted(mixture_id_set.union(*term_positive_rows.values()))
    row_to_pos = {row: pos for pos, row in enumerate(needed_rows)}
    print(f"[INFO] {len(needed_rows)} unique proteins need a forward pass "
          f"(mixture {len(mixture_id_set)} + term positives, deduped)")

    mean_chunks, max_chunks = [], []
    start = time.time()
    for batch_start in range(0, len(needed_rows), args.batch_size):
        batch_rows = needed_rows[batch_start: batch_start + args.batch_size]
        sequences = [sequences_all[r] for r in batch_rows]
        tokenized = tokenizer(sequences, max_length=args.max_length, truncation=True, padding=True, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.inference_mode():
            outputs = model.esm(input_ids=input_ids, attention_mask=attention_mask)
            mean_chunks.append(mean_pool(outputs.last_hidden_state, attention_mask).cpu().numpy().astype(np.float32))
            max_chunks.append(max_pool(outputs.last_hidden_state, attention_mask).cpu().numpy().astype(np.float32))
        if (batch_start // args.batch_size) % 20 == 0:
            elapsed = time.time() - start
            print(f"[STEP] {batch_start + len(batch_rows)}/{len(needed_rows)} elapsed={elapsed:.0f}s")

    mean_embeddings = np.concatenate(mean_chunks, axis=0)
    max_embeddings = np.concatenate(max_chunks, axis=0)
    print(f"[INFO] Forward passes done in {time.time() - start:.0f}s")

    np.save(output_dir / "mean_embeddings.npy", mean_embeddings)
    np.save(output_dir / "max_embeddings.npy", max_embeddings)
    with open(output_dir / "meta.json", "w") as f:
        json.dump({"needed_rows": needed_rows, "checkpoint_path": args.checkpoint_path}, f)

    mixture_pos = [row_to_pos[r] for r in mixture_idx]
    mixture_mean_t = torch.as_tensor(mean_embeddings[mixture_pos], device=device)
    mixture_max_t = torch.as_tensor(max_embeddings[mixture_pos], device=device)
    bandwidth_mean = median_heuristic_bandwidth(mean_embeddings[mixture_pos], seed=args.seed)
    bandwidth_max = median_heuristic_bandwidth(max_embeddings[mixture_pos], seed=args.seed)
    k_mm_mean = rbf_kernel_matrix_torch(mixture_mean_t, mixture_mean_t, bandwidth_mean)
    k_mm_max = rbf_kernel_matrix_torch(mixture_max_t, mixture_max_t, bandwidth_max)
    kappa_grid_np = np.linspace(0.0, 1.0, args.kappa_steps)
    kappa_grid_t = torch.as_tensor(kappa_grid_np, device=device, dtype=k_mm_mean.dtype)

    print()
    print("=" * 100)
    print(f"{'term':<14}{'group':<14}{'pooling':<8}{'n_pos':<8}{'centroid_AUROC':<16}"
          f"{'KM2_argmin_kappa':<18}{'KM2_monotonic_decr':<20}")
    print("=" * 100)
    results = {}
    for term_name, group, label in terms:
        rows = term_positive_rows[term_name]
        pos_positions = [row_to_pos[r] for r in rows]
        bg_positions = [p for p in mixture_pos if p not in set(pos_positions)]

        for pool_name, embeddings, mixture_t, k_mm, bandwidth in (
            ("mean", mean_embeddings, mixture_mean_t, k_mm_mean, bandwidth_mean),
            ("max", max_embeddings, mixture_max_t, k_mm_max, bandwidth_max),
        ):
            pos_emb = embeddings[pos_positions]
            bg_emb = embeddings[bg_positions]
            auroc = nearest_centroid_auroc(pos_emb, bg_emb, seed=args.seed)
            pos_emb_t = torch.as_tensor(pos_emb, device=device, dtype=k_mm.dtype)
            km2 = km2_shape_summary(pos_emb_t, mixture_t, k_mm, bandwidth, kappa_grid_t, args.fw_iters)
            results[f"{term_name}_{pool_name}"] = {
                "term": term_name, "group": group, "label": label, "pooling": pool_name,
                "n_positives": len(rows), "centroid_auroc": auroc, **km2,
            }
            print(f"{term_name:<14}{group:<14}{pool_name:<8}{len(rows):<8}{auroc:<16.4f}"
                  f"{km2['argmin_kappa']:<18.4f}{str(km2['monotonic_decreasing']):<20}")
        print("-" * 100)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] Wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
