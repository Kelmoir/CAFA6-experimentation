"""Item 14 step 1 (experiments/cafa6_next_steps_plan.md): attention-map validity check.

For proteins where the main pipeline's classifier (trial-013 checkpoint) correctly predicts an MF
term with a known, tightly-localized residue span, check whether the model's attention actually
concentrates on that span more than chance -- a cheap interpretability probe, no training.

Ground truth: experiments/mf_span/positional_mf_links.json (82,396 proteins -> localized MF-term
spans), reused read-only from the shelved mf_span project per the plan's own note -- this script
never trains or scores that project's dedicated span-prediction model. `domain_whole_protein_fallback`
entries (start=0, end=sequence_length, i.e. no real localization) are excluded; the remaining
sources (binding_ligand, text_pattern, domain_bridge, domain_fallback_union, feature_type) are
genuinely narrower than the full sequence (median width 1-160 residues depending on source).

Metric: for each (protein, term) pair where the model predicts prob > PRED_THRESHOLD, take the
last layer's CLS-token attention row (mean over heads) restricted to residue positions, and rank
the ground-truth span's mean attention against every other equal-width sliding window in the same
protein (via a cumsum trick) -- this controls for both span width and any protein-level positional
attention bias (e.g. always attending near the N-terminus) without needing a separate null model.
Chance = 50th percentile. Also reports the same percentile averaged over all layers, not just the
last one, since earlier layers still feed into what the last layer's CLS attention can draw on.

Capped to sequences <=SEQLEN_CAP residues (full output_attentions tensors are per-layer
heads*seq*seq -- quadratic, not worth lifting the cap for a diagnostic probe) and a random sample
of MAX_PROTEINS matching proteins (plenty of candidates remain under the cap: ~6,800 validation
proteins, ~54,500 candidate pairs at SEQLEN_CAP=1024).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = REPO_ROOT / "data/cafa-6-protein-function-prediction/dataset_full"
CHECKPOINT_DIR = REPO_ROOT / "models/full-label-trial013-trainNfull/best_checkpoint"
LINKS_PATH = REPO_ROOT / "experiments/mf_span/positional_mf_links.json"

NONFALLBACK_SOURCES = {"binding_ligand", "text_pattern", "domain_bridge",
                        "domain_fallback_union", "feature_type"}
SEQLEN_CAP = 1024
MAX_PROTEINS = 500
PRED_THRESHOLD = 0.5
SEED = 42


def load_candidates(label2id: dict, prot2seq: dict) -> dict:
    with open(LINKS_PATH) as f:
        links = json.load(f)
    candidates = {}
    for pid, rec in links.items():
        seq = prot2seq.get(pid)
        if seq is None:
            continue
        seqlen = len(seq)
        if seqlen > SEQLEN_CAP:
            continue
        for e in rec["localized"]:
            if e["source"] not in NONFALLBACK_SOURCES:
                continue
            if e["go_id"] not in label2id:
                continue
            start, end = e["start"], e["end"]
            if start < 0 or end <= start or end > seqlen:
                continue
            candidates.setdefault(pid, []).append((e["go_id"], start, end, e["source"]))
    return candidates


def window_percentile(attn: np.ndarray, start: int, end: int) -> float:
    """Percentile rank of the span's mean attention among every other equal-width window in the
    same sequence -- controls for span width and protein-level positional attention bias."""
    w = end - start
    n = len(attn)
    cs = np.concatenate(([0.0], np.cumsum(attn)))
    window_means = (cs[w:] - cs[:-w]) / w  # length n-w+1
    true_mean = (cs[end] - cs[start]) / w
    return 100.0 * float((window_means <= true_mean).mean())


def main():
    print("[INFO] Loading dataset_full validation split...")
    dataset = load_from_disk(str(DATA_PATH))
    val_split = dataset["validation"]
    label_names = [str(name) for name in val_split.features["labels"].feature.names]
    label2id = {name: i for i, name in enumerate(label_names)}
    prot2seq = dict(zip(val_split["protein"], val_split["sequence"]))

    print("[INFO] Matching against positional_mf_links.json...")
    candidates = load_candidates(label2id, prot2seq)
    print(f"[INFO] {len(candidates)} validation proteins have >=1 usable localized MF span "
          f"(seqlen<={SEQLEN_CAP}, non-fallback source, in-vocab term)")

    rng = random.Random(SEED)
    sampled_proteins = rng.sample(sorted(candidates), min(MAX_PROTEINS, len(candidates)))
    print(f"[INFO] sampled {len(sampled_proteins)} proteins")

    print("[INFO] Loading trial-013 checkpoint (CPU)...")
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model.eval()

    last_layer_pctl, all_layer_pctl = [], []
    n_pairs_total, n_pairs_correct = 0, 0

    with torch.inference_mode():
        for i, pid in enumerate(sampled_proteins):
            seq = prot2seq[pid]
            seqlen = len(seq)
            enc = tokenizer(seq, return_tensors="pt")
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        output_attentions=True)
            probs = torch.sigmoid(out.logits)[0]

            attentions = out.attentions  # tuple of [1, heads, seq_tok, seq_tok], one per layer
            last_layer_attn = attentions[-1][0].mean(dim=0)[0, 1:1 + seqlen].numpy()
            all_layer_attn = torch.stack(attentions).mean(dim=0)[0].mean(dim=0)[0, 1:1 + seqlen].numpy()

            for go_id, start, end, source in candidates[pid]:
                n_pairs_total += 1
                j = label2id[go_id]
                if float(probs[j]) <= PRED_THRESHOLD:
                    continue
                n_pairs_correct += 1
                last_layer_pctl.append(window_percentile(last_layer_attn, start, end))
                all_layer_pctl.append(window_percentile(all_layer_attn, start, end))

            if (i + 1) % 50 == 0:
                print(f"[INFO] {i + 1}/{len(sampled_proteins)} proteins processed, "
                      f"{len(last_layer_pctl)} correctly-predicted pairs so far", flush=True)

    print(f"[RESULT] n_candidate_pairs={n_pairs_total} "
          f"n_correctly_predicted_pairs={n_pairs_correct} "
          f"({100 * n_pairs_correct / max(n_pairs_total, 1):.1f}% coverage)")
    last_arr, all_arr = np.array(last_layer_pctl), np.array(all_layer_pctl)
    print(f"[RESULT] last-layer CLS attention percentile of GT span vs equal-width windows: "
          f"mean={last_arr.mean():.1f} median={np.median(last_arr):.1f} (chance=50.0)")
    print(f"[RESULT] all-layer-mean CLS attention percentile of GT span vs equal-width windows: "
          f"mean={all_arr.mean():.1f} median={np.median(all_arr):.1f} (chance=50.0)")


if __name__ == "__main__":
    main()
