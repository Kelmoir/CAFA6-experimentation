"""Score a trained CAFA6 checkpoint against the held-out test (or validation) split using the
same scoring formula the actual Kaggle CAFA6 competition uses: information-accretion-weighted
precision/recall, swept across confidence thresholds, reduced to the maximum F-measure (Fmax) --
the standard CAFA assessment methodology (see https://github.com/BioComputingUP/CAFA-evaluator),
driven here by this repo's own `data/cafa-6-protein-function-prediction/IA.tsv`.

IMPORTANT SCOPE CAVEAT: this replicates the official *scoring formula* exactly, but not the
official *evaluation universe*. The real competition scores a submission against the full GO
term space (tens of thousands of terms); this script can only ever score a model over the
label vocabulary it was actually trained to predict -- `training_run.py`'s 383-term coarse
label space (see CLAUDE.md's "Label space" architecture note), optionally further restricted
via --go-scope. Treat the numbers here as "how good is my model at the task it was trained on,
measured the CAFA way" -- not as a stand-in for a real leaderboard score.

Ground truth comes from this repo's own held-out dataset split (`generate_dataset.py`'s
train/validation/test DatasetDict), never the raw Kaggle `Test/` directory -- that one ships
with no labels at all (it's the blind competition test set), so nothing could be scored against
it locally.

Usage:
  # Score the promoted main-branch checkpoint against the full 383-term label space
  python evaluate_cafa_score.py

  # Score a specific trial branch, restricted to the terms the mf-span side-experiment covers
  python evaluate_cafa_score.py --revision trial-050 --go-scope mf-span

  # Just the perfect-predictor sanity check, no model/GPU needed
  python evaluate_cafa_score.py --skip-model

  # Dump the full threshold curve + a submission-shaped prediction TSV for inspection
  python evaluate_cafa_score.py --results-out experiments/logs/cafa_score_main.json \\
      --predictions-out /tmp/predictions.tsv
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import generate_dataset
import training_run

DATA_DIR = Path("data/cafa-6-protein-function-prediction")
DEFAULT_IA_FILE = DATA_DIR / "IA.tsv"
DEFAULT_DEP_GRAPH_FILE = DATA_DIR / "dataset" / "go_dependency_graph.json"
DEFAULT_MF_SPAN_VOCAB_FILE = Path("experiments/positional_mf_go_term_texts.json")


# --------------------------------------------------------------------------------------------
# GO graph + Information Accretion loading
# --------------------------------------------------------------------------------------------

def load_true_path_graph(dep_graph_file: Path) -> nx.MultiDiGraph:
    """Rebuild the is_a/part_of-only GO graph generate_dataset.py already exported, rather than
    re-parsing go-basic.obo from scratch. Edge direction is child -> parent (verified against
    generate_dataset.py's own ancestors_of(), which calls nx.descendants() on this exact graph
    shape to get a term's ancestors) -- so nx.ancestors(graph, t) gives t's ontological
    *descendants* (more specific terms), which is what prediction-confidence propagation
    (propagate_predictions() below) needs.
    """
    data = json.loads(dep_graph_file.read_text())
    graph = nx.readwrite.json_graph.node_link_graph(data["graph"], edges="links")
    return generate_dataset.restrict_to_true_path(graph)


def load_ia_weights(ia_file: Path, label_names: List[str]) -> np.ndarray:
    weights = {}
    for line in ia_file.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2:
            weights[parts[0]] = float(parts[1])
    missing = [name for name in label_names if name not in weights]
    if missing:
        print(f"[WARN] {len(missing)} label term(s) missing from {ia_file}, defaulting IA=0.0 "
              f"(contributes nothing to weighted precision/recall): {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}")
    return np.array([weights.get(name, 0.0) for name in label_names], dtype=np.float64)


def build_descendant_index(true_path_graph: nx.MultiDiGraph, label_names: List[str]) -> Dict[int, List[int]]:
    """For each label column, the indices of every OTHER label column that is one of its
    ontological descendants (a more specific term). Used to max-propagate predicted confidence
    up the DAG before thresholding -- the same fix-up the official CAFA evaluator applies to a
    submission automatically, so a model that's confident about a specific child term but not
    yet confident about its parent isn't penalized for an inconsistency the true-path rule
    says shouldn't exist. Restricted to the label vocabulary itself (383 terms, or fewer): a
    term's descendants outside that vocabulary are neither predictable nor scoreable here, so
    they're irrelevant to this propagation step regardless of --go-scope (--go-scope only
    changes which label *columns get counted* in the final Fmax below, not this precompute).
    """
    label_set = set(label_names)
    name_to_idx = {name: i for i, name in enumerate(label_names)}
    descendant_idx = {}
    for i, name in enumerate(label_names):
        descendants = nx.ancestors(true_path_graph, name) & label_set if name in true_path_graph else set()
        descendant_idx[i] = [name_to_idx[d] for d in descendants if d != name]
    return descendant_idx


def propagate_predictions(probs: np.ndarray, descendant_idx: Dict[int, List[int]]) -> np.ndarray:
    propagated = probs.copy()
    for j, desc_idxs in descendant_idx.items():
        if desc_idxs:
            propagated[:, j] = np.maximum(propagated[:, j], probs[:, desc_idxs].max(axis=1))
    return propagated


def build_scope_mask(label_names: List[str], scope: str, mf_span_vocab_file: Path) -> np.ndarray:
    if scope == "all":
        return np.ones(len(label_names), dtype=bool)
    if scope == "mf-span":
        vocab = set(json.loads(mf_span_vocab_file.read_text()).keys())
        mask = np.array([name in vocab for name in label_names], dtype=bool)
        if not mask.any():
            raise RuntimeError(
                f"--go-scope mf-span matched none of the {len(label_names)} label terms against "
                f"{mf_span_vocab_file} -- vocab file empty/stale, or the wrong dataset was loaded?"
            )
        print(f"[INFO] --go-scope mf-span: {int(mask.sum())}/{len(label_names)} label terms overlap "
              f"the mf-span vocabulary ({mf_span_vocab_file})")
        return mask
    raise ValueError(f"Unknown --go-scope {scope!r}")


# --------------------------------------------------------------------------------------------
# CAFA weighted Fmax
# --------------------------------------------------------------------------------------------

def cafa_weighted_fmax(probs: np.ndarray, labels: np.ndarray, ia_weights: np.ndarray,
                        scope_mask: np.ndarray, thresholds: np.ndarray) -> Dict:
    """Official CAFA scoring: for each threshold, weighted precision is averaged over proteins
    with >=1 predicted term (m(t) of them); weighted recall is averaged over every protein that
    has any nonzero-IA true annotation in scope (n of them, fixed across thresholds -- a
    protein with a true annotation is always in the recall denominator, whether or not the
    model predicted anything for it). Fmax is the harmonic mean's maximum over the sweep.
    """
    probs = probs[:, scope_mask]
    labels = labels[:, scope_mask].astype(bool)
    ia = ia_weights[scope_mask]

    true_ia_sum = (labels * ia).sum(axis=1)
    eval_mask = true_ia_sum > 0
    n = int(eval_mask.sum())
    if n == 0:
        raise RuntimeError("No evaluated proteins have any nonzero-IA true annotation in this "
                            "--go-scope -- nothing to score.")
    if n < labels.shape[0]:
        print(f"[INFO] {labels.shape[0] - n}/{labels.shape[0]} proteins have no nonzero-IA true "
              f"annotation in this scope and are excluded from the benchmark set (n={n}), "
              f"matching standard CAFA practice of dropping no-knowledge targets.")

    probs, labels, true_ia_sum = probs[eval_mask], labels[eval_mask], true_ia_sum[eval_mask]

    best = {"fmax": 0.0, "threshold": None, "precision": 0.0, "recall": 0.0, "coverage": 0}
    curve = []
    for t in thresholds:
        pred = probs >= t
        has_pred = pred.any(axis=1)
        m = int(has_pred.sum())
        if m == 0:
            continue

        pred_ia_sum = (pred * ia).sum(axis=1)
        tp_ia_sum = ((pred & labels) * ia).sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            precision_per_protein = np.where(pred_ia_sum > 0, tp_ia_sum / pred_ia_sum, 0.0)
        wpr = precision_per_protein[has_pred].sum() / m

        recall_per_protein = tp_ia_sum / true_ia_sum
        wrc = recall_per_protein.sum() / n

        wf = 0.0 if (wpr + wrc) == 0 else 2 * wpr * wrc / (wpr + wrc)
        curve.append({"threshold": float(t), "precision": float(wpr), "recall": float(wrc),
                       "f1": float(wf), "coverage": m})
        if wf > best["fmax"]:
            best = {"fmax": wf, "threshold": float(t), "precision": float(wpr),
                    "recall": float(wrc), "coverage": m}

    return {**best, "n_proteins": n, "n_scope_terms": int(scope_mask.sum()), "curve": curve}


# --------------------------------------------------------------------------------------------
# Ground truth extraction (independent of tokenizer/model -- needed for the perfect-predictor
# baseline even with --skip-model, and used as the ground truth for a real model run too)
# --------------------------------------------------------------------------------------------

def extract_labels_matrix(hf_split, num_labels: int) -> np.ndarray:
    """hf_split['labels'] is a datasets.Sequence(ClassLabel) column: each row is a list of
    integer class ids (already the same 383-term id space training_run.py's id2label uses).
    """
    labels_col = hf_split["labels"]
    mat = np.zeros((len(labels_col), num_labels), dtype=np.float32)
    for i, ids in enumerate(labels_col):
        for idx in ids:
            mat[i, int(idx)] = 1.0
    return mat


# --------------------------------------------------------------------------------------------
# Model inference
# --------------------------------------------------------------------------------------------

def run_inference(model, tokenizer, hf_split, label_names: List[str], device: torch.device,
                   batch_size: int, autocast_dtype: Optional[torch.dtype], use_amp: bool,
                   use_aspect_input: bool) -> np.ndarray:
    label2id = {name: i for i, name in enumerate(label_names)}
    preprocess = partial(training_run.preprocess_batch, tokenizer=tokenizer,
                          max_length=training_run.MAX_LENGTH, unique_items=label_names,
                          label2id=label2id)
    eval_ds = hf_split.with_transform(transform=preprocess)
    loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        collate_fn=partial(training_run.data_collate_function, tokenizer=tokenizer),
    )

    model.eval()
    all_probs = []
    with torch.inference_mode():
        for batch in loader:
            model_kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if use_aspect_input:
                model_kwargs["aspect_input"] = batch["aspect_input"].to(device)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                logits = model(**model_kwargs).logits
            all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def load_model_and_config(model_repo: str, revision: str, model_name_override: Optional[str],
                           use_lora_override: Optional[bool], id2label: Dict[int, str],
                           label2id: Dict[str, int], device: torch.device):
    """Mirrors experiments/scripts/diagnostic_eval_stage2.py's loading pattern: read
    trial_metrics.json (written by push_checkpoint() alongside every Hub checkpoint) for the
    model_name/use_lora/mixed_precision this checkpoint was actually trained with, rather than
    guessing -- so this works unmodified against any past or future trial's checkpoint shape.
    """
    try:
        trial_metrics_path = hf_hub_download(repo_id=model_repo, filename="trial_metrics.json",
                                              revision=revision)
        trial_config = json.loads(Path(trial_metrics_path).read_text())["config"]
    except Exception as exc:
        if model_name_override is None:
            raise RuntimeError(
                f"Could not fetch trial_metrics.json from {model_repo}@{revision} ({exc}) -- pass "
                f"--model-name explicitly (and --use-lora if applicable) to load this checkpoint "
                f"without it."
            ) from exc
        print(f"[WARN] Could not fetch trial_metrics.json from {model_repo}@{revision} ({exc}); "
              f"proceeding with explicit --model-name/--use-lora only.")
        trial_config = {}

    model_name = model_name_override or trial_config.get("model_name")
    if model_name is None:
        raise RuntimeError("No model_name available from trial_metrics.json or --model-name.")
    use_lora = trial_config.get("use_lora", False) if use_lora_override is None else use_lora_override
    use_aspect_input = trial_config.get("use_aspect_input", False)
    mixed_precision = trial_config.get("mixed_precision", "no")
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision)

    tokenizer = training_run.create_esm_tokenizer(model_name)
    if use_lora:
        model = training_run.create_esm(
            model_type=model_name, use_lora=True, lora_resume_repo_id=model_repo,
            revision=revision, id2label=id2label, label2id=label2id,
            lora_base_dtype=autocast_dtype, use_aspect_input=use_aspect_input,
        ).to(device)
    else:
        model = training_run.create_esm(
            unfreeze_last_n=0, model_type=model_repo, revision=revision, id2label=id2label,
            label2id=label2id, use_aspect_input=use_aspect_input,
        ).to(device)
    print(f"[INFO] Loaded {model_repo}@{revision} ({model_name}, use_lora={use_lora}, "
          f"use_aspect_input={use_aspect_input}) for eval, device={device}")
    return model, tokenizer, autocast_dtype, use_aspect_input


# --------------------------------------------------------------------------------------------

def write_predictions_tsv(path: Path, protein_ids: List[str], probs: np.ndarray,
                           label_names: List[str], min_confidence: float = 0.01) -> None:
    with path.open("w") as f:
        for i, protein in enumerate(protein_ids):
            row = probs[i]
            for j in np.nonzero(row >= min_confidence)[0]:
                f.write(f"{protein}\t{label_names[j]}\t{row[j]:.3f}\n")
    print(f"[INFO] Wrote predictions ({path}) at confidence >= {min_confidence}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset")
    ap.add_argument("--hf-dataset-repo", default="kelmoir/CAFA6")
    ap.add_argument("--split", choices=["test", "validation"], default="test")
    ap.add_argument("--go-scope", choices=["all", "mf-span"], default="all")
    ap.add_argument("--mf-span-vocab-file", default=str(DEFAULT_MF_SPAN_VOCAB_FILE))
    ap.add_argument("--dep-graph-file", default=str(DEFAULT_DEP_GRAPH_FILE))
    ap.add_argument("--ia-file", default=str(DEFAULT_IA_FILE))
    ap.add_argument("--model-repo", default="kelmoir/cafa6-esm")
    ap.add_argument("--revision", default="main", help="Hub branch/revision to load, e.g. trial-050")
    ap.add_argument("--model-name", default=None,
                     help="Override the ESM base model name instead of reading it from the "
                          "checkpoint's trial_metrics.json (required if that file is missing)")
    ap.add_argument("--use-lora", dest="use_lora", action="store_true", default=None,
                     help="Force LoRA loading instead of reading it from trial_metrics.json")
    ap.add_argument("--no-use-lora", dest="use_lora", action="store_false")
    ap.add_argument("--skip-model", action="store_true",
                     help="Only compute the perfect-predictor baseline; no model/GPU needed")
    ap.add_argument("--skip-perfect-baseline", action="store_true",
                     help="Only score the model; skip the perfect-predictor sanity check")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--threshold-step", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--predictions-out", default=None,
                     help="Optional path to dump the model's propagated predictions as a "
                          "sample_submission.tsv-shaped TSV (protein, term, confidence)")
    ap.add_argument("--results-out", default=None,
                     help="Optional path to dump the full score(s) + threshold curve(s) as JSON")
    args = ap.parse_args()

    if args.skip_model and args.skip_perfect_baseline:
        print("[ERROR] --skip-model and --skip-perfect-baseline together leave nothing to compute.")
        sys.exit(1)

    dataset = training_run.load_training_data(data_path=Path(args.data_path), repo_id=args.hf_dataset_repo)
    label_names = [str(name) for name in dataset["train"].features["labels"].feature.names]
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in id2label.items()}

    split = dataset[args.split]
    if args.max_samples is not None and len(split) > args.max_samples:
        split = split.select(range(args.max_samples))
    print(f"[INFO] Evaluating against {args.split} split: {len(split)} proteins, "
          f"{len(label_names)} label terms, --go-scope={args.go_scope}")

    ia_weights = load_ia_weights(Path(args.ia_file), label_names)
    scope_mask = build_scope_mask(label_names, args.go_scope, Path(args.mf_span_vocab_file))
    true_path_graph = load_true_path_graph(Path(args.dep_graph_file))
    descendant_idx = build_descendant_index(true_path_graph, label_names)
    thresholds = np.arange(args.threshold_step, 1.0 + 1e-9, args.threshold_step)

    labels_matrix = extract_labels_matrix(split, len(label_names))

    results = {"config": vars(args), "n_labels_total": len(label_names)}

    if not args.skip_perfect_baseline:
        # Passing the (already ancestor-propagated, per generate_dataset.py) true annotations
        # straight through as "predictions" -- confidence 1.0 for every true term, 0.0
        # elsewhere. No further propagate_predictions() call is needed here: every ancestor of
        # a true term is, by construction of the coarse label space, already itself true (raw
        # confidence 1.0), so max-propagation is a no-op on a perfect predictor.
        perfect_score = cafa_weighted_fmax(labels_matrix.copy(), labels_matrix, ia_weights,
                                            scope_mask, thresholds)
        results["perfect_predictor"] = perfect_score
        print(f"\n[PERFECT PREDICTOR] Fmax={perfect_score['fmax']:.4f} "
              f"@threshold={perfect_score['threshold']} "
              f"precision={perfect_score['precision']:.4f} recall={perfect_score['recall']:.4f} "
              f"(n={perfect_score['n_proteins']} proteins, {perfect_score['n_scope_terms']} scope terms)")

    if not args.skip_model:
        model, tokenizer, autocast_dtype, use_aspect_input = load_model_and_config(
            args.model_repo, args.revision, args.model_name, args.use_lora, id2label, label2id,
            torch.device(args.device),
        )
        use_amp = autocast_dtype is not None and args.device.startswith("cuda")
        raw_probs = run_inference(model, tokenizer, split, label_names, torch.device(args.device),
                                   args.batch_size, autocast_dtype, use_amp, use_aspect_input)
        propagated_probs = propagate_predictions(raw_probs, descendant_idx)

        model_score = cafa_weighted_fmax(propagated_probs, labels_matrix, ia_weights, scope_mask,
                                          thresholds)
        results["model"] = model_score
        print(f"\n[MODEL] {args.model_repo}@{args.revision} Fmax={model_score['fmax']:.4f} "
              f"@threshold={model_score['threshold']} precision={model_score['precision']:.4f} "
              f"recall={model_score['recall']:.4f} "
              f"(n={model_score['n_proteins']} proteins, {model_score['n_scope_terms']} scope terms)")
        if not args.skip_perfect_baseline:
            print(f"[MODEL] {model_score['fmax'] / results['perfect_predictor']['fmax']:.1%} of "
                  f"the perfect-predictor ceiling")

        if args.predictions_out:
            write_predictions_tsv(Path(args.predictions_out), split["protein"], propagated_probs,
                                   label_names)

    if args.results_out:
        Path(args.results_out).write_text(json.dumps(results, indent=2, default=str))
        print(f"[INFO] Wrote full results -> {args.results_out}")


if __name__ == "__main__":
    main()
