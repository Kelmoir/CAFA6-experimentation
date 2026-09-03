"""Score a trained CAFA6 checkpoint against a held-out split using the same scoring formula the
actual Kaggle CAFA6 competition uses: information-accretion-weighted precision/recall, swept
across confidence thresholds, reduced to the maximum F-measure (Fmax) -- the standard CAFA
assessment methodology (see https://github.com/BioComputingUP/CAFA-evaluator), driven here by
this repo's own `data/cafa-6-protein-function-prediction/IA.tsv`.

Works against EITHER of this repo's two model families, auto-detected from --model-repo:
  - the main classifier (kelmoir/cafa6-esm): predicts a probability over a fixed 383-term coarse
    GO label space (see CLAUDE.md's "Label space" note).
  - the mf-span side-experiment (kelmoir/cafa6-mf-span): a FiLM-conditioned span-localization
    model with no fixed output vocabulary -- scored here by aggregating its per-residue
    predictions to a per-protein presence score (max over residues) against a candidate GO term
    list, since it has no native per-protein multi-label head to lean on directly.

IMPORTANT SCOPE CAVEAT: this replicates the official *scoring formula* exactly, but not the
official *evaluation universe*. The real competition scores a submission against the full GO
term space (tens of thousands of terms); this script can only ever score a model over the label
vocabulary it can actually predict over. --go-scope controls that vocabulary, and its meaning
depends on which model is selected:
  - --go-scope all:     classifier -> all 383 coarse terms (its full native vocabulary).
                         mf-span    -> its full ~4,579-term vocabulary (experiments/
                                       positional_mf_go_term_texts.json), scored against
                                       kelmoir/cafa6_mf_placed's own ground truth (its native
                                       eval set, NOT the classifier's dataset/split).
  - --go-scope mf-span:  restricts to just the 66 terms that appear in BOTH the classifier's
                         383-term vocabulary and the mf-span vocabulary, using the classifier's
                         dataset/split as ground truth for both models -- this is the
                         apples-to-apples setting for comparing the two models head-to-head on
                         identical ground truth (same proteins, same term set).
Treat every number here as "how good is my model at the task it was trained on, measured the
CAFA way" -- not as a stand-in for a real leaderboard score.

Ground truth always comes from this repo's own held-out dataset split(s), never the raw Kaggle
`Test/` directory -- that one ships with no labels at all (it's the blind competition test set),
so nothing could be scored against it locally.

Usage:
  # Score the promoted main-branch classifier checkpoint against the full 383-term label space
  python evaluate_cafa_score.py

  # Score a specific classifier trial branch, restricted to the mf-span-overlap terms
  python evaluate_cafa_score.py --revision trial-050 --go-scope mf-span

  # Score an mf-span checkpoint against its own full ~4,579-term vocabulary
  python evaluate_cafa_score.py --model-repo kelmoir/cafa6-mf-span --revision trial-20 --go-scope all

  # Head-to-head: same 66 terms, same proteins, both models
  python evaluate_cafa_score.py --revision main --go-scope mf-span
  python evaluate_cafa_score.py --model-repo kelmoir/cafa6-mf-span --revision trial-20 --go-scope mf-span

  # Just the perfect-predictor sanity check, no model/GPU needed
  python evaluate_cafa_score.py --skip-model
  python evaluate_cafa_score.py --skip-model --model-repo kelmoir/cafa6-mf-span --go-scope all

  # Dump the full threshold curve + a submission-shaped prediction TSV for inspection
  python evaluate_cafa_score.py --results-out experiments/logs/cafa_score_main.json \\
      --predictions-out /tmp/predictions.tsv
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

import generate_dataset
import training_run

# train_mf_span_model.py lives in experiments/scripts/, not importable by plain module name from
# the repo root without this -- mirrors experiments/scripts/predict_protein_spans.py's own import
# of it, which works unmodified only because THAT script happens to sit in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments" / "scripts"))
import train_mf_span_model as mf_span  # noqa: E402

DATA_DIR = Path("data/cafa-6-protein-function-prediction")
DEFAULT_IA_FILE = DATA_DIR / "IA.tsv"
DEFAULT_DEP_GRAPH_FILE = DATA_DIR / "dataset" / "go_dependency_graph.json"
DEFAULT_MF_SPAN_VOCAB_FILE = Path("experiments/positional_mf_go_term_texts.json")


# --------------------------------------------------------------------------------------------
# Model-type detection
# --------------------------------------------------------------------------------------------

def detect_model_type(model_repo: str, revision: str) -> str:
    """'classifier' (kelmoir/cafa6-esm-shaped: transformers config.json) or 'mf-span'
    (kelmoir/cafa6-mf-span-shaped: train_mf_span_model.py's span_model_config.json). A single
    cheap file-listing call (local dir listing, or one Hub API call -- no weights downloaded)."""
    path = Path(model_repo)
    if path.exists():
        files = {p.name for p in path.iterdir()}
    else:
        files = set(HfApi().list_repo_files(model_repo, revision=revision))
    if "span_model_config.json" in files:
        return "mf-span"
    if "config.json" in files:
        return "classifier"
    raise RuntimeError(
        f"Could not tell whether {model_repo}@{revision} is a classifier or mf-span checkpoint "
        f"(found neither config.json nor span_model_config.json among {sorted(files)}).")


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


def load_ia_weights(ia_file: Path, term_names: List[str]) -> np.ndarray:
    weights = {}
    for line in ia_file.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2:
            weights[parts[0]] = float(parts[1])
    missing = [name for name in term_names if name not in weights]
    if missing:
        print(f"[WARN] {len(missing)} term(s) missing from {ia_file}, defaulting IA=0.0 "
              f"(contributes nothing to weighted precision/recall): {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}")
    return np.array([weights.get(name, 0.0) for name in term_names], dtype=np.float64)


def build_descendant_index(true_path_graph: nx.MultiDiGraph, term_names: List[str]) -> Dict[int, List[int]]:
    """For each term column, the indices of every OTHER term column that is one of its
    ontological descendants (a more specific term). Used to max-propagate predicted confidence
    up the DAG before thresholding -- the same fix-up the official CAFA evaluator applies to a
    submission automatically, so a model that's confident about a specific child term but not
    yet confident about its parent isn't penalized for an inconsistency the true-path rule says
    shouldn't exist. Restricted to term_names itself: a term's descendants outside that list are
    neither predictable nor scoreable here, so they're irrelevant to this propagation step.
    """
    term_set = set(term_names)
    name_to_idx = {name: i for i, name in enumerate(term_names)}
    descendant_idx = {}
    for i, name in enumerate(term_names):
        descendants = nx.ancestors(true_path_graph, name) & term_set if name in true_path_graph else set()
        descendant_idx[i] = [name_to_idx[d] for d in descendants if d != name]
    return descendant_idx


def propagate_predictions(probs: np.ndarray, descendant_idx: Dict[int, List[int]]) -> np.ndarray:
    propagated = probs.copy()
    for j, desc_idxs in descendant_idx.items():
        if desc_idxs:
            propagated[:, j] = np.maximum(propagated[:, j], probs[:, desc_idxs].max(axis=1))
    return propagated


def build_scope_mask(label_names: List[str], scope: str, mf_span_vocab_file: Path) -> np.ndarray:
    """Only meaningful for the classifier's fixed 383-term vocabulary -- see build_eval_context."""
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
                        thresholds: np.ndarray) -> Dict:
    """Official CAFA scoring: for each threshold, weighted precision is averaged over proteins
    with >=1 predicted term (m(t) of them); weighted recall is averaged over every protein that
    has any nonzero-IA true annotation in scope (n of them, fixed across thresholds -- a
    protein with a true annotation is always in the recall denominator, whether or not the
    model predicted anything for it). Fmax is the harmonic mean's maximum over the sweep.

    probs/labels/ia_weights must already be restricted to exactly the term columns being scored
    (the caller's job -- see build_eval_context -- since what "in scope" means differs by model).
    """
    labels = labels.astype(bool)
    ia = ia_weights

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

    return {**best, "n_proteins": n, "n_scope_terms": int(len(ia)), "curve": curve}


# --------------------------------------------------------------------------------------------
# Ground truth extraction -- classifier dataset (already ancestor-propagated 383-column labels)
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
# Ground truth extraction -- kelmoir/cafa6_mf_placed (mf-span model's own native dataset)
# --------------------------------------------------------------------------------------------

def load_mf_span_dataset(data_path: Path, hub_repo: str):
    import datasets
    if data_path.exists():
        return datasets.DatasetDict.load_from_disk(str(data_path))
    return datasets.load_dataset(hub_repo)


def extract_mf_span_labels_matrix(hf_split, candidate_go_ids: List[str]) -> np.ndarray:
    """hf_split['placed_mf_terms'] is a comma-separated string of every distinct GO id placed on
    that protein (mf_localizations deduplicated by go_id, whole_protein-scope entries INCLUDED --
    unlike train_mf_span_model.py's build_pair_examples, which excludes them because they carry
    no positional signal for span training; here we only need protein-level presence/absence, and
    a whole_protein-scope annotation is still a real, asserted "this term applies" fact).
    """
    idx = {go_id: i for i, go_id in enumerate(candidate_go_ids)}
    mat = np.zeros((len(hf_split), len(candidate_go_ids)), dtype=np.float32)
    for i, raw in enumerate(hf_split["placed_mf_terms"]):
        for go_id in (t.strip() for t in raw.split(",")):
            if go_id in idx:
                mat[i, idx[go_id]] = 1.0
    return mat


# --------------------------------------------------------------------------------------------
# Classifier inference
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
# mf-span model inference
# --------------------------------------------------------------------------------------------

def run_mf_span_inference(model, tokenizer, label_emb_matrix: torch.Tensor, sequences: List[str],
                           device: torch.device, batch_size: int, candidate_chunk_size: int,
                           max_length: int) -> np.ndarray:
    """Per-protein, per-candidate-term presence score = max over residues of the span model's
    sigmoid output for that (protein, term) pair. Cheap despite an open-ended candidate list:
    SpanLocalizationModel.forward() runs the ESM backbone BEFORE FiLM-conditioning on the term
    embedding (see its docstring/forward), so the expensive part -- one ESM pass per protein --
    is shared across every candidate term; only the small FiLM + span_head layers actually run
    once per (protein, term) pair, chunked over candidates to bound memory.
    """
    n_candidates = label_emb_matrix.shape[0]
    scores = np.zeros((len(sequences), n_candidates), dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start:start + batch_size]
            enc = tokenizer(batch_seqs, padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            b, padded_len = input_ids.shape

            # Same <cls>/<eos>-exclusion convention as make_collate_fn's valid_mask, so the max
            # here can't be pulled up by a boundary-token logit the model was never trained to
            # place a real span on.
            valid_mask = attention_mask.clone().bool()
            valid_mask[:, 0] = False
            seq_lens = attention_mask.sum(dim=1)
            valid_mask[torch.arange(b), seq_lens - 1] = False

            hidden = model.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            for c0 in range(0, n_candidates, candidate_chunk_size):
                chunk_emb = label_emb_matrix[c0:c0 + candidate_chunk_size]  # [Cc, D]
                scale = model.film_scale(chunk_emb)  # [Cc, H]
                shift = model.film_shift(chunk_emb)
                h = hidden.unsqueeze(1) * (1.0 + scale).unsqueeze(0).unsqueeze(2) + shift.unsqueeze(0).unsqueeze(2)
                logits = model.span_head(h).squeeze(-1)  # [B, Cc, L]
                probs = torch.sigmoid(logits)
                probs = probs.masked_fill(~valid_mask.unsqueeze(1), 0.0)
                scores[start:start + b, c0:c0 + candidate_chunk_size] = probs.max(dim=-1).values.cpu().numpy()

            done = min(start + batch_size, len(sequences))
            if done % (batch_size * 10) == 0 or done == len(sequences):
                print(f"[INFO] mf-span inference: {done}/{len(sequences)} proteins x {n_candidates} "
                      f"candidate terms")
    return scores


# --------------------------------------------------------------------------------------------

def write_predictions_tsv(path: Path, protein_ids: List[str], probs: np.ndarray,
                           term_names: List[str], min_confidence: float = 0.01) -> None:
    with path.open("w") as f:
        for i, protein in enumerate(protein_ids):
            row = probs[i]
            for j in np.nonzero(row >= min_confidence)[0]:
                f.write(f"{protein}\t{term_names[j]}\t{row[j]:.3f}\n")
    print(f"[INFO] Wrote predictions ({path}) at confidence >= {min_confidence}")


# --------------------------------------------------------------------------------------------
# Evaluation context: resolves --go-scope + detected model type into a concrete (term list,
# protein list, sequences, ground-truth matrix, IA weights, descendant index) tuple. This is the
# one place scope's model-dependent meaning (see module docstring) gets decided.
# --------------------------------------------------------------------------------------------

def build_eval_context(args, model_type: str, true_path_graph: nx.MultiDiGraph
                        ) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, Dict[int, List[int]]]:
    if args.go_scope == "mf-span" or model_type == "classifier":
        # Both the classifier's own scoring and the mf-span-scope cross-model comparison need
        # the classifier's dataset (383-column, already ancestor-propagated ground truth).
        dataset = training_run.load_training_data(data_path=Path(args.data_path), repo_id=args.hf_dataset_repo)
        label_names = [str(name) for name in dataset["train"].features["labels"].feature.names]
        split = dataset[args.split]
        if args.max_samples is not None and len(split) > args.max_samples:
            split = split.select(range(args.max_samples))
        scope_mask = build_scope_mask(label_names, args.go_scope, Path(args.mf_span_vocab_file))
        term_names = [label_names[i] for i in np.nonzero(scope_mask)[0]]
        protein_ids = list(split["protein"])
        sequences = list(split["sequence"])
        labels_matrix = extract_labels_matrix(split, len(label_names))[:, scope_mask]
        print(f"[INFO] Ground truth: classifier dataset, {args.split} split, {len(split)} proteins, "
              f"{len(term_names)} scope terms")
    else:
        # model_type == "mf-span" and --go-scope all: score against the mf-span model's own full
        # vocabulary and its own native dataset -- NOT the classifier's dataset/split (different
        # protein set and a much finer-grained term universe than the classifier could ever
        # express), per this module's docstring.
        vocab = json.loads(Path(args.mf_span_vocab_file).read_text())
        term_names = sorted(vocab.keys())
        dataset_dict = load_mf_span_dataset(Path(args.mf_span_dataset_path), args.mf_span_hub_dataset_repo)
        split = dataset_dict[args.split]
        if args.max_samples is not None and len(split) > args.max_samples:
            split = split.select(range(args.max_samples))
        protein_ids = list(split["protein"])
        sequences = list(split["sequence"])
        labels_matrix = extract_mf_span_labels_matrix(split, term_names)
        print(f"[INFO] Ground truth: kelmoir/cafa6_mf_placed, {args.split} split, {len(split)} proteins, "
              f"{len(term_names)} scope terms (mf-span's full vocabulary)")

    ia_weights = load_ia_weights(Path(args.ia_file), term_names)
    descendant_idx = build_descendant_index(true_path_graph, term_names)
    return term_names, protein_ids, sequences, labels_matrix, ia_weights, descendant_idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset")
    ap.add_argument("--hf-dataset-repo", default="kelmoir/CAFA6")
    ap.add_argument("--mf-span-dataset-path", default=str(mf_span.LOCAL_DATASET_PATH),
                     help="kelmoir/cafa6_mf_placed local path (used for an mf-span model with "
                          "--go-scope all)")
    ap.add_argument("--mf-span-hub-dataset-repo", default=mf_span.HUB_DATASET_ID)
    ap.add_argument("--split", choices=["test", "validation"], default="test")
    ap.add_argument("--go-scope", choices=["all", "mf-span"], default="all",
                     help="See this script's module docstring -- meaning depends on which model "
                          "--model-repo resolves to.")
    ap.add_argument("--mf-span-vocab-file", default=str(DEFAULT_MF_SPAN_VOCAB_FILE))
    ap.add_argument("--dep-graph-file", default=str(DEFAULT_DEP_GRAPH_FILE))
    ap.add_argument("--ia-file", default=str(DEFAULT_IA_FILE))
    ap.add_argument("--model-repo", default="kelmoir/cafa6-esm",
                     help="kelmoir/cafa6-esm (classifier) or kelmoir/cafa6-mf-span (span model) "
                          "-- auto-detected from the checkpoint's own files")
    ap.add_argument("--revision", default="main", help="Hub branch/revision to load, e.g. trial-050")
    ap.add_argument("--model-name", default=None,
                     help="[classifier only] Override the ESM base model name instead of reading "
                          "it from trial_metrics.json (required if that file is missing)")
    ap.add_argument("--use-lora", dest="use_lora", action="store_true", default=None,
                     help="[classifier only] Force LoRA loading instead of reading it from trial_metrics.json")
    ap.add_argument("--no-use-lora", dest="use_lora", action="store_false")
    ap.add_argument("--skip-model", action="store_true",
                     help="Only compute the perfect-predictor baseline; no model/GPU needed "
                          "(model type is still detected from --model-repo, to pick the right "
                          "ground truth for --go-scope)")
    ap.add_argument("--skip-perfect-baseline", action="store_true",
                     help="Only score the model; skip the perfect-predictor sanity check")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mf-span-candidate-chunk-size", type=int, default=32,
                     help="[mf-span only] Candidate GO terms processed per inner chunk -- bounds "
                          "the [batch, chunk, seq_len, hidden] intermediate tensor's memory")
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

    model_type = detect_model_type(args.model_repo, args.revision)
    print(f"[INFO] Detected model type: {model_type} ({args.model_repo}@{args.revision}), "
          f"--go-scope={args.go_scope}")

    true_path_graph = load_true_path_graph(Path(args.dep_graph_file))
    term_names, protein_ids, sequences, labels_matrix, ia_weights, descendant_idx = build_eval_context(
        args, model_type, true_path_graph)
    thresholds = np.arange(args.threshold_step, 1.0 + 1e-9, args.threshold_step)

    results = {"config": vars(args), "model_type": model_type, "n_scope_terms": len(term_names)}

    if not args.skip_perfect_baseline:
        # Passing the (already ancestor-propagated) true annotations straight through as
        # "predictions" -- confidence 1.0 for every true term, 0.0 elsewhere. No further
        # propagate_predictions() call is needed here: every ancestor of a true term is, by
        # construction, already itself true (raw confidence 1.0), so max-propagation is a no-op.
        perfect_score = cafa_weighted_fmax(labels_matrix.copy(), labels_matrix, ia_weights, thresholds)
        results["perfect_predictor"] = perfect_score
        print(f"\n[PERFECT PREDICTOR] Fmax={perfect_score['fmax']:.4f} "
              f"@threshold={perfect_score['threshold']} "
              f"precision={perfect_score['precision']:.4f} recall={perfect_score['recall']:.4f} "
              f"(n={perfect_score['n_proteins']} proteins, {perfect_score['n_scope_terms']} scope terms)")

    if not args.skip_model:
        if model_type == "classifier":
            label2id = {name: i for i, name in enumerate(term_names)}
            id2label = {i: name for name, i in label2id.items()}
            model, tokenizer, autocast_dtype, use_aspect_input = load_model_and_config(
                args.model_repo, args.revision, args.model_name, args.use_lora, id2label, label2id,
                torch.device(args.device),
            )
            use_amp = autocast_dtype is not None and args.device.startswith("cuda")
            # Rebuild a split object with just protein/sequence/labels for run_inference's
            # with_transform() pipeline -- build_eval_context already applied --go-scope's column
            # slicing, so re-fetch the (unsliced) split here isn't needed; run_inference tokenizes
            # from `sequences` directly instead of re-deriving them from a dataset split.
            import datasets
            eval_split = datasets.Dataset.from_dict({"protein": protein_ids, "sequence": sequences})
            raw_probs = run_inference(model, tokenizer, eval_split, term_names, torch.device(args.device),
                                       args.batch_size, autocast_dtype, use_amp, use_aspect_input)
        else:
            model = mf_span.load_span_model(args.model_repo, device=args.device, revision=args.revision)
            # save_checkpoint() (train_mf_span_model.py) always calls tokenizer.save_pretrained()
            # alongside the weights, so the checkpoint's own tokenizer files can be loaded
            # directly -- no need to separately track/guess which esm_model_name it used.
            tokenizer = training_run.EsmTokenizer.from_pretrained(args.model_repo, revision=args.revision)
            label_emb_dict, _ = mf_span.build_label_embeddings(term_names, device=args.device)
            label_emb_matrix = torch.from_numpy(np.stack([label_emb_dict[g] for g in term_names])).float().to(args.device)
            print(f"[INFO] Loaded {args.model_repo}@{args.revision} (mf-span) for eval, "
                  f"device={args.device}, {len(term_names)} candidate terms")
            raw_probs = run_mf_span_inference(model, tokenizer, label_emb_matrix, sequences,
                                               torch.device(args.device), args.batch_size,
                                               args.mf_span_candidate_chunk_size, training_run.MAX_LENGTH)

        propagated_probs = propagate_predictions(raw_probs, descendant_idx)
        model_score = cafa_weighted_fmax(propagated_probs, labels_matrix, ia_weights, thresholds)
        results["model"] = model_score
        print(f"\n[MODEL] {args.model_repo}@{args.revision} Fmax={model_score['fmax']:.4f} "
              f"@threshold={model_score['threshold']} precision={model_score['precision']:.4f} "
              f"recall={model_score['recall']:.4f} "
              f"(n={model_score['n_proteins']} proteins, {model_score['n_scope_terms']} scope terms)")
        if not args.skip_perfect_baseline:
            print(f"[MODEL] {model_score['fmax'] / results['perfect_predictor']['fmax']:.1%} of "
                  f"the perfect-predictor ceiling")

        if args.predictions_out:
            write_predictions_tsv(Path(args.predictions_out), protein_ids, propagated_probs, term_names)

    if args.results_out:
        Path(args.results_out).write_text(json.dumps(results, indent=2, default=str))
        print(f"[INFO] Wrote full results -> {args.results_out}")


if __name__ == "__main__":
    main()
