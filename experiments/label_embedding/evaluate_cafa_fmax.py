"""Score item 16's autoregressive decoder against the project's real Tier-0 baseline
(trial-013, kelmoir/cafa6-esm@main) using the actual CAFA scoring rule -- IA-weighted Fmax swept
over confidence thresholds, predictions max-propagated up the GO DAG first -- instead of AUROC.
Every number in experiments/cafa6_next_steps_plan.md item 16 so far is AUROC (threshold-
independent ranking quality); this is the first time either architecture gets scored the way the
actual competition scores a submission, and the first time item 16 is compared against the real
full-scale Tier-0 checkpoint rather than item 8's throwaway 2,000-example dense-classifier control
run (the "0.6577" baseline cited throughout item 16's status updates).

Reuses rather than reimplements:
  - compute_ia_weighted_fmax (experiments/pk_eval/scripts/pk_eval_common.py) -- the IA-weighted
    Fmax formula, already cross-checked against the real `cafaeval` 1.3.0 package.
  - load_true_path_graph / build_descendant_index / propagate_predictions / load_model_and_config
    / run_inference (evaluate_cafa_score.py) -- DAG propagation and the baseline classifier's
    inference path.
  - training_run.load_training_data / load_ia_weights / create_esm / create_esm_tokenizer.

Both models are scored on the SAME dataset_full test-split subset and the SAME 27,191-term label
space with no partial-knowledge stripping (a plain held-out test run, not the PK/LK/NK split
pk_eval_common's "known"/"novel" split exists for -- known is always all-zero here).

Usage:
  # Default: 2000-protein test subset, full-scale trial-013 baseline vs. item 16's best checkpoint
  env/bin/python experiments/label_embedding/evaluate_cafa_fmax.py

  # Full test split (14,803 proteins) -- much slower, mainly the AR decoder's sequential generate()
  env/bin/python experiments/label_embedding/evaluate_cafa_fmax.py --max-samples 14803

  # Just one side (e.g. after already having the other's results.json)
  env/bin/python experiments/label_embedding/evaluate_cafa_fmax.py --skip-baseline
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "pk_eval" / "scripts"))

import training_run  # noqa: E402
import evaluate_cafa_score as ecs  # noqa: E402
from pk_eval_common import compute_ia_weighted_fmax  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "cafa-6-protein-function-prediction"
DEFAULT_DATA_PATH = DATA_DIR / "dataset_full"
DEFAULT_IA_FILE = DATA_DIR / "IA.tsv"
DEFAULT_AR_CHECKPOINT = (REPO_ROOT / "experiments" / "label_embedding" / "checkpoints" /
                          "item16-fulldata" / "epoch_20.pt")
DEFAULT_LABEL_EMBEDDINGS_PATH = REPO_ROOT / "experiments" / "label_embedding" / "term_embeddings.pt"


@torch.no_grad()
def generate_stop_truncated(model, input_ids, attention_mask, max_steps):
    """Same decode loop as AutoregressiveLabelEmbeddingEsmForSequenceClassification.generate()
    (training_run.py), but best_logit per example stops accumulating once that example's own
    stop head has fired, instead of unconditionally maxing over the full max_steps budget --
    isolates whether the untrained-past-stop decode tail (seq_mask in make_collate_fn never
    supervises steps past a protein's true label count, yet generate() always decodes all
    max_steps "regardless of the stop head") is what's destroying Fmax calibration. See
    experiments/cafa6_next_steps_plan.md item 16's 2026-11-02 status update.
    """
    state = model.encode_pooled(input_ids, attention_mask)
    batch_size = state.shape[0]
    num_labels = model.classifier.num_labels
    device = state.device
    prev = torch.full((batch_size,), num_labels, dtype=torch.long, device=device)  # START
    emitted_mask = torch.zeros(batch_size, num_labels, dtype=torch.bool, device=device)
    best_logit = torch.full((batch_size, num_labels), float("-inf"), device=device)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for _ in range(max_steps):
        state, label_logits, stop_logit = model.classifier.step(state, prev)
        active = (~stopped).unsqueeze(1)
        # The step where stop fires is itself included (matches step_losses' is_last_step
        # training target, which is placed AT the true last index, not one before it) --
        # `stopped` only gates steps AFTER this one.
        best_logit = torch.where(active, torch.maximum(best_logit, label_logits), best_logit)
        masked_logits = label_logits.masked_fill(emitted_mask, float("-inf"))
        prev = masked_logits.argmax(dim=-1)
        emitted_mask = emitted_mask.scatter(1, prev.unsqueeze(1), True)
        stopped = stopped | (torch.sigmoid(stop_logit) > 0.5)
    return best_logit


def run_ar_inference(model, tokenizer, sequences, device, batch_size, max_length, max_steps,
                      respect_stop_head: bool = False) -> np.ndarray:
    """Same generate()-based scoring as train_autoregressive_head.run_full_label_diagnostic:
    best_logit is already a full-[batch, num_labels] score at every decode step regardless of
    max_steps, so no truncation caveat applies here -- just sigmoid it into [0,1]. With
    --respect-stop-head, uses generate_stop_truncated instead, which drops each example's
    post-stop decode tail from the max.
    """
    model.eval()
    all_best_logits = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start:start + batch_size]
            enc = tokenizer(batch_seqs, max_length=max_length, truncation=True, padding=True,
                             return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            if respect_stop_head:
                best_logit = generate_stop_truncated(model, input_ids, attention_mask, max_steps)
            else:
                _, _, best_logit, _ = model.generate(input_ids, attention_mask, max_steps)
            all_best_logits.append(best_logit.cpu().numpy())
            done = min(start + batch_size, len(sequences))
            if done % (batch_size * 20) == 0 or done == len(sequences):
                print(f"[INFO] AR decoder inference: {done}/{len(sequences)} proteins")
    logits = np.concatenate(all_best_logits, axis=0)
    return 1.0 / (1.0 + np.exp(-logits))


def load_ar_model(checkpoint_path: Path, label_embeddings_path: Path, model_name: str,
                   ar_decoder_hidden_dim: int, label_names: list, device: torch.device):
    cached = torch.load(label_embeddings_path, map_location="cpu")
    if cached["terms"] != label_names:
        raise ValueError(
            f"{label_embeddings_path}'s cached term order does not match this dataset's label "
            "order -- same check run_autoregressive_training performs before training.")
    term_embeddings = cached["embeddings"]
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    model = training_run.create_esm(
        unfreeze_last_n=0, model_type=model_name, id2label=id2label, label2id=label2id,
        use_autoregressive_label_head=True, ar_decoder_hidden_dim=ar_decoder_hidden_dim,
        term_embeddings=term_embeddings,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] Loaded AR decoder from {checkpoint_path} (checkpoint epoch {ckpt.get('epoch')})")
    return model


def score(name: str, raw_probs: np.ndarray, labels_matrix: np.ndarray, known: np.ndarray,
          ia_weights: np.ndarray, descendant_idx: dict, thresholds: np.ndarray) -> dict:
    propagated = ecs.propagate_predictions(raw_probs, descendant_idx)
    fmax, best_t, pr, rc = compute_ia_weighted_fmax(propagated, known, labels_matrix, ia_weights, thresholds)
    print(f"\n[{name}] Fmax={fmax:.4f} @threshold={best_t:.2f} precision={pr:.4f} recall={rc:.4f}")
    return {"fmax": fmax, "threshold": best_t, "precision": pr, "recall": rc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--dataset-repo-id", default="kelmoir/cafa6-full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=2000,
                     help="Test-split subset size (default matches this project's standard "
                          "diagnostic-sample-count convention). The full test split has 14,803 "
                          "proteins.")
    ap.add_argument("--ia-file", default=str(DEFAULT_IA_FILE))
    ap.add_argument("--dep-graph-file", default=None,
                     help="Defaults to <data-path>/go_dependency_graph.json")
    ap.add_argument("--threshold-step", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--baseline-repo", default="kelmoir/cafa6-esm",
                     help="Tier-0 baseline (trial-013, full dataset_full scale) promoted to @main")
    ap.add_argument("--baseline-revision", default="main")
    ap.add_argument("--baseline-batch-size", type=int, default=16)

    ap.add_argument("--ar-checkpoint", default=str(DEFAULT_AR_CHECKPOINT))
    ap.add_argument("--ar-model-name", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--ar-decoder-hidden-dim", type=int, default=256)
    ap.add_argument("--ar-max-decode-steps", type=int, default=64)
    ap.add_argument("--ar-batch-size", type=int, default=8)
    ap.add_argument("--label-embeddings-path", default=str(DEFAULT_LABEL_EMBEDDINGS_PATH))
    ap.add_argument("--max-length", type=int, default=training_run.MAX_LENGTH)
    ap.add_argument("--respect-stop-head", action="store_true",
                     help="Drop each protein's post-stop decode tail from best_logit instead of "
                          "maxing over the full --ar-max-decode-steps budget regardless of the "
                          "stop head (generate()'s default/diagnostic convention). Isolates "
                          "whether the untrained decode tail is responsible for the AR decoder's "
                          "poor Fmax calibration vs. the real baseline.")

    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-ar", action="store_true")
    ap.add_argument("--results-out", default=None)
    args = ap.parse_args()

    if args.skip_baseline and args.skip_ar:
        print("[ERROR] --skip-baseline and --skip-ar together leave nothing to compute.")
        sys.exit(1)

    device = torch.device(args.device)
    data_path = Path(args.data_path)
    dep_graph_file = Path(args.dep_graph_file) if args.dep_graph_file else data_path / "go_dependency_graph.json"

    dataset = training_run.load_training_data(data_path=data_path, repo_id=args.dataset_repo_id)
    label_names = [str(n) for n in dataset["train"].features["labels"].feature.names]
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}

    split = dataset[args.split]
    if args.max_samples is not None and len(split) > args.max_samples:
        split = split.select(range(args.max_samples))
    protein_ids = list(split["protein"])
    sequences = list(split["sequence"])
    labels_matrix = ecs.extract_labels_matrix(split, len(label_names))
    n_with_labels = int((labels_matrix.sum(axis=1) > 0).sum())
    print(f"[INFO] Scoring {len(protein_ids)} proteins from '{args.split}' split, "
          f"{len(label_names)} labels ({n_with_labels} proteins have >=1 true label)")

    ia_weights = training_run.load_ia_weights(Path(args.ia_file), label_names)
    true_path_graph = ecs.load_true_path_graph(dep_graph_file)
    descendant_idx = ecs.build_descendant_index(true_path_graph, label_names)

    thresholds = np.arange(args.threshold_step, 1.0 + 1e-9, args.threshold_step)
    known = np.zeros_like(labels_matrix, dtype=bool)  # plain test run: nothing stripped as "known"
    results = {"config": vars(args), "n_proteins": len(protein_ids), "n_labels": len(label_names)}

    if not args.skip_baseline:
        model, tokenizer, autocast_dtype, use_aspect_input = ecs.load_model_and_config(
            args.baseline_repo, args.baseline_revision, None, None, id2label, label2id, device)
        use_amp = autocast_dtype is not None and args.device.startswith("cuda")
        import datasets
        eval_split = datasets.Dataset.from_dict({"protein": protein_ids, "sequence": sequences})
        baseline_probs = ecs.run_inference(model, tokenizer, eval_split, label_names, device,
                                            args.baseline_batch_size, autocast_dtype, use_amp,
                                            use_aspect_input)
        results["baseline"] = score(f"BASELINE {args.baseline_repo}@{args.baseline_revision}",
                                     baseline_probs, labels_matrix, known, ia_weights,
                                     descendant_idx, thresholds)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_ar:
        ar_model = load_ar_model(Path(args.ar_checkpoint), Path(args.label_embeddings_path),
                                  args.ar_model_name, args.ar_decoder_hidden_dim, label_names, device)
        ar_tokenizer = training_run.create_esm_tokenizer(args.ar_model_name)
        ar_probs = run_ar_inference(ar_model, ar_tokenizer, sequences, device, args.ar_batch_size,
                                     args.max_length, args.ar_max_decode_steps,
                                     respect_stop_head=args.respect_stop_head)
        ar_label = ("AUTOREGRESSIVE, stop-head-truncated (item 16)" if args.respect_stop_head
                    else "AUTOREGRESSIVE (item 16)")
        results["autoregressive"] = score(ar_label, ar_probs, labels_matrix,
                                           known, ia_weights, descendant_idx, thresholds)

    if "baseline" in results and "autoregressive" in results:
        delta = results["autoregressive"]["fmax"] - results["baseline"]["fmax"]
        print(f"\n[COMPARISON] autoregressive - baseline Fmax = {delta:+.4f} "
              f"({results['autoregressive']['fmax']:.4f} vs {results['baseline']['fmax']:.4f})")

    if args.results_out:
        Path(args.results_out).write_text(json.dumps(results, indent=2, default=str))
        print(f"[INFO] Wrote results -> {args.results_out}")


if __name__ == "__main__":
    main()
