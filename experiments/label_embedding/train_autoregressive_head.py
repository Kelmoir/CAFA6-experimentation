"""Item 16 (see experiments/cafa6_next_steps_plan.md): cheap-scale feasibility spike for an
autoregressive decoder over item 8's term-embedding vocabulary (AutoregressiveLabelEmbeddingHead
in training_run.py). Tests whether emitting GO terms one at a time -- instead of scoring the
full label set from a single static point -- fixes the single-point multi-label capacity cap
item 8's cheap-scale trials exposed (mean_auroc stuck at chance, ~0.50, even after fixing the
head's bias-capacity gap; see the plan doc's item 8 Step 4 for the diagnosis).

Standalone, NOT wired into training_run.py's main() training loop: that loop's loss/metrics/
batching all assume a fixed-shape dense multi-hot target, which this sequential-decode mechanism
doesn't produce. Reuses create_esm()/create_esm_tokenizer()/compute_label_depth()/
build_hierarchy_edge_index()/_diagnostic_metrics_from_arrays()/load_training_data() from
training_run.py directly rather than duplicating that logic. Only promote this into
training_run.py's own CLI if this spike's result justifies it -- same validate-cheap-first
ordering item 8 itself followed.

Run (CPU smoke test, catches wiring bugs cheaply before any real GPU spend):
  env/bin/python experiments/label_embedding/train_autoregressive_head.py --device cpu \
    --max-train-samples 20 --max-val-samples 10 --epochs 1 --run-name ar-smoke-test

Run (cheap-scale GPU comparison, same config as item 8 step 4's cheap trials):
  env/bin/python experiments/label_embedding/train_autoregressive_head.py --device cuda \
    --model-name facebook/esm2_t12_35M_UR50D --unfreeze-last-n 4 --esm-dropout 0.1 \
    --attention-dropout 0.1 --weight-decay 0.01 --batch-size 2 --grad-accum-steps 8 \
    --max-train-samples 2000 --max-val-samples 500 --epochs 5 --run-diagnostic \
    --diagnostic-max-val-samples 1000 --run-name item16-cheap-autoregressive
"""
import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Running this script directly puts experiments/label_embedding/ on sys.path, not the repo
# root -- same fix as experiments/pk_eval/scripts/pk_eval_common.py etc. use for `import
# training_run`, just one directory shallower (no scripts/ subdir here).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training_run import (
    _diagnostic_metrics_from_arrays,
    build_hierarchy_edge_index,
    compute_label_depth,
    create_esm,
    create_esm_tokenizer,
    load_training_data,
)


def build_label_order(raw_label_ids: List[int], depth: torch.Tensor, max_steps: int,
                       include_roots: bool = False) -> List[int]:
    """Sort a protein's true label ids ascending by GO-DAG depth (ties broken by id, for
    determinism), then truncate to max_steps. Ascending depth guarantees every in-vocabulary
    ancestor of a kept label is emitted before it (see compute_label_depth); truncating drops the
    deepest/most specific long-tail labels first, keeping the general ones a protein almost
    certainly has.

    Depth-0 labels (the 3 GO aspect roots -- molecular_function/biological_process/
    cellular_component) are dropped unless include_roots=True: a protein is annotated under an
    aspect almost by definition, so these carry little real discriminative signal and would
    otherwise be free, near-trivially-correct predictions inflating both the CE loss and top-1
    accuracy without reflecting much actual functional-prediction ability. include_roots exists
    only to let a --ar-include-roots-in-training A/B run train with them present while still
    being scored by a root-excluding diagnostic (see main()) for a fair comparison.
    """
    pool = raw_label_ids if include_roots else [idx for idx in raw_label_ids if int(depth[idx]) > 0]
    ordered = sorted(pool, key=lambda idx: (int(depth[idx]), idx))
    return ordered[:max_steps]


def make_collate_fn(tokenizer, depth: torch.Tensor, max_steps: int, max_length: int,
                     include_roots: bool = False):
    def collate(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        sequences = [ex["sequence"] for ex in examples]
        tokenized = tokenizer(sequences, max_length=max_length, truncation=True, padding=True,
                               return_tensors="pt")
        batch_size = len(examples)
        label_seq = torch.zeros(batch_size, max_steps, dtype=torch.long)
        seq_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool)
        for i, ex in enumerate(examples):
            order = build_label_order(ex["labels"], depth, max_steps, include_roots=include_roots)
            label_seq[i, :len(order)] = torch.tensor(order, dtype=torch.long)
            seq_mask[i, :len(order)] = True
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "label_seq": label_seq,
            "seq_mask": seq_mask,
        }
    return collate


def scheduled_sampling_prob(step: int, total_steps: int, warmup_frac: float, final_prob: float) -> float:
    """Bengio et al. 2015: pure teacher forcing (prob=0) for the first warmup_frac of training,
    stabilizing early learning when the decoder's own predictions are close to random, then a
    linear ramp to final_prob -- without this ramp, training only ever exposes the decoder to
    ground-truth history while inference (generate()) only ever has its own, possibly wrong,
    history (exposure bias).
    """
    warmup_steps = warmup_frac * total_steps
    if total_steps <= warmup_steps or step < warmup_steps:
        return 0.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min(final_prob, final_prob * progress)


def step_losses(label_logits: torch.Tensor, stop_logits: torch.Tensor, label_seq: torch.Tensor,
                 seq_mask: torch.Tensor, max_steps: int):
    mask_f = seq_mask.float()
    ce = torch.nn.functional.cross_entropy(
        label_logits.reshape(-1, label_logits.shape[-1]), label_seq.reshape(-1), reduction="none"
    ).reshape(seq_mask.shape)
    ce = (ce * mask_f).sum() / mask_f.sum().clamp(min=1)

    seq_lengths = seq_mask.sum(dim=1).clamp(min=1, max=max_steps)
    is_last_step = torch.nn.functional.one_hot(seq_lengths - 1, max_steps).float()
    stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(stop_logits, is_last_step, reduction="none")
    stop_loss = (stop_loss * mask_f).sum() / mask_f.sum().clamp(min=1)
    return ce, stop_loss


def run_eval_pass(model, loader, device, max_steps):
    """Teacher-forced eval (scheduled_sampling_prob=0, i.e. pure ground-truth history) --
    cheap diagnostic in addition to loss: top-1 token accuracy (is argmax(label_logits) the
    true next label, at each real step) distinguishes "the model learned to discriminate
    between labels" from "the model just rode the prior-bias-init curve down" (both can show
    a similarly decreasing loss; only the former should show accuracy rising meaningfully
    above the single-label base rate of predicting whatever's globally most frequent)."""
    model.eval()
    ce_sum, stop_sum, correct_sum, mask_sum, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_seq = batch["label_seq"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            label_logits, stop_logits = model.forward_teacher_forced(
                input_ids, attention_mask, label_seq, seq_mask, scheduled_sampling_prob=0.0)
            ce, stop_loss = step_losses(label_logits, stop_logits, label_seq, seq_mask, max_steps)
            ce_sum += ce.item()
            stop_sum += stop_loss.item()
            mask_f = seq_mask.float()
            correct = (label_logits.argmax(dim=-1) == label_seq).float() * mask_f
            correct_sum += correct.sum().item()
            mask_sum += mask_f.sum().item()
            n_batches += 1
    top1_acc = correct_sum / max(1, mask_sum)
    return ce_sum / max(1, n_batches), stop_sum / max(1, n_batches), top1_acc


def run_dense_diagnostic(model, loader, device, max_steps, label_names, label_counts, rare_term_threshold):
    """Converts the sequential decode into the same dense [n_examples, n_labels] probability
    matrix run_diagnostic() expects, via generate()'s forced-full-length best_logit_per_label.
    Ground truth here is also truncated to each protein's first max_steps depth-ordered true
    labels (the same truncation training uses) -- NOT the full true-label set run_diagnostic()
    normally scores against for the classifier/label-embedding-head baselines. This keeps the
    comparison to "what this architecture can represent at all" rather than penalizing it for a
    cap every cheap-scale trial in this line has used; still, the resulting AUROC number is NOT
    directly apples-to-apples with the (full-label-set) baselines without noting that caveat.

    Also reports stop-head calibration (mean absolute error between the predicted stop step and
    the true, possibly-truncated sequence length) -- generate() always runs the full max_steps
    budget for AUROC purposes, so this is the only place that checks whether the dedicated stop
    head would have picked a sane cutoff in realistic (not forced-length) usage.
    """
    model.eval()
    all_best_logits, all_labels = [], []
    stop_abs_err_sum, stop_argmax_abs_err_sum, stop_n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, stop_at, best_logit, stop_logits = model.generate(input_ids, attention_mask, max_steps)
            all_best_logits.append(best_logit.cpu().numpy())
            seq_mask = batch["seq_mask"]
            label_seq = batch["label_seq"]
            true_len = seq_mask.sum(dim=1).to(device)
            # 0-based index of the true last real step (is_last_step's training target index).
            true_last_idx = (true_len - 1).clamp(min=0, max=max_steps - 1)
            stop_abs_err_sum += (stop_at - true_last_idx).abs().sum().item()
            # Ranking-only variant: the single highest-scoring step, immune to the fixed 0.5
            # threshold -- separates "the stop head hasn't learned timing at all" from "it has,
            # but 0.5 is miscalibrated given how rare a true stop step is per sequence."
            argmax_stop = stop_logits.argmax(dim=1)
            stop_argmax_abs_err_sum += (argmax_stop - true_last_idx).abs().sum().item()
            stop_n += true_len.shape[0]
            dense = np.zeros((input_ids.shape[0], len(label_names)), dtype=np.float32)
            for i in range(input_ids.shape[0]):
                ids = label_seq[i][seq_mask[i]].tolist()
                dense[i, ids] = 1.0
            all_labels.append(dense)
    probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_best_logits, axis=0)))
    labels_arr = np.concatenate(all_labels, axis=0)
    label_is_rare = label_counts.numpy() < rare_term_threshold
    result, per_label_auroc = _diagnostic_metrics_from_arrays(probs, labels_arr, label_is_rare=label_is_rare)
    print("\n" + "=" * 70)
    print(f"[CAVEAT] ground truth truncated to each protein's first {max_steps} depth-ordered "
          f"true labels -- NOT directly comparable to a full-27191-label-set baseline AUROC.")
    print(f"DIAGNOSTIC (dense adapter): {result['n_labels_evaluable']}/{result['n_labels_total']} "
          f"labels evaluable, {result['n_val_examples']} val examples")
    print("=" * 70)
    print(f"Stop-head calibration: mean |predicted_stop_step - true_last_step_idx| = "
          f"{stop_abs_err_sum / max(1, stop_n):.2f} steps (0.5-threshold) vs. "
          f"{stop_argmax_abs_err_sum / max(1, stop_n):.2f} steps (argmax-ranking, threshold-free) "
          f"(max_steps={max_steps})")
    if per_label_auroc.size:
        print(f"Per-label AUROC: mean={result['mean_auroc']:.4f} median={result['median_auroc']:.4f} "
              f"min={per_label_auroc.min():.4f} max={per_label_auroc.max():.4f}")
    print(f"Rarity-stratified AUROC: rare mean={result.get('rare_mean_auroc')} "
          f"common mean={result.get('common_mean_auroc')}")
    import json as _json
    print(f"[DIAGNOSTIC] {_json.dumps(result, default=str)}")
    return result


def make_full_label_eval_collate_fn(tokenizer, max_length: int):
    """Unlike make_collate_fn, ground truth here is each protein's FULL, untruncated true-label
    set (including roots) -- the same convention item 8's run_diagnostic() scores its classifier
    baseline against. generate()'s best_logit_per_label is already a full-vocab [batch,
    num_labels] score regardless of max_steps (every decode step scores all 27191 labels, not
    just the ones selected into the sequence), so only the ground truth needs to change here to
    get an apples-to-apples number against that baseline -- the model/inference path is
    identical to run_dense_diagnostic.
    """
    def collate(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        sequences = [ex["sequence"] for ex in examples]
        tokenized = tokenizer(sequences, max_length=max_length, truncation=True, padding=True,
                               return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": [ex["labels"] for ex in examples],
        }
    return collate


def run_full_label_diagnostic(model, loader, device, max_steps, label_names, label_counts, rare_term_threshold):
    """Same generate()-based scoring as run_dense_diagnostic, but against the FULL untruncated
    true-label set -- the fair, directly-comparable number against item 8's classifier baseline
    (0.6577 mean AUROC) and the label-embedding-head baseline, with no truncation caveat.
    """
    model.eval()
    all_best_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, _, best_logit, _ = model.generate(input_ids, attention_mask, max_steps)
            all_best_logits.append(best_logit.cpu().numpy())
            dense = np.zeros((input_ids.shape[0], len(label_names)), dtype=np.float32)
            for i, labs in enumerate(batch["labels"]):
                dense[i, labs] = 1.0
            all_labels.append(dense)
    probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_best_logits, axis=0)))
    labels_arr = np.concatenate(all_labels, axis=0)
    label_is_rare = label_counts.numpy() < rare_term_threshold
    result, per_label_auroc = _diagnostic_metrics_from_arrays(probs, labels_arr, label_is_rare=label_is_rare)
    print("\n" + "=" * 70)
    print(f"FULL-LABEL-SET DIAGNOSTIC (no truncation, directly comparable to item 8's 0.6577 "
          f"baseline): {result['n_labels_evaluable']}/{result['n_labels_total']} labels "
          f"evaluable, {result['n_val_examples']} val examples")
    print("=" * 70)
    if per_label_auroc.size:
        print(f"Per-label AUROC: mean={result['mean_auroc']:.4f} median={result['median_auroc']:.4f} "
              f"min={per_label_auroc.min():.4f} max={per_label_auroc.max():.4f}")
    print(f"Rarity-stratified AUROC: rare mean={result.get('rare_mean_auroc')} "
          f"common mean={result.get('common_mean_auroc')}")
    import json as _json
    print(f"[FULL_LABEL_DIAGNOSTIC] {_json.dumps(result, default=str)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Item 16: autoregressive label-embedding decoder spike")
    parser.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset_full")
    parser.add_argument("--dataset-repo-id", default="kelmoir/cafa6-full")
    parser.add_argument("--label-embeddings-path", default="experiments/label_embedding/term_embeddings.pt")
    parser.add_argument("--go-dependency-graph-path", default=None)
    parser.add_argument("--model-name", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--unfreeze-last-n", type=int, default=4)
    parser.add_argument("--esm-dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--output-dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4192)
    parser.add_argument("--ar-max-decode-steps", type=int, default=64,
                         help="Cap on decode steps/protein. Proteins average 43.7 true labels "
                              "(median 31) but a long tail reaches 451; sequences are truncated "
                              "to the shallowest --ar-max-decode-steps labels by GO-DAG depth "
                              "order (compute_label_depth), dropping the most specific long-tail "
                              "terms first, not the general ones.")
    parser.add_argument("--ar-include-roots-in-training", action=argparse.BooleanOptionalAction,
                         default=True,
                         help="Include the 3 depth-0 GO aspect roots as decode targets during "
                              "training. Evaluation (per-epoch top1_acc/loss and --run-diagnostic) "
                              "always excludes roots regardless of this flag, so the two settings "
                              "are a fair A/B on how well the model predicts the same non-trivial "
                              "label set either way. Default True: a controlled A/B at cheap scale "
                              "(see plan doc item 16) found training WITH root supervision scores "
                              "higher on the same root-excluded eval set (0.6430 vs 0.6158 mean "
                              "AUROC), not just free credit from trivially-true labels.")
    parser.add_argument("--ar-decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--ar-scheduled-sampling-final-prob", type=float, default=0.25)
    parser.add_argument("--ar-scheduled-sampling-warmup-frac", type=float, default=0.3)
    parser.add_argument("--run-diagnostic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostic-max-val-samples", type=int, default=1000)
    parser.add_argument("--rare-term-threshold", type=int, default=100)
    parser.add_argument("--run-name", default="ar-spike")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data_path = Path(args.data_path)
    dataset = load_training_data(data_path=data_path, repo_id=args.dataset_repo_id)
    dataset_train = dataset["train"]
    label_names = [str(name) for name in dataset_train.features["labels"].feature.names]
    label2id = {name: idx for idx, name in enumerate(label_names)}

    graph_path = (Path(args.go_dependency_graph_path) if args.go_dependency_graph_path
                  else data_path / "go_dependency_graph.json")
    depth = compute_label_depth(label2id, graph_path)
    print(f"[INFO] label depth within trainable vocabulary: min={depth.min().item()} "
          f"max={depth.max().item()} mean={depth.float().mean():.2f} "
          f"({int((depth == 0).sum())} depth-0 roots)")

    if args.ar_include_roots_in_training:
        train_label_counts_per_example = [len(labs) for labs in dataset_train["labels"]]
        root_note = "including the 3 depth-0 aspect roots, per --ar-include-roots-in-training"
    else:
        train_label_counts_per_example = [
            sum(1 for idx in labs if int(depth[idx]) > 0) for labs in dataset_train["labels"]
        ]
        root_note = "the 3 depth-0 aspect roots are excluded from this count -- they're never targets"
    n_truncated = sum(1 for n in train_label_counts_per_example if n > args.ar_max_decode_steps)
    print(f"[INFO] --ar-max-decode-steps={args.ar_max_decode_steps}: {n_truncated}/"
          f"{len(train_label_counts_per_example)} training proteins "
          f"({100 * n_truncated / len(train_label_counts_per_example):.2f}%) have more true "
          f"labels than the cap and get truncated (deepest/most-specific labels dropped first; "
          f"{root_note}).")

    cached = torch.load(args.label_embeddings_path)
    if cached["terms"] != label_names:
        raise ValueError(
            f"{args.label_embeddings_path}'s cached term order does not match this dataset's "
            "label2id ordering -- regenerate experiments/label_embedding/build_term_embeddings.py "
            "against the dataset actually in use.")
    term_embeddings = cached["embeddings"]

    tokenizer = create_esm_tokenizer(args.model_name)
    collate_fn = make_collate_fn(tokenizer, depth, args.ar_max_decode_steps, args.max_length,
                                  include_roots=args.ar_include_roots_in_training)
    # Evaluation always scores against the same root-excluded ground truth regardless of
    # --ar-include-roots-in-training, so a with-roots-training run is a fair A/B against a
    # without-roots-training run on the same non-trivial label set.
    eval_collate_fn = make_collate_fn(tokenizer, depth, args.ar_max_decode_steps, args.max_length,
                                       include_roots=False)

    val_ds = dataset["validation"]
    if args.max_val_samples is not None:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))

    model = create_esm(
        unfreeze_last_n=args.unfreeze_last_n,
        model_type=args.model_name,
        output_dropout=args.output_dropout,
        esm_dropout=args.esm_dropout,
        attention_dropout=args.attention_dropout,
        id2label={i: n for i, n in enumerate(label_names)},
        label2id=label2id,
        use_autoregressive_label_head=True,
        ar_decoder_hidden_dim=args.ar_decoder_hidden_dim,
        term_embeddings=term_embeddings,
    ).to(device)

    label_counts = torch.zeros(len(label_names))
    for labs in dataset_train["labels"]:
        for idx in labs:
            label_counts[int(idx)] += 1
    # Same prior-bias-init rationale as item 8 (LabelEmbeddingClassificationHead's docstring /
    # training_run.py main()'s get_classifier_bias_param path): BCE under this task's severe
    # imbalance otherwise wastes early gradient budget just learning the bias.
    label_priors = (label_counts / max(1, len(dataset_train))).clamp(1e-4, 1 - 1e-4)
    prior_bias = torch.log(label_priors / (1 - label_priors)).to(device)
    with torch.no_grad():
        model.classifier.bias.copy_(prior_bias)
    print(f"[INFO] Initialized classifier bias to label priors for {len(label_names)} labels "
          f"(mean={label_priors.mean().item():.4f}, min={label_priors.min().item():.4f}, "
          f"max={label_priors.max().item():.4f})")

    # Same rationale, applied to the stop head: is_last_step's target is 1 at exactly one step
    # per protein and 0 at every other real step, so with a zero-initialized bias (sigmoid=0.5)
    # the easiest way to cut loss is to predict "not last" almost everywhere -- exactly the
    # single-point-head bias-capacity failure mode item 8 hit, just for this binary target
    # instead of 27,191 multi-label ones. Prior rate = 1 positive per protein / mean capped
    # sequence length.
    capped_counts = [min(n, args.ar_max_decode_steps) for n in train_label_counts_per_example]
    total_real_steps = sum(capped_counts)
    stop_prior_rate = min(max(len(capped_counts) / max(1, total_real_steps), 1e-4), 1 - 1e-4)
    stop_prior_bias = float(np.log(stop_prior_rate / (1 - stop_prior_rate)))
    with torch.no_grad():
        model.classifier.stop_head.bias.fill_(stop_prior_bias)
    print(f"[INFO] Initialized stop_head bias to prior stop rate {stop_prior_rate:.4f} "
          f"(bias={stop_prior_bias:.4f}, mean capped sequence length "
          f"{total_real_steps / max(1, len(capped_counts)):.1f} steps)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    n_train = min(args.max_train_samples, len(dataset_train)) if args.max_train_samples else len(dataset_train)
    steps_per_epoch = max(1, n_train // args.batch_size)
    optimizer_steps_per_epoch = max(1, steps_per_epoch // args.grad_accum_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    global_step = 0
    ss_prob = 0.0

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, collate_fn=eval_collate_fn)

    for epoch in range(args.epochs):
        model.train()
        train_subset = dataset_train
        if args.max_train_samples is not None:
            train_subset = train_subset.shuffle(seed=args.seed + epoch).select(range(n_train))
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        ce_sum, stop_sum, n_batches = 0.0, 0.0, 0
        train_correct_sum, train_mask_sum = 0.0, 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_seq = batch["label_seq"].to(device)
            seq_mask = batch["seq_mask"].to(device)

            ss_prob = scheduled_sampling_prob(
                global_step, total_optimizer_steps,
                args.ar_scheduled_sampling_warmup_frac, args.ar_scheduled_sampling_final_prob)
            label_logits, stop_logits = model.forward_teacher_forced(
                input_ids, attention_mask, label_seq, seq_mask, scheduled_sampling_prob=ss_prob)
            ce, stop_loss = step_losses(label_logits, stop_logits, label_seq, seq_mask, args.ar_max_decode_steps)
            loss = (ce + stop_loss) / args.grad_accum_steps
            loss.backward()

            ce_sum += ce.item()
            stop_sum += stop_loss.item()
            n_batches += 1
            with torch.no_grad():
                mask_f = seq_mask.float()
                train_correct_sum += ((label_logits.argmax(dim=-1) == label_seq).float() * mask_f).sum().item()
                train_mask_sum += mask_f.sum().item()

            is_accum_boundary = (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader)
            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        train_ce, train_stop = ce_sum / max(1, n_batches), stop_sum / max(1, n_batches)
        train_top1_acc = train_correct_sum / max(1, train_mask_sum)
        val_ce, val_stop, val_top1_acc = run_eval_pass(model, val_loader, device, args.ar_max_decode_steps)
        print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_ce + train_stop:.4f} "
              f"(ce={train_ce:.4f} stop={train_stop:.4f}) | val_loss={val_ce + val_stop:.4f} "
              f"(ce={val_ce:.4f} stop={val_stop:.4f}) | train_top1_acc={train_top1_acc:.4f} "
              f"val_top1_acc={val_top1_acc:.4f} | scheduled_sampling_prob={ss_prob:.3f}")

    if args.run_diagnostic:
        diagnostic_ds = dataset["validation"].select(
            range(min(args.diagnostic_max_val_samples, len(dataset["validation"]))))
        diag_loader = torch.utils.data.DataLoader(diagnostic_ds, batch_size=args.batch_size, collate_fn=eval_collate_fn)
        run_dense_diagnostic(model, diag_loader, device, args.ar_max_decode_steps,
                              label_names, label_counts, args.rare_term_threshold)

        full_label_collate_fn = make_full_label_eval_collate_fn(tokenizer, args.max_length)
        full_label_loader = torch.utils.data.DataLoader(
            diagnostic_ds, batch_size=args.batch_size, collate_fn=full_label_collate_fn)
        run_full_label_diagnostic(model, full_label_loader, device, args.ar_max_decode_steps,
                                   label_names, label_counts, args.rare_term_threshold)

    print(f"[SUMMARY] run_name={args.run_name} done")


if __name__ == "__main__":
    main()
