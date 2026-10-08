# Known-Annotation (PK) Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `training_run.py` a training-time "known annotations" input channel —
Tier 1 item 2 part (B) — so the model can learn to predict a protein's remaining
unannotated GO terms conditioned on a synthetically-revealed subset of its own known
terms, instead of always predicting the full closure from sequence alone.

**Architecture:** Extend `training_run.py` in place: an on-the-fly masking transform in
`preprocess_batch` (no new dataset), a `KnownAwareClassificationHead` /
`KnownConditionedEsmForSequenceClassification` pair (direct sibling of the existing
`AspectAwareClassificationHead` pair) using `nn.EmbeddingBag(mode="sum")` to pool a
variable-size known-term set into a fixed vector injected at the classifier head, and
reuse of the existing `loss_mask`/`compute_loss` machinery to exclude revealed terms
from the loss. Final task wires `experiments/pk_eval/scripts/score_temporal_pk_eval.py`
to score a `--use-known-input` checkpoint against part (A)'s real temporal known-terms.

**Tech Stack:** PyTorch (`torch.nn.EmbeddingBag`), the project's existing
`data/cafa-6-protein-function-prediction/dataset_full` (27,191-term label space,
unchanged), the already-built `experiments/pk_eval/` harness (unchanged except for the
final wiring task).

**Spec:** `docs/superpowers/specs/2026-10-08-known-annotation-conditioning-design.md`

**Note on line numbers:** Tasks 1-4 all edit `training_run.py` sequentially, and each
edit shifts every line number below it (an inserted `import random`, a new class, a new
`argparse` block, ...). Line numbers cited in Task 2 onward are against the file state
*after* the preceding tasks' edits land, but drift slightly with exact line counts as
written — locate every insertion point by the quoted surrounding code or
function/variable name first, and treat the line number as a rough pointer, not a
guarantee. `score_temporal_pk_eval.py` (Task 6) is untouched by Tasks 1-5, so its line
numbers don't have this issue.

## Global Constraints

- No new dataset artifact, no new HF Hub dataset push — the masking transform operates
  entirely on `dataset_full`'s existing `labels` column, at preprocessing time.
- Label vocabulary: `config.num_labels` / `len(unique_items)` (27,191 terms) is both the
  prediction target space and the known-term embedding vocabulary — same space, reused
  verbatim, never a second vocabulary.
- Seed `42` everywhere a random choice is made (repo-wide convention) — `random.seed`
  joins the existing `torch.manual_seed`/`torch.cuda.manual_seed_all` calls.
- `--use-known-input` default `False` — every existing invocation of `training_run.py`
  is byte-for-byte unaffected unless the flag is passed.
- New CLI defaults: `--known-mix-zero-frac 0.3`, `--known-reveal-max 1.0`,
  `--known-embed-dim 64` — proposed starting points per the spec, not a tuned
  conclusion; `--known-reveal-max` must be in `(0, 1]`.
- `--use-known-input` is mutually exclusive with `--mask-sparse-negatives-max-total-terms`,
  `--loss-mask-from-dataset` (both already write `loss_mask`; no defined combined
  semantics), `--use-aspect-input`, and `--use-lora` (no classifier head/architecture
  supports those combinations today) — each raises a clear error, never a silent
  precedence choice.
- Known/revealed terms are excluded from the loss entirely for that example (`loss_mask`
  = 0 at those positions) — never kept as a target.
- `KnownAwareClassificationHead`/`KnownConditionedEsmForSequenceClassification` are
  direct siblings of `AspectAwareClassificationHead`/`AspectConditionedEsmForSequenceClassification`
  (`training_run.py:134-197`) — same injection point (concatenated before the
  classifier head's dense layer), same manual Linear re-init after `post_init()`, same
  "ESM backbone forward pass untouched" property.

## Review Focus

- A protein whose positive-term count is so small that
  `floor(reveal_frac * |positives|) == 0` (e.g. exactly 1 positive term) must degrade
  gracefully to a zero-known draw, not crash or call `random.sample` with a negative or
  out-of-range count.
- A batch where every example has zero known terms (realistic given
  `--known-mix-zero-frac`, and the normal shape of an all-NK eval batch) must produce an
  all-zero-vector known-embedding output via `EmbeddingBag`, not crash on an empty
  flattened `known_indices` tensor.
- `--use-known-input` combined with `--use-lora`, `--use-aspect-input`,
  `--mask-sparse-negatives-max-total-terms`, or `--loss-mask-from-dataset` must raise a
  clear error at argument-parsing/model-creation time, not silently pick an undefined
  combined behavior or crash deep into a training run.
- Reloading a `--use-known-input` checkpoint with a non-default `--known-embed-dim`
  (best-checkpoint reload block, `--run-diagnostic`, and
  `score_temporal_pk_eval.py`'s own model load) must use the exact same embed dim the
  checkpoint was trained with — a mismatch causes a `state_dict` shape error on load,
  not a silently-wrong model.
- A real eval row with genuinely empty `known_labels` (a true NK row from part (A)'s
  temporal harness, not a synthetic draw) must flow through the dataset-sourced known-
  terms path as an empty bag, identical in effect to training's zero-known draw — not
  crash or get treated as "missing data" distinct from an intentional empty set.

---

## Task 1: Synthetic known/novel masking in `preprocess_batch`

**Files:**
- Modify: `training_run.py:1-13` (imports), `training_run.py:353-468` (`preprocess_batch`),
  `training_run.py:844-933` (`main()`'s `argparse` setup and post-parse validation)
- Test: `experiments/scripts/check_known_input_masking.py` (new, standalone — matches
  this repo's `check_*.py` convention, no pytest)

**Interfaces:**
- Produces: `preprocess_batch(..., use_known_input: bool = False, known_mix_zero_frac:
  float = 0.3, known_reveal_max: float = 1.0, known_terms_from_dataset: bool = False,
  rng: Optional["random.Random"] = None)` — when `use_known_input` is set, the returned
  `output_dict` gains `"known_term_ids": List[List[int]]` (one list of known label ids
  per example, raw ids indexing into `unique_items`/`label2id`) and `"loss_mask"`
  (`List[torch.Tensor]`, same `(num_labels,)`-per-example shape the existing
  `mask_sparse_negatives_max_total_terms`/`loss_mask_from_dataset` paths already produce
  at `training_run.py:420-454` — 0 at each known position, 1 elsewhere).
- Consumes (unchanged): `labels_list` (the dense per-example label tensors already built
  at `training_run.py:399-417` when `"labels" in examples`), `examples["known_labels"]`
  (an already-encoded list of label ids per row — the exact shape
  `experiments/pk_eval/scripts/build_temporal_pk_eval.py` already writes, consumed when
  `known_terms_from_dataset=True`).

- [ ] **Step 1: Write the failing test**

Create `experiments/scripts/check_known_input_masking.py`:

```python
"""Standalone check for preprocess_batch's known/novel masking (Task 1 of
docs/superpowers/plans/2026-10-08-known-annotation-conditioning.md). Run directly:
    env/bin/python experiments/scripts/check_known_input_masking.py
"""
import random
import sys
from pathlib import Path

# training_run.py lives at the repo root; this script lives two levels down
# (experiments/scripts/), so the repo root needs to go on sys.path explicitly -
# same ruling as experiments/pk_eval/scripts/*.py (Python puts a script's own
# directory on sys.path, not the repo root, when run directly).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from training_run import create_esm_tokenizer, preprocess_batch

UNIQUE_ITEMS = ["GO:A", "GO:B", "GO:C"]
LABEL2ID = {name: i for i, name in enumerate(UNIQUE_ITEMS)}


def _run(examples, **kwargs):
    tokenizer = create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    return preprocess_batch(
        examples, tokenizer=tokenizer, label2id=LABEL2ID, unique_items=UNIQUE_ITEMS,
        **kwargs,
    )


def check_forced_zero_known():
    examples = {"sequence": ["MKT", "MKTG"], "labels": [[0, 2], [1]]}
    out = _run(examples, use_known_input=True, known_mix_zero_frac=1.0)
    assert out["known_term_ids"] == [[], []], out["known_term_ids"]
    for mask in out["loss_mask"]:
        assert torch.all(mask == 1.0), mask
    print("[OK] forced known_mix_zero_frac=1.0 -> every example has zero known terms")


def check_partial_reveal_properties():
    # known_mix_zero_frac=0.0 -> never a forced-zero draw; known_reveal_max=1.0 ->
    # reveal fraction can be anywhere in [0, 1). Assert structural properties rather
    # than exact sampled ids, since the output is genuinely randomized by design.
    examples = {"sequence": ["MKT"], "labels": [[0, 1, 2]]}
    out = _run(examples, use_known_input=True, known_mix_zero_frac=0.0,
               known_reveal_max=1.0, rng=random.Random(42))
    known_ids = out["known_term_ids"][0]
    positive_ids = {0, 1, 2}
    assert set(known_ids).issubset(positive_ids), known_ids
    assert len(known_ids) == len(set(known_ids)), "known ids must not repeat"
    mask = out["loss_mask"][0]
    for idx in range(len(UNIQUE_ITEMS)):
        expected = 0.0 if idx in known_ids else 1.0
        assert mask[idx].item() == expected, (idx, mask)
    print(f"[OK] partial reveal: known_ids={known_ids}, loss_mask matches exactly")


def check_few_positives_edge_case():
    # Exactly 1 positive term + known_reveal_max < 1.0 -> floor(reveal_frac * 1) == 0
    # for every possible reveal_frac in [0, known_reveal_max) < 1.0 -> always degrades
    # to zero known terms, deterministically, regardless of rng state.
    examples = {"sequence": ["MKT"], "labels": [[1]]}
    for seed in range(5):
        out = _run(examples, use_known_input=True, known_mix_zero_frac=0.0,
                   known_reveal_max=0.5, rng=random.Random(seed))
        assert out["known_term_ids"] == [[]], out["known_term_ids"]
    print("[OK] single-positive protein always degrades to zero known terms, no crash")


def check_known_terms_from_dataset():
    # No "labels" column at all -- matches experiments/pk_eval's validation dataset
    # shape exactly (it has known_labels/novel_labels, never a generic labels column).
    examples = {"sequence": ["MKT", "MKTG"], "known_labels": [[0, 2], []]}
    out = _run(examples, use_known_input=True, known_terms_from_dataset=True)
    assert out["known_term_ids"] == [[0, 2], []], out["known_term_ids"]
    assert torch.equal(out["loss_mask"][0], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(out["loss_mask"][1], torch.tensor([1.0, 1.0, 1.0]))
    print("[OK] known_terms_from_dataset reads known_labels directly, no randomization")


def check_missing_labels_raises():
    examples = {"sequence": ["MKT"]}  # no "labels", no "known_labels"
    try:
        _run(examples, use_known_input=True, known_mix_zero_frac=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("[OK] use_known_input without a labels column raises ValueError")


if __name__ == "__main__":
    check_forced_zero_known()
    check_partial_reveal_properties()
    check_few_positives_edge_case()
    check_known_terms_from_dataset()
    check_missing_labels_raises()
    print("[OK] all checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/scripts/check_known_input_masking.py`
Expected: `TypeError: preprocess_batch() got an unexpected keyword argument
'use_known_input'`

- [ ] **Step 3: Add `import random` and seed it in `main()`**

In `training_run.py`'s import block (`training_run.py:1-10`), insert alphabetically:

```python
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
```

In `main()`, right after the existing seeding calls (`training_run.py:935-937`):

```python
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
```

- [ ] **Step 4: Implement the masking logic in `preprocess_batch`**

Add the new parameters to `preprocess_batch`'s signature (`training_run.py:353-361`):

```python
def preprocess_batch(
    examples,
    tokenizer: EsmTokenizer,
    label2id: Dict[str, int],
    unique_items: List[str],
    max_length: int = MAX_LENGTH,
    mask_sparse_negatives_max_total_terms: Optional[int] = None,
    loss_mask_from_dataset: bool = False,
    use_known_input: bool = False,
    known_mix_zero_frac: float = 0.3,
    known_reveal_max: float = 1.0,
    known_terms_from_dataset: bool = False,
    rng: Optional[random.Random] = None,
):
```

Then, right before `return output_dict` (`training_run.py:468`), add a new independent
block (independent of the `if "labels" in examples:` block above it, so it also works
against a dataset like `experiments/pk_eval`'s validation set that has `known_labels`
but no generic `labels` column at all):

```python
    if use_known_input:
        if known_terms_from_dataset:
            if "known_labels" not in examples:
                raise ValueError("known_terms_from_dataset=True requires a known_labels column")
            raw_known = examples["known_labels"]
            if not isinstance(raw_known, list):
                raw_known = [raw_known]
            known_ids_list = [[int(i) for i in ids] for ids in raw_known]
        else:
            if "labels" not in examples:
                raise ValueError("use_known_input (synthetic mode) requires a labels column")
            # labels_list is built above whenever "labels" in examples (training_run.py:399-417) -
            # reused here rather than re-decoding raw_labels, since it's already a dense
            # per-example tensor over unique_items.
            active_rng = rng if rng is not None else random
            known_ids_list = []
            for labels in labels_list:
                positive_ids = torch.nonzero(labels, as_tuple=True)[0].tolist()
                if not positive_ids or active_rng.random() < known_mix_zero_frac:
                    known_ids_list.append([])
                    continue
                reveal_frac = active_rng.uniform(0.0, known_reveal_max)
                n_reveal = int(reveal_frac * len(positive_ids))
                known_ids_list.append(active_rng.sample(positive_ids, n_reveal))

        output_dict["known_term_ids"] = known_ids_list
        loss_mask_list = []
        for known_ids in known_ids_list:
            mask = torch.ones((len(unique_items),), dtype=torch.float)
            for idx in known_ids:
                mask[idx] = 0.0
            loss_mask_list.append(mask)
        output_dict["loss_mask"] = loss_mask_list
```

- [ ] **Step 5: Run test to verify it passes**

Run: `env/bin/python experiments/scripts/check_known_input_masking.py`
Expected: five `[OK]` lines, ending with `[OK] all checks passed`.

- [ ] **Step 6: Add the CLI flags and post-parse validation in `main()`**

In `main()`'s `argparse` setup, right after `--use-aspect-input`
(`training_run.py:888`):

```python
    parser.add_argument("--use-known-input", action=argparse.BooleanOptionalAction, default=False,
                         help="Synthetically reveal a random subset of each training example's own "
                              "positive GO terms as a 'known' input to the classifier head (see "
                              "KnownAwareClassificationHead / docs/superpowers/specs/"
                              "2026-10-08-known-annotation-conditioning-design.md), excluding revealed "
                              "terms from the loss. Re-randomized every epoch (on-the-fly, no new "
                              "dataset). Mutually exclusive with --use-aspect-input, --use-lora, "
                              "--mask-sparse-negatives-max-total-terms, and --loss-mask-from-dataset.")
    parser.add_argument("--known-mix-zero-frac", type=float, default=0.3,
                         help="Fraction of --use-known-input draws forced to zero known terms (pure "
                              "NK), so the model doesn't forget how to perform the original NK task.")
    parser.add_argument("--known-reveal-max", type=float, default=1.0,
                         help="Upper bound of the Uniform(0, known_reveal_max) reveal-fraction "
                              "sampling range for non-zero --use-known-input draws. Must be in (0, 1].")
```

Right after `args = parser.parse_args()` (`training_run.py:933`), before
`torch.manual_seed(args.seed)`:

```python
    if args.use_known_input:
        if args.mask_sparse_negatives_max_total_terms is not None or args.loss_mask_from_dataset:
            raise ValueError(
                "--use-known-input cannot be combined with --mask-sparse-negatives-max-total-terms "
                "or --loss-mask-from-dataset -- all three write loss_mask and there's no defined "
                "combined semantics (see docs/superpowers/specs/"
                "2026-10-08-known-annotation-conditioning-design.md)")
        if not (0.0 < args.known_reveal_max <= 1.0):
            raise ValueError(f"--known-reveal-max must be in (0, 1], got {args.known_reveal_max}")
```

- [ ] **Step 7: Wire the flags into `preprocess_batch_partial`**

In `main()`'s `preprocess_batch_partial` construction (`training_run.py:951-960`):

```python
    preprocess_batch_partial = partial(
        preprocess_batch,
        tokenizer=tokenizer,
        max_length=args.max_length,
        unique_items=label_names,
        label2id=label2id,
        mask_sparse_negatives_max_total_terms=args.mask_sparse_negatives_max_total_terms,
        loss_mask_from_dataset=args.loss_mask_from_dataset,
        use_known_input=args.use_known_input,
        known_mix_zero_frac=args.known_mix_zero_frac,
        known_reveal_max=args.known_reveal_max,
    )
```

(`known_terms_from_dataset`/`rng` stay at their defaults here — `training_run.py`'s own
CLI never sets them; `known_terms_from_dataset=True` is only used by
`score_temporal_pk_eval.py`'s direct call, wired in Task 6.)

- [ ] **Step 8: Commit**

```bash
git add training_run.py experiments/scripts/check_known_input_masking.py
git commit -m "feat: add synthetic known/novel masking to preprocess_batch"
```

---

## Task 2: `known_indices`/`known_offsets` batching in `data_collate_function`

**Files:**
- Modify: `training_run.py:471-502` (`data_collate_function`)
- Test: `experiments/scripts/check_known_input_collate.py` (new)

**Interfaces:**
- Consumes: `preprocessed_batch[i]["known_term_ids"]` (`List[int]`, from Task 1).
- Produces: `collated_data["known_indices"]` (`torch.LongTensor`, flattened across the
  batch), `collated_data["known_offsets"]` (`torch.LongTensor`, shape `(batch_size,)` —
  the start offset of each example's bag within `known_indices`, the standard
  `nn.EmbeddingBag(indices, offsets)` calling convention) — present only when at least
  one sample in the batch carries `"known_term_ids"`.

- [ ] **Step 1: Write the failing test**

Create `experiments/scripts/check_known_input_collate.py`:

```python
"""Standalone check for data_collate_function's known_indices/known_offsets batching
(Task 2 of docs/superpowers/plans/2026-10-08-known-annotation-conditioning.md). Run:
    env/bin/python experiments/scripts/check_known_input_collate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from training_run import create_esm_tokenizer, data_collate_function


def check_variable_length_bags():
    tokenizer = create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    preprocessed_batch = [
        {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]),
         "known_term_ids": [0, 2, 5]},
        {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]),
         "known_term_ids": []},
        {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]),
         "known_term_ids": [7]},
    ]
    collated = data_collate_function(preprocessed_batch, tokenizer=tokenizer)
    assert torch.equal(collated["known_indices"], torch.tensor([0, 2, 5, 7], dtype=torch.long))
    assert torch.equal(collated["known_offsets"], torch.tensor([0, 3, 3], dtype=torch.long))

    # EmbeddingBag must consume this exactly (verified directly: an empty bag sums to
    # the zero vector, not NaN or an error).
    eb = torch.nn.EmbeddingBag(10, 4, mode="sum")
    out = eb(collated["known_indices"], collated["known_offsets"])
    assert out.shape == (3, 4)
    assert torch.allclose(out[1], torch.zeros(4)), "example 1's empty bag must be the zero vector"
    print("[OK] variable-length known-term bags batch and pool correctly")


def check_all_empty_batch():
    tokenizer = create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    preprocessed_batch = [
        {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]), "known_term_ids": []},
        {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]), "known_term_ids": []},
    ]
    collated = data_collate_function(preprocessed_batch, tokenizer=tokenizer)
    assert collated["known_indices"].numel() == 0
    assert torch.equal(collated["known_offsets"], torch.tensor([0, 0], dtype=torch.long))
    eb = torch.nn.EmbeddingBag(10, 4, mode="sum")
    out = eb(collated["known_indices"], collated["known_offsets"])
    assert torch.allclose(out, torch.zeros(2, 4)), "an all-empty batch must not crash EmbeddingBag"
    print("[OK] an all-zero-known batch collates to an empty indices tensor without crashing")


def check_absent_when_no_known_term_ids():
    tokenizer = create_esm_tokenizer("facebook/esm2_t6_8M_UR50D")
    preprocessed_batch = [
        {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1])},
    ]
    collated = data_collate_function(preprocessed_batch, tokenizer=tokenizer)
    assert "known_indices" not in collated and "known_offsets" not in collated
    print("[OK] known_indices/known_offsets stay absent when no sample carries known_term_ids")


if __name__ == "__main__":
    check_variable_length_bags()
    check_all_empty_batch()
    check_absent_when_no_known_term_ids()
    print("[OK] all checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/scripts/check_known_input_collate.py`
Expected: `KeyError: 'known_indices'`

- [ ] **Step 3: Implement the batching in `data_collate_function`**

Insert right after the existing `loss_mask` block (`training_run.py:492-494`), before
the `protein` gathering block:

```python
    known_term_ids = [sample["known_term_ids"] for sample in preprocessed_batch if "known_term_ids" in sample]
    if known_term_ids:
        flat_indices: List[int] = []
        offsets: List[int] = []
        cursor = 0
        for ids in known_term_ids:
            offsets.append(cursor)
            flat_indices.extend(ids)
            cursor += len(ids)
        collated_data["known_indices"] = torch.tensor(flat_indices, dtype=torch.long)
        collated_data["known_offsets"] = torch.tensor(offsets, dtype=torch.long)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/scripts/check_known_input_collate.py`
Expected: three `[OK]` lines, ending with `[OK] all checks passed`.

- [ ] **Step 5: Commit**

```bash
git add training_run.py experiments/scripts/check_known_input_collate.py
git commit -m "feat: batch known_term_ids into EmbeddingBag indices/offsets"
```

---

## Task 3: `KnownAwareClassificationHead` / `KnownConditionedEsmForSequenceClassification`

**Files:**
- Modify: `training_run.py:197-199` (insert new classes between
  `AspectConditionedEsmForSequenceClassification` and `create_esm`)
- Test: `experiments/scripts/check_known_aware_head.py` (new)

**Interfaces:**
- Produces: `KnownAwareClassificationHead(config, known_embed_dim: int = 64)` (a
  `torch.nn.Module` with `.known_embedding: nn.EmbeddingBag`, `.known_embed_dim: int`,
  `.forward(features, known_indices=None, known_offsets=None, **kwargs) ->
  torch.Tensor` of shape `(batch, config.num_labels)`); `KnownConditionedEsmForSequenceClassification`
  (subclass of `EsmForSequenceClassification`, `.forward(input_ids=None,
  attention_mask=None, known_indices=None, known_offsets=None, **kwargs) ->
  SequenceClassifierOutput`).
- Consumes: `known_indices`/`known_offsets` (Task 2's collate output shape).

No dedicated unit test exercises `KnownConditionedEsmForSequenceClassification`'s full
forward pass here (that needs a real ESM backbone load) — it's verified end-to-end by
Task 5's CPU smoke run instead, the same way
`AspectConditionedEsmForSequenceClassification` has never had its own standalone
backbone-level test in this repo. This task's test covers
`KnownAwareClassificationHead` in isolation (pure `torch.nn.Module`, no backbone,
fast).

- [ ] **Step 1: Write the failing test**

Create `experiments/scripts/check_known_aware_head.py`:

```python
"""Standalone check for KnownAwareClassificationHead (Task 3 of
docs/superpowers/plans/2026-10-08-known-annotation-conditioning.md). Run:
    env/bin/python experiments/scripts/check_known_aware_head.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from training_run import KnownAwareClassificationHead


class _FakeConfig:
    hidden_size = 8
    hidden_dropout_prob = 0.0
    num_labels = 5
    initializer_range = 0.02


def check_output_shape_and_empty_bag_equivalence():
    torch.manual_seed(0)
    head = KnownAwareClassificationHead(_FakeConfig(), known_embed_dim=4)
    head.eval()

    batch, seq_len = 3, 6
    features = torch.randn(batch, seq_len, _FakeConfig.hidden_size)

    # example 0: empty known set; example 1: known={2, 4}; example 2: empty known set
    known_indices = torch.tensor([2, 4], dtype=torch.long)
    known_offsets = torch.tensor([0, 0, 2], dtype=torch.long)

    logits = head(features, known_indices=known_indices, known_offsets=known_offsets)
    assert logits.shape == (batch, _FakeConfig.num_labels), logits.shape

    # Manual cross-check for example 0: an empty EmbeddingBag bag must contribute
    # exactly the zero vector, verified independently against torch's own EmbeddingBag
    # semantics -- this checks the WIRING (forward()'s concatenation), not EmbeddingBag
    # itself.
    cls0 = features[0, 0, :]
    x0 = torch.cat([cls0, torch.zeros(4)])
    x0 = head.dense(x0)
    x0 = torch.tanh(x0)
    x0 = head.out_proj(x0)
    assert torch.allclose(logits[0], x0, atol=1e-6), (logits[0], x0)
    print("[OK] output shape correct, empty-bag example matches manual zero-vector computation")


def check_missing_known_input_raises():
    head = KnownAwareClassificationHead(_FakeConfig(), known_embed_dim=4)
    features = torch.randn(1, 3, _FakeConfig.hidden_size)
    try:
        head(features)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("[OK] missing known_indices/known_offsets raises ValueError")


if __name__ == "__main__":
    check_output_shape_and_empty_bag_equivalence()
    check_missing_known_input_raises()
    print("[OK] all checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/scripts/check_known_aware_head.py`
Expected: `ImportError: cannot import name 'KnownAwareClassificationHead' from 'training_run'`

- [ ] **Step 3: Implement the classes**

Insert into `training_run.py` right after
`AspectConditionedEsmForSequenceClassification`'s `forward()` ends and before
`def create_esm(` (`training_run.py:197-199`):

```python
class KnownAwareClassificationHead(torch.nn.Module):
    """Like AspectAwareClassificationHead (above), but the auxiliary input is a
    variable-size set of known GO term ids rather than a fixed 3-dim aspect vector:
    pooled via a learned per-term embedding + sum (nn.EmbeddingBag), not concatenated
    directly. A raw one-hot concat of the full 27,191-dim label space was rejected --
    see docs/superpowers/specs/2026-10-08-known-annotation-conditioning-design.md's
    "Alternatives considered": it would add ~27191*hidden_size parameters to one dense
    layer and tends to memorize specific known-term combinations rather than
    generalize across related GO terms. The embedding table's vocabulary size is
    config.num_labels -- every term is both a known-input candidate and a prediction
    target, the same vocabulary either way.
    """

    def __init__(self, config, known_embed_dim: int = 64):
        super().__init__()
        self.known_embed_dim = known_embed_dim
        self.known_embedding = torch.nn.EmbeddingBag(config.num_labels, known_embed_dim, mode="sum")
        self.dense = torch.nn.Linear(config.hidden_size + known_embed_dim, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor, known_indices: Optional[torch.Tensor] = None,
                known_offsets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if known_indices is None or known_offsets is None:
            raise ValueError("KnownAwareClassificationHead requires known_indices/known_offsets "
                              "(flattened EmbeddingBag inputs -- see data_collate_function)")
        x = features[:, 0, :]  # <s>/CLS token, matching EsmClassificationHead
        x = self.dropout(x)
        known_vec = self.known_embedding(known_indices, known_offsets)
        x = torch.cat([x, known_vec.to(dtype=x.dtype, device=x.device)], dim=-1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class KnownConditionedEsmForSequenceClassification(EsmForSequenceClassification):
    """EsmForSequenceClassification with its classifier head swapped for
    KnownAwareClassificationHead. Mirrors AspectConditionedEsmForSequenceClassification
    exactly (above) -- loss is computed externally by training_run.py, not here.
    """

    def __init__(self, config, known_embed_dim: int = 64):
        super().__init__(config)
        self.classifier = KnownAwareClassificationHead(config, known_embed_dim=known_embed_dim)
        std = getattr(config, "initializer_range", None) or 0.02
        with torch.no_grad():
            for module in self.classifier.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, input_ids=None, attention_mask=None, known_indices=None,
                known_offsets=None, **kwargs) -> SequenceClassifierOutput:
        outputs = self.esm(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, known_indices=known_indices, known_offsets=known_offsets)
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/scripts/check_known_aware_head.py`
Expected: two `[OK]` lines, ending with `[OK] all checks passed`.

- [ ] **Step 5: Commit**

```bash
git add training_run.py experiments/scripts/check_known_aware_head.py
git commit -m "feat: add KnownAwareClassificationHead and KnownConditionedEsmForSequenceClassification"
```

---

## Task 4: Wire `--use-known-input` through `create_esm`, the training/validation loops, `run_diagnostic`, and the best-checkpoint reload

**Files:**
- Modify: `training_run.py:200-350` (`create_esm`), `training_run.py:716-765`
  (`run_diagnostic`), `training_run.py:844-933` (`argparse` additions),
  `training_run.py:965-1010` (`model_esm = create_esm(...)` call site),
  `training_run.py:1174-1175` (`use_loss_mask`/`criterion` construction),
  `training_run.py:1219-1222` (train-loop `model_kwargs`),
  `training_run.py:1319-1322` (val-loop `model_kwargs`),
  `training_run.py:1473-1505` (best-checkpoint reload + `run_diagnostic` call site)
- Test: `experiments/scripts/check_create_esm_known_input.py` (new)

**Interfaces:**
- Consumes: `KnownAwareClassificationHead`/`KnownConditionedEsmForSequenceClassification`
  (Task 3), `known_indices`/`known_offsets` batch keys (Task 2).
- Produces: `create_esm(..., use_known_input: bool = False, known_embed_dim: int = 64)`;
  `run_diagnostic(..., use_known_input: bool = False)`.

- [ ] **Step 1: Write the failing test**

Create `experiments/scripts/check_create_esm_known_input.py`:

```python
"""Standalone check for create_esm's --use-known-input wiring (Task 4 of
docs/superpowers/plans/2026-10-08-known-annotation-conditioning.md). Loads a real tiny
ESM backbone (facebook/esm2_t6_8M_UR50D, ~8M params) -- network access required once
(cached afterward), same as this repo's own CPU smoke-test convention. Run:
    env/bin/python experiments/scripts/check_create_esm_known_input.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training_run import KnownConditionedEsmForSequenceClassification, create_esm

ID2LABEL = {0: "GO:A", 1: "GO:B", 2: "GO:C"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
TINY_MODEL = "facebook/esm2_t6_8M_UR50D"


def check_use_known_input_selects_the_right_class():
    model = create_esm(
        unfreeze_last_n=0, model_type=TINY_MODEL, id2label=ID2LABEL, label2id=LABEL2ID,
        use_known_input=True, known_embed_dim=8,
    )
    assert isinstance(model, KnownConditionedEsmForSequenceClassification), type(model)
    assert model.classifier.known_embed_dim == 8
    print("[OK] use_known_input=True selects KnownConditionedEsmForSequenceClassification")


def check_use_known_input_plus_use_lora_raises():
    try:
        create_esm(
            model_type=TINY_MODEL, id2label=ID2LABEL, label2id=LABEL2ID,
            use_known_input=True, use_lora=True,
        )
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
    print("[OK] use_known_input + use_lora raises NotImplementedError")


def check_use_known_input_plus_use_aspect_input_raises():
    try:
        create_esm(
            model_type=TINY_MODEL, id2label=ID2LABEL, label2id=LABEL2ID,
            use_known_input=True, use_aspect_input=True,
        )
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
    print("[OK] use_known_input + use_aspect_input raises NotImplementedError")


if __name__ == "__main__":
    check_use_known_input_selects_the_right_class()
    check_use_known_input_plus_use_lora_raises()
    check_use_known_input_plus_use_aspect_input_raises()
    print("[OK] all checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/scripts/check_create_esm_known_input.py`
Expected: `TypeError: create_esm() got an unexpected keyword argument 'use_known_input'`

- [ ] **Step 3: Add `use_known_input`/`known_embed_dim` to `create_esm`**

Add the two new parameters to `create_esm`'s signature (`training_run.py:200-219`),
right after `use_aspect_input: bool = False`:

```python
    use_aspect_input: bool = False,
    use_known_input: bool = False,
    known_embed_dim: int = 64,
):
```

Right after the existing `use_aspect_input and use_lora` guard
(`training_run.py:223-227`):

```python
    if use_known_input and use_lora:
        raise NotImplementedError(
            "--use-known-input + --use-lora is not implemented - KnownConditionedEsmForSequenceClassification "
            "only replaces the plain (non-LoRA) classifier head."
        )
    if use_known_input and use_aspect_input:
        raise NotImplementedError(
            "--use-known-input + --use-aspect-input is not implemented - no classifier head "
            "concatenates both auxiliary inputs today (see the design spec's non-goals)."
        )
```

Replace the non-LoRA model-class selection (`training_run.py:323-331`):

```python
    model_cls = (
        KnownConditionedEsmForSequenceClassification if use_known_input
        else AspectConditionedEsmForSequenceClassification if use_aspect_input
        else AutoModelForSequenceClassification
    )
    extra_model_kwargs = {"known_embed_dim": known_embed_dim} if use_known_input else {}
    model = model_cls.from_pretrained(
        model_type,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        revision=revision,
        **extra_model_kwargs,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/scripts/check_create_esm_known_input.py`
Expected: three `[OK]` lines, ending with `[OK] all checks passed`.

- [ ] **Step 5: Wire `--known-embed-dim` and the loop/reload call sites (no dedicated test — covered by Task 5's end-to-end run)**

Add the CLI flag right after `--known-reveal-max` (added in Task 1):

```python
    parser.add_argument("--known-embed-dim", type=int, default=64,
                         help="Embedding dimension per GO term for --use-known-input's "
                              "nn.EmbeddingBag known-term pooling.")
```

In `main()`'s `model_esm = create_esm(...)` call (`training_run.py:965-1001`), add:

```python
        use_aspect_input=args.use_aspect_input,
        use_known_input=args.use_known_input,
        known_embed_dim=args.known_embed_dim,
    ).to(device)
```

`use_loss_mask`/`criterion` construction (`training_run.py:1174-1175`) must also treat
`--use-known-input` as a `loss_mask`-writing flag (Task 1's masking writes `loss_mask`
too, and `compute_loss` requires `reduction="none"` whenever any `loss_mask` is present):

```python
    use_loss_mask = (args.mask_sparse_negatives_max_total_terms is not None
                      or args.loss_mask_from_dataset or args.use_known_input)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none" if use_loss_mask else "mean")
```

Train-loop `model_kwargs` (`training_run.py:1219-1222`):

```python
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if args.use_aspect_input:
                model_kwargs["aspect_input"] = batch["aspect_input"].to(device)
            if args.use_known_input:
                model_kwargs["known_indices"] = batch["known_indices"].to(device)
                model_kwargs["known_offsets"] = batch["known_offsets"].to(device)
            loss_mask = batch["loss_mask"].to(device) if use_loss_mask else None
```

Validation-loop `model_kwargs` (`training_run.py:1319-1322`), identical addition:

```python
                model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if args.use_aspect_input:
                    model_kwargs["aspect_input"] = batch["aspect_input"].to(device)
                if args.use_known_input:
                    model_kwargs["known_indices"] = batch["known_indices"].to(device)
                    model_kwargs["known_offsets"] = batch["known_offsets"].to(device)
                loss_mask = batch["loss_mask"].to(device) if use_loss_mask else None
```

`run_diagnostic`'s signature (`training_run.py:716-727`) gains a matching parameter,
right after `use_aspect_input: bool = False`:

```python
    use_aspect_input: bool = False,
    use_known_input: bool = False,
) -> Dict[str, Any]:
```

and its loop (`training_run.py:761-763`):

```python
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if use_aspect_input:
                model_kwargs["aspect_input"] = batch["aspect_input"].to(device)
            if use_known_input:
                model_kwargs["known_indices"] = batch["known_indices"].to(device)
                model_kwargs["known_offsets"] = batch["known_offsets"].to(device)
```

Its call site (`training_run.py:1494-1505`) gains:

```python
            use_aspect_input=args.use_aspect_input,
            use_known_input=args.use_known_input,
        )
```

Best-checkpoint reload block (`training_run.py:1473-1490`) — insert a new branch
between the existing `if args.use_lora:` and `else:`:

```python
        if args.use_lora:
            best_model = create_esm(
                model_type=args.model_name,
                use_lora=True,
                lora_resume_repo_id=str(best_checkpoint_dir),
                id2label=id2label,
                label2id=label2id,
                lora_base_dtype=lora_base_dtype,
            ).to(device)
        elif args.use_known_input:
            best_model = KnownConditionedEsmForSequenceClassification.from_pretrained(
                str(best_checkpoint_dir),
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
                problem_type="multi_label_classification",
                known_embed_dim=args.known_embed_dim,
            ).to(device)
        else:
            best_model_cls = AspectConditionedEsmForSequenceClassification if args.use_aspect_input else AutoModelForSequenceClassification
            best_model = best_model_cls.from_pretrained(
                str(best_checkpoint_dir),
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
                problem_type="multi_label_classification",
            ).to(device)
```

- [ ] **Step 6: Commit**

```bash
git add training_run.py experiments/scripts/check_create_esm_known_input.py
git commit -m "feat: wire --use-known-input through create_esm, the train/val loops, run_diagnostic, and checkpoint reload"
```

---

## Task 5: End-to-end CPU smoke test, then a real GPU training run

**Files:** none (verification-only task)

**Interfaces:** none new — exercises every interface from Tasks 1-4 together.

- [ ] **Step 1: Run the CPU smoke test**

Run:

```bash
env/bin/python training_run.py \
  --data-path data/cafa-6-protein-function-prediction/dataset_full \
  --use-known-input --known-mix-zero-frac 0.3 --known-reveal-max 1.0 --known-embed-dim 64 \
  --max-train-samples 100 --max-val-samples 50 --epochs 1 --device cpu
```

Expected: training completes without error through the single epoch, printing a
decreasing or stable `train_loss` across steps and a final validation loss — the same
shape of output `training_run.py`'s existing CPU smoke-test invocation
(`CLAUDE.md`'s "Common commands") already produces, just with the new flags exercised.
If this crashes, the error will point at a specific line in Tasks 1-4's wiring (most
likely a missed `model_kwargs` call site, per this plan's Review Focus item on
checkpoint-reload/embed-dim consistency) — fix it, re-run, and only proceed once this
passes cleanly.

- [ ] **Step 2: Ask the user to run a real GPU training trial**

This step needs real GPU throughput (per `hardware_local_gpu.md`: run GPU-assisted work
locally via a `!`-command, not in-sandbox). Hand the user this command to run themselves
via `!`:

```bash
env/bin/python training_run.py \
  --data-path data/cafa-6-protein-function-prediction/dataset_full \
  --model-name facebook/esm2_t12_35M_UR50D --unfreeze-last-n 4 \
  --esm-dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --batch-size 2 --grad-accum-steps 8 --epochs 3 \
  --use-known-input --known-mix-zero-frac 0.3 --known-reveal-max 1.0 --known-embed-dim 64 \
  --run-diagnostic --run-name known-input-trial-001 \
  --push-to-hub --hf-repo-id kelmoir/cafa6-esm --checkpoint-mode final
```

(Hyperparameters other than the new `--known-*` flags are the Tier-0 plateau sweep's
settled defaults — `experiments/full_label_sweep/sweep_log.md` — reused as-is since this
item doesn't re-sweep regularization; see this plan's Global Constraints.)

Expected: a completed trial pushed to `kelmoir/cafa6-esm@known-input-trial-001`, with a
`--run-diagnostic` AUROC/F1 summary printed and saved in that branch's
`trial_metrics.json`. Confirm with the user that this ran to completion and share the
diagnostic numbers before proceeding to Task 6 (Task 6's own verification needs this
pushed checkpoint to exist).

- [ ] **Step 3: Commit (log only — no code changed)**

Log the result in `experiments/cafa6_next_steps_plan.md` under Tier 1 item 2 part (B),
then commit:

```bash
git add experiments/cafa6_next_steps_plan.md
git commit -m "docs: log known-input-trial-001 training run"
```

---

## Task 6: Wire `score_temporal_pk_eval.py` to score a `--use-known-input` checkpoint

**Files:**
- Modify: `experiments/pk_eval/scripts/score_temporal_pk_eval.py`

**Interfaces:**
- Consumes: `training_run.preprocess_batch(..., use_known_input=True,
  known_terms_from_dataset=True)` (Task 1), `training_run.create_esm(...,
  use_known_input=True, known_embed_dim=...)` (Task 4), `KnownConditionedEsmForSequenceClassification`
  (Task 3), the pushed checkpoint from Task 5.
- Produces: `run_inference(..., use_known_input: bool = False)` — unchanged return type
  (`np.ndarray` of probabilities).

- [ ] **Step 1: Add the `use_known_input` parameter to `run_inference`**

Replace `run_inference`'s body (`experiments/pk_eval/scripts/score_temporal_pk_eval.py:79-99`):

```python
def run_inference(model, tokenizer, dataset, device, batch_size: int = 16,
                   use_known_input: bool = False) -> np.ndarray:
  import torch
  from functools import partial

  import training_run

  loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=False,
      collate_fn=partial(training_run.data_collate_function, tokenizer=tokenizer))
  was_training = model.training
  model.eval()
  all_probs = []
  with torch.inference_mode():
    for batch in loader:
      model_kwargs = {
          "input_ids": batch["input_ids"].to(device),
          "attention_mask": batch["attention_mask"].to(device),
      }
      if use_known_input:
        model_kwargs["known_indices"] = batch["known_indices"].to(device)
        model_kwargs["known_offsets"] = batch["known_offsets"].to(device)
      logits = model(**model_kwargs).logits
      all_probs.append(torch.sigmoid(logits.float()).cpu())
  if was_training:
    model.train()
  return torch.cat(all_probs).numpy()
```

- [ ] **Step 2: Detect `use_known_input` from the checkpoint's config and load the right model class**

In `main()`, right after `use_lora = trial_config.get("use_lora", False)`
(`experiments/pk_eval/scripts/score_temporal_pk_eval.py:139`):

```python
  use_known_input = trial_config.get("use_known_input", False)
  known_embed_dim = trial_config.get("known_embed_dim", 64)
```

Replace the non-LoRA model-loading branch (`experiments/pk_eval/scripts/score_temporal_pk_eval.py:148-152`):

```python
  if use_lora:
    model = training_run.create_esm(
        model_type=model_name, use_lora=True, lora_resume_repo_id=args.hf_repo_id,
        revision=args.revision, id2label=id2label, label2id=label2id,
    ).to(device)
  elif use_known_input:
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type=args.hf_repo_id, revision=args.revision,
        id2label=id2label, label2id=label2id,
        use_known_input=True, known_embed_dim=known_embed_dim,
    ).to(device)
  else:
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type=args.hf_repo_id, revision=args.revision,
        id2label=id2label, label2id=label2id,
    ).to(device)
```

- [ ] **Step 3: Feed the real t0-known terms through `preprocess_batch` and `run_inference`**

Replace the `preprocess`/`run_inference` calls
(`experiments/pk_eval/scripts/score_temporal_pk_eval.py:157-162`):

```python
  from functools import partial
  preprocess = partial(training_run.preprocess_batch, tokenizer=tokenizer,
                       unique_items=label_names, label2id=label2id,
                       use_known_input=use_known_input, known_terms_from_dataset=True)
  tokenized = validation.with_transform(transform=preprocess)

  probs = run_inference(model, tokenizer, tokenized, device, batch_size=args.batch_size,
                        use_known_input=use_known_input)
```

(`known_terms_from_dataset=True` makes Task 1's dataset-sourced branch read each row's
real `known_labels` column directly — the genuine temporal known-set part (A) already
computed, never a synthetic draw; a false `use_known_input` is a no-op exactly as
before, since Task 1's masking block only runs `if use_known_input`.)

- [ ] **Step 4: Verify against Task 5's pushed checkpoint (user-run, real GPU/CPU inference)**

Hand the user this command to run via `!`:

```bash
env/bin/python experiments/pk_eval/scripts/score_temporal_pk_eval.py \
  --hf-repo-id kelmoir/cafa6-esm --revision known-input-trial-001
```

Expected: a `[RESULT]` JSON report breaking Fmax down by aspect × PK/LK/NK/unseen
category, the same shape part (A)'s existing pure-NK run already produced — now scored
with the `--use-known-input` checkpoint actually consuming each PK/LK row's real known
terms (and an empty bag for NK/unseen rows). Confirm with the user that this ran
without error before treating the item as done.

- [ ] **Step 5: Log the comparison and commit**

Log the known-vs-pure-NK Fmax comparison in `experiments/cafa6_next_steps_plan.md`
under Tier 1 item 2 part (B), then commit:

```bash
git add experiments/pk_eval/scripts/score_temporal_pk_eval.py experiments/cafa6_next_steps_plan.md
git commit -m "feat: score --use-known-input checkpoints against real temporal known terms"
```
