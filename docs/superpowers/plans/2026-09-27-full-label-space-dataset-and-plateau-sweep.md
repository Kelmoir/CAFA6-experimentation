# Full-Label-Space Dataset + Plateau-First Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full ancestor-closure GO-term dataset variant (no depth cap, no
annotation-count threshold on the vocabulary) and launch a plateau-first regularization
sweep on it, producing the first genuinely-tuned baseline checkpoint that
`experiments/cafa6_next_steps_plan.md`'s Tier 1 work (ensembling, PK-conditioning,
hierarchy consistency) can build on.

**Architecture:** Add `generate_dataset_full.py`, a third `generate_dataset*.py` variant
alongside the existing coarse (`generate_dataset.py`, 383 terms, depth≤2, ≥50 annotations)
and raw (`generate_dataset_raw.py`, 668 terms, no propagation, ≥100 annotations) scripts.
It reuses `generate_dataset.py`'s propagation machinery (`restrict_to_true_path`,
`compute_depths`, `propagate_protein_terms`, `generate_dataset`, `export_dependency_graph`)
and `split_utils.compute_cluster_aware_split`, adding one new selection function
(`select_full_terms`) that keeps every term appearing anywhere in any protein's propagated
ancestor closure — no filtering at all. Output goes to
`data/cafa-6-protein-function-prediction/dataset_full` / Hub repo `kelmoir/cafa6-full`.
Once the dataset exists, `training_run.py` is reused completely unmodified (it is already
label-space-agnostic — `id2label`/`label2id` come from the dataset's own
`features["labels"].feature.names`) to run a plateau-first sweep, following the same
rubric/coordinate-descent process as `experiments/ancestor_sweep/sweep_log.md`, logged to a
new `experiments/full_label_sweep/sweep_log.md`.

**Tech Stack:** Python, pandas, obonet/networkx, Biopython, HuggingFace `datasets`, the
existing `training_run.py`/ESM-2 stack. No new dependencies.

**Spec:** `experiments/cafa6_next_steps_plan.md`, "Tier 0 prerequisite" (the "Open question
— RESOLVED 2026-09-26" section's decision, and the 2026-09-26 "Scope constraint: full
label space, no partial GO-term selection" section above it).

## Global Constraints

- No test framework is installed in this repo (`pytest` is not in `requirements-minimal.txt`
  and not importable in `env/`) and CLAUDE.md states there is no test suite/linter/CI
  configured. Per this repo's own established convention (`generate_dataset_raw.py`'s
  `select_raw_terms`, whose correctness is documented as "verified directly against
  `train_terms.tsv`"), verification in this plan uses standalone assertion scripts run via
  `env/bin/python <script>.py`, not pytest — but still written *before* the code they check,
  run to confirm they fail, then run again to confirm they pass, per the same red-green
  discipline TDD asks for.
- Reuse `generate_dataset.py`'s existing functions rather than re-deriving propagation
  logic — this is the established pattern `generate_dataset_raw.py` already follows.
- `export_dependency_graph`'s `selection_rule` parameter is mandatory here (its own
  docstring warns the default would "silently mislabel" a differently-selected term dict) —
  every call in this plan passes an explicit one.
- Random seed 42 throughout (dataset shuffling, cluster-aware split), per CLAUDE.md's "Data
  science conventions."
- The user has a local 16GB-VRAM GPU ([[hardware_local_gpu]]); all real training runs
  (the sweep in Task 4) happen there, not on Colab. The Bash sandbox tool itself has no GPU
  device passthrough — training commands are handed to the user as exact `!`-prefixed
  commands to run in their own terminal, never executed directly in-sandbox
  ([[feedback_no_long_cpu_runs_prefer_gpu]]).
- `data/cafa-6-protein-function-prediction/*` is gitignored — only `generate_dataset_full.py`
  itself and the sweep log get committed, not the generated dataset directory.
- Per the 2026-09-26 scope constraint: this variant must be the full label space — no depth
  cap, no annotation-count threshold. Don't reintroduce a threshold "to keep the vocabulary
  manageable" without checking with the user first — that was the exact thing this plan
  exists to fix.

## Review Focus

- Terms directly asserted in `train_terms.tsv` but absent from `go-basic.obo`'s graph
  (obsolete/unreachable term IDs) must still appear in the vocabulary with `depth=None`, not
  be silently dropped — this already bit `generate_dataset_raw.py`'s docstring once.
- A protein whose full closure ends up with zero in-vocabulary labels should be rare-to-never
  here (every protein propagates to at least an aspect root, which is always in the full
  vocabulary since roots are reachable via closure) — `generate_dataset()`'s existing
  "no labels in vocab" safety net must still report a real, small `removed` count, not
  silently swallow a large one.
- The vocabulary is expected to be very large (plausibly tens of thousands of terms, an order
  of magnitude beyond the 383/668-term existing variants) — `datasets.ClassLabel`
  construction and `Dataset.from_dict` must be checked to actually complete and produce a
  sane file size at this scale, not assumed to "just work" the way it does at 668 terms.
- The cluster-aware split must produce the *exact same* train/val/test protein ID assignment
  as `dataset`/`dataset_raw` (same `compute_cluster_aware_split(unique_protein_ids, seed=42)`
  call over the same protein universe) — a mismatch would make this variant's future results
  incomparable to the existing baselines, defeating half the point of building it.
- `export_dependency_graph`'s `selection_rule` argument must correctly describe this variant
  (`propagated: True, min_annotations: None, depth_cap: None`) — a copy-pasted wrong value
  here would silently mislabel the exported graph metadata for anyone reading it later.

---

## Task 1: `select_full_terms` — full ancestor-closure vocabulary selection

**Files:**
- Create: `generate_dataset_full.py`
- Create: `experiments/scripts/verify_select_full_terms.py`

**Interfaces:**
- Produces: `select_full_terms(propagated_terms: dict, depths: dict) -> dict`, returning
  `{term_id: {"depth": int | None, "count": int}}` for every term appearing anywhere in the
  union of all proteins' propagated closures — no filtering. Same shape as
  `generate_dataset.select_coarse_terms`/`generate_dataset_raw.select_raw_terms`, consumed by
  Task 2's `main()`.

- [ ] **Step 1: Write the failing verification script**

```python
"""Standalone verification for generate_dataset_full.select_full_terms.

This repo has no pytest/test framework (see CLAUDE.md's "no test suite, linter, or CI"
note) -- this follows the project's own established convention (generate_dataset_raw.py's
select_raw_terms, verified directly against a fixture/real data and asserted with plain
`assert`) instead of introducing a new test-framework dependency.
"""
from generate_dataset_full import select_full_terms


def test_no_threshold_and_full_union():
  propagated_terms = {
      "P1": {"GO:0000001", "GO:0000002"},
      "P2": {"GO:0000002"},
      "P3": {"GO:0000003"},  # singleton term -- must still be included, no threshold
  }
  depths = {"GO:0000001": 1, "GO:0000002": 2}  # GO:0000003 deliberately missing

  result = select_full_terms(propagated_terms, depths)

  assert set(result.keys()) == {"GO:0000001", "GO:0000002", "GO:0000003"}, result.keys()
  assert result["GO:0000001"] == {"depth": 1, "count": 1}, result["GO:0000001"]
  assert result["GO:0000002"] == {"depth": 2, "count": 2}, result["GO:0000002"]
  assert result["GO:0000003"] == {"depth": None, "count": 1}, (
      "a term missing from the depths dict (graph-external/obsolete) must still be "
      f"included, with depth=None, not dropped: {result.get('GO:0000003')}")
  print("[OK] test_no_threshold_and_full_union")


if __name__ == "__main__":
  test_no_threshold_and_full_union()
  print("[OK] all verifications passed")
```

Save this as `experiments/scripts/verify_select_full_terms.py`.

- [ ] **Step 2: Run it to verify it fails**

Run: `env/bin/python experiments/scripts/verify_select_full_terms.py`
Expected: `ModuleNotFoundError: No module named 'generate_dataset_full'` (the file doesn't
exist yet).

- [ ] **Step 3: Create `generate_dataset_full.py` with `select_full_terms`**

```python
"""Full GO-term dataset variant -- every term, no filtering of any kind.

Label vocabulary = every GO term that appears anywhere in any protein's propagated
ancestor closure (is_a/part_of propagation, the same true-path rule generate_dataset.py
already applies). No depth cap. No annotation-count threshold. Nothing is excluded for
being rare -- a term with exactly one supporting protein is in the vocabulary on exactly
the same footing as one with a hundred thousand. This is the full label space required by
the 2026-09-26 decision in experiments/cafa6_next_steps_plan.md ("Scope constraint: full
label space, no partial GO-term selection").

(For context only, not something this variant does: generate_dataset.py's coarse space
additionally restricts to depth<=2 terms with >=50 propagated annotations -- 383 terms --
and generate_dataset_raw.py drops propagation entirely and restricts to >=100
directly-asserted annotations -- 668 terms. This variant applies neither restriction.)

Reuses generate_dataset.py's sequence loading, propagation (restrict_to_true_path,
compute_depths, propagate_protein_terms), generate_dataset() (unmodified schema),
export_dependency_graph(), and split_utils.compute_cluster_aware_split (same
protein-universe split as every other generate_dataset*.py variant, so this dataset's
train/val/test protein IDs match kelmoir/cafa6 and kelmoir/cafa6-raw exactly).
generate_dataset.py's own pipeline is guarded behind `if __name__ == "__main__":`, so
importing it here does not re-run or re-push the original kelmoir/cafa6 dataset.
"""
from pathlib import Path

import datasets
import obonet
import pandas as pd
from Bio import SeqIO

from generate_dataset import (
    compute_depths,
    export_dependency_graph,
    generate_dataset,
    propagate_protein_terms,
    restrict_to_true_path,
)
from split_utils import compute_cluster_aware_split


def select_full_terms(propagated_terms: dict, depths: dict) -> dict:
  """EVERY GO term that appears anywhere in any protein's propagated ancestor closure --
  no depth cap, no annotation-count threshold, no filtering step at all. Unlike
  select_coarse_terms/select_raw_terms, there is no `if count >= min_annotations` (or
  any other) condition here: a term backed by a single protein is included exactly the
  same way as one backed by every protein in the dataset.

  Returns {term_id: {"depth", "count"}}, matching select_coarse_terms's/select_raw_terms's
  shape so export_dependency_graph's payload stays consistent across variants. depth is
  None for a term outside the true-path graph entirely (obsolete/unreachable term IDs that
  can still appear as directly-asserted train_terms.tsv rows and end up in a propagated
  closure even though propagate_protein_terms's ancestors_of() has no edges for them --
  the same edge case generate_dataset_raw.py's select_raw_terms already documents).
  """
  counts = {}
  for terms in propagated_terms.values():
    for term in terms:
      counts[term] = counts.get(term, 0) + 1
  # No threshold, no filter: every key in `counts` -- i.e. every term seen even once --
  # is returned. This dict comprehension only attaches depth metadata; it excludes nothing.
  return {
      term: {"depth": depths.get(term), "count": count}
      for term, count in counts.items()
  }
```

- [ ] **Step 4: Run the verification script again to confirm it passes**

Run: `env/bin/python experiments/scripts/verify_select_full_terms.py`
Expected:
```
[OK] test_no_threshold_and_full_union
[OK] all verifications passed
```

- [ ] **Step 5: Commit**

```bash
git add generate_dataset_full.py experiments/scripts/verify_select_full_terms.py
git commit -m "feat: add select_full_terms for the full ancestor-closure label space"
```

---

## Task 2: `main()` pipeline — build and locally verify the full-label-space dataset

**Files:**
- Modify: `generate_dataset_full.py` (add `main()`)
- Create: `experiments/scripts/verify_dataset_full_integrity.py`

**Interfaces:**
- Consumes: `select_full_terms` (Task 1), `generate_dataset.generate_dataset`,
  `generate_dataset.export_dependency_graph`, `split_utils.compute_cluster_aware_split`.
- Produces: `data/cafa-6-protein-function-prediction/dataset_full/` (a `DatasetDict` with
  `train`/`validation`/`test` splits, saved via `save_to_disk`) and
  `dataset_full/go_dependency_graph.json`, consumed by Task 3 (Hub push) and Task 4 (the
  sweep's `--data-path`).

- [ ] **Step 1: Add `main()` to `generate_dataset_full.py`**

```python
def main():
  data_path = Path("./data/cafa-6-protein-function-prediction")
  prot_sequences = {}
  for sequence in SeqIO.parse(data_path / "Train/train_sequences.fasta", "fasta"):
    prot_sequences[sequence.id.split("|")[1]] = str(sequence.seq).upper().replace(r"[UZOB]", "X")
  train_df = pd.read_csv(data_path / "Train/train_terms.tsv", delimiter="\t")
  obonet_graph = obonet.read_obo(data_path / "Train/go-basic.obo")

  true_path_graph = restrict_to_true_path(obonet_graph)
  depths = compute_depths(true_path_graph)
  propagated_terms = propagate_protein_terms(train_df, true_path_graph)
  full_terms = select_full_terms(propagated_terms, depths)
  print(f"[INFO] Selected {len(full_terms)} full GO terms "
        f"(every propagated term, no depth cap, no annotation-count threshold)")

  label2id = {term: idx for idx, term in enumerate(sorted(full_terms))}

  protein_to_labels = {
      protein_id: sorted(terms & full_terms.keys())
      for protein_id, terms in propagated_terms.items()
  }

  unique_protein_ids = list(train_df["EntryID"].unique())
  split_assignment = compute_cluster_aware_split(unique_protein_ids)
  train_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "train"]
  val_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "validation"]
  test_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "test"]
  print(f"Splitting IDs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

  dataset_dict = datasets.DatasetDict({
      "train": generate_dataset(train_ids, protein_to_labels, label2id, prot_sequences),
      "validation": generate_dataset(val_ids, protein_to_labels, label2id, prot_sequences),
      "test": generate_dataset(test_ids, protein_to_labels, label2id, prot_sequences),
  })

  out_path = data_path / "dataset_full"
  dataset_dict.save_to_disk(out_path)
  export_dependency_graph(obonet_graph, full_terms, out_path / "go_dependency_graph.json",
                          selection_rule={"min_annotations": None, "propagated": True, "depth_cap": None})

  print("\nFull-label-space dataset created successfully (not yet pushed to Hub -- see Task 3).")


if __name__ == "__main__":
  main()
```

Note: `push_to_hub` is deliberately **not** called here — Task 3 does that as its own,
separately-verified step, so a Hub/network failure can't be conflated with a local
generation bug.

- [ ] **Step 2: Run the pipeline for real**

This does real work over the full `train_terms.tsv` (537,028 rows) and is CPU-only
(pandas/networkx/Biopython, no GPU-routable code) — time-box it: run with
`run_in_background: true` and monitor. If it is still running past ~10 minutes, let it
continue in the background rather than assuming it hung (the existing `dataset`/`dataset_raw`
variants already do comparable work over the same file at a similar order of magnitude).

Run: `env/bin/python generate_dataset_full.py`

Expected output includes (exact numbers not predictable in advance — that's the point of
running it for real, not guessing):
```
[INFO] Selected N full GO terms (every propagated term, no depth cap, no annotation-count threshold)
Splitting IDs: Train=<n1>, Val=<n2>, Test=<n3>
[INFO] Generating Dataset with <count> items.
[INFO] Number of proteins removed (too long / no labels in vocab): <small number>
...
[INFO] Wrote GO dependency graph (... nodes, ... edges, N coarse terms) to .../go_dependency_graph.json

Full-label-space dataset created successfully (not yet pushed to Hub -- see Task 3).
```

Record the actual `N` (vocabulary size) and split sizes printed — expect `N` to be far
larger than the 668-term `dataset_raw` space (this is the whole point of the Tier 0 fix), but
the exact figure is only known once this run completes.

- [ ] **Step 3: Write and run the integrity verification script**

```python
"""Standalone integrity checks for data/cafa-6-protein-function-prediction/dataset_full,
run after generate_dataset_full.py's main(). No pytest in this repo -- see
verify_select_full_terms.py's header for why this is a plain assertion script.
"""
import datasets
import pandas as pd

from split_utils import compute_cluster_aware_split

DATA_PATH = "data/cafa-6-protein-function-prediction"


def test_split_matches_existing_variants():
  """dataset_full's train/val/test protein IDs must be IDENTICAL to dataset/dataset_raw's
  (same compute_cluster_aware_split(seed=42) call over the same protein universe) -- a
  mismatch would make future results on this variant incomparable to existing baselines.
  """
  train_df = pd.read_csv(f"{DATA_PATH}/Train/train_terms.tsv", delimiter="\t")
  unique_protein_ids = list(train_df["EntryID"].unique())
  expected_split = compute_cluster_aware_split(unique_protein_ids)

  dataset_full = datasets.load_from_disk(f"{DATA_PATH}/dataset_full")
  for split_name in ("train", "validation", "test"):
    actual_ids = set(dataset_full[split_name]["protein"])
    expected_ids = {pid for pid, split in expected_split.items() if split == split_name}
    unexpected = actual_ids - expected_ids
    assert not unexpected, (
        f"{split_name}: {len(unexpected)} protein IDs don't match the expected cluster-aware "
        f"split, e.g. {list(unexpected)[:5]}")
  print("[OK] test_split_matches_existing_variants")


def test_vocabulary_larger_than_existing_variants():
  """The full closure (no threshold) must be a strict superset in spirit of the existing,
  thresholded variants -- at minimum, far larger than dataset_raw's 668 terms.
  """
  dataset_full = datasets.load_from_disk(f"{DATA_PATH}/dataset_full")
  vocab_size = dataset_full["train"].features["labels"].feature.num_classes
  assert vocab_size > 668, (
      f"full-label-space vocabulary ({vocab_size} terms) is not larger than dataset_raw's "
      "668-term thresholded space -- select_full_terms may be filtering when it shouldn't")
  print(f"[OK] test_vocabulary_larger_than_existing_variants ({vocab_size} terms)")


def test_no_labels_removed_silently():
  """generate_dataset()'s 'no labels in vocab' safety net should remove close to zero
  proteins here (every protein propagates to at least an aspect root, which is always in
  a no-threshold vocabulary) -- a large removed count would mean something upstream broke.
  """
  dataset_full = datasets.load_from_disk(f"{DATA_PATH}/dataset_full")
  train_df = pd.read_csv(f"{DATA_PATH}/Train/train_terms.tsv", delimiter="\t")
  n_input_proteins = train_df["EntryID"].nunique()
  n_output_proteins = sum(len(dataset_full[s]) for s in ("train", "validation", "test"))
  removed_fraction = 1 - (n_output_proteins / n_input_proteins)
  assert removed_fraction < 0.05, (
      f"{removed_fraction:.1%} of proteins were dropped (too-long sequence or no labels) -- "
      "expected close to 0% for a no-threshold vocabulary, investigate before trusting this dataset")
  print(f"[OK] test_no_labels_removed_silently ({removed_fraction:.2%} dropped)")


def test_dependency_graph_selection_rule():
  """export_dependency_graph's own docstring warns that a wrong/default selection_rule
  would silently mislabel the exported graph's metadata for anyone reading it later --
  verify main() actually passed this variant's real rule, not a copy-pasted wrong one.
  """
  import json
  payload = json.loads(
      open(f"{DATA_PATH}/dataset_full/go_dependency_graph.json").read())
  rule = payload["coarse_selection_rule"]
  assert rule == {"min_annotations": None, "propagated": True, "depth_cap": None}, (
      f"go_dependency_graph.json's selection rule is {rule}, expected the full-label-space "
      "rule (min_annotations=None, propagated=True, depth_cap=None) -- check main()'s "
      "export_dependency_graph call")
  print("[OK] test_dependency_graph_selection_rule")


if __name__ == "__main__":
  test_split_matches_existing_variants()
  test_vocabulary_larger_than_existing_variants()
  test_no_labels_removed_silently()
  test_dependency_graph_selection_rule()
  print("[OK] all integrity checks passed")
```

Save as `experiments/scripts/verify_dataset_full_integrity.py`.

Run: `env/bin/python experiments/scripts/verify_dataset_full_integrity.py`
Expected: all three `[OK]` lines, ending with `[OK] all integrity checks passed`. If
`test_split_matches_existing_variants` fails, do not proceed to Task 3 — fix the split logic
first, since every later comparison against existing baselines depends on it.

- [ ] **Step 4: Commit**

```bash
git add generate_dataset_full.py experiments/scripts/verify_dataset_full_integrity.py
git commit -m "feat: build the full ancestor-closure dataset, verify split/vocab integrity"
```

---

## Task 3: Push to Hub and verify the remote round-trip

**Files:**
- Modify: `generate_dataset_full.py` (add the `push_to_hub` call)

**Interfaces:**
- Produces: Hub dataset repo `kelmoir/cafa6-full`, loadable via
  `datasets.load_dataset("kelmoir/cafa6-full")` — this is the `--dataset-repo-id` Task 4's
  sweep trials fall back to if no local `dataset_full` path is passed (mirrors
  `training_run.py`'s existing local-path-first-then-Hub-fallback behavior).

- [ ] **Step 1: Add the Hub push to `main()`**

```python
  dataset_dict.push_to_hub("kelmoir/cafa6-full", private=False)
```

Add this line immediately after the `export_dependency_graph(...)` call inside `main()`
(before the final `print("\nFull-label-space dataset created successfully...")`, and update
that print to drop the "not yet pushed to Hub" caveat).

- [ ] **Step 2: Re-run only the push (dataset already built locally in Task 2)**

Since Task 2 already built and saved `dataset_full` to disk, re-running the whole
`generate_dataset_full.py` script would redo all the CPU-heavy propagation work
unnecessarily just to push. Instead, run this one-off push directly:

Run:
```bash
env/bin/python -c "
import datasets
d = datasets.load_from_disk('data/cafa-6-protein-function-prediction/dataset_full')
d.push_to_hub('kelmoir/cafa6-full', private=False)
"
```

If this fails with `ImportError: cannot import name 'XetSession' from 'hf_xet'`, this is the
known env gotcha documented in CLAUDE.md — re-run with `HF_HUB_DISABLE_XET=1` set first.

- [ ] **Step 3: Verify the remote round-trip**

Run:
```bash
env/bin/python -c "
import datasets
d = datasets.load_dataset('kelmoir/cafa6-full')
print({split: len(d[split]) for split in d})
print(d['train'].features['labels'].feature.num_classes, 'terms')
"
```
Expected: split sizes matching Task 2's local run, and the same vocabulary size printed
there.

- [ ] **Step 4: Commit**

```bash
git add generate_dataset_full.py
git commit -m "feat: push the full-label-space dataset to kelmoir/cafa6-full"
```

---

## Task 4: Launch the plateau-first regularization sweep

**Files:**
- Create: `experiments/full_label_sweep/sweep_log.md`

**Interfaces:**
- Consumes: `data/cafa-6-protein-function-prediction/dataset_full` (Task 2) /
  `kelmoir/cafa6-full` (Task 3), `training_run.py` (unmodified, reused as-is).
- Produces: a first trial result (trial-001) logged in the new sweep log, and the sweep's
  ongoing rubric/next-config process — **this task's deliverable is getting the sweep
  started and its first trial judged, not finding the plateau itself.** Per
  [[feedback_plateau_first_sweep_strategy]], locating the actual plateau takes multiple
  trials whose configs depend on each prior trial's outcome, so it cannot be enumerated as
  fixed steps up front the way Tasks 1-3 could — it continues as its own log-driven loop
  after this task, exactly how `experiments/ancestor_sweep/sweep_log.md` grew one trial at a
  time rather than from a fixed plan. This task ends once trial-001 is run and judged.

- [ ] **Step 1: Write the sweep log's rubric and search-strategy header**

```markdown
# Full-Label-Space Regularization Sweep Log

Goal: find ESM-2 fine-tuning settings that train reliably on the full ancestor-closure
GO-term label space (`data/cafa-6-protein-function-prediction/dataset_full` /
`kelmoir/cafa6-full`, see `generate_dataset_full.py`), producing the first genuinely
plateau-tuned baseline for `experiments/cafa6_next_steps_plan.md`'s Tier 1 work. Each trial
runs locally on the user's GPU ([[hardware_local_gpu]]) via `training_run.py
--data-path data/cafa-6-protein-function-prediction/dataset_full`, handed to the user as an
exact `!`-prefixed command per [[feedback_no_long_cpu_runs_prefer_gpu]] (the sandbox has no
GPU device passthrough).

## Rubric

Same DEAD / OVERFITTING / HEALTHY / UNSTABLE judgment as
`experiments/ancestor_sweep/sweep_log.md` (see that file for the full rubric text and
corrective actions per verdict) — reused verbatim, not redefined here.

**One difference from the coarse-space sweep, worth calling out explicitly:** this label
space is far more extreme in class imbalance (tens of thousands of mostly-singleton terms,
vs. 383 terms with a 50-annotation floor). `--run-diagnostic`'s pooled accuracy/recall at
threshold 0.5 will look even more misleading here than it did at trial-050 in the coarse
sweep (see CLAUDE.md's "Post-training diagnostic" note) — judge HEALTHY/DEAD/OVERFITTING
from per-label AUROC distribution (mean/median) and pooled AUROC, not pooled accuracy/recall,
from the first trial onward. Don't wait for a trial-050-style false alarm to learn this again.

## Search strategy

Same coordinate-descent approach as `experiments/ancestor_sweep/sweep_log.md`: start from
that sweep's own eventual HEALTHY-config defaults as trial-001's starting point (rather than
re-deriving from scratch) — `esm2_t12_35M_UR50D`, `unfreeze_last_n=2`, `lr=5e-5`,
`output_dropout=0.2`, `esm_dropout=0.0`, `attention_dropout=0.0`, `grad_clip_norm=1.0`,
`weight_decay=0.01` — since this is a new label space on the same backbone/task family, not
a reason to assume DEAD-basin defaults again. Change one axis per trial; priority order:
(1) confirm the model learns at all on this much larger/sparser label space (per-label AUROC
mean clearly above 0.5), (2) resolve over/underfitting, (3) scale dataset size/epochs only
once HEALTHY.

## Trial index

| id | model | unfreeze | lr | out_drop | esm_drop | attn_drop | grad_clip | wd | epochs | train_n | val_n | verdict | next_action |
|----|-------|----------|----|----|----|----|----|----|----|----|----|----|----|
| trial-001 | esm2_t12_35M_UR50D | 2 | 5e-5 | 0.2 | 0.0 | 0.0 | 1.0 | 0.01 | 3 | 16000 | 4000 | *pending* | *pending* |
```

- [ ] **Step 2: Hand the user trial-001's exact command**

Give the user this command to run themselves (local GPU, not sandbox):

```
! source env/bin/activate && python training_run.py --epochs 3 --batch-size 16 \
    --learning-rate 5e-5 --unfreeze-last-n 2 --max-train-samples 16000 \
    --max-val-samples 4000 --run-name full-label-trial001 \
    --data-path data/cafa-6-protein-function-prediction/dataset_full \
    --dataset-repo-id kelmoir/cafa6-full --run-diagnostic \
    --checkpoint-mode final 2>&1 | tee experiments/full_label_sweep/logs/trial-001.out
```

(Create `experiments/full_label_sweep/logs/` first if it doesn't exist —
`mkdir -p experiments/full_label_sweep/logs`.)

- [ ] **Step 3: Judge trial-001 against the rubric once the user pastes back results**

Read the per-epoch `train_loss`/`val_loss`/per-label AUROC lines from
`experiments/full_label_sweep/logs/trial-001.out` (or whatever output the user pastes back).
Apply the DEAD/OVERFITTING/HEALTHY/UNSTABLE rubric from Step 1, fill in trial-001's row in
the Trial index table (verdict + next_action), and add a `## trial-001` section below the
table with the full per-epoch metrics table and verdict reasoning — same format as
`experiments/ancestor_sweep/sweep_log.md`'s own trial sections.

- [ ] **Step 4: Commit**

```bash
git add experiments/full_label_sweep/sweep_log.md
git commit -m "feat: launch the full-label-space regularization sweep, trial-001"
```

**After this task:** continue the sweep as its own ongoing process (one trial at a time,
each config chosen from the prior trial's verdict, per [[feedback_plateau_first_sweep_strategy]])
until a HEALTHY config is confirmed at target scale — this is explicitly out of this plan's
fixed task list, the same way `experiments/ancestor_sweep/sweep_log.md`'s 13+ trials were
never enumerated in a plan up front.
