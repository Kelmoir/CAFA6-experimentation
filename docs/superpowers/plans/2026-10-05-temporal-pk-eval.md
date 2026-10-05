# Temporal PK/LK/NK Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a genuinely temporal PK/LK/NK validation set by diffing CAFA-6's own
`train_terms.tsv` against the already-cached current Swiss-Prot dump, and score the
current trained checkpoint against it with the real CAFA IA-weighted Fmax metric, broken
down by aspect and category — cross-checked once against the real `cafaeval` package.

**Architecture:** One shared helper module (`pk_eval_common.py`) plus three scripts under
`experiments/pk_eval/scripts/`: `build_temporal_pk_eval.py` (the t0/t1 diff → labeled
dataset), `score_temporal_pk_eval.py` (checkpoint inference → Fmax breakdown), and
`crosscheck_cafaeval.py` (one-time validation of the hand-rolled Fmax math against the
real `cafaeval` 1.3.0 CLI).

**Tech Stack:** Python, Biopython (`Bio.SwissProt.parse`, `Bio.SeqIO`), `datasets`,
`networkx`/`obonet` (via `generate_dataset.py`'s existing graph helpers, imported not
duplicated), `pandas`, `numpy`, the real `cafaeval` PyPI package (installed and its CLI
verified directly during planning — see below) for the one-time cross-check only.

**Spec:** `docs/superpowers/specs/2026-10-05-temporal-pk-eval-design.md`

## Global Constraints

- Swiss-Prot (reviewed) only — no TrEMBL. Source file:
  `data/uniprot/uniprot_sprot.dat.gz` (cached 2026-08-01, **not re-fetched**).
- t0 source: `data/cafa-6-protein-function-prediction/Train/train_terms.tsv` (dated
  October 2025, no evidence column — already pre-filtered to experimental by Kaggle).
- Experimental evidence allowlist (t1 only):
  `{EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP}`.
- Label vocabulary: reuse `data/cafa-6-protein-function-prediction/dataset_full`'s
  existing `features["labels"].feature.names` (27,191 terms) verbatim — never invent a
  new vocabulary.
- Primary-accession-only matching (`record.accessions[0]`) — no secondary/historical
  accession matching.
- Seed `42` everywhere a random choice is made (repo-wide convention).
- `cafaeval` (verified: PyPI package `cafaeval==1.3.0`, CLI is
  `python -m cafaeval <obo_file> <pred_dir> <gt_file> -ia <ia_file> -out_dir <dir>`; its
  `ground_truth.tsv` format is 2-column `protein_id\tterm`, **no** aspect column; it has
  **no** known-terms-exclusion flag — known-term stripping must happen in our own script
  before calling it) is a one-time dependency for the cross-check script only; do not add
  it to `requirements-minimal.txt`.

## Review Focus

- A Swiss-Prot record with zero experimental GO annotations at all (common — many
  records are 100% electronic/IEA, e.g. the dump's very first record, Q6GZX4) must be
  skipped cleanly, not crash or get miscategorized.
- A protein in `train_terms.tsv` (has t0 annotations) with **no** matching Swiss-Prot
  record at all (removed/renamed/never-reviewed) must be excluded entirely from both
  pools — not miscounted as "unseen" just because the streaming loop never visits it.
- A GO term from `train_terms.tsv` or a Swiss-Prot `DR GO` line that is obsolete/absent
  from `go-basic.obo` (not in the true-path graph) must not crash ancestor propagation —
  treat it as having no ancestors, same documented edge case `generate_dataset_full.py`
  already handles.
- The 50/50 balancer must not crash or silently produce a degenerate dataset when one
  pool is empty or far smaller than expected — raise a clear error instead.
- A propagated novel/known term that falls outside `dataset_full`'s 27,191-term trainable
  vocabulary must be dropped (same `if label in label2id` pattern `generate_dataset.py`
  already uses) **before** categorization is decided, not after — otherwise a protein can
  end up tagged PK/LK/NK with an empty `novel_labels` once vocab-filtered, which is
  inconsistent and must instead cause that (protein, aspect) to be skipped entirely.

---

## Task 1: Shared helpers (`pk_eval_common.py`)

**Files:**
- Create: `experiments/pk_eval/scripts/pk_eval_common.py`

**Interfaces:**
- Produces: `EXPERIMENTAL_EVIDENCE_CODES: set[str]`,
  `extract_experimental_go_terms(record) -> set[str]`,
  `build_term_to_aspect(obonet_graph) -> dict[str, str]`,
  `propagate_terms(terms: set[str], true_path_graph, ancestor_cache: dict) -> set[str]`,
  `filter_to_vocab(terms: set[str], label2id: dict) -> set[str]`,
  `split_by_aspect(terms: set[str], term_to_aspect: dict[str, str]) -> dict[str, set[str]]`,
  `compute_ia_weighted_fmax(probs, known, novel, ia_weights, thresholds) -> tuple[float, float, float, float]`
  (returns `(fmax, best_threshold, precision_at_best, recall_at_best)`).

- [ ] **Step 1: Write the failing test**

Create `experiments/pk_eval/scripts/pk_eval_common.py` with just the imports and a
`if __name__ == "__main__": _self_test()` stub, then write `_self_test()` first (TDD:
the test names functions that don't exist yet):

```python
def _self_test():
    import gzip
    import numpy as np
    from Bio import SwissProt
    import obonet

    # --- extract_experimental_go_terms, against a real fixture record ---
    # P0DO85 is the first record in the cached dump with a mix of IEA (should be
    # dropped) and IDA (should be kept) GO cross-references - verified directly by
    # streaming the real file during planning.
    target = None
    with gzip.open("data/uniprot/uniprot_sprot.dat.gz", "rt") as fh:
        for record in SwissProt.parse(fh):
            if record.accessions[0] == "P0DO85":
                target = record
                break
    assert target is not None, "fixture record P0DO85 not found in cached dump"
    got = extract_experimental_go_terms(target)
    assert got == {"GO:0004497", "GO:0009821"}, got

    # --- build_term_to_aspect / propagate_terms, against real go-basic.obo ---
    obonet_graph = obonet.read_obo(
        "data/cafa-6-protein-function-prediction/Train/go-basic.obo")
    from generate_dataset import restrict_to_true_path
    true_path_graph = restrict_to_true_path(obonet_graph)
    term_to_aspect = build_term_to_aspect(obonet_graph)
    assert term_to_aspect["GO:0004497"] == "F"
    assert term_to_aspect["GO:0009821"] == "P"

    closure = propagate_terms({"GO:0004497"}, true_path_graph, {})
    assert "GO:0016491" in closure, "expected immediate is_a parent in closure"
    assert "GO:0004497" in closure

    # Obsolete/graph-absent term ID (not in go-basic.obo at all) must not crash - same
    # documented edge case generate_dataset_full.py's select_full_terms already handles.
    obsolete_closure = propagate_terms({"GO:9999999"}, true_path_graph, {})
    assert obsolete_closure == {"GO:9999999"}, obsolete_closure

    # --- filter_to_vocab ---
    label2id = {"GO:0004497": 0, "GO:0016491": 1}
    assert filter_to_vocab({"GO:0004497", "GO:9999999"}, label2id) == {"GO:0004497"}

    # --- split_by_aspect ---
    grouped = split_by_aspect({"GO:0004497", "GO:0009821"}, term_to_aspect)
    assert grouped == {"C": set(), "F": {"GO:0004497"}, "P": {"GO:0009821"}}

    # --- compute_ia_weighted_fmax, hand-verified toy case ---
    # protein 1: known={0}, novel={1}, probs=[0.9,0.8,0.1] -> pred@0.5 after stripping
    #   known = {1}; pr=1.0, rc=1.0 (uniform IA weights)
    # protein 2: known={}, novel={0,2}, probs=[0.6,0.6,0.4] -> pred@0.5={0,1};
    #   tp={0}; pr=1/2=0.5, rc=1/2=0.5
    # macro average -> wpr=0.75, wrc=0.75, f=0.75
    probs = np.array([[0.9, 0.8, 0.1], [0.6, 0.6, 0.4]])
    known = np.array([[1, 0, 0], [0, 0, 0]], dtype=bool)
    novel = np.array([[0, 1, 0], [1, 0, 1]], dtype=bool)
    ia_weights = np.array([1.0, 1.0, 1.0])
    fmax, best_t, pr, rc = compute_ia_weighted_fmax(
        probs, known, novel, ia_weights, thresholds=np.array([0.5]))
    assert abs(fmax - 0.75) < 1e-9, fmax
    assert best_t == 0.5
    assert abs(pr - 0.75) < 1e-9 and abs(rc - 0.75) < 1e-9

    print("[OK] pk_eval_common self-test passed")


if __name__ == "__main__":
    _self_test()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/pk_eval/scripts/pk_eval_common.py`
Expected: `NameError: name 'extract_experimental_go_terms' is not defined` (or similar, for
whichever function is referenced first).

- [ ] **Step 3: Write minimal implementation**

Prepend these definitions above `_self_test()` in the same file:

```python
"""Shared helpers for the temporal PK/LK/NK evaluation harness (build + score + the
one-time cafaeval cross-check). See docs/superpowers/specs/2026-10-05-temporal-pk-eval-design.md.
"""
import numpy as np

# The standard CAFA experimental-evidence group. t0 (train_terms.tsv) has no evidence
# column at all (pre-filtered to experimental by Kaggle already); this allowlist only
# ever filters t1 (the Swiss-Prot dump's DR GO lines).
EXPERIMENTAL_EVIDENCE_CODES = {
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP",
}

_NAMESPACE_TO_ASPECT = {
    "biological_process": "P",
    "molecular_function": "F",
    "cellular_component": "C",
}


def extract_experimental_go_terms(record) -> set:
  """Flat set of GO term IDs from a Bio.SwissProt record's cross_references, kept only
  if their evidence code is in EXPERIMENTAL_EVIDENCE_CODES. Not aspect-split yet -
  propagation happens on the flat set first (see propagate_terms), aspect-splitting
  happens afterward (see split_by_aspect) so ancestor terms picked up by propagation get
  the same treatment as directly-asserted ones.
  """
  terms = set()
  for xref in record.cross_references:
    if xref[0] != "GO":
      continue
    evidence = xref[3].split(":")[0]
    if evidence in EXPERIMENTAL_EVIDENCE_CODES:
      terms.add(xref[1])
  return terms


def build_term_to_aspect(obonet_graph) -> dict:
  """Every GO term's aspect letter (C/F/P), from go-basic.obo's own 'namespace' node
  attribute - the same attribute export_dependency_graph() (generate_dataset.py) already
  reads. A term outside the three known namespaces (shouldn't happen in go-basic.obo) is
  simply absent from the returned dict.
  """
  term_to_aspect = {}
  for node, attrs in obonet_graph.nodes(data=True):
    namespace = attrs.get("namespace")
    if namespace in _NAMESPACE_TO_ASPECT:
      term_to_aspect[node] = _NAMESPACE_TO_ASPECT[namespace]
  return term_to_aspect


def propagate_terms(terms: set, true_path_graph, ancestor_cache: dict) -> set:
  """True-path ancestor closure for one protein's term set - mirrors
  generate_dataset.propagate_protein_terms()'s inner loop exactly, but applied to a
  single term set directly rather than batched over a DataFrame, since this is called
  once per streamed Swiss-Prot record rather than once for the whole corpus at once.
  ancestor_cache is a plain dict the caller owns and reuses across many calls (same
  memoization strategy propagate_protein_terms uses internally), since nx.descendants is
  not free and most terms repeat across records.
  """
  import networkx as nx
  closure = set(terms)
  for term in terms:
    if term not in ancestor_cache:
      ancestor_cache[term] = nx.descendants(true_path_graph, term) if term in true_path_graph else set()
    closure |= ancestor_cache[term]
  return closure


def filter_to_vocab(terms: set, label2id: dict) -> set:
  """Drop any term absent from the trainable label vocabulary - same
  'if label in label2id' pattern generate_dataset.py's own generate_dataset() uses.
  Must run BEFORE categorization is decided (see Review Focus in the plan): a term that
  gets dropped here must not have already caused a PK/LK/NK category assignment based on
  its presence.
  """
  return {term for term in terms if term in label2id}


def split_by_aspect(terms: set, term_to_aspect: dict) -> dict:
  """Group a flat term set into {"C": set(), "F": set(), "P": set()}. A term absent from
  term_to_aspect (shouldn't happen for anything already filtered to the trainable
  vocabulary, which is itself derived from go-basic.obo) is silently skipped.
  """
  grouped = {"C": set(), "F": set(), "P": set()}
  for term in terms:
    aspect = term_to_aspect.get(term)
    if aspect is not None:
      grouped[aspect].add(term)
  return grouped


def compute_ia_weighted_fmax(probs, known, novel, ia_weights, thresholds):
  """CAFA's IA-weighted Fmax, macro-averaged over proteins (the 'cafa'-normalization
  convention: precision averaged only over proteins with >=1 prediction at a given
  threshold after known terms are stripped out, recall averaged over all proteins with
  >=1 novel/ground-truth term). probs/known/novel: (n_proteins, n_labels) arrays (known,
  novel boolean or 0/1; probs float in [0,1]). ia_weights: (n_labels,) float array.
  thresholds: 1-D array of tau values to sweep. Returns (fmax, best_threshold,
  precision_at_best, recall_at_best).
  """
  known = known.astype(bool)
  novel_mask = novel.astype(bool)
  has_novel = novel_mask.any(axis=1)
  novel_ia_total = (novel_mask * ia_weights).sum(axis=1)

  best_f, best_t, best_pr, best_rc = 0.0, float(thresholds[0]), 0.0, 0.0
  for t in thresholds:
    pred = (probs >= t) & ~known
    pred_ia_total = (pred * ia_weights).sum(axis=1)
    has_pred = pred_ia_total > 0
    tp_ia_total = (pred & novel_mask).astype(float)
    tp_ia_total = (tp_ia_total * ia_weights).sum(axis=1)

    wpr = float((tp_ia_total[has_pred] / pred_ia_total[has_pred]).mean()) if has_pred.any() else 0.0
    wrc = float((tp_ia_total[has_novel] / novel_ia_total[has_novel]).mean()) if has_novel.any() else 0.0
    f = 2 * wpr * wrc / (wpr + wrc) if (wpr + wrc) > 0 else 0.0
    if f > best_f:
      best_f, best_t, best_pr, best_rc = f, float(t), wpr, wrc
  return best_f, best_t, best_pr, best_rc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/pk_eval/scripts/pk_eval_common.py`
Expected: `[OK] pk_eval_common self-test passed`

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/pk_eval_common.py
git commit -m "feat: add shared helpers for temporal PK/LK/NK eval harness"
```

---

## Task 2: t0 index — known terms per protein per aspect

**Files:**
- Create: `experiments/pk_eval/scripts/build_temporal_pk_eval.py`

**Interfaces:**
- Consumes (from Task 1, `pk_eval_common.py`): `propagate_terms`, `filter_to_vocab`,
  `split_by_aspect`.
- Consumes (from `generate_dataset.py`, imported not duplicated): `restrict_to_true_path`,
  `propagate_protein_terms`.
- Produces: `build_t0_index(train_terms_path, true_path_graph, label2id, term_to_aspect) -> dict[str, dict[str, set]]`
  (protein_id -> {"C": set, "F": set, "P": set} of known/t0 terms, vocab-filtered).

- [ ] **Step 1: Write the failing test**

Create `experiments/pk_eval/scripts/build_temporal_pk_eval.py` with:

```python
"""Build the temporal PK/LK/NK validation set by diffing CAFA-6's train_terms.tsv (t0)
against the already-cached current Swiss-Prot dump (t1). See
docs/superpowers/specs/2026-10-05-temporal-pk-eval-design.md for the full design.
"""
from pathlib import Path

import obonet
import pandas as pd

from generate_dataset import propagate_protein_terms, restrict_to_true_path
from pk_eval_common import filter_to_vocab, split_by_aspect

DATA_ROOT = Path("data/cafa-6-protein-function-prediction")
TRAIN_TERMS_PATH = DATA_ROOT / "Train/train_terms.tsv"
GO_OBO_PATH = DATA_ROOT / "Train/go-basic.obo"


def build_t0_index(train_terms_path, true_path_graph, label2id, term_to_aspect) -> dict:
  """For every protein in train_terms.tsv: propagate its asserted terms to the full
  true-path closure (reusing generate_dataset.propagate_protein_terms - this is the
  batch, whole-corpus case it was designed for), filter to the trainable vocabulary, then
  split into per-aspect known-term buckets.
  """
  raise NotImplementedError


def _self_test():
  obonet_graph = obonet.read_obo(GO_OBO_PATH)
  true_path_graph = restrict_to_true_path(obonet_graph)
  from pk_eval_common import build_term_to_aspect
  term_to_aspect = build_term_to_aspect(obonet_graph)

  import datasets
  dataset_full = datasets.load_from_disk(str(DATA_ROOT / "dataset_full"))
  label_names = dataset_full["train"].features["labels"].feature.names
  label2id = {name: i for i, name in enumerate(label_names)}

  t0_index = build_t0_index(TRAIN_TERMS_PATH, true_path_graph, label2id, term_to_aspect)

  train_df = pd.read_csv(TRAIN_TERMS_PATH, sep="\t")
  n_unique_proteins = train_df["EntryID"].nunique()
  assert len(t0_index) == n_unique_proteins, (len(t0_index), n_unique_proteins)

  # Q5W0B1 is train_terms.tsv's first protein (GO:0000785 C, GO:0004842 F, GO:0051865 P,
  # GO:0006275 P per the raw file) - expect all three aspects non-empty after propagation.
  q5 = t0_index["Q5W0B1"]
  assert q5["C"], "expected non-empty C bucket for Q5W0B1"
  assert q5["F"], "expected non-empty F bucket for Q5W0B1"
  assert q5["P"], "expected non-empty P bucket for Q5W0B1"

  print(f"[OK] build_t0_index self-test passed ({len(t0_index)} proteins)")


if __name__ == "__main__":
  _self_test()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

Replace the `raise NotImplementedError` body:

```python
def build_t0_index(train_terms_path, true_path_graph, label2id, term_to_aspect) -> dict:
  train_df = pd.read_csv(train_terms_path, sep="\t")
  propagated = propagate_protein_terms(train_df, true_path_graph)
  t0_index = {}
  for protein_id, terms in propagated.items():
    vocab_terms = filter_to_vocab(terms, label2id)
    t0_index[protein_id] = split_by_aspect(vocab_terms, term_to_aspect)
  return t0_index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: `[OK] build_t0_index self-test passed (<N> proteins)`

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/build_temporal_pk_eval.py
git commit -m "feat: add t0 known-terms index for temporal PK/LK/NK build"
```

---

## Task 3: Per-protein categorization logic

**Files:**
- Modify: `experiments/pk_eval/scripts/build_temporal_pk_eval.py`

**Interfaces:**
- Produces: `categorize_protein(known_by_aspect: dict[str, set], t1_novel_by_aspect: dict[str, set]) -> dict[str, str]`
  (returns `{"C": category_or_None, "F": ..., "P": ...}`, category in `{"PK", "LK", "NK"}`).

- [ ] **Step 1: Write the failing test**

Add to `build_temporal_pk_eval.py` (above `_self_test`):

```python
def categorize_protein(known_by_aspect: dict, t1_novel_by_aspect: dict) -> dict:
  raise NotImplementedError
```

Extend `_self_test()`:

```python
  # --- categorize_protein: all four outcomes, hand-constructed ---
  has_any_known = {"C": {"GO:x"}, "F": set(), "P": set()}
  result = categorize_protein(has_any_known, {"C": set(), "F": {"GO:novel1"}, "P": set()})
  assert result["C"] is None, result  # no novel term in C -> no example
  assert result["F"] == "LK", result  # empty known_F, but known_C is non-empty -> LK
  assert result["P"] is None, result

  no_known_at_all = {"C": set(), "F": set(), "P": set()}
  result2 = categorize_protein(no_known_at_all, {"C": {"GO:novel2"}, "F": set(), "P": set()})
  assert result2["C"] == "NK", result2

  known_in_same_aspect = {"C": set(), "F": {"GO:known1"}, "P": set()}
  result3 = categorize_protein(known_in_same_aspect, {"C": set(), "F": {"GO:novel3"}, "P": set()})
  assert result3["F"] == "PK", result3

  print("[OK] categorize_protein self-test passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: `NotImplementedError` from the new call site (the `_self_test` runs
`build_t0_index`'s block first - leave that assertion in place, this new block runs
right after it in the same function).

- [ ] **Step 3: Write minimal implementation**

```python
def categorize_protein(known_by_aspect: dict, t1_novel_by_aspect: dict) -> dict:
  has_any_t0_annotation = any(known_by_aspect[a] for a in ("C", "F", "P"))
  result = {}
  for aspect in ("C", "F", "P"):
    if not t1_novel_by_aspect.get(aspect):
      result[aspect] = None
      continue
    if known_by_aspect[aspect]:
      result[aspect] = "PK"
    elif has_any_t0_annotation:
      result[aspect] = "LK"
    else:
      result[aspect] = "NK"
  return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: both `[OK]` lines print.

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/build_temporal_pk_eval.py
git commit -m "feat: add per-aspect PK/LK/NK categorization logic"
```

---

## Task 4: Streaming t0/t1 diff over the real Swiss-Prot dump

**Files:**
- Modify: `experiments/pk_eval/scripts/build_temporal_pk_eval.py`

**Interfaces:**
- Consumes: `build_t0_index` (Task 2), `categorize_protein` (Task 3),
  `pk_eval_common.extract_experimental_go_terms`, `propagate_terms`, `filter_to_vocab`,
  `split_by_aspect`.
- Produces: `stream_diff(dump_path, t0_index, true_path_graph, label2id, term_to_aspect, record_limit=None) -> tuple[list[dict], list[dict]]`
  (returns `(added_annotation_rows, unseen_protein_rows)`; each row is a dict with keys
  `protein_id`, `sequence`, `known_by_aspect`, `novel_by_aspect`, `category_by_aspect`,
  `pool`).

- [ ] **Step 1: Write the failing test**

Add to `build_temporal_pk_eval.py`:

```python
SWISSPROT_DUMP_PATH = Path("data/uniprot/uniprot_sprot.dat.gz")


def stream_diff(dump_path, t0_index, true_path_graph, label2id, term_to_aspect,
                record_limit=None) -> tuple:
  raise NotImplementedError
```

Extend `_self_test()` with a size-limited smoke run (the real dump is ~700MB; limiting to
the first 3000 records keeps this fast while still real data, not mocked):

```python
  added_rows, unseen_rows = stream_diff(
      SWISSPROT_DUMP_PATH, t0_index, true_path_graph, label2id, term_to_aspect,
      record_limit=3000)
  assert len(added_rows) > 0, "expected at least one added_annotation row in first 3000 records"
  assert len(unseen_rows) > 0, "expected at least one unseen_protein row in first 3000 records"
  valid_categories = {"PK", "LK", "NK", None}
  for row in added_rows:
    assert set(row["category_by_aspect"].values()) <= valid_categories
    assert row["pool"] == "added_annotation"
    assert any(row["novel_by_aspect"].values()), "added row must have >=1 novel term somewhere"
  for row in unseen_rows:
    assert row["pool"] == "unseen_protein"
    assert all(not s for s in row["known_by_aspect"].values()), "unseen rows must have empty known"
    assert any(row["novel_by_aspect"].values())
  print(f"[OK] stream_diff smoke test passed ({len(added_rows)} added, {len(unseen_rows)} unseen, first 3000 records)")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

```python
def stream_diff(dump_path, t0_index, true_path_graph, label2id, term_to_aspect,
                record_limit=None) -> tuple:
  import gzip
  import itertools

  from Bio import SwissProt

  from pk_eval_common import extract_experimental_go_terms, propagate_terms

  added_rows, unseen_rows = [], []
  ancestor_cache = {}

  with gzip.open(dump_path, "rt") as fh:
    records = SwissProt.parse(fh)
    if record_limit is not None:
      records = itertools.islice(records, record_limit)
    for record in records:
      accession = record.accessions[0]
      raw_terms = extract_experimental_go_terms(record)
      if not raw_terms:
        continue  # e.g. Q6GZX4: 100% electronic annotation, nothing to diff against
      propagated = propagate_terms(raw_terms, true_path_graph, ancestor_cache)
      vocab_terms = filter_to_vocab(propagated, label2id)
      novel_by_aspect = split_by_aspect(vocab_terms, term_to_aspect)
      if not any(novel_by_aspect.values()):
        continue  # every term dropped by vocab filtering -> no scoreable example

      if accession in t0_index:
        known_by_aspect = t0_index[accession]
        category_by_aspect = categorize_protein(known_by_aspect, novel_by_aspect)
        if not any(category_by_aspect.values()):
          continue  # no aspect actually gained a novel term -> no example
        added_rows.append({
            "protein_id": accession,
            "sequence": record.sequence,
            "known_by_aspect": known_by_aspect,
            "novel_by_aspect": novel_by_aspect,
            "category_by_aspect": category_by_aspect,
            "pool": "added_annotation",
        })
      else:
        unseen_rows.append({
            "protein_id": accession,
            "sequence": record.sequence,
            "known_by_aspect": {"C": set(), "F": set(), "P": set()},
            "novel_by_aspect": novel_by_aspect,
            "category_by_aspect": {a: ("NK" if novel_by_aspect[a] else None) for a in ("C", "F", "P")},
            "pool": "unseen_protein",
        })

  return added_rows, unseen_rows
```

Note: `categorize_protein` is defined earlier in the same module (Task 3) - no new
import needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: all three `[OK]` lines print (t0 index, categorize_protein, stream_diff smoke).

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/build_temporal_pk_eval.py
git commit -m "feat: add streaming t0/t1 diff over the cached Swiss-Prot dump"
```

---

## Task 5: 50/50 balancing + category summary

**Files:**
- Modify: `experiments/pk_eval/scripts/build_temporal_pk_eval.py`

**Interfaces:**
- Produces: `balance_pools(added_rows: list, unseen_rows: list, seed: int = 42) -> tuple[list, list]`,
  `summarize_categories(added_rows: list, unseen_rows: list) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
def balance_pools(added_rows: list, unseen_rows: list, seed: int = 42) -> tuple:
  raise NotImplementedError


def summarize_categories(added_rows: list, unseen_rows: list) -> dict:
  raise NotImplementedError
```

Extend `_self_test()`:

```python
  # --- balance_pools: synthetic unequal pools ---
  fake_added = [{"protein_id": f"A{i}"} for i in range(7)]
  fake_unseen = [{"protein_id": f"U{i}"} for i in range(3)]
  balanced_added, balanced_unseen = balance_pools(fake_added, fake_unseen, seed=42)
  assert len(balanced_added) == 3 and len(balanced_unseen) == 3
  balanced_added_2, _ = balance_pools(fake_added, fake_unseen, seed=42)
  assert [r["protein_id"] for r in balanced_added] == [r["protein_id"] for r in balanced_added_2], (
      "same seed must reproduce the same subsample")

  empty_unseen_should_fail = False
  try:
    balance_pools(fake_added, [], seed=42)
  except ValueError:
    empty_unseen_should_fail = True
  assert empty_unseen_should_fail, "balancing against an empty pool must raise, not silently produce 0 rows"

  summary = summarize_categories(added_rows, unseen_rows)
  assert summary["unseen_protein"] == len(unseen_rows)
  assert set(summary["added_annotation"].keys()) <= {"C", "F", "P"}
  print("[OK] balance_pools/summarize_categories self-test passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

```python
def balance_pools(added_rows: list, unseen_rows: list, seed: int = 42) -> tuple:
  import random
  if not added_rows or not unseen_rows:
    raise ValueError(
        f"cannot balance: added_annotation={len(added_rows)}, unseen_protein={len(unseen_rows)} "
        "- both pools must be non-empty")
  n = min(len(added_rows), len(unseen_rows))
  rng = random.Random(seed)
  balanced_added = rng.sample(added_rows, n) if len(added_rows) > n else list(added_rows)
  rng2 = random.Random(seed)
  balanced_unseen = rng2.sample(unseen_rows, n) if len(unseen_rows) > n else list(unseen_rows)
  return balanced_added, balanced_unseen


def summarize_categories(added_rows: list, unseen_rows: list) -> dict:
  summary = {"added_annotation": {"C": {}, "F": {}, "P": {}}, "unseen_protein": len(unseen_rows)}
  for row in added_rows:
    for aspect in ("C", "F", "P"):
      category = row["category_by_aspect"][aspect]
      if category is not None:
        summary["added_annotation"][aspect][category] = (
            summary["added_annotation"][aspect].get(category, 0) + 1)
  return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: all `[OK]` lines print.

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/build_temporal_pk_eval.py
git commit -m "feat: add 50/50 pool balancing and category summary"
```

---

## Task 6: Output writers (HF Dataset + flat TSVs)

**Files:**
- Modify: `experiments/pk_eval/scripts/build_temporal_pk_eval.py`

**Interfaces:**
- Produces: `write_outputs(balanced_added: list, balanced_unseen: list, label_names: list, out_dir: Path) -> None`
  — writes `out_dir/temporal_pk_validation/` (HF `Dataset`),
  `out_dir/temporal_pk_validation_flat/ground_truth.tsv`,
  `out_dir/temporal_pk_validation_flat/known_terms.tsv`.

- [ ] **Step 1: Write the failing test**

```python
def write_outputs(balanced_added: list, balanced_unseen: list, label_names: list, out_dir: Path) -> None:
  raise NotImplementedError
```

Extend `_self_test()` (uses a scratch dir, not the real `experiments/pk_eval/` output -
that's Task 7's real run):

```python
  import shutil
  import tempfile

  import datasets as hf_datasets

  scratch_dir = Path(tempfile.mkdtemp())
  try:
    toy_added = [{
        "protein_id": "T0001", "sequence": "MKV",
        "known_by_aspect": {"C": set(), "F": {label_names[0]}, "P": set()},
        "novel_by_aspect": {"C": set(), "F": set(), "P": {label_names[1]}},
        "category_by_aspect": {"C": None, "F": None, "P": "LK"},
        "pool": "added_annotation",
    }]
    toy_unseen = [{
        "protein_id": "T0002", "sequence": "MKA",
        "known_by_aspect": {"C": set(), "F": set(), "P": set()},
        "novel_by_aspect": {"C": {label_names[2]}, "F": set(), "P": set()},
        "category_by_aspect": {"C": "NK", "F": None, "P": None},
        "pool": "unseen_protein",
    }]
    write_outputs(toy_added, toy_unseen, label_names, scratch_dir)

    ds = hf_datasets.load_from_disk(str(scratch_dir / "temporal_pk_validation"))
    assert len(ds) == 2
    assert set(ds[0].keys()) >= {"protein_id", "sequence", "known_labels", "novel_labels",
                                  "category_C", "category_F", "category_P", "pool"}

    gt_path = scratch_dir / "temporal_pk_validation_flat" / "ground_truth.tsv"
    known_path = scratch_dir / "temporal_pk_validation_flat" / "known_terms.tsv"
    gt_df = pd.read_csv(gt_path, sep="\t", header=None, names=["protein_id", "term"])
    known_df = pd.read_csv(known_path, sep="\t", header=None, names=["protein_id", "term"])
    assert set(gt_df["protein_id"]) == {"T0001", "T0002"}
    assert list(known_df["protein_id"]) == ["T0001"]  # only the added_annotation row has known terms
    print("[OK] write_outputs self-test passed")
  finally:
    shutil.rmtree(scratch_dir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

```python
def write_outputs(balanced_added: list, balanced_unseen: list, label_names: list, out_dir: Path) -> None:
  import datasets as hf_datasets

  all_rows = balanced_added + balanced_unseen
  label_feature = hf_datasets.Sequence(hf_datasets.ClassLabel(names=label_names))
  features = hf_datasets.Features({
      "protein_id": hf_datasets.Value("string"),
      "sequence": hf_datasets.Value("string"),
      "known_labels": label_feature,
      "novel_labels": label_feature,
      "category_C": hf_datasets.Value("string"),
      "category_F": hf_datasets.Value("string"),
      "category_P": hf_datasets.Value("string"),
      "pool": hf_datasets.Value("string"),
  })

  columns = {key: [] for key in features}
  gt_lines, known_lines = [], []
  for row in all_rows:
    columns["protein_id"].append(row["protein_id"])
    columns["sequence"].append(row["sequence"])
    known_sorted = sorted(row["known_by_aspect"]["C"] | row["known_by_aspect"]["F"] | row["known_by_aspect"]["P"])
    novel_sorted = sorted(row["novel_by_aspect"]["C"] | row["novel_by_aspect"]["F"] | row["novel_by_aspect"]["P"])
    columns["known_labels"].append(known_sorted)
    columns["novel_labels"].append(novel_sorted)
    for aspect in ("C", "F", "P"):
      columns[f"category_{aspect}"].append(row["category_by_aspect"][aspect] or "none")
    columns["pool"].append(row["pool"])
    for term in novel_sorted:
      gt_lines.append(f"{row['protein_id']}\t{term}")
    for term in known_sorted:
      known_lines.append(f"{row['protein_id']}\t{term}")

  dataset = hf_datasets.Dataset.from_dict(columns, features=features)
  dataset.save_to_disk(str(out_dir / "temporal_pk_validation"))

  flat_dir = out_dir / "temporal_pk_validation_flat"
  flat_dir.mkdir(parents=True, exist_ok=True)
  (flat_dir / "ground_truth.tsv").write_text("\n".join(gt_lines) + "\n" if gt_lines else "")
  (flat_dir / "known_terms.tsv").write_text("\n".join(known_lines) + "\n" if known_lines else "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: all `[OK]` lines print.

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/build_temporal_pk_eval.py
git commit -m "feat: add HF Dataset + flat TSV output writers"
```

---

## Task 7: Real end-to-end build run

**Files:**
- Modify: `experiments/pk_eval/scripts/build_temporal_pk_eval.py` (add `main()`)
- Modify: `.gitignore`
- New data (generated, not hand-written):
  `experiments/pk_eval/temporal_pk_validation/`,
  `experiments/pk_eval/temporal_pk_validation_flat/`,
  `experiments/pk_eval/category_summary.json`

- [ ] **Step 1: Write `main()`**

```python
def main():
  import json

  import datasets as hf_datasets

  print("[INFO] Loading go-basic.obo and building true-path graph + term->aspect map...")
  obonet_graph = obonet.read_obo(GO_OBO_PATH)
  true_path_graph = restrict_to_true_path(obonet_graph)
  from pk_eval_common import build_term_to_aspect
  term_to_aspect = build_term_to_aspect(obonet_graph)

  dataset_full = hf_datasets.load_from_disk(str(DATA_ROOT / "dataset_full"))
  label_names = dataset_full["train"].features["labels"].feature.names
  label2id = {name: i for i, name in enumerate(label_names)}

  print("[INFO] Building t0 (train_terms.tsv) known-terms index...")
  t0_index = build_t0_index(TRAIN_TERMS_PATH, true_path_graph, label2id, term_to_aspect)
  print(f"[INFO] t0 index covers {len(t0_index)} proteins")

  print("[INFO] Streaming the full Swiss-Prot dump (one-time, ~700MB, a few minutes)...")
  added_rows, unseen_rows = stream_diff(
      SWISSPROT_DUMP_PATH, t0_index, true_path_graph, label2id, term_to_aspect)
  print(f"[INFO] Raw yields: added_annotation={len(added_rows)}, unseen_protein={len(unseen_rows)}")

  # Visibility into Review Focus item 2: a meaningful fraction of t0 (train_terms.tsv)
  # proteins are expected to have no matching Swiss-Prot record at all (CAFA-6 also
  # draws from TrEMBL, which this harness deliberately excludes - see spec), and those
  # are silently absent from added_rows by construction (stream_diff only emits a row
  # for accessions it actually visits while streaming). This count is a lower bound on
  # "matched in the dump" (it excludes matched-but-no-novel-term proteins too), but
  # still the right cheap signal that the match rate isn't suspiciously low.
  t0_proteins_with_a_novel_term = {row["protein_id"] for row in added_rows}
  print(f"[INFO] t0 proteins with >=1 novel annotation by t1 (lower bound on matched): "
        f"{len(t0_proteins_with_a_novel_term)}/{len(t0_index)} "
        f"({len(t0_proteins_with_a_novel_term) / len(t0_index):.1%})")

  balanced_added, balanced_unseen = balance_pools(added_rows, unseen_rows, seed=42)
  summary = summarize_categories(balanced_added, balanced_unseen)
  print(f"[INFO] Balanced: added_annotation={len(balanced_added)}, unseen_protein={len(balanced_unseen)}")
  print(f"[RESULT] category_summary={json.dumps(summary)}")

  out_dir = Path("experiments/pk_eval")
  write_outputs(balanced_added, balanced_unseen, label_names, out_dir)
  (out_dir / "category_summary.json").write_text(json.dumps(summary, indent=2))
  print(f"[INFO] Wrote outputs under {out_dir}")


if __name__ == "__main__":
  import sys
  if "--self-test" in sys.argv:
    _self_test()
  else:
    main()
```

Change the self-test guard at the bottom of the file to match (replace the plain
`if __name__ == "__main__": _self_test()` from Task 1 with the block above).

- [ ] **Step 2: Re-run the self-test to confirm nothing broke**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py --self-test`
Expected: all `[OK]` lines still print.

- [ ] **Step 3: Add the new data directories to `.gitignore`**

Add, matching the existing `blast_work`/`pu_cache` convention (comment above the entry
explaining why — regeneratable from the cached Swiss-Prot dump + train_terms.tsv):

```gitignore

# Temporal PK/LK/NK eval harness output (experiments/pk_eval/scripts/build_temporal_pk_eval.py)
# -- regeneratable from the cached data/uniprot/uniprot_sprot.dat.gz + train_terms.tsv.
# category_summary.json is small and kept tracked, same as class_priors_km2.json above.
/experiments/pk_eval/temporal_pk_validation/*
/experiments/pk_eval/temporal_pk_validation_flat/*
```

- [ ] **Step 4: Run the real build**

Run: `env/bin/python experiments/pk_eval/scripts/build_temporal_pk_eval.py`
Expected: completes (a few minutes, streaming ~700MB once), ends with
`[INFO] Wrote outputs under experiments/pk_eval`, and prints a `[RESULT]
category_summary=...` line with non-zero counts in both pools.

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/build_temporal_pk_eval.py .gitignore experiments/pk_eval/category_summary.json
git commit -m "feat: run the real temporal PK/LK/NK build, add main() entrypoint"
```

---

## Task 8: Scoring harness (`score_temporal_pk_eval.py`)

**Files:**
- Create: `experiments/pk_eval/scripts/score_temporal_pk_eval.py`

**Interfaces:**
- Consumes: `pk_eval_common.build_term_to_aspect`, `compute_ia_weighted_fmax`; the real
  `experiments/pk_eval/temporal_pk_validation/` dataset (Task 7); `training_run.py`'s
  `create_esm`, `create_esm_tokenizer`, `preprocess_batch`, `data_collate_function` (all
  imported, not duplicated — same reuse pattern
  `experiments/ancestor_sweep/scripts/diagnostic_eval_stage2.py` already established).
- Produces: `densify_term_lists(encoded_lists: list, n_labels: int) -> np.ndarray`,
  `run_inference(model, tokenizer, dataset, device, batch_size=16) -> np.ndarray`
  (returns `(n_proteins, n_labels)` sigmoid probabilities),
  `score_breakdown(probs, known, novel, category_columns, ia_weights, thresholds) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
"""Score a checkpoint against the temporal PK/LK/NK validation set (Task 7's output),
breaking Fmax down by aspect x category. See
docs/superpowers/specs/2026-10-05-temporal-pk-eval-design.md.
"""
import json
from pathlib import Path

import numpy as np

from pk_eval_common import compute_ia_weighted_fmax

HF_REPO_ID = "kelmoir/cafa6-esm"
DEFAULT_REVISION = "main"
DATASET_FULL_PATH = Path("data/cafa-6-protein-function-prediction/dataset_full")
VALIDATION_PATH = Path("experiments/pk_eval/temporal_pk_validation")
IA_PATH = Path("data/cafa-6-protein-function-prediction/IA.tsv")


def densify_term_lists(encoded_lists: list, n_labels: int) -> np.ndarray:
  raise NotImplementedError


def score_breakdown(probs, known, novel, category_columns, ia_weights, thresholds) -> dict:
  raise NotImplementedError


def _self_test():
  # --- densify_term_lists ---
  dense = densify_term_lists([[0, 2], [1]], n_labels=3)
  assert dense.shape == (2, 3)
  assert dense.tolist() == [[1, 0, 1], [0, 1, 0]]

  # --- score_breakdown: 2 proteins, both category "PK" on aspect "F", using the same
  # hand-verified toy numbers as pk_eval_common's compute_ia_weighted_fmax test ---
  probs = np.array([[0.9, 0.8, 0.1], [0.6, 0.6, 0.4]])
  known = np.array([[1, 0, 0], [0, 0, 0]])
  novel = np.array([[0, 1, 0], [1, 0, 1]])
  ia_weights = np.array([1.0, 1.0, 1.0])
  category_columns = {"F": ["PK", "PK"]}  # both toy proteins tagged PK on aspect F
  result = score_breakdown(probs, known, novel, category_columns, ia_weights,
                           thresholds=np.array([0.5]))
  assert abs(result["F"]["PK"]["fmax"] - 0.75) < 1e-9, result

  print("[OK] score_temporal_pk_eval self-test passed")


if __name__ == "__main__":
  _self_test()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/pk_eval/scripts/score_temporal_pk_eval.py`
Expected: `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

```python
def densify_term_lists(encoded_lists: list, n_labels: int) -> np.ndarray:
  dense = np.zeros((len(encoded_lists), n_labels), dtype=bool)
  for row_idx, indices in enumerate(encoded_lists):
    for idx in indices:
      dense[row_idx, idx] = True
  return dense


def score_breakdown(probs, known, novel, category_columns, ia_weights, thresholds) -> dict:
  results = {}
  for aspect, categories in category_columns.items():
    categories = np.array(categories, dtype=object)
    results[aspect] = {}
    for category in set(categories) - {None}:
      row_mask = categories == category
      if not row_mask.any():
        continue
      fmax, best_t, pr, rc = compute_ia_weighted_fmax(
          probs[row_mask], known[row_mask], novel[row_mask], ia_weights, thresholds)
      results[aspect][category] = {
          "fmax": fmax, "best_threshold": best_t, "precision": pr, "recall": rc,
          "n_proteins": int(row_mask.sum()),
      }
  return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/pk_eval/scripts/score_temporal_pk_eval.py`
Expected: `[OK] score_temporal_pk_eval self-test passed`

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/score_temporal_pk_eval.py
git commit -m "feat: add densify/score_breakdown helpers for PK/LK/NK scoring"
```

---

## Task 9: Real checkpoint inference + full aspect x category report

**Files:**
- Modify: `experiments/pk_eval/scripts/score_temporal_pk_eval.py` (add `run_inference`,
  `main()`, argparse)
- Modify: `experiments/cafa6_next_steps_plan.md` (log the result)

**Interfaces:**
- Produces: `run_inference(model, tokenizer, dataset, device, batch_size=16) -> np.ndarray`.

- [ ] **Step 1: Write `run_inference` and `main()`**

```python
def run_inference(model, tokenizer, dataset, device, batch_size: int = 16) -> np.ndarray:
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
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
      all_probs.append(torch.sigmoid(logits.float()).cpu())
  if was_training:
    model.train()
  return torch.cat(all_probs).numpy()


def main():
  import argparse

  import datasets as hf_datasets
  import pandas as pd
  import torch
  from huggingface_hub import hf_hub_download

  import training_run
  from pk_eval_common import build_term_to_aspect

  parser = argparse.ArgumentParser()
  parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
  parser.add_argument("--revision", default=DEFAULT_REVISION)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--th-step", type=float, default=0.01)
  args = parser.parse_args()

  dataset_full = hf_datasets.load_from_disk(str(DATASET_FULL_PATH))
  label_names = dataset_full["train"].features["labels"].feature.names
  label2id = {name: i for i, name in enumerate(label_names)}
  n_labels = len(label_names)
  id2label = {i: name for name, i in label2id.items()}

  ia_df = pd.read_csv(IA_PATH, sep="\t", header=None, names=["term", "weight"])
  ia_map = dict(zip(ia_df["term"], ia_df["weight"]))
  ia_weights = np.array([ia_map.get(name, 0.0) for name in label_names])

  # Same checkpoint-config-recovery pattern diagnostic_eval_stage2.py already
  # established: read trial_metrics.json rather than re-deriving model_name/use_lora by
  # hand, so this script works unmodified for a future LoRA checkpoint too.
  trial_metrics_path = hf_hub_download(
      repo_id=args.hf_repo_id, filename="trial_metrics.json", revision=args.revision)
  trial_config = json.loads(Path(trial_metrics_path).read_text())["config"]
  model_name = trial_config["model_name"]
  use_lora = trial_config.get("use_lora", False)

  device = torch.device(args.device)
  tokenizer = training_run.create_esm_tokenizer(model_name)
  if use_lora:
    model = training_run.create_esm(
        model_type=model_name, use_lora=True, lora_resume_repo_id=args.hf_repo_id,
        revision=args.revision, id2label=id2label, label2id=label2id,
    ).to(device)
  else:
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type=args.hf_repo_id, revision=args.revision,
        id2label=id2label, label2id=label2id,
    ).to(device)
  model.eval()
  print(f"[INFO] Loaded {args.hf_repo_id}@{args.revision} ({model_name}, use_lora={use_lora})")

  validation = hf_datasets.load_from_disk(str(VALIDATION_PATH))
  from functools import partial
  preprocess = partial(training_run.preprocess_batch, tokenizer=tokenizer,
                       unique_items=label_names, label2id=label2id)
  tokenized = validation.with_transform(transform=preprocess)

  probs = run_inference(model, tokenizer, tokenized, device, batch_size=args.batch_size)
  known = densify_term_lists(validation["known_labels"], n_labels)
  novel = densify_term_lists(validation["novel_labels"], n_labels)

  obonet_graph = __import__("obonet").read_obo(
      "data/cafa-6-protein-function-prediction/Train/go-basic.obo")
  term_to_aspect = build_term_to_aspect(obonet_graph)
  aspect_vector = np.array([term_to_aspect.get(name) for name in label_names])

  thresholds = np.arange(args.th_step, 1.0, args.th_step)
  is_unseen = np.array(list(validation["pool"])) == "unseen_protein"
  full_report = {}
  for aspect in ("C", "F", "P"):
    aspect_mask = aspect_vector == aspect
    raw_categories = list(validation[f"category_{aspect}"])
    # unseen_protein rows are stored with category "NK" in category_C/F/P (true - they
    # are NK-shaped: zero known in every aspect), but they are a distinct pool from
    # added_annotation's genuine "returning" NK rows (a protein the model trained on in
    # OTHER aspects, gaining a first annotation in this one) - conflating the two would
    # dilute exactly the distinction this harness exists to measure. Null out unseen
    # rows before the PK/LK/NK pass, then score them separately as their own "unseen"
    # bucket below.
    pk_lk_nk_categories = {aspect: [None if is_unseen[i] else c for i, c in enumerate(raw_categories)]}
    breakdown = score_breakdown(
        probs[:, aspect_mask], known[:, aspect_mask], novel[:, aspect_mask],
        pk_lk_nk_categories, ia_weights[aspect_mask], thresholds)

    unseen_only_categories = {aspect: ["unseen" if is_unseen[i] else None
                                       for i in range(len(raw_categories))]}
    unseen_breakdown = score_breakdown(
        probs[:, aspect_mask], known[:, aspect_mask], novel[:, aspect_mask],
        unseen_only_categories, ia_weights[aspect_mask], thresholds)
    full_report[aspect] = {**breakdown.get(aspect, {}), **unseen_breakdown.get(aspect, {})}

  print(f"[RESULT] {json.dumps(full_report, indent=2)}")
  return full_report


if __name__ == "__main__":
  import sys
  if "--self-test" in sys.argv:
    _self_test()
  else:
    main()
```

- [ ] **Step 2: Re-run the self-test to confirm nothing broke**

Run: `env/bin/python experiments/pk_eval/scripts/score_temporal_pk_eval.py --self-test`
Expected: `[OK] score_temporal_pk_eval self-test passed`

- [ ] **Step 3: Run the real scoring against trial-013**

Run: `env/bin/python experiments/pk_eval/scripts/score_temporal_pk_eval.py`
Expected: loads `kelmoir/cafa6-esm@main`, runs inference over the real validation set
built in Task 7, prints a `[RESULT]` JSON block with Fmax/precision/recall/n_proteins per
aspect x {PK, LK, NK, unseen} (whichever categories actually have rows — a thin category
pool, e.g. zero LK(C) rows, is expected and fine, just omitted from that aspect's dict).

- [ ] **Step 4: Log the result**

Append a new entry under Tier 1 item 2 in `experiments/cafa6_next_steps_plan.md`,
following the same pattern as other dated roadmap entries in that file (date, what was
run, the actual `[RESULT]` numbers, one-sentence interpretation).

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/score_temporal_pk_eval.py experiments/cafa6_next_steps_plan.md
git commit -m "feat: run real temporal PK/LK/NK scoring against trial-013, log result"
```

---

## Task 10: One-time `cafaeval` cross-check

**Files:**
- Create: `experiments/pk_eval/scripts/crosscheck_cafaeval.py`

**Interfaces:**
- Consumes: `experiments/pk_eval/temporal_pk_validation_flat/{ground_truth.tsv,known_terms.tsv}`
  (Task 7), the real per-protein probability matrix (re-derived by calling
  `score_temporal_pk_eval.py`'s `run_inference`/model-loading code, imported not
  duplicated), `pk_eval_common.compute_ia_weighted_fmax`.
- Produces: `write_prediction_tsv(probs, known, protein_ids, label_names, out_path, threshold_floor=0.01) -> None`,
  `main()` (runs the real `cafaeval` CLI via `subprocess`, parses
  `evaluation_best_f.tsv`, compares its `f` column to the hand-rolled aggregate Fmax).

- [ ] **Step 1: Write the failing test**

```python
"""One-time cross-check of compute_ia_weighted_fmax against the real cafaeval 1.3.0
package (installed via `pip install cafaeval` - verified directly during planning: its
CLI is `python -m cafaeval <obo> <pred_dir> <gt_file> -ia <ia_file> -out_dir <dir>`, its
ground_truth.tsv is 2-column protein_id/term with no known-exclusion flag, so known-term
stripping happens here before cafaeval ever sees the prediction file).
"""
from pathlib import Path

import numpy as np


def write_prediction_tsv(probs, known, protein_ids, label_names, out_path, threshold_floor=0.01) -> None:
  raise NotImplementedError


def _self_test():
  import tempfile

  probs = np.array([[0.9, 0.02], [0.6, 0.6]])
  known = np.array([[1, 0], [0, 0]], dtype=bool)
  protein_ids = ["T0001", "T0002"]
  label_names = ["GO:AAA", "GO:BBB"]
  with tempfile.TemporaryDirectory() as tmp:
    out_path = Path(tmp) / "pred.tsv"
    write_prediction_tsv(probs, known, protein_ids, label_names, out_path)
    lines = out_path.read_text().strip().split("\n")
    # T0001/GO:AAA has prob 0.9 but is known -> must be stripped, not written.
    assert not any(line.startswith("T0001\tGO:AAA") for line in lines), lines
    assert any(line.startswith("T0001\tGO:BBB") for line in lines), lines  # 0.02 >= floor
    assert any(line.startswith("T0002\tGO:AAA") for line in lines), lines
  print("[OK] crosscheck_cafaeval self-test passed")


if __name__ == "__main__":
  _self_test()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python experiments/pk_eval/scripts/crosscheck_cafaeval.py`
Expected: `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

```python
def write_prediction_tsv(probs, known, protein_ids, label_names, out_path, threshold_floor=0.01) -> None:
  lines = []
  for row_idx, protein_id in enumerate(protein_ids):
    for col_idx, term in enumerate(label_names):
      if known[row_idx, col_idx]:
        continue  # stripped, same as compute_ia_weighted_fmax's known-exclusion
      score = probs[row_idx, col_idx]
      if score >= threshold_floor:
        lines.append(f"{protein_id}\t{term}\t{score:.3f}")
  out_path.write_text("\n".join(lines) + "\n" if lines else "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python experiments/pk_eval/scripts/crosscheck_cafaeval.py`
Expected: `[OK] crosscheck_cafaeval self-test passed`

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/crosscheck_cafaeval.py
git commit -m "feat: add write_prediction_tsv for the cafaeval cross-check"
```

---

## Task 11: Run the real cafaeval cross-check

**Files:**
- Modify: `experiments/pk_eval/scripts/crosscheck_cafaeval.py` (add `main()`)
- Modify: `.gitignore`

- [ ] **Step 1: Install `cafaeval` (one-time, not a standing dependency)**

Run: `env/bin/pip install cafaeval`
Expected: `Successfully installed cafaeval-1.3.0` (already verified reachable from PyPI
during planning). Do **not** add this to `requirements-minimal.txt` — per the spec, this
is a one-time validation dependency, not part of the standing pipeline.

- [ ] **Step 2: Write `main()`**

```python
def main():
  import csv
  import subprocess

  import datasets as hf_datasets
  import numpy as np
  import pandas as pd

  import score_temporal_pk_eval as scorer

  dataset_full = hf_datasets.load_from_disk(str(scorer.DATASET_FULL_PATH))
  label_names = dataset_full["train"].features["labels"].feature.names
  n_labels = len(label_names)

  validation = hf_datasets.load_from_disk(str(scorer.VALIDATION_PATH))
  protein_ids = list(validation["protein_id"])
  known = scorer.densify_term_lists(validation["known_labels"], n_labels)
  novel = scorer.densify_term_lists(validation["novel_labels"], n_labels)

  # Model-loading below mirrors score_temporal_pk_eval.main()'s own checkpoint-loading
  # block verbatim (trial_metrics.json -> create_esm_tokenizer/create_esm) rather than
  # calling that main() directly, since this script only needs the resulting probs
  # matrix, not score_temporal_pk_eval's own aspect x category report.
  import json
  from pathlib import Path

  import torch
  from huggingface_hub import hf_hub_download

  import training_run

  trial_metrics_path = hf_hub_download(
      repo_id=scorer.HF_REPO_ID, filename="trial_metrics.json", revision=scorer.DEFAULT_REVISION)
  trial_config = json.loads(Path(trial_metrics_path).read_text())["config"]
  model_name = trial_config["model_name"]
  label2id = {name: i for i, name in enumerate(label_names)}
  id2label = {i: name for name, i in label2id.items()}
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = training_run.create_esm_tokenizer(model_name)
  model = training_run.create_esm(
      unfreeze_last_n=1, model_type=scorer.HF_REPO_ID, revision=scorer.DEFAULT_REVISION,
      id2label=id2label, label2id=label2id,
  ).to(device)
  model.eval()

  from functools import partial
  preprocess = partial(training_run.preprocess_batch, tokenizer=tokenizer,
                       unique_items=label_names, label2id=label2id)
  probs = scorer.run_inference(model, tokenizer, validation.with_transform(transform=preprocess), device)

  flat_dir = Path("experiments/pk_eval/temporal_pk_validation_flat")
  pred_dir = flat_dir / "predictions"
  pred_dir.mkdir(parents=True, exist_ok=True)
  write_prediction_tsv(probs, known, protein_ids, label_names, pred_dir / "model.tsv")

  cross_out_dir = Path("experiments/pk_eval/cafaeval_crosscheck")
  subprocess.run([
      "env/bin/python", "-m", "cafaeval",
      "data/cafa-6-protein-function-prediction/Train/go-basic.obo",
      str(pred_dir),
      str(flat_dir / "ground_truth.tsv"),
      "-ia", "data/cafa-6-protein-function-prediction/IA.tsv",
      "-out_dir", str(cross_out_dir),
  ], check=True)

  cafaeval_rows = pd.read_csv(cross_out_dir / "evaluation_best_f.tsv", sep="\t")
  model_rows = cafaeval_rows[cafaeval_rows["filename"] == "model.tsv"]

  ia_df = pd.read_csv(scorer.IA_PATH, sep="\t", header=None, names=["term", "weight"])
  ia_map = dict(zip(ia_df["term"], ia_df["weight"]))
  ia_weights = np.array([ia_map.get(name, 0.0) for name in label_names])
  thresholds = np.arange(0.01, 1.0, 0.01)

  from pk_eval_common import build_term_to_aspect
  import obonet
  obonet_graph = obonet.read_obo("data/cafa-6-protein-function-prediction/Train/go-basic.obo")
  term_to_aspect = build_term_to_aspect(obonet_graph)
  aspect_vector = np.array([term_to_aspect.get(name) for name in label_names])
  namespace_to_letter = {"biological_process": "P", "molecular_function": "F", "cellular_component": "C"}

  print(f"{'aspect':<8}{'hand-rolled f':<16}{'cafaeval f':<16}")
  for _, row in model_rows.iterrows():
    letter = namespace_to_letter.get(row["ns"])
    if letter is None:
      continue
    mask = aspect_vector == letter
    hand_fmax, _, _, _ = compute_ia_weighted_fmax(
        probs[:, mask], known[:, mask], novel[:, mask], ia_weights[mask], thresholds)
    print(f"{letter:<8}{hand_fmax:<16.4f}{row['f']:<16.4f}")
    assert abs(hand_fmax - row["f"]) < 0.02, (
        f"aspect {letter}: hand-rolled {hand_fmax} vs cafaeval {row['f']} - outside tolerance")
  print("[OK] cafaeval cross-check passed within tolerance")


if __name__ == "__main__":
  import sys
  if "--self-test" in sys.argv:
    _self_test()
  else:
    main()
```

Add the missing import at the top of the file: `from pk_eval_common import compute_ia_weighted_fmax`.

- [ ] **Step 3: Add the cross-check output dir to `.gitignore`**

```gitignore
/experiments/pk_eval/temporal_pk_validation_flat/predictions/*
/experiments/pk_eval/cafaeval_crosscheck/*
```

- [ ] **Step 4: Run the real cross-check**

Run: `env/bin/python experiments/pk_eval/scripts/crosscheck_cafaeval.py`
Expected: prints the per-aspect hand-rolled-vs-cafaeval `f` table and
`[OK] cafaeval cross-check passed within tolerance`. If the assertion fails, that is a
real finding (a bug in `compute_ia_weighted_fmax` or a misunderstanding of `cafaeval`'s
own normalization) to investigate before trusting Task 9's numbers — not something to
raise the tolerance to paper over.

- [ ] **Step 5: Commit**

```bash
git add experiments/pk_eval/scripts/crosscheck_cafaeval.py .gitignore
git commit -m "feat: run real cafaeval cross-check of hand-rolled IA-weighted Fmax"
```
