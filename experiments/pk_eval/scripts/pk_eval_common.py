"""Shared helpers for the temporal PK/LK/NK evaluation harness (build + score + the
one-time cafaeval cross-check). See docs/superpowers/specs/2026-10-05-temporal-pk-eval-design.md.
"""
import sys
from pathlib import Path

# Running this script directly puts experiments/pk_eval/scripts/ on sys.path, not the
# repo root - same fix experiments/ancestor_sweep/scripts/diagnostic_eval_stage2.py
# already uses, needed here so `from generate_dataset import ...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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
