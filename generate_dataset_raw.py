"""Fully raw/asserted GO-term dataset variant -- no ancestor propagation whatsoever.

generate_dataset.py's 383-term coarse space (and the depth1/aspect variants alongside it)
all label a protein via the full true-path ANCESTOR CLOSURE of its asserted terms -- a
protein counts as positive for a coarse term if any descendant term was propagated up to
it, not just what a curator actually asserted. That's a deliberate, well-justified choice
there (propagation recovers the ~98% of implied ancestor annotations that aren't spelled
out as explicit train_terms.tsv rows), but per this session's own direction it's also NOT
"the actual, human-annotated GO-terms" in the strictest sense -- it's a derived space.

This variant is the strict alternative: a protein's labels are EXACTLY its directly
asserted train_terms.tsv rows, filtered to the selected vocabulary -- no closure
expansion, no depth restriction (depth<=2 was specifically a coarse-vs-propagated-space
concept; dropping propagation removes the reason to cap depth at all). The label
vocabulary is selected purely by raw annotation count, not by graph position.

Known, accepted cost of this choice (verified directly against train_terms.tsv): at the
same >=100-annotation bar this session's dataset_refinement_pilot.py already uses for its
override-eligible-term filter, dropping propagation still leaves 668 terms (vs. 383 in the
propagated coarse space) covering 89.1% of proteins with at least one in-vocabulary label
-- richer coverage of the real annotated space, not a subset of it, but ~11% of proteins
end up with zero usable labels and are excluded (removed by generate_dataset()'s existing
"no labels in vocab" safety net, which is NOT a rare edge case here the way it is for the
propagated variants -- every protein propagates to at least an aspect root there, but a
protein's directly-asserted terms can all legitimately fall under the raw-count threshold
here). Unlike every other generate_dataset*.py variant, the 3 aspect roots are NOT
force-included: annotators essentially never assert them directly (they're almost always
reached only via propagation), so under a purely-raw-count rule they earn their place the
same way every other term does, not by fiat -- in practice they won't clear a sane
threshold and won't appear in the vocabulary at all.

Reuses generate_dataset.py's sequence loading, cluster-aware split (compute_cluster_aware_split
-- the current best-practice split, unlike the older plain-random split the depth1/aspect
variants still use), generate_dataset() (unmodified: this variant doesn't need extra
columns, so the shared protein/labels/sequence schema applies as-is), and
export_dependency_graph() (kept for future graph-embedding work per
experiments/graph_embedding_plan.md, even though propagation isn't used for labels here).
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
    restrict_to_true_path,
)
from split_utils import compute_cluster_aware_split

# Raw (unpropagated) annotation-count bar for vocabulary inclusion. Matches the threshold
# dataset_refinement_pilot.py's --min-term-annotations already settled on this session for
# the same underlying reasoning (best-supported terms first) -- 668 terms clear this bar,
# covering 89.1% of proteins with >=1 in-vocabulary label (vs. 1585 terms/92.8% at >=50,
# or 4227 terms/96.5% at >=20 -- all verified directly against train_terms.tsv).
RAW_MIN_ANNOTATIONS = 100


def select_raw_terms(train_df: pd.DataFrame, depths: dict,
                     min_annotations: int = RAW_MIN_ANNOTATIONS) -> dict:
  """Label vocabulary selected by RAW, unpropagated annotation count alone -- no
  ancestor closure, no depth restriction (depths is used only to attach informational
  depth metadata to the export, never to filter candidates). Returns
  {term_id: {"depth", "count"}}, matching select_coarse_terms's shape so
  export_dependency_graph's payload stays consistent across variants. depth is None
  for a term outside the true-path graph entirely (obsolete/unreachable term IDs
  occasionally appear as raw train_terms.tsv rows even though they can't happen in the
  propagated variants, where every candidate term comes from the depths dict itself).
  """
  counts = train_df.groupby("term").size()
  return {
      term: {"depth": depths.get(term), "count": int(count)}
      for term, count in counts.items()
      if count >= min_annotations
  }


def main():
  data_path = Path("./data/cafa-6-protein-function-prediction")
  prot_sequences = {}
  for sequence in SeqIO.parse(data_path / "Train/train_sequences.fasta", "fasta"):
    prot_sequences[sequence.id.split("|")[1]] = str(sequence.seq).upper().replace(r"[UZOB]", "X")
  train_df = pd.read_csv(data_path / "Train/train_terms.tsv", delimiter="\t")
  obonet_graph = obonet.read_obo(data_path / "Train/go-basic.obo")

  # depths is informational only here (see select_raw_terms) -- no propagation of labels
  # happens in this variant, unlike generate_dataset.py's propagate_protein_terms.
  true_path_graph = restrict_to_true_path(obonet_graph)
  depths = compute_depths(true_path_graph)
  raw_terms = select_raw_terms(train_df, depths)
  print(f"[INFO] Selected {len(raw_terms)} raw GO terms "
        f"(>={RAW_MIN_ANNOTATIONS} directly-asserted annotations, no propagation, no depth cap)")

  label2id = {term: idx for idx, term in enumerate(sorted(raw_terms))}

  # Each protein's labels are EXACTLY its asserted train_terms.tsv rows, filtered to the
  # selected vocabulary -- no ancestor closure expansion (the defining difference from
  # generate_dataset.py's propagated_terms).
  protein_to_labels = {
      protein_id: sorted(terms & raw_terms.keys())
      for protein_id, terms in train_df.groupby("EntryID")["term"].apply(set).items()
  }

  # Cluster-aware split (see generate_dataset.py's own comment on this) -- same current
  # best-practice split, applied to the same train_terms.tsv protein universe, so this
  # variant's train/val/test protein ID assignment matches kelmoir/cafa6's.
  unique_protein_ids = list(train_df["EntryID"].unique())
  split_assignment = compute_cluster_aware_split(unique_protein_ids)
  train_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "train"]
  val_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "validation"]
  test_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "test"]
  print(f"Splitting IDs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

  dataset_dict_out = datasets.DatasetDict({
      "train": generate_dataset(train_ids, protein_to_labels, label2id, prot_sequences),
      "validation": generate_dataset(val_ids, protein_to_labels, label2id, prot_sequences),
      "test": generate_dataset(test_ids, protein_to_labels, label2id, prot_sequences),
  })

  out_path = data_path / "dataset_raw"
  dataset_dict_out.save_to_disk(out_path)
  export_dependency_graph(obonet_graph, raw_terms, out_path / "go_dependency_graph.json",
                          selection_rule={"min_annotations": RAW_MIN_ANNOTATIONS, "propagated": False,
                                          "depth_cap": None})
  dataset_dict_out.push_to_hub("kelmoir/cafa6-raw", private=False)

  print("\nRaw (unpropagated) datasets created successfully.")


if __name__ == "__main__":
  main()
