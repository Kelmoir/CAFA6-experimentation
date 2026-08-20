from Bio import SeqIO
import pandas as pd
import obonet
import networkx as nx
from pathlib import Path
import numpy as np
from progressbar import progressbar
import datasets
import json

from split_utils import compute_cluster_aware_split

prot_sequences = {}
unique_prot = []

# The 3 namespace roots of go-basic.obo - depth 0 in the graph, equivalent to (and a
# strict generalization of) the "aspect" (C/F/P) label the pipeline used before this
# change. Verified directly: these are the only 3 nodes with no outgoing is_a/part_of
# edges, and no edge in the graph crosses between their namespaces.
ROOT_TERM_IDS = {"GO:0003674": "F", "GO:0005575": "C", "GO:0008150": "P"}

# GO's true-path rule (an annotation to a term implies annotation to every ancestor of
# that term) only holds for is_a/part_of edges. regulates/positively_regulates/
# negatively_regulates ("X regulates Y" does not mean "X is a kind of/part of Y") are
# excluded from propagation, but not discarded - export_dependency_graph() keeps them
# in the exported graph for anything that wants them later.
TRUE_PATH_RELATIONSHIPS = {"is_a", "part_of"}

# Initial coarse-to-fine cutoff: terms within 2 hops of their aspect root (by shortest
# path - the two conventions disagree for ~61% of terms in this ontology, by ~1.7
# levels on average, so this must be pinned down explicitly) with >=50 annotated
# proteins after ancestor propagation. Verified against train_terms.tsv: 380 terms
# clear this bar (vs. only 141 using raw, unpropagated counts - propagation matters a
# lot here, since ~98% of a protein's implied ancestor annotations aren't spelled out
# as explicit rows in the raw file).
COARSE_MAX_DEPTH = 2
COARSE_MIN_ANNOTATIONS = 50


def restrict_to_true_path(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
  """Keep only is_a/part_of edges - the subset GO's true-path rule supports."""
  restricted = nx.MultiDiGraph()
  restricted.add_nodes_from(graph.nodes(data=True))
  for u, v, key in graph.edges(keys=True):
    if key in TRUE_PATH_RELATIONSHIPS:
      restricted.add_edge(u, v, key=key)
  return restricted


def compute_depths(true_path_graph: nx.MultiDiGraph) -> dict:
  """Shortest-path distance from each term to its aspect root (root itself = 0)."""
  reversed_graph = true_path_graph.reverse(copy=False)
  depths = {}
  for root_id in ROOT_TERM_IDS:
    for term, distance in nx.single_source_shortest_path_length(reversed_graph, root_id).items():
      if term not in depths or distance < depths[term]:
        depths[term] = distance
  return depths


def propagate_protein_terms(train_df: pd.DataFrame, true_path_graph: nx.MultiDiGraph) -> dict:
  """Expand each protein's directly-annotated terms to the full true-path ancestor
  closure (raw terms union every is_a/part_of ancestor of each). Necessary because
  train_terms.tsv only lists the most-specific asserted annotations - checked directly
  against a 500-protein sample: ~98% of the ancestor annotations the true-path rule
  implies are not present as explicit rows, so skipping this step would make every
  ancestor/coarse term look far rarer than it actually is.
  """
  ancestor_cache = {}

  def ancestors_of(term: str) -> set:
    if term not in ancestor_cache:
      ancestor_cache[term] = nx.descendants(true_path_graph, term) if term in true_path_graph else set()
    return ancestor_cache[term]

  by_protein = train_df.groupby("EntryID")["term"].apply(set)
  propagated = {}
  for protein_id, terms in progressbar(by_protein.items(), total=len(by_protein), desc="propagating GO term ancestors"):
    closure = set(terms)
    for term in terms:
      closure |= ancestors_of(term)
    propagated[protein_id] = closure
  return propagated


def select_coarse_terms(propagated_terms: dict, depths: dict,
                        max_depth: int = COARSE_MAX_DEPTH,
                        min_annotations: int = COARSE_MIN_ANNOTATIONS) -> dict:
  """Pick the initial coarse-to-fine label set. Returns {term_id: {"depth", "count"}}
  for every selected term (the 3 roots always included, unconditionally - they're the
  depth-0/former-aspect task) so the selection is self-documenting downstream.
  """
  candidate_terms = {term for term, depth in depths.items() if 1 <= depth <= max_depth}
  counts = {}
  for terms in propagated_terms.values():
    for term in terms & candidate_terms:
      counts[term] = counts.get(term, 0) + 1

  selected = {
      term: {"depth": depths[term], "count": count}
      for term, count in counts.items()
      if count >= min_annotations
  }
  for root_id in ROOT_TERM_IDS:
    selected[root_id] = {
        "depth": 0,
        "count": sum(1 for terms in propagated_terms.values() if root_id in terms),
    }
  return selected


def export_dependency_graph(obonet_graph: nx.MultiDiGraph, coarse_terms: dict, out_path: Path) -> None:
  """Persist the full GO graph - every relationship type, including the
  regulates-family edges that aren't used for label propagation here but are kept
  rather than discarded - alongside the coarse-term selection. A later curriculum
  stage can load this and call networkx.descendants() on any coarse term (restricted
  to TRUE_PATH_RELATIONSHIPS) to get the remaining, finer terms under that branch,
  without recomputing the graph from scratch.
  """
  slim_graph = nx.MultiDiGraph()
  for node, attrs in obonet_graph.nodes(data=True):
    slim_graph.add_node(node, name=attrs.get("name"), namespace=attrs.get("namespace"))
  for u, v, key in obonet_graph.edges(keys=True):
    slim_graph.add_edge(u, v, key=key, relationship=key)

  payload = {
      # edges="links" pins the key name explicitly (networkx changed the default from
      # "links" to "edges" around 3.x) so this stays loadable with
      # networkx.readwrite.json_graph.node_link_graph(data["graph"], edges="links")
      # regardless of which networkx version reads it back later.
      "graph": nx.readwrite.json_graph.node_link_data(slim_graph, edges="links"),
      "true_path_relationships": sorted(TRUE_PATH_RELATIONSHIPS),
      "depth_convention": "shortest_path",
      "root_term_ids": ROOT_TERM_IDS,
      "coarse_terms": coarse_terms,
      "coarse_selection_rule": {"max_depth": COARSE_MAX_DEPTH, "min_annotations": COARSE_MIN_ANNOTATIONS},
  }
  out_path.write_text(json.dumps(payload))
  print(f"[INFO] Wrote GO dependency graph ({slim_graph.number_of_nodes()} nodes, "
        f"{slim_graph.number_of_edges()} edges, {len(coarse_terms)} coarse terms) to {out_path}")


def generate_dataset(protein_ids, protein_to_labels: dict, label2id: dict,
                     prot_sequences=prot_sequences, max_length=1024) -> datasets.Dataset:
  """
  Generates a dataset for the given protein_ids using precomputed per-protein labels.

  The labels are encoded as integer IDs using `label2id`, and the dataset
  features are annotated so Hugging Face tooling can understand the label space.
  """
  features, sequences, label_names = [], [], []
  removed = 0
  for protein_id in progressbar(protein_ids, desc="creating data as aligned lists"):
    sequence = prot_sequences.get(protein_id)
    if sequence is None or len(sequence) > max_length:
      removed += 1
      continue
    labels = sorted(label for label in protein_to_labels.get(protein_id, []) if label in label2id)
    if not labels:
      # Not expected to trigger: every protein propagates to at least its aspect
      # root(s), which are always in label2id - kept as a safety net, not a normal path.
      removed += 1
      continue
    features.append(protein_id)
    sequences.append(sequence)
    label_names.append(labels)

  print(f"[INFO] Generating Dataset with {len(features)} items.")
  print(f"[INFO] Number of proteins removed (too long / no labels in vocab): {removed}")
  label_feature = datasets.ClassLabel(
      names=sorted(label2id.keys())
  )
  dataset_dict = {
      "protein": features,
      "labels": label_names,
      "sequence": sequences,
  }
  datset_features = datasets.Features({
                                    "protein": datasets.Value("string"),
                                    "labels": datasets.Sequence(label_feature),
                                    "sequence": datasets.Value("string"),
                                })
  dataset_info = datasets.DatasetInfo(description="CAFA-6 based datasets for training ans exploring purposes",
                                citation= "Iddo Friedberg, Predrag Radivojac, Paul D Thomas, An Phan, M. Clara De Paolis Kaluza, Damiano Piovesan, Parnal Joshi, Chris Mungall, Martyna Plomecka, Walter Reade, and María Cruz. CAFA 6 Protein Function Prediction. https://kaggle.com/competitions/cafa-6-protein-function-prediction, 2025. Kaggle.",
                                features= datset_features,
                                )
  return datasets.Dataset.from_dict(
      dataset_dict,
      datset_features,
      info=dataset_info,
  )

# Create train, test, validation split
# Split the data into train and test splits
def create_train_val_split_from_dataset(dataset):
  # Shuffle first to ensure indices are randomized before splitting
  dataset = dataset.shuffle(seed=42)

  dataset_split = dataset.train_test_split(test_size=0.2, seed=42)

  # split the 80% train into train and validation (70/30 of the 80%)
  dataset_val_split = dataset_split["train"].train_test_split(test_size=0.3, seed=42)

  return datasets.DatasetDict({
      "train": dataset_val_split["train"],
      "validation": dataset_val_split["test"],
      "test": dataset_split["test"]
  })


def main():
  data_path = Path("./data/cafa-6-protein-function-prediction")
  for sequence in SeqIO.parse(data_path/"Train/train_sequences.fasta", "fasta"):
    unique_prot.append(sequence.id.split("|")[1])
    prot_sequences[sequence.id.split("|")[1]] = str(sequence.seq).upper().replace(r"[UZOB]", "X") # Reading the relevant data, and pre-formatting it for ProtBERT
  train_df = pd.read_csv(data_path/"Train/train_terms.tsv", delimiter="\t")
  obonet_graph = obonet.read_obo(data_path/"Train/go-basic.obo")

  true_path_graph = restrict_to_true_path(obonet_graph)
  depths = compute_depths(true_path_graph)
  propagated_terms = propagate_protein_terms(train_df, true_path_graph)
  coarse_terms = select_coarse_terms(propagated_terms, depths)
  print(f"[INFO] Selected {len(coarse_terms)} coarse GO terms "
        f"(3 aspect roots + depth<={COARSE_MAX_DEPTH} terms with >={COARSE_MIN_ANNOTATIONS} annotations after propagation)")

  # generate id2label and label2id datasets for the model to use (now graph-based coarse
  # GO terms instead of just the 3 aspects - the roots above subsume the old aspect task)
  label2id = {term: idx for idx, term in enumerate(sorted(coarse_terms))}
  id2label = {idx: term for term, idx in label2id.items()}

  protein_to_labels = {
      protein_id: sorted(terms & coarse_terms.keys())
      for protein_id, terms in propagated_terms.items()
  }

  # 1. Get unique protein IDs
  unique_protein_ids = list(train_df['EntryID'].unique())

  # 2. Cluster-aware split (same ~64/18/18 target proportions as the original plain-random
  # split, but assigning whole CD-HIT clusters -- ~90-99% sequence identity, from
  # data/.../Train/split.fasta.clstr -- to a single split, so no near-duplicate sequence pair
  # straddles train/val/test. See split_utils.py's docstring for why this matters: a
  # span-localization model trained against the old split scored suspiciously well, traced to
  # 17.4% of validation proteins sharing a cluster with a training protein.
  split_assignment = compute_cluster_aware_split(unique_protein_ids)
  train_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "train"]
  val_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "validation"]
  test_ids = [pid for pid in unique_protein_ids if split_assignment[pid] == "test"]

  print(f"Splitting IDs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

  # 4. Generate disjoint datasets
  dataset_dict = datasets.DatasetDict({
      "train": generate_dataset(train_ids, protein_to_labels, label2id),
      "validation": generate_dataset(val_ids, protein_to_labels, label2id),
      "test": generate_dataset(test_ids, protein_to_labels, label2id),
  })

  dataset_dict.save_to_disk(data_path/"dataset")
  export_dependency_graph(obonet_graph, coarse_terms, data_path/"dataset"/"go_dependency_graph.json")
  dataset_dict.push_to_hub("kelmoir/cafa6", private=False)

  print("\nDisjoint datasets created successfully.")


if __name__ == "__main__":
  main()
