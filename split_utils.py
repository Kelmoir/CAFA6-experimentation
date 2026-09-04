"""Cluster-aware train/val/test protein split, shared by generate_dataset.py and
experiments/mf_span/scripts/select_positional_mf_proteins.py so the two pipelines never disagree about
which proteins are train/val/test.

Replaces the original plain-random-by-accession split (random.shuffle on EntryID, no homology
awareness). Investigated directly on 2026-08-20 after a span-localization model trained on the
old split scored suspiciously well for its size: data/.../Train/split.fasta.clstr (a CD-HIT
clustering at ~90-99% sequence identity, bundled with the raw Kaggle data but never wired into
either pipeline) showed 17.4% of validation proteins in the positional-MF working set shared a
cluster with a training protein -- near-identical sequences (same conserved motif, almost
certainly at nearly the same residue offset) straddling the split boundary, letting a model
pattern-match a near-duplicate instead of genuinely generalizing.
"""
import random
import re
from pathlib import Path

CLSTR_PATH = Path("./data/cafa-6-protein-function-prediction/Train/split.fasta.clstr")

_ACCESSION_PATTERN = re.compile(r">(?:sp|tr)\|(\w+)\|")


def parse_clusters(clstr_path: Path = CLSTR_PATH) -> list[list[str]]:
  """CD-HIT .clstr file -> list of accession-lists, one list per cluster."""
  clusters = []
  current = []
  with open(clstr_path) as f:
    for line in f:
      if line.startswith(">Cluster"):
        if current:
          clusters.append(current)
        current = []
      else:
        m = _ACCESSION_PATTERN.search(line)
        if m:
          current.append(m.group(1))
  if current:
    clusters.append(current)
  return clusters


def compute_cluster_aware_split(unique_protein_ids: list[str], clstr_path: Path = CLSTR_PATH,
                                 seed: int = 42, test_frac: float = 0.18,
                                 val_frac_of_remainder: float = 0.22) -> dict[str, str]:
  """Same target proportions as the original split (test_frac of everything, then
  val_frac_of_remainder of what's left for validation, rest train), but assigning whole CD-HIT
  clusters to a single split so no near-duplicate sequence pair crosses a split boundary.

  Clusters are shuffled (not proteins) and greedily filled train-then-val-then-test against the
  target counts -- the same left-to-right threshold logic the original slice-based split used,
  just operating on whole clusters so cluster sizes only add quantization noise to the target
  proportions, never actual identity leakage across the boundary. Any protein not covered by the
  clstr file gets its own singleton cluster (defensive; verified none currently occurs).
  """
  protein_set = set(unique_protein_ids)
  clusters = parse_clusters(clstr_path)

  assigned = set()
  cluster_groups = []
  for cluster in clusters:
    members = [a for a in cluster if a in protein_set]
    if members:
      cluster_groups.append(members)
      assigned.update(members)

  for pid in unique_protein_ids:
    if pid not in assigned:
      cluster_groups.append([pid])

  rng = random.Random(seed)
  rng.shuffle(cluster_groups)

  total = len(unique_protein_ids)
  test_target = int(total * test_frac)
  train_val_target = total - test_target
  val_target = int(train_val_target * val_frac_of_remainder)
  train_target = train_val_target - val_target

  assignment = {}
  counts = {"train": 0, "validation": 0, "test": 0}
  for group in cluster_groups:
    if counts["train"] < train_target:
      split = "train"
    elif counts["validation"] < val_target:
      split = "validation"
    else:
      split = "test"
    for pid in group:
      assignment[pid] = split
    counts[split] += len(group)

  return assignment
