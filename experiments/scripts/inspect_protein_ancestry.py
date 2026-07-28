"""Ad hoc diagnostic: picks a protein from train_terms.tsv, prints its raw (as-listed)
annotations, then lists every is_a/part_of ancestor of each - to see directly whether an
"ancestor missing" observation reflects a real gap, or just the normal, expected fact
that train_terms.tsv only lists the most-specific asserted terms (ancestors are almost
never spelled out as their own explicit rows - generate_dataset.py's propagate_protein_terms()
exists specifically to add these back in computationally, not because anything is broken).

Usage: inspect_protein_ancestry.py [protein_id] [--seed N]
"""

import argparse
import random
import sys
from pathlib import Path

import networkx as nx
import obonet
import pandas as pd

TRUE_PATH_RELATIONSHIPS = {"is_a", "part_of"}
DATA_DIR = Path("data/cafa-6-protein-function-prediction/Train")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protein_id", nargs="?", default=None, help="EntryID to inspect (random if omitted)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for picking a protein (ignored if protein_id given)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    train_df = pd.read_csv(DATA_DIR / "train_terms.tsv", sep="\t")

    protein_id = args.protein_id or random.choice(train_df["EntryID"].unique().tolist())
    rows = train_df[train_df["EntryID"] == protein_id]
    if rows.empty:
        sys.exit(f"No rows found for protein {protein_id!r} in train_terms.tsv")

    print(f"Protein: {protein_id}")
    print(f"Raw annotations in train_terms.tsv ({len(rows)} rows):")
    for _, row in rows.iterrows():
        print(f"  {row['term']} ({row['aspect']})")

    graph = obonet.read_obo(DATA_DIR / "go-basic.obo")
    restricted = nx.MultiDiGraph()
    restricted.add_nodes_from(graph.nodes(data=True))
    for u, v, key in graph.edges(keys=True):
        if key in TRUE_PATH_RELATIONSHIPS:
            restricted.add_edge(u, v, key=key)

    raw_terms = set(rows["term"])
    print("\nAncestors (is_a/part_of only) per raw annotated term:")
    for term in sorted(raw_terms):
        ancestors = nx.descendants(restricted, term) if term in restricted else set()
        if not ancestors:
            print(f"  {term} -> (none - a root term, or unresolved in the graph)")
            continue
        for anc in sorted(ancestors):
            print(f"  {term} -> {anc}")

    all_ancestors = set()
    for term in raw_terms:
        all_ancestors |= (nx.descendants(restricted, term) if term in restricted else set())
    explicit = all_ancestors & raw_terms
    implied_only = all_ancestors - raw_terms
    print(f"\nOf {len(all_ancestors)} distinct ancestors implied by this protein's raw terms:")
    print(f"  {len(explicit)} are ALSO listed as their own explicit row in train_terms.tsv")
    print(f"  {len(implied_only)} are NOT listed explicitly (only implied via the true-path rule) - "
          f"expected: propagate_protein_terms() adds these back in, not a gap in the data itself.")


if __name__ == "__main__":
    main()
