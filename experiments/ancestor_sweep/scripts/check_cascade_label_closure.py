"""Sequential-cascade precondition check: for phase 1 of experiments/sequential_cascade_plan.md
(restructure the existing depth<=2, 383-term coarse label space into a 3-expert
root->depth1->depth2 cascade), each depth-2 term's expert needs a depth-1 parent already
present in the *selected coarse label set* to condition on, and each depth-1 term needs one
of the 3 aspect roots. This is a different question from inspect_protein_ancestry.py (which
checked train_terms.tsv row completeness against the full is_a/part_of ancestor closure) -
here we check whether the already-selected 383-term label space is itself closed under the
immediate parent relation, since that's what determines whether the cascade's per-level
conditioning is well-defined at all.

Reads data/cafa-6-protein-function-prediction/dataset/go_dependency_graph.json (written by
generate_dataset.py) rather than go-basic.obo directly, since it already carries the exact
coarse_terms selection alongside the full graph.

Usage: check_cascade_label_closure.py [--n N] [--seed N]
"""

import argparse
import json
import random
from pathlib import Path

import networkx as nx

GRAPH_PATH = Path("data/cafa-6-protein-function-prediction/dataset/go_dependency_graph.json")
TRUE_PATH_RELATIONSHIPS = {"is_a", "part_of"}


def build_true_path_graph(graph_json: dict) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_nodes_from(n["id"] for n in graph_json["nodes"])
    for link in graph_json["links"]:
        if link["relationship"] in TRUE_PATH_RELATIONSHIPS:
            g.add_edge(link["source"], link["target"], key=link["key"])
    return g


def immediate_parents(graph: nx.MultiDiGraph, term: str) -> set:
    return set(graph.successors(term)) if term in graph else set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="Number of depth-2 terms to check")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle depth-2 terms before taking the first N (default: dataset iteration order, i.e. the literal 'first N')")
    parser.add_argument("--all", action="store_true", help="Check every depth-2 term (and every depth-1 term's root chain), not just --n of them; suppresses per-term printout, summary only")
    args = parser.parse_args()

    with GRAPH_PATH.open() as f:
        data = json.load(f)

    root_ids = set(data["root_term_ids"].keys())
    coarse_terms = data["coarse_terms"]
    coarse_ids = set(coarse_terms.keys())
    depth1_ids = {t for t, meta in coarse_terms.items() if meta["depth"] == 1}
    depth2_ids = [t for t, meta in coarse_terms.items() if meta["depth"] == 2]

    print(f"Coarse label space: {len(coarse_ids)} terms (3 roots + {len(depth1_ids)} depth-1 + {len(depth2_ids)} depth-2)")

    graph = build_true_path_graph(data["graph"])

    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(depth2_ids)

    sample = depth2_ids if args.all else depth2_ids[: args.n]
    verbose = not args.all

    # GO is a DAG, not a tree: a depth-2 term (by shortest-path-to-root) can have MULTIPLE
    # real is_a/part_of parents, and only the one(s) on its own shortest path are guaranteed
    # to sit at depth 1 - other real parents can have a longer shortest path of their own
    # (e.g. depth 3), landing outside the depth<=2 coarse selection even though the edge
    # itself is real GO structure, not a data gap. What the cascade actually needs is AT
    # LEAST ONE usable parent per level to condition on, not every real parent in scope.
    n_fully_closed = 0
    unclosed = []
    for term in sample:
        if verbose:
            name = next((n["name"] for n in data["graph"]["nodes"] if n["id"] == term), "?")
            print(f"\n{term} ({name}), depth=2")
        parents = immediate_parents(graph, term)
        if not parents:
            if verbose:
                print(f"  {term} -> (no is_a/part_of parent found in the graph at all - unexpected)")
            unclosed.append(term)
            continue

        usable_depth1_parents = []
        for parent in sorted(parents):
            level = "depth-1" if parent in depth1_ids else ("root" if parent in root_ids else "not in coarse label space (real GO parent, but its own shortest path exceeds depth 2)")
            if verbose:
                print(f"  {term} -> {parent} [{level}]")
            if parent in depth1_ids:
                usable_depth1_parents.append(parent)

        term_closed = False
        for parent in usable_depth1_parents:
            grandparents = immediate_parents(graph, parent)
            usable_roots = [gp for gp in sorted(grandparents) if gp in root_ids]
            if verbose:
                for gp in sorted(grandparents):
                    gp_level = "root" if gp in root_ids else ("depth-1/2" if gp in coarse_ids else "not in coarse label space")
                    print(f"    {parent} -> {gp} [{gp_level}]")
            if usable_roots:
                term_closed = True
        if verbose and not usable_depth1_parents:
            print(f"  ** {term}: NO usable depth-1 parent - cascade cannot condition this term **")
        elif verbose and not term_closed:
            print(f"  ** {term}: has a depth-1 parent, but that parent has no usable root - cascade chain breaks **")
        if term_closed:
            n_fully_closed += 1
        else:
            unclosed.append(term)

    print(f"\n{n_fully_closed}/{len(sample)} {'sampled' if not args.all else ''} depth-2 terms have at least one usable "
          f"depth-2->depth-1->root chain inside the 383-term coarse label space.")
    if unclosed:
        print(f"Unclosed: {unclosed}")
    print("This determines whether phase 1's 3-expert cascade (root -> depth1 -> depth2) "
          "can condition every expert on a predecessor that's actually in the label space, "
          "without needing generate_dataset.py changes.")


if __name__ == "__main__":
    main()
