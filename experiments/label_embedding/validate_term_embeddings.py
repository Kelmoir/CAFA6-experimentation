"""Item 8 step 2: revisit Q2's distinctiveness check against the actual cached,
flag-augmented embeddings (experiments/label_embedding/term_embeddings.pt) and the full
27,191-term trainable vocabulary -- not just the 2,612-pair antonym sample the salvage
probe validated the fix on. Gates whether the model-integration step is worth building.

Checks all 4 known antonym naming patterns (not just positive/negative regulation),
plus sibling and random pairs, restricted to terms actually in the trainable vocabulary.

Run: env/bin/python experiments/label_embedding/validate_term_embeddings.py
"""
import random
import re

import obonet
import torch

random.seed(42)

OBO_PATH = "data/cafa-6-protein-function-prediction/Train/go-basic.obo"
EMB_PATH = "experiments/label_embedding/term_embeddings.pt"

# Same patterns as build_term_embeddings.py's DIRECTION_PATTERNS, split out per-pattern
# here so the breakdown can show whether the fix generalizes beyond the dominant one.
NAMED_PATTERNS = {
    "positive/negative regulation of": (r"^positive regulation of (.+)$", r"^negative regulation of (.+)$"),
    "activation of / inhibition of": (r"^activation of (.+)$", r"^inhibition of (.+)$"),
    "agonist / antagonist activity": (r"^(.+) agonist activity$", r"^(.+) antagonist activity$"),
    "stimulation of / suppression of": (r"^stimulation of (.+)$", r"^suppression of (.+)$"),
}


def summarize(label, sims):
    sims = torch.tensor(sims)
    print(f"{label:32s}: n={len(sims):<5d} mean={sims.mean():.4f} median={sims.median():.4f} "
          f"min={sims.min():.4f} max={sims.max():.4f}")


def cosine_for_pairs(pairs, idx, emb):
    return [float((emb[idx[a]] * emb[idx[b]]).sum()) for a, b in pairs]


def find_antonym_pairs(names_by_id, pos_re, neg_re):
    pos_suffix = {}
    neg_suffix = {}
    for node, name in names_by_id.items():
        m = re.match(pos_re, name)
        if m:
            pos_suffix[m.group(1)] = node
        m = re.match(neg_re, name)
        if m:
            neg_suffix[m.group(1)] = node
    return [(pos_suffix[s], neg_suffix[s]) for s in pos_suffix.keys() & neg_suffix.keys()]


def find_sibling_pairs(g, trainable_ids, n=500):
    by_parent = {}
    for child, parent, key in g.out_edges(keys=True):
        if key != "is_a" or child not in trainable_ids or parent not in trainable_ids:
            continue
        by_parent.setdefault(parent, []).append(child)
    pairs = []
    for children in by_parent.values():
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                pairs.append((children[i], children[j]))
    random.shuffle(pairs)
    return pairs[:n]


def find_random_pairs(trainable_ids, n=500):
    ids = list(trainable_ids)
    return [tuple(random.sample(ids, 2)) for _ in range(n)]


if __name__ == "__main__":
    saved = torch.load(EMB_PATH)
    terms, emb = saved["terms"], saved["embeddings"]
    idx = {t: i for i, t in enumerate(terms)}
    trainable_ids = set(terms)
    print(f"[INFO] loaded {len(terms)} term embeddings, dim={emb.shape[1]}")

    g = obonet.read_obo(OBO_PATH)
    names_by_id = {node: attrs["name"] for node, attrs in g.nodes(data=True) if "name" in attrs}

    summarize("random pairs", cosine_for_pairs(find_random_pairs(trainable_ids), idx, emb))
    summarize("sibling pairs", cosine_for_pairs(find_sibling_pairs(g, trainable_ids), idx, emb))

    print()
    for label, (pos_re, neg_re) in NAMED_PATTERNS.items():
        pairs = find_antonym_pairs(names_by_id, pos_re, neg_re)
        pairs = [(a, b) for a, b in pairs if a in trainable_ids and b in trainable_ids]
        if not pairs:
            print(f"{label:32s}: n=0 (no pairs in trainable vocabulary)")
            continue
        summarize(label, cosine_for_pairs(pairs, idx, emb))
