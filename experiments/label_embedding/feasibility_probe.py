"""Tier 3 item 8 feasibility spike: can a cheap off-the-shelf text encoder produce
GO-term-definition embeddings that are semantically distinctive enough to separate
closely related terms (siblings, parent/child, and especially positive/negative-
regulation antonym pairs)? No training -- this only tests off-the-shelf encoder quality.

Run: env/bin/python experiments/label_embedding/feasibility_probe.py
"""
import random
import re

import obonet
import torch
from transformers import AutoModel, AutoTokenizer

random.seed(42)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_defs(obo_path):
    g = obonet.read_obo(obo_path)
    defs = {}
    for node, attrs in g.nodes(data=True):
        d = attrs.get("def")
        name = attrs.get("name")
        if d and name:
            text = re.sub(r'^"|"\s*\[.*\]$', "", d.strip())
            defs[node] = (name, text)
    return g, defs


def embed(texts, tok, model, device="cpu", batch_size=64):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled)
    return torch.cat(out, dim=0)


def find_antonym_pairs(defs, g):
    """positive_regulation_of_X vs negative_regulation_of_X siblings (share a parent)."""
    pos_terms = {n: (name, txt) for n, (name, txt) in defs.items() if name.startswith("positive regulation of ")}
    neg_terms = {n: (name, txt) for n, (name, txt) in defs.items() if name.startswith("negative regulation of ")}
    pairs = []
    for n_pos, (name_pos, _) in pos_terms.items():
        suffix = name_pos[len("positive regulation of "):]
        target_name = "negative regulation of " + suffix
        for n_neg, (name_neg, _) in neg_terms.items():
            if name_neg == target_name:
                pairs.append((n_pos, n_neg))
                break
    return pairs


def find_sibling_pairs(g, defs, n=200):
    by_parent = {}
    for child, parent, key in g.out_edges(keys=True):
        if key != "is_a":
            continue
        by_parent.setdefault(parent, []).append(child)
    sibling_pairs = []
    parents = [p for p, kids in by_parent.items() if len(kids) >= 2]
    random.shuffle(parents)
    for p in parents:
        kids = [k for k in by_parent[p] if k in defs]
        if len(kids) >= 2:
            a, b = random.sample(kids, 2)
            sibling_pairs.append((a, b))
        if len(sibling_pairs) >= n:
            break
    return sibling_pairs


def find_random_pairs(defs, n=200):
    ids = list(defs.keys())
    pairs = []
    for _ in range(n):
        a, b = random.sample(ids, 2)
        pairs.append((a, b))
    return pairs


def cosine_for_pairs(pairs, id_to_idx, emb):
    sims = []
    for a, b in pairs:
        if a in id_to_idx and b in id_to_idx:
            sims.append((emb[id_to_idx[a]] @ emb[id_to_idx[b]]).item())
    return sims


def summarize(name, sims):
    if not sims:
        print(f"{name}: n=0 (no pairs found)")
        return
    t = torch.tensor(sims)
    print(f"{name}: n={len(sims)} mean={t.mean():.4f} median={t.median():.4f} "
          f"min={t.min():.4f} max={t.max():.4f}")


if __name__ == "__main__":
    obo_path = "data/cafa-6-protein-function-prediction/Train/go-basic.obo"
    g, defs = load_defs(obo_path)
    print(f"[INFO] {len(defs)}/{g.number_of_nodes()} terms have a usable def")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    ids = list(defs.keys())
    texts = [defs[i][1] for i in ids]
    id_to_idx = {i: idx for idx, i in enumerate(ids)}

    print(f"[INFO] embedding {len(texts)} term definitions with {MODEL_NAME}...")
    emb = embed(texts, tok, model)
    print(f"[INFO] embeddings shape: {emb.shape}")

    random_pairs = find_random_pairs(defs, n=500)
    sibling_pairs = find_sibling_pairs(g, defs, n=500)
    antonym_pairs = find_antonym_pairs(defs, g)
    print(f"[INFO] found {len(antonym_pairs)} positive/negative-regulation antonym sibling pairs")

    summarize("random (unrelated) pairs       ", cosine_for_pairs(random_pairs, id_to_idx, emb))
    summarize("sibling pairs (share a parent)  ", cosine_for_pairs(sibling_pairs, id_to_idx, emb))
    summarize("antonym pairs (pos_reg/neg_reg) ", cosine_for_pairs(antonym_pairs, id_to_idx, emb))

    # Show a few concrete antonym examples so similarity numbers have faces.
    print("\n[INFO] sample antonym pairs with cosine similarity:")
    for a, b in antonym_pairs[:8]:
        sim = (emb[id_to_idx[a]] @ emb[id_to_idx[b]]).item()
        print(f"  {sim:.4f}  {defs[a][0]!r:55s}  vs  {defs[b][0]!r}")
