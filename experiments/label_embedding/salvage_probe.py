"""Follow-up to feasibility_probe.py: can Q2's antonym-collapse failure be salvaged cheaply?
Tests two candidate fixes against the same antonym/sibling/random pair sets:
  (1) embed "name. definition" instead of definition alone (does naming the direction word
      prominently help the encoder separate them?)
  (2) a zero-training regex-based direction flag concatenated onto the embedding (does a
      near-free deterministic feature fully separate the one dominant antonym pattern?)

Run: env/bin/python experiments/label_embedding/salvage_probe.py
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


def embed(texts, tok, model, batch_size=64):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled)
    return torch.cat(out, dim=0)


def find_antonym_pairs(defs):
    pos_terms = {n: name for n, (name, _) in defs.items() if name.startswith("positive regulation of ")}
    neg_terms = {n: name for n, (name, _) in defs.items() if name.startswith("negative regulation of ")}
    neg_by_suffix = {name[len("negative regulation of "):]: n for n, name in neg_terms.items()}
    pairs = []
    for n_pos, name_pos in pos_terms.items():
        suffix = name_pos[len("positive regulation of "):]
        if suffix in neg_by_suffix:
            pairs.append((n_pos, neg_by_suffix[suffix]))
    return pairs


def find_sibling_pairs(g, defs, n=500):
    by_parent = {}
    for child, parent, key in g.out_edges(keys=True):
        if key != "is_a":
            continue
        by_parent.setdefault(parent, []).append(child)
    pairs = []
    parents = [p for p, kids in by_parent.items() if len(kids) >= 2]
    random.shuffle(parents)
    for p in parents:
        kids = [k for k in by_parent[p] if k in defs]
        if len(kids) >= 2:
            pairs.append(tuple(random.sample(kids, 2)))
        if len(pairs) >= n:
            break
    return pairs


def find_random_pairs(defs, n=500):
    ids = list(defs.keys())
    return [tuple(random.sample(ids, 2)) for _ in range(n)]


def cosine_for_pairs(pairs, id_to_idx, emb):
    sims = []
    for a, b in pairs:
        if a in id_to_idx and b in id_to_idx:
            sims.append((emb[id_to_idx[a]] @ emb[id_to_idx[b]]).item())
    return sims


def summarize(name, sims):
    if not sims:
        print(f"{name}: n=0")
        return
    t = torch.tensor(sims)
    print(f"{name}: n={len(sims)} mean={t.mean():.4f} median={t.median():.4f} min={t.min():.4f} max={t.max():.4f}")


DIRECTION_RE = re.compile(r"^(positive|negative) regulation of ")


def direction_flag(name):
    m = DIRECTION_RE.match(name)
    if not m:
        return 0.0  # no direction signal
    return 1.0 if m.group(1) == "positive" else -1.0


if __name__ == "__main__":
    obo_path = "data/cafa-6-protein-function-prediction/Train/go-basic.obo"
    g, defs = load_defs(obo_path)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    ids = list(defs.keys())
    id_to_idx = {i: idx for idx, i in enumerate(ids)}
    antonym_pairs = find_antonym_pairs(defs)
    sibling_pairs = find_sibling_pairs(g, defs)
    random_pairs = find_random_pairs(defs)
    print(f"[INFO] {len(antonym_pairs)} antonym pairs, {len(sibling_pairs)} sibling pairs, {len(random_pairs)} random pairs\n")

    # --- Fix attempt 1: embed "name. definition" instead of definition alone ---
    print("=== Fix 1: name + definition concatenated ===")
    texts_namedef = [f"{defs[i][0]}. {defs[i][1]}" for i in ids]
    emb_namedef = embed(texts_namedef, tok, model)
    summarize("random pairs       ", cosine_for_pairs(random_pairs, id_to_idx, emb_namedef))
    summarize("sibling pairs      ", cosine_for_pairs(sibling_pairs, id_to_idx, emb_namedef))
    summarize("antonym pairs      ", cosine_for_pairs(antonym_pairs, id_to_idx, emb_namedef))

    print("\n=== Fix 1b: name only (no definition at all) ===")
    texts_name = [defs[i][0] for i in ids]
    emb_name = embed(texts_name, tok, model)
    summarize("random pairs       ", cosine_for_pairs(random_pairs, id_to_idx, emb_name))
    summarize("sibling pairs      ", cosine_for_pairs(sibling_pairs, id_to_idx, emb_name))
    summarize("antonym pairs      ", cosine_for_pairs(antonym_pairs, id_to_idx, emb_name))

    # --- Fix attempt 2: concatenate a cheap deterministic direction-flag scalar ---
    print("\n=== Fix 2: definition embedding + 1-dim regex direction flag, renormalized ===")
    emb_def = embed([defs[i][1] for i in ids], tok, model)
    flags = torch.tensor([direction_flag(defs[i][0]) for i in ids]).unsqueeze(1)
    SCALE = 2.0  # how much weight the flag gets relative to the unit-norm text embedding
    augmented = torch.cat([emb_def, SCALE * flags], dim=1)
    augmented = torch.nn.functional.normalize(augmented, dim=-1)
    summarize("random pairs       ", cosine_for_pairs(random_pairs, id_to_idx, augmented))
    summarize("sibling pairs      ", cosine_for_pairs(sibling_pairs, id_to_idx, augmented))
    summarize("antonym pairs      ", cosine_for_pairs(antonym_pairs, id_to_idx, augmented))
    n_flagged = int((flags != 0).sum())
    print(f"[INFO] regex direction flag fires on {n_flagged}/{len(ids)} terms "
          f"(covers only the explicit 'positive/negative regulation of' pattern)")
