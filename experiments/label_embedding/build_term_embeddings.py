"""Item 8 step 1: precompute a fixed embedding for every trainable GO term, for the
label-embedding/compatibility-score output head. Definition embedding (all-MiniLM-L6-v2,
mean-pooled, L2-normalized) + a 1-dim direction flag for known antonym naming patterns
(see experiments/label_embedding/salvage_probe.py for why), concatenated and re-normalized.

The 27,191-term order is taken directly from the trainable dataset's own label feature
names -- this MUST match training_run.py's label2id ordering, since the resulting tensor
is indexed by class id at train/inference time.

Run: env/bin/python experiments/label_embedding/build_term_embeddings.py
"""
import re
from pathlib import Path

import obonet
import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OBO_PATH = "data/cafa-6-protein-function-prediction/Train/go-basic.obo"
DATASET_PATH = "data/cafa-6-protein-function-prediction/dataset_full"
OUT_PATH = Path("experiments/label_embedding/term_embeddings.pt")

# Known opposite-effect naming patterns found in the salvage probe's breadth check --
# "positive/negative regulation of" dominates (5,660/40,122 terms); the rest are cheap
# additions. Each tuple is (positive-word regex, negative-word regex) sharing one capture
# group for the common suffix, checked against the term's own name.
DIRECTION_PATTERNS = [
    (r"^positive regulation of (.+)$", r"^negative regulation of (.+)$"),
    (r"^activation of (.+)$", r"^inhibition of (.+)$"),
    (r"^(.+) agonist activity$", r"^(.+) antagonist activity$"),
    (r"^stimulation of (.+)$", r"^suppression of (.+)$"),
]
SCALE = 2.0


def load_defs(obo_path):
    g = obonet.read_obo(obo_path)
    defs = {}
    for node, attrs in g.nodes(data=True):
        d, name = attrs.get("def"), attrs.get("name")
        if d and name:
            defs[node] = (name, re.sub(r'^"|"\s*\[.*\]$', "", d.strip()))
    return defs


def direction_flag(name):
    for pos_re, neg_re in DIRECTION_PATTERNS:
        if re.match(pos_re, name):
            return 1.0
        if re.match(neg_re, name):
            return -1.0
    return 0.0


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
            out.append(torch.nn.functional.normalize(pooled, dim=-1))
    return torch.cat(out, dim=0)


if __name__ == "__main__":
    terms = load_from_disk(DATASET_PATH)["train"].features["labels"].feature.names
    print(f"[INFO] {len(terms)} trainable terms (label2id order)")

    defs = load_defs(OBO_PATH)
    missing = [t for t in terms if t not in defs]
    if missing:
        print(f"[WARN] {len(missing)}/{len(terms)} trainable terms have no go-basic.obo def "
              f"(e.g. {missing[:5]}) -- using the term id itself as filler text")

    names = [defs[t][0] if t in defs else t for t in terms]
    texts = [defs[t][1] if t in defs else t for t in terms]
    flags = torch.tensor([direction_flag(n) for n in names], dtype=torch.float32) * SCALE
    n_flagged = int((flags != 0).sum())
    print(f"[INFO] direction flag fires on {n_flagged}/{len(terms)} terms")

    print(f"[INFO] embedding {len(texts)} definitions with {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    def_emb = embed(texts, tok, model)

    combined = torch.cat([def_emb, flags.unsqueeze(1)], dim=1)
    combined = torch.nn.functional.normalize(combined, dim=-1)
    print(f"[INFO] final embedding tensor: {combined.shape}")

    torch.save({"terms": terms, "embeddings": combined}, OUT_PATH)
    print(f"[INFO] saved to {OUT_PATH}")
