"""Automated term-driven pass over the positional-MF structural gap: for every molecular_function
GO term that has never been confirmed-localized for any protein, sample up to 5 proteins that
carry it as a real annotation, look for a raw signal (domain accession, ligand, or free-text
note) that recurs across the sample, and -- only when that signal looks narrow/specific enough
to be safe to generalize -- add it to experiments/positional_mf_localization_dictionary.json.

This mirrors the manual gap-review loop's judgment calls (see build_positional_mf_links.py's
module docstring and the dictionary's own growth history): a domain accession shared by only a
handful of proteins dataset-wide is almost always a single protein/gene-family's own domain, so
mapping it broadly to a term is safe (the ancestor-closure confirmation gate in link_protein
means a wrong/overreaching mapping just fails to confirm rather than asserting a false
annotation). A domain shared by dozens+ of unrelated proteins (generic folds like LRR/WD40/a
plain kinase catalytic domain) is declined here, same as every manual round -- it would encode
"this fold correlates with the term" rather than "this specific protein/family has the term".

Every visited term (added or not) is recorded in experiments/positional_mf_visited_terms.json so
repeat runs never redo work and the loop has a clear terminating condition: run until every
never-localized term has been visited.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAW_FEATURES_PATH = Path("experiments/positional_mf_raw_features.json")
LINKS_PATH = Path("experiments/positional_mf_links.json")
DICT_PATH = Path("experiments/positional_mf_localization_dictionary.json")
VISITED_PATH = Path("experiments/positional_mf_visited_terms.json")
GO_OBO_PATH = Path("data/cafa-6-protein-function-prediction/Train/go-basic.obo")

TEXT_TIER_FEATURE_TYPES = {"ACT_SITE", "SITE", "MOTIF", "REGION"}
NARROW_DOMAIN_THRESHOLD = 15  # dataset-wide proteins carrying the accession, inclusive cutoff
GENERIC_LIGANDS = {
    "atp", "gtp", "adp", "gdp", "amp", "substrate", "product", "water", "nadh", "nad+", "nadph",
    "nadp+", "fad", "fmn", "coenzyme a", "pyridoxal 5'-phosphate", "heme", "heme b", "dna", "rna",
}
GENERIC_LIGAND_PREFIXES = ("a divalent", "an ", "a ")  # e.g. "a divalent metal cation"
GENERIC_TEXT_NOTES = {"disordered", "cleavage", "n-acetylmethionine", "phosphoserine", "phosphothreonine"}
STOPWORDS = {
    "a", "an", "the", "of", "with", "and", "or", "binding", "activity", "protein", "receptor",
    "via", "in", "to", "for", "site", "domain", "region", "motif", "interaction", "type",
    "coupled", "class", "factor", "complex", "signal", "acid", "channel", "sequence",
}
SEED = 42


def content_words(text):
  return {w for w in re.split(r"[\s\-_,()/]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def is_specific_text(low, go_name_words, min_recur_met):
  """A text note is safe to add broadly only if it's a real phrase (not a single bare word)
  and, when it's the sole (n=1) piece of evidence for the term, shares vocabulary with the GO
  term's own name -- otherwise a single incidental co-occurring note could get generalized
  with no corroboration at all."""
  if len(low) < 10 or (" " not in low and "-" not in low):
    return False
  if min_recur_met:
    return True
  return bool(content_words(low) & go_name_words)


def is_generic_ligand(lig):
  low = lig.lower().strip()
  if low in GENERIC_LIGANDS:
    return True
  if low.endswith(")") and "(" in low and len(low) <= 10:
    return True  # bare ion, e.g. "Zn(2+)"
  if low.startswith(GENERIC_LIGAND_PREFIXES):
    return True
  return False


def load_go_names():
  names = {}
  cur_id = None
  with open(GO_OBO_PATH) as f:
    for line in f:
      line = line.rstrip("\n")
      if line == "[Term]":
        cur_id = None
      elif line.startswith("id: GO:"):
        cur_id = line[4:]
      elif line.startswith("name: ") and cur_id:
        names[cur_id] = line[6:]
        cur_id = None
  return names


def protein_signal(rec):
  ligands = []
  domain_xrefs = []
  text_notes = []
  for f in rec["features"]:
    if f["type"] == "BINDING":
      lig = f["qualifiers"].get("ligand")
      if lig:
        ligands.append(lig)
    elif f["type"] in TEXT_TIER_FEATURE_TYPES:
      note = f["qualifiers"].get("note")
      if note:
        text_notes.append(note)
  for d in rec.get("domain_xrefs", []):
    domain_xrefs.append((d["db"], d["accession"], d["description"]))
  return {"ligands": ligands, "domain_xrefs": domain_xrefs, "text_notes": text_notes}


def main():
  raw_features = json.load(open(RAW_FEATURES_PATH))
  links = json.load(open(LINKS_PATH))
  dictionary = json.load(open(DICT_PATH))
  go_names = load_go_names()
  visited = json.load(open(VISITED_PATH)) if VISITED_PATH.exists() else {}
  round_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0

  domain_freq = Counter()
  for acc, rec in raw_features.items():
    seen = {f"{d['db']}:{d['accession']}" for d in rec.get("domain_xrefs", [])}
    domain_freq.update(seen)

  existing_ligands = set()
  for entry in dictionary.values():
    existing_ligands.update(entry.get("ligands", []))

  localized_real = set()
  term_accessions = defaultdict(list)
  for acc, r in links.items():
    for loc in r["localized"]:
      localized_real.add(loc["matched_real_term"])
  for acc, rec in raw_features.items():
    for g in rec["go_terms"]:
      if g["aspect"] == "F":
        term_accessions[g["go_id"]].append(acc)
  never_localized = {t for t in term_accessions if t not in localized_real}
  todo = sorted(t for t in never_localized if t not in visited)

  print(f"never-localized: {len(never_localized)}, already visited: {len(visited)}, to process this pass: {len(todo)}")

  n_added = n_no_signal = 0
  decisions = []

  for t in todo:
    accs = term_accessions[t]
    r = random.Random(SEED + (hash(t) % 1000000))
    sample = r.sample(accs, min(5, len(accs)))
    sig_counter = Counter()
    domain_desc = {}
    for acc in sample:
      sig = protein_signal(raw_features[acc])
      seen_domains = set()
      for db, accn, desc in sig["domain_xrefs"]:
        key = f"{db}:{accn}"
        if key in seen_domains:
          continue
        seen_domains.add(key)
        sig_counter[("domain", key)] += 1
        domain_desc[key] = desc
      for lig in set(sig["ligands"]):
        sig_counter[("ligand", lig)] += 1
      for note in set(sig["text_notes"]):
        sig_counter[("text", note)] += 1

    name = go_names.get(t, "?")
    go_name_words = content_words(name)
    entry = dictionary.get(t, {"name": name, "ligands": [], "feature_types": [], "domain_accessions": [], "text_patterns": []})
    added_any = False
    reasons = []

    for (kind, val), cnt in sig_counter.items():
      real_corrob = cnt >= 2  # signal recurred across >=2 independently-sampled proteins
      if kind == "domain":
        if not real_corrob:
          continue  # a domain seen in only one of the sampled proteins isn't corroborated --
                     # it's just something that protein happens to also carry, not a defining signal
        freq = domain_freq.get(val, 999999)
        if freq <= NARROW_DOMAIN_THRESHOLD and val not in entry["domain_accessions"]:
          entry["domain_accessions"].append(val)
          added_any = True
          reasons.append(f"domain {val} ({domain_desc.get(val,'')}) freq={freq} recur={cnt}/{len(sample)}")
      elif kind == "ligand":
        if is_generic_ligand(val):
          continue
        if val in existing_ligands:
          continue
        if not real_corrob and not (content_words(val) & go_name_words):
          continue  # singleton evidence: require the ligand name to echo the GO term's own name
        if val not in entry["ligands"]:
          entry["ligands"].append(val)
          existing_ligands.add(val)
          added_any = True
          reasons.append(f"ligand {val!r} recur={cnt}/{len(sample)}")
      elif kind == "text":
        low = val.lower().strip()
        if any(g in low for g in GENERIC_TEXT_NOTES):
          continue
        if not is_specific_text(low, go_name_words, real_corrob):
          continue
        if low not in entry["text_patterns"]:
          entry["text_patterns"].append(low)
          added_any = True
          reasons.append(f"text {low!r} recur={cnt}/{len(sample)}")

    if added_any:
      dictionary[t] = entry
      visited[t] = {"status": "added", "round": round_n, "reasons": reasons, "n_proteins": len(accs)}
      n_added += 1
      decisions.append((t, name, "added", reasons))
    else:
      visited[t] = {"status": "no_signal", "round": round_n, "n_proteins": len(accs)}
      n_no_signal += 1

  json.dump(dict(sorted(dictionary.items())), open(DICT_PATH, "w"), indent=1)
  json.dump(visited, open(VISITED_PATH, "w"), indent=1, sort_keys=True)

  print(f"added: {n_added}, no_signal: {n_no_signal}")
  print()
  print("=== sample of 'added' decisions (first 40) ===")
  for t, name, status, reasons in decisions[:40]:
    print(f"{t} {name}")
    for r_ in reasons:
      print(f"   + {r_}")


if __name__ == "__main__":
  main()
