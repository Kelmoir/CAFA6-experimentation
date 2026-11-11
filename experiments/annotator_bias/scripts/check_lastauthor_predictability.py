"""Item 13 step 2, take 2 (experiments/cafa6_next_steps_plan.md): the real predictability test.

check_assignedby_predictability.py tested whether `Assigned_By` (a curation DATABASE/consortium --
MGI, FlyBase, UniProt, GO_Central, Reactome, ComplexPortal, ...) can be predicted from the GO term.
That's not "can the lab/annotator be predicted" -- it's a different, less interesting question:
those values are institutional groups, not individual labs, and several are themselves topically
scoped by charter (ComplexPortal annotates complexes, Reactome annotates pathways, DisProt
annotates disordered regions, SynGO annotates synapse biology) -- "GO term predicts assigned_by"
for those is close to tautological, not evidence of persistent human bias.

This script tests the actual hypothesis: predict the PubMed `lastauthor` (an individual-researcher
identity key, from fetch_pubmed_lastauthors.py, joined in via the GOA row's REFERENCE PMID) from
the GO term/aspect alone. Caveat carried over from that script's docstring: `lastauthor` is a
surname+initials string (e.g. "Tooze SA") -- two different people can collide under it, especially
common CJK surnames, so this is a noisy proxy for true identity, not an exact one.

Confound control: restricted to TAXON ID == 9606 (human) only, rather than the prior script's
>=10-taxa "thematic" filter. A lab's own research program naturally concentrates on one organism,
so unlike assigned_by, there's no clean multi-species subset of labs to fall back on -- the only
way to fully remove organism identity as a predictor is to hold it constant. Human has by far the
most rows in this data (635,856 non-IEA rows pre-join), so this doesn't cost much coverage.

Lastauthor classes with few rows make the multiclass fit unstable (thousands of singleton labs) --
restricted to lastauthors with >= MIN_ROWS_PER_LASTAUTHOR rows in the human non-IEA data, logged.
"""
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

GOA_FILE = Path(__file__).resolve().parents[1] / "goa_attributions.tsv"
LASTAUTHOR_FILE = Path(__file__).resolve().parents[1] / "pubmed_lastauthors.tsv"
HUMAN_TAXON_ID = 9606
MIN_ROWS_PER_LASTAUTHOR = 50
SEED = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(GOA_FILE, sep="\t", usecols=[
        "GENE PRODUCT ID", "GO TERM", "GO ASPECT", "GO EVIDENCE CODE", "TAXON ID", "REFERENCE",
    ])
    df = df[(df["GO EVIDENCE CODE"] != "IEA") & (df["TAXON ID"] == HUMAN_TAXON_ID)]
    df = df[df["REFERENCE"].str.startswith("PMID:", na=False)].copy()
    df["pmid"] = df["REFERENCE"].str.slice(len("PMID:"))

    la = pd.read_csv(LASTAUTHOR_FILE, sep="\t", dtype={"pmid": str})
    la = la[la["lastauthor"].str.len() > 0]
    df = df.merge(la[["pmid", "lastauthor"]], on="pmid", how="inner")
    df = df.rename(columns={
        "GENE PRODUCT ID": "protein", "GO TERM": "go_term", "GO ASPECT": "aspect",
    })

    counts = df["lastauthor"].value_counts()
    keep = set(counts[counts >= MIN_ROWS_PER_LASTAUTHOR].index)
    print(f"[INFO] human, non-IEA, PMID-resolved: {len(df)} rows, {df['protein'].nunique()} "
          f"proteins, {df['lastauthor'].nunique()} unique lastauthors")
    print(f"[INFO] restricting to {len(keep)} lastauthors with >= {MIN_ROWS_PER_LASTAUTHOR} rows")
    df = df[df["lastauthor"].isin(keep)].reset_index(drop=True)
    return df


def run_check(df: pd.DataFrame) -> None:
    features = ["go_term", "aspect"]
    work = df[features + ["protein", "lastauthor"]].copy()
    for col in features:
        work[col] = work[col].astype("category")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(work, groups=work["protein"]))
    X_train, X_test = work.iloc[train_idx][features], work.iloc[test_idx][features]
    y_train_raw = work.iloc[train_idx]["lastauthor"]
    y_test_raw = work.iloc[test_idx]["lastauthor"]

    # Same fix as check_assignedby_predictability.py: encoder fit on the train fold only, after
    # splitting, since a protein-grouped split can leave some classes with zero train occurrences.
    encoder = LabelEncoder().fit(y_train_raw)
    y_train = encoder.transform(y_train_raw)
    test_known = y_test_raw.isin(encoder.classes_)
    n_unseen = int((~test_known).sum())
    X_test, y_test_raw = X_test[test_known], y_test_raw[test_known]
    y_test = encoder.transform(y_test_raw)

    clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.2, enable_categorical=True,
        tree_method="hist", objective="multi:softprob",
        random_state=SEED, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    majority_class_acc = (y_test == pd.Series(y_test).mode()[0]).mean()

    print(f"[RESULT] n_rows={len(work)} n_classes={len(encoder.classes_)} "
          f"n_train_proteins={work.iloc[train_idx]['protein'].nunique()} "
          f"n_test_proteins={work.iloc[test_idx]['protein'].nunique()} "
          f"n_test_rows_dropped_unseen_class={n_unseen}")
    print(f"[RESULT] accuracy={acc:.4f} macro_f1={macro_f1:.4f} "
          f"majority_class_baseline_acc={majority_class_acc:.4f}")


def main():
    df = load_data()
    run_check(df)


if __name__ == "__main__":
    main()
