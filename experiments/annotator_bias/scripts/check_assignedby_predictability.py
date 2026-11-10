"""Item 13 step 2 (experiments/cafa6_next_steps_plan.md): cheap go/no-go predictability check --
can an XGBoost classifier predict which lab/database (GO_Central, UniProt, MGI, FlyBase, ...)
produced a given GO annotation, from the GO term (+ optionally species) alone? If not learnably
distinct from chance, the user's "labs have persistent term/aspect preferences" hypothesis is
wrong and item 13 stops here before step 3's bigger PU-learning build.

IEA (electronic) rows are excluded -- no human curation to find bias in.

Confound this check is specifically designed to separate out: many Assigned_By values are
near-organism-exclusive curation databases (MGI=mouse, FlyBase=fly, SGD=yeast, ...) -- taxon_id
alone would trivially predict those, which is a known, uninteresting fact about how GOA is
organized, not evidence for the user's actual hypothesis (persistent topical/research-direction
preference independent of organism biology). So this runs three variants:
  1. WITH taxon_id as a feature (upper bound / sanity check the pipeline can learn at all).
  2. WITHOUT taxon_id (the real test: is assigned_by predictable from GO term/aspect alone?).
  3. WITHOUT taxon_id, restricted to "thematic" databases only (those annotating >=10 distinct
     species in this data, e.g. UniProt/GO_Central/IntAct/Reactome/CACAO -- excludes clearly
     organism-exclusive single-species curation groups) -- the cleanest isolation of the
     hypothesis, since within this subset organism identity can't be doing the predicting.

Split by protein ID (not row), same convention as the main pipeline's train/val/test splits --
a term/aspect/taxon combination repeats across many rows for the same protein, so a row-level
split would leak.
"""
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

GOA_FILE = Path(__file__).resolve().parents[1] / "goa_attributions.tsv"
THEMATIC_MIN_TAXA = 10
SEED = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(GOA_FILE, sep="\t", usecols=[
        "GENE PRODUCT ID", "GO TERM", "GO ASPECT", "GO EVIDENCE CODE", "TAXON ID", "ASSIGNED BY",
    ])
    df = df[df["GO EVIDENCE CODE"] != "IEA"].reset_index(drop=True)
    df = df.rename(columns={
        "GENE PRODUCT ID": "protein", "GO TERM": "go_term", "GO ASPECT": "aspect",
        "TAXON ID": "taxon_id", "ASSIGNED BY": "assigned_by",
    })
    return df


def run_check(df: pd.DataFrame, use_taxon: bool, label: str) -> None:
    features = ["go_term", "aspect"] + (["taxon_id"] if use_taxon else [])
    work = df[features + ["protein", "assigned_by"]].copy()
    for col in features:
        work[col] = work[col].astype("category")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(work, groups=work["protein"]))
    X_train, X_test = work.iloc[train_idx][features], work.iloc[test_idx][features]
    y_train_raw = work.iloc[train_idx]["assigned_by"]
    y_test_raw = work.iloc[test_idx]["assigned_by"]

    # Encoder fit on the TRAIN fold only, after splitting -- encoding against the full
    # dataset's category set first (then splitting) can leave gaps in y_train's observed
    # codes when a protein-grouped split sends every row of some rare class to the held-out
    # proteins, and XGBoost's sklearn wrapper rejects non-contiguous label ranges outright.
    # Test rows whose assigned_by never appeared in train can't be predicted correctly by
    # construction (the model has no class for them) -- dropped before scoring, count logged.
    encoder = LabelEncoder().fit(y_train_raw)
    y_train = encoder.transform(y_train_raw)
    test_known = y_test_raw.isin(encoder.classes_)
    n_unseen = int((~test_known).sum())
    X_test, y_test_raw = X_test[test_known], y_test_raw[test_known]
    y_test = encoder.transform(y_test_raw)
    n_classes = len(encoder.classes_)

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

    print(f"[RESULT] {label}: n_rows={len(work)} n_classes_in_train={n_classes} "
          f"n_train_proteins={work.iloc[train_idx]['protein'].nunique()} "
          f"n_test_proteins={work.iloc[test_idx]['protein'].nunique()} "
          f"n_test_rows_dropped_unseen_class={n_unseen}")
    print(f"[RESULT] {label}: accuracy={acc:.4f} macro_f1={macro_f1:.4f} "
          f"majority_class_baseline_acc={majority_class_acc:.4f}")


def main():
    df = load_data()
    print(f"[INFO] {len(df)} non-IEA annotation rows, {df['protein'].nunique()} proteins, "
          f"{df['assigned_by'].nunique()} assigned_by values")

    taxon_counts = df.groupby("assigned_by")["taxon_id"].nunique()
    thematic = set(taxon_counts[taxon_counts >= THEMATIC_MIN_TAXA].index)
    print(f"[INFO] {len(thematic)}/{df['assigned_by'].nunique()} assigned_by values span "
          f">={THEMATIC_MIN_TAXA} distinct taxa (\"thematic\", not organism-exclusive): "
          f"{sorted(thematic)}")

    run_check(df, use_taxon=True, label="WITH taxon_id (sanity upper bound)")
    run_check(df, use_taxon=False, label="WITHOUT taxon_id (real hypothesis test)")

    thematic_df = df[df["assigned_by"].isin(thematic)].reset_index(drop=True)
    print(f"[INFO] thematic-only subset: {len(thematic_df)} rows, "
          f"{thematic_df['protein'].nunique()} proteins")
    run_check(thematic_df, use_taxon=False,
              label="WITHOUT taxon_id, thematic-only subset (cleanest hypothesis test)")


if __name__ == "__main__":
    main()
