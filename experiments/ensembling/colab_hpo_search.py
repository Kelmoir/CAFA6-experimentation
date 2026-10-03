"""Self-contained GPU hyperparameter search for the CAFA-6 stacking meta-model --
run this on a GPU-backed Colab session, not in this repo's sandbox (see the handoff
instructions in the chat that pointed you here). Needs only numpy, scipy, xgboost,
and the uploaded meta_train_for_hpo.npz (exported by export_meta_train_for_hpo.py) --
no dataset, no BLAST, no ESM-2 checkpoint, no repo clone.

What it does: rank-transforms esm_prob/blast_score per term (matching
build_rank_features() in stacking_pipeline.py exactly), then runs a random
hyperparameter search for xgboost.XGBRanker(objective="rank:pairwise"), each config
scored by 4-fold cross-validation split BY PROTEIN (never by term -- a term's rows
must stay together within a fold, or the per-term AUROC the search optimizes for
would be meaningless) over meta-train's 2000 proteins. The monotone_constraints
variant is itself part of the search space (confirmed to matter a lot: an earlier
finding showed (1,1,0) took this model from clearly losing to a trivial baseline to
tying it).

Writes every trial's config + CV score to hpo_results.json (incrementally, after
each trial, so a disconnect doesn't lose progress) and prints a [TRIAL]/[BEST] log
line per trial. Download hpo_results.json back and hand the top entry to the
stacking_pipeline.py pipeline's fit_meta_model(..., monotone_constraints=...,
extra_params=...) for the final, honest evaluation on the untouched meta-eval slice
(done locally, not here -- this script never sees meta-eval).

Usage (on Colab, after uploading meta_train_for_hpo.npz to the same directory):
    pip install -q xgboost scipy numpy
    python colab_hpo_search.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import scipy  # noqa: F401
    import xgboost  # noqa: F401
except ImportError:
    # Mirrors training_run.py's own Colab bootstrap convention (CLAUDE.md): pip-install
    # missing heavy deps at import time rather than assuming the Colab image has them.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost", "scipy"], check=True)

import numpy as np
from scipy.stats import rankdata

NPZ_PATH = Path("meta_train_for_hpo.npz")
RESULTS_PATH = Path("hpo_results.json")
N_TRIALS = 40
K_FOLDS = 4
SEED = 42


def rank_transform(matrix):
    n_rows = matrix.shape[0]
    return np.apply_along_axis(rankdata, 0, matrix) / n_rows


def gpu_available():
    """Checks the *resolved* device on the fitted booster, not just whether .fit()
    raised -- this xgboost build doesn't raise on a missing GPU, it silently falls
    back to CPU with only a UserWarning, so "fit() didn't raise" is not evidence of
    a working GPU (caught directly by a smoke test on a CPU-only machine, 2026-10-03:
    it claimed cuda was available there)."""
    import xgboost as xgb

    try:
        m = xgb.XGBRanker(tree_method="hist", device="cuda", n_estimators=1)
        m.fit(
            np.zeros((4, 1), dtype=np.float32), np.array([0, 1, 0, 1], dtype=np.float32),
            qid=np.array([0, 0, 1, 1]),
        )
        config = m.get_booster().save_config()
        resolved_cuda = '"device":"cuda' in config
        if not resolved_cuda:
            print("[INFO] device=cuda requested but xgboost silently fell back to CPU.")
        return resolved_cuda
    except Exception as e:
        print(f"[INFO] GPU not usable ({e}); falling back to CPU.")
        return False


def sample_config(rng):
    return {
        "max_depth": int(rng.choice([3, 4, 5, 6, 7, 8])),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
        "n_estimators": int(rng.choice([50, 100, 150, 200, 300])),
        "min_child_weight": int(rng.choice([1, 3, 5, 10, 20, 50])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(10)))),
        "max_bin": int(rng.choice([128, 256, 512, 1024])),
        "monotone_constraints": rng.choice([[1, 1, 0], [1, 1, 1], [1, 1, -1]]).tolist(),
    }


def cv_score(X, y, qid, protein_idx, n_rows, config, device):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    fold_of_protein = np.arange(n_rows) % K_FOLDS
    np.random.default_rng(SEED).shuffle(fold_of_protein)
    protein_fold = fold_of_protein[protein_idx]

    extra = {k: v for k, v in config.items() if k != "monotone_constraints"}
    fold_aurocs = []
    for k in range(K_FOLDS):
        train_mask = protein_fold != k
        val_mask = ~train_mask

        order = np.argsort(qid[train_mask], kind="stable")
        model = xgb.XGBRanker(
            objective="rank:pairwise", tree_method="hist", device=device, random_state=42,
            monotone_constraints=tuple(config["monotone_constraints"]), **extra,
        )
        model.fit(X[train_mask][order], y[train_mask][order], qid=qid[train_mask][order])

        val_preds = model.predict(X[val_mask])
        val_qid = qid[val_mask]
        val_y = y[val_mask]
        aurocs = []
        for term in np.unique(val_qid):
            term_mask = val_qid == term
            term_y = val_y[term_mask]
            if term_y.min() == term_y.max():
                continue
            aurocs.append(roc_auc_score(term_y, val_preds[term_mask]))
        if aurocs:
            fold_aurocs.append(float(np.mean(aurocs)))
    return float(np.mean(fold_aurocs)) if fold_aurocs else float("nan")


def main():
    print(f"[INFO] Loading {NPZ_PATH}...")
    data = np.load(NPZ_PATH)
    esm_prob, blast_score, val_labels, term_freq = (
        data["esm_prob"].astype(np.float32), data["blast_score"].astype(np.float32),
        data["val_labels"].astype(np.float32), data["term_freq"].astype(np.float32),
    )
    n_rows, n_labels = val_labels.shape
    print(f"[INFO] n_rows={n_rows} n_labels={n_labels}")

    print("[INFO] Rank-transforming esm_prob/blast_score per term...")
    rank_esm = rank_transform(esm_prob)
    rank_blast = rank_transform(blast_score)
    X = np.stack([rank_esm.ravel(), rank_blast.ravel(), np.tile(term_freq, n_rows)], axis=1).astype(np.float32)
    y = val_labels.ravel().astype(np.float32)
    qid = np.tile(np.arange(n_labels), n_rows).astype(np.int64)
    protein_idx = np.repeat(np.arange(n_rows), n_labels)

    device = "cuda" if gpu_available() else "cpu"
    print(f"[INFO] Using device={device}")

    rng = np.random.default_rng(SEED)
    results = []
    best = None
    for trial_i in range(N_TRIALS):
        config = sample_config(rng)
        t0 = time.time()
        score = cv_score(X, y, qid, protein_idx, n_rows, config, device)
        elapsed = time.time() - t0
        results.append({"trial": trial_i, "config": config, "cv_mean_auroc": score, "seconds": elapsed})
        if best is None or (score == score and score > best["cv_mean_auroc"]):  # score==score excludes NaN
            best = results[-1]
        RESULTS_PATH.write_text(json.dumps({"trials": results, "best": best}, indent=2))
        print(f"[TRIAL {trial_i+1}/{N_TRIALS}] cv_mean_auroc={score:.4f} ({elapsed:.1f}s) config={config}")
        print(f"[BEST SO FAR] cv_mean_auroc={best['cv_mean_auroc']:.4f} config={best['config']}")

    print(f"[DONE] wrote {RESULTS_PATH}")
    print(f"[BEST] cv_mean_auroc={best['cv_mean_auroc']:.4f}")
    print(f"[BEST] config={best['config']}")


if __name__ == "__main__":
    main()
