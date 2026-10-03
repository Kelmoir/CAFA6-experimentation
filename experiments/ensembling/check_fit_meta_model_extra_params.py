"""Standalone check (no pytest in this repo): confirms fit_meta_model()'s new
`extra_params` dict actually reaches the fitted XGBRanker, so a winning
hyperparameter config found by a GPU search (run externally, see
experiments/ensembling/colab_hpo_search.py) can be applied locally without any
further code changes. Run via:
    env/bin/python experiments/ensembling/check_fit_meta_model_extra_params.py
"""
import numpy as np

from stacking_pipeline import fit_meta_model

n_rows, n_labels = 200, 50
rng = np.random.default_rng(42)

y = np.zeros((n_rows, n_labels), dtype=np.float32)
half = np.array([1.0] * (n_rows // 2) + [0.0] * (n_rows // 2))
for j in range(n_labels):
    y[:, j] = rng.permutation(half)
y_flat = y.ravel()

X = np.stack([
    (y + rng.normal(scale=0.3, size=y.shape)).ravel(),
    (y + rng.normal(scale=0.3, size=y.shape)).ravel(),
    rng.normal(size=y.shape).ravel(),
], axis=1).astype(np.float32)
qid = np.tile(np.arange(n_labels), n_rows)

model = fit_meta_model(
    X, y_flat, qid, n_labels,
    monotone_constraints=(1, 1, 0),
    extra_params={"max_depth": 4, "learning_rate": 0.2, "n_estimators": 30, "min_child_weight": 3},
)

config = model.get_booster().save_config()
# learning_rate round-trips through the booster's internal float32 repr (0.200000003),
# so check it's close to 0.2 rather than an exact string match.
for needle in ['"max_depth":"4"', '"min_child_weight":"3"']:
    assert needle in config, f"expected {needle} in fitted booster config, got: {config[:800]}"
import re  # noqa: E402
lr_match = re.search(r'"learning_rate":"([\d.]+)"', config)
assert lr_match and abs(float(lr_match.group(1)) - 0.2) < 1e-5, f"learning_rate not ~0.2: {config[:800]}"
assert model.n_estimators == 30, model.n_estimators
print("[PASS] extra_params reached the fitted XGBRanker (max_depth, learning_rate, min_child_weight, n_estimators)")
