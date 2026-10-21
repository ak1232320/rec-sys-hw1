"""The hybrid: LightGBM LambdaRank over the candidate pool of src/features.py.

Fitting happens only on splits whose answer key lies inside the weeks the
model is allowed to learn from:

  hold-out evaluation   fit on inner (11-15 -> 16-20), score valid (11-20 -> 21-30)
  final submission      fit on valid (11-20 -> 21-30), score test  (11-30 -> 31-42)

Early stopping uses a fifth of the *fitting* split's users, never the split
being scored. The number of trees found on the inner fit is reused unchanged
for the final fit.
"""
import numpy as np
import pandas as pd

import features

SEED = 20260112
PARAMS = dict(
    objective="lambdarank",
    metric="map",
    eval_at=[5],
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=2000,
    random_state=SEED,
    verbose=-1,
)


def _grouped(frame):
    frame = frame.sort_values(["user_id", "tv_show_id"], kind="mergesort").reset_index(drop=True)
    groups = frame.groupby("user_id", sort=False).size().to_numpy()
    return frame, groups


def fit(frame, n_estimators=None, binary=False, holdout=0.2, params=None):
    """Fit a ranker; with n_estimators=None, pick it by early stopping on a user hold-out."""
    import lightgbm as lgb

    cfg = dict(PARAMS, **(params or {}))
    frame, _ = _grouped(frame)
    cols = features.feature_columns(frame)
    y = (frame["label"] > 0).astype(int) if binary else frame["label"]

    if n_estimators is None:
        users = frame["user_id"].unique()
        rng = np.random.default_rng(SEED)
        held = set(rng.choice(users, size=int(len(users) * holdout), replace=False))
        mask = frame["user_id"].isin(held).to_numpy()
        tr, va = frame[~mask], frame[mask]
        model = lgb.LGBMRanker(**cfg)
        model.fit(tr[cols], y[~mask], group=tr.groupby("user_id", sort=False).size().to_numpy(),
                  eval_set=[(va[cols], y[mask])],
                  eval_group=[va.groupby("user_id", sort=False).size().to_numpy()],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        n_estimators = int(model.best_iteration_ or cfg["n_estimators"])

    cfg["n_estimators"] = n_estimators
    model = lgb.LGBMRanker(**cfg)
    model.fit(frame[cols], y, group=frame.groupby("user_id", sort=False).size().to_numpy())
    model.n_trees_ = n_estimators
    model.feature_cols_ = cols
    return model


def rank(model, frame, split, k=5):
    """Top-k per user from the ranker; any slot it cannot fill falls back to popularity."""
    scored = frame[["user_id", "tv_show_id"]].copy()
    scored["score"] = model.predict(frame[model.feature_cols_])
    scored = scored.sort_values(["user_id", "score", "tv_show_id"], ascending=[True, False, True])
    lists = scored.groupby("user_id", sort=False)["tv_show_id"].apply(lambda s: s.head(k).tolist()).to_dict()

    import methods
    pop = methods.popularity_vector(split)
    fallback = split.candidates[np.lexsort((split.candidates, -pop))].tolist()
    filled = 0
    out = {}
    for u in split.eval_users:
        top = list(lists.get(u, []))
        if len(top) < k:
            filled += 1
            top += [s for s in fallback if s not in top][:k - len(top)]
        out[u] = top
    return out, filled


def importance(model):
    gain = model.booster_.feature_importance(importance_type="gain")
    return (pd.Series(gain, index=model.feature_cols_) / gain.sum()).sort_values(ascending=False)
