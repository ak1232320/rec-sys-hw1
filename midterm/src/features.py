"""Candidate pool and features for the hybrid ranker (candidate generation + re-ranking).

Stage 1 pools, per user, the shows any component would put forward: the user's
own shows that are on air in the target weeks, each method's top-N, the global
top-N, and the shows that fill the user's habitual channel-hours in the target
schedule (src/exposure.py). Stage 2 describes every (user, show) pair in the pool with features
computed from the history weeks plus the target weeks' *schedule* - never
their viewing - and a LightGBM LambdaRank model orders the pool.

All rates are per week so that a ranker fitted on a 5-week history can score a
10- or 20-week one.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

import common
import data
import methods

COMPONENTS = ("cbf", "rules", "svd", "ppr", "exposure")
COMPONENT_PARAMS = {
    "cbf": {},
    "rules": {},
    "svd": {"factors": 256},
    "ppr": {"alpha": 0.5, "weight": "cosine"},
    "exposure": {},
}
OWN_TOP = 80        # at most this many of the user's own shows enter the pool
PER_METHOD = {"cbf": 20, "rules": 20, "svd": 20, "ppr": 20, "exposure": 60}
POP_TOP = 30        # global top-N
CHANNEL_TOP = 20    # top-N by the user's channel mix x target airtime
HALF_LIFE = 2.0     # weeks, for the decayed-seconds feature


def component_scores(split):
    return {name: methods.METHODS[name](split, **COMPONENT_PARAMS[name]) for name in COMPONENTS}


def _top_cols(M, n):
    n = min(n, M.shape[1])
    return np.argpartition(-M, n - 1, axis=1)[:, :n]


def _airtime(epg, weeks):
    span = weeks[1] - weeks[0] + 1
    air = epg.assign(sec=(epg["end"] - epg["start"]).dt.total_seconds())
    per_show = air.groupby("tv_show_id").agg(air=("sec", "sum"), broadcasts=("sec", "size"))
    per_show["air_pw"] = per_show["air"] / 3600.0 / span
    per_show["bc_pw"] = per_show["broadcasts"] / span
    by_channel = air.groupby(["tv_show_id", "channel_id"])["sec"].sum()
    hour = air["start"].dt.hour
    prime = air[(hour >= 19) & (hour < 23)].groupby("tv_show_id")["sec"].sum()
    per_show["prime_share"] = (prime.reindex(per_show.index).fillna(0) / per_show["air"]).astype(float)
    return per_show, by_channel


def channel_affinity(split):
    """users x candidates: the user's channel mix (all logged time, labelled or
    not) dotted with where each candidate airs in the target schedule."""
    log = split.hist_log
    channels = np.sort(np.union1d(log["channel_id"].unique(), split.epg_target["channel_id"].unique()))
    ch_index = {c: i for i, c in enumerate(channels)}
    per = log.groupby(["user_id", "channel_id"])["duration"].sum().reset_index()
    per = per[per["user_id"].isin(set(split.eval_users))]
    eval_index = {u: i for i, u in enumerate(split.eval_users)}
    U = sp.csr_matrix((per["duration"].to_numpy(float),
                       (per["user_id"].map(eval_index).to_numpy(), per["channel_id"].map(ch_index).to_numpy())),
                      shape=(len(split.eval_users), len(channels)))
    U = methods.normalize(U, norm="l1")

    _, by_channel = _airtime(split.epg_target, split.target_weeks)
    bc = by_channel.reset_index()
    cand_index = {s: i for i, s in enumerate(split.candidates)}
    bc = bc[bc["tv_show_id"].isin(cand_index)]
    S = sp.csr_matrix((bc["sec"].to_numpy(float),
                       (bc["channel_id"].map(ch_index).to_numpy(), bc["tv_show_id"].map(cand_index).to_numpy())),
                      shape=(len(channels), len(split.candidates)))
    S = methods.normalize(S, norm="l1", axis=0)          # where each show airs
    return np.asarray((U @ S).todense(), dtype=np.float32)


def build_pool(split, scores, chan_aff):
    n_users, n_cand = len(split.eval_users), len(split.candidates)
    parts = []

    own = split.X[split.eval_rows][:, split.cand_cols].tocsr()
    for i in range(n_users):
        lo, hi = own.indptr[i], own.indptr[i + 1]
        cols, vals = own.indices[lo:hi], own.data[lo:hi]
        if len(cols) > OWN_TOP:
            cols = cols[np.argpartition(-vals, OWN_TOP - 1)[:OWN_TOP]]
        parts.append(np.column_stack([np.full(len(cols), i), cols]))

    for name, M in scores.items():
        top = _top_cols(M, PER_METHOD[name])
        parts.append(np.column_stack([np.repeat(np.arange(n_users), top.shape[1]), top.ravel()]))

    pop = methods.popularity_vector(split)
    top_pop = np.argsort(-pop)[:POP_TOP]
    parts.append(np.column_stack([np.repeat(np.arange(n_users), len(top_pop)), np.tile(top_pop, n_users)]))

    top = _top_cols(chan_aff, CHANNEL_TOP)
    parts.append(np.column_stack([np.repeat(np.arange(n_users), top.shape[1]), top.ravel()]))

    pairs = np.unique(np.vstack(parts), axis=0)
    return pd.DataFrame({"row": pairs[:, 0], "col": pairs[:, 1]})


def build_features(split, pool, scores, chan_aff):
    lo, hi = split.history_weeks
    span = hi - lo + 1
    users = split.eval_users[pool["row"].to_numpy()]
    shows = split.candidates[pool["col"].to_numpy()]
    f = pd.DataFrame({"user_id": users, "tv_show_id": shows})

    # --- this user's history with this show -------------------------------
    h = split.hist
    h = h[h["user_id"].isin(set(split.eval_users))]
    decay = np.power(0.5, (hi - h["week"]) / HALF_LIFE)
    pair = (h.assign(decayed=h["seconds"] * decay,
                     recent=np.where(h["week"] > hi - 2, h["seconds"], 0))
            .groupby(["user_id", "tv_show_id"])
            .agg(sec=("seconds", "sum"), weeks=("week", "nunique"), last=("week", "max"),
                 decayed=("decayed", "sum"), recent=("recent", "sum")))
    f = f.join(pair, on=["user_id", "tv_show_id"])
    f["watched"] = f["sec"].notna().astype(np.int8)
    f["weeks_since"] = (hi - f["last"]).fillna(span + 1)
    f["week_share"] = f["weeks"].fillna(0) / span
    for col in ("sec", "decayed", "recent"):
        f[col] = f[col].fillna(0)

    user = h.groupby("user_id").agg(user_sec=("seconds", "sum"), user_shows=("tv_show_id", "nunique"))
    f = f.join(user, on="user_id")
    f["user_sec"] = f["user_sec"].fillna(0)
    f["user_shows"] = f["user_shows"].fillna(0)
    f["share"] = np.where(f["user_sec"] > 0, f["sec"] / f["user_sec"].clip(lower=1), 0)
    f["decayed_share"] = f["decayed"] / f.groupby("user_id")["decayed"].transform("sum").clip(lower=1)
    f["recent_share"] = f["recent"] / f.groupby("user_id")["recent"].transform("sum").clip(lower=1)
    f["sec_pw"] = np.log1p(f["sec"] / 60.0 / span)
    f["user_sec_pw"] = np.log1p(f["user_sec"] / 60.0 / span)
    f["rank_own"] = f.groupby("user_id")["sec"].rank(ascending=False, method="first")

    # --- how often, not just how long: views, airings and days -----------
    # Off for the submitted first-round hybrid, which counted preference in
    # seconds; they switch on only if the ranker is fitted to a count-based key.
    if common.TARGET != "seconds":
        ov = data.overlap_rows(split.history_weeks)
        ov = ov[ov["user_id"].isin(set(split.eval_users))]
        counts = ov.groupby(["user_id", "tv_show_id"]).agg(
            views=data.COUNTS["views"], airings=data.COUNTS["airings"], days=data.COUNTS["days"])
        f = f.join(counts, on=["user_id", "tv_show_id"])
        for col in ("views", "airings", "days"):
            f[col] = f[col].fillna(0)
            f[f"{col}_pw"] = f[col] / span
            f[f"{col}_share"] = f[col] / f.groupby("user_id")[col].transform("sum").clip(lower=1)
            f[f"rank_{col}"] = f.groupby("user_id")[col].rank(ascending=False, method="first")
        f = f.drop(columns=["views", "airings", "days"])

    # --- the show: audience in the history weeks, schedule in both -------
    show = split.hist.groupby("tv_show_id").agg(pop_sec=("seconds", "sum"), viewers=("user_id", "nunique"))
    n_active = split.hist["user_id"].nunique()
    f = f.join(show, on="tv_show_id")
    f["pop_pw"] = np.log1p(f["pop_sec"].fillna(0) / 3600.0 / span)
    f["viewer_share"] = f["viewers"].fillna(0) / n_active

    air_h, _ = _airtime(split.epg_hist, split.history_weeks)
    air_t, _ = _airtime(split.epg_target, split.target_weeks)
    f["air_hist_pw"] = f["tv_show_id"].map(air_h["air_pw"]).fillna(0)
    f["air_tgt_pw"] = f["tv_show_id"].map(air_t["air_pw"]).fillna(0)
    f["bc_tgt_pw"] = f["tv_show_id"].map(air_t["bc_pw"]).fillna(0)
    f["prime_share"] = f["tv_show_id"].map(air_t["prime_share"]).fillna(0)
    f["air_ratio"] = np.log((f["air_tgt_pw"] + 0.1) / (f["air_hist_pw"] + 0.1))
    f["is_new"] = (f["air_hist_pw"] == 0).astype(np.int8)
    # If the habit holds and the schedule changes, this is what the user would watch.
    f["expected_pw"] = np.log1p(f["sec"] / 60.0 / span * np.exp(f["air_ratio"]))

    # --- category affinity -----------------------------------------------
    cat = pd.concat([split.epg_hist, split.epg_target]).drop_duplicates("tv_show_id").set_index(
        "tv_show_id")["tv_show_category"]
    hc = h.assign(cat=h["tv_show_id"].map(cat)).groupby(["user_id", "cat"])["seconds"].sum()
    hc = (hc / hc.groupby(level=0).transform("sum")).rename("cat_share")
    f["cat"] = f["tv_show_id"].map(cat)
    f = f.join(hc, on=["user_id", "cat"])
    f["cat_share"] = f["cat_share"].fillna(0)
    f["cat"] = f["cat"].astype("category")

    # --- channel mix and the component methods ---------------------------
    rows, cols = pool["row"].to_numpy(), pool["col"].to_numpy()
    f["chan_aff"] = chan_aff[rows, cols]
    for name, M in scores.items():
        f[name] = M[rows, cols]
        f[f"{name}_rank"] = f.groupby("user_id")[name].rank(ascending=False, method="first")

    return f.drop(columns=["last", "pop_sec", "viewers", "user_sec"])


def feature_columns(frame):
    return [c for c in frame.columns if c not in ("user_id", "tv_show_id", "label")]


def labels(split, frame):
    """Graded relevance for LambdaRank: 5 for the user's #1 show ... 1 for #5."""
    grade = {}
    for u, shows in split.truth.items():
        for i, s in enumerate(shows):
            grade[(u, s)] = 5 - i
    return np.array([grade.get((u, s), 0) for u, s in zip(frame["user_id"], frame["tv_show_id"])],
                    dtype=np.int32)


def add_conversion(split, frame, pool):
    """Did the user stay for the show when their habit put them in front of it?

    `exposure_hist` is what the user's channel-hour habit predicts for the show
    on the *history* grid; `conversion` divides what they actually watched by
    it, and `expected_conv` carries that personal rate onto the target grid.
    """
    import exposure as _exposure

    hist_exp = _exposure.exposure(split, schedule="history")
    rows, cols = pool["row"].to_numpy(), pool["col"].to_numpy()
    span = split.history_weeks[1] - split.history_weeks[0] + 1
    frame["exposure_hist"] = hist_exp[rows, cols]
    watched_pw = frame["sec"].to_numpy() / span
    frame["conversion"] = watched_pw / (frame["exposure_hist"].to_numpy() + 60.0)
    frame["expected_conv"] = np.log1p(frame["exposure"].to_numpy() * np.minimum(frame["conversion"], 5.0))
    return frame


def make(split):
    """Pool + features (+ labels when the split has an answer key)."""
    scores = component_scores(split)
    chan = channel_affinity(split)
    pool = build_pool(split, scores, chan)
    frame = build_features(split, pool, scores, chan)
    frame = add_conversion(split, frame, pool)
    if split.truth is not None:
        frame["label"] = labels(split, frame)
    return frame, scores, chan
