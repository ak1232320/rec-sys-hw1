"""Watched airings and the habit-x-schedule rate model.

What counts as watching. A user has watched an airing when the seconds they
spent on it - summed over every session that overlaps it - cover at least 80%
of the airing. This is the organisers' rule as relayed in material shared with
me (see the report); the answer key is then each user's five programmes with
the most watched airings in the window.

The model (idea from the same shared material; implemented here from its
description). For a user u, a show s and a channel c:

  q  = watched airings of s on c in the history weeks, each weighted by
       recency (the weight halves every `half_life` weeks)
  e  = airings of s on c that went out on days u was in the log at all, with
       the same weights - the chances u had; a gap in logging then lowers both
       q and e instead of looking like lost interest
  rate = min(1, q / (e + prior))  - how reliably u watches s when it airs;
       `prior` shrinks rates built on few chances towards 0
  score(u, s) = sum over c of rate * (airings of s on c in the target weeks)

i.e. the expected number of airings u will watch, if their rate holds and the
schedule is as published. Shows with no score are filled by popularity: how
often a show is in users' own history top-5.
"""
import numpy as np
import pandas as pd

import common
import data

WATCHED = 0.8
# The settings of the submitted model, as given in the shared description.
SUBMITTED = {"half_life": 10.0, "prior": 30.0}


def airing_coverage(weeks):
    """(user, airing) -> share of the airing covered by the user's sessions, capped at 1."""
    ov = pd.read_parquet(common.WORK / "overlaps.parquet",
                         columns=["user_id", "tv_show_id", "week", "seconds", "prog_seconds",
                                  "channel_id", "prog_start"])
    ov = data.in_weeks(ov, weeks)
    cov = (ov.groupby(["user_id", "tv_show_id", "channel_id", "prog_start"], as_index=False)
           .agg(seconds=("seconds", "sum"), prog_seconds=("prog_seconds", "first"), week=("week", "min")))
    cov["coverage"] = np.minimum(1.0, cov["seconds"] / cov["prog_seconds"].clip(lower=1))
    return cov


def watched_airings(weeks):
    cov = airing_coverage(weeks)
    return cov[cov["coverage"] >= WATCHED]


def answer_key(weeks, k=common.TOP_K):
    """Each user's k shows with the most watched airings, most first (ties: lower id)."""
    w = watched_airings(weeks)
    per = w.groupby(["user_id", "tv_show_id"]).size().rename("n").reset_index()
    per = per.sort_values(["user_id", "n", "tv_show_id"], ascending=[True, False, True])
    top = per.groupby("user_id").head(k)
    return {u: g["tv_show_id"].tolist() for u, g in top.groupby("user_id", sort=False)}


def popularity_by_top5(split):
    """Candidates scored by how many users have them in their own history top-5."""
    key = answer_key(split.history_weeks)
    counts = pd.Series([s for shows in key.values() for s in shows]).value_counts()
    return counts.reindex(split.candidates).fillna(0).to_numpy(dtype=np.float64)


def habit_schedule(split, half_life=10.0, prior=30.0, active_days=True, schedule=True, **_):
    """users x candidates: expected watched airings in the target weeks.

    The switches exist for the ablations in experiments/run_validation.py:
    `active_days=False` counts every airing as a chance, `schedule=False`
    drops the target-week airing count (score = sum of rates).
    """
    end = split.history_weeks[1]
    weight = lambda week: np.power(0.5, (end - week) / half_life)  # noqa: E731

    # q: recency-weighted watched airings per (user, show, channel)
    w = watched_airings(split.history_weeks)
    w = w[w["user_id"].isin(set(split.eval_users))]
    q = (w.assign(wt=weight(w["week"]))
         .groupby(["user_id", "tv_show_id", "channel_id"])["wt"].sum().rename("q").reset_index())

    # e: airings of the same (show, channel) on days the user was in the log
    air = split.epg_hist[["tv_show_id", "channel_id", "start", "week"]].copy()
    air["day"] = air["start"].values.astype("datetime64[D]").astype(np.int64)
    air["wt"] = weight(air["week"])
    per_day = air.groupby(["tv_show_id", "channel_id", "day"])["wt"].sum().reset_index()

    log = split.hist_log
    active = log.assign(day=log["start"].values.astype("datetime64[D]").astype(np.int64))[
        ["user_id", "day"]].drop_duplicates()
    active = active[active["user_id"].isin(set(q["user_id"]))]

    # Join only the (user, show, channel) triples that have a q: e is never needed elsewhere.
    sc = q[["user_id", "tv_show_id", "channel_id"]]
    chances = sc.merge(per_day, on=["tv_show_id", "channel_id"])
    if active_days:
        chances = chances.merge(active, on=["user_id", "day"])
    e = (chances.groupby(["user_id", "tv_show_id", "channel_id"])["wt"].sum()
         .rename("e").reset_index())
    q = q.merge(e, on=["user_id", "tv_show_id", "channel_id"], how="left")
    q["e"] = q["e"].fillna(0.0)
    q["rate"] = np.minimum(1.0, q["q"] / (q["e"] + prior))

    # Airings of each (show, channel) in the target weeks - the schedule, nothing else.
    future = (split.epg_target.groupby(["tv_show_id", "channel_id"]).size().rename("n_future").reset_index())
    q = q.merge(future, on=["tv_show_id", "channel_id"], how="inner")
    q["score"] = q["rate"] * (q["n_future"] if schedule else 1.0)
    per = q.groupby(["user_id", "tv_show_id"])["score"].sum().reset_index()

    rows = per["user_id"].map({u: i for i, u in enumerate(split.eval_users)}).to_numpy()
    cols = per["tv_show_id"].map({s: i for i, s in enumerate(split.candidates)}).to_numpy()
    S = np.zeros((len(split.eval_users), len(split.candidates)), dtype=np.float32)
    S[rows, cols] = per["score"].to_numpy()
    return S
