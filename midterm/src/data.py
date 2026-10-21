"""Splits: what a model may see, what it is asked to rank, and the answer.

A split is defined by two week ranges. Everything a method learns from comes
from the *history* weeks; the *target* weeks contribute only their broadcast
schedule (which shows will be on air, where and for how long - metadata the
task allows) and, on the validation split, the answer key. Nothing observed in
the target weeks' viewing log ever reaches a method.

  inner: history 11-15, target 16-20  (fits the hybrid's ranker)
  valid: history 11-20, target 21-30  (the hold-out: selection only)
  test:  history 11-30, target 31-42  (answer held by Kaggle)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common


@dataclass
class Universe:
    """Fixed row/column indices shared by every split, so matrices line up."""
    users: np.ndarray
    shows: np.ndarray
    user_index: dict = field(repr=False)
    show_index: dict = field(repr=False)


@dataclass
class Split:
    name: str
    history_weeks: tuple
    target_weeks: tuple
    uni: Universe
    hist: pd.DataFrame          # (user, show, week) seconds inside the history weeks
    hist_log: pd.DataFrame      # raw sessions inside the history weeks
    epg_hist: pd.DataFrame      # real (non-zero) broadcasts inside the history weeks
    epg_target: pd.DataFrame    # real broadcasts inside the target weeks
    candidates: np.ndarray      # show ids on air in the target weeks
    eval_users: np.ndarray      # users to rank
    truth: dict | None          # user -> ordered top-5 show ids (None on test)
    X: sp.csr_matrix = None     # users x shows, history seconds

    @property
    def cand_cols(self):
        return np.array([self.uni.show_index[s] for s in self.candidates])

    @property
    def eval_rows(self):
        return np.array([self.uni.user_index[u] for u in self.eval_users])


_CACHE = {}


def interactions(min_seconds=common.WATCH_MIN_SECONDS):
    """(user, show, week) -> seconds, from overlaps of at least `min_seconds`."""
    ov = pd.read_parquet(common.WORK / "overlaps.parquet",
                         columns=["user_id", "tv_show_id", "week", "seconds"])
    ov = ov[ov["seconds"] >= min_seconds]
    inter = (ov.groupby(["user_id", "tv_show_id", "week"], as_index=False, sort=False)
             .agg(seconds=("seconds", "sum"), sessions=("seconds", "size")))
    inter["seconds"] = inter["seconds"].astype("int64")
    return inter


def load_tables():
    if "tables" not in _CACHE:
        epg = pd.read_parquet(common.WORK / "epg.parquet")
        log = pd.read_parquet(common.WORK / "log.parquet")
        inter = interactions()
        sub = pd.read_csv(common.SAMPLE_SUBMISSION, dtype={"user_id": "uint64"})
        _CACHE["tables"] = (epg, log, inter, sub)
    return _CACHE["tables"]


def universe():
    if "uni" not in _CACHE:
        epg, log, inter, _ = load_tables()
        users = np.sort(log["user_id"].unique())
        shows = np.sort(np.union1d(epg.loc[epg["tv_show_id"] != 0, "tv_show_id"].unique(),
                                   inter["tv_show_id"].unique()))
        _CACHE["uni"] = Universe(users, shows,
                                 {u: i for i, u in enumerate(users)},
                                 {s: i for i, s in enumerate(shows)})
    return _CACHE["uni"]


def in_weeks(frame, weeks):
    lo, hi = weeks
    return frame[(frame["week"] >= lo) & (frame["week"] <= hi)]


def overlap_rows(weeks, min_seconds=common.WATCH_MIN_SECONDS):
    """Every watched overlap in a week range, with airing and day identities."""
    ov = pd.read_parquet(common.WORK / "overlaps.parquet",
                         columns=["user_id", "tv_show_id", "week", "seconds",
                                  "channel_id", "prog_start", "view_start"])
    ov = in_weeks(ov, weeks)
    ov = ov[ov["seconds"] >= min_seconds].copy()
    ov["day"] = ov["view_start"] // 86400
    ov["airing"] = ov["channel_id"].astype(np.int64) * 10**10 + ov["prog_start"]
    return ov


COUNTS = {
    "seconds": ("seconds", "sum"),
    "views": ("seconds", "size"),
    "airings": ("airing", "nunique"),
    "days": ("day", "nunique"),
}


def answer_key(weeks, how=None, k=common.TOP_K):
    """Each user's k most-preferred shows in a week range, counted by `how`."""
    how = how or common.TARGET
    if how == "watched":
        import habit
        return habit.answer_key(weeks, k)
    per = overlap_rows(weeks).groupby(["user_id", "tv_show_id"]).agg(v=COUNTS[how]).reset_index()
    per = per.sort_values(["user_id", "v", "tv_show_id"], ascending=[True, False, True])
    top = per.groupby("user_id").head(k)
    return {u: g["tv_show_id"].tolist() for u, g in top.groupby("user_id", sort=False)}


def truth_lists(inter, k=common.TOP_K):
    """Each user's k most-watched shows in a window, most-watched first."""
    per = inter.groupby(["user_id", "tv_show_id"], as_index=False)["seconds"].sum()
    per = per.sort_values(["user_id", "seconds", "tv_show_id"], ascending=[True, False, True])
    top = per.groupby("user_id").head(k)
    return {u: g["tv_show_id"].tolist() for u, g in top.groupby("user_id", sort=False)}


def user_show_matrix(inter, uni, value="seconds"):
    rows = inter["user_id"].map(uni.user_index).to_numpy()
    cols = inter["tv_show_id"].map(uni.show_index).to_numpy()
    vals = inter[value].to_numpy(dtype=np.float64)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(uni.users), len(uni.shows)))


def make_split(name):
    epg, log, inter, sub = load_tables()
    uni = universe()
    if name == "valid":
        history, target = common.TRAIN_WEEKS, common.VALID_WEEKS
    elif name == "inner":
        history, target = common.INNER_HISTORY, common.INNER_TARGET
    elif name == "test":
        history, target = common.FULL_WEEKS, common.TEST_WEEKS
    else:
        raise ValueError(name)

    real = epg[epg["tv_show_id"] != 0]
    epg_target = in_weeks(real, target)
    hist = in_weeks(inter, history)
    split = Split(
        name=name, history_weeks=history, target_weeks=target, uni=uni,
        hist=hist, hist_log=in_weeks(log, history),
        epg_hist=in_weeks(real, history), epg_target=epg_target,
        candidates=np.sort(epg_target["tv_show_id"].unique()),
        eval_users=None, truth=None,
    )
    split.X = user_show_matrix(hist, uni)

    if name in ("valid", "inner"):
        truth = answer_key(target)
        active = set(hist["user_id"].unique())
        # Ranked: everyone with both a past and an answer. Reported separately for
        # the 1,260 users Kaggle scores, since they are heavier viewers than average.
        split.truth = {u: t for u, t in truth.items() if u in active}
        split.eval_users = np.array(sorted(split.truth), dtype=np.uint64)
    else:
        split.eval_users = sub["user_id"].to_numpy(dtype=np.uint64)
    return split


def submission_users():
    return set(load_tables()[3]["user_id"])
