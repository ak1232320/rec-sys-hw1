"""Measure the facts the method choice depends on, and save them for the report.

Run:  python experiments/profile_data.py   (after src/ingest.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: E402
import data  # noqa: E402


def window(frame, weeks):
    lo, hi = weeks
    return frame[(frame["week"] >= lo) & (frame["week"] <= hi)]


def top_k(inter, k=common.TOP_K):
    per = inter.groupby(["user_id", "tv_show_id"], as_index=False)["seconds"].sum()
    per = per.sort_values(["user_id", "seconds", "tv_show_id"], ascending=[True, False, True])
    return per.groupby("user_id").head(k)


def main():
    epg = pd.read_parquet(common.WORK / "epg.parquet")
    log = pd.read_parquet(common.WORK / "log.parquet")
    inter = data.interactions()
    sub = pd.read_csv(common.SAMPLE_SUBMISSION, dtype={"user_id": "uint64"})
    out = {}

    # --- how much of the grid, and of viewing, carries a programme id ------
    epg_raw = pd.concat([pd.read_csv(path, usecols=["tv_show_id", "duration"]).assign(part=label)
                         for label, path in common.EPG_FILES.items()])
    lab = epg_raw["tv_show_id"] != 0
    out["schedule_rows_labelled"] = {part: float(g["tv_show_id"].ne(0).mean())
                                     for part, g in epg_raw.groupby("part")}
    out["schedule_rows_labelled_all"] = float(lab.mean())
    out["schedule_airtime_labelled_all"] = float(epg_raw.loc[lab, "duration"].sum() / epg_raw["duration"].sum())
    ov_all = pd.read_parquet(common.WORK / "overlaps.parquet", columns=["seconds"])
    out["watched_seconds_on_labelled_programmes"] = float(
        ov_all["seconds"].sum() / log["duration"].sum())
    out["categories"] = int(epg["tv_show_category"].nunique())
    out["channels_in_schedule"] = int(epg["channel_id"].nunique())

    # --- who we are asked about --------------------------------------------
    sub_users = set(sub["user_id"])
    log_users = set(log["user_id"])
    sessions = log.groupby("user_id").size()
    out["submission_users"] = len(sub_users)
    out["submission_users_in_log"] = len(sub_users & log_users)
    out["log_users"] = len(log_users)
    s = sessions[sessions.index.isin(sub_users)]
    out["submission_sessions_quantiles"] = {q: float(s.quantile(q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)}
    last_week = log.groupby("user_id")["week"].max()
    out["submission_users_active_in_weeks_21_30"] = int((last_week[last_week.index.isin(sub_users)] >= 21).sum())
    out["submission_users_active_in_weeks_11_20"] = int(
        log[log["week"] <= 20].loc[lambda f: f["user_id"].isin(sub_users), "user_id"].nunique())

    # --- do the shows of a future window already exist in the past? ----------
    real = epg[epg["tv_show_id"] != 0]
    shows = {part: set(real.loc[real["part"] == part, "tv_show_id"]) for part in common.EPG_FILES}
    out["shows_per_slice"] = {k: len(v) for k, v in shows.items()}
    out["test_shows_seen_in_11_30_schedule"] = len(shows["31-42"] & (shows["11-20"] | shows["21-30"]))
    out["valid_shows_seen_in_11_20_schedule"] = len(shows["21-30"] & shows["11-20"])

    # Airtime of the future window that belongs to shows already broadcast before.
    air = real.assign(sec=(real["end"] - real["start"]).dt.total_seconds())
    test_air = air[air["part"] == "31-42"]
    out["test_airtime_share_on_returning_shows"] = float(
        test_air.loc[test_air["tv_show_id"].isin(shows["11-20"] | shows["21-30"]), "sec"].sum()
        / test_air["sec"].sum())

    # --- the validation analogue of the target ------------------------------
    past = window(inter, common.TRAIN_WEEKS)
    future = window(inter, common.VALID_WEEKS)
    truth = top_k(future)
    past_pairs = set(zip(past["user_id"], past["tv_show_id"]))
    past_shows = set(past["tv_show_id"])
    hits_user = np.mean([(u, s) in past_pairs for u, s in zip(truth["user_id"], truth["tv_show_id"])])
    hits_show = np.mean([s in past_shows for s in truth["tv_show_id"]])
    out["valid_truth_rows"] = int(len(truth))
    out["valid_truth_users"] = int(truth["user_id"].nunique())
    out["valid_truth_already_watched_by_same_user"] = float(hits_user)
    out["valid_truth_show_existed_in_train"] = float(hits_show)
    out["valid_truth_shows_in_21_30_schedule"] = float(np.mean(truth["tv_show_id"].isin(shows["21-30"])))
    sub_truth = truth[truth["user_id"].isin(sub_users)]
    out["valid_truth_users_from_submission"] = int(sub_truth["user_id"].nunique())

    # How concentrated is the target? (popularity is the obvious baseline)
    counts = truth["tv_show_id"].value_counts()
    out["valid_truth_distinct_shows"] = int(len(counts))
    out["valid_truth_share_top10_shows"] = float(counts.head(10).sum() / counts.sum())
    out["valid_truth_shows_per_user_hist"] = truth.groupby("user_id").size().value_counts().sort_index().to_dict()

    # --- channel loyalty: does a user's channel mix carry over? -------------
    ch_past = (window(log, common.TRAIN_WEEKS).groupby(["user_id", "channel_id"])["duration"].sum())
    ch_future = (window(log, common.VALID_WEEKS).groupby(["user_id", "channel_id"])["duration"].sum())
    top_past = ch_past.groupby(level=0).idxmax().map(lambda t: t[1])
    top_future = ch_future.groupby(level=0).idxmax().map(lambda t: t[1])
    both = top_past.index.intersection(top_future.index)
    out["favourite_channel_unchanged_21_30_vs_11_20"] = float((top_past[both] == top_future[both]).mean())

    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    (common.RESULTS / "profile.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
