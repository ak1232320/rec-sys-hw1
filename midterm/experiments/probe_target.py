"""Which definition of "most preferred" did the organisers use?

The competition only says preferences were "calculated using the subsequent
12 weeks of historical views". The first submissions showed a gap the
validation did not predict: popularity scored 0.181 on the hold-out but 0.284
on the public test, close to the personalised methods. This script rebuilds the
hold-out answer key under several plausible definitions and scores the same
fixed rankings against each, to see which definition reproduces the pattern the
public scores show. It fits nothing; it only re-labels the hold-out.

Run:  python experiments/probe_target.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: E402
import data  # noqa: E402
import methods  # noqa: E402
import metrics  # noqa: E402


def answer_key(ov, value, k=common.TOP_K):
    per = ov.groupby(["user_id", "tv_show_id"]).agg(**value).reset_index()
    col = list(value)[0]
    per = per.sort_values(["user_id", col, "tv_show_id"], ascending=[True, False, True])
    top = per.groupby("user_id").head(k)
    return {u: g["tv_show_id"].tolist() for u, g in top.groupby("user_id", sort=False)}


def main():
    split = data.make_split("valid")
    subu = data.submission_users()
    cohort = [u for u in split.eval_users if u in subu]

    ov_all = pd.read_parquet(common.WORK / "overlaps.parquet")
    ov_all = data.in_weeks(ov_all, common.VALID_WEEKS)
    ov_all["day"] = ov_all["view_start"] // 86400
    ov_all["airing"] = ov_all["channel_id"].astype(np.int64) * 10**10 + ov_all["prog_start"]
    ov_all["share"] = ov_all["seconds"] / ov_all["prog_seconds"].clip(lower=1)
    ov = ov_all[ov_all["seconds"] >= common.WATCH_MIN_SECONDS]

    keys = {
        "seconds (>=60 s)": answer_key(ov, {"v": ("seconds", "sum")}),
        "views (>=60 s)": answer_key(ov, {"v": ("seconds", "size")}),
        "views (any overlap)": answer_key(ov_all, {"v": ("seconds", "size")}),
        "airings (>=60 s)": answer_key(ov, {"v": ("airing", "nunique")}),
        "airings (any overlap)": answer_key(ov_all, {"v": ("airing", "nunique")}),
        "days (>=60 s)": answer_key(ov, {"v": ("day", "nunique")}),
        "programme share (>=60 s)": answer_key(ov, {"v": ("share", "sum")}),
    }

    with open(common.WORK / "scores_valid.pkl", "rb") as fh:
        cached = pickle.load(fh)
    assert (cached["users"] == split.eval_users).all()
    pop = methods.popularity_vector(split)
    viewers = np.asarray((split.X[:, split.cand_cols] > 0).sum(axis=0)).ravel().astype(np.float32)
    rankings = {
        "popularity (seconds)": methods.top_k(methods.popularity(split), split, tie_break=pop),
        "popularity (viewers)": methods.top_k(np.broadcast_to(viewers, (len(split.eval_users), len(viewers))),
                                              split, tie_break=pop),
        "history": methods.top_k(methods.history(split), split, tie_break=pop),
        "exposure": methods.top_k(cached["scores"]["exposure"], split, tie_break=pop),
        "ppr": methods.top_k(cached["scores"]["ppr"], split, tie_break=pop),
    }

    out = {}
    print(f"{'answer key':28s}" + "".join(f"{m:>22s}" for m in rankings))
    for kname, key in keys.items():
        users = [u for u in cohort if u in key]
        row = {m: metrics.mapk(key, preds, users=users) for m, preds in rankings.items()}
        out[kname] = row
        print(f"{kname:28s}" + "".join(f"{v:22.4f}" for v in row.values()), flush=True)
    out["public_test"] = {"popularity (seconds)": 0.28361, "exposure": 0.29113, "ppr": 0.29038}
    (common.RESULTS / "probe_target.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
