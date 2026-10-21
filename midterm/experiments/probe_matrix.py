"""Rankings that differ only in how history is counted, scored under each answer key.

"watched" = airings the user covered for >=80% (the competition's rule, see
src/habit.py). If each "history by X" ranking does best under answer key X,
then submitting two or three of them to Kaggle tells which key the organisers
used: the season shift between training and test weeks affects all of them
alike, but the aggregation does not.

Run:  python experiments/probe_matrix.py   (after experiments/probe_target.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: E402
import data  # noqa: E402
import habit  # noqa: E402
import methods  # noqa: E402
import metrics  # noqa: E402
from probe_target import answer_key  # noqa: E402

AGG = {
    "seconds": {"v": ("seconds", "sum")},
    "views": {"v": ("seconds", "size")},
    "airings": {"v": ("airing", "nunique")},
    "days": {"v": ("day", "nunique")},
}


def overlaps(weeks):
    ov = pd.read_parquet(common.WORK / "overlaps.parquet")
    ov = data.in_weeks(ov, weeks)
    ov = ov[ov["seconds"] >= common.WATCH_MIN_SECONDS].copy()
    ov["day"] = ov["view_start"] // 86400
    ov["airing"] = ov["channel_id"].astype(np.int64) * 10**10 + ov["prog_start"]
    return ov


def history_by(split, ov_hist, how):
    if how == "watched":
        w = habit.watched_airings(split.history_weeks)
        per = w.groupby(["user_id", "tv_show_id"]).size().rename("v").reset_index()
    else:
        per = ov_hist.groupby(["user_id", "tv_show_id"]).agg(**AGG[how]).reset_index()
    per = per[per["tv_show_id"].isin(set(split.candidates))]
    rows = per["user_id"].map({u: i for i, u in enumerate(split.eval_users)})
    cols = per["tv_show_id"].map({s: i for i, s in enumerate(split.candidates)})
    keep = rows.notna()
    S = np.zeros((len(split.eval_users), len(split.candidates)), dtype=np.float32)
    S[rows[keep].astype(int).to_numpy(), cols[keep].astype(int).to_numpy()] = per.loc[keep, "v"].to_numpy()
    return S


def main():
    split = data.make_split("valid")
    subu = data.submission_users()
    cohort = [u for u in split.eval_users if u in subu]
    pop = methods.popularity_vector(split)
    ov_hist, ov_tgt = overlaps(common.TRAIN_WEEKS), overlaps(common.VALID_WEEKS)
    keys = {name: answer_key(ov_tgt, agg) for name, agg in AGG.items()}
    keys["watched"] = habit.answer_key(common.VALID_WEEKS)

    out = {}
    print(f"{'ranking':22s}" + "".join(f"{'key=' + k:>14s}" for k in keys))
    for how in list(AGG) + ["watched"]:
        preds = methods.top_k(history_by(split, ov_hist, how), split, tie_break=pop)
        row = {k: metrics.mapk(key, preds, users=[u for u in cohort if u in key]) for k, key in keys.items()}
        out[f"history by {how}"] = row
        print(f"{'history by ' + how:22s}" + "".join(f"{v:14.4f}" for v in row.values()), flush=True)
    (common.RESULTS / "probe_matrix.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
