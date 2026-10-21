"""The two users of the report: every method's top-5 next to what they did.

Chosen from the hold-out cohort (users who are also in the Kaggle test file
and have an answer in weeks 21-30), by how many airings they watched to 80%
in the history weeks 11-20:

  heavy  the user at the 90th percentile - a settled habit viewer
  light  the user at the 10th percentile - a few watched airings, so co-view
         and content signals have to do more of the work

Deterministic: the user nearest each percentile, ties to the lower id. For each
user the script lists their history top-5, the answer (weeks 21-30), and each
method's five ids with title and category, marking hits.

Run:  python experiments/two_users.py   (writes experiments/two_users.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: E402
import data  # noqa: E402
import exposure  # noqa: E402
import habit  # noqa: E402
import methods  # noqa: E402


def pick(counts, q):
    target = counts.quantile(q)
    gap = (counts - target).abs()
    return gap[gap == gap.min()].index.min()


def main():
    common.TARGET = "watched"
    split = data.make_split("valid")
    subu = data.submission_users()
    cohort = [u for u in split.eval_users if u in subu]

    hist_w = habit.watched_airings(split.history_weeks)
    counts = hist_w.groupby("user_id").size().reindex(cohort).fillna(0)
    users = {"heavy": pick(counts, 0.9), "light": pick(counts, 0.1)}

    epg = pd.concat([split.epg_hist, split.epg_target])
    meta = epg.drop_duplicates("tv_show_id", keep="last").set_index("tv_show_id")[
        ["tv_show_title", "tv_show_category", "channel_title"]]
    fill = habit.popularity_by_top5(split)

    with open(common.RESULTS / f"results_{common.TARGET}.json", encoding="utf-8") as fh:
        params = {k: v["params"] for k, v in json.load(fh)["methods"].items()}
    runs = {
        "popularity": np.broadcast_to(fill.astype(np.float32), (len(split.eval_users), len(fill))).copy(),
        "CBF": methods.cbf(split, **params["CBF"]),
        "association rules": methods.rules(split, **params["association rules"]),
        "MF: PureSVD": methods.svd(split, **params["MF: PureSVD"]),
        "PPR (co-view graph)": methods.ppr(split, **params["PPR (co-view graph)"]),
        "habit x schedule (final)": habit.habit_schedule(split, **params["habit x schedule (watched-airing rate)"]),
    }
    row_of = {u: i for i, u in enumerate(split.eval_users)}

    def describe(s):
        m = meta.loc[s] if s in meta.index else None
        return {"id": int(s),
                "title": None if m is None else str(m["tv_show_title"]),
                "category": None if m is None or pd.isna(m["tv_show_category"]) else str(m["tv_show_category"]),
                "channel": None if m is None else str(m["channel_title"])}

    out = {}
    for role, u in users.items():
        i = row_of[u]
        h = hist_w[hist_w["user_id"] == u].groupby("tv_show_id").size().sort_values(ascending=False)
        truth = split.truth[u]
        entry = {
            "user_id": str(u), "role": role,
            "watched_airings_11_20": int(counts[u]),
            "distinct_shows_11_20": int(len(h)),
            "history_top5": [dict(describe(s), watched=int(n)) for s, n in h.head(5).items()],
            "answer_21_30": [describe(s) for s in truth],
            "lists": {},
        }
        for name, S in runs.items():
            order = np.lexsort((split.candidates, -fill, -S[i]))[:5]
            ids = split.candidates[order]
            scored = int((S[i] > 0).sum())
            entry["lists"][name] = {
                "items": [dict(describe(s), hit=bool(s in truth), seen_before=bool(s in h.index),
                               filled=bool(S[i, j] <= 0)) for s, j in zip(ids, order)],
                "hits": int(sum(s in truth for s in ids)),
                "shows_with_a_score": scored,
            }
        out[role] = entry
        print(f"{role}: user {u}, {entry['watched_airings_11_20']} watched airings, "
              f"{entry['distinct_shows_11_20']} shows; answer {truth}")
        for name, l in entry["lists"].items():
            print(f"   {name:26s} hits={l['hits']}  " + ", ".join(
                f"{x['id']}{'*' if x['hit'] else ''}" for x in l["items"]))
    (common.RESULTS / "two_users.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
