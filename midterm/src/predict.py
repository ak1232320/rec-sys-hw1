"""Write the Kaggle submissions.

Default - the designated submission, submissions/submission_habit.csv: the
habit-x-schedule rate model (src/habit.py) scored on history weeks 11-30
against the weeks 31-42 schedule, with popularity filling empty slots. Nothing
from weeks 31-42 enters except the schedule; the viewing log ends at week 30.

--first-submission - the four files of the first round (LambdaRank hybrid and
three single methods), built with preference counted in seconds. The hybrid is
fitted on (history 11-20 -> answer 21-30) and early-stopped on a fifth of those
users. Rerunning reproduces the files that were sent, byte for byte (md5
checked 2026-10-22). Refs and returned scores: submissions/kaggle_scores.json.

Run:  python src/predict.py [--first-submission] [--out DIR]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import data  # noqa: E402
import features  # noqa: E402
import hybrid  # noqa: E402
import habit  # noqa: E402
import methods  # noqa: E402


def align_categories(*frames):
    cats = sorted(set().union(*[set(f["cat"].astype("category").cat.categories) for f in frames]))
    for f in frames:
        f["cat"] = f["cat"].astype("category").cat.set_categories(cats)


def write_submission(preds, path):
    template = pd.read_csv(common.SAMPLE_SUBMISSION, dtype={"user_id": "uint64"})
    rows = []
    for u in template["user_id"]:
        ids = preds[u]
        assert len(ids) == common.TOP_K and len(set(ids)) == common.TOP_K and 0 not in ids, (u, ids)
        rows.append(" ".join(str(int(s)) for s in ids))
    out = pd.DataFrame({"user_id": template["user_id"].astype(str), "tv_show_id": rows})
    out.to_csv(path, index=False, lineterminator="\n")
    return out


def final_habit(out_dir):
    t0 = time.time()
    test = data.make_split("test")
    fill = habit.popularity_by_top5(test)
    S = habit.habit_schedule(test, **habit.SUBMITTED)
    preds = methods.top_k(S, test, tie_break=fill)
    write_submission(preds, out_dir / "submission_habit.csv")
    short = int(((S > 0).sum(axis=1) < 5).sum())
    empty = int(((S > 0).sum(axis=1) == 0).sum())
    log = {"users": len(test.eval_users), "users_with_fewer_than_5_scored_shows": short,
           "users_with_no_scored_show": empty, "runtime_seconds": round(time.time() - t0, 1)}
    (out_dir / "predict_habit_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"habit x schedule: {short} users needed popularity fill ({empty} entirely), "
          f"-> {out_dir / 'submission_habit.csv'} ({log['runtime_seconds']}s)")


def first_submission(out_dir, single=("popularity", "exposure", "ppr")):
    common.TARGET = "seconds"          # the first round counted preference in seconds
    t0 = time.time()
    log = {}

    print("fitting split: history 11-20 -> answer 21-30")
    fit_split = data.make_split("valid")
    fit_frame, _, _ = features.make(fit_split)
    print("scoring split: history 11-30 -> schedule 31-42")
    test = data.make_split("test")
    test_frame, test_scores, _ = features.make(test)
    align_categories(fit_frame, test_frame)

    model = hybrid.fit(fit_frame, binary=True)
    preds, filled = hybrid.rank(model, test_frame, test)
    out = write_submission(preds, out_dir / "submission_hybrid.csv")
    log["hybrid"] = {"trees": model.n_trees_, "fallback_users": filled,
                     "pool_pairs": int(len(test_frame)),
                     "pool_per_user": float(len(test_frame) / test_frame["user_id"].nunique())}
    print(f"  hybrid: {model.n_trees_} trees, {filled} users needed popularity fill, "
          f"{len(out)} rows -> {out_dir / 'submission_hybrid.csv'}")

    pop = methods.popularity_vector(test)
    for name in single:
        S = test_scores.get(name)
        if S is None:
            S = methods.METHODS[name](test)
        p = methods.top_k(S, test, tie_break=pop)
        write_submission(p, out_dir / f"submission_{name}.csv")
        positive = (S > 0).sum(axis=1)
        log[name] = {"users_with_fewer_than_5_positive_scores": int((positive < 5).sum())}
        print(f"  {name}: {out_dir / f'submission_{name}.csv'}")

    test_pairs = pd.DataFrame([(u, s, r + 1) for u, ids in preds.items() for r, s in enumerate(ids)],
                              columns=["user_id", "tv_show_id", "rank"])
    test_pairs["user_id"] = test_pairs["user_id"].astype(str)
    test_pairs.to_csv(out_dir / "hybrid_ranked_pairs.csv", index=False)
    log["runtime_seconds"] = round(time.time() - t0, 1)
    (out_dir / "predict_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-submission", action="store_true",
                    help="rebuild the four first-round files instead of the designated one")
    ap.add_argument("--out", default=str(common.SUBMISSIONS))
    args = ap.parse_args()
    common.ensure_dirs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.first_submission:
        first_submission(out_dir)
    else:
        final_habit(out_dir)


if __name__ == "__main__":
    main()
