"""Every hold-out number in the report, from one run per answer key.

  --target watched  (default) the answer key the competition uses: a user's
                    five shows with the most airings watched for >=80%.
                    Writes experiments/results_watched.json.
  --target seconds  the key of the first round (most seconds watched), plus
                    the LambdaRank hybrid that was submitted then.
                    Writes experiments/results_seconds.json.

Per run:
  1  hyperparameter sweeps per method, selected on MAP@5 of the hold-out cohort
     (the 1,2xx hold-out users who are also in the Kaggle test file)
  2  the comparison table at the selected settings: MAP@5 (cohort and all users),
     hit rate, runtime, and how many cohort users a method gives fewer than five
     non-zero scores - those slots are decided by the same popularity order
     everywhere (it breaks every tie, and a zero score ties)
  3  watched: ablations of the habit-x-schedule model
     seconds: the LambdaRank hybrid, fitted on the inner split (11-15 -> 16-20)
  4  environment

Hold-out: history weeks 11-20, answer from weeks 21-30. Validation only selects.

Run:  python experiments/run_validation.py [--target watched|seconds]
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: E402
import data  # noqa: E402
import exposure  # noqa: E402
import features  # noqa: E402
import habit  # noqa: E402
import hybrid  # noqa: E402
import methods  # noqa: E402
import metrics  # noqa: E402
from predict import align_categories  # noqa: E402


def timed(fn, *a, **kw):
    t = time.time()
    out = fn(*a, **kw)
    return out, round(time.time() - t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="watched", choices=["watched", "seconds"])
    args = ap.parse_args()
    common.TARGET = args.target
    out_path = common.RESULTS / f"results_{args.target}.json"

    split = data.make_split("valid")
    subu = data.submission_users()
    cohort = [u for u in split.eval_users if u in subu]
    # One popularity order serves as the reference baseline and as every method's fill.
    fill = habit.popularity_by_top5(split) if args.target == "watched" else methods.popularity_vector(split)
    res = {"target": args.target, "cohort_users": len(cohort), "all_users": len(split.eval_users),
           "candidates": int(len(split.candidates)),
           "fill": "history top-5 frequency" if args.target == "watched" else "total seconds"}

    def save():
        out_path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    def run(fn, **kw):
        S, sec = timed(fn, split, **kw)
        preds = methods.top_k(S, split, tie_break=fill)
        rows = {u: i for i, u in enumerate(split.eval_users)}
        return preds, {
            "map_cohort": metrics.mapk(split.truth, preds, users=cohort),
            "map_all": metrics.mapk(split.truth, preds),
            "hit_cohort": metrics.hit_rate(split.truth, preds, users=cohort),
            # A zero score is "no evidence" (ALS and SVD scores can be negative, which is still evidence).
            "fallback_users_cohort": int(sum((S[rows[u]] != 0).sum() < 5 for u in cohort)),
            "seconds": sec,
        }

    # ---- 1. sweeps --------------------------------------------------------
    ALL = ("channel", "category", "genre", "people", "title")
    grids = {
        "cbf": (methods.cbf, [
            ("log-minute profile, all fields", {"weighting": "log", "fields": ALL}),
            ("category + genres + people", {"fields": ("category", "genre", "people")}),
            ("+ channel", {"fields": ("channel", "category", "genre", "people")}),
            ("+ title", {"fields": ("category", "genre", "people", "title")}),
            ("all fields", {"fields": ALL})]),
        "rules": (methods.rules, [(f"basket>={mm} min, support>={sp}", {"min_minutes": mm, "min_support": sp})
                                  for mm in (5, 10, 30) for sp in (10, 20, 50)]),
        "mf_als": (methods.mf, [(f"positives>={mm} min, alpha={a}", {"min_minutes": mm, "alpha": a, "factors": 32})
                                for mm in (0, 120, 600, 1200) for a in (2, 10)]),
        "mf_svd": (methods.svd, [(f"k={k}", {"factors": k}) for k in (16, 32, 64, 128, 256)]),
        "ppr": (methods.ppr, [(f"{w}, alpha={a}", {"weight": w, "alpha": a})
                              for w in ("cosine", "count") for a in (0.15, 0.3, 0.5, 0.7, 0.85)]),
        "exposure": (exposure.exposure, [(f"{s // 60}-min slots", {"slot": s}) for s in (3600, 1800, 900)]),
    }
    if args.target == "watched":
        grids["habit"] = (habit.habit_schedule, [(f"half-life {h} wk, prior {p}", {"half_life": h, "prior": p})
                                                for h in (5.0, 10.0, 20.0) for p in (10.0, 30.0, 100.0)])
    res["sweeps"] = {}
    chosen = {}
    for name, (fn, grid) in grids.items():
        rows = []
        for label, kw in grid:
            _, m = run(fn, **kw)
            rows.append({"setting": label, "params": kw, **m})
            print(f"{name:9s} {label:36s} cohort={m['map_cohort']:.4f} all={m['map_all']:.4f}", flush=True)
        res["sweeps"][name] = rows
        chosen[name] = max(rows, key=lambda r: r["map_cohort"])["params"]
        save()

    # ---- 2. comparison at the selected settings ---------------------------
    table = {}
    entries = [
        ("popularity", lambda s: np.broadcast_to(fill.astype(np.float32), (len(s.eval_users), len(fill))).copy(), {}),
        ("history (re-watch baseline)", methods.history, {}),
        ("CBF", methods.cbf, chosen["cbf"]),
        ("association rules", methods.rules, chosen["rules"]),
        ("MF: implicit ALS", methods.mf, chosen["mf_als"]),
        ("MF: PureSVD", methods.svd, chosen["mf_svd"]),
        ("PPR (co-view graph)", methods.ppr, chosen["ppr"]),
        ("habit x schedule (seconds, channel-hours)", exposure.exposure, chosen["exposure"]),
    ]
    if args.target == "watched":
        # The submitted settings, not the sweep's best: the table describes the file sent to Kaggle.
        entries.append(("habit x schedule (watched-airing rate)", habit.habit_schedule, habit.SUBMITTED))
    for name, fn, kw in entries:
        _, m = run(fn, **kw)
        m["params"] = kw
        table[name] = m
        print(f"{name:44s} cohort={m['map_cohort']:.4f} all={m['map_all']:.4f} "
              f"fallback={m['fallback_users_cohort']} {m['seconds']}s", flush=True)
    res["methods"] = table
    save()

    # ---- 3a. habit ablations (watched key) ---------------------------------
    if args.target == "watched":
        base = dict(habit.SUBMITTED)
        res["habit_ablations"] = {}
        for name, kw in [("full model", {}),
                         ("no recency weighting", {"half_life": 1e9}),
                         ("no active-day correction", {"active_days": False}),
                         ("no target-week schedule", {"schedule": False}),
                         ("no prior (rate = q / e)", {"prior": 1e-9})]:
            _, m = run(habit.habit_schedule, **dict(base, **kw))
            res["habit_ablations"][name] = m
            print(f"  habit {name:28s} cohort={m['map_cohort']:.4f} all={m['map_all']:.4f}", flush=True)
        save()

    # ---- 3b. the first-round hybrid (seconds key) ----------------------------
    if args.target == "seconds":
        inner = data.make_split("inner")
        (f_inner, _, _), t_inner = timed(features.make, inner)
        (f_valid, _, _), t_valid = timed(features.make, split)
        align_categories(f_inner, f_valid)
        found = int((f_valid["label"] > 0).sum())
        total = sum(len(v) for v in split.truth.values())

        def fit_score(drop=(), binary=True):
            keep = [c for c in f_inner.columns if c not in drop]
            model, sec = timed(hybrid.fit, f_inner[keep], binary=binary)
            preds, filled = hybrid.rank(model, f_valid[keep], split)
            return model, {"map_cohort": metrics.mapk(split.truth, preds, users=cohort),
                           "map_all": metrics.mapk(split.truth, preds),
                           "trees": model.n_trees_, "fit_seconds": sec, "fallback_users": filled}

        model, full = fit_score()
        full.update({"pool_recall": found / total, "pool_per_user": len(f_valid) / f_valid["user_id"].nunique(),
                     "feature_seconds": t_inner + t_valid,
                     "importance": hybrid.importance(model).round(4).to_dict()})
        res["hybrid"] = full
        print(f"hybrid cohort={full['map_cohort']:.4f} all={full['map_all']:.4f}", flush=True)
        EXPOSURE = ("exposure", "exposure_rank", "exposure_hist", "conversion", "expected_conv")
        COURSE = tuple(f"{m}{s}" for m in ("cbf", "rules", "svd", "ppr") for s in ("", "_rank"))
        res["hybrid_ablations"] = {}
        for name, kw in [("graded labels", {"binary": False}),
                         ("without habit x schedule features", {"drop": EXPOSURE}),
                         ("without the four course methods", {"drop": COURSE})]:
            _, m = fit_score(**kw)
            res["hybrid_ablations"][name] = m
            print(f"  {name:36s} cohort={m['map_cohort']:.4f}", flush=True)
        save()

    # ---- 4. environment --------------------------------------------------------
    import implicit
    import lightgbm
    import pandas
    import scipy
    import sklearn
    res["environment"] = {
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "implicit": implicit.__version__,
        "lightgbm": lightgbm.__version__, "seeds": {"lightgbm": hybrid.SEED, "als/svd": 20260112},
    }
    save()
    print("saved", out_path)


if __name__ == "__main__":
    main()
