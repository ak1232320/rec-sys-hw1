"""Checks for the midterm pipeline. Each prints PASS/FAIL; the script exits 1 on any failure.

  1  unit self-checks: interval join, MAP@K (Kaggle reference cases), slot binning
  2  user ids survive parsing exactly; all 1,260 submission users are in the log
  3  the join is sane: overlaps never exceed the view or the programme
  4  splits respect time: history, target and answer weeks never mix
  5  no leakage: the first-round hybrid's hold-out features are identical when
     weeks 21-30 are deleted from the log
  6  no leakage: the final model's hold-out scores are identical when weeks
     21-30 are deleted from the log
  7  both submissions are well-formed (1,260 rows, 5 distinct ids on air in 31-42)
  8  the answer key follows the rule: an airing counts once covered >=80%,
     coverage is capped at 1, <=5 distinct shows per user, all on air
  9  (--repro) both rounds' submissions rebuild byte for byte

Run:  python tests/check_pipeline.py [--skip-leakage] [--repro]
"""
import dataclasses
import hashlib
import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402
import data  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail or ""))
        print(f"PASS  {name}  {detail or ''}", flush=True)
    except AssertionError as exc:
        RESULTS.append((name, False, str(exc)))
        print(f"FAIL  {name}  {exc}", flush=True)


def unit_self_checks():
    import exposure
    import intervals
    import metrics
    intervals._self_check()
    metrics._self_check()
    exposure._self_check()
    return "intervals, metrics, exposure"


def ids_exact():
    log = pd.read_parquet(common.WORK / "log.parquet", columns=["user_id"])
    text = pd.read_csv(common.LOG_FILE, usecols=["user_id"], dtype=str, nrows=300_000)["user_id"]
    parsed = log["user_id"].iloc[:len(text)].astype(str).to_numpy()
    bad = int((parsed != text.to_numpy()).sum())
    assert bad == 0, f"{bad} ids differ from the raw text"
    wanted = set(pd.read_csv(common.SAMPLE_SUBMISSION, dtype={"user_id": "uint64"})["user_id"])
    found = len(wanted & set(log["user_id"].unique()))
    assert found == 1260, f"only {found} of 1260 submission users in the log"
    return f"300,000 ids exact; {found}/1260 submission users present"


def join_sane():
    ov = pd.read_parquet(common.WORK / "overlaps.parquet",
                         columns=["seconds", "prog_seconds", "view_seconds", "tv_show_id"])
    assert (ov["seconds"] > 0).all(), "non-positive overlap"
    assert (ov["seconds"] <= ov["prog_seconds"]).all(), "overlap longer than the programme"
    assert (ov["seconds"] <= ov["view_seconds"]).all(), "overlap longer than the view"
    assert (ov["tv_show_id"] != 0).all(), "tv_show_id 0 kept as an item"
    return f"{len(ov):,} overlaps within both bounds; no id-0 items"


def splits_respect_time():
    out = []
    for name, (hlo, hhi), (tlo, thi) in [("inner", common.INNER_HISTORY, common.INNER_TARGET),
                                         ("valid", common.TRAIN_WEEKS, common.VALID_WEEKS),
                                         ("test", common.FULL_WEEKS, common.TEST_WEEKS)]:
        s = data.make_split(name)
        assert s.hist["week"].between(hlo, hhi).all(), f"{name}: interaction outside history"
        assert s.hist_log["week"].between(hlo, hhi).all(), f"{name}: session outside history"
        assert s.epg_hist["week"].between(hlo, hhi).all(), f"{name}: history schedule leaks"
        assert s.epg_target["week"].between(tlo, thi).all(), f"{name}: target schedule outside window"
        assert hhi < tlo, f"{name}: history overlaps target"
        assert set(s.candidates) <= set(s.epg_target["tv_show_id"]), f"{name}: candidate not on air"
        out.append(f"{name} {hlo}-{hhi}->{tlo}-{thi}")
    log = pd.read_parquet(common.WORK / "log.parquet", columns=["week"])
    assert log["week"].max() <= 30, "the log reaches into the test weeks"
    return "; ".join(out) + "; log ends at week 30"


def no_leakage():
    """Build hold-out features twice: normally, and with every row after week 20 deleted."""
    import features

    a = data.make_split("valid")
    fa, _, _ = features.make(a)

    epg, log, inter, sub = data.load_tables()
    data._CACHE["tables"] = (epg, log[log["week"] <= 20], inter[inter["week"] <= 20], sub)
    original = data.overlap_rows

    def truncated(weeks, min_seconds=common.WATCH_MIN_SECONDS):
        rows = original(weeks, min_seconds)
        return rows[rows["week"] <= 20]

    data.overlap_rows = truncated
    try:
        # The answer key needs weeks 21-30, so the second split borrows the first
        # one's users and key; everything a feature can read is rebuilt.
        import dataclasses
        b = dataclasses.replace(a, hist=data.in_weeks(inter[inter["week"] <= 20], a.history_weeks),
                                hist_log=data.in_weeks(log[log["week"] <= 20], a.history_weeks))
        b.X = data.user_show_matrix(b.hist, b.uni)
        fb, _, _ = features.make(b)
    finally:
        data.overlap_rows = original
        data._CACHE.pop("tables", None)

    cols = [c for c in fa.columns if c != "label"]
    assert len(fa) == len(fb), f"pool size differs: {len(fa)} vs {len(fb)}"
    fa = fa.sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)
    fb = fb.sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)
    for c in cols:
        x, y = fa[c], fb[c]
        if str(x.dtype) == "category":
            same = (x.astype(str) == y.astype(str)).all()
        else:
            same = np.allclose(x.to_numpy(float), y.to_numpy(float), equal_nan=True)
        assert same, f"feature {c} changes when weeks 21-30 are removed"
    return f"{len(cols)} features x {len(fa):,} pairs identical"


def no_leakage_final():
    """Habit x schedule on the hold-out, with every overlap after week 20 removed at the source."""
    import habit

    a = data.make_split("valid")
    sa = habit.habit_schedule(a)
    original = pd.read_parquet

    def truncated(path, *args, **kw):
        frame = original(path, *args, **kw)
        return frame[frame["week"] <= 20] if "week" in frame.columns else frame

    pd.read_parquet = truncated
    try:
        b = dataclasses.replace(a, hist_log=a.hist_log[a.hist_log["week"] <= 20])
        sb = habit.habit_schedule(b)
    finally:
        pd.read_parquet = original
    assert np.array_equal(sa, sb), "scores change when weeks 21-30 are removed"
    return f"{sa.shape[0]:,} users x {sa.shape[1]:,} candidates identical ({int((sa > 0).sum()):,} non-zero)"


def submissions_ok():
    return "; ".join(submission_ok(common.SUBMISSIONS / f) for f in ("submission_habit.csv",
                                                                      "submission_hybrid.csv"))


def submission_ok(path):
    assert path.exists(), f"{path.name} missing"
    sub = pd.read_csv(path, dtype=str)
    tpl = pd.read_csv(common.SAMPLE_SUBMISSION, dtype=str)
    assert list(sub.columns) == ["user_id", "tv_show_id"], f"header {list(sub.columns)}"
    assert len(sub) == 1260, f"{len(sub)} rows"
    assert (sub["user_id"].to_numpy() == tpl["user_id"].to_numpy()).all(), "user order differs from template"
    epg = pd.read_parquet(common.WORK / "epg.parquet", columns=["tv_show_id", "week"])
    on_air = set(data.in_weeks(epg, common.TEST_WEEKS)["tv_show_id"]) - {0}
    for ids in sub["tv_show_id"]:
        parts = [int(x) for x in ids.split(" ")]
        assert len(parts) == 5 and len(set(parts)) == 5, f"not 5 distinct ids: {ids}"
        assert all(p in on_air for p in parts), f"id not on air in weeks 31-42: {ids}"
    return f"{path.name}: 1260 x 5 distinct, on air"


def answer_key_ok():
    import habit

    cov = habit.airing_coverage(common.VALID_WEEKS)
    assert cov["coverage"].between(0, 1).all(), "coverage outside [0, 1]"
    w = habit.watched_airings(common.VALID_WEEKS)
    assert (w["coverage"] >= 0.8).all() and len(w) == int((cov["coverage"] >= 0.8).sum())
    s = data.make_split("valid")
    on_air = set(s.epg_target["tv_show_id"])
    sizes = [len(v) for v in s.truth.values()]
    assert max(sizes) <= 5 and min(sizes) >= 1, "answer key sizes out of range"
    assert all(len(set(v)) == len(v) for v in s.truth.values()), "duplicate in answer key"
    stray = sum(x not in on_air for v in s.truth.values() for x in v)
    assert stray == 0, f"{stray} answer-key shows not on the target schedule"
    return (f"{len(s.truth):,} users, key '{common.TARGET}'; {len(w):,} of {len(cov):,} "
            f"(user, airing) pairs reach 80%")


def reproducible():
    import predict

    def md5(path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        predict.final_habit(Path(tmp))
        predict.first_submission(Path(tmp))
        common.TARGET = "watched"
        names = ("submission_habit.csv", "submission_hybrid.csv", "submission_popularity.csv",
                 "submission_exposure.csv", "submission_ppr.csv")
        for name in names:
            assert md5(Path(tmp) / name) == md5(common.SUBMISSIONS / name), f"{name} differs"
    return f"{len(names)} files identical to the ones sent to Kaggle"


def main():
    check("1 unit self-checks", unit_self_checks)
    check("2 ids exact", ids_exact)
    check("3 join sane", join_sane)
    check("4 splits respect time", splits_respect_time)
    if "--skip-leakage" not in sys.argv:
        common.TARGET = "seconds"
        check("5 no leakage (first-round hybrid)", no_leakage)
        common.TARGET = "watched"
        check("6 no leakage (final model)", no_leakage_final)
    check("7 submissions well-formed", submissions_ok)
    check("8 answer key follows the 80% rule", answer_key_ok)
    if "--repro" in sys.argv:
        check("9 submissions rebuild byte for byte", reproducible)
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
