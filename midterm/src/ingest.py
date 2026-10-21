"""Turn the raw competition files into three parquet tables.

  work/epg.parquet           one row per broadcast (all three schedule slices)
  work/log.parquet           one row per viewing session
  work/interactions.parquet  (user, show, week) -> seconds watched

The interaction table is the join the task asks for: a session only names a
channel and a time span, so it is matched against every programme that channel
aired during that span (src/intervals.py) and credited with the overlap.
Programmes with tv_show_id 0 stay in the join - otherwise time spent on them
would leak onto the neighbouring real shows - and are dropped afterwards.

Run:  python src/ingest.py
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402
from intervals import check_schedule, overlap_join  # noqa: E402

EPG_COLUMNS = {
    "channel_id": "int32", "channel_title": "string", "start_time": "string",
    "duration": "int32", "tv_show_title": "string", "tv_show_id": "int64",
    "tv_show_category": "string", "tv_show_genre_1": "string",
    "tv_show_genre_2": "string", "tv_show_genre_3": "string",
    "year_of_production": "string", "director": "string", "actors": "string",
}


def load_epg():
    parts = []
    for label, path in common.EPG_FILES.items():
        frame = pd.read_csv(path, dtype=EPG_COLUMNS, encoding="utf-8")
        frame["start"] = pd.to_datetime(frame["start_time"], format="%d.%m.%Y %H:%M:%S")
        frame["end"] = frame["start"] + pd.to_timedelta(frame["duration"], unit="s")
        frame["part"] = label
        parts.append(frame.drop(columns=["start_time"]))
        print(f"  epg {label}: {len(frame):,} broadcasts, "
              f"{frame['start'].min()} .. {frame['end'].max()}")
    epg = pd.concat(parts, ignore_index=True)
    epg["week"] = common.iso_week(epg["start"])
    return epg


def clean_schedule(epg):
    """One grid per channel: sorted, deduplicated, no overlaps.

    Where two broadcasts overlap, the earlier one is cut at the later one's start
    - a channel shows one thing at a time, and the later listing is the one that
    actually began.
    """
    epg = epg.sort_values(["channel_id", "start", "end"]).drop_duplicates(
        ["channel_id", "start", "tv_show_id"], keep="last")
    epg = epg.drop_duplicates(["channel_id", "start"], keep="last").reset_index(drop=True)

    next_start = epg.groupby("channel_id")["start"].shift(-1)
    cut = next_start.notna() & (epg["end"] > next_start)
    report = {"overlaps_cut": int(cut.sum()), "rows": len(epg)}
    epg.loc[cut, "end"] = next_start[cut]
    epg = epg[epg["end"] > epg["start"]].reset_index(drop=True)
    return epg, report


def load_log():
    # The C parser, not pyarrow: user ids run up to 20 digits (past 2**63) and the
    # pyarrow engine routes them through float64, which rounds ~99% of them to a
    # neighbouring id - the user count barely changes, but almost nobody in the
    # submission file can be found in the log any more.
    frame = pd.read_csv(
        common.LOG_FILE,
        dtype={"user_id": "uint64", "vsetv_id": "int32", "duraton": "int64",
               "start_time": "string", "stop_time": "string"},
    )
    frame = frame.rename(columns={"vsetv_id": "channel_id", "duraton": "duration"})
    frame["start"] = pd.to_datetime(frame["start_time"], format="%Y-%m-%d %H:%M:%S")
    frame["end"] = pd.to_datetime(frame["stop_time"], format="%Y-%m-%d %H:%M:%S")
    frame = frame.drop(columns=["start_time", "stop_time"])
    frame["week"] = common.iso_week(frame["start"])
    return frame


def check_ids(log):
    """Fail loudly if user ids did not survive parsing exactly."""
    text = pd.read_csv(common.LOG_FILE, usecols=["user_id"], dtype=str, nrows=200_000)["user_id"]
    parsed = log["user_id"].iloc[:len(text)].astype(str).values
    assert (parsed == text.values).all(), "user_id lost precision while parsing the log"

    wanted = set(pd.read_csv(common.SAMPLE_SUBMISSION, dtype={"user_id": "uint64"})["user_id"])
    found = len(wanted & set(log["user_id"].unique()))
    print(f"  ids exact on the first {len(text):,} rows; "
          f"{found:,} of {len(wanted):,} submission users present in the log")
    assert found == len(wanted), "submission users missing from the log"


def to_seconds(series):
    return series.values.astype("datetime64[s]").astype(np.int64)


def join(log, epg):
    """Explode sessions over the programmes they overlap, channel by channel."""
    out = []
    unmatched_channels = sorted(set(log["channel_id"].unique()) - set(epg["channel_id"].unique()))
    grid_problems = {}
    for channel, views in log.groupby("channel_id", sort=True):
        grid = epg[epg["channel_id"] == channel]
        if grid.empty:
            continue
        ps, pe = to_seconds(grid["start"]), to_seconds(grid["end"])
        problems = check_schedule(ps, pe)
        if problems:
            grid_problems[int(channel)] = problems
        vi, pi, ov = overlap_join(to_seconds(views["start"]), to_seconds(views["end"]), ps, pe)
        if len(vi) == 0:
            continue
        out.append(pd.DataFrame({
            "user_id": views["user_id"].values[vi],
            "tv_show_id": grid["tv_show_id"].values[pi],
            "week": views["week"].values[vi],
            "seconds": ov.astype(np.int32),
            "prog_seconds": (pe[pi] - ps[pi]).astype(np.int32),
            "channel_id": np.full(len(vi), channel, dtype=np.int32),
            "prog_start": ps[pi],
            "view_start": to_seconds(views["start"])[vi],
            "view_seconds": (to_seconds(views["end"])[vi] - to_seconds(views["start"])[vi]).astype(np.int32),
        }))
    pairs = pd.concat(out, ignore_index=True)
    return pairs, unmatched_channels, grid_problems


def main():
    common.ensure_dirs()
    t0 = time.time()

    print("schedule")
    epg_raw = load_epg()
    epg, report = clean_schedule(epg_raw)
    print(f"  {len(epg_raw):,} rows -> {len(epg):,} after dedup; "
          f"{report['overlaps_cut']:,} overlapping listings trimmed")
    epg.to_parquet(common.WORK / "epg.parquet", index=False)

    print("viewing log")
    log = load_log()
    print(f"  {len(log):,} sessions, {log['user_id'].nunique():,} users, "
          f"{log['channel_id'].nunique():,} channels, {log['start'].min()} .. {log['end'].max()}")
    print(f"  weeks {log['week'].min()}..{log['week'].max()}; "
          f"duration column vs stop-start mismatches: "
          f"{int((log['duration'] != (log['end'] - log['start']).dt.total_seconds()).sum()):,}")
    check_ids(log)
    log.to_parquet(common.WORK / "log.parquet", index=False)

    print("join")
    pairs, unmatched, problems = join(log, epg)
    total_view = int(log["duration"].sum())
    print(f"  {len(pairs):,} session x programme rows; "
          f"{pairs['seconds'].sum() / total_view:.1%} of logged seconds land on a programme")
    print(f"  channels in the log with no schedule: {len(unmatched)} {unmatched[:20]}")
    if problems:
        print(f"  channels whose grid still has problems: {len(problems)}")

    zero = pairs["tv_show_id"] == 0
    print(f"  {pairs.loc[zero, 'seconds'].sum() / pairs['seconds'].sum():.1%} of matched seconds "
          f"are on tv_show_id 0 (irrelevant) - dropped")
    # Every single overlap on a labelled programme, so the "watched" threshold can
    # be chosen afterwards (src/data.py) instead of being baked in here.
    pairs[~zero].to_parquet(common.WORK / "overlaps.parquet", index=False)
    inter = (pairs[~zero]
             .groupby(["user_id", "tv_show_id", "week"], as_index=False, sort=False)
             .agg(seconds=("seconds", "sum"), sessions=("seconds", "size")))
    inter["seconds"] = inter["seconds"].astype("int64")
    inter["sessions"] = inter["sessions"].astype("int32")
    inter.to_parquet(common.WORK / "interactions.parquet", index=False)
    print(f"  {len(inter):,} (user, show, week) rows, {inter['user_id'].nunique():,} users, "
          f"{inter['tv_show_id'].nunique():,} shows")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
