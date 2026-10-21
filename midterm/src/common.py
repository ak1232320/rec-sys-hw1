"""Shared paths, column resolution and the week calendar for the midterm."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw"
WORK = ROOT / "work"
RESULTS = ROOT / "experiments"
SUBMISSIONS = ROOT / "submissions"

COMPETITION = "sweettv-tv-program-recommender"

# The schedule is published in three slices; the last one covers the weeks the
# competition asks about, so it is the candidate pool and never a training source.
EPG_FILES = {
    "11-20": RAW / "export_arh_11-20-final.csv",
    "21-30": RAW / "export_arh_21-30-final.csv",
    "31-42": RAW / "export_arh_31-42-final.csv",
}
LOG_FILE = RAW / "dataset11-30.csv"
SAMPLE_SUBMISSION = RAW / "submission.csv"

# Validation mirrors the competition: fit on the earlier weeks, rank the shows of
# a later window the model has never seen anyone watch.
TRAIN_WEEKS = (11, 20)
VALID_WEEKS = (21, 30)
FULL_WEEKS = (11, 30)
TEST_WEEKS = (31, 42)

# The hybrid's ranker is itself fitted, and the hold-out may only select, so it
# learns from a split nested inside the training weeks: 11-15 -> 16-20.
INNER_HISTORY = (11, 15)
INNER_TARGET = (16, 20)

TOP_K = 5

# An overlap counts as "watched" from one minute up. The cut is absolute, not a
# share of the programme: the median programme runs 100 min and the median
# overlap covers 11.6% of it, so a 50%-of-programme rule throws away 43% of all
# watched time. One minute removes channel-zapping (2.7% of overlaps, 0.07% of
# seconds). See experiments/watch_threshold.py.
WATCH_MIN_SECONDS = 60

# How "most preferred" is counted when building an answer key from the log.
# "watched": the number of airings the user covered for at least 80% - the
# organisers' rule as relayed in shared material (src/habit.py). The first
# submission counted total seconds instead; experiments/probe_*.py show why
# that did not match the public scores.
TARGET = "watched"


def resolve(columns, *candidates):
    """Pick the first candidate present in `columns`, case- and separator-insensitive."""
    norm = {str(c).strip().lower().replace("-", "_").replace(" ", "_"): c for c in columns}
    for name in candidates:
        key = name.strip().lower().replace("-", "_").replace(" ", "_")
        if key in norm:
            return norm[key]
    raise KeyError(f"none of {candidates} in {list(columns)}")


def iso_week(series):
    """Competition weeks are ISO weeks of 2020; keep them as a plain integer."""
    return pd.to_datetime(series).dt.isocalendar().week.astype("int16")


def ensure_dirs():
    for folder in (WORK, RESULTS, SUBMISSIONS):
        folder.mkdir(parents=True, exist_ok=True)
