"""Habit x schedule: how much of a show falls into the time slots a user watches TV.

Television is watched by habit - the same channels at the same times of the
week. For every (channel, slot-of-the-week) cell this module measures

  t[u, cell]  seconds per week user u spent on that channel in that slot
              (from the history weeks' log - all airtime, labelled or not), and
  a[s, cell]  seconds per week show s occupies that slot
              (from the target weeks' *schedule* only),

and scores exposure[u, s] = sum over cells of t * a / slot_length: the seconds
per week u would spend on s if their habits held and their viewing were spread
evenly within each slot. It reaches shows nobody has watched yet, because it
needs only the show's place in the grid.
"""
import numpy as np
import scipy.sparse as sp

WEEK = 7 * 24 * 3600
SLOT = 1800                # half-hour cells: the broadcast grid starts on :00 and :30
SESSION_CAP = 6 * 3600     # a session longer than this is a TV left on
_THURSDAY = 3 * 24 * 3600  # 1970-01-01 was a Thursday; shift so Monday 00:00 is slot 0


def _epoch_seconds(series):
    return series.values.astype("datetime64[s]").astype(np.int64)


def slot_bins(start, end, slot=SLOT):
    """Explode intervals into the clock slots they cover.

    Returns (interval index, slot-of-week with Monday 00:00 = 0, seconds).
    """
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    first = start // slot
    last = (np.maximum(end, start + 1) - 1) // slot
    counts = (last - first + 1).astype(np.int64)
    idx = np.repeat(np.arange(len(start)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    cell = first[idx] + offsets
    secs = np.minimum(end[idx], (cell + 1) * slot) - np.maximum(start[idx], cell * slot)
    of_week = ((cell * slot + _THURSDAY) % WEEK) // slot
    keep = secs > 0
    return idx[keep], of_week[keep], secs[keep]


def _channel_index(split):
    channels = np.union1d(split.hist_log["channel_id"].unique(),
                          np.union1d(split.epg_target["channel_id"].unique(),
                                     split.epg_hist["channel_id"].unique()))
    return {c: i for i, c in enumerate(channels)}


def user_slots(split, channel_index, last_weeks=None, slot=SLOT):
    log = split.hist_log
    weeks = split.history_weeks[1] - split.history_weeks[0] + 1
    if last_weeks:
        log = log[log["week"] > split.history_weeks[1] - last_weeks]
        weeks = last_weeks
    start = _epoch_seconds(log["start"])
    end = np.minimum(_epoch_seconds(log["end"]), start + SESSION_CAP)
    idx, of_week, secs = slot_bins(start, end, slot)
    per_week = WEEK // slot
    rows = log["user_id"].map(split.uni.user_index).to_numpy()[idx]
    cols = log["channel_id"].map(channel_index).to_numpy()[idx] * per_week + of_week
    return sp.csr_matrix((secs / weeks, (rows, cols)),
                         shape=(len(split.uni.users), len(channel_index) * per_week))


def show_slots(split, channel_index, schedule="target", slot=SLOT):
    epg, span = ((split.epg_target, split.target_weeks) if schedule == "target"
                 else (split.epg_hist, split.history_weeks))
    cand_index = {s: i for i, s in enumerate(split.candidates)}
    epg = epg[epg["tv_show_id"].isin(cand_index)]
    idx, of_week, secs = slot_bins(_epoch_seconds(epg["start"]), _epoch_seconds(epg["end"]), slot)
    per_week = WEEK // slot
    rows = epg["tv_show_id"].map(cand_index).to_numpy()[idx]
    cols = epg["channel_id"].map(channel_index).to_numpy()[idx] * per_week + of_week
    weeks = span[1] - span[0] + 1
    return sp.csr_matrix((secs / weeks, (rows, cols)),
                         shape=(len(split.candidates), len(channel_index) * per_week))


def exposure(split, last_weeks=None, schedule="target", slot=SLOT, **_):
    """users x candidates, expected seconds per week (see module docstring).

    `schedule="history"` places each candidate on the *history* grid instead:
    what the same habit would have predicted for the weeks already observed,
    which the ranker compares with what the user actually watched.
    `last_weeks` restricts the habit to the end of the history (tested; a
    longer habit window predicts better, so it is off by default).
    """
    index = _channel_index(split)
    T = user_slots(split, index, last_weeks, slot)[split.eval_rows]
    A = show_slots(split, index, schedule, slot)
    return np.asarray((T @ A.T).todense() / slot, dtype=np.float32)


def _self_check():
    monday = 4 * 86400                     # 1970-01-05 00:00 UTC was a Monday
    _, w, s = slot_bins([monday + 1800], [monday + 5400], 3600)
    assert w.tolist() == [0, 1] and s.tolist() == [1800, 1800], (w, s)
    # Sunday 23:30 -> Monday 00:30 wraps from the last slot of the week to slot 0.
    _, w, s = slot_bins([monday - 1800], [monday + 1800], 3600)
    assert w.tolist() == [167, 0] and s.tolist() == [1800, 1800], (w, s)
    _, w, s = slot_bins([monday], [monday + 3600], 3600)
    assert w.tolist() == [0] and s.tolist() == [3600], (w, s)
    # Half-hour slots split the same interval in two.
    _, w, s = slot_bins([monday], [monday + 3600], 1800)
    assert w.tolist() == [0, 1] and s.tolist() == [1800, 1800], (w, s)
    _, w, s = slot_bins([monday - 900], [monday + 900], 1800)
    assert w.tolist() == [335, 0] and s.tolist() == [900, 900], (w, s)
    print("exposure.py self-check: ok")


if __name__ == "__main__":
    _self_check()
