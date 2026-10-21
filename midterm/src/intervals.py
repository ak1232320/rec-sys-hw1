"""Overlap join between viewing sessions and a broadcast schedule.

A viewing session is (channel, from, to). The schedule is a list of programmes
(channel, show, from, to). One session can straddle several programmes, so the
join has to explode each session into every programme it overlaps and carry the
overlap length with it - that length is the only honest interaction weight we
have, since the log never says what the viewer was actually watching.

The schedule for one channel is assumed sorted by start time and non-overlapping,
which is what a broadcast grid is; `check_schedule` verifies that rather than
trusting it.
"""
import numpy as np


def check_schedule(start, end):
    """Return a list of the ways this channel's grid is not a clean schedule."""
    problems = []
    if np.any(end < start):
        problems.append("negative-length programmes")
    if np.any(np.diff(start) < 0):
        problems.append("not sorted by start")
    if len(start) > 1 and np.any(start[1:] < end[:-1]):
        problems.append("overlapping programmes")
    return problems


def _run_offsets(counts):
    """[2, 3] -> [0, 1, 0, 1, 2]: the position of each output row inside its run."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    steps = np.ones(total, dtype=np.int64)
    steps[0] = 0
    ends = np.cumsum(counts)[:-1]
    steps[ends] = 1 - counts[:-1]
    return np.cumsum(steps)


def overlap_join(view_start, view_end, prog_start, prog_end, min_overlap=1):
    """Match every view against the programmes it overlaps on one channel.

    Returns (view_idx, prog_idx, overlap) - parallel arrays, one row per
    (view, programme) pair whose overlap is at least `min_overlap`.
    """
    view_start = np.asarray(view_start, dtype=np.int64)
    view_end = np.asarray(view_end, dtype=np.int64)
    prog_start = np.asarray(prog_start, dtype=np.int64)
    prog_end = np.asarray(prog_end, dtype=np.int64)

    if len(view_start) == 0 or len(prog_start) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty

    # The first programme that has not finished when the view starts, and the
    # first one that has not begun when the view ends: everything between the
    # two overlaps the view.
    lo = np.searchsorted(prog_end, view_start, side="right")
    hi = np.searchsorted(prog_start, view_end, side="left")
    counts = np.maximum(hi - lo, 0)

    keep = counts > 0
    if not keep.any():
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    counts = counts[keep]
    lo = lo[keep]
    rows = np.nonzero(keep)[0]

    view_idx = np.repeat(rows, counts)
    prog_idx = np.repeat(lo, counts) + _run_offsets(counts)

    overlap = (np.minimum(view_end[view_idx], prog_end[prog_idx])
               - np.maximum(view_start[view_idx], prog_start[prog_idx]))

    enough = overlap >= min_overlap
    return view_idx[enough], prog_idx[enough], overlap[enough]


def _self_check():
    # Grid: [0,10) [10,20) [20,30)
    ps = np.array([0, 10, 20])
    pe = np.array([10, 20, 30])

    # One view inside a single programme.
    v, p, o = overlap_join([2], [8], ps, pe)
    assert v.tolist() == [0] and p.tolist() == [0] and o.tolist() == [6], (v, p, o)

    # A view straddling all three.
    v, p, o = overlap_join([5], [25], ps, pe)
    assert p.tolist() == [0, 1, 2] and o.tolist() == [5, 10, 5], (p, o)

    # Touching, not overlapping: a view that ends exactly when a programme starts.
    v, p, o = overlap_join([0], [10], ps, pe)
    assert p.tolist() == [0] and o.tolist() == [10], (p, o)

    # Two views at once, one of them off the end of the grid.
    v, p, o = overlap_join([2, 28], [8, 40], ps, pe)
    assert v.tolist() == [0, 1] and p.tolist() == [0, 2] and o.tolist() == [6, 2], (v, p, o)

    # A view shorter than the minimum overlap is dropped entirely.
    v, p, o = overlap_join([9], [11], ps, pe, min_overlap=2)
    assert len(v) == 0, (v, p, o)

    # Zero-length and fully-outside views contribute nothing.
    v, p, o = overlap_join([5, 100], [5, 200], ps, pe)
    assert len(v) == 0, (v, p, o)

    assert _run_offsets(np.array([2, 3])).tolist() == [0, 1, 0, 1, 2]
    assert _run_offsets(np.array([1])).tolist() == [0]
    assert check_schedule(ps, pe) == []
    assert "overlapping programmes" in check_schedule(np.array([0, 5]), np.array([10, 15]))
    print("intervals.py self-check: ok")


if __name__ == "__main__":
    _self_check()
