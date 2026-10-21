"""What should count as "watched"? Look at the distributions, then measure.

The exam asks for an explicit rule and a justification against the observed
durations. This script prints the distributions (view length, overlap length,
overlap as a share of the programme) and the effect of each candidate cut on
how much of the data survives and on how stable the answer key is.

Run:  python experiments/watch_threshold.py   (after src/ingest.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common  # noqa: E402
import data  # noqa: E402

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def top5(frame):
    return data.truth_lists(frame.groupby(["user_id", "tv_show_id", "week"], as_index=False)["seconds"].sum())


def main():
    log = pd.read_parquet(common.WORK / "log.parquet", columns=["duration"])
    ov = pd.read_parquet(common.WORK / "overlaps.parquet")
    ov["fraction"] = ov["seconds"] / ov["prog_seconds"].clip(lower=1)
    out = {
        "view_seconds": {str(q): float(log["duration"].quantile(q)) for q in QUANTILES},
        "view_seconds_max": int(log["duration"].max()),
        "overlap_seconds": {str(q): float(ov["seconds"].quantile(q)) for q in QUANTILES},
        "programme_seconds": {str(q): float(ov["prog_seconds"].quantile(q)) for q in QUANTILES},
        "overlap_fraction_of_programme": {str(q): float(ov["fraction"].quantile(q)) for q in QUANTILES},
    }

    # Candidate rules. Absolute cuts drop channel-zapping; fractional cuts drop
    # partial views of long programmes but keep a full view of a 2-minute filler.
    rules = {
        "any overlap (>=1 s)": ov["seconds"] >= 1,
        ">=60 s": ov["seconds"] >= 60,
        ">=300 s": ov["seconds"] >= 300,
        ">=10% of programme": ov["fraction"] >= 0.10,
        ">=50% of programme": ov["fraction"] >= 0.50,
        ">=60 s or >=50% of programme": (ov["seconds"] >= 60) | (ov["fraction"] >= 0.50),
    }
    target = data.in_weeks(ov, common.VALID_WEEKS)
    reference = top5(target)
    rows = []
    for name, mask in rules.items():
        kept = ov[mask]
        tgt = target[mask.loc[target.index]]
        truth = top5(tgt)
        same = np.mean([len(set(reference[u]) & set(truth.get(u, []))) / len(reference[u])
                        for u in reference])
        rows.append({
            "rule": name,
            "overlaps_kept": float(mask.mean()),
            "seconds_kept": float(kept["seconds"].sum() / ov["seconds"].sum()),
            "user_show_pairs": int(kept.groupby(["user_id", "tv_show_id"]).ngroups),
            "valid_top5_agreement_with_no_cut": float(same),
        })
    out["rules"] = rows
    print(json.dumps(out, indent=1, ensure_ascii=False))
    (common.RESULTS / "watch_threshold.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
