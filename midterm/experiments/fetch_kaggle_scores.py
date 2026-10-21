"""Record the returned Kaggle scores of this account's submissions (read-only API call).

Needs the `kaggle` package and an API token; run with the interpreter that has it.
Run:  python experiments/fetch_kaggle_scores.py
"""
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

OUT = Path(__file__).resolve().parent.parent / "submissions" / "kaggle_scores.json"
FILES = {
    "FINAL: habit x schedule": "submission_habit.csv",
    "FINAL: hybrid LambdaRank": "submission_hybrid.csv",
    "habit x schedule exposure": "submission_exposure.csv",
    "personalized PageRank (co-view graph)": "submission_ppr.csv",
    "popularity baseline": "submission_popularity.csv",
}

api = KaggleApi()
api.authenticate()
rows = []
for s in api.competition_submissions("sweettv-tv-program-recommender"):
    label = s.description.split("|", 1)[-1].strip()
    rows.append({
        "ref": s.ref, "file": FILES.get(label), "description": s.description,
        "submitted": str(s.date), "status": str(s.status).split(".")[-1],
        "public": float(s.public_score) if s.public_score else None,
        "private": float(s.private_score) if s.private_score else None,
    })
OUT.write_text(json.dumps({"competition": "sweettv-tv-program-recommender", "metric": "MAP@5",
                           "designated": "submission_habit.csv",
                           "note": "submission_hybrid.csv was the designated file of the first round "
                                   "(preference counted in seconds); it was superseded once the "
                                   "answer key turned out to count watched airings.",
                           "submissions": rows},
                          indent=1, ensure_ascii=False), encoding="utf-8")
for r in rows:
    print(r["ref"], r["file"], r["status"], r["public"], r["private"])
