# Midterm quiz — TV programme recommendation on sweet.tv

HSE LLM4Rec, individual exam. Kaggle competition
[`sweettv-tv-program-recommender`](https://www.kaggle.com/competitions/sweettv-tv-program-recommender):
for each of 1,260 users, predict the five programmes they will watch most in weeks 31–42 of 2020.
Metric MAP@5.

**Report:** [`report/kosychev_midterm_report.pdf`](report/kosychev_midterm_report.pdf) ·
**Designated submission:** [`submissions/submission_habit.csv`](submissions/submission_habit.csv)

| Submission (Kaggle ref) | Public | Private |
|---|---:|---:|
| **Habit × schedule** — final (57498272) | **0.42168** | **0.41476** |
| LambdaRank hybrid — first round (57496713) | 0.32279 | 0.29662 |
| Channel-hour exposure (57496711) | 0.29113 | 0.26829 |
| Personalised PageRank, co-view graph (57496709) | 0.29038 | 0.25733 |
| Popularity (57496705) | 0.28361 | 0.27447 |

Refs, dates and scores as returned by the API: [`submissions/kaggle_scores.json`](submissions/kaggle_scores.json).

## What happened, in three lines

1. **Join + four course methods + a LambdaRank hybrid**, with preference counted as *seconds watched*.
   The hybrid reached 0.347 on the hold-out and 0.32279 on Kaggle.
2. **The public scores did not match the hold-out.** Popularity scored 0.181 on the hold-out but 0.284 on
   Kaggle. Re-labelling the hold-out under several definitions showed that the answer key counts
   *airings watched*, not seconds ([`experiments/probe_matrix.py`](experiments/probe_matrix.py)).
3. **Final model: habit × schedule.** For each (user, programme, channel), the share of airings the user
   watched to ≥80% when they had the chance, times the programme's airings in the weeks 31–42 schedule.
   No training, ~25 s. The 80% rule and the model's idea come from material shared with me; the code is
   my own (see *Disclosure*).

## Layout

| Path | What it is |
|---|---|
| [`src/ingest.py`](src/ingest.py) | Raw CSV → parquet; the view × airing overlap join ([`src/intervals.py`](src/intervals.py)) |
| [`src/data.py`](src/data.py) | Splits (inner 11–15→16–20, hold-out 11–20→21–30, test 11–30→31–42), answer keys |
| [`src/methods.py`](src/methods.py) | Popularity, history, CBF, association rules, MF (ALS, PureSVD), PPR on the co-view graph |
| [`src/exposure.py`](src/exposure.py) | Channel × half-hour viewing habit × target-week schedule |
| [`src/habit.py`](src/habit.py) | Watched airings (≥80% coverage), the answer key, the final habit × schedule model |
| [`src/features.py`](src/features.py), [`src/hybrid.py`](src/hybrid.py) | First-round candidate pool, features and LambdaRank re-ranker |
| [`src/predict.py`](src/predict.py) | Writes the submissions (`--first-submission` rebuilds the first round) |
| [`experiments/`](experiments/) | Data profile, watch-threshold study, hold-out runs, answer-key probes, two users, figures |
| [`tests/check_pipeline.py`](tests/check_pipeline.py) | 9 checks: unit cases, id parsing, join bounds, time order, leakage, file format, answer-key rule, byte-for-byte rebuild |
| [`report/`](report/) | `report.md` → `kosychev_midterm_report.pdf` via `build_report.py` |

## Reproduce

```bash
pip install pandas pyarrow scipy scikit-learn implicit lightgbm matplotlib markdown playwright
kaggle competitions download -c sweettv-tv-program-recommender -p midterm/data_raw   # accept the rules first
cd midterm && unzip data_raw/*.zip -d data_raw

python src/ingest.py                              # ~1 min: work/epg|log|overlaps.parquet
python src/predict.py                             # ~25 s: submissions/submission_habit.csv
python src/predict.py --first-submission          # ~5 min: the four first-round files
python experiments/run_validation.py              # hold-out table under the competition's key
python experiments/run_validation.py --target seconds   # the first-round table and hybrid
python tests/check_pipeline.py --repro            # all checks, incl. byte-for-byte rebuild
```

The raw data (890 MB) and the parquet caches are not in the repository (`.gitignore`); the
competition's rules must be accepted on Kaggle before the download works (otherwise HTTP 403).

## Disclosure

- **Shared material.** The 80%-coverage rule for a watched airing, and the idea of the final model (rate of
  watched airings per user × programme × channel, recency-weighted, corrected for days the user was in the
  log, times the target-week airing count), came from a write-up of a classmate's solution
  (repository `Marrrik/sweettv-recsys`) that was shared with me. I implemented it from the description;
  no code was copied. Its settings (half-life 10 weeks, prior 30) are the described ones.
- **AI assistance.** The work was done with Claude Code; the unmodified session export is submitted with
  the report.
- **Libraries.** pandas, NumPy, SciPy, scikit-learn, implicit (ALS), LightGBM, Matplotlib.
