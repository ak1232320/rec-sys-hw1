# LLM4Rec A06 — Personalized PageRank: personal, but not better

Aleksei Kosychev (amkosychev@edu.hse.ru), HSE LLM4Rec, Week 6.

**Live demo:** https://ak1232320.github.io/rec-sys-hw1/hw6/

Starter code: [dryjins/RecSys-LLMs `week6/`](https://github.com/dryjins/RecSys-LLMs/tree/main/week6) (commit `00c032f`).
Dataset: MovieLens 100k, the same `u.data` / `u.item` and the same per-user chronological 80/10/10 split as
Weeks 2, 3 and 5 — byte-identical files. Task: implement the seven `TODO(hw6)` stubs — the bipartite graph,
the row-normalized transition, global PageRank, Personalized PageRank, movie-node scores, Top-10 retrieval
and Recall@10 — verify the propagation, select α on validation Recall@10, and compare popularity, global
PageRank, PPR and the previous assignment's MF on one shared cohort.

**What is here.** The seven stubs on TensorFlow.js tensors, propagating over the two off-diagonal blocks of
the transition so the 2,554 × 2,554 matrix is never built. Only `graph.js` and `script.js` differ from the
course files; `index.html`, `style.css`, `data.js` and `vendor/d3.min.js` are byte-identical to `starter/`.

Every evaluation sweeps the same graph at the same α over hundreds of restart users, so the first PPR call
for a `(graph, α)` pair solves **all 943 seeds in one pass** and later calls read their column:
**7.6 ms per user instead of 455 ms**, verified against the single-seed walk to 3.7×10⁻⁹.

**Headline numbers** (861 shared test users, identical candidates and exclusions, α = 0.95 selected on
validation):

| Method | Test Recall@10 | Mean training popularity | Catalogue reached |
|---|---|---|---|
| Popularity | **7.60%** | 272.0 | 5.1% |
| Global PageRank | 7.46% | 271.3 | 4.9% |
| PPR | 6.19% | 216.1 | **28.4%** |
| MF (provided reference) | 2.11% | 96.8 | 2.4% |

Three findings behind that table:

* **Global PageRank on this graph is a popularity list.** On an undirected bipartite graph the stationary
  distribution of a degree-normalized walk is proportional to degree, and a movie's degree here *is* how
  many users rated it ≥ 4.0. The two methods land on the same point — within one rating and 0.2 pp — after
  a 243-iteration power method.
* **Validation picks α = 0.95, the least personal setting.** Recall@10 is flat at 4.91–4.94% for α ≤ 0.85
  and jumps to 5.72% at 0.95, where the walk strays furthest from its seed. The tuning objective rewards
  PPR for being less personal.
* **The provided MF reference never trained.** Its train and validation RMSE are identical at all 30 epochs
  to the last digit, its factor norms are 3.6×10⁻⁵, and `mfScores` returns **one** distinct value — the
  global mean — for all 1,611 candidates. The same regularizer defect A05 diagnosed. My own A05 model on
  this same cohort and protocol reaches 1.63% (shipped settings) or 3.72% (the better configuration), still
  below PPR.

Changing only the restart user replaces **10 of 10** titles; changing only α (0.85 → 0.50) replaces **1 of
10**. Full analysis in `report/kosychev_a06_report.pdf`.

**Checks.** The page's own harness reports **12 passed / 0 pending** (the untouched starter reports 2
passed / 10 pending), and `tests/check_ppr.py` adds **14**, including both readme §4.1 reference vectors to
1.7×10⁻⁷, mass conservation with a zero-degree seed, and the batched-vs-single-seed agreement.

## Layout

| Path | What |
|---|---|
| `index.html`, `style.css`, `data.js`, `graph.js`, `script.js`, `vendor/` | Finished version (served by GitHub Pages); only `graph.js` and `script.js` differ from the starter |
| `u.data`, `u.item` | MovieLens 100k, unchanged copies from the course repo (byte-identical to `hw2/`, `hw3/`, `hw5/`) |
| `starter/` | Unchanged copies of the course files, TODO stubs and the course `readme.md` included |
| `report/graph.diff`, `report/script.diff` | The exact changes against `starter/` |
| `verification.json`, `results.csv`, `recommendations.csv` | The readme §9 deliverables, written by the page's own exporters |
| `experiments/run_experiment.py` | Drives the full protocol through the real page → `results.json` and the three exports |
| `experiments/make_figures.py`, `make_app_figure.py` | Figures for the report |
| `tests/check_ppr.py` | Browser test (Playwright); output in `tests/results.json` and `tests/run_log.txt` |
| `report/kosychev_a06_report.pdf` | Report (source `report/report.md`, build `python report/build_report.py`) |
| `manual/` | Screenshots of the checks run by hand |
| `task-slide.png` | The assignment slide |

## Run locally

TensorFlow.js is loaded from a CDN and `data.js` uses `fetch()`, so the page needs an HTTP server and
network access on first load (d3 is vendored locally):

```
cd hw6
python -m http.server 8000
```

Open http://localhost:8000/ for the finished page and http://localhost:8000/starter/ for the skeleton.
Press **Build graph**, then **Run self-checks**, **Run α sweep**, **Compare methods**, **Two-user demo**
and the three **Export** buttons. The α sweep takes about 35 seconds and the four-method comparison about
80 seconds, including training the MF reference on first use.

Reproduce the numbers (`pip install playwright numpy matplotlib markdown pillow`, plus a local Google
Chrome):

```
python experiments/run_experiment.py     # full protocol through the page (~4 min); writes results.json + the three exports
python experiments/make_figures.py       # writes report/figures/*.png
python tests/check_ppr.py                # 14 checks against the live page
python experiments/make_app_figure.py    # composes Fig. 3 from the test's screenshots
python report/build_report.py            # rebuilds the PDF
```

Dependency versions actually used: TensorFlow.js 4.22.0 (CDN), d3 v7 (vendored), Google Chrome headless on
the WebGL backend, Python 3.12 with Playwright 1.58 and Matplotlib 3.10.

## Known limitations

* The split is chronological **within** each user, not globally, so one user's "future" film can be another
  user's "past" one. MovieLens records rating times, not viewing sessions.
* This is a warm-user evaluation: every cohort user has at least one positive training edge, so nothing
  here speaks to cold start.
* PPR scores are long-run walk probabilities — not predicted ratings, not purchase probabilities, and not
  comparable in scale to MF scores.
