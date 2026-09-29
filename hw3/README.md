# LLM4Rec A03 — Collaborative Filtering: user-based vs item-based on a 94% empty matrix

Aleksei Kosychev (amkosychev@edu.hse.ru), HSE LLM4Rec, Week 3.

**Live demo:** https://ak1232320.github.io/rec-sys-hw1/hw3/

Starter code and prompt: [dryjins/RecSys-LLMs `week3/`](https://github.com/dryjins/RecSys-LLMs/tree/main/week3) (commit `c00cfa1`).
Task: fill in the four TODO stubs — build the rating matrix, write the cosine similarity, implement
User-Based and Item-Based CF — choose a missing-value strategy, and analyse the two directions, the
strategy trade-off, and why CF fails on sparse data.

**What is here.** Both recommenders with a declared missing-value strategy (`weighted`: co-rated cosine
damped by `min(n, γ)/γ`, γ = 25) plus the other two available from the UI for comparison; an evidence
count next to every prediction; and a per-column cost readout. Three bugs in the starter's data layer
were fixed: the 19-flag genre off-by-one carried over from Week 2, the CRLF-truncated last field, and
`u.item` being decoded as UTF-8 when it is ISO-8859-1.

**Headline numbers** (943 users × 1,682 movies, per-user temporal split, 80k train / 20k test):

| | User-based | Item-based |
|---|---|---|
| Pairs to compare | 444,153 | **1,413,721** (3.2×) |
| Comparisons per request (user 405) | 942 | **1,240,371** (1,317×) |
| Precision@5 (`weighted`) | 1.5% | 0.0% |
| Precision@5 (`mean` imputation) | 4.1% | 10.6% |

A non-personalised most-rated list reaches **Precision@5 of 7.9%**, so most of the specified
configurations lose to not personalising at all. What fixes that is evidence, not the similarity
function: requiring 10 ratings behind a prediction instead of 1 takes user-based from 1.5% to 8.4% and
RMSE from 1.128 to 0.980, while prediction coverage falls from 77.5% to 18.5%. Mean imputation squeezes
every user pair into a band 1.7×10⁻⁴ wide around 0.99 — 61× narrower than co-rated cosine — so its
neighbour ranking is decided by floating-point noise. Full analysis in
`report/kosychev_a03_report.pdf`.

## Layout

| Path | What |
|---|---|
| `index.html`, `style.css`, `data.js`, `cf.js`, `script.js`, `prompt.md` | Finished version (served by GitHub Pages) |
| `u.item`, `u.data` | MovieLens 100k, unchanged copies from the course repo (byte-identical to `hw2/`) |
| `starter/` | Unchanged copies of the course files, TODO stubs included; compare with the root |
| `report/*.diff` | Exact changes per file (`cf.js` is new, so it has no diff) |
| `experiments/run_experiment.py` | Offline evaluation; output in `experiments/results.json` |
| `experiments/make_figures.py` | Figures for the report (`make_manual_figure.py` composes Fig. 4) |
| `tests/check_cf.py` | Browser test (Playwright); output in `tests/results.json` and `tests/run_log.txt` |
| `report/kosychev_a03_report.pdf` | Report (source `report/report.md`, build `python report/build_report.py`) |
| `manual-test/` | Screenshots of the checks run by hand |
| `task-slide.png` | The assignment slide (the lecture `.pptx` sits in this folder but is git-ignored) |

## Run locally

`fetch()` cannot read `u.item` from `file://`, so the app needs a server:

```
cd hw3
python -m http.server 8000
```

Then open http://localhost:8000/ for the finished app and http://localhost:8000/starter/ for the
skeleton.

Reproduce the numbers and the tests (`pip install playwright numpy matplotlib markdown`, plus a local
Google Chrome):

```
python experiments/run_experiment.py     # writes experiments/results.json (~3 min)
python experiments/make_figures.py       # writes report/figures/*.png
python tests/check_cf.py                 # 36 checks against the live page
python report/build_report.py            # rebuilds the PDF
```

The browser test is what ties the two implementations together: it checks that the fast sparse
similarity in `cf.js` equals the readable dense definition in the same file, and that every Top-5 the UI
renders equals the list `run_experiment.py` computes in NumPy.
