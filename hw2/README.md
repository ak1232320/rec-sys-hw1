# LLM4Rec A02 — Content-Based Movie Recommender: cosine similarity and a user profile

Aleksei Kosychev (amkosychev@edu.hse.ru), HSE LLM4Rec, Week 2.

**Live demo:** https://ak1232320.github.io/rec-sys-hw1/hw2/

Starter code and prompt: [dryjins/RecSys-LLMs `week2/`](https://github.com/dryjins/RecSys-LLMs/tree/main/week2) (commit `af8c98a`).
Task: replace the naive Jaccard matching with cosine similarity, build a user profile by averaging several
watched movies, return a Top-5, and analyse item-to-item vs profile-based ranking, bias mitigation and
long-tail discovery.

**What changed.** Cosine similarity; an averaged profile vector; Top-5 instead of Top-2; both modes shown
side by side on the same watch history; an explicit, deterministic tie-break. Two bugs in the starter were
also fixed: it zips 18 genre names against the **19** genre flags of `u.item` (so Toy Story is stored as a
*Crime* film and the catalogue gains 71 Westerns instead of 27), and splitting on `'\n'` leaves a `\r` on
the last field after a Windows checkout rewrites the data files to CRLF, which silently deletes the Western
genre.

**Headline numbers** (1,682 movies, 100,000 ratings, 940 users with enough history):

| | Item-to-item | Profile-based |
|---|---|---|
| Hit-rate@5 / Precision@5 | 38.4% / 13.2% | **46.5% / 15.7%** |
| Recommendations in the long tail | **39.0%** | 36.1% |
| Catalogue coverage | **20.8%** | 18.7% |

The two modes share on average only 0.88 of their 5 titles, and are completely disjoint for 67% of users.

Cosine and Jaccard return the *identical* Top-5 for all 1,680 item-to-item queries — normalisation is what
removes the multi-genre blockbuster bias (raw dot product: 2.84 genres and 236 ratings per recommendation
vs 1.69 and 136), not cosine specifically. And 126 candidates tie at the top score on average, so the
tie-break, not the similarity, picks the visible list: it swings long-tail share from 24% to 57%.
Full analysis in `report/kosychev_a02_report.pdf`.

## Layout

| Path | What |
|---|---|
| `index.html`, `style.css`, `data.js`, `recommender.js`, `script.js`, `prompt.md` | Fixed version (served by GitHub Pages) |
| `u.item`, `u.data` | MovieLens 100k, unchanged copies from the course repo |
| `starter/` | Unchanged copies of the course files; compare with the root |
| `report/*.diff` | Exact changes per file (`recommender.js` is new, so it has no diff) |
| `experiments/run_experiment.py` | Offline evaluation over the whole catalogue; output in `experiments/results.json` |
| `experiments/make_figures.py` | Figures for the report (`make_manual_figure.py` composes the one below) |
| `manual/` | Screenshots of the manual check: the local server and the running app |
| `tests/check_recommender.py` | Browser test (Playwright); output in `tests/results.json` and `tests/run_log.txt` |
| `report/kosychev_a02_report.pdf` | Report (source `report/report.md`, build `python report/build_report.py`) |
| `task-slide.png` | The assignment slide this work is based on |

## Run locally

`fetch()` cannot read `u.item` from `file://`, so the app needs a server:

```
cd hw2
python -m http.server 8000
```

Then open http://localhost:8000/ for the fixed app and http://localhost:8000/starter/ for the starter.

Reproduce the numbers and the tests (`pip install playwright numpy matplotlib markdown`, plus a local
Google Chrome):

```
python experiments/run_experiment.py     # writes experiments/results.json
python experiments/make_figures.py       # writes report/figures/*.png
python tests/check_recommender.py        # 28 checks against the live page
python report/build_report.py            # rebuilds the PDF
```

The browser test is the link between the two implementations: it starts its own server, drives the real
page and asserts that every Top-5 the UI renders equals the reference list computed by
`run_experiment.py`.
