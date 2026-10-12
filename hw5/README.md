# LLM4Rec A05 — Matrix Factorization: familiar genre, unfamiliar title

Aleksei Kosychev (amkosychev@edu.hse.ru), HSE LLM4Rec, Week 5.

**Live demo:** https://ak1232320.github.io/rec-sys-hw1/hw5/

Starter code: [dryjins/RecSys-LLMs `week5/`](https://github.com/dryjins/RecSys-LLMs/tree/main/week5) (commit `a4f53be`).
Dataset: MovieLens 100k, the same `u.data` / `u.item` as Week 2 and Week 3 — byte-identical, so the algorithm
is the only thing that changes. Task: implement the six `TODO(hw5)` stubs with TensorFlow.js — the factor
variables, a batched training loop, the scalar prediction, checkpoint selection, Top-10 retrieval and
Recall@10 — then answer one question: do the recommendations help users **discover** films, or do they
mainly **repeat** what they already like?

**What is here.** The six stubs implemented with `tf.variable` / `tf.gather` / `tf.matMul` / `tf.topk` and an
Adam optimizer, seeded so a re-run reproduces every reported number. Only `script.js` differs from the
course files; `index.html`, `style.css` and `data.js` are byte-identical to `starter/`.

Two corrections to the assignment's own code sketch, both measured:

* its L2 term squares the **whole** factor matrices on every mini-batch — at the shipped `reg = 0.05` and
  1,245 batches per epoch that drives the mean item-factor norm to **1.7×10⁻¹⁹** and the test RMSE to
  **1.2289002**, i.e. exactly the mean-rating baseline's 1.2289001;
* `selectBestCheckpoint` only *names* the best epoch, so `trainMF` now snapshots the factors at that epoch
  and assigns them back — worth **0.0336 RMSE**, 18% of the whole gain over the baseline.

**Headline numbers** (the page's shipped defaults: 16 factors, 20 epochs, Adam lr 0.01, reg 0.05, batch 64):

| | |
|---|---|
| Split (per user, by timestamp) | 79,619 train / 9,596 validation / 10,785 test; **0** users excluded |
| Selected checkpoint | epoch **2** of 20, validation RMSE 1.0026 |
| Test RMSE vs mean-rating baseline | **1.0432 vs 1.2289** (15.1% better); last epoch would be 1.0768 |
| Recall@10 | **1.63%** over 861 users; 82 users and 13 relevant films excluded |
| Top-10 in one of the user's 3 most-liked genres | **6.6 of 10** (65.8%) |
| Mean training ratings: recommended film vs the user's own | **63.5 vs 170.8** — 96.4% of users get a less-known list |
| Recall@10 when candidates need ≥ 100 training ratings | **7.92%** (4.9×) |

The lists repeat the user's **genre** and randomise the **title**: with 16 free parameters per film and an
argmax over 1,611 candidates, the films whose factors are least constrained by data win. Full analysis in
`report/kosychev_a05_report.pdf`.

**Checks.** The page's own harness reports **11 passed / 0 pending** (the untouched starter reports 4 passed
/ 7 pending), and `tests/check_mf.py` adds 11 more, including a prediction recomputed outside the browser
and a proof that the model really is the restored checkpoint.

## Layout

| Path | What |
|---|---|
| `index.html`, `style.css`, `data.js`, `script.js` | Finished version (served by GitHub Pages); only `script.js` differs from the starter |
| `u.data`, `u.item` | MovieLens 100k, unchanged copies from the course repo (byte-identical to `hw2/` and `hw3/`) |
| `starter/` | Unchanged copies of the course files, TODO stubs and the course `readme.md` included |
| `report/script.diff` | The exact change against `starter/script.js` |
| `experiments/run_experiment.py` | Trains through the real page and records everything → `results.json` |
| `experiments/make_figures.py` | Figures for the report |
| `tests/check_mf.py` | Browser test (Playwright); output in `tests/results.json` and `tests/run_log.txt` |
| `report/kosychev_a05_report.pdf` | Report (source `report/report.md`, build `python report/build_report.py`) |
| `manual/` | Screenshots of the checks run by hand |
| `task-slide.jpg` | The assignment slide (the lecture `.pptx` is git-ignored) |

## Run locally

TensorFlow.js is loaded from a CDN and `data.js` uses `fetch()`, so the page needs an HTTP server and
network access on first load:

```
cd hw5
python -m http.server 8000
```

Open http://localhost:8000/ for the finished page and http://localhost:8000/starter/ for the skeleton. Press
**Run tests**, then **Train**, **Recommend Top-10** and **Evaluate Recall@10**. Training with the shipped
defaults takes about four minutes on the WebGL backend; batch 1,024 does ten epochs in under a minute.

Reproduce the numbers (`pip install playwright numpy matplotlib markdown`, plus a local Google Chrome):

```
python experiments/run_experiment.py     # trains through the page, writes results.json (~15 min)
python experiments/make_figures.py       # writes report/figures/*.png
python tests/check_mf.py                 # 11 checks against the live page
python report/build_report.py            # rebuilds the PDF
```

Both scripts drive the same page the browser serves, so every number in the report is produced by the
submitted `script.js` rather than by a Python re-implementation.

## Known defect left unfixed

`data.js` reads `u.item` with `Response.text()`, which decodes as UTF-8 although the file is ISO-8859-1, so
**9 of the 1,682 titles** are mangled — including *C'est arrivé près de chez vous*, the top recommendation
for user 655. The submission instructions list `data.js` as provided and not to be modified, so it is
unmodified here. The fix is one line: read `arrayBuffer()` and decode with `new TextDecoder('iso-8859-1')`.
