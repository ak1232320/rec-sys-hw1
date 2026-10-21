# LLM4Rec homework — Aleksei Kosychev

HSE, Recommender Systems with LLMs. Course starter code: [dryjins/RecSys-LLMs](https://github.com/dryjins/RecSys-LLMs).

**Live:** https://ak1232320.github.io/rec-sys-hw1/

| | Assignment | What it is | Links |
|---|---|---|---|
| **A01** | Week 1 — Random Lunch Generator | Finding and fixing the "images sometimes not displaying" bug: three invented Font Awesome class names that do not exist in the loaded stylesheet, hit on ~25% of clicks. | [app](https://ak1232320.github.io/rec-sys-hw1/hw1/) · [report](hw1/report/kosychev_a01_report.pdf) · [`hw1/`](hw1/) |
| **A02** | Week 2 — Content-Based Movie Recommender | Cosine similarity instead of Jaccard, a user profile averaged over several watched movies, Top-5, and a measured comparison of item-to-item vs profile-based ranking on MovieLens 100k. | [app](https://ak1232320.github.io/rec-sys-hw1/hw2/) · [report](hw2/report/kosychev_a02_report.pdf) · [`hw2/`](hw2/) |
| **A03** | Week 3 — Collaborative Filtering | User-based and item-based CF on the same ratings: a declared missing-value strategy, the evidence behind every prediction, and a measured comparison of the two directions in cost and quality. | [app](https://ak1232320.github.io/rec-sys-hw1/hw3/) · [report](hw3/report/kosychev_a03_report.pdf) · [`hw3/`](hw3/) |
| **A04** | Week 4 — Association Rules | Apriori on 17,080 UCI Online Retail baskets, checked against FP-Growth; why the top rule by lift is the one to reject, and which rules survive a later period once each basket's real date is recovered from the raw log. | [app](https://ak1232320.github.io/rec-sys-hw1/hw4/) · [report](hw4/report/kosychev_a04_report.pdf) · [`hw4/`](hw4/) |
| **A05** | Week 5 — Matrix Factorization | A 16-factor model trained in the browser with TensorFlow.js: test RMSE 1.0432 against a 1.2289 baseline, but Recall@10 of 1.63% — the lists keep the user's genre and pick films almost nobody rated. | [app](https://ak1232320.github.io/rec-sys-hw1/hw5/) · [report](hw5/report/kosychev_a05_report.pdf) · [`hw5/`](hw5/) |
| **A06** | Week 6 — Personalized PageRank | A user–movie graph walked on TensorFlow.js: PPR is genuinely personal (changing the seed replaces 10 of 10 titles) but a popularity list still wins Recall@10 — and global PageRank on this graph *is* that popularity list. | [app](https://ak1232320.github.io/rec-sys-hw1/hw6/) · [report](hw6/report/kosychev_a06_report.pdf) · [`hw6/`](hw6/) |
| **Midterm** | sweet.tv TV programme recommender (Kaggle) | A view–schedule join, the four course methods compared on one hold-out, and a habit × schedule model: public MAP@5 0.42168, up from 0.32279 once the answer key turned out to count watched airings rather than seconds. | [report](midterm/report/kosychev_midterm_report.pdf) · [`midterm/`](midterm/) |

Each folder keeps the same layout: the fixed app at its root, unchanged course files in `starter/`,
per-file diffs and the report in `report/`, and automated checks in `tests/`. See the README inside each
folder for how to run it.
