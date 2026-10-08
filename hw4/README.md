# LLM4Rec A04 — Association Rules: lift rewards rarity

Aleksei Kosychev (amkosychev@edu.hse.ru), HSE LLM4Rec, Week 4.

**Live demo:** https://ak1232320.github.io/rec-sys-hw1/hw4/

Starter code: [dryjins/RecSys-LLMs `week4/`](https://github.com/dryjins/RecSys-LLMs/tree/main/week4) (commit `c00cfa1`).
Dataset: [UCI Online Retail, id 352](https://archive.ics.uci.edu/dataset/352/online+retail), cleaned by the course
to 17,080 baskets × 3,653 items. Task: implement the seven `TODO(hw4)` stubs — basket deduplication, itemset
counting, support / confidence / lift, a frequent-itemset miner and the rule generator — justify the thresholds,
keep rules with lift > 1, and present one rule worth investigating and one to reject.

**What is here.** Apriori with bitset counting in `script.js`, a "Keep only rules with lift > 1" checkbox, and
a detail panel that prints "undefined" instead of a number on a zero denominator. Every rule the page mines is
checked against mlxtend's FP-Growth. The course export turned out to reproduce the raw UCI workbook basket for
basket, so every basket got its real `InvoiceDate` back and the rules were tested on a later period instead of
simulated data.

**Headline numbers** (support 1%, confidence 30%):

| | |
|---|---|
| Frequent itemsets / rules | 1,219 (691 / 413 / 109 / 6 by size) / **950**, mined in 0.5–1.2 s in Chrome |
| Rules removed by lift > 1 | **0** — no item reaches 12% of baskets, so confidence ≥ 30% already implies lift ≥ 2.62 |
| Rule to reject | HERB MARKER THYME → ROSEMARY: first by lift (85.06), confidence 94.6%, but a six-piece set; 176 of 186 thyme baskets already hold rosemary |
| Rule to investigate | JUMBO BAG PINK → RED: 546 baskets (the most of any pair), confidence 62.8% (34.5% reversed), lift 6.77, 834th by lift |
| Later period (rules mined Dec 2010–Aug 2011, re-measured Sep–Dec 2011) | 92% keep confidence ≥ 30%, 100% keep lift > 1, only 27.5% keep 1% support |

Full analysis in `report/kosychev_a04_report.pdf`.

## Layout

| Path | What |
|---|---|
| `index.html`, `style.css`, `script.js`, `transactions.js` | Finished version (served by GitHub Pages; works from `file://` too) |
| `starter/` | Unchanged copies of the course files, TODO stubs and the course `readme.md` included |
| `report/*.diff` | Exact changes per file against `starter/` (`transactions.js` is unchanged) |
| `experiments/run_experiment.py` | Raw-log reproduction, FP-Growth reference, threshold sweep, temporal split → `results.json`, `reference_rules.json` |
| `experiments/make_figures.py` | Figures for the report |
| `experiments/make_manual_figure.py` | Composes the screenshots in `manual/` into Fig. 4 |
| `tests/check_rules.py` | Browser test (Playwright), 24 checks; output in `tests/results.json` and `tests/run_log.txt` |
| `report/kosychev_a04_report.pdf` | Report (source `report/report.md`, build `python report/build_report.py`) |
| `manual/` | Screenshots of the checks run by hand |
| `task-slide.jpg` | The assignment slide |

## Run locally

The page needs no server: open `index.html`, press **Run tests** (11/11) and **Run rules**. The same works over
http:

```
cd hw4
python -m http.server 8000
```

Then http://localhost:8000/ for the finished page and http://localhost:8000/starter/ for the skeleton.

Reproduce the numbers and the tests (`pip install pandas openpyxl mlxtend matplotlib playwright markdown`, plus a
local Google Chrome). The experiments read the raw workbook, which is not committed (23 MB): download
[online+retail.zip](https://archive.ics.uci.edu/static/public/352/online+retail.zip) and unpack
`Online Retail.xlsx` into `hw4/data_raw/` (sha256 `43465a06…676d`, the one the course export was built from).

```
python experiments/run_experiment.py     # writes experiments/results.json (~20 s after the first xlsx read)
python experiments/make_figures.py       # writes report/figures/*.png
python tests/check_rules.py              # 24 checks against the live page
python report/build_report.py            # rebuilds the PDF
```

The browser test is what ties the two implementations together: the Apriori in `script.js` and mlxtend's
FP-Growth must produce the same rules with the same counts at every stored threshold setting.
