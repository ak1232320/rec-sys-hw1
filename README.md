# LLM4Rec A01 — Random Lunch Generator: fixing missing icons

Aleksei Kosychev (amkosychev@edu.hse.ru), HSE LLM4Rec, Week 1.

**Live demo:** https://ak1232320.github.io/rec-sys-hw1/

Starter code and prompt: [dryjins/RecSys-LLMs `week1/`](https://github.com/dryjins/RecSys-LLMs/tree/main/week1) (commit `d8a178a`).
Task: understand and fix "images sometimes not displaying".

**Root cause.** Ramen (`fa-bowl-hot`), Pasta (`fa-pasta`) and Soup (`fa-bowl`) use icon classes that do not exist
in Font Awesome 6.4.0 Free, so 3 of 12 random picks (~25%) show an empty box. The prompt never pinned the icon
library version or asked to check icon names.

| | Starter | Fixed |
|---|---|---|
| Icons rendered | 9 / 12 | 12 / 12 |
| Blank rate, 10,000 clicks | 24.91 % | 0 % |

## Layout

| Path | What |
|---|---|
| `index.html`, `prompt.md` | Fixed version (served by GitHub Pages) |
| `starter/` | Unchanged copies of the course files |
| `fixed/` | Same as root; compare with `starter/` |
| `report/index.diff`, `report/prompt.diff` | Exact changes |
| `tests/check_icons.py` | Headless Chrome test (Playwright); output in `tests/run_log.txt` |
| `report/kosychev_a01_report.pdf` | Report (source `report/report.md`, build `python report/build_report.py`) |
| `manual-verification/` | Screenshots of the manual CSS check |

## Run locally

Open `starter/index.html` or `fixed/index.html` in a browser (internet needed for the Font Awesome CDN).
Test: `pip install playwright` then `python tests/check_icons.py` (uses local Google Chrome).
