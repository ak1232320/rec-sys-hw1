# Session 2 (HW2): LLM-embedding fill on FAR-Trans — Aleksei Kosychev

Individual implementation of my Session 1 question (Encoder track). Team: name.
Report: [report/kosychev_session2_report.pdf](report/kosychev_session2_report.pdf)
(source `report/report.md`). The task statement is [homework2.md](homework2.md).

**What changes.** The provided `latest_buy` baseline = the user's past buys, most recent
first (the *prefix*), then the most popular assets (the *fill*). My method keeps the prefix
and fills the free slots with the unseen eligible assets closest to the user in a frozen
**LLM text-embedding space** (`Qwen/Qwen3-Embedding-0.6B`, revision `97b0c614`) of the
as-of asset text `name | category | subcategory | sector | industry | market`. Matched
controls change only the vector space: popularity (= `latest_buy`), the starter's
categorical tokens, TF-IDF of the same text, and `intfloat/multilingual-e5-base` (a dense
encoder that is not an LLM).

## Layout

| Path | What |
|---|---|
| `starter/` | course starter, byte-identical to `dryjins/RecSys-LLMs@799e54c` `project/far-trans` (not modified) |
| `src/export_texts.py` | exports the 736 unique as-of asset texts of the nine cutoffs → `kaggle/input/asset_texts.json` |
| `kaggle/embed_assets.py`, `kaggle/build_kernel.py` | Kaggle notebook (T4 GPU): embeds the texts with both pinned models → `kaggle/output/*.npy`, `manifest.json`, log |
| `src/fill.py` | prefix + fill arms (pure functions) |
| `src/run_baseline.py` | runs the unchanged starter CLI (four baselines), timed → `outputs/baseline/` |
| `src/run_demo.py` | HW2 protocol (100 users, 2021-06-01, 30 days): four baselines + all arms → `outputs/demo/` |
| `src/run_dev.py` | development evaluation, nine quarterly cutoffs, all warm users → `outputs/dev/dev_metrics.json` |
| `src/diagnose.py` | failure analysis of the fills on one cutoff → `outputs/dev/diagnose_<cutoff>.json` |
| `tests/test_fill.py` | unit tests, starter cross-check, future-blind test |
| `manual/CHECKLIST.md` | the hand checks for the report's verification section |

## Reproduce

Data: the FAR-Trans CSVs already in `../hw-module-2-1/data_raw/FAR-Trans/` (CC BY 4.0).
Local: Python 3.14, numpy 2.4 (scoring); the starter needs only the standard library.

```bash
cd hw-module-2-2
python src/run_baseline.py                  # unchanged starter, four baselines
python src/export_texts.py                  # asset texts for the notebook
python kaggle/build_kernel.py               # Kaggle kernel folder with the texts inlined
kaggle kernels push -p kaggle/kernel        # runs on Kaggle (GPU, internet for the HF download)
kaggle kernels output alekseykosychev/far-trans-asset-embeddings -p kaggle/output
python src/run_demo.py                      # HW2 table
python src/run_dev.py                       # nine-cutoff development table (~10 min)
python src/diagnose.py 2021-06-01
python -m unittest discover -s tests -v
python -m unittest discover -s starter -p "test_*.py"
```

The embeddings are committed (`kaggle/output/`), so every step after the Kaggle one
reruns without a model. No model is downloaded or run locally.

## Results

See the report. In short: on the HW2 100-user sample the LLM fill scores highest
(nDCG@10 0.0294 all users / 0.2672 on 11 users with purchases), but the whole gain comes
from one customer; on the nine development cutoffs (12,163 labelled user-windows) it is
the **worst** fill on the users who buy something new to them — it moves the fill into
the long tail of rarely bought assets. A negative result, reported as such.
