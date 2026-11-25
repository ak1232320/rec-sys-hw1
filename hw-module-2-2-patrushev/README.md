# LLM4Rec project on FAR-Trans, Session 2 (HW8): repeat-aware hybrid, Hybrid track

Konstantin Patrushev, HSE LLM4Rec, team Name. Session 2 milestone "Make It Run": the encoder + LLM-ranker hybrid planned
in Session 1, implemented and compared with the four provided baselines on the starter's default protocol
(100 warm demo users, cutoff 2021-06-01, 30-day window, repeats allowed, starter nDCG@10).

Arms: A0 = latest buy (rule + popularity fill), A1 = rule + LLM-encoder fill, A2 = LLM ranker over the customer's past
assets + 10 popular never-bought assets, A3 = LLM ranker over past assets + 10 encoder candidates (the hybrid).
Experiment commits: `f0dd031` (Tables 1-2 of the report), `6a729f8` (Table 3, seeded sample of 60 warm customers
with purchases). The repository history before the split commit also contains the Session 1 (HW7) files.

| Path | What |
|---|---|
| `src/hw2_hybrid.py` | the experiment: protocol through the starter's own functions, encoder, ranker, evaluation, checks |
| `src/hw2_analysis.py` | independent nDCG recompute from the raw CSV, per-arm diagnostics, the hand-check customer |
| `src/build_report_hw2.py` | renders the report (`report/patrushev_konstantin_session2_report.pdf`) from `evidence/` |
| `evidence/hw2_metrics.json` | metrics, failed cases, runtime, model revisions, prompt instruction, hardware (100 demo users) |
| `evidence/hw2_labelled_sample_60.json` | the same for the seeded labelled sample |
| `evidence/hw2_analysis.json`, `evidence/hw2_labelled_analysis.json` | diagnostics for both runs |
| `evidence/hw2_rule_only_10404users.json` | A0/A1 on all 10,404 warm users |
| `evidence/team_hw2.json`, `evidence/literature.json` | team section and verified references used by the report |
| `baseline/far-trans`, `baseline/far-trans-llm` | course starter and LLM baselines, unmodified copies of dryjins/RecSys-LLMs `project/` |
| `report/pack_session_hw2.py` | derived AI-session export for the LMS |

Models (open weights, pinned): encoder `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`,
ranker `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (greedy, thinking off, max 256 new tokens, bfloat16).
Dependencies: Python 3.12, torch 2.11.0, transformers 4.57.6, sentence-transformers 5.7.0, numpy, polars.
Hardware: Apple M5 (24 GB), PyTorch MPS.

Reproduce:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python polars numpy scikit-learn torch==2.11.0 transformers==4.57.6 sentence-transformers==5.7.0
# data: download FAR-Trans.zip from https://researchdata.gla.ac.uk/1658/ and unzip into data/raw/
# starter reference runs (the experiment asserts its users, latest-buy lists and baseline scores against them)
(cd baseline/far-trans && ../../.venv/bin/python -I starter.py run --data-dir ../../data/raw/FAR-Trans --output-dir ../../outputs/baseline_demo100)
(cd baseline/far-trans && ../../.venv/bin/python -I starter.py run --data-dir ../../data/raw/FAR-Trans --output-dir ../../outputs/baseline_warm_all --max-users 0)
.venv/bin/python -I src/hw2_hybrid.py --dtype bfloat16                                    # 100 demo users, all arms (~48 min on M5)
.venv/bin/python -I src/hw2_hybrid.py --max-users 0 --labelled-sample 60 --out outputs/hw2_labelled   # Table 3 (~33 min)
.venv/bin/python -I src/hw2_hybrid.py --max-users 0 --rule-only --out outputs/hw2_rule_warm         # A0/A1 on all warm users
.venv/bin/python -I src/hw2_analysis.py && .venv/bin/python -I src/hw2_analysis.py outputs/hw2_labelled hw2_labelled_analysis.json
.venv/bin/python -I src/build_report_hw2.py --pdf --commit "f0dd031 (Tables 1-2), 6a729f8 (Table 3)"
```

The FAR-Trans data (CC BY 4.0) and the model outputs are not redistributed here.
