# Who Should Fill the Slots Recency Leaves Empty? A Planned Test of LLM Asset Embeddings vs Categorical Features on FAR-Trans

<div class="meta">
Student: Aleksei Kosychev &nbsp;&nbsp; Team: name<br>
Email: amkosychev@edu.hse.ru &nbsp;&nbsp; Date: 2026-11-17<br>
Assignment: Session 1 — Research Question and Minimal Study Plan (LLM4Rec, Module 2)
</div>

## Abstract

Retail-investment recommendation on FAR-Trans is dominated by repetition: the median customer
has bought 2 distinct assets, so a recency ranking of past buys fills only ~3 of the 10
recommendation slots, and the remaining ~70% are decided by whatever *fills* the list. This
report fixes one research question for the shared FAR-Trans ranking task (Encoder track, partly
run): whether frozen LLM text embeddings of as-of asset metadata, used to fill those slots,
improve nDCG@10 against next-month purchases over the identical pipeline with the starter's
categorical features, a TF-IDF text space, or a popularity fill. The non-LLM arms are already
run on nine quarterly cutoffs (7,125–14,926 warm users each): a content-based fill *loses* to a
popularity fill on the subgroup whose labels contain an asset new to them (0.246 vs 0.281
nDCG@10, paired mean difference −0.036, losing 974 user-windows against 424 wins), while an
oracle fill shows +0.19 of headroom on the same subgroup. The gap the LLM arm must close is
therefore real and measurable; the arm itself is planned for Session 2 with a pinned embedding
model, ~7,200 asset texts to encode, and a budget under one US dollar.

*Index Terms* — financial asset recommendation, large language models, text embeddings, content-based filtering, nDCG, FAR-Trans

## 1. Introduction

**Problem.** The shared course task asks for a Top-10 asset list per evaluated FAR-Trans
customer, scored by nDCG@10 against purchases in the following 30 days, under a past-only
protocol: as-of metadata, pre-cutoff buy history, and an eligibility rule requiring one
pre-cutoff close price [1], [3]. Profiling the raw archive (388,048 transactions of 29,090
customers over 320 traded assets, 2018-01-02 … 2022-11-30; metadata for 806 ISINs: 285 stocks,
312 mutual funds, 209 bonds) shows the task's structural property: across nine quarterly cutoffs
the median customer holds 2 distinct past assets, 91–94% of customers hold fewer than 10, so the
recency prefix of the strongest provided baseline (`latest_buy`) covers on average only ~3 of the
10 slots — **68–70% of every list is fill**. Meanwhile 28.7% of all label pairs are assets the
customer never bought before (93.2% of them stocks), and those are exactly the pairs the prefix
cannot reach. Whatever rule fills the slots is therefore the part of the system that decides a
quarter of the achievable score, yet the provided baselines fill with raw popularity.

**Research question** (Encoder track; the control arms are run, the LLM arm is planned):

> We test whether **frozen LLM text embeddings of the as-of asset description** (intervention)
> improve **nDCG@10 against observed purchases in the next 30 days** (outcome) over **the same
> pipeline whose fill space is the starter's categorical tokens, a TF-IDF text space, or plain
> popularity order** (control) for **the warm-cohort FAR-Trans evaluation customers, overall and
> on the subgroup whose label window contains at least one new-to-them asset** (population).

Control and intervention differ only in the item vector space; users, candidate universe, cutoff,
horizon, prefix and metric are identical. A direction hypothesis: any gain should concentrate in
the with-new-asset subgroup, where the fill is the only mechanism that can hit a label; on
customers whose labels are all repeat purchases every fill ties by construction.

**Research importance.** Published LLM4Rec gains from text representations are measured on
news, books and e-commerce, where items carry rich descriptions and IDs carry little signal
[4], [5], [6]. FAR-Trans is the opposite shape: terse categorical metadata, heavy purchase
repetition, and a popularity/recency prior that is hard to beat. Whether a semantic embedding of
this metadata helps *at all* — and whether it can repair the measured failure of sparse text
features against a popularity fill — is unknown and informative in both directions. A positive
result gives a cheap, training-free way to surface new-to-user assets; a negative result says
that in repeat-dominated financial recommendation the fill budget is better spent on priors than
on semantics, which contradicts the default "embeddings improve content-based recommendation"
assumption imported from other domains.

**Business value.** The named stakeholder is the **product owner of a retail-investment
recommendation feature** deciding whether to add LLM-embedding infrastructure (model hosting,
versioning, re-embedding on metadata updates) or to keep deterministic, auditable categorical
vectors. The decision changes with either sign of the result: adoption if embeddings beat the
popularity fill on the new-asset subgroup at cent-level cost, rejection and reallocation of the
budget to candidate priors or reranking if they do not. A secondary stakeholder is the
**compliance reviewer**: a semantic component whose behaviour is measured against observed
purchases under an as-of protocol is auditable in a way a generative LLM advisor is not. No
profit, return or user-acceptance claim is made or implied anywhere in this work.

## 2. Related Work

| Study (authors, year, venue) | Method | Data and evaluation | Limitations and connection to the question |
|---|---|---|---|
| Sanz-Cruzado, Droukas, McCreadie, 2024, IJCAI Fin-RecSys WS [1] | FAR-Trans dataset release plus a benchmark of 11 classical FAR algorithms | Retail transactions, prices, profiles from a large European institution; temporal evaluation | No LLM-based method in the benchmark; supplies the data and the classical reference points this study builds on |
| Wu et al., 2024, World Wide Web [4] | Survey; taxonomy into discriminative and generative LLM4Rec | Reviews methods using LLM text representations and external knowledge for user–item correlation | Identifies textual-feature representation as the key lever but reports no measurement on financial data or repeat-purchase regimes |
| Liu, Chen, Sakai, Wu, 2024, WSDM (ONCE) [5] | Deep layers of open-source LLMs as content encoders; closed-source LLMs enrich training text | News and book recommendation; up to 19.32% relative improvement over SOTA baselines | Rich-text domains; trained end-to-end; leaves open whether *frozen* embeddings of terse categorical metadata help, and whether they survive a popularity prior |
| Li et al., 2023, KDD (RecFormer) [6] | Items flattened into "sentences" of key–value attributes; Longformer-like bidirectional encoder; pretrain + fine-tune | Six sequential-recommendation datasets; strongest gains in low-resource and cold-start settings | Requires task-specific pretraining and fine-tuning; sequential next-item objective, not a fixed-protocol Top-10 against observed purchases |
| Spadea, Seneviratne, 2025, ICAIF (FLARKO) [7] | LLM fine-tuned with Kahneman–Tversky Optimization over knowledge graphs of histories and trends | Evaluated on FAR-Trans; behavioural alignment and joint profitability | The LLM is a fine-tuned generative decision-maker; the contribution of the item representation itself is not isolated, and the metric is not nDCG@10 against next-period purchases |

**Nearest prior work.** FLARKO [7] is the closest study: same dataset, LLM inside. It answers
"can a fine-tuned LLM produce behaviourally aligned and profitable asset recommendations" and
leaves open the question asked here: whether a *representation-level* change — swapping the item
vector space for LLM embeddings while everything else stays fixed — moves ranking quality against
observed purchases, and at what cost. Under our condition (as-of protocol, frozen embeddings,
nDCG@10, popularity/recency controls) we did not find a study reporting the effect of LLM text
embeddings on next-purchase ranking over FAR-Trans; no stronger novelty claim is made.

## 3. Method

### 3.1 One actual data example

Read directly from the public archive [2]; the customer id is truncated to a non-identifying
prefix.

```
source: FAR-Trans public release [2], transactions.csv   record id: transactionID 9013789
fields: customerID, ISIN, transactionID, transactionType, timestamp, totalValue, units, channel, marketID
values: "6B15…" (truncated), GRF000045002, 9013789, Buy, 2021-06-02, 999.9910544, 52.496, Branch, AEDAK
window: label window of the 2021-06-01 cutoff (2021-06-01 … 2021-06-30); this purchase is the
        customer's only label asset and is new to her (not in her pre-cutoff history)
```

Per-field provenance, with the leakage check:

| Field | Status (timestamp) | Use and leakage check |
|---|---|---|
| customerID | observed, pseudonymised | join key only; never a feature |
| ISIN, transactionType, timestamp | observed (2021-06-02) | a pre-cutoff `Buy` builds history; an in-window `Buy` is a label — labels never enter inputs |
| totalValue, units | observed (2021-06-02) | never read: no spending, revenue or return is computed |
| channel | observed | unused |
| assetName, assetCategory, assetSubCategory, marketID | observed, as-of row dated 2018-01-02 (latest strictly before cutoff) | the exact text both encoders read: "DHLOS SYNTHESIS BEST FUND OF FUNDS ΜΕΤΟΧΙΚΟ", MTF, Money Market, AEDAK |
| sector, industry | **missing** (empty for this fund) | omitted from the text; 161 of 806 assets have 5 populated categorical fields, 56 have only 2 |
| closePrice 18.9924 | observed (2021-05-31, last pre-cutoff) | eligibility only (≥1 pre-cutoff price); never a score |
| customerType, riskLevel, investmentCapacity | observed, as-of row dated 2018-01-02 | display-only context; never a filter or score |
| lastQuestionnaireDate | anomaly: 2021-12-16, *after* the cutoff, inside the as-of row | potential future information; excluded from all arms |
| Predicted_* customer fields | predicted/estimated by the data provider | not used |
| Label assets {GRF000045002} | observed future purchase | evaluation truth only |

Every input field is knowable strictly before the cutoff; the one anomalous field is excluded.
For this customer all four provided baselines score 0.0 — the label fund sits at rank 54 in the
popularity order and rank 39 in the CBF order — because no prefix slot can hit a fund she never
bought. She is the population the fill rule serves.

### 3.2 Design: change the fill, freeze everything else

The pipeline keeps the recency prefix of the provided `latest_buy` baseline (the customer's own
past buys, most recent first) and changes only how the remaining Top-10 slots are filled. Arms:

- `pop_fill` — popularity order (control; reproduces the starter's `latest_buy` exactly, verified:
  0 of ~106k user-lists differ);
- `tokens_fill` — cosine in the starter's own categorical token space (its CBF features);
- `tfidf_fill` — cosine over unigram TF-IDF of the asset text (name + categorical attributes);
- `tfidf_topN` — the N most popular unseen assets, re-ordered by the TF-IDF cosine
  (N ∈ {15, 20, 30, 50, 100}); isolates the candidate prior from the order;
- `oracle_fill` — the customer's new-to-user label assets (ceiling, uses the label; not a method);
- `llm_fill` (**planned**) — identical to `tfidf_fill` with the vector space replaced by frozen
  embeddings of the same as-of asset text from an LLM-family encoder (candidate:
  `intfloat/multilingual-e5-base`, open weight, version pinned; fallback: a hosted
  `text-embedding-3-small`; the exact model and version will be recorded in the run's
  `protocol.json`).

**Baselines and fair comparison.** All four provided baselines (`popularity`, `cbf`,
`latest_buy`, `latest_buy_or_sell`) are retained and reported on the same protocol [3]. Every arm
sees the same users, the same pre-cutoff information, the same eligible universe and the same
time split; the only difference between control and intervention is the fill's item vectors.
Evaluation: nine quarterly cutoffs 2020-09-01 … 2022-09-01, 30-day horizon, warm cohort
(7,125 → 14,926 users, 732 → 802 eligible assets). **Measure:** common nDCG@10 over the shared
Top-10, reported on all warm users (empty-truth users frozen at 0.0 and kept), on labelled users,
and on the with-new-asset subgroup; plus wall-clock runtime and cost per arm. nDCG@10 here
measures ranking accuracy against observed purchases — a ranking metric, not investment quality
or profitability.

**Contribution check (ablation).** Removing the LLM component restores `tfidf_fill`/`tokens_fill`
with no other change, so the ablation isolates exactly the item representation. Secondary
ablations: the shortlist size N (is the effect in the candidate prior or the order?) and the text
fields fed to the encoder (name only vs full attributes).

**Feasibility.** Session 2: encode ≤ 802 as-of asset texts per cutoff (≈7,200 texts across all
nine, each < 60 tokens; cent-level API cost or free open-weight inference on CPU), cache
embeddings keyed by (model, text hash), and re-run the existing pipeline — the pilot below took
425 s for all nine cutoffs with the standard library only, so the LLM arm adds minutes, not
hours. Deterministic embeddings need no seeds; model name and version are recorded, and the run
is repeated once to confirm stability. If a generative scoring variant is added later, three
repeats per cutoff with reported variance are budgeted. No LLM judge is used anywhere; ground
truth is observed purchases only.

**Risks.** *Leakage*: the as-of rule and the excluded questionnaire-date anomaly above; history
and labels never share rows. *Contamination*: the embedding model's pretraining may contain
public descriptions of these assets — mitigated by feeding only as-of metadata text and by the
frozen, training-free setup; pretraining cannot contain the future purchase labels. *Privacy*:
ids are pseudonymised; records are truncated in the report. *Ethics*: recommendations are ranked
against past behaviour; no suitability, advice or profitability claim is made.

## 4. Experiments

**What was run.** The full pipeline and all non-LLM arms on nine quarterly cutoffs
(`experiments/pilot_control_fill.py`, numbers in `pilot_control_fill.json`): 106k user-lists,
12,163 labelled user-windows, 5,051 of them with at least one new-to-user label asset (7,276
new-to-user label pairs). The `llm_fill` arm is **planned**, not run; no numbers for it exist
yet and none are reported. Both structural invariants hold on every cutoff: `pop_fill`
reproduces the starter's `latest_buy` for 0 mismatched users, and customers whose labels are all
repeat purchases score identically under every fill (0 invariant violations) — the fill cannot
help or harm them.

**Baseline context.** On labelled users (macro mean over the nine cutoffs) the provided baselines
rank `latest_buy` 0.561 > `latest_buy_or_sell` 0.552 > `cbf` 0.312 ≈ `popularity` 0.307 — the
recency prefix is the strongest classical component, which is why it is frozen in all arms.

**Pilot results (macro mean over 9 cutoffs; min–max in parentheses).**

| Arm (fill rule) | nDCG@10 labelled (12,163) | nDCG@10 with-new (5,051) | New-to-user pairs in Top-10 |
|---|---|---|---|
| `pop_fill` (control) | 0.561 (0.465–0.623) | 0.281 (0.203–0.365) | 1,472 |
| `tokens_fill` (starter CBF space) | 0.547 (0.460–0.621) | 0.248 (0.194–0.295) | 842 |
| `tfidf_fill` (text TF-IDF) | 0.546 (0.459–0.617) | 0.246 (0.197–0.299) | 796 |
| `tfidf_top20` (popularity prior + TF-IDF order) | 0.562 (0.469–0.625) | 0.284 (0.218–0.354) | 1,526 |
| `oracle_fill` (ceiling, uses labels) | 0.639 (0.563–0.690) | 0.470 (0.402–0.541) | 4,463 |

Three findings. (i) **A content-based fill loses to a popularity fill exactly where it should
win**: on the with-new subgroup both sparse spaces score ~0.25 against popularity's 0.281, and
the paired comparison is consistently negative (424 better / 974 worse / 3,653 equal user-windows,
mean difference −0.036, negative on all nine cutoffs). Sparse categorical semantics do not just
fail to help — the personalised order actively demotes assets that popularity would have hit.
(ii) **The knob that matters so far is the candidate prior, not the order**: re-ordering the
20 most popular unseen assets by TF-IDF cosine matches the popularity fill on both groups, so the
TF-IDF order is harmless on a good prior and harmful on the full universe. (iii) **There is real
headroom**: the oracle fill adds +0.19 nDCG on the with-new subgroup, and the all-user means are
dominated by the ~89% empty-truth convention, so subgroup reporting is essential.

Runtime and cost of what was run: 425 s wall-clock for nine cutoffs (≈47 s each, standard library
only); zero monetary cost. The planned LLM arm adds one embedding pass over ≤ 7,200 short texts.

## 5. Discussion

**Team discussion** *(reported as it happened; joint points marked)*. *Options considered:* a
Ranker-track question (LLM reranks a fixed candidate list — attractive because the starter ships
a ranker seam, but it conflates the LLM's ordering with the classical candidate generator and is
hard to cost per user); a Hybrid question (dropped jointly: two LLM components cannot be ablated
cleanly within the remaining sessions); a fine-tuning question (dropped: budget and a training
pipeline we cannot verify end-to-end); and the adopted Encoder question. *Agreement and
differences:* we converged (joint) on keeping all four provided baselines, the common Top-10 and
nDCG@10, and on reporting the with-new subgroup separately because repetition hides fill effects
in the all-user mean. Still unresolved (joint): whether the with-new subgroup should be the
primary population in the final write-up or a secondary analysis — the shared task's evaluated
users are not yet fixed, and the answer may change once they are. *My contribution:* I proposed
the prefix/fill decomposition of `latest_buy`, the oracle fill as a measurable ceiling, the
invariant that only-repeat users must tie across arms (it caught one early indexing bug), and the
popularity-prior × personalised-order factorial (`tfidf_topN`); I also ran the profiling and the
pilot reported here. *My position and reasons:* I adopt the Encoder track because it isolates
exactly one change — the item vector space — inside a pipeline that already runs and is
invariant-tested, and because its cost (≈7,200 embeddings) fits the remaining sessions with
margin; the ranker alternative needs a defensible candidate generator before its LLM effect can
even be attributed.

**Threats to validity.** Label sparsity: only ~11% of warm user-windows are labelled, so all-user
means are convention-dominated and both means are always reported. Subgroup skew: 93% of
new-to-user labels are stocks, so any subgroup conclusion may not transfer to funds or bonds.
Period and institution: nine quarters of 2020–2022 from one European institution; no claim
beyond it. The subgroup is defined by a property of the label window; the paired design keeps it
honest (same users across arms), but the selection itself is descriptive, not causal.

**Meaning of the possible outcomes.** Positive (LLM fill > popularity fill on the subgroup):
frozen semantics repair the measured failure of sparse text features at cent-level cost —
adopt. Null or negative: the fill budget belongs to priors and recency, not semantics — the
popularity prior stands, the infrastructure spend is avoided, and the ranker track becomes the
more promising next question. Either outcome answers the question; none will be hidden.

## 6. AI Usage Disclosure

**AI tools used.** Kimi Code CLI (Moonshot AI) in this session; earlier groundwork used the same
assistant in a prior session. **How AI was used:** *search/reading* — locating and opening the
five cited studies; *code* — `experiments/profile_data.py`, `find_record.py` and
`pilot_control_fill.py` plus their runs; *analysis* — aggregating the pilot numbers and drafting
this report. **What I personally verified:** the record of §3.1 against the raw CSVs verbatim;
both invariants of §4 on all nine cutoffs; the `pop_fill` = `latest_buy` identity; every number
in §4 against `pilot_control_fill.json`; and that each cited paper's abstract says what §2 claims
(all five were opened). **What I trusted without verification:** the starter's as-of protocol
implementation and the dataset's own anonymisation. **Session log:**
`kosychev_session1_session.json`, the tool-native export of this session, submitted alongside.

## References

[1] J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An Investment Dataset for Financial Asset Recommendation," in *Proc. IJCAI-24 Workshop on Recommender Systems in Finance (Fin-RecSys)*, Jeju, 2024. [Online]. Available: https://arxiv.org/abs/2407.08692

[2] University of Glasgow, "FAR-Trans dataset," research data repository. [Online]. Available: https://researchdata.gla.ac.uk/1658/

[3] S. Jin, "FAR-Trans student starter," LLM4Rec course repository, GitHub, 2026. [Online]. Available: https://github.com/dryjins

[4] L. Wu, Z. Zheng, Z. Qiu, H. Wang, H. Gu, T. Shen, C. Qin, C. Zhu, H. Zhu, Q. Liu, H. Xiong, and E. Chen, "A Survey on Large Language Models for Recommendation," *World Wide Web*, vol. 27, no. 4, p. 60, 2024. [Online]. Available: https://arxiv.org/abs/2305.19860

[5] Q. Liu, N. Chen, T. Sakai, and X.-M. Wu, "ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models," in *Proc. 17th ACM Int. Conf. Web Search and Data Mining (WSDM)*, 2024, pp. 452–461. [Online]. Available: https://arxiv.org/abs/2305.06566

[6] J. Li, M. Wang, J. Li, J. Fu, X. Shen, J. Shang, and J. McAuley, "Text Is All You Need: Learning Language Representations for Sequential Recommendation," in *Proc. 29th ACM SIGKDD Conf. Knowledge Discovery and Data Mining (KDD)*, 2023, pp. 1258–1267. [Online]. Available: https://arxiv.org/abs/2305.13731

[7] F. Spadea and O. Seneviratne, "Aligning Language Models with Investor and Market Behavior for Financial Recommendations," in *Proc. 6th ACM Int. Conf. AI in Finance (ICAIF)*, 2025. [Online]. Available: https://arxiv.org/abs/2510.15993
