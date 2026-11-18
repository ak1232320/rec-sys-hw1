# Reordering What Customers Already Hold: A Planned Test of LLM Reranking for Repeat Purchases on FAR-Trans

Mark Starchenko · Team "name" (A. Kosychev, K. Patrushev, M. Starchenko, N. Vaal)
HSE University, Graduate School of Business, mastarchenko@edu.hse.ru
Session 1 report, LLM4Rec, 18 November 2026

## Abstract

The strongest provided FAR-Trans baseline, `latest_buy`, lists a customer's own past assets by recency and fills the remaining slots with popular assets. In our pilot over nine quarterly cutoffs, 58% of labelled user-windows contain only repeat purchases, and a perfect order of past assets would lift nDCG@10 on labelled users from 0.561 to 0.750, while a perfect fill reaches 0.639 and a logistic-regression order reaches 0.566. We therefore plan a Ranker-track test of whether zero-shot reranking of past assets by an open-weight LLM improves nDCG@10 over a logistic-regression ranker with the same information for warm FAR-Trans customers. The LLM arm is planned for Session 2; the non-LLM arms, one raw record with a leakage check and a written team proposal are reported here.

*Index Terms*: financial asset recommendation, repeat consumption, LLM reranking, nDCG, FAR-Trans

## I. Introduction

**Problem.** The shared course task asks for a Top-10 asset list per evaluated FAR-Trans customer [1], scored by binary nDCG@10 against the assets the customer buys in the following 30 days, with past-only inputs [7]. The provided baselines are popularity, content-based filtering (CBF), `latest_buy` and `latest_buy_or_sell`; on the larger cohorts the two recency baselines score highest [7]. A `latest_buy` list has two parts: a *prefix* with the customer's own eligible past assets, most recently bought first, and a *fill* of never-bought assets in popularity order. My teammate A. Kosychev studies the fill (Encoder track, LLM embeddings of asset text). This report studies the prefix.

**Research question** (Ranker track; non-LLM arms run as a pilot, LLM arm planned):

> We test whether **zero-shot reranking of a customer's own previously bought assets by an open-weight LLM**, given the dated Buy and Sell events and as-of asset metadata (intervention), improves **nDCG@10 against purchases in the next 30 days** (outcome) over **a logistic-regression ranker that receives the same information as features** (control) for **warm FAR-Trans customers, overall and for customers with more than ten past assets** (population).

Intervention and control reorder the same prefix and keep the same popularity fill, users, eligible universe, cutoff and metric. No direction is assumed. This question is not one of the two provided LLM baselines [8]: their ranker reorders 50 CBF candidates, mostly assets the customer never bought, and is compared with the CBF order; ours reorders only assets the customer already bought and is compared with a learned ranker.

**Research importance.** Work on repeat consumption shows that repeat items carry much of the accuracy of next-basket and session recommenders [3], [4], and LLM rankers were reported to perceive the order of a user's history poorly [5]. Rebuying a held asset is a timing question: when does a customer top up a position bought months ago? Our pilot (Section IV) shows that simple classical signals capture little of it: ordering by purchase count loses to recency, and a learned ranker gains 0.005 of the 0.188 available. Whether an LLM reading the dated event sequence does better is open for financial histories, and a null result would also be informative: it would show that the weakness found in [5] carries over to sparse investment histories.

**Business value.** The named stakeholder is the product owner of a "top up your holdings" shortlist in a retail investment app, or the advisor who reviews that shortlist before a client call. The decision is whether per-customer LLM inference is worth paying for over a logistic-regression or recency order. A positive result would justify it for heavy customers; a null or negative result keeps the cheaper ranker and moves the LLM budget to the fill. A compliance reviewer gains a narrow, auditable component: the ranker can only reorder assets the customer already held. nDCG@10 measures agreement with observed purchases, not suitability, return or profit, and no such claim is made.

## II. Related Work

**Table I. Studies checked against source texts (18.11.2026)**

| Study | Method | Data and evaluation | Limitations and connection to our question |
|---|---|---|---|
| Sanz-Cruzado, Droukas, McCreadie, 2024, Fin-RecSys @ IJCAI [1] | Public dataset release; benchmark of eleven FAR algorithms | Prices and retail investor transactions from a large European institution | Defines the data and classical reference points; no LLM method and no repeat/explore split |
| Ren et al., 2019, AAAI (RepeatNet) [3] | Encoder-decoder with a repeat-explore mechanism that chooses items from the user's history | Three session datasets, MRR and Recall; gains grow with the repeat ratio | Trained neural model on session data; its repeat mode is the part we isolate, with an LLM instead of a trained decoder |
| Li et al., 2023, ACM TOIS [4] | Repeat and explore metrics for next-basket models | Re-evaluation of published NBR methods | Shows that progress must be split into repeat and explore parts; we apply the same split to FAR-Trans |
| Hou et al., 2024, ECIR (LLMRank) [5] | Zero-shot LLM ranking conditioned on the interaction sequence | Two public datasets; candidates from other generators | LLMs struggle with the order of the history and show position and popularity bias; our candidates are the user's own past items and order matters |
| Spadea, Seneviratne, 2025, arXiv (FLARKO) [6] | LLM with knowledge graphs of histories and trends, fine-tuned with KTO | FAR-Trans; behavioural alignment and joint profitability | Fine-tuned generative recommender; repeat ordering is not isolated and the metric is not nDCG@10 against next-period purchases |

Table I lists the five studies. The nearest prior work is LLMRank [5]: it asks whether an LLM can rank candidates from a user's interaction sequence and finds that it can, with weak use of order. It leaves open the case where the candidates are the user's own past purchases, histories are short and irregular, and the comparator is a learned ranker with the same information. RepeatNet [3] answers the repeat question with a trained model, not with a frozen LLM, and FLARKO [6] uses FAR-Trans with a fine-tuned LLM but does not isolate repeat ordering. We did not find a study that reports LLM reranking of a customer's own past assets against a same-information learned ranker on next-month purchases in FAR-Trans; we make no broader novelty claim.

## III. Method

### A. One actual data example

Read from the public archive [2], `transactions.csv`; the customer id is truncated.

```
source: FAR-Trans public release, transactions.csv
record id: transactionID 9049523
fields: customerID, ISIN, transactionID, transactionType, timestamp, totalValue, units, channel, marketID
values: 5087... (truncated), GRS003003035, 9049523, Buy, 2021-06-08, 996.41, 370.0, Phone Banking, XATH
window: label window of cutoff 2021-06-01 (2021-06-01 to 2021-06-30)
```

This is a repeat purchase. The customer bought the same ISIN on 2020-03-27, 2020-04-01 and 2020-04-06 and made 24 Buy rows on five distinct ISINs before the cutoff (11, 7, 3, 2 and 1 rows; latest on 2021-04-27). In `latest_buy` the ISIN, last bought 14 months earlier, is fifth of the five past assets, so the user's nDCG@10 is 1/log2(6) = 0.387 (also in the starter's `per_user.csv`); ordered by purchase count it would be third (0.5), and a perfect prefix order would give 1.0. Table II gives the status of each field.

**Table II. Fields of the record and the leakage check**

| Field | Status (timestamp) | Use |
|---|---|---|
| customerID | observed, pseudonymised | join key only |
| ISIN, transactionType, timestamp | observed (2021-06-08) | label only; the 24 pre-cutoff Buy rows form the input |
| totalValue, units | observed (2021-06-08) | never read |
| channel, marketID | observed | unused |
| assetName, assetCategory, assetSubCategory | observed, as-of row 2018-01-02 | "NATIONAL BANK OF GREECE", Stock, Large Cap: prompt text and LR features |
| sector, industry | missing | stays "unknown" |
| closePrice 2.584 | observed (2021-05-31) | eligibility only |
| customerType, riskLevel, investmentCapacity | observed, as-of row 2021-03-10 | Professional, Aggressive, CAP_80K_300K: context only |
| lastQuestionnaireDate | observed, 2018-12-10 | unused |

Every input row is dated before 2021-06-01; the label row enters only the truth set. The profile fields here are observed, but other customers carry `Predicted_` values, which are estimates by the data provider and will be marked as such in prompts. Kosychev found a customer whose `lastQuestionnaireDate` lies after the cutoff inside an as-of row; that field is excluded from all arms.

### B. Design

**Users and data.** Warm customers at a cutoff: an as-of profile row and at least one eligible past Buy. Assets are eligible with as-of metadata and a positive pre-cutoff close price; exact duplicate rows and negative transaction ids are excluded, as in the starter [7]. Development uses the nine quarterly cutoffs 2020-09-01 to 2022-09-01 with a 30-day horizon (Kosychev's protocol, which I reproduced) and the course demo slice; the official protocol, pending from the instructor, replaces them for final numbers. Subgroups: repeat-only label windows, windows with a new-to-user asset, customers with more than ten past assets.

**Arms.** All arms keep the popularity fill and change only the prefix order: (A0) recency, which is `latest_buy`; (A1) purchase count; (A2) the control, pointwise logistic regression on as-of features of each (customer, past asset) pair: days since the last and first Buy, Buy count, mean gap between Buys, a Sell after the last Buy, recency rank, asset popularity, asset category, the customer's number of past assets and days since any Buy; (A3) the intervention, an LLM that receives the same facts as text and returns the past ISINs in order. The four provided baselines are reported for every run.

**LLM.** `Qwen/Qwen3-1.7B` at the revision pinned by the course baselines (`70d244cc...`), greedy decoding with thinking off, zero-shot, local inference on a Kaggle GPU [8]. The prompt lists each past asset with its as-of metadata, Buy dates, Buy count and last Sell date, plus the cutoff and the profile with observed or estimated status. The model must return a JSON list that is a permutation of the given ISINs. An invalid answer falls back to the A0 order and is counted; results are reported with and without fallback users. Customers with one past asset are identical in every arm and are reported as such. The prompt holds the 30 most recent past assets, and A2 ranks the same 30.

**Measure.** nDCG@10 over all users (empty truth scores 0, primary) and over labelled users, with the user count, per subgroup; macro mean over cutoffs with the min-max range and paired win/loss counts against A0. Runtime: wall-clock time to produce all lists, prompt and output tokens, GPU type. Cost: Kaggle GPU hours (no paid API).

**Ablation.** A3 against A2 removes only the LLM and keeps the information. A3 with asset names and metadata replaced by neutral ids separates pretrained knowledge about Greek listed companies from reading the dates. A3 with the event dates removed (counts only) tests whether the LLM uses the order that [5] found it perceives poorly.

**Repeats.** Greedy decoding is close to deterministic; we rerun 100 users and report how many lists change. Variation across cutoffs is reported as above.

**Feasibility.** Session 2: the LLM arm on the 100-user demo slice and on one full cutoff for customers with at least two past assets; prompts are short because the median customer holds two distinct assets (Kosychev's profiling). Session 3: optional retrieved asset descriptions as context. Session 4: LoRA on windows that end before the test cutoff, only if zero-shot fails to use dates. Session 5: the hybrid with Kosychev's fill and the full ablation. Budget: within the free Kaggle GPU quota; the classical arms run in seconds on a laptop CPU.

**Risks.** *Leakage*: inputs are rows dated before the cutoff, metadata and profiles are as-of, labels never enter prompts. *Contamination*: the model may have read news about these companies after the cutoff; the masked-name ablation estimates this effect, though it cannot remove it. *Privacy*: ids are pseudonymised, truncated in reports, and no user-level output is published. *Ethics*: rebuy suggestions can deepen concentration in assets a customer already holds; the work ranks past behaviour and is not advice.

## IV. Experiments

**What was run.** Three standard-library pilots (`pilot_gate.py`, `pilot_headroom.py`, `pilot_prefix_lr.py`) on the nine cutoffs and the starter itself on the demo slice. No LLM was run; no LLM number exists or is reported. The pilots load the archive once and took 135 s, 51 s and 33 s on a laptop CPU, at zero monetary cost.

**Consistency checks (AI-performed).** My recency lists equal the starter's `latest_buy` lists for all 100 demo users. On 2021-06-01 my warm cohort has 10,404 users, 1,178 labelled and `latest_buy` 0.0526 over all users and 0.4648 over labelled users, the values the starter README gives for that slice [7]. Pooled over the nine cutoffs I get 12,163 labelled user-windows, 5,051 with a new-to-user asset and 0.561 for `latest_buy`, the numbers in Kosychev's draft.

**Provided baselines** on the demo slice (100 users, 11 labelled): popularity 0.0286 / 0.2599, CBF 0.0201 / 0.1827, `latest_buy` 0.0277 / 0.2520, `latest_buy_or_sell` 0.0271 / 0.2464 (all users / labelled), equal to the README.

**Table III. Pilot, nDCG@10. Labelled: macro mean over 9 cutoffs. Subgroups: pooled user-windows. Oracle rows read the labels and are ceilings, not methods**

| Prefix order / fill | Labelled (12,163) | Repeat-only (7,112) | >10 past (3,688) |
|---|---:|---:|---:|
| A0 recency (`latest_buy`) | 0.561 | 0.760 | 0.495 |
| A1 purchase count | 0.530 | 0.714 | 0.445 |
| A2 logistic regression | 0.566 | 0.766 | 0.508 |
| Oracle prefix order | 0.750 | 1.000 | 0.818 |
| Co-purchase fill (A0 prefix) | 0.569 | 0.760 | 0.495 |
| Oracle fill (A0 prefix) | 0.639 | 0.760 | 0.495 |

**Results.** Table III gives the pilot. First, the prefix order holds more headroom than the fill: +0.188 against +0.077 on labelled users, and the fill cannot change the 58% repeat-only windows nor customers with more than ten past assets, whose ten slots are all prefix. Second, classical signals use little of it: purchase count loses 0.031 to recency, and A2, trained once on six windows that end before 2020-09-01, gains 0.005 (pooled: 1,776 repeat-only windows better and 1,455 worse than A0); its largest standardized weights are days since the last Buy (-0.71), Buy count (+0.42), the customer's number of assets (-0.32) and asset popularity (+0.31). Third, an item-to-item co-purchase fill beats the popularity fill (0.569 against 0.561), an option I proposed to Patrushev and Vaal. The LLM evaluation, A3 and its ablations, is planned for Session 2.

## V. Discussion

**Team discussion.** No synchronous team discussion took place before this deadline. Kosychev pushed his Session 1 draft to the team repository on 18 November (Encoder track: LLM embeddings of asset text for the fill). I read it, reproduced its non-LLM numbers and answered in writing on branch `starchenko/session1-proposals` of https://github.com/ak1232320/rec-sys-hw1, commit `809361d`. By the time of writing (18 November, afternoon) no teammate had replied. *Options considered:* in his draft, Kosychev weighed Encoder, Ranker, Hybrid and fine-tuning questions and adopted the Encoder fill; I weighed a repeat-or-explore gate (dropped, see below), a co-purchase fill (offered to Patrushev and Vaal) and the prefix reranker (adopted). *Agreement and differences:* none was agreed jointly. Shared inputs are the course starter and Kosychev's nine-cutoff protocol, which I reused with attribution. Open points: whether the official evaluation reports all warm users or labelled subgroups as primary, which components enter the team's benchmark submission, and how large an LLM the runtime tie-break allows. *My contribution:* the independent re-implementation and its checks, the gate pilot, the prefix/fill headroom split, the logistic-regression prefix ranker, the record in Section III-A, and the written proposals (team name in Kosychev's header, a note that his draft describes joint decisions that did not happen, the encoder/ranker split for a clean hybrid ablation, a shared Session-2 protocol). *My position:* I adopt the prefix reranker because it addresses the larger and untouched part of the headroom, its candidate sets are small enough for a 1.7B model, and it is orthogonal to Kosychev's fill, so the team gets baseline, encoder-only, ranker-only and both without new infrastructure.

**A failed idea.** My first question was a gate that predicts whether a customer will buy something new and then places the top-m popular new assets ahead of the prefix. Even an oracle gate that knows the answer scored 0.557 against 0.561 for `latest_buy` (m = 1), below it on 8 of 9 cutoffs; on the 5,051 windows with a new asset it was better for 527 and worse for 1,967. Root cause: the popular new assets rarely match the new purchase, and moving them up pushes down the repeat assets that do match. The position of new assets is worth nothing without a better choice of which new assets, which is Kosychev's question, so I dropped the gate.

**A failure during the work.** The script that prints the record crashed on the asset row:

```
print('ASSET ROWS:', a)
UnicodeEncodeError: 'charmap' codec
  can't encode characters in position 85-87
```

The Windows console used code page cp1251, which has no Greek letters, and the asset's short name is Greek. The fix set standard output to UTF-8; the rerun printed the full asset, customer and price rows quoted in Table II.

**Threats to validity.** About 11% of warm user-windows are labelled, so all-user means are dominated by zeros; both means are always reported. The data come from one institution and 2020-2022. The subgroups are defined by the label window and are descriptive. The oracle prefix is a loose ceiling, and a stronger learned ranker (gradient-boosted trees) could close more of the gap than logistic regression; it will be added to A2 if time allows.

**Meaning of the outcomes.** Whichever way it goes, the result will be reported. If A3 beats A2, the LLM reads purchase timing that the features miss, and the hybrid with Kosychev's fill is the next step. If it ties or loses, the cheaper learned order stays, the LLM budget goes to the fill, and the result extends the order-perception weakness of [5] to investment histories.

## VI. AI Usage Disclosure

**Tools.** Claude Code (Claude Opus 5.5, `claude-opus-5-5`, VS Code/Windows) downloaded the data, wrote and ran the pilots, fetched five arXiv abstracts, pushed proposals at my request and drafted this report.

**What the AI checked.** Claude Code checked that its recency lists equal the starter's `latest_buy` lists for the 100 demo users, reproduced the README values for the 2021-06-01 slice and Kosychev's pooled numbers, and recomputed the rank of the record's ISIN under each order (Section IV).

**What I personally verified.** I reran the three pilots (`pilot_gate.py`, `pilot_headroom.py`, `pilot_prefix_lr.py`) on my local copy of the FAR-Trans CSVs; their metrics matched the saved JSON files (I did not compare runtimes). Over the nine cutoffs I obtained 12,163 labelled windows, 7,112 of them repeat-only (58.47%), and mean nDCG@10 of 0.561337 for `latest_buy`, 0.566381 for the logistic-regression ranker and 0.749507 for the oracle order of previously bought assets. I recomputed the record from the raw CSVs: before the cutoff the customer of transaction 9049523 had 24 Buys of five distinct assets, and the label asset is fifth by date of last purchase, so nDCG@10 = 1/log2(6) = 0.386853; the asset, customer and price rows are dated before the cutoff, and the label purchase is dated 8 June 2021. On the demo slice I recomputed the means from `per_user.csv`, and all 100 `latest_buy` lists matched the starter. I checked the descriptions of the five studies (FAR-Trans, RepeatNet, Next Basket Reality Check, LLMRank, FLARKO) against the relevant sections of their source texts.

**What I trusted without verification.** Reruns of the same code do not prove every list correct. I did not audit the provider's pseudonymisation, every as-of path of the starter, or the completeness of the literature search. The planned LLM arm was not run.

**Session export.** The unmodified Claude Code JSONL of this session, submitted as `starchenko_session1_session.json`.

## References

[1] J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An investment dataset for financial asset recommendation," in *Proc. IJCAI-24 Workshop on Recommender Systems in Finance (Fin-RecSys)*, 2024. [Online]. Available: https://arxiv.org/abs/2407.08692

[2] University of Glasgow, "FAR-Trans dataset," research data repository. [Online]. Available: https://researchdata.gla.ac.uk/1658/

[3] P. Ren, Z. Chen, J. Li, Z. Ren, J. Ma, and M. de Rijke, "RepeatNet: A repeat aware neural recommendation machine for session-based recommendation," in *Proc. AAAI Conf. Artificial Intelligence*, 2019. [Online]. Available: https://arxiv.org/abs/1812.02646

[4] M. Li, S. Jullien, M. Ariannezhad, and M. de Rijke, "A next basket recommendation reality check," *ACM Trans. Information Systems*, vol. 41, no. 4, 2023. [Online]. Available: https://arxiv.org/abs/2109.14233

[5] Y. Hou, J. Zhang, Z. Lin, H. Lu, R. Xie, J. McAuley, and W. X. Zhao, "Large language models are zero-shot rankers for recommender systems," in *Proc. European Conf. Information Retrieval (ECIR)*, 2024. [Online]. Available: https://arxiv.org/abs/2305.08845

[6] F. Spadea and O. Seneviratne, "Aligning language models with investor and market behavior for financial recommendations," arXiv:2510.15993, 2025. [Online]. Available: https://arxiv.org/abs/2510.15993

[7] S. Jin, "FAR-Trans student starter," LLM4Rec course repository, 2026. [Online]. Available: https://github.com/dryjins/RecSys-LLMs/tree/main/project/far-trans

[8] S. Jin, "FAR-Trans: Encoder and Ranker baselines," LLM4Rec course repository, 2026. [Online]. Available: https://github.com/dryjins/RecSys-LLMs/tree/main/project/far-trans-llm
