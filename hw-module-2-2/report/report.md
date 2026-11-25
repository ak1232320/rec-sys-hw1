# Filling the Slots Recency Leaves Empty with an LLM Embedding Space: An Implemented, Matched Test on FAR-Trans

<div class="meta">
Student: Aleksei Kosychev &nbsp;&nbsp; Team: name &nbsp;&nbsp; Email: amkosychev@edu.hse.ru &nbsp;&nbsp; Date: 2026-11-25<br>
Repository: https://github.com/ak1232320/rec-sys-hw1 (folder <code>hw-module-2-2</code>) &nbsp;&nbsp; Experiment commit: <span class="todo">TODO after commit</span><br>
Assignment: Session 2 / Homework 2 — Implement and Test Your Idea (LLM4Rec, Module 2)
</div>

<div class="draft">PRELIMINARY DRAFT for team review. All measured numbers are generated from the run outputs; the yellow blocks wait for the team discussion and for my own verification.</div>

## Abstract

In Session 1 I asked whether frozen LLM text embeddings of as-of asset metadata, used only to
fill the Top-10 slots that the recency prefix of the provided `latest_buy` baseline leaves
empty, improve nDCG@10 over a popularity, categorical or TF-IDF fill. I implemented this
Encoder-track change with the open-weight `Qwen/Qwen3-Embedding-0.6B` (pinned revision, run on a
Kaggle T4) and compared it with the four provided baselines and matched controls. On the HW2
protocol (100 warm users, cutoff 2021-06-01, 30 days) the LLM fill scores highest:
nDCG@10 {{demo:methods.llm_fill.ndcg_at_10}} over all users and
{{demo:methods.llm_fill.ndcg_at_10_nonempty}} over the {{demo:methods.llm_fill.nonempty_truth_users|d}}
users with purchases, but the whole margin comes from one customer. On nine quarterly
development cutoffs ({{dev:labelled_user_windows|d}} labelled user-windows) it is the **worst** fill
for users who buy an asset new to them: {{dev:macro_ndcg_with_new_asset.llm_fill}} vs
{{dev:macro_ndcg_with_new_asset.pop_fill}} for popularity (paired difference
{{dev:paired_with_new_pooled.llm_fill-pop_fill.mean_diff|+.4f}}), lower on all
{{derived:cutoffs|d}} cutoffs. It fills slots with rarely bought assets. The result is negative,
and it is reported as such.

*Index Terms* — financial asset recommendation, LLM text embeddings, content-based filtering, nDCG, FAR-Trans

## 1. Introduction

**Problem and question.** Each evaluated FAR-Trans customer gets a Top-10 asset list, scored by
nDCG@10 against the purchases in the next 30 days under a past-only protocol [1], [3]. The strongest
classical baseline, `latest_buy`, is two parts: a *prefix* (the customer's own past buys, most
recent first) and a *fill* (the most popular other assets). Customers hold few distinct assets,
so most of every list is fill. Fill is also the only part that can reach an asset new to the
customer. Session 1 fixed the question (Encoder track):

> We test whether **frozen LLM text embeddings of the as-of asset description** (intervention)
> improve **nDCG@10 against purchases in the next 30 days** (outcome) over **the same pipeline whose
> fill uses popularity, the starter's categorical tokens or TF-IDF** (control) for **warm
> FAR-Trans customers, overall and on those whose label window holds an asset new to them**
> (population).

**What changed since Session 1.** The LLM arm was planned then; it is now implemented and run.
Session 1 named `multilingual-e5-base` as the candidate. That model is an XLM-RoBERTa encoder,
not an LLM, so the primary model is now an LLM-based embedder, Qwen3-Embedding-0.6B [6]. The
e5 model stays as a control and answers one question: is the effect due to the LLM, or to any
dense encoder?

**Place in the pipeline.** history → `latest_buy` prefix (unchanged) → fill candidates = eligible
assets not in the prefix → score = cosine between the asset vector and the buy-count-weighted
history profile → Top-10 → the starter's nDCG@10. Only the vector space differs between arms.

## 2. Related Work

| Study | Method | Data and evaluation | Connection to this work |
|---|---|---|---|
| Sanz-Cruzado et al., 2024 [1] | FAR-Trans release; 11 classical FAR algorithms | retail investment transactions; temporal split | supplies the data and the classical references; no LLM method |
| Lyu et al., 2024, LLM-Rec [4]; course adaptation [3] | LLM enriches item text by prompting; the course version paraphrases asset attributes with Qwen3-1.7B and ranks by TF-IDF over original + paraphrase | the paper reports gains from LLM-augmented text; the course reports its encoder on the same 100 users | **nearest prior work**: it changes the *text* and keeps a sparse space over the whole list; I keep the text and change the *space*, only in the fill slots |
| Liu et al., 2024, ONCE [5] | open LLM layers as a trained content encoder | news and books; up to 19.32% relative gain | trained end-to-end on rich text; leaves open frozen embeddings of terse metadata |
| Zhang et al., 2025, Qwen3 Embedding [6] | embedding models built on Qwen3 LLMs (0.6B/4B/8B), multi-stage training | MTEB, multilingual and cross-lingual retrieval, state of the art | the model used here; not evaluated on recommendation or finance |
| Wang et al., 2024, multilingual E5 [7] | contrastive pre-training on 1B multilingual pairs plus fine-tuning | MTEB | the non-LLM dense control |

We did not find a study that measures frozen LLM embeddings as a fill rule behind a recency
prefix on FAR-Trans, against nDCG@10 on next-month purchases.

## 3. Method

**Arms.** All arms keep the `latest_buy` prefix (its first min(10, |history|) items) and differ
only in the fill. `pop_fill` is popularity and reproduces `latest_buy` exactly (0 of 100 demo
users and 0 of {{dev:warm_user_windows|d}} development user-windows differ). `tokens_fill` uses
cosine in the starter's categorical token space. `tfidf_fill` uses cosine over unigram TF-IDF of
the asset text. `e5_fill` and **`llm_fill` (mine)** use cosine over the mE5-base and
Qwen3-Embedding vectors of the same text. `*_top20` re-orders the 20 most popular unseen assets
by the space's cosine. `oracle_fill` places the user's new-to-user label assets (a ceiling, not
a method). Ties break by popularity rank, then ISIN. Scores are rounded to 10⁻¹² before
tie-breaking: without this, float sums over Python sets followed the per-process hash seed and
flipped ties between runs (found while reproducing the Session 1 pilot).

**Model input and inference.** The text is `name | category | subcategory | sector | industry |
market` from the latest asset row strictly before the cutoff, with empty fields omitted, e.g.
`DHLOS SYNTHESIS BEST FUND OF FUNDS ΜΕΤΟΧΙΚΟ | MTF | Money Market | AEDAK`. The texts carry no
user, date or transaction field. The 736 unique texts of the nine cutoffs are embedded once.
Embedding is per text, so a text of a later cutoff cannot change an earlier vector, and each
cutoff looks up only its own as-of texts. `{{emb:models.0.repo}}` at revision
`{{emb:models.0.revision}}` (Apache-2.0, {{emb:models.0.parameters|d}} parameters): float32,
last-token pooling, L2-normalised, {{emb:models.0.dimension|d}} dimensions. Asset texts are
encoded as documents without an instruction (the repository's document prompt is empty).
Batch size is 32 and the longest input is {{emb:models.0.input_tokens_max|d}} tokens (none
truncated). `{{emb:models.1.repo}}` at `{{emb:models.1.revision}}` uses the "query: " prefix that
its card advises for symmetric similarity, with mean pooling. Inference ran on Kaggle
({{emb:environment.gpu}}, CUDA {{emb:environment.cuda}}, Python {{emb:environment.python}}, torch
{{emb:environment.torch}}, transformers {{emb:environment.transformers}}, sentence-transformers
{{emb:environment.sentence_transformers}}). A second encoding pass was bit-identical (max |Δ| =
{{emb:models.0.repeat_max_abs_diff|.1f}}). Scoring and evaluation run locally on CPU
(i5-13420H, Python 3.14, numpy 2.4). Code: `kaggle/embed_assets.py`, `src/fill.py`.

**Fair comparison and failure handling.** Users, cutoff, eligibility, labels and nDCG@10 come
from the unchanged starter, imported as a module. My run's `protocol_hash` equals that of a
separate starter CLI run, and all 4,000 baseline prediction rows are byte-identical. The fill
candidates are all eligible assets not in the prefix; the starter's 50-candidate CBF list is not
used. Nothing is generated, so there is no parsing failure. A missing, misaligned (sha256 per
text) or non-finite vector aborts the run. A user with an empty profile would get the
popularity fill and count as a failed case; none occurred. Leakage test: deleting every CSV row
dated on or after the cutoff leaves all lists unchanged. The development protocol (nine quarterly
cutoffs 2020-09-01 … 2022-09-01, all warm users, never filtered by future purchases) was fixed in
Session 1, before the LLM arm existed.

## 4. Experiments

### 4.1 HW2 protocol: 100 users, cutoff 2021-06-01, 30-day window

{{table:demo}}

Metrics are rounded to four decimals; full precision and the protocol hash are in
`outputs/demo/metrics.json`. The baseline runtime is one wall-clock run of the starter for all
four baselines together. Fill runtimes are local CPU scoring time per arm.

**Who moves.** Of the 11 users with purchases, 5 bought only assets they held before. The fill
cannot change their score: all fill arms tie, as the tests check. Three more bought a new
asset but have ten or more past assets, so no slot is free. Of the remaining three, one
customer (`6B15…`, the data example of Session 1) is rescued. He held a "DHLOS SYNTHESIS BEST
YELLOW" fund and buys the sibling "DHLOS SYNTHESIS BEST FUND OF FUNDS". Both dense spaces put
it at rank 3 (nDCG 0.5). TF-IDF reaches only rank 8, because his oil-refinery bond pulls the fill
towards Petrobras and Pemex bonds. One customer is lost: his purchase sat in the popularity
fill at rank 7. One is missed by every arm. The +0.0017 all-user margin of `llm_fill` over
`latest_buy` is therefore the net of one rescue and one loss. On the 100-user samples of the other development cutoffs
the LLM fill is below the popularity fill in {{derived:demo100_llm_below_pop|d}} of
{{derived:cutoffs|d}} and above it in {{derived:demo100_llm_above_pop|d}}.

### 4.2 Development cutoffs: 9 quarters, all warm users

{{dev:labelled_user_windows|d}} labelled user-windows out of {{dev:warm_user_windows|d}};
{{dev:with_new_user_windows|d}} of them ({{derived:with_new_share_of_labelled|pct}}) hold a
new-to-user purchase ("with-new"). Values are macro means over the cutoffs.

{{table:dev}}

{{table:paired}}

The confidence intervals come from a percentile bootstrap over pooled with-new user-windows.
A customer seen at several cutoffs is not clustered, so the intervals are optimistic.

**Findings.** (i) On the with-new users the LLM fill is the lowest of all fills: below
popularity on {{derived:llm_below_pop|d}}/{{derived:cutoffs|d}} cutoffs, below TF-IDF on
{{derived:llm_below_tfidf|d}}/{{derived:cutoffs|d}} and below mE5 on
{{derived:llm_below_e5|d}}/{{derived:cutoffs|d}}. It places
{{dev:new_pairs_in_top10.llm_fill|d}} new-to-user purchases in the Top-10, against
{{dev:new_pairs_in_top10.pop_fill|d}} for popularity. (ii) The non-LLM mE5 encoder is the best
content space, but it still loses to popularity (above it on {{derived:e5_above_pop|d}} cutoffs).
(iii) Restricting the LLM order to the 20 most popular unseen assets recovers most of the loss
(`llm_top20`), and TF-IDF inside the same prior roughly ties popularity. The prior, not the
semantic order, carries the score. (iv) An oracle fill reaches
{{dev:macro_ndcg_with_new_asset.oracle_fill}} on with-new users, so the headroom is real.

**Failure analysis (cutoff 2021-06-01; `outputs/dev/diagnose_2021-06-01.json`).** New-to-user
purchases are {{diag:new_label_category_share.Stock|pct}} stocks with a mean popularity rank of
{{diag:new_label_mean_popularity_rank|.0f}}. The mean popularity rank of the filled assets is
{{diag:arms.pop_fill.mean_popularity_rank_of_fill|.0f}} for popularity,
{{diag:arms.tokens_fill.mean_popularity_rank_of_fill|.0f}} for tokens,
{{diag:arms.e5_fill.mean_popularity_rank_of_fill|.0f}} for mE5,
{{diag:arms.tfidf_fill.mean_popularity_rank_of_fill|.0f}} for TF-IDF and
{{diag:arms.llm_fill.mean_popularity_rank_of_fill|.0f}} for the LLM. The Qwen3 space is far less
compressed: its mean pairwise cosine is {{diag:arms.llm_fill.mean_pairwise_cosine|.2f}}, against
{{diag:arms.e5_fill.mean_pairwise_cosine|.2f}} for mE5. Its nearest neighbours are
name-and-issuer siblings, and most of them are long-tail assets that almost nobody buys.
Semantic closeness is not purchase propensity in this data.

**Checks run by code (AI-performed, `tests/test_fill.py` and the run outputs).** The starter's 58
tests pass on the unchanged copy. My 19 tests pass: fill units, embedding alignment and finiteness,
`pop_fill` = `latest_buy`, the repeat-only tie, prefix preservation, the protocol and baseline
match with a fresh starter run, the future-blind test, and hash-seed independence. The non-LLM
arms match the Session 1 pilot up to tie-breaking (max |Δ| = {{dev:regression_check.max_abs_diff_vs_session1_pilot}}).

## 5. Discussion

**What improved, what failed.** The implementation works end to end: a pinned open-weight LLM,
reproducible vectors and a fair, cross-checked comparison. The question has a clear answer on the
development data: LLM embeddings of terse asset metadata are the worst fill, worse than sparse
features and worse than a smaller non-LLM encoder. The positive 100-user number is a single-user
event; I report it and do not claim it. For the product-owner decision framed in Session 1, this
argues against LLM-embedding infrastructure for the fill as it stands.

**Next controlled experiment.** *Changes:* one term, the fill score becomes
cosine + λ·log(1 + buyers). λ is chosen on the 2020-09 … 2021-03 cutoffs and tested on the six
later ones. *Fixed:* prefix, users, cutoffs, texts, model, revision and vectors. A second,
model-side variant (Qwen3 with the same retrieval instruction on both sides) waits until the
first one is settled.

**Threats to validity.** One model and one input template. The asymmetric Qwen3 model is used
symmetrically, as documents. The bootstrap ignores repeated customers. 93% of new-to-user labels
are stocks. The data cover one institution, 2020–2022. The HW2 sample has 11 labelled users, so
its ordering of methods is not evidence. No suitability, return or profit claim is made.

**Team discussion and individual contribution.** *Documented so far:* in Session 1 Mark Starchenko
sent written proposals through the repository (PR #1, merged 2026-11-18). He proposed to split
`latest_buy` into the prefix (his Ranker question: reorder it with an LLM vs logistic regression)
and the fill (my Encoder question), which gives a clean baseline / encoder-only / ranker-only /
both ablation. He also proposed the nine-cutoff development protocol and the exclusion of
`lastQuestionnaireDate`. He noted that my Session 1 report described that discussion as joint,
when it was written and asynchronous; I report it here as it happened. For HW2 the team agreed on
a process: each member builds an individual solution first, then the solutions are merged with
an AI assistant keeping the best practices, and the merge is reported as the team discussion.
*My work (individual):* everything in `hw-module-2-2`: the design of the arms, the Kaggle
inference, the controls, the tests, the development evaluation and the failure analysis.
*Shared:* the course starter and the development protocol.

<div class="todo">TODO (after the merge with Mark Starchenko, Konstantin Patrushev and Nikita Vaal):
what the team decided from the individual solutions, which parts of mine are kept or dropped,
and which questions stay open (e.g. which fill/prefix combination is submitted, whether the
with-new subgroup becomes primary).</div>

## 6. AI Usage Disclosure

**Tool.** Claude Code (model Claude Opus 5.5, Anthropic) in VS Code; the native session export
is submitted as `kosychev_session2_session.json`. **What the AI did:** it read the course
repository and starter, wrote all code in `src/`, `kaggle/` and `tests/`, pushed and ran the
Kaggle notebook with my approval, ran all experiments and the checks listed in §4, found and
fixed the hash-seed tie-breaking issue, and drafted this report. **My decisions:** the model
runs on Kaggle, not on my laptop (I stopped a local download); the model is run through the
Kaggle API; I asked for this preliminary version for the team.

<div class="todo">TODO — What I personally verified: to be written only after I do the checks in
<code>manual/CHECKLIST.md</code> (one user's IDs and nDCG@10 by hand against the raw CSV; the model
input file contains no future or customer data; the baselines equal the starter's README).</div>

## References

[1] J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An Investment Dataset for Financial Asset Recommendation," in *Proc. IJCAI-24 Workshop on Recommender Systems in Finance (Fin-RecSys)*, Jeju, 2024. [Online]. Available: https://arxiv.org/abs/2407.08692

[2] University of Glasgow, "FAR-Trans dataset," research data repository. [Online]. Available: https://researchdata.gla.ac.uk/1658/

[3] S. Jin, "RecSys-LLMs: FAR-Trans starter and LLM baselines (project/far-trans, project/far-trans-llm)," GitHub, commit 799e54c, 2026. [Online]. Available: https://github.com/dryjins/RecSys-LLMs

[4] H. Lyu, S. Jiang, H. Zeng, Y. Xia, Q. Wang, S. Zhang, R. Chen, C. Leung, J. Tang, and J. Luo, "LLM-Rec: Personalized Recommendation via Prompting Large Language Models," in *Findings of NAACL*, 2024. [Online]. Available: https://arxiv.org/abs/2307.15780

[5] Q. Liu, N. Chen, T. Sakai, and X.-M. Wu, "ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models," in *Proc. 17th ACM Int. Conf. Web Search and Data Mining (WSDM)*, 2024, pp. 452–461. [Online]. Available: https://arxiv.org/abs/2305.06566

[6] Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, and J. Zhou, "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models," arXiv:2506.05176, 2025. [Online]. Available: https://arxiv.org/abs/2506.05176

[7] L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei, "Multilingual E5 Text Embeddings: A Technical Report," arXiv:2402.05672, 2024. [Online]. Available: https://arxiv.org/abs/2402.05672
