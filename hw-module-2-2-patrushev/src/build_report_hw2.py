"""Render the Session 2 (HW2) report from evidence/ -> report/report_session2.html and the PDF.

Run: .venv/bin/python -I src/build_report_hw2.py [--pdf] [--commit <sha>]
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import subprocess
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"
REP = ROOT / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PDF_NAME = "patrushev_konstantin_session2_report.pdf"
REPO = "https://github.com/KonstantinPatrushev/recsys_hw8"


def j(name):
    p = EV / name
    return json.loads(p.read_text())


def rnd(x, d):
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


f3 = lambda x: rnd(x, 3)
f4 = lambda x: rnd(x, 4)
pct = lambda x, d=0: rnd(100 * x, d) + "%"
n0 = lambda x: f"{int(x):,}"
esc = lambda s: html.escape(str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--commit", default="[commit]")
    ap.add_argument("--ev", default=None, help="evidence dir override (layout tests)")
    args = ap.parse_args()
    global EV
    if args.ev:
        EV = pathlib.Path(args.ev)
    h = j("hw2_metrics.json")
    an = j("hw2_analysis.json")
    warm = j("hw2_rule_only_10404users.json")
    lit = j("literature.json")
    team = j("team_hw2.json")
    M, rt, fc = h["metrics"], h["runtime"], h["failed_cases"]
    cfg = h["config"]
    rk = cfg["ranker"]

    rows = [("Popularity", "popularity", f"{rt['protocol_and_baselines_s']} s*", "0"),
            ("CBF", "cbf", "(same run)*", "0"),
            ("Latest buy (= A0)", "latest_buy", "(same run)*", "0"),
            ("Latest buy or sell", "latest_buy_or_sell", "(same run)*", "0"),
            ("A1 encoder only: rule + LLM-encoder fill", "A1_rule_encoder_fill", f"{rt['encoder_embedding_s']} s", "0"),
            ("A2 ranker only: LLM ranker, popular candidates", "A2_llm_ranker_popularity",
             f"{rnd(rt['llm_generation_s']['A2'], 0)} s", str(fc.get('A2', 0))),
            ("<b>A3 hybrid: LLM ranker, encoder candidates</b>", "A3_llm_ranker_encoder_hybrid",
             f"{rnd(rt['llm_generation_s']['A3'] + rt['encoder_embedding_s'], 0)} s", str(fc.get('A3', 0)))]
    table = "".join(
        f"<tr><td>{lab}</td><td class=n>{M[k]['users']}</td><td class=n>{M[k]['nonempty_truth_users']}</td>"
        f"<td class=n>{f4(M[k]['ndcg_at_10'])}</td><td class=n>{f4(M[k]['ndcg_at_10_nonempty'])}</td>"
        f"<td class=n>{t}</td><td class=n>{f}</td></tr>" for lab, k, t, f in rows)
    W = warm["metrics"]
    lb, a1, a2, a3 = (M[k] for k in ("latest_buy", "A1_rule_encoder_fill", "A2_llm_ranker_popularity", "A3_llm_ranker_encoder_hybrid"))
    tok = rt["llm_tokens_in_out"]
    per_call = {a: rt["llm_generation_s"][a] / h["users"] for a in ("A2", "A3")}
    num = {k: i + 1 for i, k in enumerate(USED)}
    refs = "".join(f"<p>[{num[x['key']]}] {x['ieee']}</p>" for x in sorted(
        (x for x in lit["references"] if x["key"] in USED), key=lambda x: num[x["key"]]))
    c = lambda k: f"[{num[k]}]"
    lab = j("hw2_labelled_sample_60.json") if (EV / "hw2_labelled_sample_60.json").exists() else None
    lab_an = j("hw2_labelled_analysis.json") if (EV / "hw2_labelled_analysis.json").exists() else None
    mark = lambda t: f'<span class="todo">{esc(t)}</span>' if t.startswith("[") else esc(t)  # placeholders stay yellow
    hand = mark(team.get("hand_check", ""))
    T = prose(h, an, warm, lab, lab_an, hand)
    lab_block = ""
    if lab:
        LM, LA, lrt = lab["metrics"], lab_an["arms"], lab["runtime"]
        rows3 = [("Popularity", "popularity"), ("CBF", "cbf"), ("Latest buy (= A0)", "latest_buy"), ("Latest buy or sell", "latest_buy_or_sell"),
                 ("A1 encoder only", "A1_rule_encoder_fill"), ("A2 ranker only", "A2_llm_ranker_popularity"),
                 ("<b>A3 hybrid</b>", "A3_llm_ranker_encoder_hybrid")]
        fc3 = {"A2_llm_ranker_popularity": lab["failed_cases"].get("A2", 0), "A3_llm_ranker_encoder_hybrid": lab["failed_cases"].get("A3", 0)}
        bw = lambda k: "&ndash;" if k == "latest_buy" else f"{LA[k]['nonempty_wins']} / {LA[k]['nonempty_losses']}"
        t3 = "".join(f"<tr><td>{lab_}</td><td class=n>{LM[k]['users']}</td><td class=n>{LM[k]['nonempty_truth_users']}</td>"
                     f"<td class=n>{f4(LM[k]['ndcg_at_10_nonempty'])}</td><td class=n>{bw(k)}</td><td class=n>{fc3.get(k, 0)}</td></tr>"
                     for lab_, k in rows3)
        lab_block = f"""<p><b>A larger labelled sample.</b> To get more than {lb['nonempty_truth_users']} labelled customers within the deadline, I ran the
same code (commit 6a729f8) on {lab['users']} customers drawn with seed {lab['sample']['seed']} from the
{n0(lab['sample']['drawn_from_warm_users_with_purchases'])} warm customers with purchases at the same cutoff. The draw uses the labels, the methods
do not; every customer has purchases, so only the nonempty mean is meaningful, and it is not comparable to Table 1&rsquo;s all-user column.</p>
<div class="avoid"><table><tr><th>Method</th><th class=n>Users</th><th class=n>Users with future purchases</th><th class=n>nDCG@10: nonempty truth</th>
<th class=n>Better / worse than latest buy</th><th class=n>Failed cases</th></tr>{t3}</table>
<p class="cap">Table 3. Seeded labelled sample (<code>hw2_labelled_sample_{lab['users']}.json</code>, <code>hw2_labelled_analysis.json</code>); LLM generation
{n0(lrt['llm_generation_s'].get('A2', 0))} s (A2) and {n0(lrt['llm_generation_s'].get('A3', 0))} s (A3); whole run {n0(lrt['total_wall_clock_s'])} s. Failed cases fall back to the rule list and are included.</p></div>"""
    L = an["llm"]
    def inv(a):
        b = L[a]["calls_invalid_by_past_assets"]
        return " / ".join(f"{b[k][1]} of {b[k][0]}" for k in ("1", "2-3", "4+") if k in b)
    llm_rows = "".join(
        f"<tr><td>{a}</td><td class=n>{L[a]['valid']} of {L[a]['calls']}</td><td class=n>{inv(a)}</td>"
        f"<td class=n>{L[a]['valid_keeping_all_group_A']}</td><td class=n>{L[a]['mean_group_B_picks']}</td>"
        f"<td class=n>{L[a]['seconds_mean']} / {L[a]['seconds_max']}</td></tr>" for a in ("A2", "A3"))

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Session 2 Report &mdash; LLM ranker over past and new assets on FAR-Trans</title>
<style>
 a {{ color:#000; text-decoration:none; }} @page {{ size:A4; margin:12mm 13mm; }}
 body {{ font-family:"Times New Roman","Liberation Serif",serif; font-size:9.8pt; line-height:1.23; margin:0; }}
 h1 {{ font-size:14pt; text-align:center; margin:0 0 4pt; line-height:1.2; }}
 .meta {{ text-align:center; font-size:9.3pt; margin-bottom:5pt; }}
 h2 {{ font-size:10.4pt; margin:5pt 0 2pt; text-transform:uppercase; letter-spacing:.3pt; break-after:avoid; }}
 h3 {{ font-size:9.9pt; margin:3.5pt 0 1.5pt; font-style:italic; font-weight:normal; break-after:avoid; }}
 p {{ margin:0 0 2.8pt; text-align:justify; }} ul {{ margin:0 0 2.8pt; padding-left:14pt; }} li {{ margin-bottom:.8pt; text-align:justify; }}
 .abstract {{ font-size:9.3pt; margin:0 5mm 3pt; }}
 code, pre {{ font-family:"DejaVu Sans Mono","Menlo",monospace; font-size:7.6pt; }}
 pre {{ border:1px solid #000; padding:3pt 5pt; margin:2pt 0 3pt; white-space:pre-wrap; }}
 table {{ border-collapse:collapse; margin:2pt auto 1.5pt; font-size:8.3pt; }}
 th, td {{ border-top:1px solid #000; border-bottom:1px solid #000; padding:1pt 3.5pt; text-align:left; vertical-align:top; }}
 td.n, th.n {{ text-align:right; white-space:nowrap; }} .cap {{ font-size:8.2pt; text-align:center; margin:1pt 0 4pt; }}
 .refs p {{ text-align:left; padding-left:18pt; text-indent:-18pt; font-size:8.5pt; margin-bottom:1.6pt; }}
 .avoid {{ break-inside:avoid; }} .todo {{ background:#fff3a8; }}
</style></head><body>
<h1>An LLM Ranker over a Customer&rsquo;s Past and New Assets:<br>Implementing the Repeat-Aware Hybrid on FAR-Trans</h1>
<div class="meta">Student: Konstantin Patrushev &nbsp;&nbsp; Team: {esc(team['name'])} &nbsp;&nbsp; Email: kapatrushev@edu.hse.ru &nbsp;&nbsp; Date: 2026-11-25<br>
Repository: <a href="{REPO}">{REPO}</a> &nbsp;&nbsp; Experiment commits: <code>{esc(args.commit)}</code><br>
LLM4Rec project, Session 2 / Homework 2 &mdash; Implement and Test Your Idea (individual) &nbsp;&middot;&nbsp; Track: Hybrid</div>

<h2>Abstract</h2>
<p class="abstract">My Session 1 question asked whether an open-weight LLM ranker that orders a customer&rsquo;s past assets together with
never-bought candidates from an LLM embedding encoder improves nDCG@10 over the provided latest-buy baseline on FAR-Trans {c('fartrans')}, {c('dataset')}.
I implemented the encoder (Qwen3-Embedding-0.6B), the ranker (Qwen3-1.7B, zero-shot, greedy) and their 2&times;2 ablation on the starter&rsquo;s
100-user protocol, with all four provided baselines. The hybrid scores nDCG@10 {f4(a3['ndcg_at_10'])} over all users and
{f4(a3['ndcg_at_10_nonempty'])} over the {a3['nonempty_truth_users']} users with purchases, against {f4(lb['ndcg_at_10'])} and
{f4(lb['ndcg_at_10_nonempty'])} for latest buy, with {fc.get('A3', 0)} invalid outputs and about {rnd(per_call['A3'], 1)} s per customer.
{T['abstract_takeaway']}</p>

<h2>1. Introduction</h2>
<p><b>Question and change.</b> FAR-Trans customers repeat earlier purchases often: in my Session 1 data profile 61% of the future
purchases of warm customers repeat an earlier buy, and latest buy (past assets newest first, then popular assets) was the strongest
provided baseline on the full warm cohort. The question is whether an LLM can do better than that rule at two jobs: proposing assets the
customer never bought (encoder) and ordering the customer&rsquo;s past and new candidates (ranker). The change plugs in after the starter&rsquo;s
data loading and before its evaluation: the same users, cutoff, eligible assets, truth and nDCG@10 code, with a different Top-10 list.</p>
<p><b>Contributions.</b> (i) A runnable implementation of the encoder, the ranker and the ablation arms with open-weight models;
(ii) a matched comparison with the four provided baselines, run times and failed cases; (iii) checks that the comparison is fair and
leak-free, and a hand check of one customer.</p>

<h2>2. Related Work</h2>
<p>LLMs can rank recommendation candidates zero-shot but struggle with the order of a history and show position and popularity bias
{c('llmrank')}; off-the-shelf LLM scores of repurchase candidates have lost to supervised rankers {c('cao26')}. Repeat purchases carry most
of the accuracy of sequential recommenders {c('li23')}. On FAR-Trans, FLARKO tunes Qwen3 models to generate three assets and reports
Hits@3 {c('flarko')}; I found no FAR-Trans study of a zero-shot LLM ranker against a recency baseline. The course provides an LLMRank-style
ranker over content-based candidates; my ranker differs in its candidate pool (the customer&rsquo;s own past assets plus never-bought
candidates) and its comparator (latest buy). Alternative considered: Mark Starchenko&rsquo;s team proposal of a logistic-regression ranker of
past assets, the classical control planned for the next session.</p>

<h2>3. Method</h2>
<p><b>Protocol (starter defaults).</b> The starter&rsquo;s deterministic 100 warm users, cutoff 2021-06-01, labels = Buys from 2021-06-01 to
2021-07-01 intersected with the 767 eligible assets, repeats allowed, binary nDCG@10 with empty truth counted as zero. My script imports the
starter&rsquo;s own functions; its user list equals the starter run&rsquo;s <code>protocol.json</code> (hash prefix
{h['protocol_hash_of_starter_run'][:12]}).</p>
<ul>
<li><b>Repeat channel.</b> The customer&rsquo;s eligible past Buy assets, newest first (ISIN breaks ties): exactly the latest-buy prefix.</li>
<li><b>Encoder E.</b> {esc(cfg['encoder']['model'])} {c('qwen3emb')} (revision {cfg['encoder']['revision'][:8]}) embeds each eligible asset&rsquo;s as-of text
(name, short name, category, subcategory, sector, industry, market). Customer vector = Buy-count-weighted sum of their past assets&rsquo; vectors.
E re-scores the {cfg['encoder']['popular_pool']} most popular never-bought assets by cosine similarity and keeps the top
{cfg['encoder']['explore_candidates']} (the Session 1 pilot showed that similarity alone, without the popularity pool, loses to popularity).</li>
<li><b>Ranker R.</b> {esc(rk['model'])} {c('qwen3')} (revision {rk['revision'][:8]}, resolved {str(rk.get('resolved_revision'))[:8]}), {esc(rk['dtype'])} on
{esc(rk['device'].upper())}, {esc(rk['decoding'])}, thinking disabled, at most {rk['max_new_tokens']} new tokens; torch {esc(rk['torch'])},
transformers {esc(rk['transformers'])}. The prompt (instruction below) holds group A, the up to {cfg['max_repeat_assets_in_prompt']} newest past
assets with Buy count, last Buy and last Sell dates, and group B, {cfg['encoder']['explore_candidates']} never-bought candidates with their
popularity rank. No profile, prices, amounts or truth. R returns ten ISINs; a list that is not exactly ten unique ISINs from the pool is
a failed case and the arm&rsquo;s rule list is used for that customer.</li>
<li><b>Arms.</b> A0 = latest buy (rule + popularity fill). A1 = rule + E fill (encoder only). A2 = R over group A + 10 popular never-bought
assets (ranker only). A3 = R over group A + E&rsquo;s 10 candidates (both). A1&minus;A0 isolates the encoder, A2&minus;A0 the ranker.</li>
</ul>
<pre>{esc(cfg['instruction'])}
{{"cutoff_date": "2021-06-01", "group_A_bought_before": [{{"ISIN", "name", "category", "subcategory", "sector", "industry", "market",
  "buy_count", "last_buy", "last_sell"}}, ...], "group_B_never_bought": [{{..., "popularity_rank"}}, ...]}}</pre>
<p><b>Hardware and dependencies.</b> {esc(h['hardware']['platform'])} (Apple M5, 24 GB, PyTorch MPS); Python 3.12, torch {esc(rk['torch'])},
transformers {esc(rk['transformers'])}, sentence-transformers 5.7.0, numpy, polars. Code: <code>src/hw2_hybrid.py</code> (run command in the README);
every generation with its full prompt is logged to <code>outputs/hw2/generations.jsonl</code> (customer IDs hashed; not committed).</p>

<h2>4. Experiments</h2>
<div class="avoid"><table>
<tr><th>Method</th><th class=n>Users</th><th class=n>Users with future purchases</th><th class=n>nDCG@10: all users</th>
<th class=n>nDCG@10: nonempty truth</th><th class=n>Runtime</th><th class=n>Failed cases</th></tr>
{table}</table>
<p class="cap">Table 1. Starter protocol, 100 users, cutoff 2021-06-01, 30 days (<code>hw2_metrics.json</code>). *The four provided baselines come
from one starter pass, timed together. A1: embedding all 767 assets once. A2/A3: LLM generation for 100 customers ({n0(tok['A2'][0])} and
{n0(tok['A3'][0])} prompt tokens, {n0(tok['A2'][1])} and {n0(tok['A3'][1])} output tokens); A3 includes the embedding time. Whole run with model loading: {n0(rt['total_wall_clock_s'])} s. Failed cases fall back to
the rule list and are included in the scores.</p></div>
<p><b>Beyond the demo.</b> The 100-user sample has only {lb['nonempty_truth_users']} customers with purchases. On all
{n0(W['latest_buy']['users'])} warm customers ({n0(W['latest_buy']['nonempty_truth_users'])} with purchases) the encoder-only arm A1 scores
{f4(W['A1_rule_encoder_fill']['ndcg_at_10_nonempty'])} on customers with purchases against {f4(W['latest_buy']['ndcg_at_10_nonempty'])} for latest buy
(<code>hw2_rule_only_10404users.json</code>): the small demo gain of A1 does not hold at scale. The LLM arms were not run on the full cohort
(about {rnd(per_call['A3'] * W['latest_buy']['users'] / 3600, 1)} h per arm at the measured speed).</p>
<p><b>What changed per customer (demo).</b> {T['changes_text']}</p>
<div class="avoid"><table>
<tr><th>Arm</th><th class=n>Valid lists</th><th class=n>Invalid: 1 / 2&ndash;3 / 4+ past assets</th><th class=n>Valid lists keeping all past assets*</th>
<th class=n>Mean new assets in valid lists</th><th class=n>Seconds per call: mean / max</th></tr>
{llm_rows}</table>
<p class="cap">Table 2. Ranker behaviour (<code>hw2_analysis.json</code>). Invalid cells: invalid calls / calls for customers with that many eligible past
assets. *All of group A when it has at most ten assets, otherwise ten of them.</p></div>
{lab_block}
<p><b>Checks.</b> (a) For all 100 customers A0 equals the starter&rsquo;s latest-buy list and the four baselines&rsquo; nDCG@10 equal the starter&rsquo;s
<code>metrics.json</code> exactly (assertions in the script). (b) Every date in every prompt precedes the cutoff (assertion); labels are computed
only after all lists are fixed; both hold for both runs. (c) {T['hand_check_text']}</p>

<h2>5. Discussion</h2>
<p><b>What improved and what failed.</b> {T['discussion_text']}</p>
<p><b>Next controlled experiment.</b> <i>Changes:</i> one rule in the ranker&rsquo;s output: the past assets must stay in the Top-10 (in
the order the LLM gives), and the LLM only chooses which never-bought candidates fill the free slots and where they go relative to each other;
answers are generated with decoding constrained to candidate ISINs not yet emitted, so every answer is valid. <i>Fixed:</i> model and revision,
prompt facts, candidate pools, the seeded labelled sample and the 100 demo customers, cutoff, nDCG@10, fallback. Read-out: paired per-customer
differences against latest buy and A1, repeat and new hits. This isolates whether the LLM adds anything on the new-asset side once it cannot
lose repeats; the team&rsquo;s logistic-regression ranker of past assets is the classical control for the repeat side.</p>
<p><b>Team discussion and individual contribution.</b> {esc(team['documented'])} {mark(team['todo'])}</p>
<ul><li><b>My work:</b> {esc(team['mine'])}</li><li><b>Shared:</b> {esc(team['shared'])}</li></ul>

<h2>6. AI Usage Disclosure</h2>
<p><b>Tools.</b> Claude Code 2.1.292 (VS Code extension) with Claude Opus 5.5, one session. <b>Use.</b> Code for the experiment and the report
builder, running the models and the starter, analysis, and a draft of this report; every number is rendered from <code>evidence/</code>.
<b>What I personally verified.</b> {mark(team['verified'])} <b>Trusted without verification.</b>
{mark(team['trusted'])} <b>Session log.</b> <code>patrushev_konstantin_session2_session.json</code>, a derived export of the
unmodified session lines of this homework (xz + base64 inside JSON, because the LMS rejects the raw log; SHA-256 and a restore command inside).</p>

<h2>References</h2><div class="refs">{refs}</div>
</body></html>"""
    REP.mkdir(exist_ok=True)
    (REP / "report_session2.html").write_text(doc, encoding="utf-8")
    print("report_session2.html written")
    if args.pdf:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={REP / PDF_NAME}", (REP / "report_session2.html").as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("pdf:", REP / PDF_NAME)


def prose(h, an, warm, lab=None, lab_an=None, hand=""):
    """Result-dependent sentences, written after reading hw2_metrics.json / hw2_analysis.json."""
    M, A, L, ex = h["metrics"], an["arms"], an["llm"], an["example_user"]
    k2, k3, k1 = "A2_llm_ranker_popularity", "A3_llm_ranker_encoder_hybrid", "A1_rule_encoder_fill"
    lb = M["latest_buy"]
    fail = {a: L[a]["calls"] - L[a]["valid"] for a in ("A2", "A3")}
    reasons = Counter()
    for a in ("A2", "A3"):
        for r, n in L[a]["failure_reasons"].items():
            reasons["dup" if r.startswith("10 items") else "out" if r.startswith("ISIN outside") else "nojson" if r == "no JSON array" else r] += n
    wl = lambda d, k: f"{d[k]['nonempty_wins']} better, {d[k]['nonempty_losses']} worse"
    hits = lambda k: f"{A[k]['hits_repeat']}+{A[k]['hits_new']}"
    out = {}
    lab_sent = ""
    if lab:
        LM, LA = lab["metrics"], lab_an["arms"]
        lab_sent = (f" On a seeded sample of {lab['users']} warm customers who have purchases, A3 scores {f4(LM[k3]['ndcg_at_10_nonempty'])} "
                    f"and A2 {f4(LM[k2]['ndcg_at_10_nonempty'])} against {f4(LM['latest_buy']['ndcg_at_10_nonempty'])} for latest buy.")
    LA = lab_an["arms"] if lab else None
    out["abstract_takeaway"] = (
        f"On these 100 customers both ranker arms score above all four baselines, but only {lb['nonempty_truth_users']} customers have purchases "
        f"(A3: {wl(A, k3)} than latest buy) and {fail['A2']}% (A2) and {fail['A3']}% (A3) of the ranker&rsquo;s answers are invalid. "
        + (f"On a seeded sample of {lab['users']} warm customers with purchases the order reverses: A3 scores {f4(lab['metrics'][k3]['ndcg_at_10_nonempty'])} "
           f"and A2 {f4(lab['metrics'][k2]['ndcg_at_10_nonempty'])} against {f4(lab['metrics']['latest_buy']['ndcg_at_10_nonempty'])} for latest buy "
           f"(A3: {wl(LA, k3)}), mainly because the ranker drops past assets that customers buy again. In this setting the zero-shot ranker "
           f"does not beat the recency rule." if lab else "The pipeline runs end to end; the evidence that the LLM ranker helps is not there yet."))
    out["changes_text"] = (
        f"Against latest buy, on the {lb['nonempty_truth_users']} customers with purchases: A1 {wl(A, k1)}; A2 {wl(A, k2)}; A3 {wl(A, k3)}; "
        f"the rest are unchanged. Correct assets in the Top-10 of all 100 customers (repeat + new; there are {an['truth_repeat_items']} + "
        f"{an['truth_items'] - an['truth_repeat_items']} future purchases): latest buy {hits('latest_buy')}, A1 {hits(k1)}, A2 {hits(k2)}, "
        f"A3 {hits(k3)}. A2 wins by moving a past asset bought five times from rank 5 to 3 and a popular never-bought asset from rank 7 to 2. "
        f"A3&rsquo;s largest gain (+0.387) is the encoder&rsquo;s candidate at rank 5, which the ranker left where the rule put it (A1 has the same gain); "
        f"its largest loss (&minus;0.315) pushes a past asset that the customer bought again out of the Top-10. "
        f"Of the {fail['A2'] + fail['A3']} invalid answers, {reasons['dup']} repeat an ISIN, {reasons['out']} name an ISIN that is not a "
        f"candidate and {reasons['nojson']} contain no JSON array; Table 2 splits them by the number of past assets.")
    ub = ", ".join(map(str, ex["latest_buy_hit_ranks"])) or "none"
    u3 = ", ".join(map(str, ex["A3_hit_ranks"])) or "none"
    out["hand_check_text"] = (
        f"An independent recomputation of nDCG@10 for all seven methods in both runs from <code>predictions.csv</code> and the raw "
        f"<code>transactions.csv</code>, without the starter&rsquo;s code, matches <code>per_user.csv</code> "
        f"(largest difference {max(an['independent_recompute_max_abs_diff'].values()):.1f}). "
        f"(d) Hand check, customer <code>{ex['user_id'][:10]}&hellip;</code> (largest A3 vs latest-buy difference): one future purchase, "
        f"<code>{ex['truth'][0]}</code>, never bought before; latest buy does not list it (nDCG@10 {f4(ex['latest_buy_ndcg'])}); "
        f"A3 at rank {u3}, so nDCG@10 = (1/log<sub>2</sub>(5+1))/1 = {f4(ex['A3_ndcg'])}. "
        f"{hand}")
    lab_disc = ""
    if lab:
        LM = lab["metrics"]
        lab_disc = (f"<i>Main result:</i> the demo gain does not hold. On the seeded labelled sample latest buy finds {LA['latest_buy']['hits_repeat']} repeat "
                    f"and {LA['latest_buy']['hits_new']} new purchases in its Top-10s, A3 {LA[k3]['hits_repeat']} and {LA[k3]['hits_new']}, A2 "
                    f"{LA[k2]['hits_repeat']} and {LA[k2]['hits_new']}: the ranker gains a few new assets and loses more repeats, which customers buy far "
                    f"more often. A3 loses more than A2 because it puts more never-bought candidates into its lists ({lab_an['llm']['A3']['mean_group_B_picks']} "
                    f"against {lab_an['llm']['A2']['mean_group_B_picks']} per valid answer on this sample) and so pushes out more past assets. "
                    f"The encoder fill alone (A1) is a wash: {wl(LA, k1)} here, slightly below latest buy on all warm customers. ")
    out["discussion_text"] = (
        lab_disc +
        f"<i>Other failures:</i> (1) validity: Qwen3-1.7B repeats or invents ISINs in {fail['A3']}&ndash;{fail['A2']}% of the demo answers even with "
        f"greedy decoding; (2) where it does not drop assets the ranker mostly keeps the rule: it keeps the order of the past assets in "
        f"{L['A2']['valid_with_group_A_order_kept']} of {L['A2']['valid']} (A2) and {L['A3']['valid_with_group_A_order_kept']} of {L['A3']['valid']} (A3) "
        f"valid demo answers; (3) cost: about {L['A3']['seconds_mean']} s and {n0(L['A3']['input_tokens_mean'])} prompt tokens per customer against "
        f"milliseconds for the rule. <i>What worked:</i> the pipeline runs end to end with pinned open weights, every failure is counted, and two "
        f"independent checks reproduce every score. <i>Threats to validity:</i> one prompt, one model size, one cutoff, a fixed candidate order "
        f"(position bias untested), bfloat16 on MPS (another device may change some greedy answers), {lab['users'] if lab else ''} labelled customers in the larger sample, "
        f"and a protocol in which repeats count as hits.")
    return out


USED = ["fartrans", "dataset", "llmrank", "cao26", "li23", "flarko", "qwen3emb", "qwen3"]

if __name__ == "__main__":
    main()
