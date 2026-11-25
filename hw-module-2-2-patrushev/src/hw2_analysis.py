"""Analysis of the HW2 run -> evidence/hw2_analysis.json

Reads outputs/hw2/{predictions.csv, per_user.csv, generations.jsonl} and evidence/hw2_metrics.json and computes:
  * an independent nDCG@10 recompute from predictions.csv and the raw transactions (no starter code), compared
    with per_user.csv;
  * per arm vs latest buy: lists changed, wins/ties/losses on customers with purchases, hits split into
    repeat / new assets;
  * LLM behaviour: copies of the input order, group-B picks, failure reasons, seconds and tokens per call;
  * one customer for the hand check (A0 vs A3 lists, truth, hit positions, the raw LLM answer).

Run: .venv/bin/python -I src/hw2_analysis.py [outputs/<run> <evidence file name>]
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data" / "raw" / "FAR-Trans"
OUT = ROOT / "outputs" / "hw2"
CUT = "2021-06-01"
END = (date.fromisoformat(CUT) + timedelta(days=30)).isoformat()
H = lambda u: hashlib.sha256(u.encode()).hexdigest()[:12]


def ndcg(lst, rel):
    if not rel:
        return 0.0
    gain = sum(1 / math.log2(k + 2) for k, i in enumerate(lst[:10]) if i in rel)
    return gain / sum(1 / math.log2(k + 2) for k in range(min(10, len(rel))))


def main():
    import sys
    OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "outputs/hw2")
    dest = (ROOT / "evidence" / sys.argv[2] if len(sys.argv) > 2 else
            ROOT / "evidence" / "hw2_analysis.json" if len(sys.argv) == 1 else OUT / "analysis_test.json")
    pred = defaultdict(dict)
    with (OUT / "predictions.csv").open() as f:
        for r in csv.DictReader(f):
            pred[r["method"]].setdefault(r["user_id"], []).append((int(r["rank"]), r["ISIN"]))
    pred = {m: {u: [i for _, i in sorted(v)] for u, v in d.items()} for m, d in pred.items()}
    users = list(pred["latest_buy"])
    per_user = {(r["method"], r["user_id"]): float(r["ndcg_at_10"]) for r in csv.DictReader((OUT / "per_user.csv").open())}

    # truth from the raw transactions file, independently of the starter
    eligible = set()
    with (D / "asset_information.csv").open() as f:
        assets_asof = {r["ISIN"] for r in csv.DictReader(f) if r["timestamp"][:10] < CUT}
    with (D / "close_prices.csv").open() as f:
        priced = {r["ISIN"] for r in csv.DictReader(f) if r["timestamp"][:10] < CUT and float(r["closePrice"]) > 0}
    eligible = assets_asof & priced
    truth, past = defaultdict(set), defaultdict(set)
    seen_rows = set()
    with (D / "transactions.csv").open() as f:
        rd = csv.reader(f)
        head = next(rd)
        c = {k: j for j, k in enumerate(head)}
        for row in rd:
            t = tuple(row)
            if t in seen_rows:
                continue
            seen_rows.add(t)
            if int(row[c["transactionID"]]) < 0 or row[c["transactionType"]] != "Buy" or row[c["ISIN"]] not in eligible:
                continue
            day, u, i = row[c["timestamp"]][:10], row[c["customerID"]], row[c["ISIN"]]
            if day < CUT:
                past[u].add(i)
            elif day < END:
                truth[u].add(i)
    us = set(users)
    truth = {u: truth.get(u, set()) for u in us}
    nonempty = [u for u in users if truth[u]]

    methods = list(pred)
    recompute = {m: max(abs(ndcg(pred[m][u], truth[u]) - per_user[(m, u)]) for u in users) for m in methods}
    out = {"independent_recompute_max_abs_diff": recompute,
           "independent_recompute_ok": all(v < 1e-12 for v in recompute.values()),
           "users": len(users), "nonempty_users": len(nonempty),
           "truth_items": sum(len(truth[u]) for u in us),
           "truth_repeat_items": sum(len(truth[u] & past[u]) for u in us)}

    arms = {}
    for m in methods:
        hits_rep = sum(len(set(pred[m][u]) & truth[u] & past[u]) for u in us)
        hits_new = sum(len(set(pred[m][u]) & (truth[u] - past[u])) for u in us)
        rec = {"hits_repeat": hits_rep, "hits_new": hits_new,
               "mean_repeat_slots": round(statistics.mean(len(set(pred[m][u]) & past[u]) for u in users), 3)}
        if m != "latest_buy":
            d = [ndcg(pred[m][u], truth[u]) - ndcg(pred["latest_buy"][u], truth[u]) for u in nonempty]
            rec.update({"lists_changed_vs_latest_buy": sum(pred[m][u] != pred["latest_buy"][u] for u in users),
                        "nonempty_wins": sum(x > 1e-12 for x in d), "nonempty_losses": sum(x < -1e-12 for x in d),
                        "nonempty_ties": sum(abs(x) <= 1e-12 for x in d)})
        arms[m] = rec
    out["arms"] = arms

    # LLM behaviour
    gens = [json.loads(line) for line in (OUT / "generations.jsonl").open()]
    llm = {}
    for arm in ("A2", "A3"):
        g = [x for x in gens if x["arm"] == arm]
        if not g:
            continue
        copies, b_picks, a_order_kept, a_all = 0, [], 0, 0
        by_size = defaultdict(lambda: [0, 0])  # group-A size bucket -> [calls, invalid]
        for x in g:
            payload = json.loads(x["prompt"].split("\n", 1)[1])
            ga = [r["ISIN"] for r in payload["group_A_bought_before"]]
            gb = [r["ISIN"] for r in payload["group_B_never_bought"]]
            bucket = "1" if len(ga) == 1 else "2-3" if len(ga) <= 3 else "4+"
            by_size[bucket][0] += 1
            by_size[bucket][1] += not x["valid"]
            if not x["valid"]:
                continue
            arr = json.loads(re.search(r"\[[^\[\]]*\]", x["response"]["text"], re.S).group(0))  # as in parse_ranking
            copies += arr == (ga + gb)[:10]
            b_picks.append(sum(i in set(gb) for i in arr))
            a_in = [i for i in arr if i in set(ga)]
            a_order_kept += a_in == [i for i in ga if i in set(a_in)]
            a_all += len(a_in) == min(len(ga), 10)
        sec = [x["response"]["seconds"] for x in g]
        llm[arm] = {"calls": len(g), "valid": sum(x["valid"] for x in g),
                    "failure_reasons": dict(Counter(x["error"] for x in g if not x["valid"])),
                    "calls_invalid_by_past_assets": {k: by_size[k] for k in ("1", "2-3", "4+") if k in by_size},
                    "valid_copies_of_input_order": copies,
                    "valid_with_group_A_order_kept": a_order_kept,
                    "valid_keeping_all_group_A": a_all,
                    "mean_group_B_picks": round(statistics.mean(b_picks), 2) if b_picks else None,
                    "valid_with_no_group_B_pick": sum(b == 0 for b in b_picks),
                    "seconds_mean": round(statistics.mean(sec), 2), "seconds_median": round(statistics.median(sec), 2),
                    "seconds_max": round(max(sec), 2),
                    "input_tokens_mean": round(statistics.mean(x["response"]["input_tokens"] for x in g)),
                    "output_tokens_mean": round(statistics.mean(x["response"]["output_tokens"] for x in g))}
    out["llm"] = llm

    # hand-check customer: the customer with purchases where A3 differs most from latest buy (ties -> smallest hash)
    m3 = "A3_llm_ranker_encoder_hybrid"
    cand = sorted(nonempty, key=lambda u: (-abs(ndcg(pred[m3][u], truth[u]) - ndcg(pred["latest_buy"][u], truth[u])), H(u)))
    u = cand[0]
    g3 = next(x for x in gens if x["arm"] == "A3" and x["user"] == H(u))
    out["example_user"] = {
        "user_id": u, "user_hash": H(u), "past_buy_assets": sorted(past[u]), "truth": sorted(truth[u]),
        "latest_buy": pred["latest_buy"][u], "A3": pred[m3][u],
        "latest_buy_hit_ranks": [k + 1 for k, i in enumerate(pred["latest_buy"][u]) if i in truth[u]],
        "A3_hit_ranks": [k + 1 for k, i in enumerate(pred[m3][u]) if i in truth[u]],
        "latest_buy_ndcg": ndcg(pred["latest_buy"][u], truth[u]), "A3_ndcg": ndcg(pred[m3][u], truth[u]),
        "A3_valid": g3["valid"], "A3_raw_answer": g3["response"]["text"]}
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "example_user"}, indent=1))
    print("example:", json.dumps({k: v for k, v in out["example_user"].items() if k != "user_id"}, indent=1))


if __name__ == "__main__":
    main()
