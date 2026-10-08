"""Offline reference and analysis for A04 (association rules on UCI Online Retail).

What it does, in order:
  1. Rebuilds the course baskets from the raw UCI workbook and checks they equal
     transactions.js exactly, which recovers each basket's InvoiceDate and CustomerID.
  2. Mines every threshold setting of the sweep with mlxtend's FP-Growth — an
     implementation independent of the Apriori in script.js — and stores the full
     rule tables the browser test compares the page against (reference_rules.json).
  3. Brute-forces the five-basket fixture, so the test can check the page there too.
  4. Recomputes the two rules discussed in the report straight from the raw rows.
  5. Measures the rules on time: per quarter, and mined on Dec 2010 - Aug 2011 then
     re-measured on Sep - Dec 2011.

Run:  python experiments/run_experiment.py      (~1 min; writes experiments/results.json)
Needs the raw workbook in data_raw/ (see README) and: pandas openpyxl mlxtend.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import time
import warnings

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from common import APP, RAW_XLSX, clean_raw, dated_baskets, load_course_dataset, load_raw, same_multiset

warnings.filterwarnings("ignore", category=FutureWarning)

OUT = APP / "experiments" / "results.json"
# Full rule tables for the browser test, one compact row per rule:
# [antecedent, consequent, count(A u B), count(A), count(B)].
OUT_RULES = APP / "experiments" / "reference_rules.json"
SUPPORTS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
CONFIDENCES = [0.1, 0.3, 0.5, 0.6]
# Settings whose full rule table is stored for the browser test.
TABLE_SETTINGS = [(0.01, 0.3), (0.02, 0.3), (0.03, 0.6), (0.01, 0.6), (0.005, 0.1)]
CHOSEN = (0.01, 0.3)
USEFUL = ("22386", "85099B")      # JUMBO BAG PINK POLKADOT -> JUMBO BAG RED RETROSPOT
MISLEADING = ("22916", "22917")   # HERB MARKER THYME -> HERB MARKER ROSEMARY
SPLIT_DATE = "2011-09-01"


# --------------------------------------------------------------------- mining

def encode(baskets):
    te = TransactionEncoder()
    rows = [sorted(b) for b in baskets]
    matrix = te.fit(rows).transform(rows, sparse=True)
    return pd.DataFrame.sparse.from_spmatrix(matrix, columns=te.columns_)


def mine(frame, n, min_support):
    """Frequent itemsets as {frozenset: count}, via mlxtend FP-Growth."""
    found = fpgrowth(frame, min_support=min_support - 1e-12, use_colnames=True)
    counts = {frozenset(s): int(round(sup * n)) for s, sup in zip(found["itemsets"], found["support"])}
    # Same inclusive threshold as script.js: count >= ceil(minSupport * n).
    min_count = max(1, int(np.ceil(min_support * n - 1e-9)))
    return {s: c for s, c in counts.items() if c >= min_count}


def rules_from(counts, n, min_confidence):
    rules = []
    for itemset, joint in counts.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for k in range(1, len(items)):
            for ante in itertools.combinations(items, k):
                a = frozenset(ante)
                b = itemset - a
                conf = joint / counts[a]
                if conf < min_confidence:
                    continue
                rules.append({
                    "antecedent": sorted(a), "consequent": sorted(b),
                    "jointCount": joint, "antecedentCount": counts[a], "consequentCount": counts[b],
                    "support": joint / n, "confidence": conf, "lift": conf / (counts[b] / n),
                })
    return rules


def by_size(counts):
    sizes = collections.Counter(len(s) for s in counts)
    return [sizes[k] for k in range(1, max(sizes) + 1)] if sizes else []


# ----------------------------------------------------------------- the fixture

FIXTURE = [["bread", "milk", "jam", "ham"], ["bread", "milk", "jam"], ["bread", "milk"],
           ["bread", "jam", "eggs"], ["bread", "eggs"]]


def fixture_reference():
    """Enumerate every itemset of the five-basket fixture by brute force."""
    items = sorted({i for b in FIXTURE for i in b})
    n = len(FIXTURE)
    all_counts = {}
    for k in range(1, len(items) + 1):
        for combo in itertools.combinations(items, k):
            c = sum(1 for b in FIXTURE if set(combo) <= set(b))
            if c:
                all_counts[frozenset(combo)] = c
    out = []
    for s in [0.2, 0.4, 0.6]:
        min_count = int(np.ceil(s * n - 1e-9))
        counts = {k: v for k, v in all_counts.items() if v >= min_count}
        for conf in [0.1, 0.5, 0.7]:
            out.append({"minSupport": s, "minConfidence": conf,
                        "itemsets": sorted([sorted(k), v] for k, v in counts.items()),
                        "rules": sorted([r["antecedent"], r["consequent"]] for r in rules_from(counts, n, conf))})
    return out


# ------------------------------------------------------------------ analysis

def rule_report(baskets, a, b, describe):
    items = baskets["items"]
    has_a = items.map(lambda s: a in s).to_numpy()
    has_b = items.map(lambda s: b in s).to_numpy()
    joint = has_a & has_b
    n = len(baskets)
    ca, cb, cj = int(has_a.sum()), int(has_b.sum()), int(joint.sum())
    customers = collections.Counter(baskets["customer"][joint])
    quarter = baskets["date"].dt.to_period("Q").astype(str)
    per_q = []
    for q in sorted(quarter.unique()):
        m = (quarter == q).to_numpy()
        qa, qb, qj, qn = int((has_a & m).sum()), int((has_b & m).sum()), int((joint & m).sum()), int(m.sum())
        per_q.append({"quarter": q, "n": qn, "countA": qa, "countB": qb, "joint": qj,
                      "confidence": qj / qa if qa else None,
                      "lift": (qj / qa) / (qb / qn) if qa and qb else None})
    return {
        "antecedent": a, "consequent": b,
        "antecedentDescription": describe[a], "consequentDescription": describe[b],
        "n": n, "countA": ca, "countB": cb, "joint": cj,
        "support": cj / n, "confidence": cj / ca, "reverseConfidence": cj / cb,
        "lift": (cj / ca) / (cb / n),
        # Lift can never exceed N / max(count(A), count(B)): the joint count is at most
        # the smaller of the two, so two rare items get a large lift mechanically.
        "liftUpperBound": n / max(ca, cb),
        "aWithoutB": ca - cj,
        "distinctCustomers": len(customers),
        "customersBuyingA": int(baskets["customer"][has_a].nunique()),
        "topCustomerBaskets": customers.most_common(1)[0][1],
        "medianBasketSizeJoint": float(baskets["items"][joint].map(len).median()),
        "medianBasketSizeAll": float(baskets["items"].map(len).median()),
        "quarters": per_q,
    }


def hand_count_from_raw(raw_clean, a, b):
    """count(A), count(B), count(A and B) as distinct InvoiceNo, straight from the raw rows."""
    inv_a = set(raw_clean.loc[raw_clean["StockCode"] == a, "InvoiceNo"])
    inv_b = set(raw_clean.loc[raw_clean["StockCode"] == b, "InvoiceNo"])
    return {"countA": len(inv_a), "countB": len(inv_b), "joint": len(inv_a & inv_b),
            "n": int(raw_clean["InvoiceNo"].nunique()),
            "rowsA": int((raw_clean["StockCode"] == a).sum())}


def temporal_holdout(baskets, describe):
    early = baskets[baskets["date"] < SPLIT_DATE]
    late = baskets[baskets["date"] >= SPLIT_DATE]
    fe, fl = encode(early["items"]), encode(late["items"])
    out = {"split": SPLIT_DATE, "earlyBaskets": len(early), "lateBaskets": len(late),
           "earlyRange": [str(early["date"].min()), str(early["date"].max())],
           "lateRange": [str(late["date"].min()), str(late["date"].max())], "bySupport": []}
    late_sets = list(late["items"])
    late_n = len(late)

    def late_counts(itemset):
        return sum(1 for s in late_sets if itemset <= s)

    for sup in [0.005, 0.01, 0.02]:
        ce = mine(fe, len(early), sup)
        cl = mine(fl, late_n, sup)
        re_ = rules_from(ce, len(early), 0.3)
        rl = {(tuple(r["antecedent"]), tuple(r["consequent"])) for r in rules_from(cl, late_n, 0.3)}
        replicated, lift_ok, conf_ok, support_ok, deltas, examples = 0, 0, 0, 0, [], []
        cache = {}
        for r in re_:
            a, b = frozenset(r["antecedent"]), frozenset(r["consequent"])
            for key in (a, b, a | b):
                if key not in cache:
                    cache[key] = late_counts(key)
            ca, cb, cj = cache[a], cache[b], cache[a | b]
            conf_late = cj / ca if ca else 0.0
            lift_late = conf_late / (cb / late_n) if cb else 0.0
            if (tuple(r["antecedent"]), tuple(r["consequent"])) in rl:
                replicated += 1
            if lift_late > 1:
                lift_ok += 1
            if conf_late >= 0.3:
                conf_ok += 1
            if cj / late_n >= sup:
                support_ok += 1
            deltas.append(conf_late - r["confidence"])
        # Rules of the later period whose itemset was not even frequent before.
        new_in_late = [r for r in rules_from(cl, late_n, 0.3)
                       if frozenset(r["antecedent"]) | frozenset(r["consequent"]) not in ce]
        top_new = sorted(new_in_late, key=lambda r: -r["jointCount"])[:5]
        for r in top_new:
            examples.append({"antecedent": [describe[i] for i in r["antecedent"]],
                             "consequent": [describe[i] for i in r["consequent"]],
                             "jointLate": r["jointCount"], "confidenceLate": r["confidence"],
                             "earlyJoint": sum(1 for s in early["items"]
                                               if set(r["antecedent"]) | set(r["consequent"]) <= s)})
        out["bySupport"].append({
            "minSupport": sup, "minConfidence": 0.3,
            "earlyRules": len(re_), "lateRules": len(rl),
            "replicated": replicated, "replicatedShare": replicated / len(re_) if re_ else None,
            "liftAboveOneLate": lift_ok,
            "confidenceHoldsLate": conf_ok,
            "supportHoldsLate": support_ok,
            "medianConfidenceChange": float(np.median(deltas)) if deltas else None,
            "medianAbsConfidenceChange": float(np.median(np.abs(deltas))) if deltas else None,
            "lateOnlyRules": len(new_in_late),
            "lateOnlyExamples": examples,
        })
    for name, (a, b) in (("useful", USEFUL), ("misleading", MISLEADING)):
        row = {}
        for label, part in (("early", early), ("late", late)):
            r = rule_report(part, a, b, describe)
            row[label] = {k: r[k] for k in ("n", "countA", "countB", "joint", "support", "confidence", "lift")}
        out[name] = row
    return out


def seasonal_examples(rules, baskets, describe):
    """Rules of the chosen table whose joint baskets fall mostly in Sep-Dec 2011."""
    late = (baskets["date"] >= SPLIT_DATE).to_numpy()
    sets = list(baskets["items"])
    share_late_baskets = late.mean()
    out = []
    seen = set()
    for r in rules:
        key = frozenset(r["antecedent"]) | frozenset(r["consequent"])
        if key in seen:
            continue
        seen.add(key)
        mask = np.fromiter((key <= s for s in sets), bool, len(sets))
        out.append({"items": [describe[i] for i in sorted(key)], "joint": int(mask.sum()),
                    "lateShare": float(late[mask].mean()), "lift": r["lift"]})
    out.sort(key=lambda e: -e["lateShare"])
    return {"lateShareOfAllBaskets": float(share_late_baskets),
            "itemsetsMostlyLate": sum(1 for e in out if e["lateShare"] >= 0.9),
            "itemsetsInTable": len(out), "top": out[:6]}


# ---------------------------------------------------------------------- main

def main():
    t0 = time.time()
    data = load_course_dataset()
    describe = data["description"]
    n = data["N_BASKETS"]
    baskets_course = data["basket_sets"]
    results = {}

    # 1. Raw reproduction.
    raw = load_raw()
    raw_clean = clean_raw(raw)
    dated = dated_baskets(raw_clean)
    extra, missing = same_multiset(dated["items"], baskets_course)
    results["raw"] = {
        "sha256": hashlib.sha256(RAW_XLSX.read_bytes()).hexdigest(),
        "rawRows": len(raw), "keptRows": len(raw_clean),
        "keptInvoiceItemPairs": int(raw_clean[["InvoiceNo", "StockCode"]].drop_duplicates().shape[0]),
        "baskets": len(dated), "basketsOnlyInRaw": extra, "basketsOnlyInCourse": missing,
        "dateRange": [str(dated["date"].min()), str(dated["date"].max())],
        "customers": int(dated["customer"].nunique()),
        "ukShare": float((dated["country"] == "United Kingdom").mean()),
        "basketsPerMonth": {str(k): int(v) for k, v in dated.groupby(dated["date"].dt.to_period("M")).size().items()},
    }
    print("raw reproduction:", results["raw"]["baskets"], "baskets, differences", extra, missing)

    # Dataset facts quoted in the report.
    item_counts = collections.Counter(i for s in baskets_course for i in s)
    top = item_counts.most_common(5)
    sizes = [len(s) for s in baskets_course]
    results["dataset"] = {
        "n": n, "items": data["N_ITEMS"],
        "top": [{"stock": s, "description": describe[s], "count": c, "support": c / n} for s, c in top],
        "basketSize": {"median": float(np.median(sizes)), "mean": float(np.mean(sizes)), "max": max(sizes)},
        "itemsAtOnePercent": sum(1 for c in item_counts.values() if c >= np.ceil(0.01 * n - 1e-9)),
        "itemsBelow10Baskets": sum(1 for c in item_counts.values() if c < 10),
    }

    # 2. Threshold sweep with FP-Growth.
    frame = encode(baskets_course)
    sweep, tables = [], {}
    for sup in SUPPORTS:
        t = time.time()
        counts = mine(frame, n, sup)
        elapsed = time.time() - t
        for conf in CONFIDENCES:
            rules = rules_from(counts, n, conf)
            sweep.append({"minSupport": sup, "minConfidence": conf, "itemsets": len(counts),
                          "itemsetsBySize": by_size(counts), "rules": len(rules),
                          "rulesLiftAtMostOne": sum(1 for r in rules if r["lift"] <= 1),
                          "minLift": min((r["lift"] for r in rules), default=None),
                          "fpgrowthSeconds": round(elapsed, 3)})
            if (sup, conf) in TABLE_SETTINGS:
                tables[f"{sup}/{conf}"] = rules
        print(f"support {sup:.3f}: {len(counts)} itemsets {by_size(counts)}")
    results["sweep"] = sweep
    results["maxItemSupport"] = top[0][1] / n
    results["confidenceFloorForLift1"] = top[0][1] / n

    chosen = tables[f"{CHOSEN[0]}/{CHOSEN[1]}"]
    chosen_sorted = sorted(chosen, key=lambda r: -r["lift"])
    results["chosen"] = {
        "minSupport": CHOSEN[0], "minConfidence": CHOSEN[1], "rules": len(chosen),
        "minLift": min(r["lift"] for r in chosen),
        "bottomLiftConsequents": collections.Counter(
            describe[r["consequent"][0]] for r in sorted(chosen, key=lambda r: r["lift"])[:10]
            if len(r["consequent"]) == 1).most_common(3),
        "top5": [{"a": [describe[i] for i in r["antecedent"]], "b": [describe[i] for i in r["consequent"]],
                  "joint": r["jointCount"], "confidence": r["confidence"], "lift": r["lift"]}
                 for r in chosen_sorted[:5]],
        "herbMarkerRules": sum(1 for r in chosen if all(describe[i].startswith("HERB MARKER")
                                                        for i in r["antecedent"] + r["consequent"])),
        "maxJoint": max(r["jointCount"] for r in chosen),
    }
    low = [r for r in tables["0.005/0.1"] if r["lift"] <= 1]
    results["liftAtMostOne"] = [{"a": [describe[i] for i in r["antecedent"]],
                                 "b": [describe[i] for i in r["consequent"]],
                                 "joint": r["jointCount"], "countA": r["antecedentCount"],
                                 "countB": r["consequentCount"], "confidence": r["confidence"],
                                 "lift": r["lift"]} for r in low]

    # 3. Fixture.
    results["fixture"] = fixture_reference()

    # 4. The two rules of the report, from the dated baskets and from the raw rows.
    results["useful"] = rule_report(dated, *USEFUL, describe)
    results["misleading"] = rule_report(dated, *MISLEADING, describe)
    results["useful"]["fromRawRows"] = hand_count_from_raw(raw_clean, *USEFUL)
    results["misleading"]["fromRawRows"] = hand_count_from_raw(raw_clean, *MISLEADING)
    herb = [s for s in data["stocks"] if describe[s].startswith("HERB MARKER")]
    herb_mask = dated["items"].map(lambda s: len(s & set(herb)))
    results["misleading"]["herbMarkerLine"] = {
        "codes": {s: describe[s] for s in herb},
        "basketsWithAny": int((herb_mask > 0).sum()),
        "basketsWithAtLeastFive": int((herb_mask >= 5).sum()),
    }

    # 5. Time.
    results["temporal"] = temporal_holdout(dated, describe)
    results["seasonal"] = seasonal_examples(chosen, dated, describe)
    results["newProducts"] = {
        "23321 SMALL WHITE HEART OF WICKER first seen": str(dated.loc[dated["items"].map(lambda s: "23321" in s), "date"].min()),
        "23245 SET OF 3 REGENCY CAKE TINS first seen": str(dated.loc[dated["items"].map(lambda s: "23245" in s), "date"].min()),
    }
    results["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    compact = {key: [[r["antecedent"], r["consequent"], r["jointCount"], r["antecedentCount"],
                      r["consequentCount"]] for r in rules] for key, rules in tables.items()}
    OUT_RULES.write_text(json.dumps({"n": n, "tables": compact}, separators=(",", ":")), encoding="utf-8")
    print(f"saved {OUT} in {results['seconds']} s")


if __name__ == "__main__":
    main()
