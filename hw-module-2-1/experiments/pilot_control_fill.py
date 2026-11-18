"""Pilot of the NON-LLM arms of the planned comparison (no model is called here).

Planned study: keep the recency prefix of the provided `latest_buy` baseline (the customer's own
past buys, most recent first) and change only how the remaining Top-10 slots are filled:

  pop_fill     popularity order                      -> must reproduce the starter's latest_buy
  tfidf_fill   cosine in a TF-IDF space of the asset text (name + categorical attributes)
  tokens_fill  cosine in the starter's own 5-token categorical space (its CBF features)
  oracle_fill  the customer's new-to-user label assets (ceiling, uses the label - not a method)
  tfidf_topN   the N most popular unseen assets, re-ordered by the TF-IDF cosine (N = 15..100):
               a popularity prior with a personalised order; N is the one knob of the fill rule

The LLM arm (same pipeline, asset vectors from an LLM text-embedding model) is NOT run here; it
is the Session 2 work. This script exists to (a) check that the pipeline and its invariants hold
and (b) see whether a content-based fill is a credible competitor to the popularity fill at all.

Invariant checked on every run: a customer whose label assets were all bought before gets the
same nDCG@10 under every fill, because only the prefix can hit a repeat asset.

Run:  python -I experiments/pilot_control_fill.py [cutoff ...]     (from hw-module-2-1/)
Writes experiments/pilot_control_fill.json.
"""
import importlib.util
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data_raw" / "FAR-Trans"

spec = importlib.util.spec_from_file_location("starter", ROOT / "starter" / "starter.py")
starter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(starter)

HORIZON = 30
DEFAULT_CUTOFFS = ["2021-03-01", "2021-06-01", "2021-09-01"]
TOP_N = (15, 20, 30, 50, 100)   # shortlist sizes for the popularity-prior fill
TEXT_FIELDS = ("assetName", "assetCategory", "assetSubCategory", "sector", "industry", "marketID")
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def asset_text(row):
    """The text both encoders will read: as-of name and categorical attributes; missing -> omitted."""
    return " | ".join(row[f].strip() for f in TEXT_FIELDS if row.get(f, "").strip())


def tfidf_vectors(texts):
    """Unigram TF-IDF, smooth idf, L2-normalised. texts: {isin: str} -> {isin: {token: weight}}."""
    docs = {i: Counter(t for t in WORD.findall(s.casefold()) if len(t) > 1) for i, s in texts.items()}
    df = Counter(t for d in docs.values() for t in d)
    n = len(docs)
    out = {}
    for isin, tf in docs.items():
        vec = {t: c * (math.log((1 + n) / (1 + df[t])) + 1) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        out[isin] = {t: v / norm for t, v in vec.items()}
    return out


def token_vectors(features):
    """The starter's E1 item vectors: binary categorical tokens, L2-normalised."""
    out = {}
    for isin, values in features.items():
        unique = set(values)
        out[isin] = {v: 1 / math.sqrt(len(unique)) for v in unique} if unique else {}
    return out


def profile_scores(history_counts, vectors, inverted):
    """Cosine-like score of every asset against the buy-count-weighted sum of history vectors."""
    profile = defaultdict(float)
    for isin, weight in history_counts.items():
        for token, value in vectors.get(isin, {}).items():
            profile[token] += weight * value
    norm = math.sqrt(sum(v * v for v in profile.values()))
    scores = defaultdict(float)
    if norm:
        for token, pv in profile.items():
            for isin, value in inverted[token]:
                scores[isin] += pv * value / norm
    return scores


def invert(vectors):
    inverted = defaultdict(list)
    for isin, vec in vectors.items():
        for token, value in vec.items():
            inverted[token].append((isin, value))
    return inverted


def run_cutoff(cutoff):
    assets, profiles, history, truth, end, _ = starter.load_data(DATA, cutoff, HORIZON)
    users = starter.select_users(profiles, history, "warm", 0, None)
    demo_users = set(starter.select_users(profiles, history, "warm", 100, None))
    eligible = sorted(assets)
    eligible_set = set(eligible)
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    pop = starter.popularity_scores(history, assets)
    pop_order = starter.order(pop)
    pop_rank = {isin: r for r, isin in enumerate(pop_order)}
    events = starter.scan_recent_events(DATA, cutoff, assets)
    contexts = {uid: {"eligible_asset_ids": eligible} for uid in users}
    recent = starter.recency_rankings(contexts, pop, events)

    spaces = {"tfidf_fill": tfidf_vectors({i: asset_text(r) for i, r in assets.items()}),
              "tokens_fill": token_vectors(features)}
    inverted = {name: invert(vec) for name, vec in spaces.items()}
    arms = (["pop_fill", "tfidf_fill", "tokens_fill", "oracle_fill"]
            + [f"tfidf_top{n}" for n in TOP_N])

    per_user = {}
    new_hits, new_pairs = Counter(), 0
    mismatches = 0
    for uid in users:
        relevant = truth.get(uid, set()) & eligible_set
        hist = history[uid]
        starter_list = recent[uid]["latest_buy"]
        k_prefix = min(10, len(hist))
        prefix = starter_list[:k_prefix]
        taken = set(prefix)
        free = 10 - k_prefix
        lists = {"pop_fill": prefix + [i for i in pop_order if i not in taken][:free]}
        if lists["pop_fill"] != starter_list:
            mismatches += 1
        if not relevant:
            continue
        new = {i for i in relevant if i not in hist}
        for name in spaces:
            scores = profile_scores(hist, spaces[name], inverted[name]) if free else {}
            # Stable order: score, then popularity, then ISIN (pop_rank already breaks ties by ISIN).
            cands = sorted((i for i in eligible if i not in taken),
                           key=lambda i: (-scores.get(i, 0.0), pop_rank[i]))[:free] if free else []
            lists[name] = prefix + cands
            if name == "tfidf_fill":
                unseen_by_pop = [i for i in pop_order if i not in taken]
                for n in TOP_N:
                    short = sorted(unseen_by_pop[:max(n, free)],
                                   key=lambda i: (-scores.get(i, 0.0), pop_rank[i]))[:free]
                    lists[f"tfidf_top{n}"] = prefix + short
        oracle = prefix + sorted(new)[:free]
        lists["oracle_fill"] = oracle + [i for i in pop_order if i not in set(oracle)][:10 - len(oracle)]
        row = {"relevant": len(relevant), "new": len(new), "free_slots": free}
        for arm in arms:
            assert len(lists[arm]) == 10 and len(set(lists[arm])) == 10
            row[arm] = starter.ndcg10(lists[arm], relevant)
            new_hits[arm] += len(new & set(lists[arm]))
        new_pairs += len(new)
        per_user[uid] = row

    labelled = list(per_user)
    with_new = [u for u in labelled if per_user[u]["new"]]
    only_repeat = [u for u in labelled if not per_user[u]["new"]]
    # Invariant: the fill cannot change the score of a customer with only repeat labels.
    broken = [u for u in only_repeat
              if len({round(per_user[u][a], 12) for a in arms}) != 1]

    def mean(group, arm, denom=None):
        denom = denom or len(group)
        return sum(per_user[u][arm] for u in group) / denom if denom else None

    def paired(group, a, b):
        diffs = [per_user[u][a] - per_user[u][b] for u in group]
        return {"better": sum(d > 1e-12 for d in diffs), "worse": sum(d < -1e-12 for d in diffs),
                "equal": sum(abs(d) <= 1e-12 for d in diffs),
                "mean_diff": sum(diffs) / len(diffs) if diffs else None}

    demo_labelled = [u for u in labelled if u in demo_users]
    return {
        "cutoff": cutoff, "end_exclusive": end, "eligible_assets": len(assets),
        "warm_users": len(users), "labelled_users": len(labelled),
        "labelled_with_new_asset": len(with_new), "labelled_only_repeat": len(only_repeat),
        "new_to_user_label_pairs": new_pairs,
        "with_new_users_having_free_slots": sum(1 for u in with_new if per_user[u]["free_slots"]),
        "new_pairs_reachable_by_fill": sum(min(per_user[u]["new"], per_user[u]["free_slots"])
                                           for u in with_new),
        "pop_fill_differs_from_starter_latest_buy_users": mismatches,
        "invariant_only_repeat_users_broken": len(broken),
        "ndcg_all_warm": {a: mean(labelled, a, len(users)) for a in arms},
        "ndcg_labelled": {a: mean(labelled, a) for a in arms},
        "ndcg_with_new_asset": {a: mean(with_new, a) for a in arms},
        "ndcg_only_repeat": {a: mean(only_repeat, a) for a in arms},
        "new_pairs_in_top10": {a: new_hits[a] for a in arms},
        "paired_with_new_tfidf_vs_pop": paired(with_new, "tfidf_fill", "pop_fill"),
        "paired_with_new_tokens_vs_pop": paired(with_new, "tokens_fill", "pop_fill"),
        "demo_100": {"users": len(demo_users), "labelled": len(demo_labelled),
                     "ndcg_all": {a: mean(demo_labelled, a, len(demo_users)) for a in arms},
                     "ndcg_labelled": {a: mean(demo_labelled, a) for a in arms}},
    }


def main():
    cutoffs = sys.argv[1:] or DEFAULT_CUTOFFS
    t0 = time.time()
    out = {"note": "non-LLM arms only; the LLM-embedding arm is planned, not run",
           "text_fields": list(TEXT_FIELDS), "horizon_days": HORIZON, "cutoffs": []}
    for cutoff in cutoffs:
        block = run_cutoff(cutoff)
        out["cutoffs"].append(block)
        print(cutoff, json.dumps({k: block[k] for k in (
            "warm_users", "labelled_users", "labelled_with_new_asset",
            "pop_fill_differs_from_starter_latest_buy_users", "invariant_only_repeat_users_broken",
            "ndcg_labelled", "ndcg_with_new_asset", "new_pairs_in_top10",
            "new_to_user_label_pairs")}, indent=1), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (HERE / "pilot_control_fill.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote pilot_control_fill.json in", out["seconds"], "s")


if __name__ == "__main__":
    sys.exit(main())
