"""Profile FAR-Trans under the course starter's rules: how much of the label is repeat buying?

Everything here reuses the unchanged course starter (../starter/starter.py) for loading,
eligibility, popularity, CBF, the recency baselines and nDCG@10, so the numbers are computed
under exactly the starter's teaching protocol. Standard library only.

For every cutoff it reports, on the full warm cohort (every as-of customer with a past Buy):
  * how many (user, asset) label pairs are repeats (asset bought by that user before the cutoff)
    and how many are new to the user;
  * how many slots of the latest_buy Top-10 are popularity fill;
  * nDCG@10 of the four provided baselines, on all labelled users and on the two subgroups
    "only repeat labels" / "at least one new-to-user label";
  * how many new-to-user label pairs each baseline places in its Top-10;
  * an oracle that keeps the latest_buy prefix and fills the remaining slots with the user's
    new-to-user label assets: the ceiling for any change that touches only the fill.

Run:  python -I experiments/profile_data.py            (from hw-module-2-1/)
Writes experiments/profile.json.
"""
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data_raw" / "FAR-Trans"

spec = importlib.util.spec_from_file_location("starter", ROOT / "starter" / "starter.py")
starter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(starter)

MAIN_CUTOFF = "2021-06-01"
HORIZON = 30
# Monthly cutoffs around the starter default; each one is an independent as-of split.
CUTOFFS = ["2020-09-01", "2020-12-01", "2021-03-01", "2021-06-01",
           "2021-09-01", "2021-12-01", "2022-03-01", "2022-06-01", "2022-09-01"]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def profile_cutoff(cutoff, detailed=False):
    assets, profiles, history, truth, end, excluded = starter.load_data(DATA, cutoff, HORIZON)
    users = starter.select_users(profiles, history, "warm", 0, None)
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    pop = starter.popularity_scores(history, assets)
    contexts, rankings = starter.contexts_for(users, assets, profiles, history, features, pop,
                                              cutoff, 50, False)
    events = starter.scan_recent_events(DATA, cutoff, assets)
    recent = starter.recency_rankings(contexts, pop, events)
    for uid in users:
        rankings[uid].update(recent[uid])

    methods = ["popularity", "cbf", "latest_buy", "latest_buy_or_sell"]
    eligible = set(assets)
    pairs = repeat_pairs = 0
    labelled, with_new, only_repeat = [], [], []
    new_hits = Counter()
    repeat_hits = Counter()
    scores = {m: {} for m in methods + ["oracle_fill"]}
    fill_slots = []
    distinct_hist = []
    new_by_category = Counter()
    examples = []
    for uid in users:
        hist = history[uid]
        distinct_hist.append(len(hist))
        buy_prefix = min(10, len(hist))
        fill_slots.append(10 - buy_prefix)
        relevant = truth.get(uid, set()) & eligible
        if not relevant:
            continue
        labelled.append(uid)
        new = {i for i in relevant if i not in hist}
        pairs += len(relevant)
        repeat_pairs += len(relevant) - len(new)
        (with_new if new else only_repeat).append(uid)
        for isin in new:
            new_by_category[assets[isin]["assetCategory"]] += 1
        for m in methods:
            top = rankings[uid][m]
            scores[m][uid] = starter.ndcg10(top, relevant)
            new_hits[m] += len(new & set(top))
            repeat_hits[m] += len((relevant - new) & set(top))
        # Oracle fill: keep the latest_buy prefix (the user's own past buys, most recent first),
        # then put this user's new-to-user label assets into the free slots, then popularity.
        prefix = rankings[uid]["latest_buy"][:buy_prefix]
        oracle = prefix + sorted(new)[:10 - buy_prefix]
        for isin in rankings[uid]["popularity"]:
            if len(oracle) >= 10:
                break
            if isin not in oracle:
                oracle.append(isin)
        scores["oracle_fill"][uid] = starter.ndcg10(oracle, relevant)
        if detailed and new and len(hist) < 10:
            examples.append((uid, sorted(new), len(hist)))

    def block(group):
        return {m: mean(scores[m][u] for u in group) for m in scores}

    n_users = len(users)
    out = {
        "cutoff": cutoff, "end_exclusive": end, "excluded_rows": excluded,
        "eligible_assets": len(assets),
        "warm_users": n_users,
        "labelled_users": len(labelled),
        "labelled_users_only_repeat": len(only_repeat),
        "labelled_users_with_new_asset": len(with_new),
        "label_pairs": pairs,
        "label_pairs_repeat": repeat_pairs,
        "label_pairs_new_to_user": pairs - repeat_pairs,
        "share_pairs_new_to_user": (pairs - repeat_pairs) / pairs if pairs else None,
        "new_pairs_by_asset_category": dict(new_by_category),
        "history_distinct_assets": {
            "median": statistics.median(distinct_hist),
            "mean": mean(distinct_hist),
            "share_users_fewer_than_10": mean(1 if d < 10 else 0 for d in distinct_hist),
            "share_users_at_most_2": mean(1 if d <= 2 else 0 for d in distinct_hist),
        },
        "latest_buy_fill_slots": {
            "mean_per_user": mean(fill_slots),
            "share_of_all_slots": sum(fill_slots) / (10 * n_users),
        },
        "ndcg_all_warm_users": {m: sum(scores[m].values()) / n_users for m in scores},
        "ndcg_labelled_users": block(labelled),
        "ndcg_labelled_only_repeat": block(only_repeat),
        "ndcg_labelled_with_new_asset": block(with_new),
        "new_pairs_in_top10": {m: new_hits[m] for m in methods},
        "repeat_pairs_in_top10": {m: repeat_hits[m] for m in methods},
    }
    if detailed:
        out["_examples"] = examples
        out["_objects"] = (assets, profiles, history, truth, features, users, rankings)
    return out


def asset_text_profile(assets, features):
    """How much do the starter's five categorical tokens tell assets apart?"""
    by_tokens = Counter(tuple(t) for t in features.values())
    sizes = sorted(by_tokens.values(), reverse=True)
    greek = sum(1 for a in assets.values()
                if any("Ͱ" <= ch <= "Ͽ" or "ἀ" <= ch <= "῿" for ch in a["assetName"]))
    cats = Counter(a["assetCategory"] for a in assets.values())
    missing = {}
    for cat in cats:
        rows = [a for a in assets.values() if a["assetCategory"] == cat]
        missing[cat] = {
            "assets": len(rows),
            "sector_missing": sum(1 for a in rows if not a["sector"].strip()),
            "industry_missing": sum(1 for a in rows if not a["industry"].strip()),
            "subcategory_missing": sum(1 for a in rows if not a["assetSubCategory"].strip()),
        }
    return {
        "eligible_assets": len(assets),
        "by_category": dict(cats),
        "missing_by_category": missing,
        "names_with_greek_letters": greek,
        "distinct_token_sets": len(by_tokens),
        "assets_sharing_their_token_set_with_another": sum(s for s in sizes if s > 1),
        "largest_identical_token_groups": [
            {"tokens": list(t), "assets": n} for t, n in by_tokens.most_common(5)],
    }


def raw_counts():
    kinds, cats, dates, customers, isins = Counter(), Counter(), [], set(), set()
    n = 0
    for row in starter.rows(DATA / "transactions.csv"):
        n += 1
        kinds[row["transactionType"]] += 1
        dates.append(row["timestamp"][:10])
        customers.add(row["customerID"])
        isins.add(row["ISIN"])
    asset_rows = list(starter.rows(DATA / "asset_information.csv"))
    for isin in {r["ISIN"] for r in asset_rows}:
        cats[next(r["assetCategory"] for r in asset_rows if r["ISIN"] == isin)] += 1
    cust_rows = list(starter.rows(DATA / "customer_information.csv"))
    return {
        "transactions_rows": n, "transaction_types": dict(kinds),
        "transactions_first_date": min(dates), "transactions_last_date": max(dates),
        "transactions_customers": len(customers), "transactions_assets": len(isins),
        "asset_information_rows": len(asset_rows),
        "asset_information_distinct_isin": len({r["ISIN"] for r in asset_rows}),
        "asset_categories_distinct_isin": dict(cats),
        "customer_information_rows": len(cust_rows),
        "customer_information_distinct": len({r["customerID"] for r in cust_rows}),
    }


def main():
    t0 = time.time()
    result = {"protocol": "course starter rules (teaching defaults, not the official protocol)",
              "horizon_days": HORIZON, "raw": raw_counts(), "cutoffs": []}
    for cutoff in CUTOFFS:
        detailed = cutoff == MAIN_CUTOFF
        block = profile_cutoff(cutoff, detailed)
        if detailed:
            assets, profiles, history, truth, features, users, rankings = block.pop("_objects")
            block.pop("_examples")
            result["asset_text_at_main_cutoff"] = asset_text_profile(assets, features)
        result["cutoffs"].append(block)
        print(f"{cutoff}: warm {block['warm_users']:>6}  labelled {block['labelled_users']:>5}  "
              f"pairs {block['label_pairs']:>5}  new-to-user {block['share_pairs_new_to_user']:.3f}  "
              f"latest_buy {block['ndcg_labelled_users']['latest_buy']:.4f}  "
              f"oracle fill {block['ndcg_labelled_users']['oracle_fill']:.4f}", flush=True)
    result["seconds"] = round(time.time() - t0, 1)
    (HERE / "profile.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(f"wrote profile.json in {result['seconds']} s")


if __name__ == "__main__":
    sys.exit(main())
