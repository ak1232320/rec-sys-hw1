"""Pick the one raw record shown in the report (homework section 4) and print its context.

The record is a label-window Buy of an asset the customer had never bought before the cutoff
("new-to-user"), made by a customer of the starter's default 100-user demo cohort. Among the
candidates the script takes the first one in (customerID hash order, timestamp, transactionID)
order, so the choice is deterministic and not tuned by hand.

It prints the verbatim transaction row, the as-of asset_information row, the as-of
customer_information row, the customer's pre-cutoff Buy history and the four baseline Top-10
lists for that customer. The customerID is printed masked (first 4 characters) - the public
release already pseudonymises it, the report does not repeat the full value.

Run:  python -I experiments/find_record.py     (from hw-module-2-1/)
Writes experiments/record.json.
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data_raw" / "FAR-Trans"

spec = importlib.util.spec_from_file_location("starter", ROOT / "starter" / "starter.py")
starter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(starter)

CUTOFF, HORIZON = "2021-06-01", 30


def mask(customer_id):
    return customer_id[:4] + "…"


def main():
    assets, profiles, history, truth, end, _ = starter.load_data(DATA, CUTOFF, HORIZON)
    users = starter.select_users(profiles, history, "warm", 100, None)
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    pop = starter.popularity_scores(history, assets)
    contexts, rankings = starter.contexts_for(users, assets, profiles, history, features, pop,
                                              CUTOFF, 50, False)
    events = starter.scan_recent_events(DATA, CUTOFF, assets)
    recent = starter.recency_rankings(contexts, pop, events)
    for uid in users:
        rankings[uid].update(recent[uid])

    order_of = {uid: i for i, uid in enumerate(users)}
    labelled = [u for u in users if truth.get(u, set()) & set(assets)]
    summary = []
    for uid in labelled:
        rel = truth[uid] & set(assets)
        new = sorted(i for i in rel if i not in history[uid])
        summary.append({"cohort_position": order_of[uid], "customer": mask(uid),
                        "distinct_past_assets": len(history[uid]),
                        "label_assets": len(rel), "new_to_user": len(new),
                        "ndcg": {m: round(starter.ndcg10(rankings[uid][m], rel), 4)
                                 for m in ("popularity", "cbf", "latest_buy", "latest_buy_or_sell")}})
    print("labelled users in the 100-user demo cohort:", len(labelled))
    for row in summary:
        print(" ", row)

    candidates = []
    for row in starter.rows(DATA / "transactions.csv"):
        uid, isin, day = row["customerID"], row["ISIN"], row["timestamp"][:10]
        if (uid in order_of and row["transactionType"] == "Buy" and CUTOFF <= day < end
                and int(row["transactionID"]) >= 0 and isin in assets and isin not in history[uid]):
            candidates.append((order_of[uid], day, int(row["transactionID"]), row))
    candidates.sort(key=lambda c: c[:3])
    print("new-to-user label Buy rows in the demo cohort:", len(candidates))
    _, _, _, tx = candidates[0]
    uid, isin = tx["customerID"], tx["ISIN"]

    asset_rows = [r for r in starter.rows(DATA / "asset_information.csv") if r["ISIN"] == isin]
    cust_rows = [r for r in starter.rows(DATA / "customer_information.csv") if r["customerID"] == uid]
    prices = [r for r in starter.rows(DATA / "close_prices.csv") if r["ISIN"] == isin]
    before = [r for r in prices if r["timestamp"][:10] < CUTOFF]
    past = sorted((t, i, k) for t, i, k in events.get(uid, ()))
    rel = truth[uid] & set(assets)
    out = {
        "cutoff": CUTOFF, "end_exclusive": end,
        "transaction_row_verbatim": dict(tx, customerID=mask(uid)),
        "asset_information_rows_for_isin": asset_rows,
        "asset_row_as_of_cutoff": assets[isin],
        "starter_tokens_for_asset": features[isin],
        "customer_information_rows": [dict(r, customerID=mask(uid)) for r in cust_rows],
        "customer_row_as_of_cutoff": dict(profiles[uid], customerID=mask(uid)),
        "close_price_rows_for_isin": len(prices),
        "close_price_first_date": prices[0]["timestamp"] if prices else None,
        "last_close_before_cutoff": before[-1] if before else None,
        "customer_past_events_before_cutoff": [
            {"date": t, "ISIN": i, "type": k, "assetName": assets[i]["assetName"],
             "tokens": features[i]} for t, i, k in past],
        "customer_distinct_past_buy_assets": len(history[uid]),
        "label_assets": sorted(rel),
        "label_assets_new_to_user": sorted(i for i in rel if i not in history[uid]),
        "top10": {m: [{"ISIN": i, "name": assets[i]["assetName"], "hit": i in rel}
                      for i in rankings[uid][m]]
                  for m in ("popularity", "cbf", "latest_buy", "latest_buy_or_sell")},
        "ndcg_at_10": {m: starter.ndcg10(rankings[uid][m], rel)
                       for m in ("popularity", "cbf", "latest_buy", "latest_buy_or_sell")},
        "assets_sharing_this_assets_token_set": sum(
            1 for f in features.values() if f == features[isin]),
        "rank_of_label_asset": {
            "popularity": starter.order(pop).index(isin) + 1,
            "cbf": starter.order(starter.content_scores(history[uid], features)).index(isin) + 1,
        },
        "token_set_sizes": dict(Counter(len(f) for f in features.values())),
    }
    (HERE / "record.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
