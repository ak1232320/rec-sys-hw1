#!/usr/bin/env python3
"""Session 1 pilot for the repeat-or-explore gate on FAR-Trans. Standard library only.

Row policy follows the course starter (starter.py): exact duplicate rows removed,
negative transactionID rows excluded, history = Buy rows strictly before the cutoff,
truth = Buy rows in [cutoff, cutoff + horizon), both restricted to eligible assets
(as-of metadata and a positive pre-cutoff close price). Warm cohort = customers with
an as-of profile row and at least one eligible past Buy. Popularity = distinct past
buyers per asset over all customers.

Lists per user (P = past bought assets ordered by latest Buy date, then ISIN;
F = never-bought eligible assets in popularity order):
  repeat_first      = (P + F)[:10]            identical to the starter's latest_buy
  explore_first_m   = (F[:m] + P + F[m:])[:10]
Gates choose repeat_first or explore_first_m per user:
  lr_gate           logistic regression on as-of features, trained only on earlier
                    windows whose labels end on or before the test cutoff
  oracle_gate       explore iff the label window holds a new-to-user asset (ceiling,
                    reads the label, not a method)
"""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

TEST_CUTOFFS = ['2020-09-01', '2020-12-01', '2021-03-01', '2021-06-01', '2021-09-01',
                '2021-12-01', '2022-03-01', '2022-06-01', '2022-09-01']
TRAIN_ONLY = ['2019-03-01', '2019-06-01', '2019-09-01', '2019-12-01', '2020-03-01',
              '2020-06-01']
MS = (1, 3, 5)
PROFILE_KEYS = ('customerType', 'riskLevel', 'investmentCapacity')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def ndcg10(ranking, truth):
    if not truth:
        return 0.0
    gain = sum(1 / math.log2(i + 2) for i, isin in enumerate(ranking[:10]) if isin in truth)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(truth))))
    return gain / ideal


def load_all(data_dir):
    tx, seen = [], set()
    for r in rows(data_dir / 'transactions.csv'):
        key = tuple(sorted(r.items()))
        if key in seen:
            continue
        seen.add(key)
        if int(r['transactionID']) < 0:
            continue
        tx.append((r['timestamp'][:10], r['customerID'], r['ISIN'], r['transactionType']))
    asset_rows = list(rows(data_dir / 'asset_information.csv'))
    first_price = {}
    for r in rows(data_dir / 'close_prices.csv'):
        if float(r['closePrice']) > 0:
            d = r['timestamp'][:10]
            if d < first_price.get(r['ISIN'], '9999'):
                first_price[r['ISIN']] = d
    profile_rows = defaultdict(list)
    for r in rows(data_dir / 'customer_information.csv'):
        profile_rows[r['customerID']].append(r)
    return tx, asset_rows, first_price, profile_rows


def window(cutoff, horizon, tx, asset_rows, first_price, profile_rows):
    end = (date.fromisoformat(cutoff) + timedelta(days=horizon)).isoformat()
    assets = {r['ISIN'] for r in asset_rows
              if r['timestamp'][:10] < cutoff and first_price.get(r['ISIN'], '9999') < cutoff}
    history, truth = defaultdict(Counter), defaultdict(set)
    buy_last, buy_first, buy_dates = defaultdict(dict), defaultdict(dict), defaultdict(list)
    for d, c, i, t in tx:
        if t != 'Buy' or i not in assets:
            continue
        if d < cutoff:
            history[c][i] += 1
            if d > buy_last[c].get(i, ''):
                buy_last[c][i] = d
            if d < buy_first[c].get(i, '9999'):
                buy_first[c][i] = d
            buy_dates[c].append(d)
        elif d < end:
            truth[c].add(i)
    profiles = {}
    for c, rs in profile_rows.items():
        best = None
        for r in rs:
            if r['timestamp'][:10] < cutoff and (best is None or r['timestamp'] > best['timestamp']):
                best = r
        if best is not None:
            profiles[c] = best
    users = [u for u in profiles if history.get(u)]
    pop = Counter(i for counts in history.values() for i in counts)
    pop_order = sorted(assets, key=lambda i: (-pop[i], i))
    cut = date.fromisoformat(cutoff)
    out = {'cutoff': cutoff, 'end': end, 'eligible_assets': len(assets), 'users': {}}
    for u in users:
        last = buy_last[u]
        P = sorted(last, key=lambda i: (-date.fromisoformat(last[i]).toordinal(), i))
        Pset = set(P)
        F = [i for i in pop_order if i not in Pset]
        lists = {'repeat_first': (P + F)[:10]}
        for m in MS:
            lists[f'explore_first_{m}'] = (F[:m] + P + F[m:])[:10]
        t = truth.get(u, set())
        dates = sorted(buy_dates[u])
        n_buys, n_dist = len(dates), len(P)
        recent90 = (cut - timedelta(days=90)).isoformat()
        prof = profiles[u]
        rec = {
            'nonempty': bool(t),
            'new_in_truth': bool(t - Pset),
            'repeat_only_truth': bool(t) and not (t - Pset),
            'n_distinct': n_dist,
            'features': {
                'log_distinct': math.log1p(n_dist),
                'log_buys': math.log1p(n_buys),
                'repeat_ratio': 1 - n_dist / n_buys,
                'log_days_since_last': math.log1p((cut - date.fromisoformat(dates[-1])).days),
                'log_days_since_first': math.log1p((cut - date.fromisoformat(dates[0])).days),
                'log_buys_90d': math.log1p(sum(d >= recent90 for d in dates)),
                'log_new_assets_90d': math.log1p(sum(f >= recent90 for f in buy_first[u].values())),
                **{f'{k}={prof.get(k, "")}': 1.0 for k in PROFILE_KEYS},
            },
            'ndcg': {name: ndcg10(lst, t) for name, lst in lists.items()},
            'repeat_first_list': lists['repeat_first'],
        }
        out['users'][u] = rec
    return out


def fit_lr(X, y, iters=200, lr=0.5, l2=1e-3):
    keys = sorted({k for x in X for k in x})
    n = len(X)
    mean = {k: sum(x.get(k, 0.0) for x in X) / n for k in keys}
    std = {k: math.sqrt(sum((x.get(k, 0.0) - mean[k]) ** 2 for x in X) / n) or 1.0 for k in keys}
    Z = [[(x.get(k, 0.0) - mean[k]) / std[k] for k in keys] for x in X]
    w, b = [0.0] * len(keys), 0.0
    for _ in range(iters):
        gw, gb = [0.0] * len(keys), 0.0
        for z, yi in zip(Z, y):
            s = b + sum(wi * zi for wi, zi in zip(w, z))
            p = 1 / (1 + math.exp(-max(-30, min(30, s))))
            e = p - yi
            gb += e
            for j, zj in enumerate(z):
                gw[j] += e * zj
        b -= lr * gb / n
        w = [wi - lr * (g / n + l2 * wi) for wi, g in zip(w, gw)]

    def predict(x):
        s = b + sum(wi * (x.get(k, 0.0) - mean[k]) / std[k] for wi, k in zip(w, keys))
        return 1 / (1 + math.exp(-max(-30, min(30, s))))
    return predict, dict(zip(keys, w))


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--starter-predictions', type=Path,
                    help='predictions.csv of the starter demo run at 2021-06-01 for the identity check')
    ap.add_argument('--horizon-days', type=int, default=30)
    args = ap.parse_args()
    t0 = time.time()
    data = load_all(args.data_dir)
    t_load = time.time() - t0
    windows = {}
    for c in TRAIN_ONLY + TEST_CUTOFFS:
        windows[c] = window(c, args.horizon_days, *data)
    t_windows = time.time() - t0 - t_load

    result = {'horizon_days': args.horizon_days, 'test_cutoffs': TEST_CUTOFFS,
              'train_only_cutoffs': TRAIN_ONLY, 'per_cutoff': [],
              'timing_s': {'load': round(t_load, 1), 'windows': round(t_windows, 1)}}
    methods = ['repeat_first'] + [f'explore_first_{m}' for m in MS] + \
              [f'lr_gate_{m}' for m in MS] + [f'oracle_gate_{m}' for m in MS]
    pooled = defaultdict(list)  # labelled user-windows over test cutoffs
    explore_by_bucket = defaultdict(lambda: [0, 0])
    for c in TEST_CUTOFFS:
        w = windows[c]
        train_cs = [x for x in TRAIN_ONLY + TEST_CUTOFFS
                    if (date.fromisoformat(x) + timedelta(days=args.horizon_days)).isoformat() <= c]
        X, y, train_recs = [], [], []
        for x in train_cs:
            for rec in windows[x]['users'].values():
                if rec['nonempty']:
                    X.append(rec['features'])
                    y.append(1 if rec['new_in_truth'] else 0)
                    train_recs.append(rec)
        predict, weights = fit_lr(X, y)
        p_train = [predict(r['features']) for r in train_recs]
        thresholds = {}
        for m in MS:
            best = None
            for k in range(5, 96, 5):
                tau = k / 100
                score = sum(r['ndcg'][f'explore_first_{m}'] if p > tau else r['ndcg']['repeat_first']
                            for r, p in zip(train_recs, p_train))
                if best is None or score > best[0]:
                    best = (score, tau)
            thresholds[m] = best[1]
        users = w['users']
        labelled = {u: r for u, r in users.items() if r['nonempty']}
        sums = Counter()
        p_test, y_test = [], []
        for u, r in labelled.items():
            p = predict(r['features'])
            p_test.append(p)
            y_test.append(r['new_in_truth'])
            vals = {'repeat_first': r['ndcg']['repeat_first']}
            for m in MS:
                vals[f'explore_first_{m}'] = r['ndcg'][f'explore_first_{m}']
                vals[f'lr_gate_{m}'] = r['ndcg'][f'explore_first_{m}'] if p > thresholds[m] else r['ndcg']['repeat_first']
                vals[f'oracle_gate_{m}'] = r['ndcg'][f'explore_first_{m}'] if r['new_in_truth'] else r['ndcg']['repeat_first']
            for k, v in vals.items():
                sums[k] += v
                pooled[k].append(v)
            pooled['_new_in_truth'].append(r['new_in_truth'])
            bucket = '1' if r['n_distinct'] == 1 else '2' if r['n_distinct'] == 2 else \
                '3-5' if r['n_distinct'] <= 5 else '6+'
            explore_by_bucket[bucket][0] += r['new_in_truth']
            explore_by_bucket[bucket][1] += 1
        n_all, n_lab = len(users), len(labelled)
        n_new = sum(r['new_in_truth'] for r in labelled.values())
        result['per_cutoff'].append({
            'cutoff': c, 'label_end_exclusive': w['end'], 'eligible_assets': w['eligible_assets'],
            'warm_users': n_all, 'labelled_users': n_lab, 'labelled_with_new_asset': n_new,
            'train_cutoffs': train_cs, 'train_labelled_windows': len(train_recs),
            'train_positive_rate': round(sum(y) / len(y), 4), 'thresholds': thresholds,
            'test_auc': auc(p_test, y_test),
            'ndcg_all_users': {k: sums[k] / n_all for k in methods},
            'ndcg_labelled': {k: sums[k] / n_lab for k in methods},
        })
    n = len(pooled['repeat_first'])
    new_flags = pooled['_new_in_truth']

    def paired(a, b, subset=None):
        idx = [i for i in range(n) if subset is None or subset[i]]
        diffs = [pooled[a][i] - pooled[b][i] for i in idx]
        return {'n': len(idx), 'mean_diff': sum(diffs) / len(idx),
                'better': sum(d > 1e-12 for d in diffs), 'worse': sum(d < -1e-12 for d in diffs),
                'equal': sum(abs(d) <= 1e-12 for d in diffs)}

    result['pooled_labelled_user_windows'] = n
    result['pooled_with_new_asset'] = sum(new_flags)
    result['macro_mean_labelled'] = {k: sum(pc['ndcg_labelled'][k] for pc in result['per_cutoff']) / len(TEST_CUTOFFS)
                                     for k in methods}
    result['macro_mean_all_users'] = {k: sum(pc['ndcg_all_users'][k] for pc in result['per_cutoff']) / len(TEST_CUTOFFS)
                                      for k in methods}
    result['paired_vs_repeat_first'] = {k: paired(k, 'repeat_first') for k in methods if k != 'repeat_first'}
    result['paired_vs_repeat_first_with_new'] = {k: paired(k, 'repeat_first', new_flags)
                                                 for k in methods if k != 'repeat_first'}
    result['paired_vs_repeat_first_repeat_only'] = {k: paired(k, 'repeat_first', [not f for f in new_flags])
                                                    for k in methods if k != 'repeat_first'}
    result['explore_rate_by_past_distinct_assets'] = {
        b: {'labelled_user_windows': v[1], 'with_new_asset': v[0], 'rate': round(v[0] / v[1], 4)}
        for b, v in sorted(explore_by_bucket.items())}

    if args.starter_predictions:
        w = windows['2021-06-01']
        starter = defaultdict(list)
        for r in rows(args.starter_predictions):
            if r['method'] == 'latest_buy':
                starter[r['user_id']].append((int(r['rank']), r['ISIN']))
        mism = 0
        for u, ranked in starter.items():
            mine = w['users'][u]['repeat_first_list']
            if [i for _, i in sorted(ranked)] != mine:
                mism += 1
        result['starter_identity_check'] = {'cutoff': '2021-06-01', 'users_compared': len(starter),
                                            'latest_buy_list_mismatches': mism}
    result['timing_s']['total'] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('pooled_labelled_user_windows', 'pooled_with_new_asset',
                                              'macro_mean_labelled', 'timing_s')}, indent=1))
    if 'starter_identity_check' in result:
        print(result['starter_identity_check'])


if __name__ == '__main__':
    main()
