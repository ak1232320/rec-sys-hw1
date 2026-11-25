#!/usr/bin/env python3
"""Session 1 pilot: a classical learned ranker for the user's own past assets (prefix).

Pointwise logistic regression over (user, past asset) pairs with as-of features only.
Trained once on the six training-only windows (cutoffs 2019-03-01 ... 2020-06-01, all
labels end before 2020-09-01) and applied unchanged to the nine test cutoffs. The prefix
is re-ordered by predicted probability; the fill stays the popularity fill of latest_buy.
"""
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_gate import TEST_CUTOFFS, TRAIN_ONLY, fit_lr, load_all, ndcg10  # noqa: E402


def build(cutoff, data, horizon=30):
    tx, asset_rows, first_price, profile_rows = data
    end = (date.fromisoformat(cutoff) + timedelta(days=horizon)).isoformat()
    meta = {}
    for r in asset_rows:
        if r['timestamp'][:10] < cutoff and first_price.get(r['ISIN'], '9999') < cutoff:
            if r['ISIN'] not in meta or r['timestamp'] > meta[r['ISIN']]['timestamp']:
                meta[r['ISIN']] = r
    history, truth = defaultdict(Counter), defaultdict(set)
    last, first, last_sell, user_last = defaultdict(dict), defaultdict(dict), defaultdict(dict), {}
    for d, c, i, t in tx:
        if i not in meta or d >= end:
            continue
        if d < cutoff:
            if t == 'Buy':
                history[c][i] += 1
                last[c][i] = max(d, last[c].get(i, ''))
                first[c][i] = min(d, first[c].get(i, '9999'))
                user_last[c] = max(d, user_last.get(c, ''))
            elif t == 'Sell':
                last_sell[c][i] = max(d, last_sell[c].get(i, ''))
        elif t == 'Buy':
            truth[c].add(i)
    has_profile = {c for c, rs in profile_rows.items() if any(r['timestamp'][:10] < cutoff for r in rs)}
    pop = Counter(i for counts in history.values() for i in counts)
    pop_order = sorted(meta, key=lambda i: (-pop[i], i))
    cut = date.fromisoformat(cutoff)
    days = lambda d: (cut - date.fromisoformat(d)).days
    users = []
    for u in has_profile:
        if not history.get(u) or not truth.get(u):
            continue  # unlabelled users score 0 under every ordering
        lu = last[u]
        P = sorted(lu, key=lambda i: (-date.fromisoformat(lu[i]).toordinal(), i))
        feats = []
        for rank, i in enumerate(P):
            n = history[u][i]
            span = days(first[u][i]) - days(lu[i])
            feats.append({
                'log_days_since_last': math.log1p(days(lu[i])),
                'log_buy_count': math.log1p(n),
                'log_days_since_first': math.log1p(days(first[u][i])),
                'log_recency_rank': math.log1p(rank),
                'log_mean_gap': math.log1p(span / (n - 1)) if n > 1 else 0.0,
                'single_buy': 1.0 if n == 1 else 0.0,
                'sold_after_last_buy': 1.0 if last_sell[u].get(i, '') > lu[i] else 0.0,
                'log_user_days_since_any_buy': math.log1p(days(user_last[u])),
                'log_user_distinct': math.log1p(len(P)),
                'log_asset_popularity': math.log1p(pop[i]),
                f'cat={meta[i]["assetCategory"]}': 1.0,
            })
        users.append((u, P, feats, truth[u], [i for i in pop_order if i not in set(P)]))
    return users


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()
    data = load_all(Path(sys.argv[1]))
    X, y = [], []
    for c in TRAIN_ONLY:
        for u, P, feats, t, F in build(c, data):
            for i, f in zip(P, feats):
                X.append(f)
                y.append(1 if i in t else 0)
    t_fit = time.time()
    predict, weights = fit_lr(X, y, iters=150)
    fit_s = time.time() - t_fit
    per_cutoff, pooled = [], defaultdict(list)
    for c in TEST_CUTOFFS:
        sums, n = Counter(), 0
        t_rank = time.time()
        for u, P, feats, t, F in build(c, data):
            n += 1
            probs = {i: predict(f) for i, f in zip(P, feats)}
            lr_prefix = sorted(P, key=lambda i: (-probs[i], P.index(i)))
            vals = {'repeat_first': ndcg10((P + F)[:10], t), 'lr_prefix': ndcg10((lr_prefix + F)[:10], t)}
            for k, v in vals.items():
                sums[k] += v
                pooled[k].append(v)
            pooled['_new'].append(bool(t - set(P)))
            pooled['_over10'].append(len(P) > 10)
        per_cutoff.append({'cutoff': c, 'labelled': n, 'ndcg_labelled': {k: sums[k] / n for k in sums},
                           'rank_seconds': round(time.time() - t_rank, 1)})
    N = len(pooled['repeat_first'])

    def summary(flag=None, val=True):
        idx = [i for i in range(N) if flag is None or pooled[flag][i] == val]
        d = [pooled['lr_prefix'][i] - pooled['repeat_first'][i] for i in idx]
        return {'n': len(idx), 'repeat_first': sum(pooled['repeat_first'][i] for i in idx) / len(idx),
                'lr_prefix': sum(pooled['lr_prefix'][i] for i in idx) / len(idx),
                'better': sum(x > 1e-12 for x in d), 'worse': sum(x < -1e-12 for x in d)}

    res = {'train_pairs': len(X), 'train_positive_rate': sum(y) / len(y), 'fit_seconds': round(fit_s, 1),
           'weights_standardized': {k: round(v, 3) for k, v in sorted(weights.items(), key=lambda p: -abs(p[1]))},
           'per_cutoff': per_cutoff,
           'macro_mean_labelled': {k: sum(p['ndcg_labelled'][k] for p in per_cutoff) / len(per_cutoff)
                                   for k in ('repeat_first', 'lr_prefix')},
           'pooled_all_labelled': summary(), 'pooled_repeat_only': summary('_new', False),
           'pooled_with_new': summary('_new', True), 'pooled_over10': summary('_over10', True),
           'total_seconds': round(time.time() - t0, 1)}
    Path(sys.argv[2]).write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('train_pairs', 'macro_mean_labelled', 'pooled_repeat_only',
                                          'pooled_with_new', 'pooled_over10', 'weights_standardized',
                                          'total_seconds')}, indent=1))


if __name__ == '__main__':
    main()
