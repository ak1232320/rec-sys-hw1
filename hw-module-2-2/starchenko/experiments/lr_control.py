#!/usr/bin/env python3
"""Matched classical control (A2): logistic-regression order of the same past-asset prefix.

Training is the Session 1 pilot unchanged: six windows with cutoffs 2019-03-01 ... 2020-06-01,
labels end by 2020-07-01, i.e. eleven months before the 2021-06-01 evaluation cutoff.
At the evaluation cutoff the features are computed from the starter's own past events
(strictly before the cutoff) for the same 100 users and the same prefix as build_inputs.py.
Usage: python lr_control.py STARTER_DIR DATA_DIR INPUTS_DIR
"""
import json
import math
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

STARTER_DIR, DATA_DIR, INPUTS_DIR = map(Path, sys.argv[1:4])
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STARTER_DIR))
import starter  # noqa: E402
from pilot_gate import TRAIN_ONLY, fit_lr, load_all  # noqa: E402
from pilot_prefix_lr import build  # noqa: E402

CUTOFF = '2021-06-01'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()
    data = load_all(DATA_DIR)
    X, y = [], []
    for c in TRAIN_ONLY:
        for u, P, feats, t, F in build(c, data):
            for i, f in zip(P, feats):
                X.append(f)
                y.append(1 if i in t else 0)
    predict, weights = fit_lr(X, y, iters=150)
    fit_s = time.time() - t0

    t1 = time.time()
    assets, profiles, history, _labels_not_used, _e, _x = starter.load_data(DATA_DIR, CUTOFF, 30)
    del _labels_not_used
    pop = starter.popularity_scores(history, assets)
    events = starter.scan_recent_events(DATA_DIR, CUTOFF, assets)
    cut = date.fromisoformat(CUTOFF)
    days = lambda d: (cut - date.fromisoformat(d)).days
    rows = [json.loads(l) for l in (INPUTS_DIR / 'prefix_inputs.jsonl').open(encoding='utf-8')]
    out = []
    for r in rows:
        uid, P = r['user_id'], r['prefix']
        buys, sells = {}, {}
        for d, isin, kind in sorted(events.get(uid, ())):
            (buys if kind == 'Buy' else sells).setdefault(isin, []).append(d)
        user_last = max(b[-1] for b in buys.values())
        probs = {}
        for rank, i in enumerate(P):
            b = buys[i]
            n = len(b)
            span = days(b[0]) - days(b[-1])
            f = {'log_days_since_last': math.log1p(days(b[-1])),
                 'log_buy_count': math.log1p(n),
                 'log_days_since_first': math.log1p(days(b[0])),
                 'log_recency_rank': math.log1p(rank),
                 'log_mean_gap': math.log1p(span / (n - 1)) if n > 1 else 0.0,
                 'single_buy': 1.0 if n == 1 else 0.0,
                 'sold_after_last_buy': 1.0 if sells.get(i, [''])[-1] > b[-1] else 0.0,
                 'log_user_days_since_any_buy': math.log1p(days(user_last)),
                 'log_user_distinct': math.log1p(len(P)),
                 'log_asset_popularity': math.log1p(pop[i]),
                 f'cat={assets[i]["assetCategory"]}': 1.0}
            probs[i] = predict(f)
        lr_prefix = sorted(P, key=lambda i: (-probs[i], P.index(i)))
        out.append({'user_id': uid, 'lr_prefix': lr_prefix})
    with (INPUTS_DIR / 'lr_orders.jsonl').open('w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    meta = {'train_cutoffs': TRAIN_ONLY, 'train_pairs': len(X),
            'train_positive_rate': round(sum(y) / len(y), 4),
            'fit_seconds_incl_data_load': round(fit_s, 1),
            'scoring_seconds_incl_data_load': round(time.time() - t1, 1),
            'users': len(out),
            'users_where_lr_order_differs_from_recency': sum(
                r['lr_prefix'] != p['prefix'] for r, p in zip(out, rows)),
            'weights_standardized': {k: round(v, 3) for k, v in
                                     sorted(weights.items(), key=lambda p: -abs(p[1]))}}
    (INPUTS_DIR / 'lr_meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
