#!/usr/bin/env python3
"""Session 1 headroom pilot: where can a component placed after the recency prefix gain?

Same row policy and cutoffs as pilot_gate.py. All arms keep the user's past assets P
(latest Buy first) and differ only in what follows, or in P's internal order:
  repeat_first        (P + F_pop)[:10]           = starter latest_buy
  freq_prefix         P ordered by past Buy count, then recency, + F_pop
  oracle_prefix       P with label assets moved first (ceiling of reordering P)
  copurchase_fill     P + never-bought assets ordered by item-item co-purchase cosine
                      with P over all customers' pre-cutoff histories (popularity breaks ties)
  oracle_cat_fill     P + F_pop restricted first to the (category, subcategory) of the
                      user's new label assets (ceiling of predicting the category to explore)
  oracle_fill         P + new label assets first (ceiling of the fill, uses labels)
Oracle arms read the label and are ceilings, not methods.
"""
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_gate import TEST_CUTOFFS, load_all, ndcg10  # noqa: E402

ARMS = ['repeat_first', 'freq_prefix', 'oracle_prefix', 'copurchase_fill',
        'oracle_cat_fill', 'oracle_fill']


def run(data_dir, horizon=30):
    tx, asset_rows, first_price, profile_rows = load_all(data_dir)
    per_cutoff, pooled = [], defaultdict(list)
    for cutoff in TEST_CUTOFFS:
        end = (date.fromisoformat(cutoff) + timedelta(days=horizon)).isoformat()
        meta = {}
        for r in asset_rows:
            if r['timestamp'][:10] < cutoff and first_price.get(r['ISIN'], '9999') < cutoff:
                if r['ISIN'] not in meta or r['timestamp'] > meta[r['ISIN']]['timestamp']:
                    meta[r['ISIN']] = r
        assets = set(meta)
        cat = {i: (meta[i]['assetCategory'], meta[i]['assetSubCategory']) for i in assets}
        history, truth, last = defaultdict(Counter), defaultdict(set), defaultdict(dict)
        for d, c, i, t in tx:
            if t != 'Buy' or i not in assets:
                continue
            if d < cutoff:
                history[c][i] += 1
                if d > last[c].get(i, ''):
                    last[c][i] = d
            elif d < end:
                truth[c].add(i)
        has_profile = {c for c, rs in profile_rows.items() if any(r['timestamp'][:10] < cutoff for r in rs)}
        users = [u for u in has_profile if history.get(u)]
        pop = Counter(i for counts in history.values() for i in counts)
        pop_order = sorted(assets, key=lambda i: (-pop[i], i))
        co = defaultdict(Counter)
        for counts in history.values():
            items = list(counts)
            for a in items:
                for b in items:
                    if a != b:
                        co[a][b] += 1
        sums, n_lab, n_new, over10 = Counter(), 0, 0, 0
        for u in users:
            t = truth.get(u)
            if not t:
                continue
            n_lab += 1
            lu = last[u]
            P = sorted(lu, key=lambda i: (-date.fromisoformat(lu[i]).toordinal(), i))
            Pset = set(P)
            over10 += len(P) > 10
            F = [i for i in pop_order if i not in Pset]
            new = t - Pset
            n_new += bool(new)
            lists = {'repeat_first': (P + F)[:10]}
            cnt = history[u]
            freq = sorted(P, key=lambda i: (-cnt[i], -date.fromisoformat(lu[i]).toordinal(), i))
            lists['freq_prefix'] = (freq + F)[:10]
            lists['oracle_prefix'] = ([i for i in P if i in t] + [i for i in P if i not in t] + F)[:10]
            score = Counter()
            for a in P:
                for b, n in co[a].items():
                    if b not in Pset:
                        score[b] += n / math.sqrt(pop[a] * pop[b])
            cf = sorted(F, key=lambda i: (-score[i], -pop[i], i))
            lists['copurchase_fill'] = (P + cf)[:10]
            cats = {cat[i] for i in new}
            catf = [i for i in F if cat[i] in cats] + [i for i in F if cat[i] not in cats]
            lists['oracle_cat_fill'] = (P + catf)[:10]
            lists['oracle_fill'] = (P + sorted(new) + [i for i in F if i not in new])[:10]
            for k, lst in lists.items():
                v = ndcg10(lst, t)
                sums[k] += v
                pooled[k].append(v)
            pooled['_new'].append(bool(new))
            pooled['_over10'].append(len(P) > 10)
        per_cutoff.append({'cutoff': cutoff, 'warm_users': len(users), 'labelled': n_lab,
                           'with_new': n_new, 'labelled_with_over10_past_assets': over10,
                           'ndcg_labelled': {k: sums[k] / n_lab for k in ARMS}})
    n = len(pooled['repeat_first'])

    def mean_on(k, flag=None, val=True):
        idx = [i for i in range(n) if flag is None or pooled[flag][i] == val]
        return {'n': len(idx), 'mean': sum(pooled[k][i] for i in idx) / len(idx)}

    def paired(k, flag=None, val=True):
        idx = [i for i in range(n) if flag is None or pooled[flag][i] == val]
        d = [pooled[k][i] - pooled['repeat_first'][i] for i in idx]
        return {'n': len(idx), 'mean_diff': sum(d) / len(idx), 'better': sum(x > 1e-12 for x in d),
                'worse': sum(x < -1e-12 for x in d)}

    return {
        'per_cutoff': per_cutoff,
        'macro_mean_labelled': {k: sum(p['ndcg_labelled'][k] for p in per_cutoff) / len(per_cutoff) for k in ARMS},
        'pooled_labelled': {k: mean_on(k) for k in ARMS},
        'pooled_with_new': {k: mean_on(k, '_new', True) for k in ARMS},
        'pooled_repeat_only': {k: mean_on(k, '_new', False) for k in ARMS},
        'pooled_over10_past_assets': {k: mean_on(k, '_over10', True) for k in ARMS},
        'paired_vs_repeat_first_with_new': {k: paired(k, '_new', True) for k in ARMS[1:]},
        'paired_vs_repeat_first_repeat_only': {k: paired(k, '_new', False) for k in ARMS[1:]},
    }


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()
    res = run(Path(sys.argv[1]))
    res['timing_s'] = round(time.time() - t0, 1)
    Path(sys.argv[2]).write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('macro_mean_labelled', 'pooled_with_new', 'pooled_repeat_only',
                                          'pooled_over10_past_assets', 'timing_s')}, indent=1))
