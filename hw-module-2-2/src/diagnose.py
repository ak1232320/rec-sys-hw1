"""Failure analysis of the fill arms on one development cutoff (aggregates only).

For labelled warm users whose label window holds an asset new to them and who have at
least one free slot, compare what each fill puts into those slots with what they bought:
popularity rank of the filled assets, asset-category mix, share of fill items from the
same issuer/fund family as a history asset (first word of the name), and how anisotropic
each embedding space is (mean pairwise cosine). No customer identifier is written.

Run (from hw-module-2-2/):  python src/diagnose.py [cutoff]   -> outputs/dev/diagnose_<cutoff>.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from common import DATA, HORIZON_DAYS, ROOT, asset_text, load_starter
from embeddings import load_tables, vectors_for
from fill import DenseSpace, SparseSpace, prefix_of, tfidf_vectors, token_vectors, user_lists

ARMS = ('pop_fill', 'tokens_fill', 'tfidf_fill', 'e5_fill', 'llm_fill', 'llm_top20')


def family(row):
    words = row['assetName'].replace('A/K', '').split()
    return words[0].casefold() if words else ''


def main():
    cutoff = sys.argv[1] if len(sys.argv) > 1 else '2021-06-01'
    starter = load_starter()
    assets, profiles, history, truth, end, _ = starter.load_data(DATA, cutoff, HORIZON_DAYS)
    users = starter.select_users(profiles, history, 'warm', 0, None)
    eligible = sorted(assets)
    features = {i: starter.tokens(r) for i, r in assets.items()}
    pop = starter.popularity_scores(history, assets)
    pop_order = starter.order(pop)
    pop_rank = {i: r for r, i in enumerate(pop_order)}
    events = starter.scan_recent_events(DATA, cutoff, assets)
    recent = starter.recency_rankings({u: {'eligible_asset_ids': eligible} for u in users}, pop, events)
    tables, _ = load_tables()
    spaces = {'tokens_fill': SparseSpace(token_vectors(features)),
              'tfidf_fill': SparseSpace(tfidf_vectors({i: asset_text(r) for i, r in assets.items()}))}
    for arm, table in tables.items():
        spaces[arm] = DenseSpace(vectors_for(assets, table))

    stats = {a: {'fill_items': 0, 'pop_rank_sum': 0, 'category': Counter(), 'same_family': 0,
                 'new_hits': 0} for a in ARMS}
    label_cat, n_users, n_new = Counter(), 0, 0
    for uid in users:
        relevant = truth.get(uid, set()) & set(eligible)
        hist = history[uid]
        new = {i for i in relevant if i not in hist}
        prefix = prefix_of(recent[uid]['latest_buy'], hist)
        if not new or len(prefix) == 10:
            continue
        n_users += 1
        n_new += len(new)
        label_cat.update(assets[i]['assetCategory'] for i in new)
        fams = {family(assets[i]) for i in hist}
        lists, _ = user_lists(recent[uid]['latest_buy'], hist, eligible, pop_order, pop_rank, spaces)
        for arm in ARMS:
            fill = lists[arm][len(prefix):]
            s = stats[arm]
            s['fill_items'] += len(fill)
            s['pop_rank_sum'] += sum(pop_rank[i] for i in fill)
            s['category'].update(assets[i]['assetCategory'] for i in fill)
            s['same_family'] += sum(family(assets[i]) in fams for i in fill)
            s['new_hits'] += len(new & set(fill))
    out = {'cutoff': cutoff, 'users_with_new_asset_and_free_slots': n_users,
           'new_label_pairs': n_new,
           'new_label_category_share': {k: v / n_new for k, v in label_cat.items()},
           'new_label_mean_popularity_rank': None, 'arms': {}}
    ranks = []
    for uid in users:
        hist = history[uid]
        relevant = truth.get(uid, set()) & set(eligible)
        if len(prefix_of(recent[uid]['latest_buy'], hist)) < 10:
            ranks += [pop_rank[i] for i in relevant if i not in hist]
    out['new_label_mean_popularity_rank'] = sum(ranks) / len(ranks)
    for arm, s in stats.items():
        n = s['fill_items']
        out['arms'][arm] = {'mean_popularity_rank_of_fill': s['pop_rank_sum'] / n,
                            'fill_category_share': {k: v / n for k, v in s['category'].items()},
                            'fill_share_same_name_family_as_history': s['same_family'] / n,
                            'new_label_hits': s['new_hits']}
    for arm in ('e5_fill', 'llm_fill'):
        m = spaces[arm].matrix
        sims = m @ m.T
        off = sims[~np.eye(len(m), dtype=bool)]
        out['arms'][arm]['mean_pairwise_cosine'] = float(off.mean())
        out['arms'][arm]['p10_p90_pairwise_cosine'] = [float(np.quantile(off, .1)), float(np.quantile(off, .9))]
    path = ROOT / 'outputs' / 'dev' / f'diagnose_{cutoff}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
