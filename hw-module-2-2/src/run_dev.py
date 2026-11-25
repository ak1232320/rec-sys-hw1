"""Development evaluation: the same arms on nine quarterly cutoffs, all warm users.

Selection rule fixed before any LLM result (Session 1 plan): cutoffs 2020-09-01 ...
2022-09-01 (quarterly), 30-day window, every as-of customer with pre-cutoff Buy history
(starter `select_users(..., 'warm', 0)`), never filtered by future purchases. These are
development results, not official benchmark scores.

Unlabelled users score 0 under every list (E3), so lists are built only for labelled
users and all-user means divide by the full warm count; the `pop_fill == latest_buy`
identity is still checked on every warm user. Non-LLM arms are compared with the Session 1
pilot (hw-module-2-1/experiments/pilot_control_fill.json) as a regression check.

Run (from hw-module-2-2/):  python src/run_dev.py          -> outputs/dev/dev_metrics.json
Only aggregates are written; no customer identifier leaves the run.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from common import DATA, DEMO_USERS, DEV_CUTOFFS, EMBED_DIR, HORIZON_DAYS, ROOT, asset_text, load_starter
from embeddings import load_tables, vectors_for
from fill import ARMS, PRIMARY, DenseSpace, SparseSpace, tfidf_vectors, token_vectors, user_lists

PILOT = ROOT.parent / 'hw-module-2-1' / 'experiments' / 'pilot_control_fill.json'
ALL_ARMS = ARMS + ('oracle_fill',)
PAIRS = (('llm_fill', 'pop_fill'), ('llm_fill', 'tfidf_fill'), ('llm_fill', 'e5_fill'),
         ('llm_fill', 'tokens_fill'), ('e5_fill', 'pop_fill'), ('tfidf_fill', 'pop_fill'),
         ('llm_top20', 'pop_fill'), ('llm_top20', 'tfidf_top20'))


def run_cutoff(starter, data_dir, cutoff, tables):
    assets, profiles, history, truth, end, _ = starter.load_data(data_dir, cutoff, HORIZON_DAYS)
    users = starter.select_users(profiles, history, 'warm', 0, None)
    demo = set(starter.select_users(profiles, history, 'warm', DEMO_USERS, None))
    eligible = sorted(assets)
    eligible_set = set(eligible)
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    pop = starter.popularity_scores(history, assets)
    pop_order = starter.order(pop)
    pop_rank = {isin: r for r, isin in enumerate(pop_order)}
    events = starter.scan_recent_events(data_dir, cutoff, assets)
    recent = starter.recency_rankings({uid: {'eligible_asset_ids': eligible} for uid in users},
                                      pop, events)
    spaces = {'tokens_fill': SparseSpace(token_vectors(features)),
              'tfidf_fill': SparseSpace(tfidf_vectors({i: asset_text(r) for i, r in assets.items()}))}
    for arm, table in tables.items():
        spaces[arm] = DenseSpace(vectors_for(assets, table))

    mismatches, rows, fallbacks = 0, {}, {a: 0 for a in ALL_ARMS}
    for uid in users:
        relevant = truth.get(uid, set()) & eligible_set
        hist = history[uid]
        latest = recent[uid]['latest_buy']
        if not relevant:
            lists, _ = user_lists(latest, hist, eligible, pop_order, pop_rank, {})
            mismatches += lists['pop_fill'] != latest
            continue
        new = {i for i in relevant if i not in hist}
        lists, failed = user_lists(latest, hist, eligible, pop_order, pop_rank, spaces, new)
        mismatches += lists['pop_fill'] != latest
        for arm in failed:
            fallbacks[arm] += 1
        row = {'new': len(new), 'free': 10 - min(10, len(hist)), 'demo': uid in demo}
        for arm, ranking in lists.items():
            starter.validate_ranking(ranking, eligible)
            row[arm] = starter.ndcg10(ranking, relevant)
            row[arm + '__new_hits'] = len(new & set(ranking))
        rows[uid] = row

    arms = [a for a in ALL_ARMS if a in next(iter(rows.values()))]
    labelled = list(rows.values())
    with_new = [r for r in labelled if r['new']]
    only_repeat = [r for r in labelled if not r['new']]
    broken = sum(len({round(r[a], 12) for a in arms}) != 1 for r in only_repeat)
    mean = lambda group, arm, denom=None: (sum(r[arm] for r in group) / (denom or len(group))
                                           if (denom or group) else None)
    demo_rows = [r for r in labelled if r['demo']]
    return {
        'cutoff': cutoff, 'end_exclusive': end, 'eligible_assets': len(assets),
        'warm_users': len(users), 'labelled_users': len(labelled),
        'labelled_with_new_asset': len(with_new), 'labelled_only_repeat': len(only_repeat),
        'new_to_user_label_pairs': sum(r['new'] for r in labelled),
        'pop_fill_differs_from_latest_buy_users': mismatches,
        'invariant_only_repeat_users_broken': broken, 'fallbacks': fallbacks,
        'ndcg_all_warm': {a: mean(labelled, a, len(users)) for a in arms},
        'ndcg_labelled': {a: mean(labelled, a) for a in arms},
        'ndcg_with_new_asset': {a: mean(with_new, a) for a in arms},
        'new_pairs_in_top10': {a: sum(r[a + '__new_hits'] for r in labelled) for a in arms},
        'demo_100': {'labelled': len(demo_rows),
                     'ndcg_all': {a: mean(demo_rows, a, DEMO_USERS) for a in arms}},
    }, [{k: v for k, v in r.items() if '__' not in k} for r in with_new]


def paired(windows, a, b, reps=2000, seed=0):
    """Paired difference a - b over user-windows, with a percentile bootstrap 95% CI."""
    diffs = [w[a] - w[b] for w in windows]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return {'better': sum(d > 1e-12 for d in diffs), 'worse': sum(d < -1e-12 for d in diffs),
            'equal': sum(abs(d) <= 1e-12 for d in diffs), 'mean_diff': sum(diffs) / n,
            'ci95': [boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]],
            'note': 'user-windows resampled independently; a customer present at several '
                    'cutoffs is not clustered, so the interval is optimistic'}


def regression_vs_pilot(blocks):
    if not PILOT.exists():
        return None
    pilot = {c['cutoff']: c for c in json.loads(PILOT.read_text(encoding='utf-8'))['cutoffs']}
    worst = 0.0
    for block in blocks:
        old = pilot.get(block['cutoff'])
        for key in ('ndcg_labelled', 'ndcg_with_new_asset'):
            for arm in ('pop_fill', 'tokens_fill', 'tfidf_fill', 'tfidf_top20', 'oracle_fill'):
                worst = max(worst, abs(old[key][arm] - block[key][arm]))
    return {'arms': ['pop_fill', 'tokens_fill', 'tfidf_fill', 'tfidf_top20', 'oracle_fill'],
            'max_abs_diff_vs_session1_pilot': worst}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--embed-dir', type=Path, default=EMBED_DIR)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs' / 'dev' / 'dev_metrics.json')
    parser.add_argument('cutoffs', nargs='*', default=list(DEV_CUTOFFS))
    args = parser.parse_args()
    starter = load_starter()
    tables, manifest = load_tables(args.embed_dir)
    started = time.perf_counter()
    blocks, windows = [], []
    for cutoff in args.cutoffs:
        t = time.perf_counter()
        block, with_new = run_cutoff(starter, args.data_dir, cutoff, tables)
        block['seconds'] = round(time.perf_counter() - t, 1)
        blocks.append(block)
        windows.extend(with_new)
        print(cutoff, json.dumps({k: block[k] for k in ('labelled_users', 'labelled_with_new_asset',
              'pop_fill_differs_from_latest_buy_users', 'invariant_only_repeat_users_broken')}),
              {a: round(v, 4) for a, v in block['ndcg_with_new_asset'].items()}, flush=True)
    arms = list(blocks[0]['ndcg_labelled'])
    macro = lambda key: {a: sum(b[key][a] for b in blocks) / len(blocks) for a in arms}
    span = lambda key: {a: [min(b[key][a] for b in blocks), max(b[key][a] for b in blocks)] for a in arms}
    summary = {
        'cutoffs': args.cutoffs, 'primary_method': PRIMARY,
        'selection_rule': 'all warm as-of customers per cutoff; never filtered by future purchases',
        'labelled_user_windows': sum(b['labelled_users'] for b in blocks),
        'with_new_user_windows': sum(b['labelled_with_new_asset'] for b in blocks),
        'warm_user_windows': sum(b['warm_users'] for b in blocks),
        'pop_fill_differs_from_latest_buy_users': sum(b['pop_fill_differs_from_latest_buy_users'] for b in blocks),
        'invariant_only_repeat_users_broken': sum(b['invariant_only_repeat_users_broken'] for b in blocks),
        'fallbacks': {a: sum(b['fallbacks'][a] for b in blocks) for a in ALL_ARMS},
        'macro_ndcg_all_warm': macro('ndcg_all_warm'), 'macro_ndcg_labelled': macro('ndcg_labelled'),
        'macro_ndcg_with_new_asset': macro('ndcg_with_new_asset'),
        'range_ndcg_with_new_asset': span('ndcg_with_new_asset'),
        'new_pairs_in_top10': {a: sum(b['new_pairs_in_top10'][a] for b in blocks) for a in arms},
        'new_to_user_label_pairs': sum(b['new_to_user_label_pairs'] for b in blocks),
        'cutoffs_where_llm_beats_pop_with_new': sum(
            b['ndcg_with_new_asset']['llm_fill'] > b['ndcg_with_new_asset']['pop_fill'] + 1e-12
            for b in blocks) if 'llm_fill' in arms else None,
        'paired_with_new_pooled': {f'{a}-{b}': paired(windows, a, b)
                                   for a, b in PAIRS if a in arms and b in arms},
        'regression_check': regression_vs_pilot(blocks),
        'embedding_models': [{k: m[k] for k in ('key', 'repo', 'revision')} for m in manifest['models']],
        'seconds': round(time.perf_counter() - started, 1),
        'per_cutoff': blocks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('macro with-new:', {a: round(v, 4) for a, v in summary['macro_ndcg_with_new_asset'].items()})
    print('regression vs pilot:', summary['regression_check'])
    print(f"wrote {args.out} in {summary['seconds']} s")


if __name__ == '__main__':
    main()
