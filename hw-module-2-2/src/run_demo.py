"""HW2 run: four provided baselines + prefix/fill arms on the starter's 100-user protocol.

Everything except the fill rule comes from the unchanged starter: data loading, as-of
metadata, eligibility, the deterministic warm 100-user sample, the four baselines,
`validate_ranking`, nDCG@10 (E3) and the aggregate (E5). The protocol dict is built the
same way as in `starter.py main()`, and the run is cross-checked against a separate
starter CLI run (`outputs/baseline`): equal protocol_hash and byte-equal baseline rows.

Run (from hw-module-2-2/):  python src/run_demo.py
Embeddings are read from kaggle/output (computed on Kaggle; see kaggle/embed_assets.py).
`--fake-embeddings` replaces them with deterministic hash vectors (tests only).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from common import (CANDIDATE_COUNT, DATA, DEMO_CUTOFF, DEMO_USERS, EMBED_DIR, HORIZON_DAYS,
                    ROOT, TEXT_TEMPLATE, asset_text, load_starter)
from embeddings import MODEL_KEYS, load_tables, vectors_for
from fill import (ARMS, PRIMARY, SHORTLIST_ARMS, SPACE_ARMS, DenseSpace, SparseSpace,
                  tfidf_vectors, token_vectors, user_lists)

BASELINES = ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell')


def fake_vectors(assets, dim=64):
    """Deterministic stand-in embeddings keyed by the asset text (tests only)."""
    out = {}
    for isin, row in assets.items():
        seed = int(hashlib.sha256(asset_text(row).encode('utf-8')).hexdigest()[:8], 16)
        out[isin] = np.random.default_rng(seed).standard_normal(dim)
    return out


def starter_protocol(starter, args, users, assets, end):
    """Same construction as starter.py main(), so the hash is comparable."""
    protocol = {'teaching_defaults_not_official_grading_protocol': True,
                'cutoff': args.cutoff, 'end_exclusive': end,
                'users': users, 'cohort_selection': 'warm',
                'eligible_assets': sorted(assets), 'exclude_seen': False,
                'candidate_count': CANDIDATE_COUNT, 'k': 10,
                'empty_truth_policy': 'zero in primary mean; nonempty mean also reported',
                'initial_positions': 'excluded', 'duplicate_policy': 'exact full rows only',
                'profile_use': 'display only in classical baselines',
                'asset_feature_fields': ['assetCategory', 'assetSubCategory', 'sector',
                                         'industry', 'marketID']}
    protocol['source_sha256'] = {
        name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
        for name in ('transactions.csv', 'asset_information.csv',
                     'customer_information.csv', 'close_prices.csv')}
    protocol['protocol_hash'] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


def build(args):
    """Return everything the run produces, without writing files (tests reuse this)."""
    starter = load_starter()
    timings = {}
    started = time.perf_counter()
    assets, profiles, history, truth, end, excluded = starter.load_data(
        args.data_dir, args.cutoff, args.horizon_days)
    users = starter.select_users(profiles, history, 'warm', args.max_users, None)
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    pop = starter.popularity_scores(history, assets)
    contexts, predictions = starter.contexts_for(users, assets, profiles, history, features, pop,
                                                 args.cutoff, CANDIDATE_COUNT, False)
    events = starter.scan_recent_events(args.data_dir, args.cutoff, assets)
    recent = starter.recency_rankings(contexts, pop, events)
    for uid in users:
        predictions[uid].update(recent[uid])
    protocol = starter_protocol(starter, args, users, assets, end)
    timings['shared_data_and_four_baselines_seconds'] = time.perf_counter() - started

    eligible = sorted(assets)
    pop_order = starter.order(pop)
    pop_rank = {isin: r for r, isin in enumerate(pop_order)}
    started = time.perf_counter()
    spaces = {'tokens_fill': SparseSpace(token_vectors(features)),
              'tfidf_fill': SparseSpace(tfidf_vectors({i: asset_text(r) for i, r in assets.items()}))}
    manifest = None
    if args.fake_embeddings:
        for arm in MODEL_KEYS:
            spaces[arm] = DenseSpace(fake_vectors(assets))
    else:
        tables, manifest = load_tables(args.embed_dir)
        for arm, table in tables.items():
            spaces[arm] = DenseSpace(vectors_for(assets, table))
    timings['build_vector_spaces_seconds'] = time.perf_counter() - started

    failures = {arm: [] for arm in ARMS}
    mismatch = []
    started = time.perf_counter()
    for uid in users:
        lists, _ = user_lists(predictions[uid]['latest_buy'], history[uid], eligible,
                              pop_order, pop_rank, {})
        predictions[uid]['pop_fill'] = lists['pop_fill']
        if lists['pop_fill'] != predictions[uid]['latest_buy']:
            mismatch.append(uid)
    timings['scoring_seconds'] = {'pop_fill': time.perf_counter() - started}
    for arm in SPACE_ARMS:
        if arm not in spaces:
            continue
        started = time.perf_counter()
        for uid in users:
            lists, fallbacks = user_lists(predictions[uid]['latest_buy'], history[uid], eligible,
                                          pop_order, pop_rank, {arm: spaces[arm]})
            for name, ranking in lists.items():
                if name != 'pop_fill':
                    predictions[uid][name] = ranking
            for name in fallbacks:
                failures[name].append({'user_id': uid, 'error': 'empty profile; popularity fill used'})
        timings['scoring_seconds'][arm] = time.perf_counter() - started
    for arm in SHORTLIST_ARMS:
        timings['scoring_seconds'][arm] = 'included in the scoring time of its space'

    per_user = []
    for uid in users:
        relevant = truth.get(uid, set()) & set(contexts[uid]['eligible_asset_ids'])
        new = {i for i in relevant if i not in history[uid]}
        free = 10 - min(10, len(history[uid]))
        for method, ranking in predictions[uid].items():
            starter.validate_ranking(ranking, contexts[uid]['eligible_asset_ids'])
            per_user.append({'user_id': uid, 'method': method, 'relevant_count': len(relevant),
                             'new_to_user_count': len(new), 'free_slots': free,
                             'ndcg_at_10': starter.ndcg10(ranking, relevant)})
    methods = list(BASELINES) + [m for m in ARMS if m in predictions[users[0]]]
    metrics = {}
    for m in methods:
        rows = [r for r in per_user if r['method'] == m]
        block = starter.aggregate(rows)
        with_new = [r['ndcg_at_10'] for r in rows if r['new_to_user_count']]
        block['with_new_asset_users'] = len(with_new)
        block['ndcg_at_10_with_new_asset'] = sum(with_new) / len(with_new) if with_new else None
        block['failed_cases'] = len(failures.get(m, []))
        metrics[m] = block
    return {'starter': starter, 'protocol': protocol, 'users': users, 'predictions': predictions,
            'per_user': per_user, 'metrics': metrics, 'methods': methods, 'failures': failures,
            'pop_fill_mismatch_users': mismatch, 'timings': timings, 'excluded': excluded,
            'embedding_manifest': manifest}


def cross_check(result, baseline_dir):
    """Compare with a separate, unchanged starter CLI run on the same settings."""
    if not (baseline_dir / 'protocol.json').exists():
        return {'baseline_run_found': False}
    theirs = json.loads((baseline_dir / 'protocol.json').read_text(encoding='utf-8'))
    with (baseline_dir / 'predictions.csv').open(encoding='utf-8', newline='') as stream:
        starter_rows = [tuple(r) for r in csv.reader(stream)][1:]
    ours = [(m, uid, str(rank), isin) for uid in result['users']
            for m in BASELINES for rank, isin in enumerate(result['predictions'][uid][m], 1)]
    with (baseline_dir / 'per_user.csv').open(encoding='utf-8', newline='') as stream:
        starter_ndcg = {(r['user_id'], r['method']): float(r['ndcg_at_10']) for r in csv.DictReader(stream)}
    ndcg_equal = all(starter_ndcg[(r['user_id'], r['method'])] == r['ndcg_at_10']
                     for r in result['per_user'] if r['method'] in BASELINES)
    return {'baseline_run_found': True,
            'protocol_hash_equal': theirs['protocol_hash'] == result['protocol']['protocol_hash'],
            'baseline_prediction_rows_equal': sorted(starter_rows) == sorted(ours),
            'baseline_rows_compared': len(ours),
            'baseline_per_user_ndcg_equal': ndcg_equal}


def write(result, args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'protocol.json').write_text(json.dumps(result['protocol'], ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    with (out / 'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'user_id', 'rank', 'ISIN'])
        for uid in result['users']:
            for m in result['methods']:
                writer.writerows((m, uid, rank, isin)
                                 for rank, isin in enumerate(result['predictions'][uid][m], 1))
    with (out / 'per_user.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['per_user'][0]))
        writer.writeheader()
        writer.writerows(result['per_user'])
    check = cross_check(result, args.baseline_dir)
    metrics = {'protocol_hash': result['protocol']['protocol_hash'], 'primary_method': PRIMARY,
               'methods': result['metrics'],
               'failures': {k: v for k, v in result['failures'].items() if v},
               'pop_fill_equals_latest_buy_mismatched_users': len(result['pop_fill_mismatch_users']),
               'cross_check_with_starter_run': check, 'excluded_rows': result['excluded'],
               'timings_local': result['timings'],
               'note': 'Offline purchase relevance, not profitability or asset suitability.'}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    manifest = {'protocol': 'starter defaults: cutoff 2021-06-01, 30 days, warm, 100 users, '
                            'repeats allowed, candidate_count 50 (unused by the fill arms)',
                'candidate_policy': 'prefix = latest_buy recency prefix; fill candidates = all '
                                    'eligible assets not in the prefix; ties by popularity rank, ISIN',
                'invalid_policy': 'missing/non-finite embedding aborts the run; an empty user profile '
                                  'falls back to the popularity fill and is counted as a failed case',
                'text_template': TEXT_TEMPLATE, 'fake_embeddings': args.fake_embeddings,
                'local_environment': {'python': platform.python_version(),
                                      'numpy': np.__version__, 'platform': platform.platform(),
                                      'processor': platform.processor()},
                'embedding_manifest': {k: v for k, v in (result['embedding_manifest'] or {}).items()
                                       if k != 'texts_sha256'}}
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'demo')
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'outputs' / 'baseline')
    parser.add_argument('--embed-dir', type=Path, default=EMBED_DIR)
    parser.add_argument('--cutoff', default=DEMO_CUTOFF)
    parser.add_argument('--horizon-days', type=int, default=HORIZON_DAYS)
    parser.add_argument('--max-users', type=int, default=DEMO_USERS)
    parser.add_argument('--fake-embeddings', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    result = build(args)
    check = write(result, args)
    total = time.perf_counter() - started
    print(f"{'method':<20}{'users':>6}{'nonempty':>9}{'with-new':>9}{'all':>9}{'nonempty':>10}"
          f"{'with-new':>10}{'failed':>7}")
    for m, b in result['metrics'].items():
        fmt = lambda v: f'{v:.4f}' if v is not None else 'n/a'
        print(f"{m:<20}{b['users']:>6}{b['nonempty_truth_users']:>9}{b['with_new_asset_users']:>9}"
              f"{fmt(b['ndcg_at_10']):>9}{fmt(b['ndcg_at_10_nonempty']):>10}"
              f"{fmt(b['ndcg_at_10_with_new_asset']):>10}{b['failed_cases']:>7}")
    print('protocol_hash', result['protocol']['protocol_hash'][:12], '| cross-check', check)
    print('pop_fill != latest_buy users:', len(result['pop_fill_mismatch_users']),
          f'| whole local run {total:.1f} s')


if __name__ == '__main__':
    main()
