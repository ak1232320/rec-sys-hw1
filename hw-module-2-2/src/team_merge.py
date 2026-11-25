"""Team merge (Session 2): Starchenko's prefix arms x my fill arms, without rerunning his model.

Why the combination is exact. Every arm of both solutions keeps the `latest_buy` layout:
positions 1..p hold the user's own past assets (p = min(10, |history|)), positions p+1..10
the fill. A prefix arm (his) only reorders the past assets, or re-chooses ten of them when
|history| > 10 (then there is no fill). A fill arm (mine) only changes positions p+1..10
and sees the prefix only as a set, which every prefix arm keeps when p < 10. Binary DCG
is a sum over positions, so for every user

    nDCG(prefix A + fill B) = nDCG(A + pop) + nDCG(recency + B) - nDCG(recency + pop),

and, means being linear, the same holds for any mean over the same users.

Checks (the script stops if one fails):
 1. My rebuild of his model-free prompt-order control (past assets in ISIN order, then the
    popularity fill) gives his per-user nDCG@10 on the 11 labelled users and his mean:
    both codebases see the same users, labels and histories.
 2. The identity on real lists: prompt-order prefix x each of my fills, built as lists and
    scored by the starter, equals the formula for all 100 users.
 3. Development cutoffs: his Session 1 `repeat_first` arm equals my `pop_fill` per cutoff
    (labelled count and nDCG), so his LR-prefix deltas and my fill deltas are measured on
    the same user-windows.

Inputs: his committed results at commit bb64f11 (read with `git show`), his Session 1 pilot
`../hw-module-2-1/experiments/starchenko/pilot_prefix_lr.json`, my `outputs/dev/dev_metrics.json`
and a fresh build of my demo run. Output: `outputs/team/merge.json`.
Run (from hw-module-2-2/):  python src/team_merge.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from common import DATA, DEMO_CUTOFF, EMBED_DIR, HORIZON_DAYS, DEMO_USERS, ROOT
import run_demo

STARCHENKO_COMMIT = 'bb64f11'
STARCHENKO_RESULTS = 'hw-module-2-2/starchenko/results'
PILOT = ROOT.parent / 'hw-module-2-1' / 'experiments' / 'starchenko' / 'pilot_prefix_lr.json'
OUT = ROOT / 'outputs' / 'team' / 'merge.json'
FILLS_DEMO = ('pop_fill', 'llm_fill', 'e5_fill', 'tfidf_top20')
FILLS_DEV = ('pop_fill', 'tfidf_top20', 'e5_fill', 'llm_fill')
PREFIXES_DEMO = {'recency': 'latest_buy', 'lr': 'change_lr', 'input_order': 'change_input_order',
                 'llm': 'change_llm'}
TOL = 1e-12


def git_json(path):
    text = subprocess.run(['git', 'show', f'{STARCHENKO_COMMIT}:{path}'], cwd=ROOT.parent,
                          check=True, capture_output=True, text=True, encoding='utf-8').stdout
    return json.loads(text.lstrip('﻿'))


def require(ok, message):
    if not ok:
        raise SystemExit(f'CHECK FAILED: {message}')


def demo_merge(result):
    starter = result['starter']
    users = result['users']
    preds, hist, rel = result['predictions'], result['history'], result['relevant']
    ndcg = {(uid, m): starter.ndcg10(preds[uid][m], rel[uid]) for uid in users for m in FILLS_DEMO}

    # Check 1: rebuild his prompt-order control and compare with his committed numbers.
    analysis = git_json(f'{STARCHENKO_RESULTS}/analysis.json')
    theirs = {r['user_id'].rstrip('.'): r for r in analysis['labelled_users']}
    input_order = {}
    for uid in users:
        p = min(10, len(hist[uid]))
        prefix = sorted(hist[uid])[:p]
        input_order[uid] = prefix + preds[uid]['pop_fill'][p:]
        require(len(set(input_order[uid])) == 10, f'{uid}: prompt-order list')
    mine_io = {uid: starter.ndcg10(input_order[uid], rel[uid]) for uid in users}
    matched = []
    for short, row in theirs.items():
        uid = next(u for u in users if u.startswith(short))
        matched.append(uid)
        require(len(hist[uid]) == row['n_past_assets'], f'{short}: past-asset count')
        require(round(ndcg[(uid, 'pop_fill')], 4) == row['latest_buy'], f'{short}: latest_buy nDCG')
        require(round(mine_io[uid], 4) == row['input_order'], f'{short}: prompt-order nDCG')
    labelled = [u for u in users if rel[u]]
    require(sorted(matched) == sorted(labelled), 'labelled users differ')
    metrics = {arm: git_json(f'{STARCHENKO_RESULTS}/{folder}/metrics.json')['methods']
               for arm, folder in PREFIXES_DEMO.items() if arm != 'recency'}
    base = git_json(f'{STARCHENKO_RESULTS}/baseline/metrics.json')['methods']['latest_buy']
    his = {'recency': base, **{arm: m['custom'] for arm, m in metrics.items()}}
    require(abs(sum(mine_io.values()) / len(users) - his['input_order']['ndcg_at_10']) < TOL,
            'prompt-order mean')
    require(abs(result['metrics']['latest_buy']['ndcg_at_10'] - base['ndcg_at_10']) < TOL,
            'latest_buy mean')

    # Check 2: the identity on real lists (prompt-order prefix x my fills).
    worst = 0.0
    for uid in users:
        p = min(10, len(hist[uid]))
        for fill in FILLS_DEMO:
            combined = input_order[uid][:p] + preds[uid][fill][p:]
            require(len(set(combined)) == 10, f'{uid}: combined list')
            direct = starter.ndcg10(combined, rel[uid])
            formula = mine_io[uid] + ndcg[(uid, fill)] - ndcg[(uid, 'pop_fill')]
            worst = max(worst, abs(direct - formula))
    require(worst < TOL, f'additivity on real lists (max diff {worst})')

    # The 100-user grid: his exact means + my exact fill deltas.
    n, n_lab = len(users), len(labelled)
    delta = {f: sum(ndcg[(u, f)] - ndcg[(u, 'pop_fill')] for u in users) for f in FILLS_DEMO}
    grid = {}
    for arm, block in his.items():
        grid[arm] = {f: {'ndcg_at_10': block['ndcg_at_10'] + delta[f] / n,
                         'ndcg_at_10_nonempty': block['ndcg_at_10_nonempty'] + delta[f] / n_lab}
                     for f in FILLS_DEMO}
    movers = {f: {'better': sum(ndcg[(u, f)] > ndcg[(u, 'pop_fill')] + TOL for u in users),
                  'worse': sum(ndcg[(u, f)] < ndcg[(u, 'pop_fill')] - TOL for u in users)}
              for f in FILLS_DEMO}
    return {'users': n, 'labelled_users': n_lab,
            'starchenko_protocol_hash': git_json(f'{STARCHENKO_RESULTS}/change_lr/metrics.json')['protocol_hash'],
            'my_protocol_hash': result['protocol']['protocol_hash'],
            'checks': {'labelled_users_match': True, 'past_asset_counts_match': True,
                       'latest_buy_per_user_match': True,
                       'prompt_order_control_rebuilt_per_user_match': True,
                       'prompt_order_control_mean_abs_diff': abs(
                           sum(mine_io.values()) / n - his['input_order']['ndcg_at_10']),
                       'additivity_max_abs_diff_real_lists': worst},
            'fill_movers_vs_pop': movers, 'grid': grid,
            'llm_prefix_valid_answers': analysis['llm_valid_output_agreement']['valid'],
            'llm_prefix_echo_input_order': analysis['llm_valid_output_agreement']['equals_input_order'],
            'llm_prefix_invalid': analysis['llm_status_counts_all_100_users']['fallback_missing_ids'],
            'llm_prefix_calls': (analysis['llm_status_counts_all_100_users']['ok']
                                 + analysis['llm_status_counts_all_100_users']['fallback_missing_ids'])}


def dev_merge():
    dev = json.loads((ROOT / 'outputs' / 'dev' / 'dev_metrics.json').read_text(encoding='utf-8'))
    pilot = json.loads(PILOT.read_text(encoding='utf-8'))
    # Check 3: his repeat_first == my pop_fill on every cutoff.
    require([b['cutoff'] for b in pilot['per_cutoff']] == dev['cutoffs'], 'cutoffs differ')
    for mine, his in zip(dev['per_cutoff'], pilot['per_cutoff']):
        require(mine['labelled_users'] == his['labelled'], f"{mine['cutoff']}: labelled count")
        require(abs(mine['ndcg_labelled']['pop_fill'] - his['ndcg_labelled']['repeat_first']) < TOL,
                f"{mine['cutoff']}: base nDCG")
    pooled = {}
    for group, count_key, value_key in (('labelled', 'labelled_users', 'ndcg_labelled'),
                                        ('with_new', 'labelled_with_new_asset', 'ndcg_with_new_asset')):
        total = sum(b[count_key] for b in dev['per_cutoff'])
        pooled[group] = {'n': total, **{f: sum(b[count_key] * b[value_key][f] for b in dev['per_cutoff'])
                                        / total for f in FILLS_DEV}}
    his = {'labelled': pilot['pooled_all_labelled'], 'with_new': pilot['pooled_with_new']}
    for group in pooled:
        require(his[group]['n'] == pooled[group]['n'], f'{group}: pooled count')
        require(abs(his[group]['repeat_first'] - pooled[group]['pop_fill']) < TOL, f'{group}: pooled base')
    grid = {prefix: {group: {f: his[group][key] + pooled[group][f] - pooled[group]['pop_fill']
                             for f in FILLS_DEV} for group in pooled}
            for prefix, key in (('recency', 'repeat_first'), ('lr', 'lr_prefix'))}
    return {'cutoffs': dev['cutoffs'], 'labelled_user_windows': pooled['labelled']['n'],
            'with_new_user_windows': pooled['with_new']['n'],
            'lr_prefix_paired': {g: {'better': his[g]['better'], 'worse': his[g]['worse']} for g in his},
            'lr_train_cutoffs': '2019-03-01 ... 2020-06-01 (labels end by 2020-07-01)',
            'checks': {'repeat_first_equals_pop_fill_every_cutoff': True,
                       'pooled_counts_and_base_match': True},
            'grid_pooled': grid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    args = argparse.Namespace(data_dir=DATA, output_dir=None, baseline_dir=None, embed_dir=EMBED_DIR,
                              cutoff=DEMO_CUTOFF, horizon_days=HORIZON_DAYS, max_users=DEMO_USERS,
                              fake_embeddings=False)
    merged = {'starchenko_commit': STARCHENKO_COMMIT,
              'identity': 'nDCG(A+B) = nDCG(A+pop) + nDCG(recency+B) - nDCG(recency+pop)',
              'demo_100': demo_merge(run_demo.build(args)), 'dev_9_cutoffs': dev_merge()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(merged, indent=2) + '\n', encoding='utf-8')
    d, v = merged['demo_100'], merged['dev_9_cutoffs']
    print('100 users, nDCG@10 all users (nonempty):')
    for arm, row in d['grid'].items():
        print(f'  {arm:<12}' + ''.join(f"  {f}={row[f]['ndcg_at_10']:.4f} ({row[f]['ndcg_at_10_nonempty']:.4f})"
                                         for f in FILLS_DEMO))
    print('9 cutoffs, pooled labelled | with-new:')
    for arm, row in v['grid_pooled'].items():
        print(f'  {arm:<12}' + ''.join(f"  {f}={row['labelled'][f]:.4f}|{row['with_new'][f]:.4f}"
                                         for f in FILLS_DEV))
    print('checks', d['checks'], v['checks'], f'-> {OUT}')


if __name__ == '__main__':
    main()
