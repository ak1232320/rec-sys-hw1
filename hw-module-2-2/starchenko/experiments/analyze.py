#!/usr/bin/env python3
"""Compare the prefix arms user by user and describe the LLM outputs.

Usage: python analyze.py OUTPUTS_DIR
Reads OUTPUTS_DIR/{change_lr,change_llm}/per_user.csv, OUTPUTS_DIR/inputs/* and the
LLM run metadata; writes OUTPUTS_DIR/analysis.json. Labels are read here only from the
starter's per_user.csv (the evaluation output), never by the plugin or prompt builder.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_parse import parse_llm  # noqa: E402

OUT = Path(sys.argv[1])
INP = OUT / 'inputs'


def jl(path):
    return {r['user_id']: r for r in map(json.loads, path.read_text(encoding='utf-8').splitlines()) if r}


def per_user(run):
    res = {}
    with (OUT / run / 'per_user.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            res.setdefault(r['user_id'], {})[r['method']] = float(r['ndcg_at_10'])
            res[r['user_id']]['_rel'] = int(r['relevant_count'])
    return res


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    inputs, prompts, raw = jl(INP / 'prefix_inputs.jsonl'), jl(INP / 'prompts.jsonl'), jl(INP / 'raw_outputs.jsonl')
    lr, log = jl(INP / 'lr_orders.jsonl'), jl(INP / 'plugin_log_llm.jsonl')
    pu_llm, pu_lr, pu_in = per_user('change_llm'), per_user('change_lr'), per_user('change_input_order')
    statuses = Counter(r['status'] for r in log.values())
    agree = Counter()
    for uid, p in prompts.items():
        order, status = parse_llm(raw[uid]['raw_text'], p['alias_to_isin'])
        if order is None:
            continue
        shown = sorted(p['alias_to_isin'].values())
        rec = [i for i in inputs[uid]['prefix'] if i in set(shown)]
        agree['valid'] += 1
        agree['equals_input_order'] += order == shown
        agree['equals_recency_order'] += order == rec
        agree['equals_lr_order'] += order == [i for i in lr[uid]['lr_prefix'] if i in set(shown)]
        agree['same_first_as_recency'] += order[0] == rec[0]
        agree['top10_differs_from_latest_buy'] += (log[uid]['ranking'] !=
                                                   (inputs[uid]['prefix'] + inputs[uid]['fill'])[:10])
    labelled = []
    for uid, r in pu_llm.items():
        if r['_rel']:
            labelled.append({'user_id': uid[:8] + '...', 'relevant': r['_rel'],
                             'n_past_assets': inputs[uid]['n_past_assets'],
                             'llm_status': log[uid]['status'],
                             'latest_buy': round(r['latest_buy'], 4), 'lr': round(pu_lr[uid]['custom'], 4),
                             'input_order': round(pu_in[uid]['custom'], 4),
                             'llm': round(r['custom'], 4)})
    wins = Counter()
    for uid in pu_llm:
        a, b, c = pu_llm[uid]['latest_buy'], pu_lr[uid]['custom'], pu_llm[uid]['custom']
        d = pu_in[uid]['custom']
        wins['llm>input_order'] += c > d + 1e-12
        wins['llm<input_order'] += c < d - 1e-12
        wins['input_order>latest'] += d > a + 1e-12
        wins['input_order<latest'] += d < a - 1e-12
        wins['llm>latest'] += c > a + 1e-12
        wins['llm<latest'] += c < a - 1e-12
        wins['llm>lr'] += c > b + 1e-12
        wins['llm<lr'] += c < b - 1e-12
        wins['lr>latest'] += b > a + 1e-12
        wins['lr<latest'] += b < a - 1e-12
    toks = [r['output_tokens'] for r in raw.values()]
    res = {'llm_status_counts_all_100_users': dict(statuses),
           'llm_valid_output_agreement': dict(agree),
           'paired_counts_100_users': dict(wins),
           'labelled_users': sorted(labelled, key=lambda r: -r['n_past_assets']),
           'finish_reason': dict(Counter(r['finish_reason'] for r in raw.values())),
           'output_tokens_max': max(toks), 'output_tokens_mean': round(sum(toks) / len(toks), 1)}
    (OUT / 'analysis.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
