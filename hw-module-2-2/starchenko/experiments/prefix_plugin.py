"""Starter plugin for the prefix reranking experiment (Session 2, Ranker track).

Every arm keeps the latest_buy structure: the customer's own past assets first, then the
same popularity fill of never-bought assets. Only the order of the past assets changes.

Environment:
  PREFIX_ARM  recency | lr | input_order | llm   (recency must reproduce latest_buy exactly;
              input_order is the prompt's listing order, by ISIN, without a model)
  PREFIX_DIR  folder with prefix_inputs.jsonl, prompts.jsonl, lr_orders.jsonl and
              raw_outputs.jsonl (raw Qwen3-1.7B text from run_prefix_llm.py)
The starter must be run with --candidate-count equal to the number of eligible assets,
because the popularity fill is not inside the default 50 CBF candidates.

LLM output policy (strict): the first JSON array in the raw text must be a permutation of
the shown aliases. Anything else (no array, bad JSON, unknown, missing or duplicate ids,
truncated output) is invalid; the user then keeps the recency prefix and is logged.
Plugins receive only past context; this module reads no labels.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_parse import parse_llm  # noqa: E402

ARM = os.environ.get('PREFIX_ARM', 'llm')
DIR = Path(os.environ.get('PREFIX_DIR', Path(__file__).resolve().parent.parent / 'outputs' / 'inputs'))
LOG = DIR / f'plugin_log_{ARM}.jsonl'


def _jsonl(name):
    path = DIR / name
    if not path.exists():
        return {}
    return {r['user_id']: r for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())
            if r}


INPUTS = _jsonl('prefix_inputs.jsonl')
PROMPTS = _jsonl('prompts.jsonl')
LR = _jsonl('lr_orders.jsonl')
RAW = _jsonl('raw_outputs.jsonl') if ARM == 'llm' else {}
_LOG_STARTED = False  # the log is reset on the first rerun() call, never on import


def prefix_order(uid):
    prefix = INPUTS[uid]['prefix']
    if ARM == 'recency' or len(prefix) < 2:
        return prefix, 'no_call_single_asset' if len(prefix) < 2 else 'recency'
    if ARM == 'lr':
        return LR[uid]['lr_prefix'], 'lr'
    if ARM == 'input_order':
        # Control without a model: the order in which the prompt lists the assets (by ISIN).
        shown = sorted(prefix[:30])
        return shown + [i for i in prefix if i not in set(shown)], 'input_order'
    prompt = PROMPTS[uid]
    if prompt['protocol_hash'] != RAW[uid]['protocol_hash']:
        raise ValueError('raw output from another protocol')
    ordered, status = parse_llm(RAW[uid]['raw_text'], prompt['alias_to_isin'])
    if RAW[uid].get('finish_reason') == 'length' and ordered is None:
        status = 'truncated_' + status
    if ordered is None:
        return prefix, 'fallback_' + status
    shown = set(ordered)
    return ordered + [i for i in prefix if i not in shown], 'ok'


def rerank(context, candidate_ids):
    global _LOG_STARTED
    uid = context['user_id']
    order, status = prefix_order(uid)
    ranking = (order + INPUTS[uid]['fill'])[:10]
    with LOG.open('a' if _LOG_STARTED else 'w', encoding='utf-8') as f:
        _LOG_STARTED = True
        f.write(json.dumps({'user_id': uid, 'arm': ARM, 'status': status,
                            'n_past_assets': len(INPUTS[uid]['prefix']),
                            'ranking': ranking}) + '\n')
    if not set(ranking) <= set(candidate_ids):
        raise ValueError('ranking outside the candidate pool; use the full candidate count')
    return ranking
