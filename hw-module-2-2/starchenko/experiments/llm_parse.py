"""Strict parser for the LLM prefix order (no side effects, shared by plugin and analysis)."""
import json
import re


def parse_llm(raw_text, alias_to_isin):
    """Return (ordered ISINs or None, status).

    The first JSON array in the text must be a permutation of the shown aliases.
    """
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.S).strip()
    match = re.search(r'\[.*?\]', text, flags=re.S)
    if not match:
        return None, 'no_json_array'
    try:
        ids = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 'invalid_json'
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        return None, 'non_string_ids'
    if len(set(ids)) != len(ids):
        return None, 'duplicate_ids'
    if set(ids) - set(alias_to_isin):
        return None, 'unknown_ids'
    if set(alias_to_isin) - set(ids):
        return None, 'missing_ids'
    return [alias_to_isin[a] for a in ids], 'ok'
