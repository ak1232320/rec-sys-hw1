"""Load the embeddings computed on Kaggle and map them to assets of a given cutoff.

The notebook writes one `<model_key>.npy` matrix per model, row-aligned with
`manifest.json["texts_sha256"]`. Loading re-checks that alignment against the exported
texts, that every value is finite, and that every eligible asset's as-of text has a row;
a missing or broken vector is an error, never a silent zero.
"""
from __future__ import annotations

import json

import numpy as np

from common import EMBED_DIR, TEXTS_FILE, asset_text, text_sha

MODEL_KEYS = {'llm_fill': 'qwen3_embedding_0.6b', 'e5_fill': 'multilingual_e5_base'}


def load_manifest(embed_dir=EMBED_DIR):
    return json.loads((embed_dir / 'manifest.json').read_text(encoding='utf-8'))


def load_tables(embed_dir=EMBED_DIR, texts_file=TEXTS_FILE):
    """Return {arm: {text_sha256: vector}} for the arms whose matrix is present."""
    manifest = load_manifest(embed_dir)
    exported = [t['sha256'] for t in json.loads(texts_file.read_text(encoding='utf-8'))['texts']]
    if manifest['texts_sha256'] != exported:
        raise ValueError('Embedding rows are not aligned with the exported asset texts')
    tables = {}
    for arm, key in MODEL_KEYS.items():
        path = embed_dir / f'{key}.npy'
        if not path.exists():
            continue
        matrix = np.load(path)
        if matrix.shape[0] != len(exported) or not np.isfinite(matrix).all():
            raise ValueError(f'{key}: wrong shape or non-finite values')
        tables[arm] = dict(zip(exported, matrix))
    return tables, manifest


def vectors_for(assets, table):
    """{isin: vector} for every asset of one cutoff, looked up by its as-of text."""
    out, missing = {}, []
    for isin, row in assets.items():
        vector = table.get(text_sha(asset_text(row)))
        if vector is None:
            missing.append(isin)
        else:
            out[isin] = vector
    if missing:
        raise ValueError(f'{len(missing)} eligible assets have no embedding, e.g. {missing[:3]}')
    return out
