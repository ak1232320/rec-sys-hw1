"""Shared pieces: paths, the unchanged course starter, and the as-of asset text.

The starter is imported as a module (exactly as the course's far-trans-llm adapter
does) so that data loading, eligibility, user selection, the four baselines and
nDCG@10 are the provided implementations, not re-implementations.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # hw-module-2-2/
DATA = ROOT.parent / 'hw-module-2-1' / 'data_raw' / 'FAR-Trans'
STARTER_PATH = ROOT / 'starter' / 'starter.py'
KAGGLE_DIR = ROOT / 'kaggle'
TEXTS_FILE = KAGGLE_DIR / 'input' / 'asset_texts.json'
EMBED_DIR = KAGGLE_DIR / 'output'

# HW2 teaching protocol (starter defaults) and the Session 1 development cutoffs.
DEMO_CUTOFF = '2021-06-01'
HORIZON_DAYS = 30
DEMO_USERS = 100
CANDIDATE_COUNT = 50
DEV_CUTOFFS = ('2020-09-01', '2020-12-01', '2021-03-01', '2021-06-01', '2021-09-01',
               '2021-12-01', '2022-03-01', '2022-06-01', '2022-09-01')

# The text every text-based fill reads (TF-IDF and both embedding models): the as-of
# name plus categorical attributes, missing fields omitted. Identical to the Session 1
# pilot (hw-module-2-1/experiments/pilot_control_fill.py), so only the vector space changes.
TEXT_FIELDS = ('assetName', 'assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID')
TEXT_TEMPLATE = ' | '.join(TEXT_FIELDS) + '  (empty fields omitted)'


def load_starter():
    spec = importlib.util.spec_from_file_location('provided_far_trans_starter', STARTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asset_text(row):
    return ' | '.join(row[f].strip() for f in TEXT_FIELDS if row.get(f, '').strip())


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
