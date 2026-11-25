"""Prefix + fill: keep the `latest_buy` recency prefix, change only how free slots are filled.

`latest_buy` (provided baseline) = the user's past-bought eligible assets, most recent
first, then popularity. Its first min(10, |history|) items are the *prefix*; the rest
is the *fill*. Every arm here keeps that prefix and differs only in the fill rule:

  pop_fill     popularity order (control; must reproduce `latest_buy` exactly)
  tokens_fill  cosine in the starter's categorical token space (its CBF features)
  tfidf_fill   cosine over unigram TF-IDF of the as-of asset text
  e5_fill      cosine over multilingual-e5-base embeddings of the same text (non-LLM encoder)
  llm_fill     cosine over Qwen3-Embedding-0.6B embeddings of the same text  <- my method
  *_top20      the 20 most popular unseen assets re-ordered by that space's cosine
  oracle_fill  the user's new-to-user label assets (ceiling; reads labels, not a method)

A user profile is the buy-count-weighted sum of the history assets' vectors (the same
weighting as the starter's CBF, E1); candidates are eligible assets not in the prefix;
ties break by popularity rank, then ISIN.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

WORD = re.compile(r'[^\W_]+', re.UNICODE)
K = 10


# ----------------------------------------------------------------------------- spaces
def tfidf_vectors(texts):
    """Unigram TF-IDF, smooth idf, L2-normalised; {isin: text} -> {isin: {token: w}}."""
    docs = {i: Counter(t for t in WORD.findall(s.casefold()) if len(t) > 1)
            for i, s in texts.items()}
    df = Counter(t for d in docs.values() for t in d)
    n = len(docs)
    out = {}
    for isin, tf in docs.items():
        vec = {t: c * (math.log((1 + n) / (1 + df[t])) + 1) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        out[isin] = {t: v / norm for t, v in vec.items()}
    return out


def token_vectors(features):
    """The starter's E1 item vectors: binary categorical tokens, L2-normalised."""
    out = {}
    for isin, values in features.items():
        unique = sorted(set(values))          # sorted: iteration order must not follow the hash seed
        out[isin] = {v: 1 / math.sqrt(len(unique)) for v in unique} if unique else {}
    return out


def _check_weight(weight):
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0):
        raise ValueError('History weights must be finite and nonnegative')


class SparseSpace:
    def __init__(self, vectors):
        self.vectors = vectors
        self.inverted = defaultdict(list)
        for isin, vec in vectors.items():
            for token, value in vec.items():
                self.inverted[token].append((isin, value))

    def scores(self, history_counts):
        """Cosine of every asset with the normalised count-weighted history profile."""
        profile = defaultdict(float)
        for isin, weight in history_counts.items():
            _check_weight(weight)
            for token, value in self.vectors.get(isin, {}).items():
                profile[token] += weight * value
        norm = math.sqrt(sum(v * v for v in profile.values()))
        scores = defaultdict(float)
        if norm:
            for token, pv in profile.items():
                for isin, value in self.inverted[token]:
                    scores[isin] += pv * value / norm
        return scores


class DenseSpace:
    """Rows are L2-normalised embeddings; computed in float64 for a stable order."""

    def __init__(self, vectors):
        self.ids = sorted(vectors)
        self.index = {isin: r for r, isin in enumerate(self.ids)}
        matrix = np.asarray([vectors[i] for i in self.ids], dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError('Non-finite embedding')
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError('Zero embedding')
        self.matrix = matrix / norms

    def scores(self, history_counts):
        profile = np.zeros(self.matrix.shape[1])
        for isin, weight in history_counts.items():
            _check_weight(weight)
            row = self.index.get(isin)
            if row is not None:
                profile += weight * self.matrix[row]
        norm = np.linalg.norm(profile)
        if not norm:
            return {}
        values = self.matrix @ (profile / norm)
        return dict(zip(self.ids, values.tolist()))


# ----------------------------------------------------------------------------- fills
def prefix_of(latest_buy_list, history_counts):
    """The recency prefix of the provided `latest_buy` list."""
    return list(latest_buy_list[:min(K, len(history_counts))])


def popularity_fill(prefix, pop_order):
    taken = set(prefix)
    return prefix + [i for i in pop_order if i not in taken][:K - len(prefix)]


def _key(scores, pop_rank):
    """Sort key: score rounded to 1e-12, then popularity rank (which already breaks by ISIN).

    Rounding makes mathematically equal scores tie exactly. Without it the last bits of a
    float sum depend on set/dict iteration order, which for strings follows Python's
    per-process hash seed, so ties flipped between runs (found against the Session 1 pilot).
    """
    return lambda i: (-round(scores.get(i, 0.0), 12), pop_rank[i])


def score_fill(prefix, eligible, scores, pop_rank):
    free = K - len(prefix)
    if free <= 0:
        return prefix[:K]
    taken = set(prefix)
    cands = sorted((i for i in eligible if i not in taken), key=_key(scores, pop_rank))
    return prefix + cands[:free]


def shortlist_fill(prefix, pop_order, scores, pop_rank, n):
    """Popularity prior (top-n unseen) with a personalised order inside it."""
    free = K - len(prefix)
    if free <= 0:
        return prefix[:K]
    taken = set(prefix)
    unseen = [i for i in pop_order if i not in taken]
    short = sorted(unseen[:max(n, free)], key=_key(scores, pop_rank))
    return prefix + short[:free]


def oracle_fill(prefix, new_label_assets, pop_order):
    """Ceiling: the user's new-to-user label assets fill the free slots. Reads labels."""
    chosen = prefix + sorted(new_label_assets)[:K - len(prefix)]
    taken = set(chosen)
    return chosen + [i for i in pop_order if i not in taken][:K - len(chosen)]


SPACE_ARMS = ('tokens_fill', 'tfidf_fill', 'e5_fill', 'llm_fill')
SHORTLIST_ARMS = {'tfidf_top20': ('tfidf_fill', 20), 'llm_top20': ('llm_fill', 20)}
ARMS = ('pop_fill',) + SPACE_ARMS + tuple(SHORTLIST_ARMS)
PRIMARY = 'llm_fill'


def user_lists(latest_buy_list, history_counts, eligible, pop_order, pop_rank, spaces,
               new_label_assets=None):
    """All arms for one user. `spaces` maps a SPACE_ARMS name to a Sparse/DenseSpace.

    Returns (lists, fallbacks): fallbacks names the arms whose profile was empty and
    that therefore used the popularity fill (recorded as a failed case, never dropped).
    """
    prefix = prefix_of(latest_buy_list, history_counts)
    lists, fallbacks, cache = {'pop_fill': popularity_fill(prefix, pop_order)}, [], {}
    free = K - len(prefix)
    for arm in SPACE_ARMS:
        if arm not in spaces:
            continue
        scores = spaces[arm].scores(history_counts) if free else {}
        if free and not scores:
            fallbacks.append(arm)
        cache[arm] = scores
        lists[arm] = score_fill(prefix, eligible, scores, pop_rank)
    for arm, (space, n) in SHORTLIST_ARMS.items():
        if space in cache:
            lists[arm] = shortlist_fill(prefix, pop_order, cache[space], pop_rank, n)
    if new_label_assets is not None:
        lists['oracle_fill'] = oracle_fill(prefix, new_label_assets, pop_order)
    for name, ranking in lists.items():
        if len(ranking) != K or len(set(ranking)) != K or ranking[:len(prefix)] != prefix:
            raise AssertionError(f'{name}: malformed list')
    return lists, fallbacks
