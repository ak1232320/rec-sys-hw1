"""Offline evaluation for A03: user-based vs item-based CF, missing-value
strategies, and the cost of sparsity.

Reads ../u.item and ../u.data (MovieLens 100k), reproduces exactly the scoring
rules implemented in ../cf.js, and writes results.json.

Run:  python experiments/run_experiment.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
OUT_JSON = HERE / "results.json"

TEST_FRACTION = 0.20      # most recent share of each user's ratings, held out
LIKE_THRESHOLD = 4.0      # a held-out rating of 4-5 counts as "relevant"
TOP_K = 5
NEIGHBOURS = 20           # N most similar users for user-based CF
GAMMA = 25                # significance-weighting constant of the `weighted` strategy
STRATEGIES = ("corated", "weighted", "mean")
HEAD_FRACTION = 0.20      # top 20% of items by rating count = the head


# --------------------------------------------------------------------------- data

def load():
    titles = {}
    for line in (DATA / "u.item").read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        f = line.split("|")
        titles[int(f[0])] = f[1]
    rows = [l.split("\t") for l in (DATA / "u.data").read_text(encoding="ascii").splitlines()
            if l.strip()]
    ratings = np.array([[int(r[0]), int(r[1]), float(r[2]), int(r[3])] for r in rows])
    return titles, ratings


def temporal_split(ratings, n_users, n_items):
    """Per user, the most recent TEST_FRACTION of their ratings becomes the test set."""
    train = np.zeros((n_users, n_items), dtype=np.float32)
    test_pairs = []
    order = np.lexsort((ratings[:, 3], ratings[:, 0]))       # by user, then timestamp
    ratings = ratings[order]
    starts = np.searchsorted(ratings[:, 0], np.arange(1, n_users + 2))
    for u in range(1, n_users + 1):
        block = ratings[starts[u - 1]:starts[u]]
        if len(block) == 0:
            continue
        n_test = int(round(TEST_FRACTION * len(block)))
        n_test = min(n_test, len(block) - 1)                  # always leave a profile behind
        cut = len(block) - n_test
        for _, item, rating, _ts in block[:cut]:
            train[u - 1, int(item) - 1] = rating
        for _, item, rating, _ts in block[cut:]:
            test_pairs.append((u - 1, int(item) - 1, rating))
    return train, np.array(test_pairs)


# --------------------------------------------------------------------- similarity

def similarity_matrix(R, strategy, gamma=GAMMA):
    """Pairwise cosine between the ROWS of R, under one missing-value strategy.

    Mirrors cosineSimilarity() in cf.js: `corated` and `weighted` sum only over
    entries both rows rated (a missing entry contributes to neither the dot
    product nor the norms); `mean` replaces every missing entry with the row's
    own mean and takes the cosine over the full vector.
    """
    B = (R > 0).astype(np.float32)

    if strategy == "mean":
        counts = B.sum(axis=1)
        means = np.divide(R.sum(axis=1), np.maximum(counts, 1), dtype=np.float32)
        M = R + (1.0 - B) * means[:, None]
        norms = np.linalg.norm(M, axis=1)
        safe = np.where(norms == 0, 1.0, norms)
        S = (M @ M.T) / np.outer(safe, safe)
        S[norms == 0, :] = 0.0
        S[:, norms == 0] = 0.0
    else:
        dot = R @ R.T
        sq = (R * R) @ B.T                  # sq[a, b] = sum of a's squares over co-rated
        denom = np.sqrt(sq * sq.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.where(denom > 0, dot / np.where(denom == 0, 1, denom), 0.0)
        if strategy == "weighted":
            n_co = B @ B.T
            S = S * (np.minimum(n_co, gamma) / gamma)

    np.fill_diagonal(S, 0.0)
    return S.astype(np.float32)


def co_rated_counts(R):
    B = (R > 0).astype(np.float32)
    return (B @ B.T).astype(np.int32)


# ------------------------------------------------------------------ predictors

def user_based_scores(train, S, u, neighbours=NEIGHBOURS):
    """Similarity-weighted average over the N most similar users with sim > 0.

    Returns (predicted score per item, how many neighbours rated it).
    """
    sims = S[u]
    positive = np.flatnonzero(sims > 0)
    if positive.size == 0:
        n_items = train.shape[1]
        return np.zeros(n_items), np.zeros(n_items, dtype=np.int32)
    top = positive[np.argsort(-sims[positive], kind="stable")[:neighbours]]
    weights = sims[top].astype(np.float64)
    rated = (train[top] > 0).astype(np.float64)
    numerator = weights @ train[top].astype(np.float64)
    denominator = weights @ rated
    support = rated.sum(axis=0).astype(np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denominator > 0, numerator / np.where(denominator == 0, 1, denominator), 0.0)
    return scores, support


def item_based_scores(train, S_items, u):
    """Similarity-weighted average over every movie the user rated (sim > 0)."""
    rated = np.flatnonzero(train[u] > 0)
    n_items = train.shape[1]
    if rated.size == 0:
        return np.zeros(n_items), np.zeros(n_items, dtype=np.int32)
    block = S_items[:, rated].astype(np.float64)
    block = np.where(block > 0, block, 0.0)
    numerator = block @ train[u, rated].astype(np.float64)
    denominator = block.sum(axis=1)
    support = (block > 0).sum(axis=1).astype(np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denominator > 0, numerator / np.where(denominator == 0, 1, denominator), 0.0)
    return scores, support


def item_based_knn_scores(train, S_items, u, k=20):
    """Item-based CF restricted to the k most similar rated movies per candidate.

    The specification aggregates over every movie the user rated. That lets a
    candidate accumulate a high weighted average out of many weak, single-co-rater
    similarities, which is how films with one rating in the whole dataset reach
    the top. Keeping only the k strongest links per candidate is the textbook fix.
    """
    rated = np.flatnonzero(train[u] > 0)
    n_items = train.shape[1]
    if rated.size == 0:
        return np.zeros(n_items), np.zeros(n_items, dtype=np.int32)
    block = np.where(S_items[:, rated] > 0, S_items[:, rated], 0.0).astype(np.float64)
    if block.shape[1] > k:
        weakest = np.argpartition(-block, k, axis=1)[:, k:]
        np.put_along_axis(block, weakest, 0.0, axis=1)
    numerator = block @ train[u, rated].astype(np.float64)
    denominator = block.sum(axis=1)
    support = (block > 0).sum(axis=1).astype(np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denominator > 0, numerator / np.where(denominator == 0, 1, denominator), 0.0)
    return scores, support


def train_mf(train, factors=20, epochs=30, lr=0.005, reg=0.02, seed=0):
    """Biased matrix factorisation by SGD — the 'advanced technique' row of the
    lecture's missing-value table, kept offline because it needs training."""
    rng = np.random.default_rng(seed)
    users, items = np.nonzero(train)
    values = train[users, items]
    mu = float(values.mean())
    bu = np.zeros(train.shape[0])
    bi = np.zeros(train.shape[1])
    P = rng.normal(0, 0.05, (train.shape[0], factors))
    Q = rng.normal(0, 0.05, (train.shape[1], factors))
    order = np.arange(len(values))
    for _ in range(epochs):
        rng.shuffle(order)
        for k in order:
            u, i, r = users[k], items[k], values[k]
            error = r - (mu + bu[u] + bi[i] + P[u] @ Q[i])
            bu[u] += lr * (error - reg * bu[u])
            bi[i] += lr * (error - reg * bi[i])
            pu = P[u].copy()
            P[u] += lr * (error * Q[i] - reg * P[u])
            Q[i] += lr * (error * pu - reg * Q[i])
    return {"mu": mu, "bu": bu, "bi": bi, "P": P, "Q": Q}


def mf_scores(mf, u):
    return mf["mu"] + mf["bu"][u] + mf["bi"] + mf["Q"] @ mf["P"][u]


# --------------------------------------------------------------------- metrics

def gini(counts):
    x = np.sort(counts.astype(float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def evaluate(score_fn, train, test_pairs, relevant, popularity, is_tail,
             min_support=1, min_item_ratings=0, rank_fn=None, label=""):
    """One pass over every user: rating-prediction error and Top-5 quality.

    `rank_fn` lets a model rank by something other than its predicted rating
    (the popularity baseline predicts the item mean but ranks by rating count).
    `min_item_ratings` drops candidates the training set barely knows.
    """
    n_users, n_items = train.shape
    test_by_user = {}
    for u, i, r in test_pairs:
        test_by_user.setdefault(int(u), []).append((int(i), r))

    squared, absolute, predicted, total = 0.0, 0.0, 0, 0
    hits, precision, recall, evaluated = 0, 0.0, 0.0, 0
    rec_counts = np.zeros(n_items, dtype=np.int64)
    supports = []
    started = time.perf_counter()

    for u in range(n_users):
        scores, support = score_fn(u)
        if scores is None:
            continue

        for item, rating in test_by_user.get(u, []):
            total += 1
            if support[item] >= min_support and scores[item] > 0:
                predicted += 1
                error = scores[item] - rating
                squared += error * error
                absolute += abs(error)

        wanted = relevant.get(u)
        if not wanted:
            continue
        candidates = (scores if rank_fn is None else rank_fn(u)).astype(float).copy()
        candidates[train[u] > 0] = -np.inf
        if min_item_ratings > 0:
            candidates[popularity < min_item_ratings] = -np.inf
        candidates[support < min_support] = -np.inf
        candidates[candidates <= 0] = -np.inf
        finite = np.flatnonzero(np.isfinite(candidates))
        if finite.size == 0:
            evaluated += 1
            continue
        # score desc, then evidence desc, then id asc — the tie-break cf.js uses
        order = np.lexsort((finite, -support[finite], -candidates[finite]))
        top = finite[order[:TOP_K]]
        rec_counts[top] += 1
        supports.extend(support[top].tolist())
        hit = len(set(top.tolist()) & wanted)
        hits += hit > 0
        precision += hit / TOP_K
        recall += hit / len(wanted)
        evaluated += 1

    elapsed = time.perf_counter() - started
    flat = rec_counts.sum()
    return {
        "label": label,
        "rmse": float(np.sqrt(squared / predicted)) if predicted else None,
        "mae": float(absolute / predicted) if predicted else None,
        "prediction_coverage_pct": float(100.0 * predicted / total) if total else 0.0,
        "hit_rate_at_5_pct": float(100.0 * hits / evaluated) if evaluated else 0.0,
        "precision_at_5_pct": float(100.0 * precision / evaluated) if evaluated else 0.0,
        "recall_at_5_pct": float(100.0 * recall / evaluated) if evaluated else 0.0,
        "users_evaluated": evaluated,
        "catalogue_coverage_pct": float(100.0 * (rec_counts > 0).sum() / n_items),
        "gini": gini(rec_counts),
        "mean_popularity_of_rec": float(np.average(popularity, weights=rec_counts)) if flat else 0.0,
        "long_tail_share_pct": float(100.0 * np.average(is_tail, weights=rec_counts)) if flat else 0.0,
        "median_evidence": float(np.median(supports)) if supports else 0.0,
        "seconds": elapsed,
    }


# ------------------------------------------------------------------------ main

def main():
    titles, ratings = load()
    n_users = int(ratings[:, 0].max())
    n_items = max(titles)
    train, test_pairs = temporal_split(ratings, n_users, n_items)

    popularity = (train > 0).sum(axis=0)
    head = np.argsort(-popularity, kind="stable")[:int(round(HEAD_FRACTION * n_items))]
    is_tail = np.ones(n_items, dtype=bool)
    is_tail[head] = False

    relevant = {}
    for u, i, r in test_pairs:
        if r >= LIKE_THRESHOLD:
            relevant.setdefault(int(u), set()).add(int(i))

    results = {
        "dataset": {
            "users": n_users, "items": n_items, "ratings": int(len(ratings)),
            "train_ratings": int((train > 0).sum()), "test_ratings": int(len(test_pairs)),
            "density_pct": float(100.0 * (train > 0).sum() / (n_users * n_items)),
            "users_with_relevant_test_items": len(relevant),
            "mean_profile": float((train > 0).sum(axis=1).mean()),
            "median_profile": float(np.median((train > 0).sum(axis=1))),
            "items_with_under_5_ratings": int((popularity < 5).sum()),
            "gamma": GAMMA, "neighbours": NEIGHBOURS, "top_k": TOP_K,
        },
        "experiments": {},
    }

    # ---- cost of the two directions ---------------------------------------
    cost = {}
    for name, matrix in (("user_based", train), ("item_based", train.T)):
        started = time.perf_counter()
        S = similarity_matrix(matrix, "weighted")
        cost[name] = {
            "rows": int(matrix.shape[0]),
            "pairs": int(matrix.shape[0] * (matrix.shape[0] - 1) / 2),
            "similarity_seconds": time.perf_counter() - started,
            "matrix_mb": float(S.nbytes / 1048576),
            "nonzero_similarity_pct": float(100.0 * (S > 0).sum() / S.size),
        }
        del S
    cost["item_to_user_pair_ratio"] = cost["item_based"]["pairs"] / cost["user_based"]["pairs"]
    results["experiments"]["cost"] = cost

    # ---- baselines ---------------------------------------------------------
    item_mean = np.divide(train.sum(axis=0), np.maximum((train > 0).sum(axis=0), 1))
    global_mean = float(train[train > 0].mean())
    item_mean = np.where((train > 0).sum(axis=0) > 0, item_mean, global_mean)

    item_support = (train > 0).sum(axis=0).astype(np.int32)

    def item_mean_fn(u):
        return item_mean * (item_support > 0), item_support

    baselines = {
        "item_mean": evaluate(item_mean_fn, train, test_pairs, relevant, popularity, is_tail,
                              label="Item mean (no personalisation)"),
        "popularity": evaluate(item_mean_fn, train, test_pairs, relevant, popularity, is_tail,
                               rank_fn=lambda u: popularity.astype(float),
                               label="Most-rated items"),
    }
    results["experiments"]["baselines"] = baselines

    # ---- user-based vs item-based, three strategies ------------------------
    grid = {}
    for strategy in STRATEGIES:
        S_users = similarity_matrix(train, strategy)
        grid[f"user_based/{strategy}"] = evaluate(
            lambda u, S=S_users: user_based_scores(train, S, u),
            train, test_pairs, relevant, popularity, is_tail,
            label=f"User-based / {strategy}")
        del S_users

        S_items = similarity_matrix(train.T, strategy)
        grid[f"item_based/{strategy}"] = evaluate(
            lambda u, S=S_items: item_based_scores(train, S, u),
            train, test_pairs, relevant, popularity, is_tail,
            label=f"Item-based / {strategy}")
        del S_items
        print(f"  done {strategy}")
    results["experiments"]["strategies"] = grid

    # ---- matrix factorisation ---------------------------------------------
    started = time.perf_counter()
    mf = train_mf(train)
    train_seconds = time.perf_counter() - started
    full_support = np.full(n_items, 10_000, dtype=np.int32)   # MF always predicts
    mf_result = evaluate(lambda u: (mf_scores(mf, u), full_support),
                         train, test_pairs, relevant, popularity, is_tail,
                         label="Matrix factorisation (k=20)")
    mf_result["train_seconds"] = train_seconds
    results["experiments"]["matrix_factorisation"] = mf_result
    print("  done MF")

    # ---- gamma sweep for the weighted strategy -----------------------------
    sweep = {}
    for gamma in (1, 5, 10, 25, 50, 100):
        S = similarity_matrix(train, "weighted", gamma=gamma)
        sweep[str(gamma)] = evaluate(lambda u, S=S: user_based_scores(train, S, u),
                                     train, test_pairs, relevant, popularity, is_tail,
                                     label=f"User-based / weighted, gamma={gamma}")
        del S
    results["experiments"]["gamma_sweep"] = sweep
    print("  done gamma sweep")

    # ---- evidence threshold ------------------------------------------------
    evidence = {}
    S_users = similarity_matrix(train, "weighted")
    S_items = similarity_matrix(train.T, "weighted")
    for min_support in (1, 3, 5, 10):
        evidence[f"user_based/{min_support}"] = evaluate(
            lambda u: user_based_scores(train, S_users, u), train, test_pairs, relevant,
            popularity, is_tail, min_support=min_support, label=f"User-based, >= {min_support}")
        evidence[f"item_based/{min_support}"] = evaluate(
            lambda u: item_based_scores(train, S_items, u), train, test_pairs, relevant,
            popularity, is_tail, min_support=min_support, label=f"Item-based, >= {min_support}")
    results["experiments"]["evidence_threshold"] = evidence
    print("  done evidence sweep")

    # ---- how well does the catalogue side have to be known? ----------------
    item_evidence = {}
    for minimum in (0, 5, 10, 25):
        item_evidence[f"user_based/{minimum}"] = evaluate(
            lambda u: user_based_scores(train, S_users, u), train, test_pairs, relevant,
            popularity, is_tail, min_item_ratings=minimum,
            label=f"User-based, candidates with >= {minimum} ratings")
        item_evidence[f"item_based/{minimum}"] = evaluate(
            lambda u: item_based_scores(train, S_items, u), train, test_pairs, relevant,
            popularity, is_tail, min_item_ratings=minimum,
            label=f"Item-based, candidates with >= {minimum} ratings")
    results["experiments"]["item_evidence"] = item_evidence
    results["experiments"]["item_based_knn"] = evaluate(
        lambda u: item_based_knn_scores(train, S_items, u, k=20),
        train, test_pairs, relevant, popularity, is_tail,
        label="Item-based, 20 most similar rated movies per candidate")
    print("  done item-evidence sweep")

    # ---- cold start and sparsity ------------------------------------------
    profile = (train > 0).sum(axis=1)
    buckets = [(0, 20), (21, 50), (51, 100), (101, 200), (201, 10_000)]
    cold = {"by_profile_size": {}}
    for low, high in buckets:
        members = set(np.flatnonzero((profile >= low) & (profile <= high)).tolist())
        subset = {u: v for u, v in relevant.items() if u in members}
        if not subset:
            continue
        key = f"{low}-{high if high < 10_000 else '+'}"
        cold["by_profile_size"][key] = {
            "users": len(members),
            "user_based": evaluate(lambda u: user_based_scores(train, S_users, u),
                                   train, test_pairs, subset, popularity, is_tail,
                                   label=key)["precision_at_5_pct"],
            "item_based": evaluate(lambda u: item_based_scores(train, S_items, u),
                                   train, test_pairs, subset, popularity, is_tail,
                                   label=key)["precision_at_5_pct"],
        }

    # how much evidence the similarities themselves rest on
    n_co = co_rated_counts(train)
    plain = similarity_matrix(train, "corated")
    triu = np.triu_indices(n_users, k=1)
    pair_co = n_co[triu]
    pair_sim = plain[triu]
    positive = pair_sim > 0
    cold["similarity_evidence"] = {
        "user_pairs": int(positive.sum()),
        "share_with_at_most_2_co_rated_pct": float(100.0 * (pair_co[positive] <= 2).mean()),
        "share_exactly_1_0_pct": float(100.0 * (pair_sim[positive] >= 0.999999).mean()),
        "co_rated_of_perfect_pairs_median": float(np.median(pair_co[pair_sim >= 0.999999]))
        if (pair_sim >= 0.999999).any() else 0.0,
        "mean_similarity_by_co_rated": {
            label: float(pair_sim[positive][mask].mean())
            for label, mask in (
                ("1", pair_co[positive] == 1), ("2", pair_co[positive] == 2),
                ("3-5", (pair_co[positive] >= 3) & (pair_co[positive] <= 5)),
                ("6-25", (pair_co[positive] >= 6) & (pair_co[positive] <= 25)),
                ("26+", pair_co[positive] >= 26))
            if mask.any()
        },
    }
    # items nobody can reach: never rated in train
    cold["cold_items"] = {
        "items_with_no_train_rating": int((popularity == 0).sum()),
        "items_with_under_5_train_ratings": int((popularity < 5).sum()),
        "share_of_catalogue_pct": float(100.0 * (popularity < 5).sum() / n_items),
    }
    results["experiments"]["cold_start"] = cold
    del S_users, S_items, plain, n_co

    # ---- reference Top-5 lists for the browser test ------------------------
    # Computed on the FULL rating matrix (no split), because that is what the
    # page has loaded. Same formulas, same tie-break as cf.js.
    full = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i, r, _ts in ratings:
        full[int(u) - 1, int(i) - 1] = r
    full_popularity = (full > 0).sum(axis=0)

    def top5(scores, support, u, min_support):
        candidates = scores.astype(float).copy()
        candidates[full[u] > 0] = -np.inf
        candidates[support < min_support] = -np.inf
        candidates[candidates <= 0] = -np.inf
        finite = np.flatnonzero(np.isfinite(candidates))
        order = np.lexsort((finite, -support[finite], -candidates[finite]))
        return [{"id": int(i + 1), "title": titles[i + 1],
                 "score": round(float(scores[i]), 6), "support": int(support[i])}
                for i in finite[order[:TOP_K]]]

    ref = {}
    for strategy in STRATEGIES:
        S_u = similarity_matrix(full, strategy)
        S_i = similarity_matrix(full.T, strategy)
        for user in (1, 100, 405):
            u = user - 1
            for min_support in (1, 5):
                key = f"{user}/{strategy}/{min_support}"
                ref[key] = {
                    "user_based": top5(*user_based_scores(full, S_u, u), u, min_support),
                    "item_based": top5(*item_based_scores(full, S_i, u), u, min_support),
                }
        del S_u, S_i
    results["reference_top5"] = ref
    results["reference_popularity"] = {
        str(int(i + 1)): int(full_popularity[i]) for i in np.argsort(-full_popularity)[:5]
    }
    print("  done reference lists")

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_JSON}")

    print("\n-- cost --")
    for name in ("user_based", "item_based"):
        c = cost[name]
        print(f"{name:11s} rows {c['rows']:5d}  pairs {c['pairs']:9,d}  "
              f"{c['similarity_seconds']:6.2f}s  {c['matrix_mb']:6.1f} MB")
    print("\n-- accuracy (min. evidence 1) --")
    header = f"{'model':28s} {'RMSE':>6s} {'cover':>7s} {'hit@5':>7s} {'P@5':>6s} {'tail':>6s} {'gini':>6s}"
    print(header)
    rows = [("popularity", baselines["popularity"])] + list(grid.items()) + \
           [("matrix_factorisation", mf_result)]
    for name, r in rows:
        print(f"{name:28s} {r['rmse'] or 0:6.3f} {r['prediction_coverage_pct']:6.1f}% "
              f"{r['hit_rate_at_5_pct']:6.1f}% {r['precision_at_5_pct']:5.1f}% "
              f"{r['long_tail_share_pct']:5.1f}% {r['gini']:6.3f}")


if __name__ == "__main__":
    main()
