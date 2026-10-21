"""The recommenders compared on one split under one protocol.

Every method returns a dense score matrix, rows = split.eval_users and
columns = split.candidates (shows on air in the target weeks), and is free to
score shows the user has already watched: the target is "the shows this person
will watch most", and most of those are shows they already watch.

  popularity   total seconds watched by everyone in the history weeks
  history      this user's own seconds per show (re-watch baseline)
  cbf          content-based: TF-IDF profile over category / genres / channel / title
  rules        association rules X -> Y over viewing baskets, confidence-weighted
  mf           implicit ALS matrix factorisation on log-minutes
  ppr          personalised PageRank on the user-show bipartite graph
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def log_minutes(X):
    """Seconds -> log(1 + minutes): one long binge should not outweigh steady habit."""
    Y = X.copy().astype(np.float64)
    Y.data = np.log1p(Y.data / 60.0)
    return Y


def popularity_vector(split):
    return np.asarray(split.X[:, split.cand_cols].sum(axis=0)).ravel()


# --- baselines -------------------------------------------------------------

def popularity(split, **_):
    pop = popularity_vector(split).astype(np.float32)
    return np.broadcast_to(pop, (len(split.eval_users), len(pop))).copy()


def history(split, **_):
    return split.X[split.eval_rows][:, split.cand_cols].toarray().astype(np.float32)


# --- content-based ---------------------------------------------------------

_GENERIC = {"сезон", "серия", "серии", "эп", "часть", "фильм", "выпуск", "программа",
            "сериал", "передача", "the", "and", "of", "новый"}


def _title_tokens(title):
    words = re.findall(r"[a-zа-яёіїєґ0-9]+", str(title).lower())
    return [w for w in words if len(w) >= 3 and not w.isdigit() and w not in _GENERIC]


def show_documents(split, fields=("channel", "category", "genre", "people", "title")):
    """One bag of metadata tokens per show, from every broadcast the split may see."""
    epg = pd.concat([split.epg_hist, split.epg_target], ignore_index=True)
    epg = epg.assign(air=(epg["end"] - epg["start"]).dt.total_seconds())

    # The channel that carries most of a show's airtime, and its first listing.
    main_channel = (epg.groupby(["tv_show_id", "channel_id"])["air"].sum()
                    .reset_index().sort_values("air", ascending=False)
                    .drop_duplicates("tv_show_id").set_index("tv_show_id")["channel_id"])
    first = epg.drop_duplicates("tv_show_id").set_index("tv_show_id")

    docs = {}
    for show, row in first.iterrows():
        tokens = []
        if "channel" in fields:
            tokens.append(f"ch={main_channel.get(show)}")
        if "category" in fields and pd.notna(row["tv_show_category"]):
            tokens.append(f"cat={row['tv_show_category']}")
        if "genre" in fields:
            for g in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
                if pd.notna(row[g]):
                    tokens.append(f"gen={row[g]}")
        if "people" in fields:
            for col, tag in (("director", "dir"), ("actors", "act")):
                if pd.notna(row[col]):
                    tokens += [f"{tag}={d.strip()}" for d in str(row[col]).split(",") if d.strip()]
        if "title" in fields:
            tokens += [f"t={w}" for w in _title_tokens(row["tv_show_title"])]
        docs[show] = tokens
    return docs


def item_features(split, fields=("channel", "category", "genre", "people", "title")):
    docs = show_documents(split, fields)
    shows = split.uni.shows
    vec = TfidfVectorizer(analyzer=lambda toks: toks, min_df=2, sublinear_tf=True)
    F = vec.fit_transform([docs.get(s, []) for s in shows])
    return normalize(F), vec


def cbf(split, weighting="seconds", fields=("channel", "category", "genre", "people", "title"), **_):
    """Cosine between a user profile and each candidate's metadata vector.

    The profile is the time-weighted mean of the user's watched shows. Weighting
    by raw seconds (the default) lets the few shows that take most of a person's
    time define the profile; log-minutes flattens it, and a long tail of
    once-watched films - each with three genres and a director - then outvotes
    the two or three programmes the person actually lives on.
    """
    F, _ = item_features(split, fields)
    X = split.X[split.eval_rows]
    W = normalize(X if weighting == "seconds" else log_minutes(X), norm="l1")
    profile = normalize(W @ F)
    return (profile @ F[split.cand_cols].T).toarray().astype(np.float32)


# --- association rules -----------------------------------------------------

def rules(split, min_minutes=10, min_support=20, min_pair=5, **_):
    """Confidence of X -> Y, summed over the rules this user's basket fires.

    A basket is the set of shows a user watched for at least `min_minutes`.
    Rules need an antecedent seen in `min_support` baskets and a pair seen in
    `min_pair`; the trivial rule Y -> Y is excluded, so this method only ever
    recommends through co-viewing, never through the user's own repeat habit.
    """
    B = (split.X >= 60 * min_minutes).astype(np.float32).tocsc()
    support = np.asarray(B.sum(axis=0)).ravel()
    cand = split.cand_cols
    co = (B.T @ B[:, cand]).toarray()                     # shows x candidates
    co[co < min_pair] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        conf = np.where(support[:, None] >= min_support, co / support[:, None], 0.0)
    conf[cand, np.arange(len(cand))] = 0.0                # drop Y -> Y
    basket = normalize(log_minutes(split.X[split.eval_rows]).multiply(
        B[split.eval_rows].tocsr() > 0), norm="l1")
    return np.asarray(basket @ conf, dtype=np.float32)


# --- matrix factorisation --------------------------------------------------

def mf(split, factors=64, regularization=0.05, iterations=20, alpha=2.0, min_minutes=0,
       seed=20260112, **_):
    """Implicit ALS (Hu, Koren & Volinsky 2008), confidence = alpha * log-minutes.

    ALS fits a binary preference - watched or not - and uses time only as
    confidence, so a film sampled for ten minutes and the daily news both carry
    target 1. `min_minutes` keeps only pairs above a habit threshold as positives.
    """
    from implicit.als import AlternatingLeastSquares

    X = split.X.copy()
    if min_minutes:
        X.data[X.data < 60 * min_minutes] = 0
        X.eliminate_zeros()
    C = log_minutes(X).tocsr()
    C.data *= alpha
    model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                    iterations=iterations, random_state=seed,
                                    calculate_training_loss=False)
    model.fit(C.astype(np.float32), show_progress=False)
    U = np.asarray(model.user_factors)
    V = np.asarray(model.item_factors)
    return (U[split.eval_rows] @ V[split.cand_cols].T).astype(np.float32)


def svd(split, factors=64, **_):
    """PureSVD (Cremonesi et al. 2010): truncated SVD of the log-minute matrix.

    Unlike ALS it reconstructs the magnitude, so it can tell a habit from a
    sample - which is what a top-by-watch-time target rewards.
    """
    from scipy.sparse.linalg import svds

    M = log_minutes(split.X).tocsc()
    _, _, vt = svds(M, k=factors, random_state=20260112)
    V = vt.T                                            # shows x factors
    P = split.X[split.eval_rows]
    P = log_minutes(P)
    return np.asarray((P @ V) @ V[split.cand_cols].T, dtype=np.float32)


# --- personalised PageRank -------------------------------------------------

def coview_graph(split, min_minutes=10, neighbours=50, weight="cosine"):
    """Programme co-view graph: shows linked by the users who watched both.

    An edge i-j carries the number of shared viewers (a viewer counts once they
    spent `min_minutes` on a show), cosine-normalised by default so that a
    blockbuster is not everyone's neighbour. Each show keeps its `neighbours`
    strongest edges; rows are then normalised into a transition matrix.
    """
    B = (split.X >= 60 * min_minutes).astype(np.float32).tocsc()
    n = np.asarray(B.sum(axis=0)).ravel()
    C = (B.T @ B).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    if weight == "cosine":
        inv = 1.0 / np.sqrt(np.maximum(n, 1))
        C = sp.diags(inv) @ C @ sp.diags(inv)
    C = C.tocsr()

    # Keep the strongest `neighbours` edges per row.
    rows, cols, vals = [], [], []
    for i in range(C.shape[0]):
        lo, hi = C.indptr[i], C.indptr[i + 1]
        if hi == lo:
            continue
        idx, w = C.indices[lo:hi], C.data[lo:hi]
        if len(w) > neighbours:
            keep = np.argpartition(-w, neighbours)[:neighbours]
            idx, w = idx[keep], w[keep]
        rows.append(np.full(len(idx), i)); cols.append(idx); vals.append(w)
    G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=C.shape)
    deg = np.asarray(G.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return (sp.diags(inv) @ G).tocsr(), deg == 0


def ppr(split, alpha=0.5, iterations=30, batch=1024, min_minutes=10, neighbours=50,
        weight="cosine", **_):
    """Personalised PageRank on the programme co-view graph.

    The walk restarts from the user's own history (each show in proportion to
    the seconds spent on it), so `alpha` is a dial: at 0 the scores are the
    user's history, as it grows mass diffuses to shows their shows co-occur with.
    Mass that reaches a show with no edges is returned to the restart vector.
    """
    P, dangling = coview_graph(split, min_minutes, neighbours, weight)
    PT = P.T.tocsr().astype(np.float32)
    dangling = dangling.astype(np.float32)
    R_all = normalize(split.X[split.eval_rows], norm="l1").astype(np.float32)

    cand = split.cand_cols
    out = np.zeros((R_all.shape[0], len(cand)), dtype=np.float32)
    for start in range(0, R_all.shape[0], batch):
        restart = R_all[start:start + batch].T.toarray()        # shows x users
        p = restart.copy()
        for _ in range(iterations):
            lost = dangling @ p
            p = alpha * (PT @ p + restart * lost) + (1 - alpha) * restart
        out[start:start + restart.shape[1]] = p[cand].T
    return out


def exposure(split, **kw):
    import exposure as _exposure
    return _exposure.exposure(split, **kw)


def habit(split, **kw):
    import habit as _habit
    return _habit.habit_schedule(split, **kw)


METHODS = {
    "popularity": popularity,
    "history": history,
    "cbf": cbf,
    "rules": rules,
    "mf": mf,
    "svd": svd,
    "ppr": ppr,
    "exposure": exposure,
    "habit": habit,
}


def top_k(scores, split, k=5, tie_break=None):
    """Top-k candidate show ids per user; ties go to the more popular show."""
    if tie_break is None:
        tie_break = popularity_vector(split)
    cand = split.candidates
    preds = {}
    for i, user in enumerate(split.eval_users):
        row = scores[i]
        order = np.lexsort((cand, -tie_break, -row))[:k]
        preds[user] = cand[order].tolist()
    return preds
