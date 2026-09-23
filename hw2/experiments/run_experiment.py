"""Offline evaluation for A02: Jaccard vs cosine, item-to-item vs profile-based.

Reads ../u.item and ../u.data (MovieLens 100k), reproduces exactly the scoring rules
implemented in ../recommender.js, and writes results.json + figures/.

Run:  python experiments/run_experiment.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
OUT_JSON = HERE / "results.json"

GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]
TOP_N = 5
LIKE_THRESHOLD = 4.0      # a rating of 4 or 5 counts as "watched and liked"
PROFILE_SIZE = 3          # the assignment's "e.g. 3 watched titles"
HEAD_FRACTION = 0.20      # top 20% of items by rating count = the head, the rest = long tail
SHRINK_M = 25.0           # Bayesian shrinkage strength for the quality tie-break


# --------------------------------------------------------------------------- data

def load_movies(shift_bug: bool = False):
    """Parse u.item. With shift_bug=True, reproduce the starter's off-by-one genre mapping.

    Fields 5..23 are 19 flags: [unknown, Action, ..., Western]. The starter zips the 18
    genre names against all 19 flags, so every genre label is shifted by one position.
    """
    text = (DATA / "u.item").read_text(encoding="latin-1")
    ids, titles, vectors = [], [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        f = line.split("|")
        flags = [int(v) for v in f[5:24]]
        vec = np.zeros(len(GENRE_NAMES), dtype=float)
        for i in range(len(GENRE_NAMES)):
            # correct: name i <- flag i+1 (flag 0 is "unknown"); starter: name i <- flag i
            if flags[i if shift_bug else i + 1] == 1:
                vec[i] = 1.0
        ids.append(int(f[0]))
        titles.append(f[1])
        vectors.append(vec)
    return np.array(ids), titles, np.array(vectors)


def load_ratings():
    text = (DATA / "u.data").read_text(encoding="ascii")
    rows = [line.split("\t") for line in text.splitlines() if line.strip()]
    arr = np.array([[int(r[0]), int(r[1]), float(r[2]), int(r[3])] for r in rows])
    return arr  # userId, itemId, rating, timestamp


# ----------------------------------------------------------------------- scoring

def cosine_matrix(G: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(G, axis=1)
    safe = np.where(norms == 0, 1.0, norms)
    U = G / safe[:, None]
    S = U @ U.T
    S[norms == 0, :] = 0.0
    S[:, norms == 0] = 0.0
    return S


def jaccard_matrix(G: np.ndarray) -> np.ndarray:
    inter = G @ G.T
    sizes = G.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(union > 0, inter / np.where(union == 0, 1, union), 0.0)
    return S


def overlap_matrix(G: np.ndarray) -> np.ndarray:
    """Unnormalised dot product: the 'naive matching' baseline."""
    return G @ G.T


def cosine_to_vector(G: np.ndarray, v: np.ndarray) -> np.ndarray:
    nv = np.linalg.norm(v)
    if nv == 0:
        return np.zeros(len(G))
    norms = np.linalg.norm(G, axis=1)
    safe = np.where(norms == 0, 1.0, norms)
    return np.where(norms == 0, 0.0, (G @ v) / (safe * nv))


def rank_top(scores: np.ndarray, exclude: set[int], tiebreak: np.ndarray, n: int = TOP_N):
    """Rank by score desc, then by tiebreak desc, then by index asc. Returns row indices."""
    mask = np.ones(len(scores), dtype=bool)
    for i in exclude:
        mask[i] = False
    idx = np.flatnonzero(mask)
    order = np.lexsort((idx, -tiebreak[idx], -scores[idx]))
    return idx[order[:n]]


# ----------------------------------------------------------------------- metrics

def gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def summarise(rec_lists, popularity, genre_counts, is_tail, n_items):
    flat = np.concatenate(rec_lists) if rec_lists else np.array([], dtype=int)
    counts = np.bincount(flat, minlength=n_items)
    return {
        "recommendations": int(len(flat)),
        "mean_genres_per_rec": float(genre_counts[flat].mean()),
        "mean_popularity": float(popularity[flat].mean()),
        "median_popularity": float(np.median(popularity[flat])),
        "long_tail_share_pct": float(100.0 * is_tail[flat].mean()),
        "catalog_coverage_pct": float(100.0 * (counts > 0).sum() / n_items),
        "gini": float(gini(counts)),
        "top_item_share_pct": float(100.0 * counts.max() / len(flat)),
    }


# -------------------------------------------------------------------- experiments

def main():
    ids, titles, G = load_movies()
    _, _, G_bug = load_movies(shift_bug=True)
    ratings = load_ratings()
    n_items = len(ids)
    index_of = {int(m): i for i, m in enumerate(ids)}

    # popularity and shrunk quality per item
    popularity = np.zeros(n_items)
    rating_sum = np.zeros(n_items)
    for item_id, rating in zip(ratings[:, 1].astype(int), ratings[:, 2]):
        k = index_of[item_id]
        popularity[k] += 1
        rating_sum[k] += rating
    global_mean = float(ratings[:, 2].mean())
    mean_rating = np.where(popularity > 0, rating_sum / np.maximum(popularity, 1), global_mean)
    quality = (popularity * mean_rating + SHRINK_M * global_mean) / (popularity + SHRINK_M)

    genre_counts = G.sum(axis=1)
    head_size = int(round(HEAD_FRACTION * n_items))
    head_idx = np.argsort(-popularity, kind="stable")[:head_size]
    is_tail = np.ones(n_items, dtype=bool)
    is_tail[head_idx] = False

    results: dict = {
        "dataset": {
            "movies": n_items,
            "ratings": int(len(ratings)),
            "users": int(len(np.unique(ratings[:, 0]))),
            "genres": len(GENRE_NAMES),
            "movies_without_genre": int((genre_counts == 0).sum()),
            "mean_genres_per_movie": float(genre_counts.mean()),
            "head_items": head_size,
            "head_share_of_ratings_pct": float(100 * popularity[head_idx].sum() / popularity.sum()),
            "global_mean_rating": global_mean,
        }
    }

    # ---- 0. the genre off-by-one bug in the starter -------------------------
    disagree = int((G != G_bug).any(axis=1).sum())
    results["genre_bug"] = {
        "movies_with_wrong_genres": disagree,
        "movies_with_wrong_genres_pct": float(100.0 * disagree / n_items),
        "western_movies_correct": int(G[:, GENRE_NAMES.index("Western")].sum()),
        "western_movies_starter": int(G_bug[:, GENRE_NAMES.index("Western")].sum()),
        "examples": [
            {
                "title": titles[index_of[mid]],
                "correct": [GENRE_NAMES[j] for j in np.flatnonzero(G[index_of[mid]])],
                "starter": [GENRE_NAMES[j] for j in np.flatnonzero(G_bug[index_of[mid]])],
            }
            for mid in (1, 50, 121)
        ],
    }

    # ---- 1. item-to-item: overlap vs Jaccard vs cosine ----------------------
    matrices = {
        "overlap": overlap_matrix(G),
        "jaccard": jaccard_matrix(G),
        "cosine": cosine_matrix(G),
    }
    exp1 = {}
    for name, S in matrices.items():
        rec_lists, tie_sizes = [], []
        for q in range(n_items):
            if genre_counts[q] == 0:
                continue
            recs = rank_top(S[q], {q}, quality)
            rec_lists.append(recs)
            scores = np.delete(S[q], q)
            tie_sizes.append(int((scores == scores.max()).sum()))
        stats = summarise(rec_lists, popularity, genre_counts, is_tail, n_items)
        stats["queries"] = len(rec_lists)
        stats["mean_top_score_ties"] = float(np.mean(tie_sizes))
        stats["median_top_score_ties"] = float(np.median(tie_sizes))
        exp1[name] = stats
    results["experiment_1_item_to_item"] = exp1

    # how often the three scorers return the same list
    def top_sets(S, n):
        return [set(rank_top(S[q], {q}, quality, n).tolist())
                for q in range(n_items) if genre_counts[q] > 0]

    agreement = {}
    for n in (TOP_N, 20):
        sets = {k: top_sets(S, n) for k, S in matrices.items()}
        for a, b in (("jaccard", "cosine"), ("overlap", "cosine"), ("overlap", "jaccard")):
            agreement[f"top{n}_{a}_vs_{b}"] = float(
                np.mean([len(x & y) / n for x, y in zip(sets[a], sets[b])]))
    # do the raw scores (not just the Top-N sets) ever disagree on the order?
    disagreements = 0
    for q in range(n_items):
        if genre_counts[q] == 0:
            continue
        c, j = matrices["cosine"][q], matrices["jaccard"][q]
        order = np.lexsort((np.arange(n_items), -quality, -c))
        cs, js = c[order], j[order]
        keep = np.arange(n_items) != q
        cs, js = cs[keep[order]], js[keep[order]]
        # a pair is an inversion if cosine and Jaccard disagree strictly
        if np.any(np.diff(js[:20]) > 1e-12):
            disagreements += 1
    agreement["queries_where_jaccard_not_monotone_in_cosine_top20"] = disagreements
    # why there are so many ties: how many distinct genre sets exist at all
    distinct_sets = len({tuple(row) for row in G.astype(int)})
    agreement["distinct_genre_vectors"] = distinct_sets
    agreement["exact_genre_twins_mean"] = float(np.mean([
        int((np.abs(G - G[q]).sum(axis=1) == 0).sum() - 1)
        for q in range(n_items) if genre_counts[q] > 0]))
    results["experiment_1_agreement"] = agreement

    # ---- 2. tie-break sensitivity (cosine, item-to-item) -------------------
    tiebreaks = {
        "id": -np.arange(n_items, dtype=float),   # ascending id == descending -id
        "popularity": popularity,
        "quality": quality,
    }
    exp2 = {}
    for name, tb in tiebreaks.items():
        rec_lists = [rank_top(matrices["cosine"][q], {q}, tb)
                     for q in range(n_items) if genre_counts[q] > 0]
        exp2[name] = summarise(rec_lists, popularity, genre_counts, is_tail, n_items)
    results["experiment_2_tiebreak"] = exp2

    # ---- 3. item-to-item vs profile-based on real users --------------------
    users = {}
    for u, i, r, t in ratings:
        if r >= LIKE_THRESHOLD:
            users.setdefault(int(u), []).append((int(t), index_of[int(i)]))

    seeds_per_user, active_per_user, heldout_per_user = [], [], []
    for u, events in sorted(users.items()):
        events.sort()                                   # chronological
        liked = [k for _, k in events if genre_counts[k] > 0]
        # keep the order but drop duplicates
        seen, ordered = set(), []
        for k in liked:
            if k not in seen:
                seen.add(k)
                ordered.append(k)
        if len(ordered) < PROFILE_SIZE + 1:
            continue
        seeds = ordered[:PROFILE_SIZE]                  # 3 earliest liked movies
        seeds_per_user.append(seeds)
        active_per_user.append(seeds[-1])               # the "single active item"
        heldout_per_user.append(set(ordered[PROFILE_SIZE:]))

    exp3 = {"users_evaluated": len(seeds_per_user)}
    rec_counts_for_figure = {}
    for tb_name, tb in tiebreaks.items():
        item_recs, profile_recs, overlaps = [], [], []
        item_hits, profile_hits = [], []
        item_ties, profile_ties = [], []
        item_prec, profile_prec = [], []
        for seeds, active, heldout in zip(seeds_per_user, active_per_user, heldout_per_user):
            exclude = set(seeds)
            s_item = matrices["cosine"][active]
            profile = G[seeds].mean(axis=0)
            s_prof = cosine_to_vector(G, profile)

            r_item = rank_top(s_item, exclude, tb)
            r_prof = rank_top(s_prof, exclude, tb)
            item_recs.append(r_item)
            profile_recs.append(r_prof)
            overlaps.append(len(set(r_item.tolist()) & set(r_prof.tolist())) / TOP_N)

            hit_i = len(set(r_item.tolist()) & heldout)
            hit_p = len(set(r_prof.tolist()) & heldout)
            item_hits.append(hit_i > 0)
            profile_hits.append(hit_p > 0)
            item_prec.append(hit_i / TOP_N)
            profile_prec.append(hit_p / TOP_N)

            for s, store in ((s_item, item_ties), (s_prof, profile_ties)):
                free = np.array([s[k] for k in range(n_items) if k not in exclude])
                store.append(int((free == free.max()).sum()))

        block = {}
        for name, recs, hits, prec, ties in (
            ("item_to_item", item_recs, item_hits, item_prec, item_ties),
            ("profile_based", profile_recs, profile_hits, profile_prec, profile_ties),
        ):
            stats = summarise(recs, popularity, genre_counts, is_tail, n_items)
            stats["hit_rate_at_5_pct"] = float(100.0 * np.mean(hits))
            stats["precision_at_5_pct"] = float(100.0 * np.mean(prec))
            stats["mean_top_score_ties"] = float(np.mean(ties))
            stats["median_top_score_ties"] = float(np.median(ties))
            block[name] = stats
            rec_counts_for_figure[f"{name}/{tb_name}"] = np.bincount(
                np.concatenate(recs), minlength=n_items).tolist()
        block["mean_top5_overlap"] = float(np.mean(overlaps))
        block["identical_top5_pct"] = float(100.0 * np.mean([o == 1.0 for o in overlaps]))
        block["disjoint_top5_pct"] = float(100.0 * np.mean([o == 0.0 for o in overlaps]))
        exp3[f"tiebreak_{tb_name}"] = block
    results["experiment_3_item_vs_profile"] = exp3
    (HERE / "rec_counts.json").write_text(
        json.dumps({"popularity": popularity.tolist(), "counts": rec_counts_for_figure}),
        encoding="utf-8")

    # ---- 4. worked example for the report and the browser test -------------
    demo_seeds = [1, 50, 275]      # Toy Story (1995), Star Wars (1977), Fargo (1996)
    demo = {"seed_movies": [], "item_to_item": [], "profile_based": []}
    seed_idx = [index_of[m] for m in demo_seeds]
    for m in demo_seeds:
        k = index_of[m]
        demo["seed_movies"].append({
            "id": m, "title": titles[k],
            "genres": [GENRE_NAMES[j] for j in np.flatnonzero(G[k])],
        })
    active = seed_idx[-1]
    profile_vec = G[seed_idx].mean(axis=0)
    demo["profile_vector"] = {GENRE_NAMES[j]: round(float(profile_vec[j]), 4)
                              for j in np.flatnonzero(profile_vec)}
    demo["active_item"] = {"id": int(ids[active]), "title": titles[active]}
    for key, scores in (("item_to_item", matrices["cosine"][active]),
                        ("profile_based", cosine_to_vector(G, profile_vec))):
        for k in rank_top(scores, set(seed_idx), quality):
            demo[key].append({
                "id": int(ids[k]), "title": titles[k], "score": round(float(scores[k]), 4),
                "genres": [GENRE_NAMES[j] for j in np.flatnonzero(G[k])],
                "ratings": int(popularity[k]),
            })
    results["worked_example"] = demo

    # Reference Top-5 lists that the browser test replays against the live app.
    # The app always excludes every movie in the watch history from both columns,
    # so the references are built with the same exclusion set.
    def as_list(scores, exclude):
        return [{"id": int(ids[j]), "title": titles[j], "score": round(float(scores[j]), 6)}
                for j in rank_top(scores, exclude, quality)]

    reference = {"single_movie": {}, "history": {}}
    for m in (1, 50, 121, 275, 313):
        k = index_of[m]
        reference["single_movie"][str(m)] = as_list(matrices["cosine"][k], {k})
    for combo in ([1, 50, 275], [7, 98, 181], [100, 258, 286]):
        idxs = [index_of[m] for m in combo]
        exclude = set(idxs)
        vec = G[idxs].mean(axis=0)
        reference["history"]["+".join(map(str, combo))] = {
            "active": int(ids[idxs[-1]]),
            "item_to_item": as_list(matrices["cosine"][idxs[-1]], exclude),
            "profile_based": as_list(cosine_to_vector(G, vec), exclude),
            "profile_vector": [round(float(v), 6) for v in vec],
        }
    # the same three histories under the two other tie-breaks, to prove the app's
    # tie-break switch is wired to the same rule as the offline script
    reference["tiebreaks"] = {}
    for tb_name, tb in (("popularity", popularity), ("id", tiebreaks["id"])):
        idxs = [index_of[m] for m in (1, 50, 275)]
        vec = G[idxs].mean(axis=0)
        reference["tiebreaks"][tb_name] = [
            {"id": int(ids[j]), "title": titles[j]}
            for j in rank_top(cosine_to_vector(G, vec), set(idxs), tb)
        ]
    results["reference_top5"] = reference

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUT_JSON}")

    # a compact console summary
    print("\n-- item-to-item, all 1682 queries, Top-5 --")
    for name, s in exp1.items():
        print(f"{name:9s} genres/rec {s['mean_genres_per_rec']:.2f}  "
              f"pop {s['mean_popularity']:7.1f}  tail {s['long_tail_share_pct']:5.1f}%  "
              f"cover {s['catalog_coverage_pct']:5.1f}%  gini {s['gini']:.3f}  "
              f"ties {s['mean_top_score_ties']:.0f}")
    print(f"\n-- item vs profile, {exp3['users_evaluated']} real users --")
    for tb_name in tiebreaks:
        block = exp3[f"tiebreak_{tb_name}"]
        for name in ("item_to_item", "profile_based"):
            s = block[name]
            print(f"{tb_name:10s} {name:13s} pop {s['mean_popularity']:7.1f}  "
                  f"tail {s['long_tail_share_pct']:5.1f}%  cover {s['catalog_coverage_pct']:5.1f}%  "
                  f"gini {s['gini']:.3f}  ties {s['mean_top_score_ties']:6.1f}  "
                  f"hit@5 {s['hit_rate_at_5_pct']:.1f}%  P@5 {s['precision_at_5_pct']:.1f}%")
        print(f"{'':10s} mean Top-5 overlap between the two modes: "
              f"{block['mean_top5_overlap']:.3f}")


if __name__ == "__main__":
    main()
