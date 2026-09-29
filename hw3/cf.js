/* Collaborative-filtering maths: similarity and the two recommenders.
 *
 * Pure functions only — no DOM, no fetching. Everything works on the structures
 * built by data.js, so the same code is driven by the page, by the browser test
 * and (in a line-by-line port) by experiments/run_experiment.py.
 *
 * MISSING-VALUE STRATEGY — the app declares ONE default and offers the other two
 * for comparison, because the assignment's analysis section asks for the
 * trade-off between them. The declared default is `weighted`:
 *
 *   weighted  cosine over co-rated entries, damped by min(n, gamma) / gamma
 *             (significance weighting, Herlocker et al. 1999). Two users who
 *             agree on two films get a similarity of 1.0 under plain cosine;
 *             this multiplies that by 2/25, because two films are not evidence.
 *   corated   plain cosine over co-rated entries only — the simplest strategy
 *             and the one the week3 specification names as its default.
 *   mean      every missing entry is replaced by the row's mean rating and the
 *             cosine is taken over the full vector.
 *
 * The three are never mixed inside one calculation: a run uses exactly one.
 */

const DEFAULT_GAMMA = 25;       // co-rated count at which `weighted` stops damping
const DEFAULT_NEIGHBOURS = 20;  // N most similar users for user-based CF
const DEFAULT_TOP_K = 5;

const MISSING_VALUE_STRATEGIES = {
    weighted: {
        label: 'Weighted by co-rated count',
        note: 'co-rated cosine damped by min(n, γ)/γ'
    },
    corated: {
        label: 'Co-rated entries only',
        note: 'plain cosine over the entries both sides rated'
    },
    mean: {
        label: 'Mean imputation',
        note: 'missing entries replaced by the row mean, cosine over the full vector'
    }
};

// --------------------------------------------------------------- reference

/**
 * Cosine similarity between two dense rating vectors, where 0 means "not rated".
 *
 * This is the readable definition of every strategy, written exactly as the
 * specification asks for it. It is what the browser test checks the fast path
 * against; the app itself calls similaritiesAgainstAll below, which computes
 * the same numbers without scanning entries that are missing on both sides.
 *
 * options: { strategy, gamma, meanA, meanB, from, length }
 *   meanA / meanB are only used by the `mean` strategy.
 *   from / length bound the real positions: the rating matrix is indexed by raw
 *   id, so column 0 is padding that no one can rate and must not count as a
 *   missing value under `mean`. The app passes from = 1.
 */
function cosineSimilarity(a, b, options) {
    const settings = Object.assign(
        { strategy: 'weighted', gamma: DEFAULT_GAMMA, meanA: 0, meanB: 0, from: 0 },
        options || {});
    const end = settings.length !== undefined
        ? settings.from + settings.length
        : Math.min(a.length, b.length);

    let dot = 0;
    let normA = 0;
    let normB = 0;
    let coRated = 0;

    for (let k = settings.from; k < end; k++) {
        const ratedA = a[k] > 0;
        const ratedB = b[k] > 0;
        if (ratedA && ratedB) coRated++;

        let valueA;
        let valueB;
        if (settings.strategy === 'mean') {
            // every position counts; a missing entry becomes the row mean
            valueA = ratedA ? a[k] : settings.meanA;
            valueB = ratedB ? b[k] : settings.meanB;
        } else {
            // co-rated entries only; anything else contributes nothing at all,
            // not even to the norms
            if (!ratedA || !ratedB) continue;
            valueA = a[k];
            valueB = b[k];
        }

        dot += valueA * valueB;
        normA += valueA * valueA;
        normB += valueB * valueB;
    }

    if (normA === 0 || normB === 0) return 0;
    const cosine = dot / Math.sqrt(normA * normB);
    return settings.strategy === 'weighted'
        ? cosine * Math.min(coRated, settings.gamma) / settings.gamma
        : cosine;
}

// ------------------------------------------------------------- fast path

/**
 * Similarity of one row against every other row of the same index.
 *
 * `primary` maps a row id to the sparse list of positions it rated;
 * `cross` is the transpose, mapping a position to the rows that rated it.
 * For user-based CF, primary = userIndex and cross = itemIndex; for item-based
 * CF the two swap, which is the whole difference between the two algorithms.
 *
 * The loop visits only entries that are actually rated, so its cost is
 *   sum over positions p rated by x of (number of rows that rated p),
 * instead of (number of rows) x (vector length). For the `mean` strategy the
 * missing entries still have to count, but their contribution is a closed form
 * built from the per-row totals in `stats`, so no scan is needed for them.
 *
 * Returns a Float32Array indexed by row id; entry x itself is left at 0.
 */
function similaritiesAgainstAll(primary, cross, x, stats, options) {
    const settings = Object.assign(
        { strategy: 'weighted', gamma: DEFAULT_GAMMA }, options || {});
    const rows = primary.length;

    const dot = new Float64Array(rows);
    const sqX = new Float64Array(rows);     // sum of x's squared values over co-rated
    const sqY = new Float64Array(rows);     // sum of y's squared values over co-rated
    const sumX = new Float64Array(rows);    // sum of x's values over co-rated
    const sumY = new Float64Array(rows);    // sum of y's values over co-rated
    const shared = new Int32Array(rows);

    const own = primary[x];
    if (own) {
        for (let p = 0; p < own.ids.length; p++) {
            const position = own.ids[p];
            const valueX = own.values[p];
            const peers = cross[position];
            if (!peers) continue;
            for (let q = 0; q < peers.ids.length; q++) {
                const y = peers.ids[q];
                const valueY = peers.values[q];
                dot[y] += valueX * valueY;
                sqX[y] += valueX * valueX;
                sqY[y] += valueY * valueY;
                sumX[y] += valueX;
                sumY[y] += valueY;
                shared[y]++;
            }
        }
    }

    const out = new Float32Array(rows);
    const meanX = stats.mean[x];
    const totalX = stats.sum[x];
    const countX = stats.count[x];
    const sqTotalX = stats.sumSquares[x];
    const length = stats.length;          // number of positions in a full vector

    for (let y = 0; y < rows; y++) {
        if (y === x) continue;
        if (settings.strategy !== 'mean') {
            if (shared[y] === 0 || sqX[y] === 0 || sqY[y] === 0) continue;
            let similarity = dot[y] / Math.sqrt(sqX[y] * sqY[y]);
            if (settings.strategy === 'weighted') {
                similarity *= Math.min(shared[y], settings.gamma) / settings.gamma;
            }
            out[y] = similarity;
            continue;
        }

        // mean imputation, in closed form:
        //   dot = (co-rated part) + meanY * (x's total outside the overlap)
        //                         + meanX * (y's total outside the overlap)
        //                         + meanX * meanY * (positions neither rated)
        const countY = stats.count[y];
        if (countX === 0 && countY === 0) continue;
        const meanY = stats.mean[y];
        const neither = length - countX - countY + shared[y];
        const full = dot[y]
            + meanY * (totalX - sumX[y])
            + meanX * (stats.sum[y] - sumY[y])
            + meanX * meanY * neither;
        const normX = Math.sqrt(sqTotalX + meanX * meanX * (length - countX));
        const normY = Math.sqrt(stats.sumSquares[y] + meanY * meanY * (length - countY));
        if (normX === 0 || normY === 0) continue;
        out[y] = full / (normX * normY);
    }

    return out;
}

// ------------------------------------------------------------ user-based

/**
 * User-Based CF: "users similar to you liked these items".
 *
 * Compares the active user's rating vector against every other user (the
 * columns of the rating matrix), keeps the N most similar with positive
 * similarity, and predicts each unseen movie as the similarity-weighted
 * average of their ratings.
 *
 * `support` — how many of the N neighbours actually rated the movie — is
 * returned alongside the score, because a prediction backed by one neighbour
 * is not the same claim as one backed by fifteen.
 */
function getUserBasedRecommendations(model, activeUserId, options) {
    const settings = Object.assign({
        strategy: 'weighted', gamma: DEFAULT_GAMMA,
        neighbours: DEFAULT_NEIGHBOURS, topK: DEFAULT_TOP_K, minSupport: 1
    }, options || {});

    const similarities = similaritiesAgainstAll(
        model.userIndex, model.itemIndex, activeUserId, model.userStats, settings);

    const neighbours = [];
    for (let v = 1; v < similarities.length; v++) {
        if (v !== activeUserId && similarities[v] > 0) {
            neighbours.push({ id: v, similarity: similarities[v] });
        }
    }
    neighbours.sort((p, q) => q.similarity - p.similarity || p.id - q.id);
    const selected = neighbours.slice(0, settings.neighbours);

    const seen = model.userIndex[activeUserId];
    const alreadyRated = new Set(seen ? Array.from(seen.ids) : []);

    const numerator = new Float64Array(model.numMovies + 1);
    const denominator = new Float64Array(model.numMovies + 1);
    const support = new Int32Array(model.numMovies + 1);

    for (const neighbour of selected) {
        const rated = model.userIndex[neighbour.id];
        if (!rated) continue;
        for (let p = 0; p < rated.ids.length; p++) {
            const movieId = rated.ids[p];
            if (alreadyRated.has(movieId)) continue;
            numerator[movieId] += neighbour.similarity * rated.values[p];
            denominator[movieId] += neighbour.similarity;
            support[movieId]++;
        }
    }

    const scored = [];
    for (let movieId = 1; movieId <= model.numMovies; movieId++) {
        if (denominator[movieId] === 0 || support[movieId] < settings.minSupport) continue;
        scored.push({
            id: movieId,
            score: numerator[movieId] / denominator[movieId],
            support: support[movieId]
        });
    }

    return {
        top: rankAndDecorate(model, scored, settings.topK),
        neighbours: selected.length,
        bestSimilarity: selected.length > 0 ? selected[0].similarity : 0,
        candidates: scored.length
    };
}

// ------------------------------------------------------------ item-based

/**
 * Item-Based CF: "items similar to the ones you liked".
 *
 * Compares movie rating columns instead of user rows. For every movie the
 * active user rated, the similarity of that movie to all others is computed and
 * accumulated into the candidates, weighted by what the user gave it.
 *
 * This is the expensive direction on MovieLens 100k: there are 1,682 movies
 * against 943 users, so the item-item comparison space is roughly 3.2x larger.
 */
function getItemBasedRecommendations(model, activeUserId, options) {
    const settings = Object.assign({
        strategy: 'weighted', gamma: DEFAULT_GAMMA,
        topK: DEFAULT_TOP_K, minSupport: 1
    }, options || {});

    const rated = model.userIndex[activeUserId];
    if (!rated || rated.ids.length === 0) {
        return { top: [], ratedMovies: 0, comparisons: 0, candidates: 0 };
    }

    const alreadyRated = new Set(Array.from(rated.ids));
    const numerator = new Float64Array(model.numMovies + 1);
    const denominator = new Float64Array(model.numMovies + 1);
    const support = new Int32Array(model.numMovies + 1);
    let comparisons = 0;

    for (let p = 0; p < rated.ids.length; p++) {
        const likedId = rated.ids[p];
        const userRating = rated.values[p];
        const similarities = similaritiesAgainstAll(
            model.itemIndex, model.userIndex, likedId, model.itemStats, settings);
        comparisons += similarities.length;

        for (let movieId = 1; movieId <= model.numMovies; movieId++) {
            const similarity = similarities[movieId];
            if (similarity <= 0 || alreadyRated.has(movieId)) continue;
            numerator[movieId] += similarity * userRating;
            denominator[movieId] += similarity;
            support[movieId]++;
        }
    }

    const scored = [];
    for (let movieId = 1; movieId <= model.numMovies; movieId++) {
        if (denominator[movieId] === 0 || support[movieId] < settings.minSupport) continue;
        scored.push({
            id: movieId,
            score: numerator[movieId] / denominator[movieId],
            support: support[movieId]
        });
    }

    return {
        top: rankAndDecorate(model, scored, settings.topK),
        ratedMovies: rated.ids.length,
        comparisons: comparisons,
        candidates: scored.length
    };
}

// ----------------------------------------------------------------- shared

/**
 * Sort by predicted score, then by how many ratings back it up, then by id.
 * Predicted ratings land on a coarse grid (a single neighbour rating a film 5
 * predicts exactly 5.000), so without an explicit tie-break the visible Top-5
 * would depend on array order.
 */
function rankAndDecorate(model, scored, topK) {
    scored.sort((p, q) =>
        q.score - p.score || q.support - p.support || p.id - q.id);
    return scored.slice(0, topK).map(entry => {
        const movie = model.movieById.get(entry.id);
        return {
            id: entry.id,
            title: movie ? movie.title : `Movie ${entry.id}`,
            genres: movie ? movie.genres : [],
            score: entry.score,
            support: entry.support,
            ratingCount: movie ? movie.ratingCount : 0
        };
    });
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        cosineSimilarity, similaritiesAgainstAll,
        getUserBasedRecommendations, getItemBasedRecommendations,
        MISSING_VALUE_STRATEGIES, DEFAULT_GAMMA, DEFAULT_NEIGHBOURS
    };
}
