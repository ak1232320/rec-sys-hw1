/* Similarity and ranking logic.
 *
 * Pure functions only: no DOM, no globals from data.js. Everything here works on
 * genre vectors (arrays of 18 numbers) so that the same code serves both
 * item-to-item recommendations (a binary movie vector) and profile-based
 * recommendations (a real-valued average of several movie vectors).
 *
 * Mirrored 1:1 by experiments/run_experiment.py, which is the offline reference.
 */

// ---------------------------------------------------------------- similarity

/** Dot product. This is the "naive matching" baseline: it just counts shared genres. */
function dotProduct(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
}

function norm(a) {
    return Math.sqrt(dotProduct(a, a));
}

/**
 * Cosine similarity: dot(a, b) / (|a| * |b|).
 * The division by the vector lengths is what stops movies with many genres from
 * winning just because they overlap with everything. Returns 0 for a zero vector
 * (2 movies in u.item carry no genre at all), where the cosine is undefined.
 */
function cosineSimilarity(a, b) {
    const denominator = norm(a) * norm(b);
    return denominator === 0 ? 0 : dotProduct(a, b) / denominator;
}

/**
 * Jaccard index on binary vectors: |intersection| / |union|. This is what the
 * starter used. It is only defined for sets, so it cannot score a profile vector
 * whose entries are fractions (see buildProfileVector).
 */
function jaccardSimilarity(a, b) {
    let intersection = 0;
    let union = 0;
    for (let i = 0; i < a.length; i++) {
        if (a[i] > 0 && b[i] > 0) intersection++;
        if (a[i] > 0 || b[i] > 0) union++;
    }
    return union === 0 ? 0 : intersection / union;
}

const SIMILARITY_METRICS = {
    cosine: { label: 'Cosine', fn: cosineSimilarity, supportsProfile: true },
    jaccard: { label: 'Jaccard (starter)', fn: jaccardSimilarity, supportsProfile: false },
    overlap: { label: 'Raw overlap (unnormalised)', fn: dotProduct, supportsProfile: true }
};

// ------------------------------------------------------------------- profile

/**
 * Average the genre vectors of the watched movies into one user profile vector.
 * Entry g of the result is the share of watched movies carrying genre g, so a
 * genre the user keeps picking outweighs a genre seen once. Cosine similarity is
 * scale invariant, so using the mean rather than the sum changes no ranking.
 */
function buildProfileVector(vectors) {
    if (vectors.length === 0) return [];
    const profile = new Array(vectors[0].length).fill(0);
    for (const vector of vectors) {
        for (let i = 0; i < vector.length; i++) profile[i] += vector[i];
    }
    return profile.map(value => value / vectors.length);
}

// ------------------------------------------------------------------- ranking

/**
 * Score every movie against `target` and return the Top-N.
 *
 * Genre vectors are coarse (18 binary flags, 1.72 of them set on average), so a
 * large group of movies reaches the very same top score. The ranking is therefore
 * made deterministic by an explicit tie-break, applied in this order:
 *   1. similarity, descending
 *   2. tie-break key, descending (default: shrunk average rating, i.e. quality)
 *   3. movie id, ascending
 * The tie-break is not cosmetic: it decides most of the visible list.
 */
function recommend(movies, target, options) {
    const settings = Object.assign({
        topN: 5,
        excludeIds: [],
        metric: 'cosine',
        tieBreak: movie => movie.quality
    }, options || {});

    const metric = SIMILARITY_METRICS[settings.metric];
    if (!metric) throw new Error(`Unknown similarity metric: ${settings.metric}`);
    const excluded = new Set(settings.excludeIds);

    const scored = [];
    for (const movie of movies) {
        if (excluded.has(movie.id)) continue;
        scored.push({ movie: movie, score: metric.fn(target, movie.vector) });
    }

    scored.sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        const tie = settings.tieBreak(b.movie) - settings.tieBreak(a.movie);
        if (tie !== 0) return tie;
        return a.movie.id - b.movie.id;
    });

    // How many candidates share the best score: the size of the group the
    // tie-break has to resolve.
    const best = scored.length > 0 ? scored[0].score : 0;
    let tiedAtTop = 0;
    for (const entry of scored) {
        if (entry.score !== best) break;
        tiedAtTop++;
    }

    return { top: scored.slice(0, settings.topN), tiedAtTop: tiedAtTop, candidates: scored.length };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        dotProduct, cosineSimilarity, jaccardSimilarity,
        buildProfileVector, recommend, SIMILARITY_METRICS
    };
}
