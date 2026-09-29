/* Data loading, parsing and the rating structures for collaborative filtering. */

// Global data, filled in by loadData()
let movies = [];
let ratings = [];
let numUsers = 0;          // highest user id found in u.data
let numMovies = 0;         // number of parsed movies
let ratingMatrix = null;   // (numUsers + 1) x (numMovies + 1); 0 = "not rated"
let model = null;          // the sparse view of the same data, used by cf.js

// The 18 genre names, in the order they appear in u.item / u.genre.
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

async function loadData() {
    try {
        const itemResponse = await fetch('u.item');
        if (!itemResponse.ok) {
            throw new Error(`Failed to load movie data: ${itemResponse.status}`);
        }
        // u.item is ISO-8859-1, not UTF-8. Response.text() always decodes as UTF-8,
        // which turns every accented title into replacement characters
        // ("C'est arrive pres de chez vous" and 30 others). Decode it explicitly.
        parseItemData(new TextDecoder('iso-8859-1').decode(await itemResponse.arrayBuffer()));

        const dataResponse = await fetch('u.data');
        if (!dataResponse.ok) {
            throw new Error(`Failed to load rating data: ${dataResponse.status}`);
        }
        parseRatingData(await dataResponse.text());   // u.data is pure ASCII

        numUsers = ratings.reduce((max, rating) => Math.max(max, rating.userId), 0);
        numMovies = movies.length;
        buildRatingMatrix();
    } catch (error) {
        console.error('Error loading data:', error);
        const target = document.getElementById('status');
        if (target) {
            target.textContent = `Error: ${error.message}. u.item and u.data must be served ` +
                `over http:// (run "python -m http.server" in this folder), not opened from file://.`;
            target.className = 'error';
        }
        throw error;
    }
}

/**
 * Parse u.item: id | title | release | video release | IMDb url | 19 genre flags.
 *
 * The 19 flags are [unknown, Action, ..., Western]: one "unknown genre"
 * placeholder followed by the 18 named genres, so genre i is flag i + 1. Zipping
 * the 18 names against all 19 flags shifts every label by one position and drops
 * Western — the bug carried over from the week2 starter. Collaborative filtering
 * never reads the genres, but the app shows them next to each recommendation, so
 * they have to be right.
 *
 * Splitting on /\r?\n/ rather than '\n' matters for the same reason as in A02: a
 * git checkout with core.autocrlf=true rewrites the files to CRLF and the
 * trailing \r lands on the last genre flag.
 */
function parseItemData(text) {
    movies = [];
    for (const rawLine of text.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (line === '') continue;

        const fields = line.split('|');
        if (fields.length < 24) continue;

        const flags = fields.slice(5, 24);
        const genres = genreNames.filter((_, index) => flags[index + 1] === '1');

        movies.push({
            id: parseInt(fields[0], 10),
            title: fields[1],
            genres: genres,
            ratingCount: 0,
            meanRating: 0
        });
    }
}

function parseRatingData(text) {
    ratings = [];
    for (const rawLine of text.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (line === '') continue;

        const fields = line.split('\t');
        if (fields.length < 4) continue;

        ratings.push({
            userId: parseInt(fields[0], 10),
            itemId: parseInt(fields[1], 10),
            rating: parseFloat(fields[2]),
            timestamp: parseInt(fields[3], 10)
        });
    }
}

/**
 * Build the rating structures.
 *
 * `ratingMatrix` is the dense (numUsers + 1) x (numMovies + 1) grid the
 * specification asks for, indexed by raw id, with 0 meaning "not rated".
 * MovieLens ratings are 1-5, so 0 is unambiguous and no separate mask is needed.
 *
 * `model` is the same data as sparse per-row lists plus the per-row totals.
 * It holds no information the matrix does not, but it lets cf.js visit only
 * entries that exist: the dense form has 1.59M cells of which 100k are ratings,
 * so 94% of any full scan is spent on nothing. Item-based CF is unusable in a
 * browser without it (measured: 5.5 s per click dense, 0.05 s sparse).
 */
function buildRatingMatrix() {
    ratingMatrix = [];
    for (let userId = 0; userId <= numUsers; userId++) {
        ratingMatrix.push(new Float32Array(numMovies + 1));
    }

    const movieById = new Map(movies.map(movie => [movie.id, movie]));
    const perUser = Array.from({ length: numUsers + 1 }, () => []);
    const perItem = Array.from({ length: numMovies + 1 }, () => []);

    for (const entry of ratings) {
        if (entry.userId < 1 || entry.userId > numUsers) continue;
        if (!movieById.has(entry.itemId)) continue;
        ratingMatrix[entry.userId][entry.itemId] = entry.rating;
        perUser[entry.userId].push(entry.itemId);
        perItem[entry.itemId].push(entry.userId);
    }

    const userIndex = buildIndex(perUser, (userId, movieId) => ratingMatrix[userId][movieId]);
    const itemIndex = buildIndex(perItem, (movieId, userId) => ratingMatrix[userId][movieId]);

    for (const movie of movies) {
        const column = itemIndex[movie.id];
        movie.ratingCount = column ? column.ids.length : 0;
        movie.meanRating = column && column.ids.length > 0
            ? column.values.reduce((sum, value) => sum + value, 0) / column.ids.length
            : 0;
    }

    model = {
        numUsers: numUsers,
        numMovies: numMovies,
        movieById: movieById,
        ratingMatrix: ratingMatrix,
        userIndex: userIndex,
        itemIndex: itemIndex,
        // `length` is the number of real positions in a vector: column 0 of the
        // matrix is padding, not a movie anybody failed to rate.
        userStats: summarise(userIndex, numMovies),
        itemStats: summarise(itemIndex, numUsers),
        totalRatings: ratings.length,
        globalMean: ratings.length > 0
            ? ratings.reduce((sum, entry) => sum + entry.rating, 0) / ratings.length
            : 0
    };
}

/** Turn lists of ids into sorted typed arrays of (id, value) pairs. */
function buildIndex(lists, valueOf) {
    return lists.map((ids, rowId) => {
        const sorted = Int32Array.from(ids).sort();
        const values = new Float32Array(sorted.length);
        for (let k = 0; k < sorted.length; k++) values[k] = valueOf(rowId, sorted[k]);
        return { ids: sorted, values: values };
    });
}

/** Per-row count, sum, sum of squares and mean — everything the mean-imputation
 *  closed form in cf.js needs so that it never has to scan missing entries. */
function summarise(index, length) {
    const count = new Int32Array(index.length);
    const sum = new Float64Array(index.length);
    const sumSquares = new Float64Array(index.length);
    const mean = new Float64Array(index.length);

    for (let rowId = 0; rowId < index.length; rowId++) {
        const row = index[rowId];
        for (let k = 0; k < row.values.length; k++) {
            sum[rowId] += row.values[k];
            sumSquares[rowId] += row.values[k] * row.values[k];
        }
        count[rowId] = row.ids.length;
        mean[rowId] = count[rowId] > 0 ? sum[rowId] / count[rowId] : 0;
    }

    return { count, sum, sumSquares, mean, length };
}
