/* UI wiring: the user dropdown, the two recommendation columns and the
 * per-column diagnostics (neighbours, evidence and wall-clock cost).
 *
 * The missing-value strategy in force is whatever #strategy says; the declared
 * default is `weighted`, documented at the top of cf.js. One strategy is used
 * for a whole run — the selector switches it, it never mixes them.
 */

window.onload = async function () {
    const status = document.getElementById('status');
    try {
        status.textContent = 'Loading movie data…';
        status.className = 'loading';

        await loadData();
        populateUserDropdown();

        const sparsity = 100 * model.totalRatings / (model.numUsers * model.numMovies);
        status.textContent = `Loaded ${model.numUsers} users × ${model.numMovies} movies = ` +
            `${model.totalRatings.toLocaleString('en-US')} ratings. The matrix is ` +
            `${sparsity.toFixed(1)}% full, so ${(100 - sparsity).toFixed(1)}% of it is the ` +
            `missing values both columns have to cope with.`;
        status.className = 'success';

        setMessage('user-based-result', 'Data loaded. Select a user.');
        setMessage('item-based-result', 'Data loaded. Select a user.');
    } catch (error) {
        console.error('Initialization error:', error);
        // data.js has already written the message into #status
    }
};

// Populate the user dropdown with one option per user id found in u.data
function populateUserDropdown() {
    const selectElement = document.getElementById('user-select');
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    for (let userId = 1; userId <= numUsers; userId++) {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId} — ${model.userStats.count[userId]} ratings`;
        selectElement.appendChild(option);
    }
}

function getRecommendations() {
    const status = document.getElementById('status');
    const userId = parseInt(document.getElementById('user-select').value, 10);

    if (isNaN(userId)) {
        setMessage('user-based-result', 'Please select a user first.');
        setMessage('item-based-result', 'Please select a user first.');
        return;
    }

    const options = {
        strategy: document.getElementById('strategy').value,
        minSupport: parseInt(document.getElementById('min-support').value, 10)
    };
    const rated = model.userStats.count[userId];

    if (rated === 0) {
        const message = `User ${userId} has rated nothing. Collaborative filtering has no ` +
            `signal to work from — this is the cold-start case, and no similarity can be computed.`;
        setMessage('user-based-result', message);
        setMessage('item-based-result', message);
        document.getElementById('user-based-diagnostics').textContent = '';
        document.getElementById('item-based-diagnostics').textContent = '';
        return;
    }

    // A · user-based: compare the active user against every other user.
    let started = performance.now();
    const userBased = getUserBasedRecommendations(model, userId, options);
    const userMs = performance.now() - started;

    // B · item-based: compare every movie the user rated against every other movie.
    started = performance.now();
    const itemBased = getItemBasedRecommendations(model, userId, options);
    const itemMs = performance.now() - started;

    document.getElementById('user-based-subtitle').textContent =
        `Because users similar to User ${userId} liked them — ` +
        `${userBased.neighbours} neighbours out of ${model.numUsers - 1} candidates.`;
    document.getElementById('item-based-subtitle').textContent =
        `Because User ${userId} liked ${itemBased.ratedMovies} movies — each compared ` +
        `against all ${model.numMovies}.`;

    // The two columns average over different things, so the evidence badge is
    // named after what it actually counts in each.
    renderList('user-based-result', userBased.top, 'neighbour');
    renderList('item-based-result', itemBased.top, 'of your movies');

    document.getElementById('user-based-diagnostics').textContent =
        `${userBased.candidates.toLocaleString('en-US')} candidate movies scored in ` +
        `${userMs.toFixed(0)} ms. Strongest neighbour similarity ` +
        `${userBased.bestSimilarity.toFixed(3)}.`;
    document.getElementById('item-based-diagnostics').textContent =
        `${itemBased.candidates.toLocaleString('en-US')} candidate movies scored in ` +
        `${itemMs.toFixed(0)} ms, after ${itemBased.comparisons.toLocaleString('en-US')} ` +
        `movie-to-movie comparisons — ${(itemMs / Math.max(userMs, 0.001)).toFixed(0)}× ` +
        `the user-based cost.`;

    const strategy = MISSING_VALUE_STRATEGIES[options.strategy];
    status.textContent = `User ${userId} rated ${rated} movies. Missing values: ` +
        `${strategy.label.toLowerCase()} (${strategy.note}). A prediction is shown only when at ` +
        `least ${options.minSupport} rating${options.minSupport === 1 ? '' : 's'} backs it.`;
    status.className = 'success';
}

// ----------------------------------------------------------------- rendering

function setMessage(elementId, message) {
    document.getElementById(elementId).innerHTML = `<p>${message}</p>`;
}

/**
 * Render a list of recommendations. Each row carries the predicted score and
 * the evidence behind it, because "5.000 from one neighbour" and "4.2 from
 * fourteen" look identical once you print only the number.
 */
function renderList(elementId, items, supportNoun) {
    const element = document.getElementById(elementId);

    if (!items || items.length === 0) {
        element.innerHTML = '<p>No recommendation passes the evidence threshold for this user.</p>';
        return;
    }

    const list = document.createElement('ol');
    list.className = 'recommendations';
    for (const item of items) {
        const row = document.createElement('li');

        const title = document.createElement('span');
        title.className = 'rec-title';
        title.textContent = item.title;

        const score = document.createElement('span');
        score.className = 'rec-score';
        score.textContent = item.score.toFixed(3);

        const meta = document.createElement('span');
        meta.className = 'rec-meta';
        meta.textContent = `${item.genres.length > 0 ? item.genres.join(', ') : 'no genre listed'} ` +
            `· ${item.ratingCount} rating${item.ratingCount === 1 ? '' : 's'} in the dataset`;

        const evidence = document.createElement('span');
        evidence.className = item.support < 5 ? 'badge thin' : 'badge solid';
        evidence.textContent = supportNoun === 'neighbour'
            ? `${item.support} neighbour${item.support === 1 ? '' : 's'}`
            : `${item.support} of your movies`;
        evidence.title = 'How many ratings this prediction is averaged over';

        row.append(title, score, meta, evidence);
        list.appendChild(row);
    }

    element.innerHTML = '';
    element.appendChild(list);
}
