/* UI wiring: watch history, profile vector display and the two Top-5 columns. */

// Toy Story / Star Wars / Sense and Sensibility: three deliberately unrelated
// tastes, so the difference between the two columns is visible straight away.
const DEFAULT_HISTORY = [1, 50, 275];
const TOP_N = 5;

// How each tie-break option scores a movie (higher wins).
const TIE_BREAKS = {
    quality: movie => movie.quality,
    popularity: movie => movie.ratingCount,
    id: movie => -movie.id
};

let optionTemplate = null;   // one prepared <option> list, cloned per row
let activeRowId = null;      // which history row is the "active" movie
let nextRowId = 0;

window.onload = async function () {
    const status = document.getElementById('status');
    try {
        status.textContent = 'Loading movie data…';
        status.className = 'loading';

        await loadData();
        buildOptionTemplate();
        DEFAULT_HISTORY.forEach(id => addHistoryRow(id));
        setActiveRow(nextRowId - 1);   // the most recently added movie is the active one

        document.getElementById('add-movie').addEventListener('click', () => addHistoryRow());
        document.getElementById('metric').addEventListener('change', updateMetricHint);
        updateMetricHint();

        const stats = datasetStats;
        status.textContent = `Loaded ${stats.movies} movies, ${stats.ratings.toLocaleString('en-US')} ` +
            `ratings from ${stats.users} users. The top ${Math.round(100 * stats.headSize / stats.movies)}% ` +
            `most-rated movies collect ${(100 * stats.headShareOfRatings).toFixed(0)}% of all ratings.`;
        status.className = 'success';
        document.getElementById('result').textContent =
            'Data loaded. Pick your watched movies and press "Get Recommendations".';
    } catch (error) {
        console.error('Initialization error:', error);
        // data.js has already written the message into #status
    }
};

// ------------------------------------------------------------- watch history

function buildOptionTemplate() {
    optionTemplate = document.createDocumentFragment();
    const sorted = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    for (const movie of sorted) {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        optionTemplate.appendChild(option);
    }
}

function addHistoryRow(movieId) {
    const rowId = nextRowId++;
    const chosen = new Set(readHistoryIds());
    const fallback = movies.find(movie => !chosen.has(movie.id));
    const selectedId = movieId !== undefined ? movieId : (fallback ? fallback.id : movies[0].id);

    const row = document.createElement('div');
    row.className = 'history-row';
    row.dataset.rowId = rowId;

    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'active-movie';
    radio.title = 'Use this movie as the active item';
    radio.addEventListener('change', () => setActiveRow(rowId));

    const select = document.createElement('select');
    select.className = 'movie-select';
    select.appendChild(optionTemplate.cloneNode(true));
    select.value = selectedId;

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'remove-row';
    remove.textContent = '×';
    remove.title = 'Remove this movie';
    remove.addEventListener('click', () => removeHistoryRow(rowId));

    row.append(radio, select, remove);
    document.getElementById('watch-history').appendChild(row);

    if (activeRowId === null) setActiveRow(rowId);
    updateRemoveButtons();
    return rowId;
}

function removeHistoryRow(rowId) {
    const rows = [...document.querySelectorAll('.history-row')];
    if (rows.length <= 1) return;
    const row = rows.find(element => Number(element.dataset.rowId) === rowId);
    row.remove();
    if (activeRowId === rowId) {
        setActiveRow(Number(document.querySelector('.history-row').dataset.rowId));
    }
    updateRemoveButtons();
}

function setActiveRow(rowId) {
    activeRowId = rowId;
    for (const row of document.querySelectorAll('.history-row')) {
        const isActive = Number(row.dataset.rowId) === rowId;
        row.classList.toggle('active', isActive);
        row.querySelector('input[type="radio"]').checked = isActive;
    }
}

function updateRemoveButtons() {
    const rows = document.querySelectorAll('.history-row');
    for (const row of rows) {
        row.querySelector('.remove-row').disabled = rows.length <= 1;
    }
}

/** Ids currently selected, in row order, without duplicates. */
function readHistoryIds() {
    const ids = [];
    for (const select of document.querySelectorAll('.movie-select')) {
        const id = parseInt(select.value, 10);
        if (!Number.isNaN(id) && !ids.includes(id)) ids.push(id);
    }
    return ids;
}

function readActiveId() {
    const row = document.querySelector(`.history-row[data-row-id="${activeRowId}"]`);
    return row ? parseInt(row.querySelector('.movie-select').value, 10) : null;
}

function updateMetricHint() {
    const metric = document.getElementById('metric').value;
    const supportsProfile = SIMILARITY_METRICS[metric].supportsProfile;
    document.getElementById('profile-panel').classList.toggle('muted', !supportsProfile);
}

// ------------------------------------------------------------------ the core

function getRecommendations() {
    const status = document.getElementById('status');
    const resultElement = document.getElementById('result');

    const watchedIds = readHistoryIds();
    const activeId = readActiveId();
    const watched = watchedIds.map(id => movies.find(movie => movie.id === id)).filter(Boolean);
    const activeMovie = movies.find(movie => movie.id === activeId);

    if (watched.length === 0 || !activeMovie) {
        status.textContent = 'Please select at least one movie.';
        status.className = 'error';
        return;
    }

    const metricKey = document.getElementById('metric').value;
    const metric = SIMILARITY_METRICS[metricKey];
    const tieBreakKey = document.getElementById('tiebreak').value;
    const options = {
        topN: TOP_N,
        // Both columns exclude every watched movie, so the two lists are comparable.
        excludeIds: watchedIds,
        metric: metricKey,
        tieBreak: TIE_BREAKS[tieBreakKey]
    };

    // A · item-to-item: the active movie's own genre vector is the query.
    const itemResult = recommend(movies, activeMovie.vector, options);
    document.getElementById('item-subtitle').textContent =
        `Query: "${activeMovie.title}" — ${describeGenres(activeMovie.genres)}`;
    renderRecommendations('item-results', itemResult.top);
    renderDiagnostics('item-diagnostics', itemResult, metric.label);

    // B · profile-based: the average of every watched movie's vector is the query.
    const profileVector = buildProfileVector(watched.map(movie => movie.vector));
    renderProfile(profileVector, watched);
    if (metric.supportsProfile) {
        const profileResult = recommend(movies, profileVector, options);
        document.getElementById('profile-subtitle').textContent =
            `Query: the average of ${watched.length} watched movie${watched.length === 1 ? '' : 's'}`;
        renderRecommendations('profile-results', profileResult.top);
        renderDiagnostics('profile-diagnostics', profileResult, metric.label);
        resultElement.textContent =
            `Because you liked "${activeMovie.title}", we recommend: ` +
            `${itemResult.top.map(entry => entry.movie.title).join(', ')}. ` +
            `Because your profile averages ${watched.map(movie => `"${movie.title}"`).join(', ')}, ` +
            `we recommend: ${profileResult.top.map(entry => entry.movie.title).join(', ')}.`;
    } else {
        document.getElementById('profile-subtitle').textContent =
            `${metric.label} is defined on sets, not on vectors`;
        document.getElementById('profile-results').innerHTML = '';
        document.getElementById('profile-diagnostics').textContent =
            `A profile vector holds fractions such as 0.33, so it is not a set and ` +
            `${metric.label} cannot score it. Switch to cosine to use this column: handling a ` +
            `real-valued profile is exactly what the normalised dot product buys us.`;
        resultElement.textContent =
            `Because you liked "${activeMovie.title}", we recommend: ` +
            `${itemResult.top.map(entry => entry.movie.title).join(', ')}.`;
    }

    document.getElementById('results').hidden = false;
    document.getElementById('profile-panel').hidden = false;
    status.textContent = `Ranked ${itemResult.candidates} candidate movies with ` +
        `${metric.label.toLowerCase()} similarity, ties broken by ` +
        `${document.getElementById('tiebreak').selectedOptions[0].textContent.toLowerCase()}.`;
    status.className = 'success';
}

// ------------------------------------------------------------------ rendering

function describeGenres(genres) {
    return genres.length > 0 ? genres.join(', ') : 'no genre listed';
}

function renderRecommendations(elementId, entries) {
    const list = document.getElementById(elementId);
    list.innerHTML = '';
    for (const entry of entries) {
        const item = document.createElement('li');

        const title = document.createElement('span');
        title.className = 'rec-title';
        title.textContent = entry.movie.title;

        const score = document.createElement('span');
        score.className = 'rec-score';
        score.textContent = entry.score.toFixed(3);

        const meta = document.createElement('span');
        meta.className = 'rec-meta';
        meta.textContent = `${describeGenres(entry.movie.genres)} · ${entry.movie.ratingCount} ratings`;

        const badge = document.createElement('span');
        badge.className = entry.movie.isLongTail ? 'badge tail' : 'badge head';
        badge.textContent = entry.movie.isLongTail ? 'long tail' : 'head';

        item.append(title, score, meta, badge);
        list.appendChild(item);
    }
}

function renderDiagnostics(elementId, result, metricLabel) {
    const element = document.getElementById(elementId);
    const best = result.top.length > 0 ? result.top[0].score.toFixed(3) : '0';
    element.textContent = `${result.tiedAtTop} of ${result.candidates} candidates share the top ` +
        `${metricLabel.toLowerCase()} score of ${best}` +
        (result.tiedAtTop > TOP_N ? ', so the tie-break decides which five you see.' : '.');
}

function renderProfile(vector, watched) {
    document.getElementById('profile-formula').textContent =
        `Average of ${watched.length} genre vector${watched.length === 1 ? '' : 's'}: ` +
        `${watched.map(movie => `"${movie.title}"`).join(' + ')}, divided by ${watched.length}. ` +
        `A genre in every watched movie scores 1.00; one seen once scores ` +
        `${(1 / watched.length).toFixed(2)}.`;

    const container = document.getElementById('profile-bars');
    container.innerHTML = '';
    const entries = genreNames
        .map((name, index) => ({ name: name, weight: vector[index] }))
        .filter(entry => entry.weight > 0)
        .sort((a, b) => b.weight - a.weight || a.name.localeCompare(b.name));

    for (const entry of entries) {
        const row = document.createElement('div');
        row.className = 'profile-row';

        const label = document.createElement('span');
        label.className = 'profile-label';
        label.textContent = entry.name;

        const track = document.createElement('span');
        track.className = 'profile-track';
        const fill = document.createElement('span');
        fill.className = 'profile-fill';
        fill.style.width = `${(100 * entry.weight).toFixed(1)}%`;
        track.appendChild(fill);

        const value = document.createElement('span');
        value.className = 'profile-value';
        value.textContent = entry.weight.toFixed(2);

        row.append(label, track, value);
        container.appendChild(row);
    }
}
