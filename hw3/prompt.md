You are an expert full-stack web developer who creates robust, well-commented, and modular web applications using only vanilla HTML, CSS, and JavaScript.

Your task is to generate the complete code for a "Collaborative Filtering Movie Recommender" web application based on the detailed specifications below. The application logic will be split into three separate JavaScript files: `data.js` for data loading and the rating structures, `cf.js` for the similarity and recommendation maths, and `script.js` for UI logic. Please provide the code for each of the five files—`index.html`, `style.css`, `data.js`, `cf.js`, and `script.js`—separately and clearly labeled.

---

### **Project Specification: Collaborative Filtering Movie Recommender (Modular)**

#### **1. Overall Goal**

Build a single-page web application that recommends movies using **collaborative filtering**. The application will use `data.js` to load and parse the same MovieLens 100K files as the previous exercise (`u.item`, `u.data`)—the dataset is deliberately unchanged so that the **algorithm** is the only thing that changes between the Content-Based assignment and this one.

Unlike the Content-Based version, which compared movie **genres**, this version uses the **rating patterns of users**. It must produce a Top-5 recommendation list **two ways** for the same active user—**User-Based CF** and **Item-Based CF**—so the two lists can be compared side by side.

Every recommendation must show, besides its predicted score, **how much evidence stands behind it**, and each column must report **what it cost to compute**. Both are needed for the analysis the assignment asks for, and neither is visible from a ranked list of titles alone.

#### **2. File `index.html` - The Application Structure**

-   **DOCTYPE and Language:** The document should start with `<!DOCTYPE html>` and the `<html>` tag should specify `lang="en"`.
-   **Title:** The page title should be "Collaborative Filtering Movie Recommender".
-   **Main Heading:** Include an `<h1>` with the text "Collaborative Filtering Movie Recommender".
-   **Instructions:** Add a `<p>` explaining that the user picks a user and receives two Top-5 lists, one per CF approach.
-   **User Dropdown:** A `<select>` with the ID `user-select`, populated with one option per user ID in `u.data`, each labelled with how many ratings that user has.
-   **Scoring Controls:** A `<select id="strategy">` offering `weighted` (default), `corated` and `mean`, and a `<select id="min-support">` offering 1 (default), 3 and 5 as the minimum number of ratings a prediction must rest on.
-   **Button:** A `<button>` with the text "Get Recommendations". When clicked, it must call the `getRecommendations()` JavaScript function.
-   **Status line:** A `<p id="status">` for loading, error and run-summary messages.
-   **Result Display Areas:** A `<div>` with the ID `result-box` holding two clearly labelled sections:
    -   `<div id="user-based-result">` — for the User-Based CF Top-5
    -   `<div id="item-based-result">` — for the Item-Based CF Top-5

    Each is preceded by a subtitle explaining what was compared and followed by a `.diagnostics` paragraph reporting the cost.
-   **File Linking:** Link the three scripts at the end of the `<body>`, in dependency order:
    ```
    <script src="data.js"></script>
    <script src="cf.js"></script>
    <script src="script.js"></script>
    ```

#### **3. File `style.css` - The Application Design**

-   **Layout:** Professional, modern and user-friendly, centred in a main container of up to 940 px.
-   **Background:** The `<body>` should have a light, neutral background color (e.g., `#f4f7f6`).
-   **Container:** White background, rounded corners, a subtle box shadow.
-   **Typography:** A clean, sans-serif font like 'Helvetica' or 'Arial'.
-   **Controls:** The three `<select>` elements sit in one row that collapses to a column below 720 px; the button spans the full width.
-   **Button:** Distinct background colour (e.g., a shade of blue), white text, hover effect.
-   **Result Areas:** Two columns on wide screens, stacked on narrow ones, each with a light background and a clear heading. Each recommendation shows title, predicted score, genres, how many ratings the movie has in the dataset, and an evidence badge — amber when the prediction rests on fewer than five ratings, green otherwise.

#### **4. File `data.js` - The Data Handling Module**

1.  **Global Variables:** `movies`, `ratings`, `numUsers`, `numMovies`, `ratingMatrix` and `model`.

2.  **Primary Function: `loadData()`**
    -   Must be `async`, use `fetch()` on `u.item` and `u.data`, and `try...catch` with the error written into `#status`, including the reminder that `fetch` needs an http server and will not work from `file://`.
    -   **`u.item` is ISO-8859-1, not UTF-8.** `Response.text()` always decodes as UTF-8 and turns every accented title into replacement characters. Read it with `arrayBuffer()` and decode it with `new TextDecoder('iso-8859-1')`. `u.data` is pure ASCII and needs no such care.
    -   After parsing: set `numUsers` (max user ID), `numMovies` (number of parsed movies) and call `buildRatingMatrix()`.

3.  **Parsing Function: `parseItemData(text)`**
    -   Split the text on `/\r?\n/`, not on `'\n'`, and trim each line: a git checkout with `core.autocrlf=true` rewrites the files to CRLF and the trailing `\r` would corrupt the last field of every line.
    -   Split each line by `|`, taking `id` (field 0) and `title` (field 1).
    -   **Fields 5 to 23 are 19 flags, not 18: `[unknown, Action, …, Western]`.** The first is an "unknown genre" placeholder with no name in the list of 18, so genre `i` is flag `i + 1`. Zipping the 18 names against all 19 flags shifts every label by one position (Toy Story comes out as a *Crime* film) and drops Western entirely.
    -   Push `{ id, title, genres, ratingCount, meanRating }`.

4.  **Parsing Function: `parseRatingData(text)`**
    -   Split on `/\r?\n/`, split each line by `\t`, push `{ userId, itemId, rating, timestamp }` as numbers.

5.  **Matrix Function: `buildRatingMatrix()`**
    -   Build the dense `(numUsers + 1) × (numMovies + 1)` grid indexed by raw id, where a missing rating is `0`. MovieLens ratings are 1-5, so `0` is unambiguous and no separate mask is needed. Store it in `ratingMatrix`.
    -   Also build `model`: the same data as sparse per-row lists (`userIndex`, `itemIndex`, each `{ids, values}` typed arrays) plus per-row `count`, `sum`, `sumSquares` and `mean` (`userStats`, `itemStats`). This holds no information the matrix does not, but the dense grid is 94% empty, so a full scan spends 94% of its time on nothing — item-based CF takes about 5 seconds per click over the dense matrix and about 0.3 seconds over the sparse one.
    -   Record `count` against the number of **real** positions: column 0 of the matrix is padding, not a movie anybody failed to rate.

#### **5. File `cf.js` - The Collaborative-Filtering Module**

Pure functions only: no DOM access, so the same code can be tested outside the browser.

1.  **Missing-value strategy.** State the choice in a comment at the top of the file. The declared default is **`weighted`**: cosine over co-rated entries, multiplied by `min(n, γ) / γ` with `γ = 25` (significance weighting). Two users who agree on two films score 1.0 under plain cosine, and two films are not evidence. The other two strategies — `corated` (plain co-rated cosine) and `mean` (missing entries replaced by the row mean, cosine over the full vector) — are implemented so the assignment's comparison can be reproduced from the UI, but one run uses exactly one strategy and never mixes them.

2.  `cosineSimilarity(a, b, options)` — the readable definition on two dense rating vectors, supporting all three strategies, with `from`/`length` bounding the real positions and a guard returning `0` when either norm is zero.

3.  `similaritiesAgainstAll(primary, cross, x, stats, options)` — one row against every other row, visiting only entries that were actually rated. `primary` maps a row to the positions it rated and `cross` is the transpose; for user-based CF they are `userIndex` and `itemIndex`, and for item-based CF the two swap — that swap is the entire difference between the two algorithms. Under `mean`, the missing part must still count, so express its contribution in closed form from the per-row totals rather than scanning it. This function must return exactly what `cosineSimilarity` would.

4.  `getUserBasedRecommendations(model, activeUserId, options)` — compare the active user against every other user, keep the `N = 20` most similar with positive similarity, and predict each unrated movie as the similarity-weighted average of their ratings. Return the Top-5 together with **how many of those neighbours actually rated each movie**.

5.  `getItemBasedRecommendations(model, activeUserId, options)` — for every movie the user rated, compute its similarity to all other movies and accumulate the candidates weighted by the user's rating. Return the Top-5 with **how many of the user's movies contributed** to each.

6.  **Deterministic ranking.** Predicted ratings land on a coarse grid — a single neighbour who rated a film 5 predicts exactly 5.000 — so sort by score descending, then by evidence descending, then by id ascending. Without an explicit tie-break the visible Top-5 depends on array order.

#### **6. File `script.js` - The UI Module**

1.  **Initialization:** `window.onload` runs an `async` function that awaits `loadData()`, calls `populateUserDropdown()`, and reports in `#status` how full the rating matrix is.
2.  **`getRecommendations()`:** reads the user, the strategy and the evidence threshold; times each column with `performance.now()`; renders both Top-5 lists; and writes per-column diagnostics — how many candidates were scored, in how many milliseconds, how strong the best neighbour was, and how many movie-to-movie comparisons the item-based column needed.
3.  **Cold start:** if the selected user has rated nothing, say so plainly rather than returning an empty list — collaborative filtering has no signal at all for such a user, and that is the point worth showing.
4.  The evidence badge counts different things in the two columns (neighbours on the left, the user's own movies on the right), so name it accordingly rather than printing a bare number.

#### **7. Notes on This Exercise**

-   Do **not** change the dataset files.
-   Keep the modular split; do not move logic between files.
-   The code must run from static hosting (GitHub Pages) with no build step and no external libraries. `fetch` will not read the data files from `file://`, so a local run needs `python -m http.server`.

---
Please now generate the complete code for the `index.html`, `style.css`, `data.js`, `cf.js`, and `script.js` files based on these final, detailed specifications.
