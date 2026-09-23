You are an expert full-stack web developer who creates robust, well-commented, and modular web applications using only vanilla HTML, CSS, and JavaScript.

Your task is to generate the complete code for a "Content-Based Movie Recommender" web application based on the detailed specifications below. The application logic will be split into three separate JavaScript files: `data.js` for data loading and parsing, `recommender.js` for the similarity and ranking maths, and `script.js` for UI logic. Please provide the code for each of the five files—`index.html`, `style.css`, `data.js`, `recommender.js`, and `script.js`—separately and clearly labeled.

---

### **Project Specification: Content-Based Movie Recommender (Modular)**

#### **1. Overall Goal**

Build a single-page web application that recommends movies. The application will use `data.js` to load and parse movie and rating data from local files (`u.item`, `u.data`). The `recommender.js` file holds pure scoring functions. The `script.js` file will then use this parsed data to populate the UI and calculate content-based recommendations using **cosine similarity** when a user makes a selection.

The app must show, side by side, the two ways of asking for recommendations, because comparing them is the point of the exercise:

-   **A — item-to-item:** similarity between one active movie and every candidate.
-   **B — profile-based:** similarity between the average of everything the user has watched and every candidate.

Both columns return a **Top-5**.

#### **2. File `index.html` - The Application Structure**

-   **DOCTYPE and Language:** The document should start with `<!DOCTYPE html>` and the `<html>` tag should specify `lang="en"`.
-   **Title:** The page title should be "Content-Based Movie Recommender".
-   **Main Heading:** Include an `<h1>` with the text "Content-Based Movie Recommender".
-   **Instructions:** Add a `<p>` tag explaining that the user enters what they watched and gets two Top-5 lists to compare.
-   **Watch history:** A `<div>` with the ID `watch-history` holding one row per watched movie. Each row has a radio button (name `active-movie`) marking the *active* item, a `<select class="movie-select">` populated dynamically, and a remove button. A button `#add-movie` appends a row. Start with three rows.
-   **Scoring controls:** A `<select id="metric">` offering `cosine` (default), `jaccard` and `overlap`, and a `<select id="tiebreak">` offering `quality` (default), `popularity` and `id`.
-   **Button:** Include a `<button id="recommend-btn">` with the text "Get Recommendations". When clicked, it must call the `getRecommendations()` JavaScript function.
-   **Result Display Area:** A `#status` paragraph for loading and error messages; a `#profile-panel` showing the profile vector; two result columns with `<ol id="item-results">` and `<ol id="profile-results">`, each followed by a `.diagnostics` paragraph; and a `<div id="result-box">` containing `<p id="result">` with the plain-text summary sentence.
-   **File Linking:** This is a critical step. At the end of the `<body>`, link to all three JavaScript files in this order, because each depends on the ones before it:
    ```
    <script src="data.js"></script>
    <script src="recommender.js"></script>
    <script src="script.js"></script>
    ```

#### **3. File `style.css` - The Application Design**

-   **Layout:** Create a professional, modern, and user-friendly layout. All content should be centered on the page within a main container of up to 900 px, wide enough for the two result columns.
-   **Background:** The `<body>` should have a light, neutral background color (e.g., `#f4f7f6`).
-   **Container:** The main container holding all elements should have a white background, rounded corners (`border-radius`), and a subtle box shadow to make it pop.
-   **Typography:** Use a clean, sans-serif font like 'Helvetica' or 'Arial'.
-   **Controls:** The `<select>` dropdowns and `<button>` should have consistent styling, with adequate padding and a clear visual hierarchy. The active history row is outlined in the accent colour.
-   **Button:** The button should be inviting, with a distinct background color (e.g., a shade of blue), white text, and a hover effect (e.g., slightly darker background) to indicate interactivity.
-   **Result Area:** The two result columns sit in a two-column grid that collapses to one column below 700 px. Each recommendation shows title, score, genres, rating count and a `head` / `long tail` badge. The profile vector is drawn as horizontal weight bars.

#### **4. File `data.js` - The Data Handling Module**

This file is responsible only for fetching and parsing the data from local files.

1.  **Global Variables:**
    -   Declare global `let` variables `movies`, `ratings` and `datasetStats`.

2.  **Primary Function: `loadData()`**
    -   This must be an `async` function.
    -   It will use the `fetch()` API to read `u.item` and `u.data`. Assume these files are in the same directory as `index.html`.
    -   Implement `try...catch` error handling to manage potential file loading failures. If a file fails to load, display an error message in `#status` that also reminds the reader that `fetch` needs an http server and will not work from `file://`.
    -   Inside the `try` block, `await` the fetch for `u.item`, pass the text to `parseItemData`, then do the same for `u.data` and `parseRatingData`, and finally call `computePopularity()`.

3.  **Parsing Function: `parseItemData(text)`**
    -   This function takes the raw text from `u.item` as input.
    -   It should define an array of the 18 genre names (from "Action" to "Western").
    -   Split the text on `/\r?\n/`, not on `'\n'`, and trim each line. The files are stored with LF, but a git checkout with `core.autocrlf=true` (the Windows default) rewrites them to CRLF, and the trailing `\r` would then corrupt the last field on every line — which is the Western genre flag.
    -   For each line, split by `|` and take the movie `id` (field 0) and `title` (field 1).
    -   **Fields 5 to 23 are 19 flags, not 18: `[unknown, Action, Adventure, …, Western]`.** The first flag is an "unknown genre" placeholder that has no name in the list of 18. Genre `i` is therefore flag `i + 1`. Do not zip the 18 names against all 19 flags — that shifts every label by one position (Toy Story comes out as a *Crime* film) and drops Western entirely.
    -   Build a `vector` of 18 numbers (1 / 0) as well as the human-readable `genres` array, and push `{ id, title, genres, vector, ratingCount, meanRating, quality, isLongTail }`.

4.  **Parsing Function: `parseRatingData(text)`**
    -   Split on `/\r?\n/` as above, split each line by `\t`, and push `{ userId, itemId, rating, timestamp }`.

5.  **Aggregation Function: `computePopularity()`**
    -   For every movie compute `ratingCount` and `meanRating` from `ratings`.
    -   Compute a shrunk `quality = (n·mean + m·globalMean) / (n + m)` with `m = 25`, so a movie with three ratings of 5.0 does not outrank a well-known favourite.
    -   Mark the 20% most-rated movies as the head (`isLongTail = false`) and the rest as the long tail.
    -   Fill `datasetStats` with the movie / rating / user counts and the share of ratings the head collects.

#### **5. File `recommender.js` - The Scoring Module**

Pure functions only: no DOM access, no reliance on globals, so the same code can be unit-tested outside the browser.

1.  `dotProduct(a, b)` — the unnormalised count of shared genres, kept as the "naive matching" baseline.
2.  `cosineSimilarity(a, b)` — `dotProduct(a, b) / (|a| · |b|)`, returning 0 when either vector is all zeros (two movies in `u.item` carry no genre at all). Dividing by the vector lengths is what stops a movie tagged with six genres from beating a focused match purely because it overlaps with everything.
3.  `jaccardSimilarity(a, b)` — `|intersection| / |union|` on binary vectors, kept so the starter's behaviour can be reproduced from the UI.
4.  `buildProfileVector(vectors)` — the element-wise **average** of the watched movies' genre vectors. Entry *g* is then the share of watched movies carrying genre *g*. Note in a comment that cosine is scale-invariant, so the mean and the sum rank identically.
5.  `recommend(movies, target, options)` — score every non-excluded movie against `target` with the chosen metric and return `{ top, tiedAtTop, candidates }`. Sorting must be **fully deterministic**: similarity descending, then the tie-break key descending, then movie id ascending. With 18 binary genres a large group of movies reaches the identical top score, so state in a comment that the tie-break, not the similarity, decides most of the visible list.

#### **6. File `script.js` - The UI Module**

1.  **Initialization:** `window.onload` runs an `async` function that awaits `loadData()`, builds the dropdown options once and clones them per row, seeds the history with three movies, marks the last one active, and writes a dataset summary into `#status`.
2.  **Watch history:** functions to add, remove and re-activate rows, and to read the selected ids without duplicates.
3.  **Core Logic: `getRecommendations()`**
    -   **Step 1:** Read the watch-history ids and the active id; show an error if the history is empty.
    -   **Step 2:** Read the metric and the tie-break from the two selects.
    -   **Step 3 (column A):** call `recommend()` with the *active movie's* vector as the target.
    -   **Step 4 (column B):** build the profile vector with `buildProfileVector()` and call `recommend()` with it as the target.
    -   Both columns exclude every movie already in the watch history, so the two lists are comparable and never recommend something the user just told you they watched.
    -   **Step 5:** Render each Top-5 with rank, title, score to three decimals, genres, rating count and the head / long-tail badge; draw the profile vector as weight bars; and write under each column how many candidates share the top score.
    -   **Step 6:** Write a plain summary into `#result`: "Because you liked '[Active Movie]', we recommend: … Because your profile averages …, we recommend: …".
4.  **Honest degradation:** Jaccard is defined on sets, so it cannot score a profile vector whose entries are fractions. When the user selects it, leave column B empty and say why, rather than silently binarising the profile.

---
Please now generate the complete code for the `index.html`, `style.css`, `data.js`, `recommender.js`, and `script.js` files based on these final, detailed specifications.
