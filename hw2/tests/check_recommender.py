"""Browser test for the A02 recommender.

Serves the hw2 folder over http, drives the real page in headless Chrome and checks
that what the UI shows equals the offline reference in experiments/results.json.
Also re-measures the starter's genre mapping to document the off-by-one bug.

Run:  python tests/check_recommender.py      (needs: pip install playwright)
"""
from __future__ import annotations

import functools
import http.server
import json
import socket
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
APP = HERE.parent
REFERENCE = json.loads((APP / "experiments" / "results.json").read_text(encoding="utf-8"))
OUT_JSON = HERE / "results.json"
OUT_LOG = HERE / "run_log.txt"
FIGURES = APP / "report" / "figures"

checks: list[dict] = []
log_lines: list[str] = []


def log(message: str) -> None:
    print(message)
    log_lines.append(message)


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    log(f"[{'PASS' if passed else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


# ------------------------------------------------------------------ http server

def serve(directory: Path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{port}"


# --------------------------------------------------------------- page helpers

def set_history(page, movie_ids, active_index=-1):
    """Make the watch history hold exactly movie_ids and mark one row active."""
    while page.locator(".history-row").count() > len(movie_ids):
        page.locator(".history-row .remove-row").last.click()
    while page.locator(".history-row").count() < len(movie_ids):
        page.click("#add-movie")
    for row, movie_id in enumerate(movie_ids):
        page.locator(".movie-select").nth(row).select_option(str(movie_id))
    index = active_index % len(movie_ids)
    page.locator('.history-row input[type="radio"]').nth(index).check()


def read_column(page, column):
    """Read one result column as [(title, score string)]."""
    rows = page.locator(f"#{column}-results li")
    out = []
    for i in range(rows.count()):
        out.append((
            rows.nth(i).locator(".rec-title").inner_text().strip(),
            rows.nth(i).locator(".rec-score").inner_text().strip(),
        ))
    return out


def expected_column(entries):
    return [(entry["title"], f"{entry['score']:.3f}") for entry in entries]


def recommend_now(page):
    page.click("#recommend-btn")
    page.wait_for_selector("#item-results li")


# ------------------------------------------------------------------- the test

def run(page, base_url):
    page.goto(f"{base_url}/index.html", wait_until="networkidle")
    page.wait_for_selector("#status.success", timeout=30_000)
    status = page.inner_text("#status")
    stats = REFERENCE["dataset"]
    check("data loads over http",
          f"{stats['movies']} movies" in status and "100,000 ratings" in status, status)

    # ---- 1. the genre off-by-one fix -----------------------------------------
    expected_genres = {
        1: ["Animation", "Children's", "Comedy"],
        50: ["Action", "Adventure", "Romance", "Sci-Fi", "War"],
        121: ["Action", "Sci-Fi", "War"],
    }
    actual = page.evaluate(
        "ids => ids.map(id => movies.find(m => m.id === id).genres)",
        list(expected_genres),
    )
    check("genres match u.genre order in the fixed app",
          actual == list(expected_genres.values()),
          f"Toy Story -> {actual[0]}")

    western_fixed = page.evaluate(
        "() => movies.filter(m => m.genres.includes('Western')).length")
    check("Western is no longer over-assigned",
          western_fixed == REFERENCE["genre_bug"]["western_movies_correct"],
          f"{western_fixed} movies (starter: {REFERENCE['genre_bug']['western_movies_starter']})")

    # ---- 2. Top-5 against the offline reference ------------------------------
    for movie_id, entries in REFERENCE["reference_top5"]["single_movie"].items():
        set_history(page, [int(movie_id)])
        recommend_now(page)
        item = read_column(page, "item")
        profile = read_column(page, "profile")
        check(f"item-to-item Top-5 for movie {movie_id} matches the offline reference",
              item == expected_column(entries), f"{[t for t, _ in item]}")
        check(f"a one-movie profile reproduces item-to-item for movie {movie_id}",
              profile == item)

    for combo, block in REFERENCE["reference_top5"]["history"].items():
        ids = [int(part) for part in combo.split("+")]
        set_history(page, ids)
        recommend_now(page)
        check(f"item-to-item Top-5 for history {combo} matches the offline reference",
              read_column(page, "item") == expected_column(block["item_to_item"]))
        check(f"profile-based Top-5 for history {combo} matches the offline reference",
              read_column(page, "profile") == expected_column(block["profile_based"]),
              f"{[entry['title'] for entry in block['profile_based']]}")
        shown = page.evaluate("() => Array.from(document.querySelectorAll('.profile-row'))"
                              ".map(r => [r.querySelector('.profile-label').textContent,"
                              " r.querySelector('.profile-value').textContent])")
        genre_names = page.evaluate("() => genreNames")
        expected_profile = sorted(
            [(name, f"{weight:.2f}") for name, weight in zip(genre_names, block["profile_vector"])
             if weight > 0],
            key=lambda pair: (-float(pair[1]), pair[0]))
        check(f"profile vector shown for history {combo} equals the averaged vector",
              [tuple(pair) for pair in shown] == expected_profile)

    # ---- 3. the tie-break switch --------------------------------------------
    for tiebreak, entries in REFERENCE["reference_top5"]["tiebreaks"].items():
        set_history(page, [1, 50, 275])
        page.select_option("#tiebreak", tiebreak)
        recommend_now(page)
        titles = [title for title, _ in read_column(page, "profile")]
        check(f"tie-break '{tiebreak}' reproduces the offline ranking",
              titles == [entry["title"] for entry in entries], f"{titles}")
    page.select_option("#tiebreak", "quality")

    # ---- 4. normalisation removes the multi-genre bias -----------------------
    set_history(page, [1, 50, 275])
    genres_per_rec = {}
    for metric in ("cosine", "overlap"):
        page.select_option("#metric", metric)
        recommend_now(page)
        genres_per_rec[metric] = page.evaluate(
            """() => {
                const titles = Array.from(document.querySelectorAll('#item-results .rec-title'))
                    .map(node => node.textContent);
                const picked = titles.map(t => movies.find(m => m.title === t));
                return picked.reduce((sum, m) => sum + m.genres.length, 0) / picked.length;
            }""")
    check("raw overlap recommends more multi-genre movies than cosine",
          genres_per_rec["overlap"] > genres_per_rec["cosine"],
          f"genres per recommendation: overlap {genres_per_rec['overlap']:.2f} "
          f"vs cosine {genres_per_rec['cosine']:.2f}")

    # ---- 5. Jaccard cannot score a profile vector ----------------------------
    page.select_option("#metric", "jaccard")
    recommend_now(page)
    check("Jaccard disables the profile column with an explanation",
          page.locator("#profile-results li").count() == 0
          and "cannot score it" in page.inner_text("#profile-diagnostics"))
    page.select_option("#metric", "cosine")

    # ---- 6. ties are reported to the user ------------------------------------
    set_history(page, [275])
    recommend_now(page)
    diagnostics = page.inner_text("#item-diagnostics")
    tied = int(diagnostics.split()[0])
    check("the app reports how many candidates tie at the top score",
          tied > 5 and "tie-break decides" in diagnostics, diagnostics)

    # ---- 7. screenshots of the fixed app for the report ----------------------
    set_history(page, [1, 50, 275])
    recommend_now(page)
    page.screenshot(path=str(FIGURES / "fixed_app.png"), full_page=True)
    page.locator("#results").screenshot(path=str(FIGURES / "comparison.png"))
    log(f"saved {FIGURES / 'fixed_app.png'} and comparison.png")

    # ---- 8. the same three movies in the starter -----------------------------
    page.goto(f"{base_url}/starter/index.html", wait_until="networkidle")
    page.wait_for_selector("#result.success", timeout=30_000)
    starter_genres = page.evaluate(
        "ids => ids.map(id => movies.find(m => m.id === id).genres)", list(expected_genres))
    check("the starter really does shift every genre label",
          starter_genres != list(expected_genres.values()),
          f"Toy Story -> {starter_genres[0]}")
    page.select_option("#movie-select", "275")
    page.click("#recommend-btn")
    page.wait_for_function(
        "() => document.getElementById('result').textContent.startsWith('Because')",
        timeout=15_000)
    starter_text = page.inner_text("#result")
    log(f"starter output: {starter_text}")
    page.locator(".container").screenshot(path=str(FIGURES / "starter_app.png"))
    log(f"saved {FIGURES / 'starter_app.png'}")
    return {"starter_output": starter_text,
            "starter_genres": dict(zip(map(str, expected_genres), starter_genres)),
            "genres_per_recommendation": genres_per_rec}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    httpd, base_url = serve(APP)
    log(f"serving {APP} at {base_url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            log(f"chrome {browser.version}, playwright {p.__class__.__module__.split('.')[0]}")
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.on("pageerror", lambda error: check("no uncaught page error", False, str(error)))
            extra = run(page, base_url)
            browser.close()
    finally:
        httpd.shutdown()

    passed = sum(1 for c in checks if c["passed"])
    log(f"\n{passed}/{len(checks)} checks passed")
    OUT_JSON.write_text(json.dumps(
        {"checks": checks, "passed": passed, "total": len(checks), **extra},
        indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    raise SystemExit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    main()
