"""Browser test for the A03 collaborative-filtering recommender.

Serves the hw3 folder over http, drives the real page in headless Chrome and checks
that (a) the fast sparse similarity agrees with the readable dense definition in
cf.js, and (b) both Top-5 columns equal the offline reference in
experiments/results.json. Also records what the untouched starter does.

Run:  python tests/check_cf.py      (needs: pip install playwright)
"""
from __future__ import annotations

import functools
import http.server
import json
import re
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
    log(f"[{'PASS' if passed else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


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

def recommend(page, user_id, strategy, min_support):
    page.select_option("#user-select", str(user_id))
    page.select_option("#strategy", strategy)
    page.select_option("#min-support", str(min_support))
    page.click("#recommend-btn")
    page.wait_for_function(
        "() => document.querySelector('#user-based-result li, #user-based-result p') !== null")


def read_column(page, column):
    """Read one column as [(title, score string, support int)]."""
    return page.evaluate(
        """column => Array.from(document.querySelectorAll(`#${column}-result li`)).map(li => [
             li.querySelector('.rec-title').textContent,
             li.querySelector('.rec-score').textContent,
             parseInt(li.querySelector('.badge').textContent, 10)
           ])""", column)


def expected_column(entries):
    return [[e["title"], f"{e['score']:.3f}", e["support"]] for e in entries]


# ------------------------------------------------------------------- the test

def run(page, base_url):
    page.goto(f"{base_url}/index.html", wait_until="networkidle")
    page.wait_for_selector("#status.success", timeout=60_000)
    status = page.inner_text("#status")
    stats = REFERENCE["dataset"]
    check("data loads over http",
          f"{stats['users']} users" in status and "100,000 ratings" in status, status)

    # ---- 1. the genre off-by-one fix ----------------------------------------
    expected_genres = {1: ["Animation", "Children's", "Comedy"],
                       50: ["Action", "Adventure", "Romance", "Sci-Fi", "War"]}
    actual = page.evaluate("ids => ids.map(id => movies.find(m => m.id === id).genres)",
                           list(expected_genres))
    check("genres follow the u.genre order in the fixed app",
          actual == list(expected_genres.values()), f"Toy Story -> {actual[0]}")

    accented = page.evaluate("() => movies.find(m => m.id === 1104).title")
    check("accented titles survive the ISO-8859-1 decode",
          "�" not in accented and "arrivé" in accented, accented)

    # ---- 2. the fast sparse path equals the dense definition ----------------
    # This is the check that matters most: similaritiesAgainstAll() skips every
    # entry neither side rated, and for mean imputation replaces the missing part
    # with a closed form. Both must reproduce cosineSimilarity() exactly.
    for strategy in ("corated", "weighted", "mean"):
        worst = page.evaluate(
            """strategy => {
                let worst = 0, checked = 0;
                for (const kind of ['user', 'item']) {
                    const isUser = kind === 'user';
                    const primary = isUser ? model.userIndex : model.itemIndex;
                    const cross = isUser ? model.itemIndex : model.userIndex;
                    const stats = isUser ? model.userStats : model.itemStats;
                    const length = isUser ? model.numMovies : model.numUsers;
                    const x = isUser ? 7 : 50;
                    const fast = similaritiesAgainstAll(primary, cross, x, stats, {strategy});
                    const vector = id => isUser
                        ? model.ratingMatrix[id]
                        : Float32Array.from({length: model.numUsers + 1},
                                            (_, u) => model.ratingMatrix[u][id]);
                    const vx = vector(x);
                    for (let y = 1; y < fast.length; y += 29) {
                        if (y === x) continue;
                        const reference = cosineSimilarity(vx, vector(y), {
                            strategy, from: 1, length,
                            meanA: stats.mean[x], meanB: stats.mean[y]});
                        worst = Math.max(worst, Math.abs(reference - fast[y]));
                        checked++;
                    }
                }
                return [worst, checked];
            }""", strategy)
        check(f"sparse similarity equals the dense definition ({strategy})",
              worst[0] < 1e-5, f"max difference {worst[0]:.2e} over {worst[1]} pairs")

    # ---- 3. both columns against the offline reference ----------------------
    # `mean` is compared on similarity values only, not on list identity: mean
    # imputation squeezes every user pair into a band ~1e-4 wide (see check 4),
    # so which 20 neighbours land inside the cut is decided by float32 rounding
    # and legitimately differs between the browser and NumPy.
    mismatches = 0
    for key, block in REFERENCE["reference_top5"].items():
        user_id, strategy, min_support = key.split("/")
        if strategy == "mean":
            continue
        recommend(page, int(user_id), strategy, int(min_support))
        for column, entries in (("user-based", block["user_based"]),
                                ("item-based", block["item_based"])):
            actual = read_column(page, column)
            expected = expected_column(entries)
            ok = actual == expected
            if not ok:
                mismatches += 1
                log(f"    expected {expected}")
                log(f"    actual   {actual}")
            check(f"{column} Top-5 matches the offline reference for {key}", ok,
                  ", ".join(row[0] for row in actual) if ok else "")
    compared = sum(1 for k in REFERENCE["reference_top5"] if k.split("/")[1] != "mean") * 2
    log(f"reference comparison: {compared - mismatches} of {compared} columns matched "
        f"(the `mean` configurations are checked numerically instead)")

    # ---- 4. mean imputation barely discriminates between users --------------
    spreads = page.evaluate(
        """() => {
            const out = {};
            for (const strategy of ['corated', 'weighted', 'mean']) {
                const sims = similaritiesAgainstAll(model.userIndex, model.itemIndex, 1,
                                                    model.userStats, {strategy});
                const positive = [];
                for (let v = 1; v < sims.length; v++) if (v !== 1 && sims[v] > 0) positive.push(sims[v]);
                positive.sort((a, b) => b - a);
                out[strategy] = {top: positive[0], twentieth: positive[19],
                                 spread: positive[0] - positive[19]};
            }
            return out;
        }""")
    check("mean imputation collapses the similarity range to near-nothing",
          spreads["mean"]["spread"] < 1e-3
          and spreads["corated"]["spread"] > 10 * spreads["mean"]["spread"],
          f"spread across the top 20 neighbours: mean {spreads['mean']['spread']:.2e}, "
          f"co-rated {spreads['corated']['spread']:.2e} "
          f"({spreads['corated']['spread'] / spreads['mean']['spread']:.0f}x wider)")

    # ---- 5. mean imputation is the popularity-biased strategy ---------------
    popularity = {}
    for strategy in ("corated", "mean"):
        recommend(page, 405, strategy, 1)
        popularity[strategy] = page.evaluate(
            """() => {
                const titles = Array.from(document.querySelectorAll('#item-based-result .rec-title'))
                    .map(node => node.textContent);
                const picked = titles.map(t => movies.find(m => m.title === t));
                return picked.reduce((sum, m) => sum + m.ratingCount, 0) / picked.length;
            }""")
    check("mean imputation recommends far better-known movies than co-rated cosine",
          popularity["mean"] > 10 * popularity["corated"],
          f"mean {popularity['mean']:.0f} ratings vs co-rated {popularity['corated']:.0f}")

    # ---- 6. the evidence threshold is enforced ------------------------------
    recommend(page, 405, "weighted", 5)
    supports = [row[2] for column in ("user-based", "item-based")
                for row in read_column(page, column)]
    check("raising the evidence threshold removes thinly-backed predictions",
          all(value >= 5 for value in supports), f"evidence values {supports}")

    # ---- 7. the two directions report their cost ----------------------------
    recommend(page, 405, "weighted", 1)
    user_diag = page.inner_text("#user-based-diagnostics")
    item_diag = page.inner_text("#item-based-diagnostics")
    user_ms = int(re.search(r"in (\d+) ms", user_diag).group(1))
    item_ms = int(re.search(r"in (\d+) ms", item_diag).group(1))
    check("item-based is measurably the expensive direction",
          item_ms > user_ms, f"user-based {user_ms} ms vs item-based {item_ms} ms")

    page.screenshot(path=str(FIGURES / "fixed_app.png"), full_page=True)
    page.locator("#result-box").screenshot(path=str(FIGURES / "comparison.png"))
    log(f"saved {FIGURES / 'fixed_app.png'} and comparison.png")

    # ---- 8. the untouched starter ------------------------------------------
    page.goto(f"{base_url}/starter/index.html", wait_until="networkidle")
    page.wait_for_function(
        "() => document.getElementById('user-based-result').textContent.includes('Data loaded')",
        timeout=60_000)
    starter_genres = page.evaluate("() => movies.find(m => m.id === 1).genres")
    check("the starter still shifts every genre label",
          starter_genres != expected_genres[1], f"Toy Story -> {starter_genres}")
    page.select_option("#user-select", "405")
    page.click("#recommend-btn")
    page.wait_for_timeout(500)
    starter_text = page.inner_text("#user-based-result")
    check("the starter's TODO stubs return nothing",
          "Implement the TODO" in starter_text, starter_text.strip())
    page.locator(".container").screenshot(path=str(FIGURES / "starter_app.png"))
    log(f"saved {FIGURES / 'starter_app.png'}")

    return {"similarity_spreads": spreads,
            "starter_output": starter_text.strip(),
            "starter_genres": starter_genres,
            "item_based_mean_popularity": popularity,
            "browser_ms": {"user_based": user_ms, "item_based": item_ms}}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    httpd, base_url = serve(APP)
    log(f"serving {APP} at {base_url}")
    extra = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            log(f"chrome {browser.version}")
            page = browser.new_page(viewport={"width": 1100, "height": 1000})
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
