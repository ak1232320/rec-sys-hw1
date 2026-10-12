"""Browser test for the A05 matrix-factorization page.

Serves `hw5/`, drives the real page in headless Chrome and checks the six
implemented stubs beyond what the page's own 11 self-checks cover: that the run
is reproducible, that the model really is the selected checkpoint, that nothing
leaks between runs, and that one prediction agrees with a dot product computed
outside the browser.

Training here uses a short configuration (3 epochs, batch 1024) so the test runs
in under a minute; the reported headline numbers come from
`experiments/run_experiment.py`, which uses the page's own defaults.

Run:  python tests/check_mf.py      (needs: pip install playwright)
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
OUT_JSON = HERE / "results.json"
OUT_LOG = HERE / "run_log.txt"
FIGURES = APP / "report" / "figures"

SHORT = {"epochs": 3, "lr": 0.01, "reg": 0.05, "batchSize": 1024}

checks: list[dict] = []
log_lines: list[str] = []


def log(message: str) -> None:
    print(message)
    log_lines.append(message)


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    log(f"[{'PASS' if passed else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


def serve(directory: Path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{port}"


TRAIN = """
(opts) => {
  initializeFactors(trainSet, NUM_FACTORS);
  const history = trainMF(trainSet, valSet, opts);
  return {
    history,
    best: selectBestCheckpoint(history),
    valNow: computeRMSEFor(valSet),
    testNow: computeRMSEFor(testSet),
    tensors: tf.memory().numTensors,
  };
}
"""


def run(page, base_url):
    page.goto(f"{base_url}/index.html", wait_until="networkidle")
    page.wait_for_function(
        "() => typeof tf !== 'undefined' && trainSet.length > 0", timeout=120_000)

    backend = page.evaluate("() => tf.getBackend()")
    shape = page.evaluate(
        "() => [numUsers, movies.length, ratings.length, trainSet.length, valSet.length, testSet.length]")
    check("page loads the dataset and a TensorFlow.js backend",
          backend in ("webgl", "cpu") and shape[2] == 100_000,
          f"backend {backend}, {shape[0]} users x {shape[1]} movies, "
          f"split {shape[3]}/{shape[4]}/{shape[5]}")

    # ---- 1. the page's own self-checks --------------------------------------
    result = page.evaluate("() => runTests()")
    check("all 11 self-checks pass (0 pending stubs)",
          result["passed"] == 11 and result["failed"] == 0 and result["pending"] == 0,
          f"{result['passed']} passed, {result['failed']} failed, {result['pending']} pending")

    # ---- 2. the split is a per-user 80/10/10 by timestamp -------------------
    split = page.evaluate("""() => {
      const byUser = new Map();
      for (const r of ratings) byUser.set(r.userId, (byUser.get(r.userId) || 0) + 1);
      let okCounts = 0, okOrder = 0, users = 0;
      const lastTrain = new Map(), firstTest = new Map();
      for (const r of trainSet) lastTrain.set(r.userId, Math.max(lastTrain.get(r.userId) ?? -1, r.timestamp));
      for (const r of testSet) firstTest.set(r.userId, Math.min(firstTest.get(r.userId) ?? Infinity, r.timestamp));
      const trainN = new Map(), valN = new Map(), testN = new Map();
      for (const r of trainSet) trainN.set(r.userId, (trainN.get(r.userId) || 0) + 1);
      for (const r of valSet) valN.set(r.userId, (valN.get(r.userId) || 0) + 1);
      for (const r of testSet) testN.set(r.userId, (testN.get(r.userId) || 0) + 1);
      for (const [userId, n] of byUser) {
        users++;
        const nTrain = Math.floor(0.8 * n), nVal = Math.floor(0.1 * n);
        if ((trainN.get(userId) || 0) === nTrain && (valN.get(userId) || 0) === nVal
            && (testN.get(userId) || 0) === n - nTrain - nVal) okCounts++;
        if ((lastTrain.get(userId) ?? -1) <= (firstTest.get(userId) ?? Infinity)) okOrder++;
      }
      return { users, okCounts, okOrder, total: trainSet.length + valSet.length + testSet.length };
    }""")
    check("every user is split 80/10/10 with training strictly before test",
          split["okCounts"] == split["users"] and split["okOrder"] == split["users"]
          and split["total"] == 100_000,
          f"{split['okCounts']}/{split['users']} users, {split['total']} ratings kept")

    # ---- 3. the same settings reproduce the same run ------------------------
    first = page.evaluate(TRAIN, SHORT)
    second = page.evaluate(TRAIN, SHORT)
    same = all(
        abs(a["trainRMSE"] - b["trainRMSE"]) < 1e-12 and abs(a["valRMSE"] - b["valRMSE"]) < 1e-12
        for a, b in zip(first["history"], second["history"]))
    check("two runs with the same hyper-parameters produce the same curve",
          same, f"val RMSE {[round(h['valRMSE'], 6) for h in first['history']]}")

    # ---- 4. nothing leaks between runs --------------------------------------
    third = page.evaluate(TRAIN, SHORT)
    check("repeated training does not leak tensors",
          second["tensors"] == third["tensors"],
          f"live tensors {first['tensors']} -> {second['tensors']} -> {third['tensors']}")

    # ---- 5. the model IS the selected checkpoint ----------------------------
    best = third["best"]
    worst_epoch = max(third["history"], key=lambda h: h["valRMSE"])
    check("after training the model is the best-validation checkpoint, not the last epoch",
          abs(third["valNow"] - best["valRMSE"]) < 1e-6 and best["valRMSE"] <= worst_epoch["valRMSE"],
          f"epoch {best['epoch']} val {best['valRMSE']:.6f}, model now {third['valNow']:.6f}")

    # ---- 6. one prediction, recomputed outside the browser ------------------
    probe = page.evaluate("""() => {
      const r = testSet[0];
      const f = model.userFactors.shape[1];
      return {
        rating: r,
        userRow: Array.from(model.userFactors.slice([r.userId, 0], [1, f]).dataSync()),
        itemRow: Array.from(model.itemFactors.slice([r.movieId, 0], [1, f]).dataSync()),
        globalMean: model.globalMean,
        predicted: predictRating(r.userId, r.movieId),
      };
    }""")
    dot = sum(a * b for a, b in zip(probe["userRow"], probe["itemRow"]))
    by_hand = min(5.0, max(1.0, probe["globalMean"] + dot))
    check("predictRating equals the dot product recomputed in Python",
          abs(by_hand - probe["predicted"]) < 1e-5,
          f"user {probe['rating']['userId']}, movie {probe['rating']['movieId']}: "
          f"page {probe['predicted']:.6f} vs hand {by_hand:.6f}")

    # ---- 7. unknown ids fall back instead of throwing -----------------------
    fallback = page.evaluate("""() => ({
      bigUser: predictRating(10 ** 6, 1),
      bigMovie: predictRating(1, 10 ** 6),
      mean: model.globalMean,
      empty: recommendTopK(10 ** 6, 10).length,
    })""")
    check("out-of-range ids return the global mean and an empty list",
          abs(fallback["bigUser"] - fallback["mean"]) < 1e-9
          and abs(fallback["bigMovie"] - fallback["mean"]) < 1e-9
          and fallback["empty"] == 0,
          f"globalMean {fallback['mean']:.4f}")

    # ---- 8. recommendations never repeat what the user already rated -------
    leak = page.evaluate("""(K) => {
      const users = Array.from(new Set(trainSet.map((r) => r.userId))).slice(0, 40);
      let bad = 0, lists = 0, nonDecreasing = 0, outsideCatalogue = 0;
      const known = new Set(movies.map((m) => m.id));
      for (const userId of users) {
        const seen = new Set();
        for (const r of trainSet) if (r.userId === userId) seen.add(r.movieId);
        for (const r of valSet) if (r.userId === userId) seen.add(r.movieId);
        const recs = recommendTopK(userId, K);
        lists++;
        if (recs.length !== K) bad++;
        for (let i = 0; i < recs.length; i++) {
          if (seen.has(recs[i].movieId)) bad++;
          if (!known.has(recs[i].movieId)) outsideCatalogue++;
          if (i > 0 && recs[i - 1].score < recs[i].score) nonDecreasing++;
        }
      }
      return { lists, bad, nonDecreasing, outsideCatalogue };
    }""", 10)
    check("40 Top-10 lists exclude train/val items, stay in the catalogue and are sorted",
          leak["bad"] == 0 and leak["nonDecreasing"] == 0 and leak["outsideCatalogue"] == 0,
          f"{leak['lists']} lists checked")

    # ---- 9. Recall@10 bookkeeping adds up ----------------------------------
    recall = page.evaluate("""(K) => {
      const out = recallAtK(K);
      const testUsers = new Set(testSet.map((r) => r.userId));
      const trainItems = new Set(trainSet.map((r) => r.movieId));
      const unseenReallyUnseen = out.excluded.unseenIds.every((id) => !trainItems.has(id));
      const relevantAllCounted = out.perUser.every((r) => r.hit <= r.relevant && r.relevant > 0);
      const manual = out.perUser.length
        ? out.perUser.reduce((s, r) => s + r.hit / r.relevant, 0) / out.perUser.length
        : 0;
      return {
        overall: out.overallRecall,
        manual,
        reported: out.perUser.length,
        without: out.excluded.usersWithoutRelevant.length,
        testUsers: testUsers.size,
        unseenReallyUnseen,
        relevantAllCounted,
      };
    }""", 10)
    check("Recall@10 bookkeeping is complete and the mean matches a manual recomputation",
          recall["reported"] + recall["without"] == recall["testUsers"]
          and abs(recall["overall"] - recall["manual"]) < 1e-12
          and recall["unseenReallyUnseen"] and recall["relevantAllCounted"],
          f"{recall['reported']} reported + {recall['without']} without relevant = "
          f"{recall['testUsers']} test users; recall {recall['overall']:.4f}")

    # ---- 10. the starter still refuses to run ------------------------------
    page.goto(f"{base_url}/starter/index.html", wait_until="networkidle")
    page.wait_for_function(
        "() => typeof tf !== 'undefined' && trainSet.length > 0", timeout=120_000)
    starter = page.evaluate("() => runTests()")
    check("the untouched starter reports 4 passed / 7 pending",
          starter["passed"] == 4 and starter["pending"] == 7 and starter["failed"] == 0,
          f"{starter['passed']} passed, {starter['failed']} failed, {starter['pending']} pending")
    page.locator("#testLog").screenshot(path=str(FIGURES / "starter_checks.png"))

    # ---- panel screenshots of the finished page ----------------------------
    # The full page is 33,000 px tall (the recall panel lists every user), so the
    # report uses the two panels that carry the evidence.
    page.goto(f"{base_url}/index.html", wait_until="networkidle")
    page.wait_for_function(
        "() => typeof tf !== 'undefined' && trainSet.length > 0", timeout=120_000)
    page.evaluate(TRAIN, SHORT)
    page.evaluate("() => runTests()")
    page.locator("#testLog").screenshot(path=str(FIGURES / "fixed_checks.png"))
    page.evaluate("() => renderTopK(655)")
    page.locator("#recommendations").screenshot(path=str(FIGURES / "fixed_topk.png"))
    log(f"saved {FIGURES / 'fixed_checks.png'}, starter_checks.png and fixed_topk.png")

    return {
        "backend": backend,
        "selfChecks": result["passed"],
        "starterSelfChecks": {"passed": starter["passed"], "pending": starter["pending"]},
        "handCheck": {"page": probe["predicted"], "python": by_hand},
        "recall": recall,
    }


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    httpd, base_url = serve(APP)
    log(f"serving {APP} at {base_url}")
    extra = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            log(f"chrome {browser.version}")
            page = browser.new_page(viewport={"width": 1200, "height": 1400})
            page.set_default_timeout(900_000)
            page.on("pageerror", lambda e: check("no uncaught page error", False, str(e)))
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
