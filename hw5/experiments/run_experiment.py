"""Offline evaluation for A05, driven through the real page.

The model is trained by `hw5/script.js` in a headless browser, so every number
reported here is produced by the submitted code rather than by a Python
re-implementation. The script serves `hw5/`, trains once with the page's own
default hyper-parameters, and records:

  * the 11 self-checks,
  * the training curve, the selected checkpoint, test RMSE and the baseline,
  * an independent hand-check of one prediction and its squared error,
  * Top-10 lists for two contrasting users and their rating history,
  * Recall@10 with its exclusions,
  * the familiar-vs-new analysis over every evaluable user,
  * what the regulariser suggested in the week5 readme does instead,
  * batch-size cost and the tensor-leak check.

Run:  python experiments/run_experiment.py
"""
from __future__ import annotations

import functools
import http.server
import json
import math
import socket
import socketserver
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
APP = HERE.parent
OUT_JSON = HERE / "results.json"

# The hyper-parameters index.html ships with; §7.4 of the week5 readme asks for
# the numbers produced by these.
DEFAULTS = {"epochs": 20, "lr": 0.01, "reg": 0.05, "batchSize": 64}


def serve(directory: Path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{port}"


# --------------------------------------------------------------- in-page scripts

TRAIN = """
(opts) => {
  const t0 = performance.now();
  initializeFactors(trainSet, NUM_FACTORS);
  const history = trainMF(trainSet, valSet, opts);
  const seconds = (performance.now() - t0) / 1000;
  const best = selectBestCheckpoint(history);
  return {
    seconds,
    history,
    best,
    testRMSE: computeRMSEFor(testSet),
    valRMSE: computeRMSEFor(valSet),
    trainRMSE: computeRMSEFor(trainSet),
    baseline: {
      train: computeBaselineRMSE(trainSet),
      val: computeBaselineRMSE(valSet),
      test: computeBaselineRMSE(testSet),
    },
    globalMean: model.globalMean,
    liveTensors: tf.memory().numTensors,
    backend: tf.getBackend(),
  };
}
"""

# A second training loop that keeps everything the same except the penalty: it
# uses the L2 term exactly as sketched in week5/readme.md §4.2, which squares the
# WHOLE factor matrices on every mini-batch.
TRAIN_GLOBAL_L2 = """
(opts) => {
  initializeFactors(trainSet, NUM_FACTORS);
  const n = trainSet.length;
  const size = Math.min(opts.batchSize, n);
  const optimizer = tf.train.adam(opts.lr);
  const history = [];
  const order = Int32Array.from({ length: n }, (_, k) => k);
  const t0 = performance.now();
  for (let epoch = 1; epoch <= opts.epochs; epoch++) {
    for (let start = 0; start < n; start += size) {
      const count = Math.min(size, n - start);
      const u = new Int32Array(count), i = new Int32Array(count), y = new Float32Array(count);
      for (let k = 0; k < count; k++) {
        const r = trainSet[order[start + k]];
        u[k] = r.userId; i[k] = r.movieId; y[k] = r.rating;
      }
      const uIdx = tf.tensor1d(u, "int32"), iIdx = tf.tensor1d(i, "int32"), yT = tf.tensor1d(y);
      optimizer.minimize(() => tf.tidy(() => {
        const uu = tf.gather(model.userFactors, uIdx);
        const ii = tf.gather(model.itemFactors, iIdx);
        const pred = tf.sum(uu.mul(ii), 1).add(model.globalMean);
        const mse = tf.losses.meanSquaredError(yT, pred);
        const l2 = model.userFactors.square().sum()
          .add(model.itemFactors.square().sum()).mul(opts.reg);
        return mse.add(l2);
      }), false, [model.userFactors, model.itemFactors]);
      tf.dispose([uIdx, iIdx, yT]);
    }
    history.push({ epoch, trainRMSE: computeRMSEFor(trainSet), valRMSE: computeRMSEFor(valSet) });
  }
  const seconds = (performance.now() - t0) / 1000;
  optimizer.dispose();
  const factorNorm = tf.tidy(() =>
    Math.sqrt(model.itemFactors.square().sum().dataSync()[0] / model.itemFactors.shape[0]));
  return { seconds, history, testRMSE: computeRMSEFor(testSet), factorNorm };
}
"""

FACTOR_NORM = """
() => tf.tidy(() =>
  Math.sqrt(model.itemFactors.square().sum().dataSync()[0] / model.itemFactors.shape[0]))
"""

# Everything the familiar-vs-new analysis needs, computed from the page's own
# recommendTopK over every user that Recall@10 reports on.
ANALYSIS = """
(K) => {
  const trainCount = new Map();      // movieId -> ratings in the training split
  const byUser = new Map();          // userId -> training ratings
  for (const r of trainSet) {
    trainCount.set(r.movieId, (trainCount.get(r.movieId) || 0) + 1);
    if (!byUser.has(r.userId)) byUser.set(r.userId, []);
    byUser.get(r.userId).push(r);
  }
  const genresOf = new Map(movies.map((m) => [m.id, m.genres]));
  const popularity = Array.from(trainCount.values()).sort((a, b) => b - a);
  const headSize = Math.max(1, Math.round(0.2 * popularity.length));
  const headCut = popularity[headSize - 1];

  const recall = recallAtK(K);
  const rows = [];
  const recCounts = new Map();

  for (const entry of recall.perUser) {
    const userId = entry.userId;
    const own = byUser.get(userId) || [];
    if (own.length === 0) continue;

    // The user's taste as the training split shows it: how often each genre
    // appears among the movies they rated 4 or 5.
    const liked = own.filter((r) => r.rating >= 4);
    const genreLiked = new Map();
    const genreAny = new Set();
    for (const r of own) for (const g of genresOf.get(r.movieId) || []) genreAny.add(g);
    for (const r of liked) {
      for (const g of genresOf.get(r.movieId) || []) {
        genreLiked.set(g, (genreLiked.get(g) || 0) + 1);
      }
    }
    const topGenres = Array.from(genreLiked.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 3)
      .map(([g]) => g);

    const recs = recommendTopK(userId, K);
    let familiar = 0, brandNew = 0, headItems = 0, popSum = 0;
    for (const rec of recs) {
      recCounts.set(rec.movieId, (recCounts.get(rec.movieId) || 0) + 1);
      const gs = genresOf.get(rec.movieId) || [];
      if (gs.some((g) => topGenres.includes(g))) familiar++;
      if (gs.length > 0 && gs.every((g) => !genreAny.has(g))) brandNew++;
      const pop = trainCount.get(rec.movieId) || 0;
      popSum += pop;
      if (pop >= headCut) headItems++;
    }
    const ownPop = own.reduce((s, r) => s + (trainCount.get(r.movieId) || 0), 0) / own.length;

    rows.push({
      userId,
      trainRatings: own.length,
      likedRatings: liked.length,
      topGenres,
      distinctGenresRated: genreAny.size,
      recommended: recs.length,
      familiarGenreHits: familiar,
      unseenGenreItems: brandNew,
      headItems,
      meanRecPopularity: recs.length ? popSum / recs.length : 0,
      meanOwnPopularity: ownPop,
      relevant: entry.relevant,
      hit: entry.hit,
    });
  }

  const mean = (f) => (rows.length ? rows.reduce((s, r) => s + f(r), 0) / rows.length : 0);
  const counts = Array.from(recCounts.values());
  const total = counts.reduce((s, c) => s + c, 0);
  const sorted = counts.slice().sort((a, b) => a - b);
  let gini = 0;
  if (total > 0) {
    const distinctItems = trainCount.size;
    const padded = new Array(Math.max(0, distinctItems - sorted.length)).fill(0).concat(sorted);
    const n = padded.length;
    const sum = padded.reduce((s, c) => s + c, 0);
    let acc = 0;
    for (let k = 0; k < n; k++) acc += (2 * (k + 1) - n - 1) * padded[k];
    gini = sum > 0 ? acc / (n * sum) : 0;
  }

  return {
    K,
    users: rows.length,
    headCut,
    catalogueItems: trainCount.size,
    distinctRecommended: recCounts.size,
    coveragePct: trainCount.size ? (100 * recCounts.size) / trainCount.size : 0,
    gini,
    meanFamiliarPct: 10 * mean((r) => r.familiarGenreHits),
    meanUnseenGenrePct: 10 * mean((r) => r.unseenGenreItems),
    meanHeadPct: 10 * mean((r) => r.headItems),
    meanRecPopularity: mean((r) => r.meanRecPopularity),
    meanOwnPopularity: mean((r) => r.meanOwnPopularity),
    perUser: rows,
  };
}
"""

# Recall@K and the familiar/new mix when candidates must carry at least
# `minCount` training ratings. This re-scores with the SAME trained factors; it
# is analysis code, deliberately kept out of the submitted script.js.
EVIDENCE_SWEEP = """
([K, minCount]) => {
  const trainCount = new Map();
  for (const r of trainSet) trainCount.set(r.movieId, (trainCount.get(r.movieId) || 0) + 1);
  const genresOf = new Map(movies.map((m) => [m.id, m.genres]));
  const byUser = new Map();
  for (const r of trainSet) {
    if (!byUser.has(r.userId)) byUser.set(r.userId, []);
    byUser.get(r.userId).push(r);
  }

  const candidates = new Set();
  for (const [id, c] of trainCount) if (c >= minCount) candidates.add(id);

  const relevantByUser = new Map();
  for (const r of testSet) {
    if (r.rating < 4.0 || !candidates.has(r.movieId)) continue;
    if (!relevantByUser.has(r.userId)) relevantByUser.set(r.userId, new Set());
    relevantByUser.get(r.userId).add(r.movieId);
  }

  const itemRows = model.itemFactors.shape[0];
  const f = model.userFactors.shape[1];
  let recallSum = 0, users = 0, popSum = 0, familiarSum = 0, shown = 0;
  const recommended = new Set();

  for (const [userId, relevant] of Array.from(relevantByUser).sort((a, b) => a[0] - b[0])) {
    const seen = new Set();
    for (const r of trainSet) if (r.userId === userId) seen.add(r.movieId);
    for (const r of valSet) if (r.userId === userId) seen.add(r.movieId);
    const mask = new Float32Array(itemRows);
    let eligible = 0;
    for (const id of candidates) {
      if (id < itemRows && !seen.has(id)) { mask[id] = 1; eligible++; }
    }
    if (eligible === 0) continue;

    const picked = tf.tidy(() => {
      const row = model.userFactors.slice([userId, 0], [1, f]);
      const raw = model.itemFactors.matMul(row, false, true).squeeze([1]).add(model.globalMean);
      const keep = tf.tensor1d(mask, "float32");
      const masked = raw.mul(keep).add(tf.scalar(1).sub(keep).mul(-1e9));
      const top = tf.topk(masked, Math.min(K, eligible));
      return Array.from(top.indices.dataSync());
    }).filter((id) => mask[id] === 1);

    const own = byUser.get(userId) || [];
    const liked = own.filter((r) => r.rating >= 4);
    const genreLiked = new Map();
    for (const r of liked) {
      for (const g of genresOf.get(r.movieId) || []) genreLiked.set(g, (genreLiked.get(g) || 0) + 1);
    }
    const topGenres = Array.from(genreLiked.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 3).map(([g]) => g);

    let hit = 0;
    for (const id of picked) {
      recommended.add(id);
      popSum += trainCount.get(id) || 0;
      shown++;
      if ((genresOf.get(id) || []).some((g) => topGenres.includes(g))) familiarSum++;
      if (relevant.has(id)) hit++;
    }
    recallSum += hit / relevant.size;
    users++;
  }

  return {
    minCount,
    candidatePool: candidates.size,
    users,
    recall: users ? recallSum / users : 0,
    meanRecPopularity: shown ? popSum / shown : 0,
    familiarPct: shown ? (100 * familiarSum) / shown : 0,
    distinctRecommended: recommended.size,
    coveragePct: candidates.size ? (100 * recommended.size) / candidates.size : 0,
  };
}
"""

SHOWCASE = """
([userId, K]) => {
  const own = trainSet.concat(valSet).filter((r) => r.userId === userId);
  const genresOf = new Map(movies.map((m) => [m.id, m.genres]));
  const trainCount = new Map();
  for (const r of trainSet) trainCount.set(r.movieId, (trainCount.get(r.movieId) || 0) + 1);
  const top = own.slice().sort((a, b) => b.rating - a.rating || a.movieId - b.movieId).slice(0, 8);
  return {
    userId,
    ratedInTrainVal: own.length,
    meanRating: own.reduce((s, r) => s + r.rating, 0) / Math.max(1, own.length),
    favourites: top.map((r) => ({
      movieId: r.movieId,
      title: movieTitle(r.movieId),
      rating: r.rating,
      genres: genresOf.get(r.movieId) || [],
    })),
    recommendations: recommendTopK(userId, K).map((rec) => ({
      ...rec,
      genres: genresOf.get(rec.movieId) || [],
      trainRatings: trainCount.get(rec.movieId) || 0,
      alsoInTest: testSet.some((r) => r.userId === userId && r.movieId === rec.movieId),
    })),
    leakCheck: recommendTopK(userId, K).every(
      (rec) => !own.some((r) => r.movieId === rec.movieId)),
  };
}
"""


def main():
    httpd, base_url = serve(APP)
    results: dict = {}
    print(f"serving {APP} at {base_url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            page = browser.new_page(viewport={"width": 1200, "height": 1000})
            page.set_default_timeout(3_600_000)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base_url}/index.html", wait_until="networkidle")
            page.wait_for_function(
                "() => typeof tf !== 'undefined' && trainSet.length > 0", timeout=120_000)

            # ---- dataset and split ------------------------------------------
            results["dataset"] = page.evaluate("""() => {
              const maxMovieId = movies.reduce((m, x) => Math.max(m, x.id), 0);
              const cells = (numUsers + 1) * (maxMovieId + 1);
              const trainItems = new Set(trainSet.map((r) => r.movieId));
              const trainUsers = new Set(trainSet.map((r) => r.userId));
              return {
                users: numUsers, movies: movies.length, maxMovieId,
                ratings: ratings.length,
                train: trainSet.length, val: valSet.length, test: testSet.length,
                densityPct: (100 * ratings.length) / (numUsers * movies.length),
                matrixCells: cells,
                splitExclusions: SPLIT_EXCLUSIONS.length,
                trainItems: trainItems.size, trainUsers: trainUsers.size,
                testItemsNotInTrain: new Set(testSet.map((r) => r.movieId))
                  .size - new Set(testSet.map((r) => r.movieId).filter((id) => trainItems.has(id))).size,
                backend: tf.getBackend(),
                tfVersion: tf.version.tfjs,
              };
            }""")
            print("dataset:", json.dumps(results["dataset"]))

            # ---- self-checks -------------------------------------------------
            checks = page.evaluate("() => runTests()")
            results["self_checks"] = {
                "passed": checks["passed"], "failed": checks["failed"],
                "pending": checks["pending"],
                "names": [f"{c['status']} {c['name']}" for c in checks["checks"]],
            }
            print(f"self-checks: {checks['passed']} passed, {checks['failed']} failed, "
                  f"{checks['pending']} pending")

            # ---- the headline run, with the page's own defaults ---------------
            started = time.time()
            run = page.evaluate(TRAIN, DEFAULTS)
            print(f"default run: {run['seconds']:.0f}s in page ({time.time()-started:.0f}s wall)")
            run["options"] = DEFAULTS
            run["factorNorm"] = page.evaluate(FACTOR_NORM)
            results["training"] = run
            print(f"  best epoch {run['best']['epoch']} val {run['best']['valRMSE']:.4f} | "
                  f"test {run['testRMSE']:.4f} vs baseline {run['baseline']['test']:.4f}")

            # What the same 20 epochs are worth WITHOUT checkpoint selection: an
            # empty validation split switches the snapshot off, so the factors
            # stay at the last epoch. Training is otherwise bit-identical.
            final_epoch = page.evaluate("""(opts) => {
              initializeFactors(trainSet, NUM_FACTORS);
              const history = trainMF(trainSet, [], opts);
              return { testRMSE: computeRMSEFor(testSet), epochs: history.length };
            }""", DEFAULTS)
            results["training"]["finalEpochTestRMSE"] = final_epoch["testRMSE"]
            print(f"  without checkpoint selection (epoch {final_epoch['epochs']}): "
                  f"test {final_epoch['testRMSE']:.4f}")
            # restore the selected-checkpoint model for everything that follows
            page.evaluate(TRAIN, DEFAULTS)

            # ---- hand-check of one prediction --------------------------------
            probe = page.evaluate("""() => {
              const r = testSet[0];
              const f = model.userFactors.shape[1];
              const u = Array.from(model.userFactors.slice([r.userId, 0], [1, f]).dataSync());
              const i = Array.from(model.itemFactors.slice([r.movieId, 0], [1, f]).dataSync());
              return { rating: r, userRow: u, itemRow: i,
                       globalMean: model.globalMean, predicted: predictRating(r.userId, r.movieId) };
            }""")
            dot = sum(a * b for a, b in zip(probe["userRow"], probe["itemRow"]))
            by_hand = min(5.0, max(1.0, probe["globalMean"] + dot))
            probe["handDot"] = dot
            probe["handPrediction"] = by_hand
            probe["agreement"] = abs(by_hand - probe["predicted"])
            probe["handSquaredError"] = (by_hand - probe["rating"]["rating"]) ** 2
            results["hand_check"] = probe
            print(f"  hand-check: predicted {probe['predicted']:.6f} vs hand {by_hand:.6f} "
                  f"(delta {probe['agreement']:.2e})")

            # ---- Recall@10 and the discovery analysis ------------------------
            analysis = page.evaluate(ANALYSIS, 10)
            recall = page.evaluate("(K) => recallAtK(K)", 10)
            results["recall"] = {
                "K": 10,
                "overallRecall": recall["overallRecall"],
                "usersReported": len(recall["perUser"]),
                "excludedUnseenIds": len(recall["excluded"]["unseenIds"]),
                "excludedUsersWithoutRelevant": len(recall["excluded"]["usersWithoutRelevant"]),
                "usersWithAtLeastOneHit": sum(1 for r in recall["perUser"] if r["hit"] > 0),
            }
            results["analysis"] = analysis
            print(f"  recall@10 {recall['overallRecall']:.4f} over "
                  f"{len(recall['perUser'])} users; familiar {analysis['meanFamiliarPct']:.1f}%, "
                  f"head {analysis['meanHeadPct']:.1f}%, coverage {analysis['coveragePct']:.1f}%")

            # ---- two contrasting users for the report ------------------------
            rows = sorted(analysis["perUser"], key=lambda r: r["trainRatings"])
            eligible = [r for r in rows if r["relevant"] >= 3]
            picks = [eligible[-1]["userId"], eligible[0]["userId"]] if len(eligible) >= 2 else \
                    [rows[-1]["userId"], rows[0]["userId"]]
            results["showcase"] = [page.evaluate(SHOWCASE, [uid, 10]) for uid in picks]
            for case in results["showcase"]:
                print(f"  user {case['userId']}: {case['ratedInTrainVal']} rated, "
                      f"no leak = {case['leakCheck']}")

            # ---- what the readme's regulariser does --------------------------
            compare_opts = {"epochs": 10, "lr": 0.01, "reg": 0.05, "batchSize": 1024}
            mine = page.evaluate(TRAIN, compare_opts)
            mine["factorNorm"] = page.evaluate(FACTOR_NORM)
            theirs = page.evaluate(TRAIN_GLOBAL_L2, compare_opts)
            results["regularisation"] = {
                "options": compare_opts,
                "batch_rows_l2": {
                    "testRMSE": mine["testRMSE"], "valRMSE": mine["best"]["valRMSE"],
                    "factorNorm": mine["factorNorm"], "seconds": mine["seconds"],
                },
                "whole_matrix_l2": {
                    "testRMSE": theirs["testRMSE"],
                    "valRMSE": min(h["valRMSE"] for h in theirs["history"]),
                    "factorNorm": theirs["factorNorm"], "seconds": theirs["seconds"],
                },
                "baselineTest": mine["baseline"]["test"],
            }
            print(f"  L2 on batch rows: test {mine['testRMSE']:.4f} | "
                  f"whole matrices: test {theirs['testRMSE']:.4f} "
                  f"(baseline {mine['baseline']['test']:.4f})")

            # ---- batch size cost and the leak check --------------------------
            cost = []
            for batch in (64, 256, 1024):
                one = page.evaluate(TRAIN, {"epochs": 1, "lr": 0.01, "reg": 0.05, "batchSize": batch})
                cost.append({
                    "batchSize": batch,
                    "steps": math.ceil(results["dataset"]["train"] / batch),
                    "secondsPerEpoch": one["seconds"],
                    "valRMSEAfter1Epoch": one["history"][0]["valRMSE"],
                    "liveTensors": one["liveTensors"],
                })
                print(f"  batch {batch}: {one['seconds']:.1f}s/epoch, "
                      f"{one['liveTensors']} live tensors")
            results["batch_cost"] = cost

            # Are the shipped defaults the right ones? Same code, same seed, a
            # larger batch and fewer epochs — measured on both metrics.
            alt_opts = {"epochs": 10, "lr": 0.01, "reg": 0.05, "batchSize": 1024}
            alt = page.evaluate(TRAIN, alt_opts)
            alt_recall = page.evaluate("(K) => recallAtK(K)", 10)
            results["alternative_config"] = {
                "options": alt_opts,
                "seconds": alt["seconds"],
                "bestEpoch": alt["best"]["epoch"],
                "valRMSE": alt["best"]["valRMSE"],
                "testRMSE": alt["testRMSE"],
                "recallAt10": alt_recall["overallRecall"],
                "usersReported": len(alt_recall["perUser"]),
            }
            print(f"  alt config (batch 1024, 10 epochs, {alt['seconds']:.0f}s): "
                  f"best epoch {alt['best']['epoch']} val {alt['best']['valRMSE']:.4f}, "
                  f"test {alt['testRMSE']:.4f}, recall@10 {alt_recall['overallRecall']:.4f}")

            # the selected-checkpoint model again, then the evidence sweep
            page.evaluate(TRAIN, DEFAULTS)
            sweep = [page.evaluate(EVIDENCE_SWEEP, [10, m]) for m in (1, 5, 20, 50, 100)]
            results["evidence_sweep"] = sweep
            for row in sweep:
                print(f"  candidates with >= {row['minCount']:3d} train ratings "
                      f"({row['candidatePool']:4d} movies): recall {row['recall']:.4f}, "
                      f"mean pop {row['meanRecPopularity']:6.1f}, familiar {row['familiarPct']:.1f}%")

            # how many titles the provided data layer mangles by decoding
            # ISO-8859-1 bytes as UTF-8 (data.js is provided and left unmodified)
            results["encoding_audit"] = page.evaluate("""() => {
              const bad = movies.filter((m) => m.title.includes("\uFFFD"));
              return { affectedTitles: bad.length, examples: bad.slice(0, 3).map((m) => m.title) };
            }""")
            print(f"  titles mangled by the UTF-8 decode: "
                  f"{results['encoding_audit']['affectedTitles']}")

            results["page_errors"] = errors
            browser.close()
    finally:
        httpd.shutdown()

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
