"""Offline evaluation for A06, driven through the real page.

Everything is computed by the submitted `graph.js` / `script.js` in a headless
browser, so the report's numbers and the exported CSV/JSON files come from the
same code a grader runs. The script serves `hw6/`, builds the graph, selects
alpha on validation Recall@10, compares the four methods on the test split,
runs the one-factor-at-a-time demonstration, and writes:

  * `experiments/results.json`  — everything the report quotes,
  * `verification.json`, `results.csv`, `recommendations.csv` in `hw6/`
    (the deliverables required by readme section 9, produced by the page's own
    exporters).

Run:  python experiments/run_experiment.py
"""
from __future__ import annotations

import functools
import http.server
import json
import socket
import socketserver
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
APP = HERE.parent
OUT_JSON = HERE / "results.json"


def serve(directory: Path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{port}"


DATASET = """
() => {
  const graph = getGraph();
  const degrees = Array.from(graph.degree);
  const userDeg = degrees.slice(0, graph.nUsers);
  const movieDeg = degrees.slice(graph.nUsers);
  const mean = (xs) => xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
  return {
    users: numUsers, movies: movies.length, ratings: ratings.length,
    trainUsers: HW6_STATE.trainUserIds.length,
    candidateMovies: HW6_STATE.candidateIds.length,
    positivePairs: HW6_STATE.positivePairs,
    splitCounts: HW6_STATE.splitCounts || null,
    sparsityPct: (100 * ratings.length) / (numUsers * movies.length),
    graph: {
      nUsers: graph.nUsers, nMovies: graph.nMovies, nodeCount: graph.nodeCount,
      positiveEdges: graph.positiveEdges,
      zeroDegreeUsers: graph.zeroDegreeUsers.length,
      zeroDegreeMovies: graph.zeroDegreeMovies.length,
      meanUserDegree: mean(userDeg), meanMovieDegree: mean(movieDeg),
      maxUserDegree: Math.max(...userDeg), maxMovieDegree: Math.max(...movieDeg),
      graphDensityPct: (100 * graph.positiveEdges) / (graph.nUsers * graph.nMovies),
    },
    backend: tf.getBackend(),
    tfVersion: tf.version.tfjs,
    cohortTest: hw6SharedCohort("test").length,
    cohortVal: hw6SharedCohort("val").length,
    exclusionsTest: hw6CohortExclusions("test"),
    exclusionsVal: hw6CohortExclusions("val"),
  };
}
"""

# Agreement between the memoised all-seeds batch and the single-seed reference.
BATCH_CHECK = """
() => {
  const graph = getGraph();
  let worst = 0, checked = 0;
  for (const userId of HW6_STATE.trainUserIds.slice(0, 8)) {
    const node = graph.nodeOfUser(userId);
    const batched = personalizedPageRank(node, 0.85);
    const single = personalizedPageRank(node, 0.85, { noBatch: true });
    for (let i = 0; i < batched.p.length; i++) {
      worst = Math.max(worst, Math.abs(batched.p[i] - single.p[i]));
    }
    checked++;
  }
  return { worst, checked };
}
"""

# A hand-checkable propagation step on the 4-node reference fixture.
HAND_STEP = """
() => {
  const t = REFERENCE_GRAPH.transition;
  const n = t.Q.length;
  const v = new Array(n).fill(1 / n);
  const alpha = 0.85;
  const p0 = v.slice();
  const p1 = new Array(n).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) p1[j] += p0[i] * t.Q[i][j];
  for (let j = 0; j < n; j++) p1[j] = alpha * p1[j] + (1 - alpha) * v[j];
  const converged = pageRank(alpha, { transition: t });
  return { Q: t.Q, v, firstStep: p1, final: Array.from(converged.p),
           iterations: converged.iterations, residual: converged.residual,
           converged: converged.converged };
}
"""

MF_REFERENCE = """
async () => {
  await hw6EnsureMfForNeed();
  const history = MF_INFO.history || [];
  const first = history.length ? history[0] : null;
  const constant = history.length > 1 && history.every(
    (row) => row.trainRMSE === first.trainRMSE && row.valRMSE === first.valRMSE);
  const scored = mfScores(HW6_STATE.trainUserIds[0]);
  const values = HW6_STATE.candidateIds.map((id) => scored[id]).filter(Number.isFinite);
  const distinct = new Set(values.map((v) => v.toFixed(9)));
  const norms = HW6_MF_MODEL ? tf.tidy(() => ({
    user: Math.sqrt(HW6_MF_MODEL.userFactors.square().sum().dataSync()[0]
      / HW6_MF_MODEL.userFactors.shape[0]),
    item: Math.sqrt(HW6_MF_MODEL.itemFactors.square().sum().dataSync()[0]
      / HW6_MF_MODEL.itemFactors.shape[0]),
  })) : null;
  return {
    info: { ...MF_INFO, history: undefined },
    epochs: history.length,
    rmseConstantAcrossEpochs: constant,
    trainRMSE: first ? first.trainRMSE : null,
    valRMSE: first ? first.valRMSE : null,
    distinctCandidateScores: distinct.size,
    candidatesScored: values.length,
    scoreMin: values.length ? Math.min(...values) : null,
    scoreMax: values.length ? Math.max(...values) : null,
    globalMean: HW6_MF_MODEL ? HW6_MF_MODEL.globalMean : null,
    factorNorms: norms,
  };
}
"""

# The A05 cohort rule had no "at least one positive training edge" condition.
# If both counts agree, the previous assignment's MF number is comparable.
COHORT_COMPARABILITY = """
() => {
  const grades = HW6_STATE.gradesBySplit.test;
  let withoutRule = 0;
  HW6_STATE.trainUserIds.forEach((userId) => {
    const gradeMap = grades.get(userId);
    if (!gradeMap) return;
    const excluded = hw6ExclusionSet("test", userId);
    let relevant = 0;
    gradeMap.forEach((rating, movieId) => {
      if (rating >= 4 && HW6_STATE.candidateSet.has(movieId) && !excluded.has(movieId)) relevant++;
    });
    if (relevant > 0) withoutRule++;
  });
  return { withRule: hw6SharedCohort("test").length, withoutRule };
}
"""

POPULARITY_PROFILE = """
([methods, alpha]) => {
  const cohort = hw6SharedCohort("test");
  const out = {};
  for (const method of methods) {
    let sumPop = 0, lists = 0, items = 0, zeroScores = 0, shortLists = 0;
    const counts = new Map();
    for (const userId of cohort) {
      const recs = recommendTopK(method, userId, HW6_K, "test");
      if (recs.length < HW6_K) shortLists++;
      lists++;
      for (const rec of recs) {
        items++;
        sumPop += rec.popularity;
        if (rec.score === 0) zeroScores++;
        counts.set(rec.movieId, (counts.get(rec.movieId) || 0) + 1);
      }
    }
    const values = Array.from(counts.values()).sort((a, b) => a - b);
    const total = values.reduce((s, x) => s + x, 0);
    const pool = HW6_STATE.candidateIds.length;
    const padded = new Array(Math.max(0, pool - values.length)).fill(0).concat(values);
    let acc = 0;
    for (let k = 0; k < padded.length; k++) acc += (2 * (k + 1) - padded.length - 1) * padded[k];
    out[method] = {
      lists, items,
      meanPopularity: items ? sumPop / items : 0,
      zeroScoreItems: zeroScores,
      shortLists,
      distinctRecommended: counts.size,
      coveragePct: pool ? (100 * counts.size) / pool : 0,
      gini: total ? acc / (padded.length * total) : 0,
    };
  }
  return out;
}
"""

USER_PROFILE = """
(userId) => {
  const liked = [];
  const trainGrades = HW6_STATE.gradesBySplit.train.get(userId);
  if (trainGrades) trainGrades.forEach((rating, movieId) => {
    if (rating >= 4.0) liked.push({ movieId, rating, title: hw6Title(movieId),
                                    genres: hw6Genres(movieId),
                                    popularity: hw6PopularityOf(movieId) });
  });
  liked.sort((a, b) => b.popularity - a.popularity || a.movieId - b.movieId);
  const genreCount = new Map();
  for (const item of liked) for (const g of item.genres) genreCount.set(g, (genreCount.get(g) || 0) + 1);
  const topGenres = Array.from(genreCount.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 4);
  const testGrades = HW6_STATE.gradesBySplit.test.get(userId);
  let relevantTest = 0;
  if (testGrades) testGrades.forEach((rating, movieId) => {
    if (rating >= 4.0 && HW6_STATE.candidateSet.has(movieId)
        && !hw6ExclusionSet("test", userId).has(movieId)) relevantTest++;
  });
  return {
    userId,
    positiveTrainEdges: liked.length,
    topGenres: topGenres.map(([g, n]) => g + " (" + n + ")"),
    favourites: liked.slice(0, 6),
    relevantTestMovies: relevantTest,
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
            page = browser.new_page(viewport={"width": 1320, "height": 1000})
            page.set_default_timeout(3_600_000)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base_url}/index.html", wait_until="networkidle")
            page.wait_for_function(
                "() => typeof tf !== 'undefined' && HW6_STATE.splitReady", timeout=180_000)

            # ---- graph ------------------------------------------------------
            started = time.time()
            page.evaluate("() => hw6HandleBuildGraph()")
            build_seconds = time.time() - started
            results["dataset"] = page.evaluate(DATASET)
            results["dataset"]["buildSeconds"] = build_seconds
            g = results["dataset"]["graph"]
            print(f"graph: {g['nodeCount']} nodes ({g['nUsers']}+{g['nMovies']}), "
                  f"{g['positiveEdges']} edges, zero-degree {g['zeroDegreeUsers']}u/"
                  f"{g['zeroDegreeMovies']}m, built in {build_seconds:.1f}s")
            print(f"cohort: test {results['dataset']['cohortTest']}, "
                  f"val {results['dataset']['cohortVal']}")

            # ---- verification ------------------------------------------------
            checks = page.evaluate("() => runSelfChecks()")
            results["self_checks"] = {
                "passed": checks["passed"], "failed": checks["failed"],
                "pending": checks["pending"],
                "names": [f"{c['status']} {c['name']}" for c in checks["checks"]],
            }
            print(f"self-checks: {checks['passed']} passed, {checks['failed']} failed, "
                  f"{checks['pending']} pending")

            results["hand_step"] = page.evaluate(HAND_STEP)
            results["batch_agreement"] = page.evaluate(BATCH_CHECK)
            print(f"batched vs single-seed PPR: max |diff| = "
                  f"{results['batch_agreement']['worst']:.2e}")

            # ---- alpha selection on validation -------------------------------
            started = time.time()
            sweep = page.evaluate("() => hw6RunAlphaSweep()")
            sweep["seconds"] = time.time() - started
            results["alpha_sweep"] = sweep
            for row in sweep["rows"]:
                value = row["recall"]
                print(f"  alpha {row['alpha']:.2f}: validation Recall@10 "
                      f"{'n/a' if value is None else format(value, '.4f')} "
                      f"over {row['cohortSize']} users")
            selected = sweep["best"]
            print(f"selected alpha = {selected} ({sweep['seconds']:.0f}s)")

            # ---- the four-method comparison on test --------------------------
            page.evaluate("(a) => hw6SetAlpha(a)", selected)
            started = time.time()
            comparison = page.evaluate("async () => await hw6RunComparison('test')")
            comparison_seconds = time.time() - started
            results["comparison"] = comparison
            results["comparison_seconds"] = comparison_seconds
            results["mf_info"] = page.evaluate("() => MF_INFO")
            rows = comparison["rows"] if isinstance(comparison, dict) else comparison
            print(f"\ntest comparison at alpha={selected} ({comparison_seconds:.0f}s):")
            for row in rows:
                value = row.get("recall")
                print(f"  {row['method']:11s} Recall@10 "
                      f"{'n/a' if value is None else format(value, '.4f')}  "
                      f"mean popularity {row.get('meanPopularity')}")

            # ---- the one-factor-at-a-time demonstration ----------------------
            demo = page.evaluate("() => hw6RunTwoUserDemo()")
            results["two_user_demo"] = demo
            if demo.get("users"):
                results["user_profiles"] = [
                    page.evaluate(USER_PROFILE, demo["users"][0]),
                    page.evaluate(USER_PROFILE, demo["users"][1]),
                ]
                print(f"two-user demo: users {demo['users'][0]} and {demo['users'][1]}")

            # ---- how popular the recommended films are -----------------------
            results["popularity_profile"] = page.evaluate(
                POPULARITY_PROFILE, [["popularity", "pagerank", "ppr", "mf"], selected])
            for method, stats in results["popularity_profile"].items():
                print(f"  {method:11s} mean popularity {stats['meanPopularity']:7.1f}, "
                      f"coverage {stats['coveragePct']:5.1f}%, "
                      f"zero-score items {stats['zeroScoreItems']}")

            # Record a convergence run at the selected alpha so results.csv carries
            # the iteration count and residual the readme asks to report.
            results["convergence"] = page.evaluate("""(alpha) => {
              const graph = getGraph();
              hw6SetAlpha(alpha);
              hw6RunAndRecordConvergence("pagerank", graph, alpha);
              hw6RunAndRecordConvergence("ppr", graph, alpha, hw6SharedCohort("test")[0]);
              return HW6_STATE.convergence;
            }""", selected)
            for kind, record in results["convergence"].items():
                if record:
                    print(f"  {kind}: {record['iterations']} iterations, "
                          f"residual {record['residual']:.2e}, converged {record['converged']}")

            results["mf_reference"] = page.evaluate(MF_REFERENCE)
            mf = results["mf_reference"]
            print(f"  MF reference: {mf['epochs']} epochs, RMSE constant across them "
                  f"= {mf['rmseConstantAcrossEpochs']}, "
                  f"{mf['distinctCandidateScores']} distinct score(s) over "
                  f"{mf['candidatesScored']} candidates")

            results["cohort_comparability"] = page.evaluate(COHORT_COMPARABILITY)
            print(f"  cohort with/without the positive-edge rule: "
                  f"{results['cohort_comparability']['withRule']} / "
                  f"{results['cohort_comparability']['withoutRule']}")

            # ---- the readme section 9 deliverables ---------------------------
            exports = page.evaluate("""() => ({
              verification: serializeVerification(),
              results: serializeResults(),
              recommendations: serializeRecommendations(),
            })""")
            (APP / "verification.json").write_text(exports["verification"], encoding="utf-8")
            (APP / "results.csv").write_text(exports["results"], encoding="utf-8")
            (APP / "recommendations.csv").write_text(exports["recommendations"], encoding="utf-8")
            print(f"\nexports: verification.json "
                  f"({len(exports['verification'])}B), results.csv "
                  f"({len(exports['results'].splitlines())} rows), recommendations.csv "
                  f"({len(exports['recommendations'].splitlines())} rows)")

            results["tensors_live"] = page.evaluate("() => tf.memory().numTensors")
            results["page_errors"] = errors
            browser.close()
    finally:
        httpd.shutdown()

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
