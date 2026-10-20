"""Browser test for the A06 Personalized PageRank page.

Serves `hw6/`, drives the real page in headless Chrome and checks the seven
implemented stubs beyond the page's own 12 self-checks: that the readme 4.1
reference vectors are reproduced, that mass is conserved and the zero-degree
policy is identical in PageRank and PPR, that the memoised all-seeds batch
equals the single-seed reference, that the split and the exclusion rules are
what the protocol says, and that the exporters produce the readme section 9
files.

Run:  python tests/check_ppr.py      (needs: pip install playwright)
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


def run(page, base_url):
    page.goto(f"{base_url}/index.html", wait_until="networkidle")
    page.wait_for_function(
        "() => typeof tf !== 'undefined' && HW6_STATE.splitReady", timeout=180_000)
    backend = page.evaluate("() => tf.getBackend()")
    page.evaluate("() => hw6HandleBuildGraph()")

    graph = page.evaluate("""() => {
      const g = getGraph();
      return { nUsers: g.nUsers, nMovies: g.nMovies, nodeCount: g.nodeCount,
               edges: g.positiveEdges, zeroU: g.zeroDegreeUsers.length,
               zeroM: g.zeroDegreeMovies.length,
               trainUsers: HW6_STATE.trainUserIds.length,
               candidates: HW6_STATE.candidateIds.length };
    }""")
    check("the graph has one node per observed training id and no padding node",
          graph["nodeCount"] == graph["nUsers"] + graph["nMovies"]
          and graph["nUsers"] == graph["trainUsers"]
          and graph["nMovies"] == graph["candidates"],
          f"{graph['nodeCount']} nodes = {graph['nUsers']} users + {graph['nMovies']} movies, "
          f"{graph['edges']} positive edges, backend {backend}")

    # ---- 1. the page's own self-checks --------------------------------------
    result = page.evaluate("() => runSelfChecks()")
    check("all 12 page self-checks pass (0 pending stubs)",
          result["passed"] == 12 and result["failed"] == 0 and result["pending"] == 0,
          f"{result['passed']} passed, {result['failed']} failed, {result['pending']} pending")

    # ---- 2. the readme 4.1 reference vectors --------------------------------
    reference = page.evaluate("""() => {
      const t = REFERENCE_GRAPH.transition;
      const uniform = pageRank(0.85, { transition: t });
      const seeded = personalizedPageRank(0, 0.85, { transition: t });
      return { uniform: Array.from(uniform.p), seeded: Array.from(seeded.p),
               uIter: uniform.iterations, sIter: seeded.iterations,
               uConv: uniform.converged, sConv: seeded.converged };
    }""")
    expected_uniform = [0.3725268513, 0.1958239118, 0.3941492369, 0.0375]
    expected_seeded = [0.4522328999, 0.1921989825, 0.3555681176, 0.0]
    worst_u = max(abs(a - b) for a, b in zip(reference["uniform"], expected_uniform))
    worst_s = max(abs(a - b) for a, b in zip(reference["seeded"], expected_seeded))
    check("the 4-node reference graph reproduces both readme 4.1 vectors",
          worst_u < 1e-6 and worst_s < 1e-6
          and reference["uConv"] and reference["sConv"],
          f"max |diff| uniform {worst_u:.2e} ({reference['uIter']} iterations), "
          f"restart-at-A {worst_s:.2e} ({reference['sIter']})")
    check("a node nothing points to keeps only its restart share",
          abs(reference["uniform"][3] - 0.0375) < 1e-9 and reference["seeded"][3] == 0.0,
          f"D = {reference['uniform'][3]:.6f} under uniform restart, "
          f"{reference['seeded'][3]:.6f} under restart at A")

    # ---- 3. mass conservation and the zero-degree policy --------------------
    mass = page.evaluate("""() => {
      const g = getGraph();
      const global_ = pageRank(0.85);
      const seedUser = g.nodeOfUser(HW6_STATE.trainUserIds[0]);
      const seeded = personalizedPageRank(seedUser, 0.85);
      const zeroUser = g.zeroDegreeUsers.length ? g.nodeOfUser(g.zeroDegreeUsers[0]) : -1;
      const danglingSeeded = zeroUser >= 0 ? personalizedPageRank(zeroUser, 0.85) : null;
      const sum = (p) => { let s = 0; for (let i = 0; i < p.length; i++) s += p[i]; return s; };
      const minimum = (p) => { let m = Infinity; for (let i = 0; i < p.length; i++) m = Math.min(m, p[i]); return m; };
      return {
        globalMass: sum(global_.p), seededMass: sum(seeded.p),
        danglingMass: danglingSeeded ? sum(danglingSeeded.p) : null,
        danglingUserId: zeroUser >= 0 ? g.zeroDegreeUsers[0] : null,
        minGlobal: minimum(global_.p), minSeeded: minimum(seeded.p),
        globalIter: global_.iterations, globalConv: global_.converged,
        globalResidual: global_.residual,
      };
    }""")
    masses = [mass["globalMass"], mass["seededMass"]]
    if mass["danglingMass"] is not None:
        masses.append(mass["danglingMass"])
    check("probability is conserved and nonnegative, including a zero-degree seed",
          all(abs(m - 1) < 1e-5 for m in masses)
          and mass["minGlobal"] >= 0 and mass["minSeeded"] >= 0,
          f"masses {[round(m, 8) for m in masses]}; seeding the zero-degree user "
          f"{mass['danglingUserId']} still sums to 1")
    check("global PageRank converges below the 1e-6 tolerance inside the cap",
          mass["globalConv"] and mass["globalResidual"] < 1e-6 and mass["globalIter"] < 500,
          f"{mass['globalIter']} iterations, residual {mass['globalResidual']:.2e}")

    # ---- 4. the all-seeds batch equals the single-seed reference ------------
    agreement = page.evaluate("""() => {
      const g = getGraph();
      let worst = 0, checked = 0, iterGap = 0;
      for (const userId of HW6_STATE.trainUserIds.slice(0, 10)) {
        const node = g.nodeOfUser(userId);
        const batched = personalizedPageRank(node, 0.85);
        const single = personalizedPageRank(node, 0.85, { noBatch: true });
        for (let i = 0; i < batched.p.length; i++) {
          worst = Math.max(worst, Math.abs(batched.p[i] - single.p[i]));
        }
        iterGap = Math.max(iterGap, Math.abs(batched.iterations - single.iterations));
        checked++;
      }
      return { worst, checked, iterGap };
    }""")
    check("the memoised all-seeds batch reproduces the single-seed walk",
          agreement["worst"] < 1e-6,
          f"max |diff| {agreement['worst']:.2e} over {agreement['checked']} users "
          f"(iteration counts differ by at most {agreement['iterGap']})")

    # ---- 5. PPR really is personal; global PageRank is not -----------------
    # The two users' LISTS differ under both methods, because their exclusion
    # sets differ. The property that separates the methods is the score vector:
    # global PageRank hands every user the same one, PPR does not.
    personal = page.evaluate("""() => {
      const a = HW6_STATE.trainUserIds[0], b = HW6_STATE.trainUserIds[1];
      const alpha = 0.85;
      const prA = hw6ScoreVector("pagerank", a, alpha, "test").scores;
      const prB = hw6ScoreVector("pagerank", b, alpha, "test").scores;
      const pprA = hw6ScoreVector("ppr", a, alpha, "test").scores;
      const pprB = hw6ScoreVector("ppr", b, alpha, "test").scores;
      let prGap = 0, pprGap = 0;
      for (let i = 0; i < prA.length; i++) {
        prGap = Math.max(prGap, Math.abs(prA[i] - prB[i]));
        pprGap = Math.max(pprGap, Math.abs(pprA[i] - pprB[i]));
      }
      const rank = (scores) => Array.from(scores.keys())
        .sort((x, y) => scores[y] - scores[x]).slice(0, 10).join(",");
      return { prGap, pprGap, a, b,
               prSameOrder: rank(prA) === rank(prB),
               pprSameOrder: rank(pprA) === rank(pprB) };
    }""")
    check("global PageRank scores every user alike while PPR does not",
          personal["prGap"] == 0 and personal["prSameOrder"]
          and personal["pprGap"] > 0 and not personal["pprSameOrder"],
          f"users {personal['a']} / {personal['b']}: global PageRank score gap "
          f"{personal['prGap']:.2e} (identical ranking), PPR gap {personal['pprGap']:.2e}")

    # ---- 6. the exclusion rules ---------------------------------------------
    filtering = page.evaluate("""() => {
      const cohort = hw6SharedCohort("test").slice(0, 25);
      let badTest = 0, badVal = 0, outside = 0, unsorted = 0, lists = 0;
      for (const userId of cohort) {
        for (const split of ["test", "val"]) {
          const recs = recommendTopK("ppr", userId, 10, split);
          lists++;
          const excluded = hw6ExclusionSet(split, userId);
          for (let i = 0; i < recs.length; i++) {
            if (excluded.has(recs[i].movieId)) { if (split === "test") badTest++; else badVal++; }
            if (!HW6_STATE.candidateSet.has(recs[i].movieId)) outside++;
            if (i > 0) {
              const prev = recs[i - 1];
              if (prev.score < recs[i].score) unsorted++;
              else if (prev.score === recs[i].score && prev.movieId >= recs[i].movieId) unsorted++;
            }
          }
        }
      }
      // the validation rule must be strictly weaker than the test rule
      const userId = cohort[0];
      const valOnly = hw6ExclusionSet("val", userId);
      const testOnly = hw6ExclusionSet("test", userId);
      let nested = true;
      valOnly.forEach((id) => { if (!testOnly.has(id)) nested = false; });
      return { lists, badTest, badVal, outside, unsorted, nested,
               valSize: valOnly.size, testSize: testOnly.size };
    }""")
    check("both splits exclude the right movies, stay in the pool and sort correctly",
          filtering["badTest"] == 0 and filtering["badVal"] == 0
          and filtering["outside"] == 0 and filtering["unsorted"] == 0
          and filtering["nested"],
          f"{filtering['lists']} lists; validation excludes {filtering['valSize']} films "
          f"and test {filtering['testSize']} for the first cohort user")

    # ---- 7. Recall@10 bookkeeping -------------------------------------------
    recall = page.evaluate("""() => {
      const report = recallAtK("popularity", 10, "test");
      const manual = report.perUser.length
        ? report.perUser.reduce((s, r) => s + r.hit / r.relevant, 0) / report.perUser.length
        : 0;
      const cohort = hw6SharedCohort("test");
      const sameCohort = report.cohort.length === cohort.length
        && report.cohort.every((id, i) => id === cohort[i]);
      return { recall: report.recall, manual, sameCohort,
               reported: report.perUser.length, cohort: cohort.length,
               allPositive: report.perUser.every((r) => r.relevant > 0 && r.hit <= r.relevant) };
    }""")
    check("Recall@10 is the macro-average over the shared cohort",
          abs(recall["recall"] - recall["manual"]) < 1e-12 and recall["sameCohort"]
          and recall["allPositive"],
          f"{recall['reported']} users reported out of a {recall['cohort']}-user cohort, "
          f"recall {recall['recall']:.4f}")

    # ---- 8. every method is scored on the same cohort and candidates --------
    shared = page.evaluate("""() => {
      const out = {};
      for (const method of ["popularity", "pagerank", "ppr", "mf"]) {
        try {
          const report = recallAtK(method, 10, "test");
          out[method] = { cohort: report.cohort.length, users: report.perUser.length };
        } catch (error) { out[method] = { error: String(error.message || error) }; }
      }
      return out;
    }""")
    sizes = {m: v.get("cohort") for m, v in shared.items() if "cohort" in v}
    check("the four methods are evaluated on one shared cohort",
          len(set(sizes.values())) == 1 and len(sizes) == 4,
          ", ".join(f"{m} {n}" for m, n in sizes.items()))

    # ---- 9. the readme section 9 exporters ----------------------------------
    # The exporters serialize what the page has actually produced, so a list and
    # a convergence record are rendered first; `run_experiment.py` is what fills
    # them from the full four-method comparison.
    page.evaluate("""() => {
      const g = getGraph();
      hw6SetAlpha(0.85);
      hw6RunAndRecordConvergence("pagerank", g, 0.85);
      hw6RunAndRecordConvergence("ppr", g, 0.85, HW6_STATE.trainUserIds[0]);
      hw6RenderRecommendations();
    }""")
    exports = page.evaluate("""() => {
      const verification = serializeVerification();
      const results = serializeResults();
      const recommendations = serializeRecommendations();
      let parsed = null;
      try { parsed = JSON.parse(verification); } catch (error) { parsed = null; }
      return {
        verificationIsJson: parsed !== null,
        verificationChecks: parsed && Array.isArray(parsed.checks) ? parsed.checks.length : 0,
        resultsHeader: results.split("\\n")[0],
        resultsRows: results.trim().split("\\n").length - 1,
        recommendationsHeader: recommendations.split("\\n")[0],
        recommendationsRows: recommendations.trim().split("\\n").length - 1,
      };
    }""")
    check("the three readme section 9 exports are produced and well formed",
          exports["verificationIsJson"] and exports["verificationChecks"] >= 12
          and exports["recommendationsRows"] >= 1
          and exports["recommendationsHeader"].startswith("user_id,method,alpha,rank")
          and exports["resultsHeader"].startswith("method,"),
          f"verification.json carries {exports['verificationChecks']} checks; "
          f"results.csv {exports['resultsRows']} rows; "
          f"recommendations.csv {exports['recommendationsRows']} rows")

    # ---- 10. repeated runs must not accumulate tensors ----------------------
    leak = page.evaluate("""() => {
      const g = getGraph();
      const before = tf.memory().numTensors;
      for (let k = 0; k < 3; k++) {
        pageRank(0.85);
        personalizedPageRank(g.nodeOfUser(HW6_STATE.trainUserIds[k]), 0.85, { noBatch: true });
      }
      const after = tf.memory().numTensors;
      return { before, after };
    }""")
    check("repeated propagation does not accumulate tensors",
          leak["after"] <= leak["before"],
          f"live tensors {leak['before']} -> {leak['after']}")

    # ---- panel screenshots --------------------------------------------------
    page.evaluate("() => runSelfChecks()")
    page.locator("#self-checks").screenshot(path=str(FIGURES / "fixed_checks.png"))
    page.evaluate("() => { hw6SetAlpha(0.85); hw6RenderRecommendations(); }")
    page.locator("#recommendations").screenshot(path=str(FIGURES / "fixed_topk.png"))
    log(f"saved {FIGURES / 'fixed_checks.png'} and fixed_topk.png")

    # ---- the untouched starter ---------------------------------------------
    page.goto(f"{base_url}/starter/index.html", wait_until="networkidle")
    page.wait_for_function(
        "() => typeof tf !== 'undefined' && HW6_STATE.splitReady", timeout=180_000)
    starter = page.evaluate("() => runSelfChecks()")
    check("the untouched starter reports pending stubs rather than passes",
          starter["pending"] > 0 and starter["passed"] < 12,
          f"{starter['passed']} passed, {starter['failed']} failed, {starter['pending']} pending")
    page.locator("#self-checks").screenshot(path=str(FIGURES / "starter_checks.png"))

    return {
        "backend": backend,
        "graph": graph,
        "selfChecks": result["passed"],
        "starterSelfChecks": {"passed": starter["passed"], "pending": starter["pending"]},
        "referenceAgreement": {"uniform": worst_u, "restartAtA": worst_s},
        "batchAgreement": agreement,
        "mass": mass,
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
            page = browser.new_page(viewport={"width": 1320, "height": 1200})
            page.set_default_timeout(1_800_000)
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
