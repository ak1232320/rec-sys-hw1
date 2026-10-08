"""Browser test for the A04 association-rule miner.

Serves the hw4 folder over http, drives the real page in headless Chrome and checks
that (a) every rule table the page mines equals the independent FP-Growth reference
in experiments/results.json, count for count, (b) the UI shows those numbers,
(c) the reverse-direction and zero-denominator paths behave, and (d) the page still
works from file://. Also records what the untouched starter does, and saves the two
screenshots used in the report.

Run:  python tests/check_rules.py      (needs: pip install playwright, and Google Chrome)
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
RULES = json.loads((APP / "experiments" / "reference_rules.json").read_text(encoding="utf-8"))
OUT_JSON = HERE / "results.json"
OUT_LOG = HERE / "run_log.txt"
FIGURES = APP / "report" / "figures"
TOL = 1e-12

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


# ------------------------------------------------------------------ helpers

MINE_JS = """([s, c]) => {
  const started = performance.now();
  const itemsets = findFrequentItemsets(TRANSACTIONS, s);
  const rules = generateRules(itemsets, c);
  return {
    ms: performance.now() - started,
    itemsets: itemsets.length,
    rules: rules.map(r => [r.antecedent.slice().sort().join('+'), r.consequent.slice().sort().join('+'),
                           r.jointCount, r.antecedentCount, r.consequentCount, r.support, r.confidence, r.lift]),
  };
}"""


def keyed(rows):
    return {(r[0], r[1]): r[2:] for r in rows}


def reference_rows(rules, n=RULES["n"]):
    """Expand compact reference rows; the metrics use the same arithmetic as the definitions."""
    return [["+".join(sorted(a)), "+".join(sorted(b)), j, ca, cb, j / n, j / ca, (j / ca) / (cb / n)]
            for a, b, j, ca, cb in rules]


def compare_tables(page_rows, ref_rows):
    mine, ref = keyed(page_rows), keyed(ref_rows)
    if mine.keys() != ref.keys():
        return False, f"{len(mine)} rules on the page vs {len(ref)} in the reference; " \
                      f"{len(mine.keys() - ref.keys())} extra, {len(ref.keys() - mine.keys())} missing"
    worst = 0.0
    for key, values in mine.items():
        expected = ref[key]
        if values[:3] != expected[:3]:
            return False, f"counts differ for {key}: {values[:3]} vs {expected[:3]}"
        worst = max(worst, *(abs(a - b) for a, b in zip(values[3:], expected[3:])))
    return worst < TOL, f"{len(mine)} rules, identical counts, max metric difference {worst:.1e}"


def set_thresholds(page, support_pct, confidence_pct, lift_filter=True):
    page.evaluate(
        """([s, c, f]) => {
             const sup = document.getElementById('min-support');
             const conf = document.getElementById('min-confidence');
             sup.value = String(s); conf.value = String(c);
             sup.dispatchEvent(new Event('input')); conf.dispatchEvent(new Event('input'));
             const box = document.getElementById('lift-filter');
             if (box) box.checked = f;
           }""", [support_pct, confidence_pct, lift_filter])


def run_rules(page):
    page.click("#run-rules")
    page.wait_for_function("() => /^Done|failed|TODO/.test(document.getElementById('status').textContent)")


def table_rows(page):
    return page.evaluate("""() => Array.from(document.querySelectorAll('#results tbody tr'))
                              .map(tr => Array.from(tr.cells).map(td => td.textContent.trim()))""")


def pct(x):
    return f"{x * 100:.2f}%"


# --------------------------------------------------------------------- run

def main():
    httpd, base = serve(APP)
    log(f"serving {APP} at {base}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        log(f"chrome {browser.version}")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        # ---- the untouched starter, for the record
        page.goto(f"{base}/starter/index.html")
        starter_tests = page.evaluate("() => { const r = runTests(); return [r.passed, r.failed, r.pending]; }")
        check("starter: harness reports 2 passed / 0 failed / 9 pending, as the course README says",
              starter_tests == [2, 0, 9], f"{starter_tests}")
        page.click("#run-tests")
        run_rules(page)
        failed_text = page.text_content("#results")
        check("starter: Run rules fails with the TODO message", "Run failed" in failed_text,
              page.text_content("#status"))
        page.screenshot(path=str(FIGURES / "starter_app.png"))
        errors.clear()

        # ---- the finished page
        page.goto(f"{base}/index.html")
        summary = page.text_content("#dataset-summary-body")
        check("dataset summary: 17,080 baskets and 3,653 distinct items",
              "17,080" in summary and "3,653" in summary)
        top = page.evaluate("() => countItem(DATASET_INDEX, '85123A')")
        check("anchor fact: 85123A is in 1,959 baskets (11.47%)", top == 1959, f"{top}")

        result = page.evaluate("() => { const r = runTests(); return [r.passed, r.failed, r.pending]; }")
        check("finished: harness reports 11 passed / 0 failed / 0 pending", result == [11, 0, 0], f"{result}")

        # Fixture against brute force.
        bad = []
        for case in REFERENCE["fixture"]:
            got = page.evaluate(
                """([s, c]) => {
                     const fx = tinyWorkedExample().baskets;
                     const sets = findFrequentItemsets(fx, s);
                     const rules = generateRules(sets, c);
                     return {
                       itemsets: sets.map(x => [x.items.slice().sort(), x.count]).sort(),
                       rules: rules.map(r => [r.antecedent.slice().sort(), r.consequent.slice().sort()]),
                     };
                   }""", [case["minSupport"], case["minConfidence"]])
            if sorted(map(json.dumps, got["itemsets"])) != sorted(map(json.dumps, case["itemsets"])) or \
               sorted(map(json.dumps, got["rules"])) != sorted(map(json.dumps, case["rules"])):
                bad.append((case["minSupport"], case["minConfidence"]))
        check("fixture: itemsets and rules equal brute-force enumeration at 9 threshold pairs", not bad,
              f"mismatches: {bad}" if bad else "38 rules at 20% / 50%")

        # Real data against FP-Growth, every stored setting.
        timings = {}
        for key, ref_rules in RULES["tables"].items():
            s, c = (float(x) for x in key.split("/"))
            got = page.evaluate(MINE_JS, [s, c])
            timings[key] = round(got["ms"])
            ok, detail = compare_tables(got["rules"], reference_rows(ref_rules))
            check(f"support {s:.1%} / confidence {c:.0%}: page rules equal mlxtend FP-Growth", ok,
                  f"{detail}; mined in {got['ms']:.0f} ms")
            lifts = [r[7] for r in got["rules"]]
            if key == "0.01/0.3":
                check("rules are sorted by lift, strongest first",
                      all(a >= b for a, b in zip(lifts, lifts[1:])))
        sweep_ok = []
        for row in REFERENCE["sweep"]:
            n_rules = page.evaluate("([s, c]) => generateRules(findFrequentItemsets(TRANSACTIONS, s), c).length",
                                    [row["minSupport"], row["minConfidence"]])
            sweep_ok.append(n_rules == row["rules"])
        check("threshold sweep: rule counts equal the reference at all 24 settings", all(sweep_ok),
              f"{sum(sweep_ok)}/24")

        # ---- the UI at the chosen thresholds
        set_thresholds(page, 1, 30)
        run_rules(page)
        status = page.text_content("#status")
        rows = table_rows(page)
        check("UI at 1% / 30%: the table lists 950 rules", len(rows) == 950, status)
        top_rule = sorted(reference_rows(RULES["tables"]["0.01/0.3"]), key=lambda r: -r[7])[0]
        expected_first = [str(top_rule[2]), str(top_rule[3]), str(top_rule[4]), pct(top_rule[5]),
                          pct(top_rule[6]), f"{top_rule[7]:.4f}"]
        check("first row shows the reference's strongest rule", rows[0][2:] == expected_first,
              f"{rows[0][0]} -> {rows[0][1]}: {rows[0][2:]}")
        in_bounds = all(float(r[5].rstrip('%')) >= 1.0 - 0.005 and float(r[6].rstrip('%')) >= 30 - 0.005
                        and float(r[7]) > 1 for r in rows)
        check("every displayed rule meets support >= 1%, confidence >= 30%, lift > 1", in_bounds)

        # Select the investigated rule and reverse it.
        index = next(i for i, r in enumerate(rows) if r[0] == "22386 — JUMBO BAG PINK POLKADOT"
                     and r[1] == "85099B — JUMBO BAG RED RETROSPOT")
        page.locator("#results tbody tr").nth(index).click()
        detail = page.text_content("#rule-detail")
        check("detail panel: JUMBO BAG PINK -> RED shows 546 / 869 / 1584, 62.83%, lift 6.7749",
              all(t in detail for t in ["546", "869", "1584", "62.83%", "6.7749"]))
        check("detail panel: reverse direction quoted as 34.47% with the same lift",
              "34.47%" in page.text_content("#rule-detail .comparison"))
        page.locator("#rule-detail").scroll_into_view_if_needed()
        page.screenshot(path=str(FIGURES / "fixed_app.png"), full_page=False)
        page.click("#reverse-rule")
        detail = page.text_content("#rule-detail")
        check("Reverse button: B -> A has confidence 34.47% and lift 6.7749",
              "85099B" in page.text_content("#rule-detail dd") and "34.47%" in detail and "6.7749" in detail)
        hand = REFERENCE["useful"]["fromRawRows"]
        counted = page.evaluate("() => [countItem(DATASET_INDEX,'22386'), countItem(DATASET_INDEX,'85099B'),"
                                " countPair(DATASET_INDEX,'22386','85099B')]")
        check("page counts equal distinct InvoiceNo counted from the raw workbook rows",
              counted == [hand["countA"], hand["countB"], hand["joint"]], f"{counted}")

        # Zero denominators.
        text = page.evaluate("""() => {
            const box = document.createElement('div');
            renderRuleDetail({ antecedent: ['NO-SUCH-STOCK'], consequent: ['85123A'] }, DATASET_INDEX, box);
            return box.textContent; }""")
        check("count(A) = 0: the panel says 'undefined' with an inline note, never NaN or Infinity",
              "undefined" in text and "count(A) = 0" in text and "NaN" not in text and "Infinity" not in text)

        # The lift filter only matters at the lowest confidence.
        set_thresholds(page, 0.5, 10, lift_filter=False)
        run_rules(page)
        unfiltered = len(table_rows(page))
        set_thresholds(page, 0.5, 10, lift_filter=True)
        run_rules(page)
        filtered = len(table_rows(page))
        check("lift filter at 0.5% / 10% drops exactly the 2 rules with lift <= 1",
              (unfiltered, filtered) == (19058, 19056), f"{unfiltered} -> {filtered}")

        check("no page or console errors on the finished page", not errors, "; ".join(errors[:3]))

        # ---- file:// works too (no fetch in this app)
        page.goto((APP / "index.html").as_uri())
        result = page.evaluate("() => { const r = runTests(); return [r.passed, r.failed, r.pending]; }")
        check("opened from file:// the page loads and passes 11/11", result == [11, 0, 0], f"{result}")
        browser.close()
    httpd.shutdown()

    passed = sum(c["passed"] for c in checks)
    log(f"\n{passed}/{len(checks)} checks passed")
    OUT_JSON.write_text(json.dumps({"passed": passed, "total": len(checks), "timingsMs": timings,
                                    "checks": checks}, indent=1, ensure_ascii=False), encoding="utf-8")
    OUT_LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
