"""Verify which lunch icons actually render in starter/ vs fixed/ index.html.

Run:  python tests/check_icons.py
Needs: playwright (python) + local Google Chrome, internet (cdnjs Font Awesome 6.4.0).
"""
import json
import random
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

HW = Path(__file__).resolve().parent.parent
FIG = HW / "report" / "figures"
N_ITEMS = 12

# Read what ::before glyph the displayed <i> got and how wide it is.
PROBE = """() => {
  const i = document.querySelector('.food-icon i');
  const r = i.getBoundingClientRect();
  return { cls: i.className,
           name: document.querySelector('.food-name').textContent,
           before: getComputedStyle(i, '::before').content,
           width: Math.round(r.width * 10) / 10 };
}"""


def pick(page, idx):
    """Force Math.random to select item idx, click, wait for the 500 ms timeout."""
    page.evaluate(f"() => {{ Math.random = () => ({idx} + 0.5) / {N_ITEMS}; }}")
    page.click("#generateBtn")
    page.wait_for_timeout(1100)  # 500 ms delay + 500 ms fade-in animation
    return page.evaluate(PROBE)


def double_click_check(browser, label, html_path):
    """Click Pizza, then Sushi 100 ms later. At ~550 ms after the first click the
    second pick is still 'Thinking...'; a stale Pizza result there is the timer race."""
    page = browser.new_page()
    page.goto(html_path.as_uri(), wait_until="networkidle")
    page.wait_for_timeout(1100)
    shown = page.evaluate(f"""async () => {{
        const btn = document.getElementById('generateBtn');
        const sleep = ms => new Promise(r => setTimeout(r, ms));
        Math.random = () => 0.5 / {N_ITEMS}; btn.click();
        await sleep(100);
        Math.random = () => 1.5 / {N_ITEMS}; btn.click();
        await sleep(450);
        const mid = document.querySelector('.food-name').textContent;
        await sleep(300);
        return [mid, document.querySelector('.food-name').textContent];
    }}""")
    page.close()
    print(f"[{label}] double click: at 550 ms shows '{shown[0]}', final '{shown[1]}'")
    return shown


def run_variant(browser, label, html_path, patch=None):
    page = browser.new_page(viewport={"width": 600, "height": 700})
    if patch:  # serve a modified copy of the file (used to test the fallback)
        src = html_path.read_text(encoding="utf-8")
        for a, b in patch:
            src = src.replace(a, b)
        page.route("**/index.html", lambda route: route.fulfill(
            status=200, content_type="text/html", body=src))
    page.goto(html_path.as_uri(), wait_until="networkidle")
    page.wait_for_timeout(1100)  # initial generateRandomLunch() on load
    rows = []
    for idx in range(N_ITEMS):
        r = pick(page, idx)
        r["rendered"] = r["before"] not in ("none", "normal", '""') and r["width"] > 0
        rows.append(r)
        if label in ("starter", "fixed") and r["name"] in ("Ramen", "Pizza"):
            page.locator(".container").screenshot(
                path=str(FIG / f"{label}_{r['name'].lower()}.png"))
    page.close()
    ok = sum(r["rendered"] for r in rows)
    print(f"\n== {label}: {ok}/{N_ITEMS} icons rendered")
    for r in rows:
        # glyphs live in the Private Use Area; print the code point instead of the raw char
        before = r["before"].encode("ascii", "backslashreplace").decode()
        print(f"  {r['name']:9s} {r['cls']:24s} before={before:10s} "
              f"width={r['width']:5}  {'OK' if r['rendered'] else 'BLANK'}")
    return rows


def css_check(html_path, css_text):
    """Independent check: does each fa-* class used in the HTML exist in the CSS file?"""
    # only real class strings ("fas fa-xxx"), not names mentioned in comments
    used = sorted(set(re.findall(r"fas fa-([a-z0-9-]+)", html_path.read_text(encoding="utf-8"))))
    missing = [c for c in used
               if not re.search(r"\.fa-" + re.escape(c) + r"[,:{]", css_text)]
    return used, missing


def monte_carlo(rows, clicks=10_000, seed=42):
    """Simulate random clicks with the page's own selection formula."""
    rng = random.Random(seed)
    blank = sum(not rows[int(rng.random() * N_ITEMS)]["rendered"] for _ in range(clicks))
    return blank / clicks


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    css = (HW / "tests" / "fa-6.4.0-all.min.css").read_text(encoding="utf-8")
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        print("Chrome", browser.version)
        results["starter"] = run_variant(browser, "starter", HW / "starter" / "index.html")
        results["fixed"] = run_variant(browser, "fixed", HW / "fixed" / "index.html")
        # Re-introduce one bad class into the fixed page: fallback must kick in.
        results["fixed_bad_ramen"] = run_variant(
            browser, "fixed_bad_ramen", HW / "fixed" / "index.html",
            patch=[('"fas fa-bowl-rice"', '"fas fa-bowl-hot"')])
        print()
        double = {label: double_click_check(browser, label, HW / label / "index.html")
                  for label in ("starter", "fixed")}
        browser.close()

    summary = {}
    for label in ("starter", "fixed"):
        used, missing = css_check(HW / label / "index.html", css)
        rate = monte_carlo(results[label])
        summary[label] = {"rendered": sum(r["rendered"] for r in results[label]),
                          "missing_in_css": missing, "mc_blank_rate": rate}
        print(f"\n[{label}] classes missing in FA 6.4.0 CSS: {missing or 'none'}")
        print(f"[{label}] Monte-Carlo blank rate over 10000 clicks: {rate:.4f}")
    ramen = results["fixed_bad_ramen"][5]
    print(f"\n[fallback] Ramen with fa-bowl-hot -> shown class '{ramen['cls']}', "
          f"rendered={ramen['rendered']}")
    summary["fallback_ramen"] = ramen
    summary["double_click"] = double

    out = HW / "tests" / "results.json"
    out.write_text(json.dumps({"summary": summary, "rows": results}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
