"""Build kosychev_midterm_report.pdf from report.md (Markdown -> HTML -> PDF via headless Chrome).

The method table and the two-user tables are rendered from the experiment
outputs (experiments/results_*.json, two_users.json) at build time, so the
numbers and programme ids in the PDF are the ones the scripts wrote.

Run:  python report/build_report.py
"""
import json
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"

METHOD_ROWS = [
    ("popularity", "Popularity (reference)", "users' history top-5 frequency"),
    ("history (re-watch baseline)", "History (re-watch)", "own seconds per show"),
    ("CBF", "CBF", "TF-IDF: channel, category, genres, people, title"),
    ("association rules", "Association rules", None),
    ("MF: implicit ALS", "MF: implicit ALS", None),
    ("MF: PureSVD", "MF: PureSVD", None),
    ("PPR (co-view graph)", "PPR, co-view graph", None),
    ("habit x schedule (seconds, channel-hours)", "Channel-hour exposure", None),
    ("habit x schedule (watched-airing rate)", "<b>Habit x schedule (final)</b>", None),
]


def _setting(name, params):
    if name == "association rules":
        return f"basket &ge;{params['min_minutes']} min, support &ge;{params['min_support']}"
    if name == "MF: implicit ALS":
        return f"positives &ge;{params['min_minutes']} min, &alpha;={params['alpha']}, 32 factors"
    if name == "MF: PureSVD":
        return f"k={params['factors']}, log-minutes"
    if name == "PPR (co-view graph)":
        return f"{params['weight']} edges, &alpha;={params['alpha']}, 50 neighbours"
    if name.startswith("habit x schedule (seconds"):
        return f"{params.get('slot', 1800) // 60}-min slots, channel x slot"
    if name.startswith("habit x schedule (watched"):
        return f"half-life {params['half_life']:g} wk, prior {params['prior']:g}"
    return ""


def methods_table():
    watched = json.loads((EXPERIMENTS / "results_watched.json").read_text(encoding="utf-8"))
    seconds = json.loads((EXPERIMENTS / "results_seconds.json").read_text(encoding="utf-8"))
    head = ("<table><tr><th rowspan=2>Method</th><th rowspan=2>Setting (selected on the hold-out)</th>"
            f"<th colspan=2>Competition key (watched airings)</th><th rowspan=2>&lt;5 scored</th>"
            "<th rowspan=2>Time</th><th rowspan=2>Seconds key</th></tr>"
            f"<tr><th>cohort ({watched['cohort_users']:,})</th><th>all ({watched['all_users']:,})</th></tr>")
    rows = []
    for key, label, fixed in METHOD_ROWS:
        w = watched["methods"][key]
        sec = seconds["methods"].get(key)
        rows.append(
            f"<tr><td>{label}</td><td>{fixed or _setting(key, w['params'])}</td>"
            f"<td class=n>{w['map_cohort']:.4f}</td><td class=n>{w['map_all']:.4f}</td>"
            f"<td class=n>{w['fallback_users_cohort']}</td><td class=n>{w['seconds']:.1f} s</td>"
            f"<td class=n>{'&ndash;' if sec is None else format(sec['map_cohort'], '.4f')}</td></tr>")
    h = seconds["hybrid"]
    rows.append(f"<tr><td>LambdaRank hybrid (1st round)</td><td>{h['trees']} trees, pool recall "
                f"{h['pool_recall']:.1%}</td><td colspan=4><i>fitted to the seconds key only</i></td>"
                f"<td>{h['map_cohort']:.4f}</td></tr>")
    return head + "".join(rows) + "</table>"


def _short(title, n=30):
    t = re.sub(r"^(Т/с|Х/ф|М/с|Д/ф)\s*", "", str(title)).replace("'", "").replace("’", "")
    t = t.strip(" .")
    return t if len(t) <= n else t[:n - 1].rstrip() + "&hellip;"


def two_user_tables():
    data = json.loads((EXPERIMENTS / "two_users.json").read_text(encoding="utf-8"))
    legend = {}
    blocks = []
    for role in ("heavy", "light"):
        e = data[role]
        answer = {x["id"] for x in e["answer_21_30"]}
        def cells(items, mark=True):
            out = []
            for x in items:
                legend[x["id"]] = (x["title"], x["category"])
                txt = str(x["id"])
                if mark and x["id"] in answer:
                    txt = f"<b>{txt}&#10003;</b>"
                if x.get("seen_before") is False:
                    txt += "<sup>n</sup>"
                out.append(f"<td>{txt}</td>")
            return "".join(out)
        rows = [f"<tr><td><i>history 11&ndash;20 (watched&times;)</i></td>"
                + "".join(f"<td>{x['id']} ({x['watched']})</td>" for x in e["history_top5"]) + "<td></td></tr>",
                "<tr><td><i>answer 21&ndash;30</i></td>" + cells(e["answer_21_30"], mark=False) + "<td></td></tr>"]
        for x in e["history_top5"]:
            legend[x["id"]] = (x["title"], x["category"])
        for name, l in e["lists"].items():
            label = "<b>Habit x schedule</b>" if name.startswith("habit") else name
            rows.append(f"<tr><td>{label}</td>{cells(l['items'])}<td>{l['hits']}</td></tr>")
        title = (("Heavy viewer" if role == "heavy" else "Light viewer")
                 + f" &mdash; {e['watched_airings_11_20']} watched airings of "
                   f"{e['distinct_shows_11_20']} programmes in weeks 11&ndash;20")
        blocks.append(f"<p class=tcap><b>{title}</b> (user {e['user_id']})</p><table class=ids>"
                      "<tr><th></th><th>#1</th><th>#2</th><th>#3</th><th>#4</th><th>#5</th><th>hits</th></tr>"
                      + "".join(rows) + "</table>")
    items = sorted(legend.items())
    cells = [f"<td><b>{i}</b></td><td>{_short(t)}</td><td>{c or ''}</td>" for i, (t, c) in items]
    per_row = 3
    rows = ["<tr>" + "".join(cells[k:k + per_row]) + "</tr>" for k in range(0, len(cells), per_row)]
    blocks.append("<table class=legend><tr>" + "<th>id</th><th>title</th><th>category</th>" * per_row
                  + "</tr>" + "".join(rows) + "</table>")
    return "".join(blocks)
OUT_PDF = HERE / "kosychev_midterm_report.pdf"

CSS = """
@page { size: A4; margin: 15mm 15mm 14mm 15mm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 9.4pt; line-height: 1.17; color: #111; }
h1 { font-size: 15pt; text-align: center; margin: 0 0 5px; line-height: 1.18; }
.meta { text-align: center; font-style: italic; font-size: 9pt; margin-bottom: 8px; }
h2 { font-size: 11pt; margin: 9px 0 3px; color: #1f3b63; border-bottom: 1px solid #c9d3e0; }
p { margin: 2px 0 4px; text-align: justify; }
ul { margin: 2px 0 4px; padding-left: 16px; }
li { margin: 1px 0; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.2pt; background: #f2f4f7; padding: 0 2px; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 3px 0 5px; }
th, td { border: 1px solid #8fa3bf; padding: 1px 4px; text-align: left; vertical-align: top; }
th { background: #dde6f2; }
.fig { text-align: center; margin: 4px 0; page-break-inside: avoid; }
.fig img { width: 31%; margin: 0 1%; border: 1px solid #ccc; }
.fig.wide img { width: 47%; }
.fig.full img { width: 68%; border: none; }
.fig.manual img { width: 78%; border: 1px solid #ccc; }
.fig.shot img { width: 62%; border: 1px solid #ccc; }
.fig.app img { width: 88%; border: 1px solid #ccc; }
.fig.pair img { border: 1px solid #ccc; vertical-align: top; margin: 0 1%; }
.fig.pair img:first-child { width: 27%; }
.fig.pair img:last-child { width: 48%; }
.fig p { text-align: center; font-size: 8.4pt; margin-top: 2px; }
.todo { background: #fff3a8; font-style: italic; }
table.ids td, table.ids th { text-align: center; padding: 0 3px; }
table.ids td:first-child { text-align: left; white-space: nowrap; }
table.legend { font-size: 7.4pt; }
table.legend td { padding: 0 3px; }
p.tcap { margin: 4px 0 1px; font-size: 8.6pt; text-align: left; }
.fig.kaggle img { width: 84%; border: none; }
td.n { white-space: nowrap; }
sup { line-height: 0; font-size: 70%; }
.fig.wide2 img { width: 100%; border: none; }
"""


def main():
    md_text = (HERE / "report.md").read_text(encoding="utf-8")
    md_text = md_text.replace("<!-- METHODS_TABLE -->", methods_table())
    md_text = md_text.replace("<!-- TWO_USERS -->", two_user_tables())
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "md_in_html"])
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Midterm report — Aleksei Kosychev</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    html_path = HERE / "report.html"
    html_path.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.pdf(path=str(OUT_PDF), format="A4", print_background=True,
                 display_header_footer=True, header_template="<span></span>",
                 footer_template='<div style="font-size:8px;width:100%;text-align:center">'
                                 '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                 margin={"top": "16mm", "bottom": "16mm", "left": "17mm", "right": "17mm"})
        browser.close()
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
