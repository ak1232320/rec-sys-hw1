"""Build kosychev_a01_report.pdf from report.md (Markdown -> HTML -> PDF via headless Chrome).

Run:  python report/build_report.py
"""
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "kosychev_a01_report.pdf"

CSS = """
@page { size: A4; margin: 18mm 17mm 16mm 17mm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 10.5pt; line-height: 1.32; color: #111; }
h1 { font-size: 16.5pt; text-align: center; margin: 0 0 6px; line-height: 1.2; }
.meta { text-align: center; font-style: italic; font-size: 9.5pt; margin-bottom: 10px; }
h2 { font-size: 12pt; margin: 12px 0 4px; color: #1f3b63; border-bottom: 1px solid #c9d3e0; }
p { margin: 3px 0 5px; text-align: justify; }
ul { margin: 2px 0 5px; padding-left: 18px; }
li { margin: 1px 0; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt; background: #f2f4f7; padding: 0 2px; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 4px 0 8px; }
th, td { border: 1px solid #8fa3bf; padding: 2px 5px; text-align: left; vertical-align: top; }
th { background: #dde6f2; }
.fig { text-align: center; margin: 6px 0; page-break-inside: avoid; }
.fig img { width: 44%; margin: 0 1.5%; border: 1px solid #ccc; }
.fig p { text-align: center; font-size: 9pt; }
.todo { background: #fff3a8; font-style: italic; }
"""


def main():
    md_text = (HERE / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "md_in_html"])
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>A01 report — Aleksei Kosychev</title><style>{CSS}</style></head>
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
