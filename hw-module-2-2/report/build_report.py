"""Build kosychev_session2_report.pdf from report.md (Markdown -> HTML -> PDF via headless Chrome).

Every measured number is pulled from the run outputs, never typed by hand:
  {{src:path.to.value|fmt}}  value from a JSON source (fmt: .4f default, d, .1f, +.4f, pct)
  {{table:name}}             a generated table (demo, dev, paired)
Run:  python report/build_report.py        (needs `markdown` and `playwright`; Python 3.12 here)
"""
import json
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_PDF = HERE / 'kosychev_session2_report.pdf'
SOURCES = {'demo': ROOT / 'outputs' / 'demo' / 'metrics.json',
           'dev': ROOT / 'outputs' / 'dev' / 'dev_metrics.json',
           'diag': ROOT / 'outputs' / 'dev' / 'diagnose_2021-06-01.json',
           'base': ROOT / 'outputs' / 'baseline' / 'timing.json',
           'emb': ROOT / 'kaggle' / 'output' / 'manifest.json'}
DATA = {k: json.loads(p.read_text(encoding='utf-8')) for k, p in SOURCES.items() if p.exists()}


def _count(blocks, cond):
    return sum(1 for b in blocks if cond(b))


_cuts = DATA['dev']['per_cutoff']
_new = lambda b, a: b['ndcg_with_new_asset'][a]
_demo = lambda b, a: b['demo_100']['ndcg_all'][a]
DATA['derived'] = {
    'cutoffs': len(_cuts),
    'llm_below_pop': _count(_cuts, lambda b: _new(b, 'llm_fill') < _new(b, 'pop_fill')),
    'llm_below_tfidf': _count(_cuts, lambda b: _new(b, 'llm_fill') < _new(b, 'tfidf_fill')),
    'llm_below_e5': _count(_cuts, lambda b: _new(b, 'llm_fill') < _new(b, 'e5_fill')),
    'e5_above_pop': _count(_cuts, lambda b: _new(b, 'e5_fill') > _new(b, 'pop_fill')),
    'demo100_llm_below_pop': _count(_cuts, lambda b: _demo(b, 'llm_fill') < _demo(b, 'pop_fill')),
    'demo100_llm_above_pop': _count(_cuts, lambda b: _demo(b, 'llm_fill') > _demo(b, 'pop_fill')),
    'with_new_share_of_labelled': DATA['dev']['with_new_user_windows'] / DATA['dev']['labelled_user_windows'],
}

NAMES = {'popularity': 'Popularity', 'cbf': 'CBF', 'latest_buy': 'Latest buy',
         'latest_buy_or_sell': 'Latest buy or sell', 'pop_fill': 'Popularity fill (= Latest buy)',
         'tokens_fill': 'Token fill (starter CBF space)', 'tfidf_fill': 'TF-IDF fill',
         'e5_fill': 'mE5-base fill (non-LLM encoder)', 'llm_fill': '**LLM fill (Qwen3-Emb.) — mine**',
         'tfidf_top20': 'TF-IDF order in top-20 popular', 'llm_top20': 'LLM order in top-20 popular',
         'oracle_fill': 'Oracle fill (ceiling, reads labels)'}


def lookup(source, path):
    value = DATA[source]
    for key in path.split('.'):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def fmt(value, spec='.4f'):
    if isinstance(value, str):
        return value
    if spec == 'd':
        return f'{value:,}'
    if spec == 'pct':
        return f'{100 * value:.0f}%'
    return format(value, spec)


def demo_table():
    m = DATA['demo']['methods']
    t = DATA['demo']['timings_local']
    emb = DATA['emb']['models'][0]
    base = DATA['base']['wall_clock_seconds']
    runtime = {b: f'{base:.1f} s (whole starter run, all four)' for b in
               ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell')}
    shared = t['shared_data_and_four_baselines_seconds'] + t['build_vector_spaces_seconds']
    for arm, sec in t['scoring_seconds'].items():
        if isinstance(sec, (int, float)):
            runtime[arm] = f'{sec:.2f} s scoring'
    runtime['llm_fill'] = (f"{t['scoring_seconds']['llm_fill']:.2f} s scoring + "
                           f"{emb['encode_seconds_first']:.1f} s one-off embedding of "
                           f"{emb['texts']} texts (T4)")
    rows = ['| Method | Users | Users with future purchases | nDCG@10: all users | '
            'nDCG@10: nonempty truth | Runtime | Failed cases |',
            '|---|---:|---:|---:|---:|---|---:|']
    order = ['popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell', 'llm_fill',
             'tokens_fill', 'tfidf_fill', 'e5_fill', 'tfidf_top20', 'llm_top20']
    for name in order:
        b = m[name]
        rows.append(f"| {NAMES[name]} | {b['users']} | {b['nonempty_truth_users']} | "
                    f"{b['ndcg_at_10']:.4f} | {b['ndcg_at_10_nonempty']:.4f} | "
                    f"{runtime.get(name, 'with its space')} | {b['failed_cases']} |")
    rows.insert(7, '| *Matched controls (same prefix, other fill space)* | | | | | | |')
    return '\n'.join(rows) + (f'\n\nShared by all fill arms: data loading and space building '
                              f'{shared:.1f} s (local CPU); the LLM model load on Kaggle took '
                              f"{emb['load_seconds']:.1f} s.")


def dev_table():
    d = DATA['dev']
    rows = ['| Fill rule (same `latest_buy` prefix) | nDCG@10 all warm | nDCG@10 labelled | '
            'nDCG@10 with-new (min–max over cutoffs) | New-to-user purchases in Top-10 |',
            '|---|---:|---:|---:|---:|']
    for arm in ('pop_fill', 'tokens_fill', 'tfidf_fill', 'e5_fill', 'llm_fill', 'tfidf_top20',
                'llm_top20', 'oracle_fill'):
        lo, hi = d['range_ndcg_with_new_asset'][arm]
        rows.append(f"| {NAMES[arm]} | {d['macro_ndcg_all_warm'][arm]:.4f} | "
                    f"{d['macro_ndcg_labelled'][arm]:.4f} | "
                    f"{d['macro_ndcg_with_new_asset'][arm]:.4f} ({lo:.4f}–{hi:.4f}) | "
                    f"{d['new_pairs_in_top10'][arm]:,} |")
    return '\n'.join(rows)


def paired_table():
    rows = ['| Paired on with-new user-windows | Better | Worse | Equal | Mean diff. | 95% CI |',
            '|---|---:|---:|---:|---:|---|']
    for key in ('llm_fill-pop_fill', 'llm_fill-tfidf_fill', 'llm_fill-e5_fill', 'e5_fill-pop_fill',
                'tfidf_fill-pop_fill', 'llm_top20-pop_fill'):
        p = DATA['dev']['paired_with_new_pooled'][key]
        a, b = key.split('-')
        rows.append(f"| `{a}` − `{b}` | {p['better']:,} | {p['worse']:,} | {p['equal']:,} | "
                    f"{p['mean_diff']:+.4f} | [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] |")
    return '\n'.join(rows)


TABLES = {'demo': demo_table, 'dev': dev_table, 'paired': paired_table}


def render(md_text):
    md_text = re.sub(r'\{\{table:(\w+)\}\}', lambda m: TABLES[m.group(1)](), md_text)
    return re.sub(r'\{\{(\w+):([^|}]+)(?:\|([^}]+))?\}\}',
                  lambda m: fmt(lookup(m.group(1), m.group(2)), m.group(3) or '.4f'), md_text)

CSS = """
@page { size: A4; }
body { font-family: 'Times New Roman', Times, serif; font-size: 9.4pt; line-height: 1.17; color: #111; }
h1 { font-size: 14pt; text-align: center; margin: 0 0 5px; line-height: 1.18; }
.meta { text-align: center; font-style: italic; font-size: 9pt; margin-bottom: 8px; }
h2 { font-size: 11pt; margin: 9px 0 3px; color: #1f3b63; border-bottom: 1px solid #c9d3e0; }
h3 { font-size: 9.8pt; margin: 7px 0 2px; }
p { margin: 2px 0 4px; text-align: justify; }
ul, ol { margin: 2px 0 4px; padding-left: 16px; }
li { margin: 1px 0; }
blockquote { margin: 3px 12px; padding: 2px 8px; border-left: 3px solid #1f3b63; background: #f2f4f7; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.2pt; background: #f2f4f7; padding: 0 2px; }
pre { font-family: Consolas, 'Courier New', monospace; font-size: 7.8pt; background: #f2f4f7;
      padding: 4px 6px; margin: 3px 0 5px; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 3px 0 5px; }
th, td { border: 1px solid #8fa3bf; padding: 1px 4px; text-align: left; vertical-align: top; }
td.num, th.num { text-align: right; }
th { background: #dde6f2; }
tr.mine td { background: #eef5ea; }
.todo { background: #fff3b0; border: 1px solid #e0b000; padding: 3px 6px; margin: 4px 0; }
span.todo { padding: 0 3px; margin: 0; }
.draft { background: #fde2e1; border: 1px solid #d9534f; padding: 3px 6px; margin: 0 0 6px;
         text-align: center; font-size: 8.6pt; }
.small { font-size: 8.2pt; }
"""


def main():
    md_text = render((HERE / 'report.md').read_text(encoding='utf-8'))
    body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'md_in_html', 'attr_list'])
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Session 2 report — Aleksei Kosychev</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    html_path = HERE / 'report.html'
    html_path.write_text(html, encoding='utf-8')

    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome')
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until='networkidle')
        page.pdf(path=str(OUT_PDF), format='A4', print_background=True,
                 display_header_footer=True, header_template='<span></span>',
                 footer_template='<div style="font-size:8px;width:100%;text-align:center">'
                                 '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                 margin={'top': '15mm', 'bottom': '15mm', 'left': '16mm', 'right': '16mm'})
        browser.close()
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
