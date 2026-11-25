"""Export the unique as-of asset texts that the Kaggle notebook embeds.

For every cutoff (the HW2 demo cutoff is one of the nine development cutoffs) the
eligible universe is the starter's: latest asset row strictly before the cutoff and a
positive pre-cutoff close price. Each eligible asset's text is built from that as-of row
only. The union of distinct texts is exported: embedding is per text (no corpus is
fitted), so embedding a text that exists at a later cutoff cannot change the vector of
an earlier one, and each cutoff looks up only its own as-of texts.

Only public asset metadata leaves the machine; no customer or transaction data does.

Run (from hw-module-2-2/):  python src/export_texts.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (DATA, DEV_CUTOFFS, TEXT_FIELDS, TEXT_TEMPLATE, TEXTS_FILE, asset_text,
                    file_sha, load_starter, text_sha)


def eligible_assets(starter, data_dir, cutoff):
    assets = starter.latest_before(data_dir / 'asset_information.csv', 'ISIN', cutoff)
    priced = {r['ISIN'] for r in starter.rows(data_dir / 'close_prices.csv')
              if r['timestamp'][:10] < cutoff and float(r['closePrice']) > 0}
    return {k: v for k, v in assets.items() if k in priced}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--out', type=Path, default=TEXTS_FILE)
    args = parser.parse_args()
    starter = load_starter()
    texts, per_cutoff = {}, {}
    for cutoff in DEV_CUTOFFS:
        assets = eligible_assets(starter, args.data_dir, cutoff)
        assert all(row['timestamp'][:10] < cutoff for row in assets.values())
        shas = set()
        for row in assets.values():
            text = asset_text(row)
            if not text:
                raise ValueError(f'{row["ISIN"]}: empty asset text')
            texts[text_sha(text)] = text
            shas.add(text_sha(text))
        per_cutoff[cutoff] = {'eligible_assets': len(assets), 'distinct_texts': len(shas)}
        print(cutoff, per_cutoff[cutoff])
    payload = {'text_fields': list(TEXT_FIELDS), 'text_template': TEXT_TEMPLATE,
               'source': 'FAR-Trans asset_information.csv, as-of rows (timestamp < cutoff)',
               'asset_information_sha256': file_sha(args.data_dir / 'asset_information.csv'),
               'cutoffs': per_cutoff,
               'texts': [{'sha256': sha, 'text': texts[sha]} for sha in sorted(texts)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{len(texts)} unique texts -> {args.out}')


if __name__ == '__main__':
    main()
