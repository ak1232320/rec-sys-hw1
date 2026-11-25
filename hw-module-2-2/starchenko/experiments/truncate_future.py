#!/usr/bin/env python3
"""Leakage test helper: copy the four FAR-Trans CSVs keeping only rows dated before the cutoff.

If prompts and prefix inputs built from the truncated copy are byte-identical to those built
from the full data, no row at or after the cutoff can have reached the model input.
Usage: python truncate_future.py DATA_DIR OUT_DIR [CUTOFF]
"""
import csv
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cutoff = sys.argv[3] if len(sys.argv) > 3 else '2021-06-01'
dst.mkdir(parents=True, exist_ok=True)
for name in ('transactions.csv', 'asset_information.csv', 'customer_information.csv',
             'close_prices.csv'):
    kept = dropped = 0
    with (src / name).open(encoding='utf-8-sig', newline='') as fin, \
            (dst / name).open('w', encoding='utf-8', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['timestamp'][:10] < cutoff:
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1
    print(f'{name}: kept {kept}, dropped {dropped} rows dated on or after {cutoff}')
