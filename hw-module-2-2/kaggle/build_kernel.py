"""Build the Kaggle kernel folder: embed_assets.py with the asset texts inlined.

Run (from hw-module-2-2/):  python kaggle/build_kernel.py
Then:  kaggle kernels push -p kaggle/kernel
       kaggle kernels status alekseykosychev/far-trans-asset-embeddings
       kaggle kernels output alekseykosychev/far-trans-asset-embeddings -p kaggle/output
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL = HERE / 'kernel'
SLUG = 'far-trans-asset-embeddings'
OWNER = 'alekseykosychev'


def main():
    texts = (HERE / 'input' / 'asset_texts.json').read_bytes()
    blob = base64.b64encode(gzip.compress(texts, mtime=0)).decode('ascii')
    source = (HERE / 'embed_assets.py').read_text(encoding='utf-8')
    marker = 'EMBEDDED_TEXTS_B64 = None  # filled in by build_kernel.py'
    if source.count(marker) != 1:
        raise ValueError('Marker not found exactly once')
    KERNEL.mkdir(exist_ok=True)
    (KERNEL / 'embed_assets.py').write_text(
        source.replace(marker, f'EMBEDDED_TEXTS_B64 = "{blob}"'), encoding='utf-8')
    metadata = {'id': f'{OWNER}/{SLUG}', 'title': SLUG, 'code_file': 'embed_assets.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
                'machine_shape': 'NvidiaTeslaT4',
                'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [],
                'model_sources': []}
    (KERNEL / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=1) + '\n',
                                                 encoding='utf-8')
    print(f'kernel folder ready: {KERNEL} ({len(blob)} base64 chars of texts)')


if __name__ == '__main__':
    main()
