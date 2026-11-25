"""Embed FAR-Trans as-of asset texts with two pinned open-weight models (Kaggle GPU).

Input : the unique asset texts exported by src/export_texts.py (public asset metadata
        only). `build_kernel.py` inlines them below as EMBEDDED_TEXTS_B64 so the Kaggle
        kernel needs no dataset; locally `--texts` can point at asset_texts.json.
Output: <model_key>.npy (float32, one L2-normalised row per text, in texts_sha256 order)
        and manifest.json (models, pinned revisions, versions, hardware, timings).

Both models are frozen; nothing is trained. Each model encodes the texts twice to
record run-to-run stability. No user, transaction or label data is involved.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

EMBEDDED_TEXTS_B64 = None  # filled in by build_kernel.py

MODELS = [
    {'key': 'qwen3_embedding_0.6b', 'repo': 'Qwen/Qwen3-Embedding-0.6B',
     'revision': '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3', 'input_prefix': '',
     'role': 'primary: LLM-based embedder (Qwen3-0.6B backbone, last-token pooling); '
             'asset texts are encoded as documents, without a query instruction'},
    {'key': 'multilingual_e5_base', 'repo': 'intfloat/multilingual-e5-base',
     'revision': '835193815a3936a24a0ee7dc9e3d48c1fbb19c55', 'input_prefix': 'query: ',
     'role': 'secondary: non-LLM dense encoder (XLM-RoBERTa base, mean pooling); the model card '
             'advises the "query: " prefix for symmetric similarity'},
]
BATCH_SIZE = 32


def load_texts(path):
    if EMBEDDED_TEXTS_B64 is not None:
        payload = json.loads(gzip.decompress(base64.b64decode(EMBEDDED_TEXTS_B64)))
    else:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    for item in payload['texts']:
        if hashlib.sha256(item['text'].encode('utf-8')).hexdigest() != item['sha256']:
            raise ValueError('Text does not match its sha256')
    return payload


def environment(torch):
    import numpy, sentence_transformers, transformers
    info = {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'sentence_transformers': sentence_transformers.__version__,
            'numpy': numpy.__version__, 'cuda_available': torch.cuda.is_available()}
    if torch.cuda.is_available():
        info.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                    gpu_count=torch.cuda.device_count())
    try:
        with open('/proc/cpuinfo') as stream:
            info['cpu'] = next(line.split(':', 1)[1].strip() for line in stream
                               if line.startswith('model name'))
    except (OSError, StopIteration):
        info['cpu'] = platform.processor()
    return info


def embed(spec, texts, device, torch):
    import numpy as np
    import transformers
    from huggingface_hub import try_to_load_from_cache
    from sentence_transformers import SentenceTransformer
    dtype_kw = 'dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'
    started = time.perf_counter()
    model = SentenceTransformer(spec['repo'], revision=spec['revision'], device=device,
                                model_kwargs={dtype_kw: torch.float32})
    load_seconds = time.perf_counter() - started
    config_path = try_to_load_from_cache(spec['repo'], 'config.json', revision=spec['revision'])
    resolved = Path(str(config_path)).parent.name if isinstance(config_path, str) else None
    inputs = [spec['input_prefix'] + t for t in texts]
    lengths = [len(ids) for ids in model.tokenizer(inputs)['input_ids']]
    runs, seconds = [], []
    for _ in range(2):
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        runs.append(model.encode(inputs, batch_size=BATCH_SIZE, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False))
        if device == 'cuda':
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
    matrix = runs[0].astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f'{spec["key"]}: non-finite embedding')
    record = dict(spec, resolved_snapshot=resolved,
                  modules=[type(m).__name__ for m in model],
                  pooling=getattr(model[1], 'get_pooling_mode_str', lambda: None)(),
                  max_seq_length=model.max_seq_length,
                  padding_side=getattr(model.tokenizer, 'padding_side', None),
                  parameter_dtype=str(next(model.parameters()).dtype),
                  parameters=sum(p.numel() for p in model.parameters()),
                  dimension=int(matrix.shape[1]), texts=len(texts),
                  input_tokens_max=max(lengths), input_tokens_mean=sum(lengths) / len(lengths),
                  truncated_texts=sum(n > model.max_seq_length for n in lengths),
                  normalize_embeddings=True, batch_size=BATCH_SIZE,
                  load_seconds=round(load_seconds, 2),
                  encode_seconds_first=round(seconds[0], 3),
                  encode_seconds_repeat=round(seconds[1], 3),
                  repeat_max_abs_diff=float(np.abs(runs[0] - runs[1]).max()),
                  row_norm_min=float(np.linalg.norm(matrix, axis=1).min()),
                  row_norm_max=float(np.linalg.norm(matrix, axis=1).max()))
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return matrix, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--texts', default='asset_texts.json')
    parser.add_argument('--out', default=None)
    args, _ = parser.parse_known_args()
    out = Path(args.out or ('/kaggle/working' if Path('/kaggle/working').is_dir() else 'output'))
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    payload = load_texts(args.texts)
    texts = [t['text'] for t in payload['texts']]
    manifest = {'texts_sha256': [t['sha256'] for t in payload['texts']],
                'text_template': payload['text_template'],
                'asset_information_sha256': payload['asset_information_sha256'],
                'environment': environment(torch), 'device': device, 'models': []}
    print(json.dumps(manifest['environment'], indent=1), flush=True)
    for spec in MODELS:
        matrix, record = embed(spec, texts, device, torch)
        np.save(out / f'{spec["key"]}.npy', matrix)
        record['npy_sha256'] = hashlib.sha256((out / f'{spec["key"]}.npy').read_bytes()).hexdigest()
        manifest['models'].append(record)
        print(json.dumps(record, indent=1), flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    print('done', out, flush=True)


if __name__ == '__main__':
    sys.exit(main())
