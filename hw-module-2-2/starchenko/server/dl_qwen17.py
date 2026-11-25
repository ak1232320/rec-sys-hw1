"""Download Qwen/Qwen3-1.7B at the course-pinned revision with curl (resume, rate limit)."""
import json, os, subprocess, sys, time, urllib.request

REPO = 'Qwen/Qwen3-1.7B'
REV = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
OUT = '/home/mstarchenko/xai_x/models/qwen3-1.7b'
LIMIT = os.environ.get('DL_LIMIT', '8M')
SKIP = ('.md', '.gitattributes', '.png', '.pdf')

info = json.load(urllib.request.urlopen(
    'https://huggingface.co/api/models/%s/revision/%s?blobs=true' % (REPO, REV), timeout=60))
print('resolved sha', info.get('sha'), flush=True)
files = [(s['rfilename'], s.get('size') or (s.get('lfs') or {}).get('size'))
         for s in info['siblings'] if not s['rfilename'].endswith(SKIP)]
print('files', len(files), 'total %.2f GB' % (sum(sz or 0 for _, sz in files) / 1e9), flush=True)
os.makedirs(OUT, exist_ok=True)
t = time.time()
for f, sz in files:
    out = os.path.join(OUT, f)
    for attempt in range(20):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if sz and have == sz:
            break
        url = 'https://huggingface.co/%s/resolve/%s/%s' % (REPO, REV, f)
        subprocess.run(['curl', '-s', '-L', '-f', '-C', '-', '--limit-rate', LIMIT,
                        '--connect-timeout', '30', '--retry', '5', '-o', out, url])
    print(time.strftime('%H:%M:%S'), f, os.path.getsize(out), 'expected', sz, flush=True)
json.dump({'repo': REPO, 'revision': REV, 'resolved_sha': info.get('sha'),
           'files': files}, open(os.path.join(OUT, 'download_manifest.json'), 'w'), indent=1)
open(os.path.join(OUT, '.ok'), 'w').write(time.strftime('%Y-%m-%d %H:%M:%S'))
print('DONE in %.0f s' % (time.time() - t), flush=True)
