"""Run the provided starter CLI unchanged (four baselines) and time the whole run.

The starter itself writes no timing, so this wrapper measures wall-clock for the
complete command (data loading + all four baselines together) and stores it next
to the starter's own outputs, together with the console summary.

Run (from hw-module-2-2/):  python src/run_baseline.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STARTER = ROOT / 'starter' / 'starter.py'
DATA = ROOT.parent / 'hw-module-2-1' / 'data_raw' / 'FAR-Trans'
OUT = ROOT / 'outputs' / 'baseline'


def main():
    cmd = [sys.executable, str(STARTER), 'run', '--data-dir', str(DATA), '--output-dir', str(OUT)]
    started = time.perf_counter()
    done = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    seconds = time.perf_counter() - started
    print(done.stdout, end='')
    if done.returncode:
        print(done.stderr, file=sys.stderr)
        sys.exit(done.returncode)
    (OUT / 'console.txt').write_text(done.stdout, encoding='utf-8')
    timing = {'scope': 'whole starter run: data loading plus all four baselines together; '
                       'not a per-method timing',
              'wall_clock_seconds': round(seconds, 2),
              'command': 'python starter/starter.py run --data-dir <FAR-Trans> --output-dir outputs/baseline',
              'python': platform.python_version(), 'platform': platform.platform(),
              'processor': platform.processor()}
    (OUT / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n', encoding='utf-8')
    print(f'wall-clock {seconds:.2f} s (whole run)')


if __name__ == '__main__':
    main()
