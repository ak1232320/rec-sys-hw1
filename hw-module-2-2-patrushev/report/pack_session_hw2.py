"""Derived AI-session export for HW8 (project Session 2). Run AFTER the Claude Code session ends:

    python3 ~/Documents/вшэ/recsys_course/hw7/report/pack_session.py

The Claude Code session that produced this homework also contains the earlier midterm and Session 1 work, so the
export is derived: it keeps the original JSON Lines unmodified from the first user message of this
homework ("…в этом репо части ДЗ моих коллег… сделай дз") to the end of the session. Writes, next to this script:
  patrushev_konstantin_session1_session.jsonl  - the slice itself (native JSONL, unmodified lines)
  patrushev_konstantin_session1_session.json   - upload file for the LMS, which rejects raw logs by
      content: {what, derived, sha256_of_full_session, sha256_of_slice, restore, data}, where data is
      base64 of the xz-compressed slice
and checks that decoding the upload file gives back the slice byte for byte.
"""
import base64
import hashlib
import json
import lzma
import pathlib

SESSION = pathlib.Path.home() / ".claude/projects/-Users-konstantin-Documents-----recsys-course/8c34e5a1-52dd-4d84-beda-c2ce400d5147.jsonl"
DST = pathlib.Path(__file__).resolve().parent
STEM = "patrushev_konstantin_session2_session"
MARKER = "в этом репо части ДЗ моих коллег"

raw = SESSION.read_bytes()
lines = raw.splitlines(keepends=True)
start = None
for i, line in enumerate(lines):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("type") != "user":
        continue
    c = d.get("message", {}).get("content")
    text = c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict)) if isinstance(c, list) else ""
    if MARKER in text:
        start = i
        break
assert start is not None, "HW8 start not found"
part = b"".join(lines[start:])
(DST / f"{STEM}.jsonl").write_bytes(part)
sha = lambda b: hashlib.sha256(b).hexdigest()
doc = {
    "what": "Claude Code session log (JSON Lines) for LLM4Rec project Session 2 (HW8), Konstantin Patrushev",
    "derived": (f"lines {start + 1}-{len(lines)} of {len(lines)} of the original session file "
                f"{SESSION.name}, unmodified; earlier lines belong to the midterm and Session 1 (HW7) and are omitted"),
    "sha256_of_full_session": sha(raw),
    "sha256_of_slice": sha(part),
    "restore": ("python3 -c \"import json,base64,lzma;d=json.load(open('" + STEM + ".json'));"
                "open('" + STEM + ".jsonl','wb').write(lzma.decompress(base64.b64decode(d['data'])))\""),
    "data": base64.b64encode(lzma.compress(part, preset=9)).decode("ascii"),
}
out = DST / f"{STEM}.json"
out.write_text(json.dumps(doc, indent=1))
back = lzma.decompress(base64.b64decode(json.loads(out.read_text())["data"]))
print("slice     :", f"lines {start + 1}-{len(lines)}", f"{len(part) / 1e6:.1f} MB", "sha256", sha(part))
print("upload    :", out.name, f"{out.stat().st_size / 1e6:.1f} MB")
print("roundtrip identical:", back == part)
print("contains a Kaggle token pattern (KGAT_):", b"KGAT_" in part)
