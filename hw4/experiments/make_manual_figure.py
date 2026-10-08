"""Compose the manual-verification screenshots (manual/) into one report figure.

Run:  python experiments/make_manual_figure.py
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
MANUAL = HERE.parent / "manual"
OUT = HERE.parent / "report" / "figures" / "manual_check.png"

GAP = 14
BORDER = (200, 200, 200)
# Two rows: the 1% / 30% run next to the self-check log, then the selected rule next
# to the 0.5% / 10% run without the lift filter.
ROWS = [["image copy.png", "image.png"], ["image copy 3.png", "image copy 4.png"]]


def row_image(names):
    images = [Image.open(MANUAL / name).convert("RGB") for name in names]
    height = images[0].height
    scaled = [im if im.height == height else im.resize((round(im.width * height / im.height), height))
              for im in images]
    canvas = Image.new("RGB", (sum(im.width for im in scaled) + GAP * (len(scaled) - 1), height), "white")
    x = 0
    for im in scaled:
        canvas.paste(im, (x, 0))
        x += im.width + GAP
    return canvas


def main():
    rows = [row_image(names) for names in ROWS]
    width = max(r.width for r in rows)
    rows = [r.resize((width, round(r.height * width / r.width))) for r in rows]
    canvas = Image.new("RGB", (width, sum(r.height for r in rows) + GAP), "white")
    canvas.paste(rows[0], (0, 0))
    canvas.paste(rows[1], (0, rows[0].height + GAP))
    framed = Image.new("RGB", (canvas.width + 2, canvas.height + 2), BORDER)
    framed.paste(canvas, (1, 1))
    framed.save(OUT)
    print(f"saved {OUT} {framed.size}")


if __name__ == "__main__":
    main()
