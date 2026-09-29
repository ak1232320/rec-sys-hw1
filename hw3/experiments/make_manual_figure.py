"""Compose the manual-verification screenshots into one report figure.

Run:  python experiments/make_manual_figure.py
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
MANUAL = HERE.parent / "manual-test"
OUT = HERE.parent / "report" / "figures" / "manual_check.png"

GAP = 14
BORDER = (200, 200, 200)
# Crop each screenshot down to its two result columns; the controls above them are
# already described in the caption and would only shrink the readable part.
CROPS = {"4.png": 288, "5.png": 276}


def main():
    panels = []
    for name, top in CROPS.items():
        image = Image.open(MANUAL / name).convert("RGB")
        panels.append(image.crop((0, top, image.width, image.height)))

    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels) + GAP * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + GAP

    framed = Image.new("RGB", (canvas.width + 2, canvas.height + 2), BORDER)
    framed.paste(canvas, (1, 1))
    framed.save(OUT)
    print(f"saved {OUT} {framed.size}")


if __name__ == "__main__":
    main()
