"""Compose the page screenshots taken by tests/check_mf.py into one report figure.

Run:  python experiments/make_app_figure.py   (after tests/check_mf.py)
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "report" / "figures"
OUT = FIGURES / "app.png"

GAP = 16
BORDER = (200, 200, 200)
# The starter's harness beside the finished one. The Top-10 panel is captured
# too (fixed_topk.png) but is deliberately NOT in this figure: the test suite
# trains with a short configuration, so its list differs from the default-config
# list quoted in the report, and showing both would invite a false comparison.
PANELS = ("starter_checks.png", "fixed_checks.png")


def main():
    images = [Image.open(FIGURES / name).convert("RGB") for name in PANELS]
    # Scale every panel to a common height so the row reads as one figure.
    height = max(image.height for image in images)
    scaled = [
        image if image.height == height
        else image.resize((round(image.width * height / image.height), height), Image.LANCZOS)
        for image in images
    ]

    width = sum(image.width for image in scaled) + GAP * (len(scaled) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for image in scaled:
        canvas.paste(image, (x, 0))
        x += image.width + GAP

    framed = Image.new("RGB", (canvas.width + 2, canvas.height + 2), BORDER)
    framed.paste(canvas, (1, 1))
    framed.save(OUT)
    print(f"saved {OUT} {framed.size}")


if __name__ == "__main__":
    main()
