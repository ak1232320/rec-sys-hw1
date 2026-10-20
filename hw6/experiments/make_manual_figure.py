"""Compose the manual-verification screenshots into one report figure.

Takes the two self-check panels captured on the published site — the starter and
the submitted version — keeps each one's address bar so the figure shows where
it was taken, and stacks them. Wide screenshots do not survive being scaled into
an A4 column, so each panel is cropped to the lines that carry the evidence.

Run:  python experiments/make_manual_figure.py
"""
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
MANUAL = HERE.parent / "manual"
OUT = HERE.parent / "report" / "figures" / "manual_check.png"

GAP = 14
BORDER = (200, 200, 200)
TEXT_WIDTH = 960      # enough for "HW6 self-checks - pass 12, fail 0, pending 0 (of 12)"
HEADER_ROWS = 150     # the summary line plus the backend / MF reference lines
BODY_ROWS = 170       # the first few check lines underneath

# (file, address-bar band, first row of the code box)
PANELS = [
    ("image copy 2.png", (0, 0, 1232, 42), 268),
    ("image copy 3.png", (0, 0, 1206, 42), 345),
]


def trim_left(image):
    """First column that is not page background, so the gutter is dropped."""
    background = Image.new("RGB", image.size, image.getpixel((2, 2)))
    box = ImageChops.difference(image, background).convert("L").point(
        lambda v: 255 if v > 18 else 0).getbbox()
    return box[0] if box else 0


def build(name, url_band, body_top):
    image = Image.open(MANUAL / name).convert("RGB")
    url = image.crop(url_band)

    band = image.crop((0, body_top, image.width, body_top + HEADER_ROWS + BODY_ROWS))
    left = trim_left(band)
    body = band.crop((left, 0, min(left + TEXT_WIDTH, band.width), band.height))

    width = max(url.width, body.width)
    canvas = Image.new("RGB", (width, url.height + body.height), "white")
    canvas.paste(url, (0, 0))
    canvas.paste(body, (0, url.height))
    return canvas


def main():
    panels = [build(*panel) for panel in PANELS]
    width = max(panel.width for panel in panels)
    scaled = [
        panel if panel.width == width
        else panel.resize((width, round(panel.height * width / panel.width)), Image.LANCZOS)
        for panel in panels
    ]

    height = sum(panel.height for panel in scaled) + GAP * (len(scaled) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in scaled:
        canvas.paste(panel, (0, y))
        y += panel.height + GAP

    framed = Image.new("RGB", (canvas.width + 2, canvas.height + 2), BORDER)
    framed.paste(canvas, (1, 1))
    framed.save(OUT)
    print(f"saved {OUT} {framed.size}")


if __name__ == "__main__":
    main()
