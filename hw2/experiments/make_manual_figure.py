"""Compose the two manual-verification screenshots into one report figure.

Run:  python experiments/make_manual_figure.py
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
MANUAL = HERE.parent / "manual"
OUT = HERE.parent / "report" / "figures" / "manual_check.png"

GAP = 10
BORDER = (200, 200, 200)


def main():
    terminal = Image.open(MANUAL / "1.png").convert("RGB")
    browser = Image.open(MANUAL / "2.png").convert("RGB")

    # Keep the browser shot down to the green status line; the profile-vector
    # panel below it is cut off by the viewport anyway.
    browser = browser.crop((0, 0, browser.width, 900))

    width = browser.width
    scale = width / terminal.width
    terminal = terminal.resize((width, round(terminal.height * scale)), Image.LANCZOS)

    canvas = Image.new("RGB", (width, terminal.height + GAP + browser.height), "white")
    canvas.paste(terminal, (0, 0))
    canvas.paste(browser, (0, terminal.height + GAP))

    # hairline frame so the white browser chrome does not bleed into the page
    framed = Image.new("RGB", (canvas.width + 2, canvas.height + 2), BORDER)
    framed.paste(canvas, (1, 1))
    framed.save(OUT)
    print(f"saved {OUT} {framed.size}")


if __name__ == "__main__":
    main()
