"""Crop the Kaggle leaderboard screenshot for the report.

Keeps the competition header and the "Your Recent Submission" card (file name,
account, public and private score); drops the cookie banner and the desktop.

Run:  python experiments/make_kaggle_figure.py
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "manual" / "kaggle_leaderboard.png"
OUT = HERE.parent / "report" / "figures" / "kaggle_score.png"
LEFT, RIGHT = 30, 1270
BANDS = [(0, 302), (448, 648)]  # header + tabs; the score card (rows in the 1475 x 1012 shot)

shot = Image.open(SRC).convert("RGB")
parts = [shot.crop((LEFT, top, RIGHT, bottom)) for top, bottom in BANDS]
image = Image.new("RGB", (RIGHT - LEFT, sum(p.height for p in parts)), "white")
y = 0
for part in parts:
    image.paste(part, (0, y))
    y += part.height
framed = Image.new("RGB", (image.width + 2, image.height + 2), (200, 200, 200))
framed.paste(image, (1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
framed.save(OUT)
print("saved", OUT, framed.size)
