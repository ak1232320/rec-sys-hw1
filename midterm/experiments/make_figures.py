"""Figures for the midterm report, from experiments/results_watched.json and probe_matrix.json.

Run:  python experiments/make_figures.py   (after run_validation.py and probe_matrix.py)
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "report" / "figures"

# dataviz reference palette, light mode (same as the homework reports)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, GREY = "#9ec5f4", "#c3c2b7"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_MUTED,
    "text.color": INK,
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

SHORT = {
    "popularity": ("Popularity", GREY),
    "history (re-watch baseline)": ("History (re-watch)", GREY),
    "CBF": ("CBF", BLUE),
    "association rules": ("Association rules", BLUE),
    "MF: implicit ALS": ("MF: implicit ALS", BLUE),
    "MF: PureSVD": ("MF: PureSVD", BLUE),
    "PPR (co-view graph)": ("PPR (co-view)", BLUE),
    "habit x schedule (seconds, channel-hours)": ("Channel-hour exposure", BLUE_LIGHT),
    "habit x schedule (watched-airing rate)": ("Habit x schedule (final)", ORANGE),
}


def strip(ax, axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def methods_panel(ax, res):
    rows = [(SHORT[k][0], v["map_cohort"], SHORT[k][1]) for k, v in res["methods"].items() if k in SHORT]
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.66)
    for yi, (_, v, _) in zip(y, rows):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK_MUTED)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, max(r[1] for r in rows) * 1.18)
    ax.set_xlabel("MAP@5, hold-out cohort (weeks 11-20 -> 21-30)")
    ax.set_title("a. Methods under the competition's answer key", loc="left", fontsize=9)
    strip(ax)


def probe_panel(ax, probe):
    order = ["seconds", "views", "airings", "days", "watched"]
    names = {"seconds": "seconds", "views": "views", "airings": "airings", "days": "days",
             "watched": "watched\n>=80%"}
    M = np.array([[probe[f"history by {r}"][c] for c in order] for r in order])
    ax.imshow(M, cmap="Blues", vmin=M.min() - 0.05, vmax=M.max() + 0.02)
    for i in range(len(order)):
        for j in range(len(order)):
            best = M[i, j] == M[:, j].max()
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.2,
                    color="white" if M[i, j] > M.mean() + 0.03 else INK,
                    fontweight="bold" if best else "normal")
    ax.set_xticks(range(len(order)), [names[c] for c in order])
    ax.set_yticks(range(len(order)), [names[r].replace("\n", " ") for r in order])
    ax.set_xlabel("answer key counts ...")
    ax.set_ylabel("history counted by ...")
    ax.set_title("b. One history, counted five ways", loc="left", fontsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    res = json.loads((HERE / "results_watched.json").read_text(encoding="utf-8"))
    probe = json.loads((HERE / "probe_matrix.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), gridspec_kw={"width_ratios": [1.05, 1]})
    methods_panel(axes[0], res)
    probe_panel(axes[1], probe)
    fig.tight_layout(w_pad=2.5)
    out = FIGURES / "fig_methods_probe.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
