"""Figures for the A02 report, built from experiments/results.json.

Run:  python experiments/make_figures.py   (after run_experiment.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "report" / "figures"

# dataviz reference palette, light mode: categorical slots 1-3 (validated all-pairs)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#9ec5f4"
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


def strip(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def figure_bias(results):
    """Three one-measure panels: what normalising the dot product changes."""
    exp = results["experiment_1_item_to_item"]
    rows = [("Raw overlap", "overlap"), ("Jaccard", "jaccard"), ("Cosine", "cosine")]
    panels = [
        ("Genres per recommended movie", "mean_genres_per_rec", "{:.2f}"),
        ("Rating count of recommended movie", "mean_popularity", "{:.0f}"),
        ("Share of recommendations in the long tail", "long_tail_share_pct", "{:.0f}%"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.85))
    y = np.arange(len(rows))
    for ax, (title, key, fmt) in zip(axes, panels):
        values = [exp[k][key] for _, k in rows]
        colors = [BLUE if k == "cosine" else BLUE_LIGHT for _, k in rows]
        bars = ax.barh(y, values, height=0.6, color=colors)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.03, bar.get_y() + bar.get_height() / 2,
                    fmt.format(value), va="center", ha="left", fontsize=8, color=INK)
        ax.set_yticks(y, [label for label, _ in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.28)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=6)
        ax.set_xticks([])
        strip(ax)
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)

    fig.tight_layout(pad=0.4)
    out = FIGURES / "bias.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def figure_concentration(results):
    """Where the recommended mass sits along the popularity curve."""
    raw = json.loads((HERE / "rec_counts.json").read_text(encoding="utf-8"))
    popularity = np.array(raw["popularity"])
    order = np.argsort(-popularity, kind="stable")
    n = len(order)
    x = 100.0 * np.arange(1, n + 1) / n
    head_x = 100.0 * results["dataset"]["head_items"] / n

    def curve(key):
        counts = np.array(raw["counts"][key], dtype=float)[order]
        return 100.0 * np.cumsum(counts) / counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)

    series = [
        (axes[0], "Averaged profile vs. one active item (quality tie-break)",
         [("Profile-based", "profile_based/quality", ORANGE),
          ("Item-to-item", "item_to_item/quality", BLUE)]),
        (axes[1], "Item-to-item cosine under three tie-breaks",
         [("Rating count", "item_to_item/popularity", ORANGE),
          ("Shrunk rating", "item_to_item/quality", BLUE),
          ("Movie id", "item_to_item/id", AQUA)]),
    ]
    head_index = int(round(head_x / 100 * n)) - 1
    for ax, title, entries in series:
        ax.plot([0, 100], [0, 100], color="#c3c2b7", linewidth=1,
                linestyle=(0, (4, 3)), zorder=1)
        ax.axvline(head_x, color=GRID, linewidth=1, zorder=1)
        ax.text(head_x + 1.5, 3, "head ends", fontsize=7, color="#898781")
        # The key readout is how much of the recommended mass sits inside the
        # popular head, so it is labelled once per series instead of on the curve.
        ax.text(54, 34, "inside the popular head:", fontsize=7, color="#898781", va="top")
        for row, (label, key, color) in enumerate(entries):
            y = curve(key)
            ax.plot(x, y, color=color, linewidth=2, zorder=3, label=label)
            ax.plot([head_x], [y[head_index]], marker="o", markersize=5, color=color,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=4)
            ax.text(54, 27 - 7 * row, f"{y[head_index]:.0f}%  {label}",
                    fontsize=7.5, color=color, fontweight="bold", va="top")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=6)
        ax.set_xlabel("Catalogue, ordered from most to least rated (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 101)
        strip(ax, grid_axis="y")
    axes[0].set_ylabel("Cumulative share of\nrecommendations (%)")

    fig.tight_layout(pad=0.4)
    out = FIGURES / "concentration.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    figure_bias(results)
    figure_concentration(results)


if __name__ == "__main__":
    main()
