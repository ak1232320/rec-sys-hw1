"""Figures for the A06 report, built from experiments/results.json.

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
BLUE_LIGHT, GREY = "#9ec5f4", "#c3c2b7"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

LABELS = {
    "popularity": "Popularity",
    "pagerank": "Global\nPageRank",
    "ppr": "PPR",
    "mf": "MF",
}
ORDER = ["popularity", "pagerank", "ppr", "mf"]
COLOURS = {"popularity": GREY, "pagerank": BLUE_LIGHT, "ppr": BLUE, "mf": ORANGE}

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


def strip(ax, axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def rows_by_method(results):
    rows = results["comparison"]
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    return {row["method"]: row for row in rows}


def figure_recall(results):
    """How alpha was chosen, and what the four methods score on test."""
    sweep = results["alpha_sweep"]["rows"]
    selected = results["alpha_sweep"]["best"]
    methods = rows_by_method(results)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    ax = axes[0]
    alphas = [row["alpha"] for row in sweep]
    recalls = [100 * row["recall"] for row in sweep]
    ax.plot(alphas, recalls, color=BLUE, linewidth=2, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1.3)
    for a, value in zip(alphas, recalls):
        ax.annotate(f"{value:.2f}%", xy=(a, value), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=BLUE, fontweight="bold")
    ax.plot([selected], [100 * next(r["recall"] for r in sweep if r["alpha"] == selected)],
            marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=ORANGE, markeredgewidth=2)
    ax.text(selected, 100 * min(r["recall"] for r in sweep) - 0.09,
            f"selected\nα = {selected:.2f}", ha="center", fontsize=7.5,
            color=ORANGE, fontweight="bold")
    ax.set_xlabel("α (link-following probability)")
    ax.set_ylabel("Validation Recall@10 (%)")
    ax.set_xticks(alphas, [f"{a:.2f}" for a in alphas])
    ax.set_ylim(100 * min(r["recall"] for r in sweep) - 0.25,
                100 * max(r["recall"] for r in sweep) + 0.25)
    ax.set_title("PPR scores best where it is least personal",
                 fontsize=8.5, color=INK, loc="left", pad=6)
    strip(ax)

    ax = axes[1]
    values = [100 * methods[m]["recall"] for m in ORDER]
    bars = ax.bar(np.arange(len(ORDER)), values, width=0.6,
                  color=[COLOURS[m] for m in ORDER])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.1, f"{value:.2f}%",
                ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(np.arange(len(ORDER)), [LABELS[m] for m in ORDER], fontsize=7.5)
    ax.set_ylabel("Test Recall@10 (%)")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title(f"Test cohort: {methods['ppr']['cohortSize']} users, same candidates",
                 fontsize=8.5, color=INK, loc="left", pad=6)
    strip(ax)

    fig.tight_layout(pad=0.4)
    out = FIGURES / "recall.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def figure_character(results):
    """What the lists are made of, and which knob actually changes them."""
    profile = results["popularity_profile"]
    demo = results["two_user_demo"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    # (a) catalogue coverage against how well known the recommended films are
    ax = axes[0]
    # Popularity and global PageRank land almost on top of each other - that near
    # coincidence IS the finding, so their labels are split instead of stacked.
    offsets = {"popularity": (0, 14), "pagerank": (0, -22), "ppr": (0, 14), "mf": (0, 14)}
    for method in ORDER:
        stats = profile[method]
        ax.scatter([stats["meanPopularity"]], [stats["coveragePct"]], s=130,
                   color=COLOURS[method], edgecolor="white", linewidth=1.4, zorder=3)
        ax.annotate(LABELS[method].replace("\n", " "),
                    xy=(stats["meanPopularity"], stats["coveragePct"]),
                    xytext=offsets[method], textcoords="offset points", ha="center",
                    fontsize=7.5, color=INK_MUTED if method == "popularity" else COLOURS[method],
                    fontweight="bold")
    ax.annotate("the same point, within\n1 rating and 0.2 pp",
                xy=(268, 5.5), xytext=(170, 13), textcoords="data", fontsize=7,
                color="#898781", ha="center",
                arrowprops={"arrowstyle": "->", "color": "#c3c2b7", "linewidth": 1})
    ax.set_xlabel("Mean training popularity of a recommended film")
    ax.set_ylabel("Catalogue reached (%)")
    ax.set_xlim(0, 320)
    ax.set_ylim(-3.5, 34)
    ax.set_title("PPR is the only method that leaves the head",
                 fontsize=8.5, color=INK, loc="left", pad=6)
    strip(ax)

    # (b) one factor at a time
    ax = axes[1]
    pairs = [c for c in demo["common"] if c["pair"] != "change seed only vs change alpha only"]
    labels = ["Change the restart\nuser (1 → 2)", "Change α\n(0.85 → 0.50)"]
    shared = [c["shared"] for c in pairs]
    bars = ax.bar(np.arange(len(pairs)), shared, width=0.5, color=[ORANGE, BLUE])
    for bar, value in zip(bars, shared):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25,
                f"{value} of 10 titles\nunchanged", ha="center", fontsize=8,
                color=INK, fontweight="bold")
    ax.set_xticks(np.arange(len(pairs)), labels, fontsize=7.5)
    ax.set_ylabel("Titles shared with the baseline")
    ax.set_ylim(0, 12.5)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_title("The seed decides the list; α only re-scores it",
                 fontsize=8.5, color=INK, loc="left", pad=6)
    strip(ax)

    fig.tight_layout(pad=0.4)
    out = FIGURES / "character.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    figure_recall(results)
    figure_character(results)


if __name__ == "__main__":
    main()
