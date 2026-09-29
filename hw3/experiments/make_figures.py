"""Figures for the A03 report, built from experiments/results.json.

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


def strip(ax, axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def figure_models(results):
    """What each direction and each missing-value strategy is worth."""
    e = results["experiments"]
    rows = [
        ("Most-rated (no CF)", e["baselines"]["popularity"], GREY),
        ("User-based / co-rated", e["strategies"]["user_based/corated"], BLUE_LIGHT),
        ("User-based / weighted", e["strategies"]["user_based/weighted"], BLUE),
        ("User-based / mean imp.", e["strategies"]["user_based/mean"], BLUE_LIGHT),
        ("Item-based / co-rated", e["strategies"]["item_based/corated"], "#f5b79b"),
        ("Item-based / weighted", e["strategies"]["item_based/weighted"], ORANGE),
        ("Item-based / mean imp.", e["strategies"]["item_based/mean"], "#f5b79b"),
        ("Matrix factorisation", e["matrix_factorisation"], AQUA),
    ]
    panels = [
        ("Precision@5 (%)", "precision_at_5_pct", "{:.1f}"),
        ("Rating RMSE (lower is better)", "rmse", "{:.3f}"),
        ("Recommendations in the long tail (%)", "long_tail_share_pct", "{:.0f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    y = np.arange(len(rows))
    for ax, (title, key, fmt) in zip(axes, panels):
        values = [row[1][key] or 0 for row in rows]
        bars = ax.barh(y, values, height=0.62, color=[row[2] for row in rows])
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.03, bar.get_y() + bar.get_height() / 2,
                    fmt.format(value), va="center", ha="left", fontsize=7.5, color=INK)
        ax.set_yticks(y, [row[0] for row in rows], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.3)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=6)
        ax.set_xticks([])
        strip(ax)
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        if ax is not axes[0]:
            ax.set_yticklabels([])

    fig.tight_layout(pad=0.4)
    out = FIGURES / "models.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def figure_evidence(results):
    """Cold start on the user side, and unreliable similarity on the pair side."""
    e = results["experiments"]
    cold = e["cold_start"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    # (a) accuracy against how much the active user has rated
    buckets = list(cold["by_profile_size"].keys())
    x = np.arange(len(buckets))
    user_values = [cold["by_profile_size"][b]["user_based"] for b in buckets]
    item_values = [cold["by_profile_size"][b]["item_based"] for b in buckets]
    ax = axes[0]
    ax.bar(x - 0.19, user_values, width=0.36, color=BLUE, label="User-based")
    ax.bar(x + 0.19, item_values, width=0.36, color=ORANGE, label="Item-based")
    for xi, value in zip(x, user_values):
        ax.text(xi - 0.19, value + 0.12, f"{value:.1f}", ha="center", fontsize=7.5, color=BLUE)
    counts = [cold["by_profile_size"][b]["users"] for b in buckets]
    ax.set_xticks(x, [f"{b}\n{n} users" for b, n in zip(buckets, counts)], fontsize=7.5)
    ax.set_ylabel("Precision@5 (%)")
    ax.set_xlabel("Ratings the active user has in the training set")
    ax.set_title("Cold start: nothing works without a profile",
                 fontsize=8.5, color=INK, loc="left", pad=6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    strip(ax, axis="y")

    # (b) similarity is highest exactly where the evidence is thinnest
    evidence = cold["similarity_evidence"]["mean_similarity_by_co_rated"]
    labels = list(evidence.keys())
    values = [evidence[k] for k in labels]
    ax = axes[1]
    colors = [ORANGE if k in ("1", "2") else BLUE for k in labels]
    bars = ax.bar(np.arange(len(labels)), values, width=0.6, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}",
                ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0.9, 1.02)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xlabel("Movies the two users both rated")
    ax.set_title("A pair that overlaps on one film looks perfectly similar",
                 fontsize=8.5, color=INK, loc="left", pad=6)
    share = cold["similarity_evidence"]["share_exactly_1_0_pct"]
    ax.text(0.34, 0.99, f"{share:.0f}% of all positive user pairs\nscore exactly 1.000",
            transform=ax.transAxes, fontsize=7.5, color=ORANGE, va="top", fontweight="bold")
    strip(ax, axis="y")

    fig.tight_layout(pad=0.4)
    out = FIGURES / "evidence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    figure_models(results)
    figure_evidence(results)


if __name__ == "__main__":
    main()
