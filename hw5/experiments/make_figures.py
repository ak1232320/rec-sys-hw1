"""Figures for the A05 report, built from experiments/results.json.

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


def strip(ax, axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def figure_training(results):
    """The curve, the checkpoint it picks, and what the extra epochs cost."""
    training = results["training"]
    history = training["history"]
    best = training["best"]
    epochs = [h["epoch"] for h in history]
    train = [h["trainRMSE"] for h in history]
    val = [h["valRMSE"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    ax = axes[0]
    ax.plot(epochs, train, color=BLUE, linewidth=2, label="Training")
    ax.plot(epochs, val, color=ORANGE, linewidth=2, label="Validation")
    ax.axhline(training["baseline"]["val"], color=GREY, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(epochs[-1], training["baseline"]["val"] + 0.012,
            f"mean-rating baseline {training['baseline']['val']:.3f}",
            fontsize=7.5, color="#898781", ha="right")
    ax.plot([best["epoch"]], [best["valRMSE"]], marker="o", markersize=7, color=ORANGE,
            markeredgecolor="white", markeredgewidth=1.5, zorder=4)
    ax.annotate(f"selected: epoch {best['epoch']}\nval {best['valRMSE']:.4f}",
                xy=(best["epoch"], best["valRMSE"]), xytext=(14, 14),
                textcoords="offset points", fontsize=7.5, color=ORANGE, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_xticks([1] + list(range(5, max(epochs) + 1, 5)))
    ax.set_title("Validation RMSE bottoms out after two epochs",
                 fontsize=8.5, color=INK, loc="left", pad=6)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    strip(ax)

    ax = axes[1]
    labels = ["Mean-rating\nbaseline", "Last epoch\n(no selection)", f"Epoch {best['epoch']}\n(selected)"]
    values = [training["baseline"]["test"], training["finalEpochTestRMSE"], training["testRMSE"]]
    colors = [GREY, BLUE_LIGHT, BLUE]
    bars = ax.bar(np.arange(3), values, width=0.6, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.4f}",
                ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(np.arange(3), labels, fontsize=7.5)
    ax.set_ylim(0.9, max(values) * 1.06)
    ax.set_ylabel("Test RMSE")
    ax.set_title("What the checkpoint is worth on the test split",
                 fontsize=8.5, color=INK, loc="left", pad=6)
    strip(ax)

    fig.tight_layout(pad=0.4)
    out = FIGURES / "training.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def figure_discovery(results):
    """Who the recommended films are, and what an evidence floor does to them."""
    analysis = results["analysis"]
    sweep = results["evidence_sweep"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    # (a) how well known the recommended films are
    ax = axes[0]
    labels = ["Films the user\nalready rated", "Films the model\nrecommends"]
    values = [analysis["meanOwnPopularity"], analysis["meanRecPopularity"]]
    bars = ax.bar(np.arange(2), values, width=0.55, color=[BLUE, ORANGE])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.0f}",
                ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.axhline(analysis["headCut"], color=GREY, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(1.46, analysis["headCut"] + 4, f"top-20% cut: {analysis['headCut']}",
            fontsize=7.5, color="#898781", ha="right")
    ax.set_xticks(np.arange(2), labels, fontsize=7.5)
    ax.set_ylabel("Ratings in the training split")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("Recommended films are far less rated than the user's own",
                 fontsize=8.5, color=INK, loc="left", pad=6)
    strip(ax)

    # (b) what a minimum-evidence floor buys
    ax = axes[1]
    x = np.arange(len(sweep))
    recall = [100 * row["recall"] for row in sweep]
    ax.plot(x, recall, color=BLUE, linewidth=2, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1.2)
    for xi, value, row in zip(x, recall, sweep):
        ax.annotate(f"{value:.1f}%", xy=(xi, value), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=BLUE, fontweight="bold")
    ax.set_xticks(x, [f"≥ {row['minCount']}\n{row['candidatePool']} films" for row in sweep],
                  fontsize=7.5)
    ax.set_xlabel("Training ratings a candidate must have")
    ax.set_ylabel("Recall@10 (%)")
    ax.set_ylim(0, max(recall) * 1.3)
    ax.set_title("Requiring evidence from the candidate, not the user",
                 fontsize=8.5, color=INK, loc="left", pad=6)
    strip(ax)

    fig.tight_layout(pad=0.4)
    out = FIGURES / "discovery.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    figure_training(results)
    figure_discovery(results)


if __name__ == "__main__":
    main()
