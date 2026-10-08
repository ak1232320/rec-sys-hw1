"""Figures for the A04 report, built from experiments/results.json.

Run:  python experiments/make_figures.py   (after run_experiment.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "report" / "figures"

# dataviz reference palette, light mode: categorical slots 1-3 (validated all-pairs)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_1, BLUE_2, BLUE_3 = "#9ec5f4", "#5598e7", "#1c5cab"   # ordinal: itemset size
GREY = "#c3c2b7"
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.85}
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


def figure_thresholds(results):
    sweep = results["sweep"]
    supports = sorted({r["minSupport"] for r in sweep})
    xs = [s * 100 for s in supports]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 2.7))

    for conf, color in ((0.1, AQUA), (0.3, BLUE), (0.6, ORANGE)):
        ys = [next(r["rules"] for r in sweep if r["minSupport"] == s and r["minConfidence"] == conf)
              for s in supports]
        keep = [(x, y) for x, y in zip(xs, ys) if y > 0]
        left.plot([k[0] for k in keep], [k[1] for k in keep], color=color, linewidth=2,
                  marker="o", markersize=4.5, label=f"confidence ≥ {conf:.0%}")
    left.scatter([1], [950], s=70, facecolor="none", edgecolor=INK, linewidth=1, zorder=5)
    left.annotate("chosen: 1% / 30%\n950 rules", (1, 950), xytext=(14, 14), textcoords="offset points",
                  fontsize=7.5, color=INK, bbox=LABEL_BOX)
    left.set_yscale("log")
    left.set_xlabel("minimum support, % of baskets")
    left.set_ylabel("rules kept")
    left.set_title("Rules against the two thresholds", fontsize=9, loc="left")
    left.legend(frameon=False, fontsize=7.5, loc="upper right")
    strip(left)

    sizes = {s: next(r["itemsetsBySize"] for r in sweep if r["minSupport"] == s) for s in supports}
    one = [sizes[s][0] for s in supports]
    two = [sizes[s][1] if len(sizes[s]) > 1 else 0 for s in supports]
    more = [sum(sizes[s][2:]) for s in supports]
    pos = range(len(supports))
    right.bar(pos, one, color=BLUE_1, width=0.62, label="1 item", edgecolor="white", linewidth=1)
    right.bar(pos, two, bottom=one, color=BLUE_2, width=0.62, label="2 items", edgecolor="white", linewidth=1)
    right.bar(pos, more, bottom=[a + b for a, b in zip(one, two)], color=BLUE_3, width=0.62,
              label="3–6 items", edgecolor="white", linewidth=1)
    for p, a, b, c in zip(pos, one, two, more):
        right.annotate(f"{a + b + c:,}", (p, a + b + c), xytext=(0, 2), textcoords="offset points",
                       ha="center", fontsize=7, color=INK_MUTED)
    right.set_xticks(list(pos), [f"{x:g}%" for x in xs])
    right.set_xlabel("minimum support")
    right.set_ylabel("frequent itemsets")
    right.set_title("Frequent itemsets by size (downward closure)", fontsize=9, loc="left")
    right.legend(frameon=False, fontsize=7.5)
    strip(right)
    fig.tight_layout()
    fig.savefig(FIGURES / "thresholds.png", dpi=200)
    plt.close(fig)


def figure_rules(results):
    data = json.loads((HERE.parent / "transactions.js").read_text(encoding="utf-8").split("window.HW4 = ", 1)[1].rstrip().rstrip(";"))
    describe = dict(zip(data["stocks"], data["descriptions"]))
    n = results["dataset"]["n"]
    compact = json.loads((HERE / "reference_rules.json").read_text(encoding="utf-8"))["tables"]["0.01/0.3"]
    rules = [{"antecedent": a, "consequent": b, "jointCount": j, "lift": (j / ca) / (cb / n)}
             for a, b, j, ca, cb in compact]

    def kind(r):
        names = [describe[i] for i in r["antecedent"] + r["consequent"]]
        if all(x.startswith("HERB MARKER") for x in names):
            return "herb"
        if sorted(r["antecedent"] + r["consequent"]) == ["22386", "85099B"]:
            return "useful"
        return "other"

    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    other = [r for r in rules if kind(r) == "other"]
    ax.scatter([r["jointCount"] for r in other], [r["lift"] for r in other], s=9, color=GREY,
               alpha=0.8, linewidth=0, label="other rules at 1% / 30% (lift > 1 for all 950)")
    herb = [r for r in rules if kind(r) == "herb"]
    ax.scatter([r["jointCount"] for r in herb], [r["lift"] for r in herb], s=30, color=ORANGE,
               edgecolor="white", linewidth=1, zorder=4, label="HERB MARKER pairs (rejected)")
    useful = [r for r in rules if kind(r) == "useful"]
    ax.scatter([r["jointCount"] for r in useful], [r["lift"] for r in useful], s=40, color=BLUE,
               edgecolor="white", linewidth=1, zorder=4, label="JUMBO BAG PINK → RED (investigated)")
    xs = list(range(165, 600, 5))
    ax.plot(xs, [n / x for x in xs], color=INK_MUTED, linewidth=1, linestyle="--")
    ax.annotate("ceiling: lift ≤ N / count(A∪B)", (300, n / 300), xytext=(2, -14),
                textcoords="offset points", fontsize=7.5, color=INK_MUTED)
    ax.annotate("THYME → ROSEMARY\nlift 85.1, conf 94.6%", (176, 85.06), xytext=(14, -26),
                textcoords="offset points", fontsize=7.5, color=INK, bbox=LABEL_BOX)
    ax.annotate("PINK → RED: conf 62.8%\nRED → PINK: conf 34.5%\nlift 6.77 both ways",
                (546, 6.77), xytext=(-104, -24), textcoords="offset points", fontsize=7.5, color=INK, bbox=LABEL_BOX)
    ax.set_yscale("log")
    ax.set_xlabel("count(A∪B): baskets that contain the whole rule")
    ax.set_ylabel("lift (log scale)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    strip(ax, "both")
    fig.tight_layout()
    fig.savefig(FIGURES / "rules.png", dpi=200)
    plt.close(fig)


def figure_quarters(results):
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 2.5))
    for key, color, label in (("useful", BLUE, "JUMBO BAG PINK → RED"),
                              ("misleading", ORANGE, "HERB MARKER THYME → ROSEMARY")):
        q = results[key]["quarters"]
        names = [r["quarter"].replace("20", "", 1) for r in q]
        left.plot(names, [r["confidence"] * 100 for r in q], color=color, linewidth=2, marker="o",
                  markersize=4.5, label=label)
        right.plot(names, [r["lift"] for r in q], color=color, linewidth=2, marker="o",
                   markersize=4.5, label=label)
        for name, r in zip(names, q):
            right.annotate(f"{r['joint']}", (name, r["lift"]), xytext=(0, 5), textcoords="offset points",
                           ha="center", fontsize=6.8, color=INK_MUTED)
    left.axhline(30, color=INK_MUTED, linewidth=0.8, linestyle=":")
    left.annotate("threshold 30%", (0, 30), xytext=(0, 3), textcoords="offset points", fontsize=7,
                  color=INK_MUTED)
    left.set_ylim(0, 105)
    left.set_ylabel("confidence, %")
    left.set_title("Confidence per quarter", fontsize=9, loc="left")
    left.legend(frameon=False, fontsize=7.5, loc="lower right")
    right.set_yscale("log")
    right.set_ylabel("lift (log scale)")
    right.set_title("Lift per quarter (labels: count(A∪B))", fontsize=9, loc="left")
    strip(left)
    strip(right)
    fig.tight_layout()
    fig.savefig(FIGURES / "quarters.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    figure_thresholds(results)
    figure_rules(results)
    figure_quarters(results)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
