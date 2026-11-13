"""Figures for the Stage 1 degradation benchmark.

Reads results/stage1_accuracy.csv and writes two panels:
  (a) accuracy against severity, one line per corruption
  (b) accuracy drop at severity 5, ranked

The point the figure has to make is the spread: corruptions sharing a nominal
severity label cost very different amounts of accuracy. Two corruptions are
drawn in colour to anchor that, the rest in grey.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, RESULTS_DIR

HIGHLIGHT = {"gaussian_noise": "#b2453a", "brightness": "#2f6f8f"}


def main():
    df = pd.read_csv(RESULTS_DIR / "stage1_accuracy.csv")
    ref_row = df[df.corruption.isin(["clean", "clean_published"])]
    clean = float(ref_row.accuracy.iloc[0]) if len(ref_row) else None
    d = df[df.severity > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=[1, 1.15])

    # --- (a) accuracy-severity curves ---------------------------------
    for name, g in d.groupby("corruption"):
        g = g.sort_values("severity")
        colour = HIGHLIGHT.get(name)
        ax1.plot(g.severity, g.accuracy,
                 color=colour or "#c4c4c4", lw=2.0 if colour else 1.0,
                 zorder=3 if colour else 1,
                 marker="o" if colour else None, ms=4)
    if clean is not None:
        ax1.axhline(clean, color="#555", ls=":", lw=1, zorder=2)
        # Labelled on the left: the right-hand end of this line is where the
        # brightness curve and its own label sit.
        ax1.text(1.05, clean + 0.015, "clean reference", va="bottom", fontsize=8, color="#555")

    for name, colour in HIGHLIGHT.items():
        row = d[(d.corruption == name) & (d.severity == 5)].iloc[0]
        ax1.annotate(name.replace("_", " "), (5, row.accuracy),
                     xytext=(-6, 8 if name == "brightness" else -14),
                     textcoords="offset points", fontsize=9, color=colour, ha="right")

    ax1.set_xlabel("severity")
    ax1.set_ylabel("top-1 accuracy")
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_ylim(0, 1)
    ax1.set_title("Accuracy under corruption (19 types)", fontsize=10, loc="left")
    ax1.spines[["top", "right"]].set_visible(False)

    # --- (b) ranked drop at the highest severity ----------------------
    s5 = d[d.severity == 5].sort_values("accuracy_drop")
    colours = [HIGHLIGHT.get(n, "#9aa5ad") for n in s5.corruption]
    ax2.barh(range(len(s5)), s5.accuracy_drop * 100, color=colours, height=0.72)
    ax2.set_yticks(range(len(s5)))
    ax2.set_yticklabels([n.replace("_", " ") for n in s5.corruption], fontsize=8)
    ax2.set_xlabel("accuracy drop at severity 5 (percentage points)")
    ax2.set_title("Same severity label, very different cost", fontsize=10, loc="left")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.margins(y=0.01)

    fig.tight_layout()
    out = FIGURES_DIR / "stage1_degradation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
