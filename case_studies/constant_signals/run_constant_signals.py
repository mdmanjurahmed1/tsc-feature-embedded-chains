"""
Reproduces the Case Study on Constant Data (Section 5.1.5, Fig. 18).

Generates five synthetic, structurally trivial signals (sine wave,
triangle wave, ramp-up, downward parabola, straight line) with no
genuine evolving structure. TSC'17 and TSC'22 baseline chain indices
below are hardcoded from those methods' own runs (they are external
baselines, not part of this codebase, and both incorrectly discover
chains in these trivial signals).

The proposed method's column is NOT hardcoded: it is computed live by
calling should_skip / run_full_pipeline on each signal, demonstrating
that the preprocessing filter (Section 4.1) correctly rejects all five
signals and produces no chains, matching the paper's claim.

Usage:
    python run_constant_signals.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.preprocessing import should_skip
from src.chain_discovery import run_full_pipeline

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

plt.rcParams.update({
    "axes.edgecolor": "gray",
    "axes.linewidth": 0.8,
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "axes.grid": True,
    "font.size": 10,
})

CHAIN_COLORS = ["#ffac03", "#f3681f", "#f53155", "#f9618e",
                "#20bbfd", "#01ccbd", "#01c159", "#9ecb1f"]

N = 501
M = 50


def build_signals():
    """Builds the five trivial synthetic signals used in Fig. 18."""
    t = np.linspace(0, 10 * np.pi, N)

    ts_sine = np.sin(t) * 10 + 50
    ts_triangle = 2 * np.abs(np.mod(np.arange(N), 20) - 10) - 10
    ts_ramp = np.concatenate([np.linspace(0, 50, 250), np.full(250, 50)])
    ts_parabola = 1 - np.linspace(0, 1, N) ** 2
    ts_line = np.full(N, 50)

    return {
        # name: (series, ylim, TSC'17 chain starts, TSC'22 chain starts)
        "Sine Wave": (ts_sine, (40, 60),
                      [26, 126, 226, 326, 426],
                      [76, 176, 276, 376]),
        "Triangle Wave": (ts_triangle, (-12, 12),
                           [1, 21, 41],
                           []),
        "Ramp-up": (ts_ramp, (0, 55),
                    [202, 216, 230, 244],
                    [55, 129, 211]),
        "Parabola Down": (ts_parabola, (0, 1.1),
                           list(range(1, 450, 14)),
                           [94, 145, 196, 247, 298, 349, 400, 451]),
        "Straight Line": (ts_line, (48, 52),
                           [2, 17],
                           []),
    }


def run_proposed_method(ts, m):
    """
    Runs the actual proposed pipeline on a signal and returns the
    final chain's start indices (empty if the signal is rejected by
    preprocessing, as expected for all five trivial signals here).
    """
    skipped = should_skip(ts)
    if skipped:
        return []
    result = run_full_pipeline(ts, m)
    return result["final_chain"]


def plot_row(axes_row, title, ts, ylim, tsc17_chains, tsc22_chains, proposed_chains, m):
    df = pd.Series(ts)

    for ax, chains, label in zip(
        axes_row,
        [tsc17_chains, tsc22_chains, proposed_chains],
        ["TSC'17", "TSC'22", "Feature-Based Embedded (Proposed)"],
    ):
        ax.plot(df, linewidth=1, color="black")
        for i, chain in enumerate(chains):
            color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
            end_idx = min(chain + m, len(df))
            ax.plot(range(chain, end_idx), df.iloc[chain:end_idx],
                     linewidth=3, color=color)
        ax.set_title(f"{title}: {label}")
        ax.set_ylim(ylim)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    signals = build_signals()

    fig, axes = plt.subplots(len(signals), 3, figsize=(16, 10))

    for idx, (title, (ts, ylim, tsc17_chains, tsc22_chains)) in enumerate(signals.items()):
        proposed_chains = run_proposed_method(ts, M)
        print(f"{title}: proposed method chain = {proposed_chains} "
              f"({'skipped by preprocessing' if not proposed_chains else 'NOT skipped'})")

        plot_row(axes[idx], title, ts, ylim, tsc17_chains, tsc22_chains, proposed_chains, M)

    plt.tight_layout()

    out_pdf = os.path.join(OUTPUT_DIR, "Fig18.pdf")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {out_pdf}")


if __name__ == "__main__":
    main()
