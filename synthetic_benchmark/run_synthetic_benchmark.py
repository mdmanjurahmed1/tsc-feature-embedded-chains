"""
Reproduces the Synthetic Benchmark Evaluation (Section 5.2, Fig. 19,
Table 2).

Runs the proposed method across 101 noise levels (0% to 100%, in 1%
steps) on the synthetic ground-truth chain series, computing precision,
recall, and F1-score at each level.

TSC'17's F1 curve is an external baseline (not part of this codebase)
and is loaded from data/tsc17_baseline_f1.csv, whose values were
cross-checked against the 11 points published in Table 2 of the paper
(all matched exactly). This is the same 101-point resolution used to
produce Fig. 19's bottom panel in the original submission.

Usage:
    python run_synthetic_benchmark.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.synthetic_data import generate_synthetic_chain, add_noise, GROUND_TRUTH_CHAIN_INDICES
from src.chain_discovery import run_full_pipeline
from src.metrics import compute_performance_metrics

M = 50
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
TSC17_BASELINE_PATH = os.path.join(os.path.dirname(__file__), "data", "tsc17_baseline_f1.csv")

# Table 2 is reported at these 10%-increment noise levels
TABLE2_NOISE_LEVELS = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]


def run_proposed_method_across_noise():
    """
    Runs the proposed method across 101 noise levels (0% to 100%, 1%
    steps) on the synthetic benchmark series, exactly mirroring the
    original evaluation loop: the synthetic generator is reseeded
    (seed=42) on every iteration so the underlying series is always
    identical, while noise is drawn from the running numpy random
    state (not reseeded per-iteration), matching the original code.
    Running the full 101-iteration loop from a fresh process is fully
    deterministic and reproducible.
    """
    results = []

    for i in range(101):
        noise_level = round(i / 100, 2)

        time_series_data = generate_synthetic_chain(n_series=100, length=100, k=20, m=M, seed=42)
        ts_original = time_series_data[0]
        ts_noisy = add_noise(ts_original, noise_level)  # no reseed: matches original loop

        pipeline_result = run_full_pipeline(ts_noisy, M)
        detected_indices = pipeline_result["final_chain"]

        precision, recall, f1 = compute_performance_metrics(
            GROUND_TRUTH_CHAIN_INDICES, detected_indices, M
        )

        print(f"Noise Level: {int(noise_level * 100)}% | "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        results.append({
            "noise_level": noise_level,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

    return pd.DataFrame(results)


def build_table2(proposed_df, tsc17_df):
    """
    Builds Table 2: precision/recall/F1 for both methods at 10%
    noise-level increments.
    """
    proposed_subset = proposed_df[proposed_df["noise_level"].isin(TABLE2_NOISE_LEVELS)]
    tsc17_subset = tsc17_df[tsc17_df["noise_level"].isin(TABLE2_NOISE_LEVELS)]

    table = pd.merge(
        tsc17_subset, proposed_subset, on="noise_level",
        suffixes=("_tsc17", "_proposed"),
    )
    return table.sort_values("noise_level").reset_index(drop=True)


def plot_fig19(ts_original, ts_noisy_100pct, proposed_df, tsc17_df):
    """Reproduces Fig. 19: (top) original vs. 100%-noise series, (bottom) F1 vs. noise."""
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 7))

    # Top panel
    ax_top.plot(ts_original, color="black", label="Original Time Series")
    ax_top.plot(ts_noisy_100pct, color="red", alpha=0.6, label="Noisy Time Series (100% noise)")
    ax_top.legend()
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Bottom panel
    noise_pct = (proposed_df["noise_level"] * 100).astype(int)
    ax_bottom.plot(noise_pct, proposed_df["f1_score"], label="Feature-Embedded (Proposed)",
                    marker="o", color="#D62728", linewidth=1)
    ax_bottom.plot((tsc17_df["noise_level"] * 100).astype(int), tsc17_df["f1_score"],
                    label="TSC'17", marker="^", color="#9467BD", linewidth=1)
    ax_bottom.set_xlabel("Noise Level (%)")
    ax_bottom.set_ylabel("F1 Score")
    ax_bottom.set_xticks(range(0, 101, 10))
    ax_bottom.set_yticks([round(x * 0.1, 1) for x in range(0, 11)])
    ax_bottom.set_ylim(0, 1.05)
    ax_bottom.grid(True, linestyle="--", alpha=0.6)
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)
    ax_bottom.legend(fontsize=10)

    fig.tight_layout()
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Live-run the proposed method across all 101 noise levels
    proposed_df = run_proposed_method_across_noise()
    proposed_df.to_csv(os.path.join(OUTPUT_DIR, "proposed_method_f1_101.csv"), index=False)

    # Load the TSC'17 baseline (external method, cached results)
    tsc17_df = pd.read_csv(TSC17_BASELINE_PATH)

    # Table 2
    table2 = build_table2(proposed_df, tsc17_df)
    table2.to_excel(os.path.join(OUTPUT_DIR, "table2_precision_recall_f1.xlsx"), index=False)
    print("\nTable 2:\n", table2.to_string(index=False))

    # Fig. 19 top panel data: regenerate the same example series/100% noise
    time_series_data = generate_synthetic_chain(n_series=100, length=100, k=20, m=M, seed=42)
    ts_original = time_series_data[0]
    ts_noisy_100pct = add_noise(ts_original, 1.0, seed=42)

    fig = plot_fig19(ts_original, ts_noisy_100pct, proposed_df, tsc17_df)
    fig_path = os.path.join(OUTPUT_DIR, "Fig19.pdf")
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    print(f"\nSaved Fig. 19 to {fig_path}")


if __name__ == "__main__":
    main()
