"""
Reproduces the Battery Discharge Capacity Degradation case study
(Section 5.1.4, new case study added in response to reviewer Comment 1).

Data: NASA PCoE Battery Dataset, cell B0006 (Saha and Goebel, 2007).
A cached, pre-extracted discharge-capacity series is provided in
data/Discharge_Capacity.csv so reviewers do not need to re-run the
.mat extraction or re-download the original dataset.

Usage:
    python run_battery.py --m 18
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.chain_discovery import run_full_pipeline
from src.plotting import plot_chain

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Discharge_Capacity.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_data():
    """Loads the cached NASA B0006 discharge capacity series."""
    df = pd.read_csv(DATA_PATH)
    ts = df["Discharge_Capacity"].to_numpy().flatten()
    return ts


def main(m):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ts = load_data()
    print(f"Loaded discharge capacity series of length {len(ts)}; m={m}")

    result = run_full_pipeline(ts, m)

    print("Base chain:", result["base_chain"])
    print("K-D Tree augmented chain:", result["kd_chain"])
    print("Final chain:", result["final_chain"])

    fig = plot_chain(
        ts, result["final_chain"], m,
        title=f"NASA B0006 Discharge Capacity — Final Discovered Chain (m={m})",
    )
    out_path = os.path.join(OUTPUT_DIR, f"battery_chain_m{m}.pdf")
    fig.savefig(out_path)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=18,
                         help="Subsequence length (18 in the paper)")
    args = parser.parse_args()
    main(args.m)
