"""
Reproduces the Web Query Volume case study (Section 5.1.1, Fig. 10/11).

Usage:
    python run_web_query.py --m 25
    python run_web_query.py --m 50
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.chain_discovery import run_full_pipeline
from src.plotting import plot_chain

DATA_URL = ("https://zenodo.org/record/4276348/files/"
            "Time_Series_Chains_Kohls_data.csv?download=1")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_data():
    """Loads the Web Query (Kohl's) dataset directly from Zenodo."""
    df = pd.read_csv(DATA_URL)
    ts = df["volume"].to_numpy().flatten()
    return df, ts


def main(m):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, ts = load_data()
    print(f"Loaded time series of length {len(ts)}; subsequence length m={m}")

    result = run_full_pipeline(ts, m)

    print("Base chain:", result["base_chain"])
    print("K-D Tree augmented chain:", result["kd_chain"])
    print("Final chain:", result["final_chain"])

    fig = plot_chain(
        ts, result["final_chain"], m,
        title=f"Web Query Volume — Final Discovered Chain (m={m})",
        year_labels=True,
    )
    out_path = os.path.join(OUTPUT_DIR, f"web_query_chain_m{m}.pdf")
    fig.savefig(out_path)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=25,
                         help="Subsequence length (25 or 50 in the paper)")
    args = parser.parse_args()
    main(args.m)
