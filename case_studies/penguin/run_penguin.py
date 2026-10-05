"""
Reproduces the Penguin Activity case study (Section 5.1.2, Fig. 12).

Data: Magellanic penguin dive telemetry, 22.5-second dive sampled at
40 Hz, X-axis acceleration channel (Ponganis et al., 2015;
Williams et al., 2012).

Place the original 'penguinshort.mat' file (MATLAB struct with key
'penguinshort') under data/ before running. A copy is not bundled in
this repository for licensing reasons; see data/README.md for the
source.

Usage:
    python run_penguin.py --m 25
"""

import argparse
import os
import sys

import numpy as np
import scipy.io

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.chain_discovery import run_full_pipeline
from src.plotting import plot_chain

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "penguinshort.mat")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_data():
    """Loads the penguin dive acceleration series from the cached .mat file."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Expected penguinshort.mat at {DATA_PATH}. "
            "See data/README.md for how to obtain and cache it."
        )
    data = scipy.io.loadmat(DATA_PATH)
    ts = data["penguinshort"].flatten().astype(np.float64)
    return ts


def main(m):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ts = load_data()
    print(f"Loaded penguin acceleration series of length {len(ts)}; m={m}")

    result = run_full_pipeline(ts, m)

    print("Base chain:", result["base_chain"])
    print("K-D Tree augmented chain:", result["kd_chain"])
    print("Final chain:", result["final_chain"])

    fig = plot_chain(
        ts, result["final_chain"], m,
        title=f"Penguin Activity — Final Discovered Chain (m={m})",
    )
    out_path = os.path.join(OUTPUT_DIR, f"penguin_chain_m{m}.pdf")
    fig.savefig(out_path)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=25,
                         help="Subsequence length (m=25, ~0.6s, in the paper)")
    args = parser.parse_args()
    main(args.m)
