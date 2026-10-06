"""
Reproduces the Tilt Table Data case study (Section 5.1.3, Fig. 13-16).

Data: arterial blood pressure (ABP) signal recorded during a tilt table
test, transitioning from a steady (pre-tilt) to a drifting (post-tilt)
physiological state (dataset from TSC'17, Zhu et al. 2017).

Also reproduces the noise-robustness check (Fig. 15, 16): fixed-seed
Gaussian noise scaled to 5% of the series' standard deviation is added
to the clean signal.

Place the original 'tilt_table.mat' file (MATLAB struct with key
'tilt_table') under data/ before running. A copy is not bundled in
this repository for licensing reasons; see data/README.md for the
source.

Usage:
    python run_tilt_table.py --m 200
    python run_tilt_table.py --m 200 --noise
"""

import argparse
import os
import sys

import numpy as np
import scipy.io

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.chain_discovery import run_full_pipeline
from src.plotting import plot_chain

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tilt_table.mat")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_data():
    """Loads the tilt table ABP series from the cached .mat file."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Expected tilt_table.mat at {DATA_PATH}. "
            "See data/README.md for how to obtain and cache it."
        )
    data = scipy.io.loadmat(DATA_PATH)
    ts = data["tilt_table"].flatten().astype(np.float64)
    return ts


def inject_noise(ts, seed=42, noise_pct=0.05):
    """
    Injects fixed-seed Gaussian noise scaled to `noise_pct` of the
    series' standard deviation, exactly as used to produce Fig. 15/16.
    """
    np.random.seed(seed)
    std_dev = np.std(ts)
    noise = np.random.normal(loc=0, scale=noise_pct * std_dev, size=ts.shape)
    return ts + noise


def main(m, noisy):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ts = load_data()
    label = "clean"
    if noisy:
        ts = inject_noise(ts, seed=42, noise_pct=0.05)
        label = "noisy_pm5pct"

    print(f"Loaded tilt table ABP series ({label}), length {len(ts)}; m={m}")

    result = run_full_pipeline(ts, m)

    print("Base chain:", result["base_chain"])
    print("K-D Tree augmented chain:", result["kd_chain"])
    print("Final chain:", result["final_chain"])

    fig = plot_chain(
        ts, result["final_chain"], m,
        title=f"Tilt Table ABP ({label}) — Final Discovered Chain (m={m})",
    )
    out_path = os.path.join(OUTPUT_DIR, f"tilt_table_chain_{label}_m{m}.pdf")
    fig.savefig(out_path)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=200,
                         help="Subsequence length (m=200, ~1 cardiac cycle, in the paper)")
    parser.add_argument("--noise", action="store_true",
                         help="Inject +/-5%% noise before running (Fig. 15/16)")
    args = parser.parse_args()
    main(args.m, args.noise)
