"""
Reproduces the Real-World Benchmark Evaluation (Section 5.2,
Table 4 / tab:realworld_results).

Evaluates the proposed method on seven UCR Time Series Archive
datasets (TwoLeadECG, ECG200, Wafer, Plane, FreezerRegularTrain,
FreezerSmallTrain, TwoPatterns), comparing against TSC'22 and TSC'17
using precision/recall/F1 with 40%-overlap ground-truth matching.

For each dataset, five reconstructed time series are evaluated and the
results averaged, following the paper's protocol: ground-truth chains
are derived from TSC'22's own chain node indices for each
reconstruction.

*** DATA AND GROUND TRUTH NOT YET FILLED IN ***
This script was not built against your original data files, since only
the pipeline code and a single Wafer-loading example (m=152) were
shared in the working session that produced this repository, not the
actual 35 reconstructed CSV files (7 datasets x 5 reconstructions) or
the TSC'22 ground-truth chain node indices used to build them.

To complete this:
  1. Place each dataset's 5 reconstructed CSVs under
     data/<dataset_name>/<dataset_name>_{1..5}.csv, each with a
     "ts_final" column (matching your Wafer example).
  2. Fill in DATASET_CONFIG below with the correct m (subsequence
     length) for each dataset, and the ground-truth chain node
     indices for each reconstruction (from TSC'22's own output).

Usage:
    python run_real_world_benchmark.py
    python run_real_world_benchmark.py --dataset Wafer
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.matrix_profile import discover_stump_chain
from src.c22_features import preprocess_feature_profiles
from src.kd_tree_augmentation import augment_with_kd_tree
from src.chain_discovery import discover_feature_based_chains, augment_chain
from src.final_filtering import filter_final_chain
from src.metrics import compute_performance_metrics_one_to_one

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# TODO: fill in m (subsequence length) and ground_truth (TSC'22 chain
# node indices) for every dataset. ground_truth is a SINGLE list shared
# across all 5 reconstructions of a dataset (confirmed against the
# Plane dataset: one idx_tsc list, evaluated against all 5
# reconstructions via the 40%-overlap matching criterion, exactly
# reproducing the paper's reported per-reconstruction P/R/F1 values).
# Only Plane and Wafer's m are known from shared examples; the other
# five datasets' m values and ground_truth entries are placeholders
# and MUST be replaced.
DATASET_CONFIG = {
    "TwoLeadECG":       {"category": "ECG",       "m": 82,
                          "ground_truth": [100, 625, 1704, 1946, 2673, 3632, 4936, 6237, 8179, 8638]},
    "ECG200":           {"category": "ECG",       "m": 96,
                          "ground_truth": [100, 639, 1732, 1988, 2729, 3702, 5020, 6335, 8291, 8764]},
    "Wafer":            {"category": "Sensor",    "m": 152,
                          "ground_truth": [100, 695, 1844, 2156, 2953, 3982, 5356, 6727, 8739, 9268]},
    "Plane":            {"category": "Sensor",    "m": 144,
                          "ground_truth": [100, 687, 1828, 2132, 2921, 3942, 5308, 6671, 8675, 9196]},
    "FreezerRegularTrain": {"category": "Sensor", "m": 301,
                          "ground_truth": [100, 844, 2142, 2603, 3549, 4727, 6250, 7770, 9931, 10609]},
    "FreezerSmallTrain":   {"category": "Sensor", "m": 301,
                             "ground_truth": [100, 844, 2142, 2603, 3549, 4727, 6250, 7770, 9931, 10609]},
    "TwoPatterns":      {"category": "Simulated", "m": 128,
                          "ground_truth": [100, 671, 1796, 2084, 2857, 3862, 5212, 6559, 8547, 9052]},
}


def load_reconstruction(dataset_name, recon_idx):
    """Loads one reconstructed time series (1-indexed) for a dataset."""
    path = os.path.join(DATA_DIR, dataset_name, f"{dataset_name}_{recon_idx}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected reconstructed series at {path}. "
            "See this script's module docstring for what's needed."
        )
    df = pd.read_csv(path)
    return df["ts_final"].to_numpy().flatten()


def run_proposed_method(ts, m):
    """Runs the full proposed pipeline, returning the final chain indices."""
    stump_chain = discover_stump_chain(ts, m)
    if len(stump_chain) == 0:
        return []

    feature_profiles = preprocess_feature_profiles(ts, m)
    kd_chain = augment_with_kd_tree(stump_chain, feature_profiles, k=5, delta=0.4)
    feature_chains = discover_feature_based_chains(feature_profiles, m)
    feature_pool = sorted(set(sum(feature_chains.values(), [])) | set(kd_chain))
    augmented = augment_chain(ts, stump_chain, feature_pool, m)
    augmented = [int(i) for i in augmented]
    return filter_final_chain(stump_chain, augmented, m)


def evaluate_dataset(dataset_name):
    """
    Evaluates the proposed method on all 5 reconstructions of one
    dataset, matching each against the dataset's single shared
    ground-truth chain (TSC'22's chain node indices), and returns the
    averaged precision/recall/F1.
    """
    config = DATASET_CONFIG[dataset_name]
    m = config["m"]
    ground_truth = config["ground_truth"]

    if m is None or ground_truth is None:
        raise ValueError(
            f"{dataset_name}: m or ground_truth not filled in. "
            "See DATASET_CONFIG and this script's module docstring."
        )

    ground_truth = np.array(ground_truth)

    precisions, recalls, f1s = [], [], []
    for recon_idx in range(1, 6):
        ts = load_reconstruction(dataset_name, recon_idx)
        detected = run_proposed_method(ts, m)

        p, r, f1 = compute_performance_metrics_one_to_one(ground_truth, detected, m)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        print(f"  {dataset_name} recon {recon_idx}: P={p:.2f} R={r:.2f} F1={f1:.2f}")

    return {
        "Dataset": dataset_name,
        "Category": config["category"],
        "Precision": round(np.mean(precisions), 2),
        "Recall": round(np.mean(recalls), 2),
        "F1": round(np.mean(f1s), 2),
    }


def main(dataset_filter=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datasets = [dataset_filter] if dataset_filter else list(DATASET_CONFIG.keys())

    results = []
    for dataset_name in datasets:
        print(f"\n=== {dataset_name} ===")
        results.append(evaluate_dataset(dataset_name))

    results_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "table4_realworld_results.xlsx")
    results_df.to_excel(out_path, index=False)
    print("\nResults:\n", results_df.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                         choices=list(DATASET_CONFIG.keys()),
                         help="Evaluate a single dataset instead of all seven")
    args = parser.parse_args()
    main(args.dataset)
