"""
Reproduces the Ablation and Efficiency Analysis (Section 5.3, Table 3).

Runs four pipeline variants across 101 noise levels (0% to 100%, 1%
steps) on the synthetic benchmark, tracking F1-score, runtime, and
peak memory for each:

  - "Base only (TSC'17)"      : base chain only (no augmentation at all)
  - "w/o Feature-Based Chain" : base chain + K-D Tree augmentation only
  - "w/o KD-Tree"             : base chain + feature-based chain
                                 augmentation only (no K-D Tree)
  - "Full Model"              : the complete proposed pipeline

Runtime and memory are measured only around the chain-discovery
computation itself (matching the original notebook), excluding
synthetic data generation and noise injection.

Usage:
    python run_ablation.py
    python run_ablation.py --variant "w/o KD-Tree"   # run a single variant
"""

import argparse
import os
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import stumpy

from src.synthetic_data import generate_synthetic_chain, add_noise, GROUND_TRUTH_CHAIN_INDICES
from src.matrix_profile import discover_stump_chain
from src.c22_features import preprocess_feature_profiles
from src.kd_tree_augmentation import augment_with_kd_tree
from src.chain_discovery import discover_feature_based_chains, augment_chain
from src.final_filtering import filter_final_chain
from src.metrics import compute_performance_metrics

M = 50
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

VARIANTS = [
    "Base only (TSC'17)",
    "w/o Feature-Based Chain",
    "w/o KD-Tree",
    "Full Model",
]


def run_variant(ts, m, variant):
    """
    Runs one ablation variant on a single (already noised) series,
    returning the detected chain indices. Only the steps relevant to
    the variant are executed.

    All four variants use the same discover_stump_chain (with the
    should_skip structural filter applied), matching the master
    "Full C22MP Ablation" notebook, where all four variants are
    switchable blocks sharing one discover_stump_chain definition.
    (A separate standalone notebook for the Base-only variant defines
    discover_stump_chain without should_skip; empirically this makes
    no difference here, since should_skip(ts) is False at all 101
    noise levels for this synthetic benchmark series, so both
    versions are equivalent for this experiment.)
    """
    stump_chain = discover_stump_chain(ts, m)

    if len(stump_chain) == 0:
        return []

    if variant == "Base only (TSC'17)":
        # No augmentation of any kind: equivalent to TSC'17's own
        # chain discovery.
        return sorted(int(i) for i in stump_chain)

    feature_profiles = preprocess_feature_profiles(ts, m)

    if variant == "w/o Feature-Based Chain":
        # K-D Tree augmentation only (no per-feature MP chain
        # discovery), but the structural-similarity augment_chain step
        # still runs, using the K-D Tree candidates as its pool.
        kd_chain = augment_with_kd_tree(stump_chain, feature_profiles, k=5, delta=0.4)
        feature_pool = sorted(set(kd_chain))
        augmented = augment_chain(ts, stump_chain, feature_pool, m)
        augmented = [int(i) for i in augmented]
        return filter_final_chain(stump_chain, augmented, m)

    if variant == "w/o KD-Tree":
        # Feature-based chain discovery and augmentation only, no
        # K-D Tree augmentation.
        feature_chains = discover_feature_based_chains(feature_profiles, m)
        feature_pool = sorted(set(sum(feature_chains.values(), [])))
        augmented = augment_chain(ts, stump_chain, feature_pool, m)
        augmented = [int(i) for i in augmented]
        return filter_final_chain(stump_chain, augmented, m)

    if variant == "Full Model":
        kd_chain = augment_with_kd_tree(stump_chain, feature_profiles, k=5, delta=0.4)
        feature_chains = discover_feature_based_chains(feature_profiles, m)
        feature_pool = sorted(set(sum(feature_chains.values(), [])) | set(kd_chain))
        augmented = augment_chain(ts, stump_chain, feature_pool, m)
        augmented = [int(i) for i in augmented]
        return filter_final_chain(stump_chain, augmented, m)

    raise ValueError(f"Unknown variant: {variant}")


def run_variant_across_noise(variant, noise_levels=None):
    """
    Runs one variant across all noise levels, tracking F1, runtime,
    and peak memory at each level.
    """
    if noise_levels is None:
        noise_levels = [round(i / 100, 2) for i in range(101)]

    results = []
    for noise_level in noise_levels:
        # Note: this outer seed call mirrors the original notebook; it
        # has no practical effect since generate_synthetic_chain
        # reseeds internally (seed=42) as its first step, so the
        # underlying series is always identical regardless.
        np.random.seed(int(noise_level * 100))

        time_series_data = generate_synthetic_chain(n_series=100, length=100, k=20, m=M, seed=42)
        ts_original = time_series_data[0]
        ts = add_noise(ts_original, noise_level)

        tracemalloc.start()
        start = time.perf_counter()

        detected_indices = run_variant(ts, M, variant)

        end = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        runtime = round(end - start, 4)
        memory_mb = round(peak / 1024 / 1024, 4)

        precision, recall, f1 = compute_performance_metrics(
            GROUND_TRUTH_CHAIN_INDICES, detected_indices, M
        )

        print(f"[{variant}] Noise {int(noise_level * 100)}% | "
              f"F1={f1:.4f} | Runtime={runtime}s | Memory={memory_mb}MB")

        results.append({
            "noise_level_pct": int(noise_level * 100),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "runtime_s": runtime,
            "memory_mb": memory_mb,
        })

    return pd.DataFrame(results)


def warm_up_numba():
    """
    stumpy's Matrix Profile computation is numba-JIT-compiled on its
    first call in a fresh process, which can take tens of seconds and
    would otherwise contaminate the first timed measurement. Running
    one throwaway call here absorbs that one-time cost before timing
    begins, matching best practice for benchmarking numba-jitted code.
    """
    import stumpy
    dummy = np.random.rand(200)
    stumpy.stump(dummy, 20)
    print("Numba warm-up complete.")


def main(variant_filter=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    warm_up_numba()
    variants_to_run = [variant_filter] if variant_filter else VARIANTS

    summary_rows = []
    for variant in variants_to_run:
        df = run_variant_across_noise(variant)

        safe_name = variant.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").replace("'", "")
        out_path = os.path.join(OUTPUT_DIR, f"ablation_{safe_name}.xlsx")
        df.to_excel(out_path, index=False)
        print(f"Saved {out_path}")

        summary_rows.append({
            "Variant": variant,
            "Avg F1": round(df["f1_score"].mean(), 4),
            "Avg Runtime (s)": round(df["runtime_s"].mean(), 4),
            "Avg Memory (MB)": round(df["memory_mb"].mean(), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "table3_ablation_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)
    print("\nTable 3 (summary):\n", summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default=None,
                         choices=VARIANTS,
                         help="Run a single variant instead of all four")
    args = parser.parse_args()
    main(args.variant)
