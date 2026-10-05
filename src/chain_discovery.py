"""
Feature-based chain discovery via per-feature Matrix Profile, plus the
similarity-based augmentation step (Section 4.5 of the paper).

Also contains the pipeline orchestration function
`run_full_pipeline`, which ties together every stage described in
Algorithm 1 of the paper (preprocessing -> base chain -> C22 features ->
K-D Tree augmentation -> feature-based augmentation -> final filtering).
"""

import numpy as np
import antropy as ant
import stumpy
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .preprocessing import should_skip
from .matrix_profile import discover_stump_chain
from .c22_features import preprocess_feature_profiles
from .kd_tree_augmentation import augment_with_kd_tree
from .final_filtering import filter_final_chain

# ---------------------------------------------------------------- #
# Per-feature chain discovery
# ---------------------------------------------------------------- #

def discover_feature_based_chains(feature_profiles, m):
    """
    Runs Matrix Profile / ALLC chain discovery independently on each of
    the 22 normalized C22 feature trajectories. Features that fail the
    structural complexity check are skipped.
    """
    feature_chain_results = {}

    for feature_idx in tqdm(range(22), desc="Evaluating Features"):
        feature_values = feature_profiles[:, feature_idx]

        if should_skip(feature_values):
            continue

        mp = stumpy.stump(feature_values, m=m)
        _, chain = stumpy.allc(mp[:, 2], mp[:, 3])

        if isinstance(chain, (int, np.integer)):
            chain = [int(chain)]
        elif isinstance(chain, np.ndarray):
            chain = chain.tolist()
        else:
            chain = list(map(int, chain))

        feature_chain_results[feature_idx] = chain

    return feature_chain_results


# ---------------------------------------------------------------- #
# Similarity-based augmentation (range-restricted)
# ---------------------------------------------------------------- #

def has_noise_sampen(ts, threshold=1.0):
    """Flags noisy series via sample entropy (switches similarity metric)."""
    return ant.sample_entropy(ts) > threshold


def is_structurally_similar(subseq1, subseq2, threshold=1.0):
    """Euclidean similarity on min-max scaled subsequences (low-noise)."""
    s1 = MinMaxScaler().fit_transform(subseq1.reshape(-1, 1)).flatten()
    s2 = MinMaxScaler().fit_transform(subseq2.reshape(-1, 1)).flatten()
    return euclidean(s1, s2) < threshold


def is_structurally_similar_p(subseq1, subseq2, threshold=0.5):
    """Pearson correlation similarity (high-noise)."""
    corr, _ = pearsonr(subseq1, subseq2)
    return corr > threshold


def define_ranges(base_chain, ts_length, m):
    """Splits the series into intervals bounded by consecutive base-chain indices."""
    ranges = []
    base_chain = sorted(base_chain)

    if base_chain[0] > 0:
        ranges.append((0, base_chain[0]))
    for i in range(len(base_chain) - 1):
        ranges.append((base_chain[i], base_chain[i + 1]))
    ranges.append((base_chain[-1], ts_length - m))

    return ranges


def evaluate_range(start, end, ref_indices, candidate_pool, ts, m, use_corr):
    """Evaluates candidates in [start, end) against two reference subsequences."""
    candidates = sorted(i for i in candidate_pool if start <= i < end)
    augmented = []

    ref1 = ts[ref_indices[0]:ref_indices[0] + m]
    ref2 = ts[ref_indices[1]:ref_indices[1] + m]

    for candidate in candidates:
        s = ts[candidate:candidate + m]
        if use_corr:
            if is_structurally_similar_p(s, ref1) and is_structurally_similar_p(s, ref2):
                augmented.append(candidate)
        else:
            if is_structurally_similar(s, ref1) and is_structurally_similar(s, ref2):
                augmented.append(candidate)

    return augmented


def augment_chain(ts, base_chain, candidate_pool, m):
    """
    Builds the feature-augmented chain by evaluating candidate indices
    (from feature-based chains + K-D Tree chain) within each interval
    of the base chain, against the two bounding base-chain reference
    subsequences.
    """
    base_chain = sorted(base_chain)
    ranges = define_ranges(base_chain, len(ts), m)
    use_corr = has_noise_sampen(ts)

    augmented = []
    for i, (start, end) in enumerate(ranges):
        if i == 0 or i == 1:
            ref_indices = [base_chain[0], base_chain[1]]
        elif i < len(base_chain) - 1:
            ref_indices = [base_chain[i - 1], base_chain[i]]
        else:
            ref_indices = [base_chain[-2], base_chain[-1]]

        augmented.extend(
            evaluate_range(start, end, ref_indices, candidate_pool, ts, m, use_corr)
        )

    if not augmented:
        return sorted(base_chain)
    return sorted(set(list(base_chain) + list(augmented)))


# ---------------------------------------------------------------- #
# Full pipeline orchestration (Algorithm 1)
# ---------------------------------------------------------------- #

def run_full_pipeline(ts, m, kd_k=5, kd_delta=0.4):
    """
    Runs the complete proposed framework end-to-end:
      1. Base chain discovery (STUMP / ALLC)
      2. C22 feature extraction and normalization
      3. K-D Tree augmentation
      4. Feature-based chain discovery + similarity-based augmentation
      5. Final redundancy filtering

    Returns a dict with intermediate and final results, useful for both
    reproducing figures and for the ablation study.
    """
    result = {
        "base_chain": [],
        "kd_chain": [],
        "feature_chains": {},
        "feature_augmented_chain": [],
        "final_chain": [],
    }

    base_chain = discover_stump_chain(ts, m)
    result["base_chain"] = list(base_chain)
    if len(base_chain) == 0:
        return result  # series skipped; no chain to discover

    feature_profiles = preprocess_feature_profiles(ts, m)

    kd_chain = augment_with_kd_tree(base_chain, feature_profiles, k=kd_k, delta=kd_delta)
    result["kd_chain"] = kd_chain

    feature_chains = discover_feature_based_chains(feature_profiles, m)
    result["feature_chains"] = feature_chains

    feature_pool = sorted(set(sum(feature_chains.values(), [])) | set(kd_chain))
    feature_augmented = augment_chain(ts, base_chain, feature_pool, m)
    result["feature_augmented_chain"] = [int(i) for i in feature_augmented]

    final_chain = filter_final_chain(base_chain, result["feature_augmented_chain"], m)
    result["final_chain"] = final_chain

    return result
