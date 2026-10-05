"""
Evaluation metrics for comparing discovered chains against ground truth,
used in Section 5.2 (Synthetic Benchmark Evaluation, Table 2) and
Section 5.3 (Ablation and Efficiency Analysis, Table 3).

A detected subsequence counts as a true positive if it overlaps a
ground-truth subsequence by more than 40% of the subsequence length
(overlap-based greedy matching).
"""

import numpy as np


def compute_performance_metrics(ground_truth_indices, detected_indices, subseq_len,
                                 overlap_threshold=0.4):
    """
    Computes precision, recall, and F1-score between detected chain
    indices and ground-truth chain indices, using overlap-based
    matching.

    Returns (precision, recall, f1).
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    def overlap(idx_a, idx_b):
        return max(0, min(idx_a + subseq_len, idx_b + subseq_len) - max(idx_a, idx_b))

    for detected_idx in detected_indices:
        hit = any(overlap(detected_idx, ground_idx) > overlap_threshold * subseq_len
                   for ground_idx in ground_truth_indices)
        if hit:
            true_positives += 1
        else:
            false_positives += 1

    for ground_idx in ground_truth_indices:
        hit = any(overlap(detected_idx, ground_idx) > overlap_threshold * subseq_len
                   for detected_idx in detected_indices)
        if not hit:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) \
        if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) \
        if (true_positives + false_negatives) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) \
        if (precision + recall) > 0 else 0

    return precision, recall, f1


def compute_performance_metrics_one_to_one(ground_truth_indices, detected_indices, subseq_len,
                                             overlap_threshold=0.4):
    """
    Computes precision, recall, and F1-score with STRICT one-to-one
    matching: each ground-truth point can be claimed by at most one
    detected point. If multiple detected points overlap the same
    ground-truth point, only the first (in detection order) counts as
    a true positive; the rest count as false positives.

    This differs from compute_performance_metrics (used in
    synthetic_benchmark/ and ablation/), which counts true positives
    independently on the detected side and the ground-truth side,
    allowing a single ground-truth point to be "hit" by more than one
    detected point without penalty. The two give identical results
    whenever every ground-truth point is matched by at most one
    detected point (true for the large majority of reconstructions
    across all seven real-world datasets), but diverge when two
    detected points independently fall within the overlap threshold of
    the same ground-truth point, as happens for two of the five
    FreezerSmallTrain reconstructions. Cross-checking against the
    paper's published Table 4 values for exactly those two
    reconstructions confirmed this one-to-one version is the correct
    match, so it is used throughout real_world_benchmark/ for
    consistency across all seven datasets.

    Returns (precision, recall, f1).
    """
    ground_truth_indices = list(ground_truth_indices)
    claimed = set()
    true_positives = 0
    false_positives = 0

    def overlap(idx_a, idx_b):
        return max(0, min(idx_a + subseq_len, idx_b + subseq_len) - max(idx_a, idx_b))

    for detected_idx in detected_indices:
        matched_gt = None
        for ground_idx in ground_truth_indices:
            if ground_idx in claimed:
                continue
            if overlap(detected_idx, ground_idx) > overlap_threshold * subseq_len:
                matched_gt = ground_idx
                break
        if matched_gt is not None:
            claimed.add(matched_gt)
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(ground_truth_indices) - len(claimed)

    precision = true_positives / (true_positives + false_positives) \
        if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) \
        if (true_positives + false_negatives) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) \
        if (precision + recall) > 0 else 0

    return precision, recall, f1
