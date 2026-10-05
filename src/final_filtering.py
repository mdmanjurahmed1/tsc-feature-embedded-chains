"""
Final chain filtering (Section 4.6 of the paper). Enforces temporal
non-redundancy using an exclusion window of +/- 3m/4 around each
selected index (see Table 1, Parameter Summary).
"""


def filter_final_chain(base_chain, feature_augmented_chain, m):
    """
    Starts from the base chain, then greedily accepts augmented
    candidates that fall outside the temporal exclusion window of any
    already-accepted index.
    """
    window = (3 * m) // 4

    filtered_chain = []
    visited_ranges = set()

    for idx in sorted(base_chain):
        idx = int(idx)
        filtered_chain.append(idx)
        visited_ranges.add((idx - window, idx + window))

    augmented_candidates = sorted(set(feature_augmented_chain) - set(base_chain))

    for idx in augmented_candidates:
        idx = int(idx)
        if any(lo <= idx <= hi for lo, hi in visited_ranges):
            continue
        filtered_chain.append(idx)
        visited_ranges.add((idx - window, idx + window))

    filtered_chain.sort()
    return filtered_chain
