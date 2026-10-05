"""
Base chain discovery via the Matrix Profile (STUMP / ALLC), corresponding
to Section 4.2 of the paper.
"""

import stumpy
from .preprocessing import should_skip


def discover_stump_chain(ts, m):
    """
    Computes the Matrix Profile for `ts` with subsequence length `m` and
    extracts the base chain via the ALLC algorithm.

    Returns an empty list if the series fails the structural complexity
    check (see preprocessing.should_skip). This structural filter is
    part of the proposed framework (Section 4.1), not of TSC'17 itself;
    use discover_stump_chain_raw() to reproduce TSC'17's own behavior
    with no such filtering (as used in the "Base only (TSC'17)"
    ablation variant).
    """
    if should_skip(ts):
        print("Skipping STUMP calculation: low complexity or periodicity.")
        return []

    matrix_profile = stumpy.stump(ts, m)
    _, stump_chain = stumpy.allc(matrix_profile[:, 2], matrix_profile[:, 3])
    return stump_chain


def discover_stump_chain_raw(ts, m):
    """
    Computes the base chain via Matrix Profile + ALLC with NO
    structural complexity filtering. TSC'17 itself has no equivalent
    of the proposed framework's should_skip preprocessing filter, so
    this is the most literal reproduction of TSC'17's own behavior.

    Not used by ablation/run_ablation.py: that script's "Base only
    (TSC'17)" variant instead uses the standard, should_skip-filtered
    discover_stump_chain, matching the master "Full C22MP Ablation"
    notebook where all four variants share one discover_stump_chain
    definition. Empirically the two are equivalent for the synthetic
    benchmark series used throughout this repo, since should_skip(ts)
    is False at all 101 noise levels tested. This function is kept
    for exact fidelity to the separate standalone notebook that
    defines Base-only without should_skip, and for use on any other
    series where the distinction might actually matter.
    """
    matrix_profile = stumpy.stump(ts, m)
    _, stump_chain = stumpy.allc(matrix_profile[:, 2], matrix_profile[:, 3])
    return stump_chain
