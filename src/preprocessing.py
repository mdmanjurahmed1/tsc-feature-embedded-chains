"""
Structural filtering for the proposed TSC discovery framework.

Implements the preprocessing criteria described in Section 4.1 of the paper:
  - Constant signal detection
  - High-periodicity detection (FFT energy ratio)
  - Oscillatory / waveform structure detection
  - Permutation entropy (also catches constant-slope / monotonic ramps)
"""

import math
import numpy as np
from scipy.stats import entropy
from scipy.fftpack import fft


def _permutation_entropy(x, m=4, delay=2):
    """Computes normalized permutation entropy (Bandt & Pompe, 2002)."""
    n = len(x)
    permutations = []
    for i in range(n - (m - 1) * delay):
        segment = x[i:i + m * delay:delay]
        permutation = tuple(np.argsort(segment))
        permutations.append(permutation)

    _, counts = np.unique(permutations, return_counts=True, axis=0)
    if len(counts) < 2:
        return 0.0  # Perfectly structured order (e.g. monotonic ramp)

    probs = counts / counts.sum()
    return entropy(probs, base=2) / np.log2(math.factorial(m))


def _fft_energy_ratio(x):
    """Dominant-frequency energy ratio, used for periodicity detection."""
    n = len(x)
    fft_values = np.abs(fft(x))[:n // 2]
    fft_values[0] = 0  # ignore DC component
    return np.max(fft_values) / np.sum(fft_values)


def _is_waveform(x, tolerance=0.1):
    """Checks for regular oscillatory structure via peak/trough spacing."""
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0]
    troughs = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0]

    peak_diff = np.diff(peaks)
    trough_diff = np.diff(troughs)

    if len(peak_diff) > 1 and len(trough_diff) > 1:
        peak_std = np.std(peak_diff) / np.mean(peak_diff)
        trough_std = np.std(trough_diff) / np.mean(trough_diff)
        return peak_std < tolerance and trough_std < tolerance
    return False


def should_skip(series, pe_threshold=0.4, m=4, delay=2,
                 fft_energy_threshold=0.7, decimals=4):
    """
    Determines whether a (sub)series lacks sufficient structural complexity
    to be worth passing through chain discovery.

    Detects: constant signals, constant-slope (monotonic) signals,
    strongly periodic signals (e.g. sine waves), and regular waveforms.

    Returns True if the series should be SKIPPED.
    """
    x = np.round(series, decimals)

    # Constant amplitude
    if np.ptp(x) == 0:
        return True

    # Strong periodicity (FFT)
    if _fft_energy_ratio(x) > fft_energy_threshold:
        return True

    # Regular oscillatory waveform
    if _is_waveform(x):
        return True

    # Permutation entropy: catches low-complexity signals, including
    # constant-slope / monotonic ramps, which collapse to a single
    # repeating ordinal pattern and thus have near-zero entropy.
    pe = _permutation_entropy(x, m, delay)
    return pe < pe_threshold
