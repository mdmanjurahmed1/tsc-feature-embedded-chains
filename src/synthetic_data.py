"""
Synthetic benchmark data generation for Section 5.2 (Synthetic Benchmark
Evaluation, Fig. 19, Table 2) and the ablation/noise-robustness study
(Section 5.3, Table 3).

Each generated series embeds k=20 ground-truth chain elements of length
m=50, built from distorted sine waves, following the synthetic
evaluation protocol adapted from TSC'17 (Zhu et al., 2017).
"""

import numpy as np


def generate_synthetic_chain(n_series=100, length=100, k=20, m=50, seed=42):
    """
    Generates `n_series` synthetic time series, each containing a
    ground-truth chain of `k` elements of length `m`, embedded in
    random noise segments.

    Returns an array of shape (n_series, length * 2).
    """
    np.random.seed(seed)
    time_series_set = []

    interval = 6.3 / (m - 1)
    sine_wave = np.sin(np.arange(0, 6.3 + interval, interval))[:m]

    temp = np.random.rand(m)
    d = np.where(temp > 0.5, 1, -1)
    d[0] = 0
    d = np.cumsum(d)
    d = np.concatenate([d[:np.ceil(m / 2).astype(int)],
                         np.flip(d[:np.floor(m / 2).astype(int)])])
    d = np.convolve(d, np.ones(5) / 5, mode='same')  # smoothing

    period = 2
    x = m * period

    for _ in range(n_series):
        randseq = np.random.rand(x * k)
        L = np.zeros_like(randseq)

        for i in range(k):
            randseq[i * x: i * x + m] = (
                (randseq[i * x: i * x + m] - np.mean(randseq[i * x: i * x + m]))
                / np.std(randseq[i * x: i * x + m])
            )
            L[i * x: i * x + m] = 1
            randseq[i * x + m: (i + 1) * x] = (
                (randseq[i * x + m: (i + 1) * x] - np.mean(randseq[i * x + m: (i + 1) * x]))
                / np.std(randseq[i * x + m: (i + 1) * x])
            )

        data1 = []
        for i in range(k):
            rand1seq = randseq[i * x: i * x + m]
            rand2seq = randseq[i * x + m: (i + 1) * x]
            data1.extend((sine_wave * (k - i) + d * (i - 1)) + rand1seq * 0.01 * 100)
            data1.extend(rand2seq * 0.6)

        time_series_set.append(np.array(data1))

    return np.array(time_series_set)


def add_noise(ts, noise_level, seed=None):
    """
    Adds Gaussian noise scaled to `noise_level` (0.0-1.0) times the
    series' standard deviation.
    """
    if seed is not None:
        np.random.seed(seed)
    noise_scale = noise_level * np.std(ts)
    noise = np.random.normal(loc=0, scale=noise_scale, size=ts.shape)
    return ts + noise


# Ground-truth chain start indices for the default generator settings
# (k=20, m=50, period=2, length=100 -> chain elements at every 100
# samples: 0, 100, 200, ..., 1900).
GROUND_TRUTH_CHAIN_INDICES = np.arange(0, 2000, 100)
