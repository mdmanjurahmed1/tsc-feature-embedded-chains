"""
C22 (catch22) feature extraction and normalization, corresponding to
Section 4.3 of the paper.
"""

import numpy as np
import pycatch22
from sklearn.preprocessing import MinMaxScaler


def cal_catch22(x):
    """Extracts all 22 catch22 features for a single subsequence."""
    return pycatch22.catch22_all(x)['values']


def extract_catch22_features(ts, subseq_len):
    """Extracts catch22 features for every subsequence of `ts`."""
    n_subseqs = len(ts) - subseq_len + 1
    features = [cal_catch22(ts[i:i + subseq_len]) for i in range(n_subseqs)]
    return np.array(features)


def ffill(arr):
    """Forward-fills NaN values along axis 1."""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return arr[np.arange(idx.shape[0])[:, None], idx]


def bfill(arr):
    """Backward-fills NaN values along axis 1."""
    return ffill(arr[:, ::-1])[:, ::-1]


def preprocess_feature_profiles(ts, m):
    """
    Extracts catch22 features for all subsequences of `ts`, imputes any
    missing feature values, and min-max normalizes the resulting matrix.
    """
    feature_profiles = extract_catch22_features(ts, m)
    feature_profiles = ffill(feature_profiles)
    feature_profiles = bfill(feature_profiles)
    return MinMaxScaler().fit_transform(feature_profiles)
