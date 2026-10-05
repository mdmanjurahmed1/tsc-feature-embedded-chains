"""
K-D Tree-based chain augmentation, corresponding to Section 4.4 of the
paper. Parameter values (k, delta) follow Table 1 (Parameter Summary).
"""

from scipy.spatial import KDTree


def build_kd_tree(feature_profiles):
    """Builds a K-D Tree over the normalized C22 feature matrix."""
    return KDTree(feature_profiles)


def augment_with_kd_tree(base_chain, feature_profiles, k=5, delta=0.4):
    """
    Augments `base_chain` with additional structurally similar indices
    found via K-D Tree nearest-neighbor search in C22 feature space.

    k     : number of nearest neighbors queried per base chain index
    delta : Euclidean distance threshold for accepting a neighbor
    """
    kd_tree = build_kd_tree(feature_profiles)
    kd_chain = list(base_chain)

    for idx in base_chain:
        query_feature = feature_profiles[idx].reshape(1, -1)
        distances, neighbors = kd_tree.query(query_feature, k=k)

        for dist, neighbor_idx in zip(distances[0], neighbors[0]):
            if dist < delta and neighbor_idx not in kd_chain:
                kd_chain.append(neighbor_idx)

    kd_chain = sorted(set(int(i) for i in kd_chain))
    return kd_chain
