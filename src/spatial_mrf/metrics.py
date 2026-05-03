#metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    silhouette_score,
)

# 1. Label-level: ARI + NMI
def compute_ari_nmi(y_true, y_pred):
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
    }


# 2. Label-level: purity
def compute_purity(y_true, y_pred):
    ct = pd.crosstab(y_pred, y_true)
    purity = ct.max(axis=1).sum() / ct.values.sum()
    return {"purity": purity}


# 3. Label-level: homogeneity & completeness
def compute_homogeneity_completeness(y_true, y_pred):
    return {
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
    }


# 4. Cluster-level: silhouette（embedding）
def compute_silhouette(X, y_pred):
    if len(np.unique(y_pred)) < 2:
        return {"silhouette": np.nan}
    return {"silhouette": silhouette_score(X, y_pred)}


# 5. Spatial-level: neighbor consistency
def compute_spatial_consistency(labels, W):
    import numpy as np
    from scipy.sparse import csr_matrix

    if not isinstance(W, csr_matrix):
        W = csr_matrix(W)

    same = 0
    total = 0

    for i in range(W.shape[0]):
        start, end = W.indptr[i], W.indptr[i + 1]
        neighbors = W.indices[start:end]

        if len(neighbors) == 0:
            continue

        total += len(neighbors)
        same += np.sum(labels[neighbors] == labels[i])

    return {"spatial_consistency": same / total if total > 0 else 0}