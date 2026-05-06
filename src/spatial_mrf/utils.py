from __future__ import annotations

import numpy as np
from pathlib import Path



def validate_weight_matrix(weight_matrix: np.ndarray) -> np.ndarray:
    """Validate and return a symmetric nonnegative weight matrix."""
    matrix = np.asarray(weight_matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("weight_matrix must be a square matrix")
    if np.any(matrix < 0):
        raise ValueError("weight_matrix cannot contain negative entries")
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("weight_matrix must be symmetric")

    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    return matrix


def normalize_weights(weight_matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize a validated weight matrix while preserving zeros."""
    matrix = validate_weight_matrix(weight_matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums + eps, where=row_sums > 0)
    normalized[row_sums.squeeze(axis=1) == 0] = 0.0
    return normalized

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


def build_anatomy_weighted_knn_graph(
    adata,
    coord_key="spatial",
    label_key="ccf_parcellation_index",
    k=8,
    alpha=0.2,
    symmetric=True,
):
    """
    Build anatomy-weighted kNN graph.

    Parameters
    ----------
    adata : AnnData
    coord_key : str
        Key in adata.obsm for spatial coordinates.
    label_key : str
        Column in adata.obs for anatomical labels.
    k : int
        Number of neighbors (excluding self).
    alpha : float
        Penalty weight for cross-region edges or missing labels.
    symmetric : bool
        Whether to symmetrize adjacency (recommended for most graph methods).

    Returns
    -------
    W : csr_matrix
        Weighted adjacency matrix.
    """

    # ----------------------------
    # 1. coordinates
    # ----------------------------
    coords = adata.obsm[coord_key][:, :2]

    knn = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="ball_tree"
    ).fit(coords)

    _, indices = knn.kneighbors(coords)

    # ----------------------------
    # 2. labels
    # ----------------------------
    labels = adata.obs[label_key].values

    if pd.isna(labels).all():
        raise ValueError(f"All values in {label_key} are NaN.")

    # vectorized NaN mask (faster than pd.isna inside loop)
    is_nan = pd.isna(labels)

    n = coords.shape[0]
    rows, cols, weights = [], [], []

    # ----------------------------
    # 3. build graph
    # ----------------------------
    for i in range(n):
        for j in indices[i, 1:]:  # skip self

            rows.append(i)
            cols.append(j)

            if is_nan[i] or is_nan[j]:
                w = alpha
            elif labels[i] == labels[j]:
                w = 1.0
            else:
                w = alpha

            weights.append(w)

    W = csr_matrix((weights, (rows, cols)), shape=(n, n))

    # ----------------------------
    # 4. symmetrize (important for MRF/GNN)
    # ----------------------------
    if symmetric:
        W = (W + W.T) * 0.5

    print("AW graph built successfully")
    print("shape:", W.shape, "nnz:", W.nnz)

    return W

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def remap_clusters_to_atlas(adata, cluster_col, label_col):
    """
    Hungarian matching for best label matching between cluster labels and atlas labels.
    """

    clusters = adata.obs[cluster_col].astype(str)
    labels = adata.obs[label_col].astype(str)

    # row: label, column: cluster 
    cm_df = pd.crosstab(labels, clusters)  
    
    cm = cm_df.values
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-cm)

    # mapping from cluster to label using index/columns of cm_df
    mapping = {
        cm_df.columns[c]: cm_df.index[r]
        for r, c in zip(row_ind, col_ind)
    }

    # unmatched_clusters as unassigned if clusters > labels
    unmatched_clusters = set(cm_df.columns) - set(mapping.keys())
    for uc in unmatched_clusters:
        mapping[uc] = "Unassigned"

    adata.obs[f'{cluster_col}_named'] = clusters.map(mapping)

    return mapping

import pandas as pd

def align_visualization_colors(adata, target_col, reference_col):
    """
    for target_col use reference_col （based on category alignment）
    """

    # 1. categorical dtype
    adata.obs[reference_col] = adata.obs[reference_col].astype('category')
    adata.obs[target_col] = adata.obs[target_col].astype('category')

    # 2. reference color 
    ref_colors = adata.uns.get(f'{reference_col}_colors', None)
    if ref_colors is None:
        raise ValueError(f"{reference_col}_colors not found in adata.uns")

    ref_categories = adata.obs[reference_col].cat.categories
    color_map = dict(zip(ref_categories, ref_colors))

    # 3.target color list
    target_categories = adata.obs[target_col].cat.categories

    new_colors = []
    for cat in target_categories:
        if cat in color_map:
            new_colors.append(color_map[cat])
        else:
            new_colors.append("#cccccc")  # fallback grey

    # 4. write back
    adata.uns[f'{target_col}_colors'] = new_colors

import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from spatial_mrf.utils import remap_clusters_to_atlas, align_visualization_colors


def plot_hmrf_spatial_results(
    adata,
    results,
    label_key="parcellation_structure",
    k_to_plot=None,
    spot_size=0.034,
    do_remap=True,
    do_color_align=True,
    figsize=None,
    output_dir=None,
    show=True,
):
    """
    Visualize HMRF spatial results across beta values.
    """

    # ----------------------------
    # 1. choose betas
    # ----------------------------
    betas = list(results.keys())
    if k_to_plot is not None:
        betas = [b for b in betas if b in k_to_plot]

    # ----------------------------
    # 2. write obs columns
    # ----------------------------
    for beta in betas:
        col = f"AW_HMRF_beta_{beta}"
        adata.obs[col] = pd.Categorical(results[beta].states)

        col = f"AW_HMRF_beta_{beta}"

        if do_remap:
            remap_clusters_to_atlas(
                adata,
                col,
                label_key
            )

        # remapped column is assumed created as *_named
            named_col = f"{col}_named"

            adata.obs[named_col] = adata.obs[named_col].astype("category")

            if do_color_align:
                align_visualization_colors(
                    adata,
                    named_col,
                    label_key
                )

    # ----------------------------
    # 4. plotting
    # ----------------------------
    for beta in betas:
        col = f"AW_HMRF_beta_{beta}"
        named_col = f"{col}_named"
        print(f"Plotting beta={beta}")

        sc.pl.spatial(
            adata,
            color=named_col,
            title=f"HMRF Spatial Domains (beta={beta})",
            spot_size=spot_size,
            frameon=False,
            show=False
        )

        fig = plt.gcf()
        if figsize is not None:
            fig.set_size_inches(*figsize)

        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                out_dir / f"spatial_beta-{beta}.png",
                dpi=200,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close(fig)
