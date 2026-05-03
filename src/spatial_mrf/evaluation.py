import matplotlib.pyplot as plt

def plot_energy_history(result):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(result.energy_history) + 1), result.energy_history, marker='o', color='#2c3e50')
    plt.title("AW-HMRF Convergence: Total Gibbs Energy")
    plt.xlabel("EM Iterations")
    plt.ylabel("Total Energy (Unary + Weighted Pairwise)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# evaluation.py

import pandas as pd
""" from utils.metrics import (
    compute_ari_nmi,
    compute_purity,
    compute_homogeneity_completeness,
    compute_silhouette,
    compute_spatial_consistency,
) """


def evaluate_all(
    adata,
    label_key="parcellation_structure",
    pred_key="AW_HMRF",
    embedding_key="X_pca",
    W=None,
):
    """
    Returns:
        dict
    """
    from spatial_mrf.metrics import (
        compute_ari_nmi,
        compute_purity,
        compute_homogeneity_completeness,
        compute_silhouette,
        compute_spatial_consistency,
    )
    y_true = adata.obs[label_key]
    y_pred = adata.obs[pred_key]

    results = {}

    # --- label-level ---
    results.update(compute_ari_nmi(y_true, y_pred))
    results.update(compute_purity(y_true, y_pred))
    results.update(compute_homogeneity_completeness(y_true, y_pred))

    # --- cluster-level ---
    if embedding_key in adata.obsm:
        X = adata.obsm[embedding_key]
        results.update(compute_silhouette(X, y_pred))

    # --- spatial-level ---
    if W is not None:
        results.update(compute_spatial_consistency(y_pred, W))

    return results


def print_results(results):
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")