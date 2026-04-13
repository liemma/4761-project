from __future__ import annotations

import numpy as np


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
