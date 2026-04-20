from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

@dataclass
class HMRFResult:
    states: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    energy_history: list[float]
    converged: bool
    n_iter: int

class AW_HMRF:
    """Anatomy-Weighted Multi-class HMRF with Gaussian emission and sparse spatial graph."""

    def __init__(
        self,
        n_regions: int,
        beta: float = 1.0,
        max_em_iter: int = 20,
        max_icm_iter: int = 5,
        tol: float = 1e-3,
    ) -> None:
        if n_regions <= 1:
            raise ValueError("n_regions (K) must be greater than 1")
        if beta < 0:
            raise ValueError("beta must be nonnegative")

        self.K = int(n_regions)
        self.beta = float(beta)
        self.max_em_iter = int(max_em_iter)
        self.max_icm_iter = int(max_icm_iter)
        self.tol = float(tol)

        # emission parameters (Gaussian parameters)
        self.mu = None
        self.sigma = None

    def fit(self, embeddings: np.ndarray, weight_matrix: sp.spmatrix) -> HMRFResult:
        """
        Fit the HMRF model using Expectation(ICM)-Maximization.
        
        embeddings: shape (N_cells, d_features)
        weight_matrix: scipy sparse matrix of shape (N_cells, N_cells)
        """
        embeddings = np.asarray(embeddings, dtype=float)
        n_cells, d_features = embeddings.shape

        if not sp.isspmatrix_csr(weight_matrix):
            weight_matrix = weight_matrix.tocsr()

        # K-means initialization for states
        kmeans = KMeans(n_clusters=self.K, n_init=10)
        states = kmeans.fit_predict(embeddings)
        
        # parameters initialization (can be updated in the M-step)
        self.mu = np.zeros((self.K, d_features))
        self.sigma = np.zeros((self.K, d_features, d_features))
        
        energy_history = []
        converged = False

        for em_iter in range(1, self.max_em_iter + 1):
            previous_states = states.copy()

            # M-step: gaussian parameters estimation based on current states
            self._m_step(embeddings, states)

            # E-step: states update using ICM
            states = self._icm_step(embeddings, states, weight_matrix)

            # energy_history may be computed here if needed, e.g., using a pseudo-energy function that combines data likelihood and spatial consistency
            # current_energy = self._pseudo_energy(embeddings, states, weight_matrix)
            # energy_history.append(current_energy)

            # check convergence based on state changes
            changed_ratio = np.sum(states != previous_states) / n_cells
            if changed_ratio <= self.tol:
                converged = True
                break

        return HMRFResult(
            states=states,
            mu=self.mu,
            sigma=self.sigma,
            energy_history=energy_history,
            converged=converged,
            n_iter=em_iter,
        )

    def _m_step(self, embeddings: np.ndarray, states: np.ndarray) -> None:
        """parameter estimation (MLE)"""
        d_features = embeddings.shape[1]
        eps = 1e-6 * np.eye(d_features)  # regularization to prevent singular covariance

        for k in range(self.K):
            mask = (states == k)
            if np.sum(mask) > 1:
                cluster_data = embeddings[mask]
                self.mu[k] = np.mean(cluster_data, axis=0)
                self.sigma[k] = np.cov(cluster_data.T) + eps
            else:
                # if a cluster has 0 or 1 member, we cannot estimate covariance; use identity and mean of all data
                self.mu[k] = np.zeros(d_features)
                self.sigma[k] = np.eye(d_features)

    def _icm_step(self, embeddings: np.ndarray, states: np.ndarray, W: sp.csr_matrix) -> np.ndarray:
        """Iterated Conditional Modes"""
        n_cells = embeddings.shape[0]
        
        # Unary Potentials (Negative Log-Likelihood)
        unary_energy = np.zeros((n_cells, self.K))
        for k in range(self.K):
            try:
                rv = multivariate_normal(self.mu[k], self.sigma[k])
                unary_energy[:, k] = -rv.logpdf(embeddings)
            except np.linalg.LinAlgError:
                # in case of singular covariance, assign high energy to this cluster to avoid assignment
                unary_energy[:, k] = np.inf 

        # ICM iterations for every cell
        for _ in range(self.max_icm_iter):
            # randomly permute the order of cells to update
            for i in np.random.permutation(n_cells):
                # find neighbors and their states
                start_ptr, end_ptr = W.indptr[i], W.indptr[i+1]
                neighbors = W.indices[start_ptr:end_ptr]
                weights = W.data[start_ptr:end_ptr]

                neighbor_states = states[neighbors]

                # Pairwise Potentials (匹配奖励)
                spatial_energy = np.zeros(self.K)
                for k in range(self.K):
                    # Penalty on unmatching labels (Potts model)
                    # weighted sum of neighbors that match state k
                    match_weight = np.sum(weights[neighbor_states == k])
                    spatial_energy[k] = -self.beta * match_weight

                # Update state for cell i by minimizing the total energy (unary + pairwise)
                states[i] = np.argmin(unary_energy[i] + spatial_energy)

        return states