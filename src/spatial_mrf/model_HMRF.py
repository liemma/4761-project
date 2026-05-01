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
        energy_tol: float | None = None,
        random_state: int | None = 0,
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
        self.energy_tol = None if energy_tol is None else float(energy_tol)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

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
        weight_matrix = self._validate_weight_matrix(weight_matrix, n_cells=n_cells)

        # K-means initialization for states
        kmeans = KMeans(n_clusters=self.K, n_init=10, random_state=self.random_state)
        states = kmeans.fit_predict(embeddings)
        
        # parameters initialization (can be updated in the M-step)
        self.mu = np.zeros((self.K, d_features))
        self.sigma = np.zeros((self.K, d_features, d_features))
        
        energy_history = []
        converged = False

        for em_iter in range(1, self.max_em_iter + 1):
            previous_states = states.copy()
            previous_energy = energy_history[-1] if energy_history else None

            # M-step: gaussian parameters estimation based on current states
            self._m_step(embeddings, states)

            # E-step: states update using ICM
            states = self._icm_step(embeddings, states, weight_matrix)

            current_energy = self._pseudo_energy(embeddings, states, weight_matrix)
            energy_history.append(current_energy)

            # check convergence based on state changes
            changed_ratio = np.sum(states != previous_states) / n_cells
            energy_converged = (
                False
                if self.energy_tol is None or previous_energy is None
                else abs(previous_energy - current_energy) <= self.energy_tol
            )
            if changed_ratio <= self.tol or energy_converged:
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
        global_mu = np.mean(embeddings, axis=0)
        global_sigma = np.cov(embeddings.T) + eps

        for k in range(self.K):
            mask = (states == k)
            if np.sum(mask) > 1:
                cluster_data = embeddings[mask]
                self.mu[k] = np.mean(cluster_data, axis=0)
                self.sigma[k] = np.cov(cluster_data.T) + eps
            else:
                # if a cluster has 0 or 1 member, we cannot estimate covariance reliably
                self.mu[k] = global_mu
                self.sigma[k] = global_sigma

    def _icm_step(self, embeddings: np.ndarray, states: np.ndarray, W: sp.csr_matrix) -> np.ndarray:
        """Iterated Conditional Modes"""
        n_cells = embeddings.shape[0]
        
        # Unary Potentials (Negative Log-Likelihood)
        unary_energy = np.zeros((n_cells, self.K))
        for k in range(self.K):
            try:
                rv = multivariate_normal(self.mu[k], self.sigma[k], allow_singular=True)
                unary_energy[:, k] = -rv.logpdf(embeddings)
            except (np.linalg.LinAlgError, ValueError):
                # in case of singular covariance, assign high energy to this cluster to avoid assignment
                unary_energy[:, k] = np.inf 

        # ICM iterations for every cell
        for _ in range(self.max_icm_iter):
            # randomly permute the order of cells to update
            for i in self._rng.permutation(n_cells):
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

    def _pseudo_energy(self, embeddings: np.ndarray, states: np.ndarray, W: sp.csr_matrix) -> float:
        """
        Compute a scalar objective for monitoring:
        E(z; mu, Sigma) = sum_i -log p(e_i | z_i) - beta * sum_{i<j} W_ij * 1[z_i = z_j]
        Lower is better.
        """
        embeddings = np.asarray(embeddings, dtype=float)
        states = np.asarray(states, dtype=int)

        n_cells = embeddings.shape[0]
        if states.shape != (n_cells,):
            raise ValueError("states must be a vector of length N_cells")

        unary_sum = 0.0
        for k in range(self.K):
            mask = states == k
            if not np.any(mask):
                continue
            try:
                rv = multivariate_normal(self.mu[k], self.sigma[k], allow_singular=True)
                unary_sum += float(np.sum(-rv.logpdf(embeddings[mask])))
            except (np.linalg.LinAlgError, ValueError):
                return float("inf")

        # Pairwise Potts reward: count matching edges; divide by 2 for symmetric graphs.
        match_sum = 0.0
        W = W.tocsr()
        for i in range(n_cells):
            start_ptr, end_ptr = W.indptr[i], W.indptr[i + 1]
            nbrs = W.indices[start_ptr:end_ptr]
            wts = W.data[start_ptr:end_ptr]
            if nbrs.size == 0:
                continue
            match_sum += float(np.sum(wts[states[nbrs] == states[i]]))

        pairwise_term = -0.5 * self.beta * match_sum
        return float(unary_sum + pairwise_term)

    @staticmethod
    def _validate_weight_matrix(W: sp.csr_matrix, n_cells: int) -> sp.csr_matrix:
        W = W.tocsr()
        if W.shape != (n_cells, n_cells):
            raise ValueError("weight_matrix must have shape (N_cells, N_cells)")
        if W.nnz > 0 and np.any(W.data < 0):
            raise ValueError("weight_matrix cannot contain negative entries")
        # Many KNN constructions yield a directed graph; enforce an undirected spatial prior.
        W = W.maximum(W.T).tocsr()
        W = W.copy()
        W.setdiag(0.0)
        W.eliminate_zeros()
        return W