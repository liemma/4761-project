from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import validate_weight_matrix


@dataclass
class MRFResult:
    states: np.ndarray
    energy_history: list[float]
    converged: bool
    n_iter: int


class BinarySpatialMRF:
    """Binary spatial MRF with weighted edges and ICM inference."""

    def __init__(
        self,
        weight_matrix: np.ndarray,
        beta: float = 1.0,
        max_iter: int = 200,
        tol: float = 0.0,
    ) -> None:
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol < 0:
            raise ValueError("tol must be nonnegative")

        self.weight_matrix = validate_weight_matrix(weight_matrix)
        self.beta = float(beta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, theta: np.ndarray, init_states: np.ndarray | None = None) -> MRFResult:
        theta = np.asarray(theta, dtype=float)
        n_regions = self.weight_matrix.shape[0]

        if theta.ndim != 1 or theta.shape[0] != n_regions:
            raise ValueError("theta must be a vector with one entry per region")

        states = self._initialize_states(theta, init_states)
        energy_history = [self.energy(theta, states)]
        converged = False

        for iteration in range(1, self.max_iter + 1):
            previous_states = states.copy()

            for region in range(n_regions):
                local_field = theta[region] + self.beta * np.dot(
                    self.weight_matrix[region], states
                )
                states[region] = 1 if local_field >= 0 else -1

            current_energy = self.energy(theta, states)
            energy_history.append(current_energy)

            max_state_change = np.max(np.abs(states - previous_states))
            if max_state_change <= self.tol:
                converged = True
                break

        return MRFResult(
            states=states,
            energy_history=energy_history,
            converged=converged,
            n_iter=iteration,
        )

    def local_field(self, theta: np.ndarray, states: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        states = np.asarray(states, dtype=int)
        return theta + self.beta * self.weight_matrix @ states

    def energy(self, theta: np.ndarray, states: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        states = np.asarray(states, dtype=int)

        unary_term = -np.dot(theta, states)
        pairwise_term = -0.5 * self.beta * np.sum(
            self.weight_matrix * np.outer(states, states)
        )
        return float(unary_term + pairwise_term)

    @staticmethod
    def from_uniform_graph(n_regions: int, beta: float = 1.0, max_iter: int = 200) -> "BinarySpatialMRF":
        if n_regions <= 0:
            raise ValueError("n_regions must be positive")

        matrix = np.ones((n_regions, n_regions), dtype=float) - np.eye(n_regions)
        return BinarySpatialMRF(weight_matrix=matrix, beta=beta, max_iter=max_iter)

    @staticmethod
    def _initialize_states(theta: np.ndarray, init_states: np.ndarray | None) -> np.ndarray:
        if init_states is None:
            return np.where(theta >= 0, 1, -1).astype(int)

        states = np.asarray(init_states, dtype=int)
        if states.shape != theta.shape:
            raise ValueError("init_states must have the same shape as theta")
        if not np.all(np.isin(states, (-1, 1))):
            raise ValueError("init_states entries must be -1 or 1")
        return states.copy()
