from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from spatial_mrf import BinarySpatialMRF


REGION_NAMES = [
    "PFC",
    "Motor",
    "Somatosensory",
    "Temporal",
    "Parietal",
    "Occipital",
]


def make_weighted_graph() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.9, 0.5, 0.7, 0.8, 0.2],
            [0.9, 0.0, 0.8, 0.4, 0.6, 0.2],
            [0.5, 0.8, 0.0, 0.3, 0.5, 0.2],
            [0.7, 0.4, 0.3, 0.0, 0.7, 0.3],
            [0.8, 0.6, 0.5, 0.7, 0.0, 0.4],
            [0.2, 0.2, 0.2, 0.3, 0.4, 0.0],
        ]
    )


def main() -> None:
    theta = np.array([1.2, 0.9, -0.4, -0.8, 0.3, -0.2])

    weighted_model = BinarySpatialMRF(weight_matrix=make_weighted_graph(), beta=0.9)
    uniform_model = BinarySpatialMRF.from_uniform_graph(n_regions=len(theta), beta=0.9)

    weighted_result = weighted_model.fit(theta)
    uniform_result = uniform_model.fit(theta)

    print("Unary evidence (theta):")
    for region, score in zip(REGION_NAMES, theta):
        print(f"  {region:15s} {score:+.2f}")

    print("\nWeighted MRF states:")
    for region, state in zip(REGION_NAMES, weighted_result.states):
        print(f"  {region:15s} {state:+d}")

    print("\nUniform MRF states:")
    for region, state in zip(REGION_NAMES, uniform_result.states):
        print(f"  {region:15s} {state:+d}")

    print(f"\nWeighted energy: {weighted_result.energy_history[-1]:.3f}")
    print(f"Uniform energy:  {uniform_result.energy_history[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(REGION_NAMES, theta, color=["#2a6f97" if x >= 0 else "#c44536" for x in theta])
    axes[0].set_title("Unary Evidence")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].axhline(0.0, color="black", linewidth=1)

    x = np.arange(len(REGION_NAMES))
    width = 0.35
    axes[1].bar(x - width / 2, weighted_result.states, width=width, label="Weighted")
    axes[1].bar(x + width / 2, uniform_result.states, width=width, label="Uniform")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(REGION_NAMES, rotation=35)
    axes[1].set_yticks([-1, 1])
    axes[1].set_title("Inferred Latent States")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
