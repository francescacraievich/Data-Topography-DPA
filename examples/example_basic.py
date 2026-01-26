"""
Basic example of DPA clustering on synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from src.dpa import DPA


def generate_blobs(n_samples=500, centers=3, random_state=42):
    """Generate synthetic blob data."""
    np.random.seed(random_state)

    points_per_cluster = n_samples // centers
    X = []
    y = []

    for i in range(centers):
        center = np.random.randn(2) * 5
        cluster_points = center + np.random.randn(points_per_cluster, 2) * 0.5
        X.append(cluster_points)
        y.extend([i] * points_per_cluster)

    return np.vstack(X), np.array(y)


if __name__ == "__main__":
    # Generate data
    X, y_true = generate_blobs(n_samples=300, centers=3)

    # Fit DPA
    dpa = DPA(k=15)
    labels = dpa.fit_predict(X)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
    axes[0].set_title('Ground Truth')

    axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    axes[1].scatter(X[dpa.cluster_centers_, 0], X[dpa.cluster_centers_, 1],
                    c='red', marker='x', s=200, linewidths=3, label='Density Peaks')
    axes[1].set_title(f'DPA Clustering ({len(dpa.cluster_centers_)} clusters found)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('dpa_example.png', dpi=150)
    plt.show()

    print(f"Number of clusters found: {len(dpa.cluster_centers_)}")
    print(f"Cluster centers indices: {dpa.cluster_centers_}")
