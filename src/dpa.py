"""
Density Peak Advanced (DPA) - Automatic Topography of High-Dimensional Data Sets

Implementation based on:
d'Errico, M., Facco, E., Laio, A., & Rodriguez, A. (2021).
"Automatic topography of high-dimensional data sets by non-parametric density peak clustering"
Information Sciences, 560, 476-492.

Main components:
1. k-NN based density estimation (PAk - Point Adaptive k-NN)
2. Density Peak identification
3. Cluster assignment with topographic representation
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


class DPA:
    """
    Density Peak Advanced clustering algorithm.

    Parameters
    ----------
    k : int, default=20
        Number of neighbors for density estimation
    z : float, default=1.65
        Z-score threshold for statistical significance (95% confidence)
    metric : str, default='euclidean'
        Distance metric to use

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point
    density_ : ndarray of shape (n_samples,)
        Estimated density for each point
    cluster_centers_ : ndarray
        Indices of cluster centers (density peaks)
    """

    def __init__(self, k=20, z=1.65, metric='euclidean'):
        self.k = k
        self.z = z
        self.metric = metric

        self.labels_ = None
        self.density_ = None
        self.cluster_centers_ = None
        self.topography_ = None

    def fit(self, X):
        """
        Fit the DPA clustering model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Step 1: Compute k-NN distances
        nn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Step 2: Estimate density using PAk estimator
        self.density_ = self._estimate_density(distances, n_features)

        # Step 3: Find density peaks
        self.cluster_centers_ = self._find_density_peaks(indices)

        # Step 4: Assign clusters
        self.labels_ = self._assign_clusters(indices)

        return self

    def _estimate_density(self, distances, d):
        """
        Estimate density using the PAk (Point Adaptive k-NN) estimator.

        The density is estimated as:
        rho_i = k / (V_d * r_k^d)

        where V_d is the volume of d-dimensional unit sphere
        and r_k is the distance to the k-th neighbor.
        """
        n_samples = distances.shape[0]

        # Volume of d-dimensional unit sphere
        # V_d = pi^(d/2) / Gamma(d/2 + 1)
        from scipy.special import gamma
        V_d = (np.pi ** (d / 2)) / gamma(d / 2 + 1)

        # Distance to k-th neighbor (excluding self)
        r_k = distances[:, self.k]

        # Avoid division by zero
        r_k = np.maximum(r_k, 1e-10)

        # Density estimation
        density = self.k / (V_d * (r_k ** d))

        # Log-density for numerical stability
        log_density = np.log(density)

        return log_density

    def _find_density_peaks(self, indices):
        """
        Find density peaks - points with higher density than all their neighbors.
        """
        n_samples = len(self.density_)
        is_peak = np.ones(n_samples, dtype=bool)

        for i in range(n_samples):
            neighbors = indices[i, 1:]  # Exclude self
            # A point is a peak if its density is higher than all neighbors
            if np.any(self.density_[neighbors] >= self.density_[i]):
                is_peak[i] = False

        peaks = np.where(is_peak)[0]
        return peaks

    def _assign_clusters(self, indices):
        """
        Assign each point to the cluster of its nearest neighbor with higher density.
        """
        n_samples = len(self.density_)
        labels = -np.ones(n_samples, dtype=int)

        # Assign peaks to their own clusters
        for cluster_id, peak_idx in enumerate(self.cluster_centers_):
            labels[peak_idx] = cluster_id

        # Sort points by decreasing density
        sorted_indices = np.argsort(-self.density_)

        # Assign each point to the cluster of its nearest higher-density neighbor
        for i in sorted_indices:
            if labels[i] >= 0:
                continue  # Already assigned (is a peak)

            neighbors = indices[i, 1:]
            higher_density_neighbors = neighbors[self.density_[neighbors] > self.density_[i]]

            if len(higher_density_neighbors) > 0:
                # Find nearest neighbor with higher density that is already assigned
                for neighbor in higher_density_neighbors:
                    if labels[neighbor] >= 0:
                        labels[i] = labels[neighbor]
                        break

        return labels

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels
        """
        self.fit(X)
        return self.labels_
