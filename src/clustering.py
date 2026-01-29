"""
Clustering algorithms for DPA.

Implementation of the three main algorithms from the DPA paper:
- Algorithm 1 (Heuristic 1): Automatic detection of cluster centers
- Algorithm 2 (Heuristic 2): Finding saddle points between clusters
- Algorithm 3 (Heuristic 3): Merging insignificant peaks using Z-score

Based on: d'Errico, M., Facco, E., Laio, A., & Rodriguez, A. (2021).
"Automatic topography of high-dimensional data sets by non-parametric
density peak clustering" Information Sciences, 560, 476-492.
"""

import numpy as np
from collections import defaultdict


def find_cluster_centers(log_density, epsilon, distances, indices, k_hat):
    """
    Algorithm 1 (Heuristic 1): Automatic detection of cluster centers.

    A point i is a cluster center if:
    1. delta_i > r_{k_hat_i}: the nearest point with higher g is outside
       the optimal neighborhood
    2. i is not in the neighborhood of any point j with g_j > g_i

    Parameters
    ----------
    log_density : ndarray of shape (n_samples,)
        Log-density log(rho_i) for each point.

    epsilon : ndarray of shape (n_samples,)
        Error estimates for each point.

    distances : ndarray of shape (n_samples, k_max + 1)
        Distances to neighbors.

    indices : ndarray of shape (n_samples, k_max + 1)
        Neighbor indices.

    k_hat : ndarray of shape (n_samples,)
        Optimal k for each point.

    Returns
    -------
    centers : ndarray
        Indices of detected cluster centers.

    delta : ndarray of shape (n_samples,)
        Delta values for each point.

    nearest_higher : ndarray of shape (n_samples,)
        Index of nearest point with higher g (-1 if none in neighbors).
    """
    n_samples = len(log_density)

    # Compute g = log(rho) - epsilon (error-adjusted log-density)
    g = log_density - epsilon

    # Compute delta: min distance to point with higher g
    delta = np.full(n_samples, np.inf)
    nearest_higher = np.full(n_samples, -1, dtype=int)

    # Sort points by decreasing g
    sorted_idx = np.argsort(-g)

    # For each point, find nearest neighbor with higher g
    for rank, i in enumerate(sorted_idx):
        if rank == 0:
            # Point with highest g has no point with higher g
            continue

        # Look through neighbors for point with higher g
        for j_pos in range(1, min(len(indices[i]), k_hat[i] + 10)):
            j = indices[i, j_pos]
            if g[j] > g[i]:
                delta[i] = distances[i, j_pos]
                nearest_higher[i] = j
                break

        # If not found in neighbors, need broader search
        if nearest_higher[i] == -1:
            # Search all points with higher g
            higher_g_mask = g > g[i]
            if np.any(higher_g_mask):
                # Find minimum distance to any point with higher g
                # This is expensive, so we use a heuristic: check if any
                # neighbor of neighbors has higher g
                min_dist = np.inf
                for j_pos in range(1, len(indices[i])):
                    j = indices[i, j_pos]
                    if g[j] > g[i] and distances[i, j_pos] < min_dist:
                        min_dist = distances[i, j_pos]
                        nearest_higher[i] = j
                        delta[i] = min_dist

    # Set delta for highest-g point to max delta (following DP convention)
    max_g_idx = sorted_idx[0]
    finite_deltas = delta[np.isfinite(delta)]
    if len(finite_deltas) > 0:
        delta[max_g_idx] = np.max(finite_deltas) * 1.1
    else:
        delta[max_g_idx] = np.max(distances[:, -1])

    # Find centers using two conditions from Heuristic 1
    centers = []

    for i in range(n_samples):
        # Get optimal neighborhood radius for point i
        k_i = min(k_hat[i], distances.shape[1] - 1)
        r_k_i = distances[i, k_i]

        # Condition 1: delta_i > r_{k_hat_i}
        cond1 = delta[i] > r_k_i

        # Condition 2: i not in neighborhood of any point with higher g
        cond2 = True
        for j in range(n_samples):
            if g[j] > g[i]:
                # Check if i is in j's optimal neighborhood
                k_j = min(k_hat[j], distances.shape[1] - 1)
                # i is in NN_j if i appears in j's first k_j neighbors
                if i in indices[j, 1:k_j + 1]:
                    cond2 = False
                    break

        if cond1 and cond2:
            centers.append(i)

    return np.array(centers), delta, nearest_higher


def assign_to_clusters(g, nearest_higher, centers, n_samples):
    """
    Assign all points to clusters following density gradient.

    Each non-center point is assigned to the same cluster as its
    nearest neighbor with higher g.

    Parameters
    ----------
    g : ndarray of shape (n_samples,)
        Error-adjusted log-density.

    nearest_higher : ndarray of shape (n_samples,)
        Index of nearest point with higher g.

    centers : ndarray
        Cluster center indices.

    n_samples : int
        Number of samples.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    """
    labels = np.full(n_samples, -1, dtype=int)

    # Create center lookup
    center_to_cluster = {c: idx for idx, c in enumerate(centers)}

    # Assign centers to their clusters
    for cluster_id, center_idx in enumerate(centers):
        labels[center_idx] = cluster_id

    # Sort by decreasing g
    sorted_idx = np.argsort(-g)

    # Assign following density gradient
    for i in sorted_idx:
        if labels[i] >= 0:
            continue  # already assigned

        # Follow chain until we reach an assigned point
        current = i
        chain = [i]
        while labels[current] < 0 and nearest_higher[current] >= 0:
            current = nearest_higher[current]
            if current in chain:  # prevent infinite loop
                break
            chain.append(current)

        # Assign all points in chain
        if labels[current] >= 0:
            cluster = labels[current]
            for point in chain:
                labels[point] = cluster

    return labels


def find_saddle_points(labels, g, log_density, epsilon, distances, indices, k_hat):
    """
    Algorithm 2 (Heuristic 2): Find saddle points between clusters.

    For each pair of adjacent clusters, find the border points and
    identify the saddle point as the border point with highest g.

    Parameters
    ----------
    labels : ndarray of shape (n_samples,)
        Cluster assignments.

    g : ndarray of shape (n_samples,)
        Error-adjusted log-density.

    log_density : ndarray of shape (n_samples,)
        Log-density values.

    epsilon : ndarray of shape (n_samples,)
        Error estimates.

    distances : ndarray of shape (n_samples, k_max + 1)
        Distances to neighbors.

    indices : ndarray of shape (n_samples, k_max + 1)
        Neighbor indices.

    k_hat : ndarray of shape (n_samples,)
        Optimal k for each point.

    Returns
    -------
    saddles : dict
        Dictionary {(c1, c2): {'index': saddle_idx, 'g': saddle_g,
                              'log_density': saddle_log_rho, 'epsilon': saddle_eps}}

    border_points : dict
        Dictionary {(c1, c2): [list of border point indices]}
    """
    n_samples = len(labels)
    n_clusters = len(np.unique(labels[labels >= 0]))

    saddles = {}
    border_points = defaultdict(list)

    # Find border points for each cluster pair
    for i in range(n_samples):
        if labels[i] < 0:
            continue  # skip unassigned points

        c_i = labels[i]
        k_i = min(k_hat[i], indices.shape[1] - 1)

        # Check neighbors within optimal neighborhood
        for j_pos in range(1, k_i + 1):
            j = indices[i, j_pos]
            if labels[j] >= 0 and labels[j] != c_i:
                # i and j are in different clusters
                c_j = labels[j]

                # Check if this is a valid border point (Heuristic 2)
                # i is border if j is its closest point in cluster c_j
                # and i is the closest point to j among those in c_i

                # Simplified: just record border points
                pair = tuple(sorted([c_i, c_j]))
                border_points[pair].append(i)
                break  # only count once per point

    # Find saddle point for each cluster pair (highest g among border points)
    for pair, points in border_points.items():
        if len(points) == 0:
            continue

        # Saddle is border point with highest g
        g_values = g[points]
        best_idx = np.argmax(g_values)
        saddle_idx = points[best_idx]

        saddles[pair] = {
            'index': saddle_idx,
            'g': g[saddle_idx],
            'log_density': log_density[saddle_idx],
            'epsilon': epsilon[saddle_idx]
        }

    return saddles, dict(border_points)


def merge_insignificant_peaks(centers, saddles, log_density, epsilon, Z):
    """
    Algorithm 3 (Heuristic 3): Merge clusters with insignificant peaks.

    A cluster c is merged with cluster c' if:
        log(rho_c) - log(rho_{cc'}) < Z * sqrt(epsilon_c² + epsilon_{cc'}²)

    This tests whether the peak height above the saddle is statistically
    significant given the estimation errors. The error propagation uses
    quadrature (root sum of squares) since the errors are independent.

    Parameters
    ----------
    centers : ndarray
        Cluster center indices.

    saddles : dict
        Saddle point information from find_saddle_points.

    log_density : ndarray
        Log-density for all points.

    epsilon : ndarray
        Error estimates for all points.

    Z : float
        Significance threshold. Higher Z = more conservative (fewer clusters).
        - Z=1.0: ~68% confidence
        - Z=2.0: ~95% confidence
        - Z=3.0: ~99.7% confidence

    Returns
    -------
    merged_centers : ndarray
        Remaining centers after merging.

    merge_map : dict
        Mapping from original cluster ID to merged cluster ID.

    merge_history : list
        List of merge events [(lower_cluster, higher_cluster, saddle_info), ...]
    """
    n_clusters = len(centers)

    # Track which clusters are still active
    active = np.ones(n_clusters, dtype=bool)

    # Track cluster merges (maps original cluster to current root)
    parent = np.arange(n_clusters)

    # History of merges
    merge_history = []

    def find_root(i):
        """Find root of cluster i with path compression."""
        if parent[i] != i:
            parent[i] = find_root(parent[i])
        return parent[i]

    # Sort saddles by decreasing saddle density
    sorted_saddles = sorted(
        saddles.items(),
        key=lambda x: -x[1]['log_density']
    )

    # Process saddles in order
    for pair, saddle_info in sorted_saddles:
        c1, c2 = pair

        # Find current roots
        r1 = find_root(c1)
        r2 = find_root(c2)

        if r1 == r2:
            continue  # already merged

        # Get peak densities
        peak1_density = log_density[centers[r1]]
        peak2_density = log_density[centers[r2]]

        # Determine which has lower peak
        if peak1_density < peak2_density:
            lower_cluster, higher_cluster = r1, r2
        else:
            lower_cluster, higher_cluster = r2, r1

        # Get densities and errors
        peak_density = log_density[centers[lower_cluster]]
        peak_epsilon = epsilon[centers[lower_cluster]]
        saddle_density = saddle_info['log_density']
        saddle_epsilon = saddle_info['epsilon']

        # Merge criterion (Heuristic 3)
        # if log(rho_c) - log(rho_{cc'}) < Z * sqrt(epsilon_c² + epsilon_{cc'}²)
        # Error propagation uses quadrature for independent Gaussian errors
        height_diff = peak_density - saddle_density
        error_threshold = Z * np.sqrt(peak_epsilon**2 + saddle_epsilon**2)

        if height_diff < error_threshold:
            # Merge lower peak into higher peak
            parent[lower_cluster] = higher_cluster
            active[lower_cluster] = False
            merge_history.append((lower_cluster, higher_cluster, saddle_info))

    # Build final cluster mapping
    merge_map = {}
    new_id = 0
    for old_id in range(n_clusters):
        root = find_root(old_id)
        if root not in merge_map:
            merge_map[root] = new_id
            new_id += 1
        merge_map[old_id] = merge_map[root]

    # Get merged centers
    merged_centers = [centers[i] for i in range(n_clusters) if active[i]]

    return np.array(merged_centers), merge_map, merge_history


def identify_halo_points(labels, log_density, saddles, centers):
    """
    Identify halo points with unreliable cluster assignment.

    A point i in cluster c is marked as HALO if:
        rho_i < max{rho_{cc'}} for all clusters c' adjacent to c

    Halo points are not noise - they ARE assigned to a cluster, but with
    low confidence because their density is below the highest saddle.

    Parameters
    ----------
    labels : ndarray of shape (n_samples,)
        Cluster assignments.

    log_density : ndarray of shape (n_samples,)
        Log-density for all points.

    saddles : dict
        Saddle point information.

    centers : ndarray
        Cluster center indices.

    Returns
    -------
    halo_mask : ndarray of bool
        True for halo points.

    cluster_max_border_density : dict
        Maximum border density for each cluster.
    """
    n_samples = len(labels)
    n_clusters = len(centers)

    # Find maximum border density for each cluster
    cluster_max_border_density = {c: -np.inf for c in range(n_clusters)}

    for pair, info in saddles.items():
        c1, c2 = pair
        saddle_density = info['log_density']

        if c1 < n_clusters:
            cluster_max_border_density[c1] = max(
                cluster_max_border_density[c1], saddle_density
            )
        if c2 < n_clusters:
            cluster_max_border_density[c2] = max(
                cluster_max_border_density[c2], saddle_density
            )

    # Mark halo points
    halo_mask = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        c = labels[i]
        if c < 0:
            continue  # already unassigned

        max_border = cluster_max_border_density.get(c, -np.inf)
        if np.isfinite(max_border) and log_density[i] < max_border:
            halo_mask[i] = True

    return halo_mask, cluster_max_border_density


def relabel_after_merge(labels, merge_map):
    """
    Update labels after cluster merging.

    Parameters
    ----------
    labels : ndarray of shape (n_samples,)
        Original cluster labels.

    merge_map : dict
        Mapping from old cluster ID to new cluster ID.

    Returns
    -------
    new_labels : ndarray of shape (n_samples,)
        Updated cluster labels.
    """
    new_labels = labels.copy()
    for i in range(len(labels)):
        if labels[i] >= 0 and labels[i] in merge_map:
            new_labels[i] = merge_map[labels[i]]
    return new_labels


# =============================================================================
# Standard Density Peaks (DP) - Rodriguez & Laio (2014)
# For comparison with DPA
# =============================================================================

class DensityPeaks:
    """
    Standard Density Peaks clustering (Rodriguez & Laio, 2014).

    This is the ORIGINAL algorithm that DPA improves upon.
    Key limitation: requires MANUAL selection of cluster centers
    from the decision graph (rho vs delta plot).

    DPA's improvements over standard DP:
    1. Adaptive k selection (PAk) instead of fixed k-NN
    2. Automatic center detection (no manual decision graph inspection)
    3. Statistical error bounds for density estimates
    4. Automatic cluster merging based on significance

    Parameters
    ----------
    k : int, default=20
        Number of neighbors for density estimation.

    percent : float, default=2.0
        Percentage of points to select as centers (gamma threshold).
        In standard DP, this is done manually by visual inspection.
        Here we automate it for fair comparison.

    metric : str, default='euclidean'
        Distance metric.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point.

    cluster_centers_ : ndarray
        Indices of cluster centers.

    rho_ : ndarray of shape (n_samples,)
        Local density for each point.

    delta_ : ndarray of shape (n_samples,)
        Distance to nearest higher-density point.

    gamma_ : ndarray of shape (n_samples,)
        Decision graph score: gamma = rho * delta.

    References
    ----------
    Rodriguez, A., & Laio, A. (2014). Clustering by fast search and find
    of density peaks. Science, 344(6191), 1492-1496.
    """

    def __init__(self, k=20, percent=2.0, metric='euclidean'):
        self.k = k
        self.percent = percent
        self.metric = metric

        # Attributes set after fit
        self.labels_ = None
        self.cluster_centers_ = None
        self.rho_ = None
        self.delta_ = None
        self.gamma_ = None
        self.n_clusters_ = None

    def fit(self, X):
        """
        Fit the Density Peaks model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        from sklearn.neighbors import NearestNeighbors

        n_samples = X.shape[0]

        # Step 1: Compute k-NN density (fixed k, unlike DPA's adaptive PAk)
        nn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Local density: inverse of average distance to k neighbors
        # (simpler than DPA's PAk estimator)
        self.rho_ = 1.0 / (np.mean(distances[:, 1:self.k+1], axis=1) + 1e-10)

        # Step 2: Compute delta (distance to nearest higher-density point)
        self.delta_ = np.zeros(n_samples)
        nearest_higher = np.zeros(n_samples, dtype=int)

        # Sort by density (descending)
        rho_order = np.argsort(-self.rho_)

        for i, idx in enumerate(rho_order):
            if i == 0:
                # Highest density point: delta = max distance
                self.delta_[idx] = np.max(distances[idx])
                nearest_higher[idx] = idx
            else:
                # Find nearest point with higher density
                higher_density_points = rho_order[:i]
                min_dist = np.inf
                nearest = idx

                for hdp in higher_density_points:
                    # Compute distance to this higher-density point
                    dist = np.linalg.norm(X[idx] - X[hdp])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = hdp

                self.delta_[idx] = min_dist
                nearest_higher[idx] = nearest

        # Step 3: Compute gamma = rho * delta (decision graph score)
        # Normalize for numerical stability
        rho_norm = self.rho_ / np.max(self.rho_)
        delta_norm = self.delta_ / np.max(self.delta_)
        self.gamma_ = rho_norm * delta_norm

        # Step 4: Select cluster centers
        # In STANDARD DP: this is done MANUALLY by inspecting the decision graph!
        # Here we automate it using a percentile threshold for fair comparison
        n_centers = max(1, int(n_samples * self.percent / 100))
        center_indices = np.argsort(-self.gamma_)[:n_centers]
        self.cluster_centers_ = center_indices
        self.n_clusters_ = len(center_indices)

        # Step 5: Assign remaining points by following density gradient
        self.labels_ = np.full(n_samples, -1, dtype=int)

        # Assign centers first
        for c_id, center in enumerate(self.cluster_centers_):
            self.labels_[center] = c_id

        # Assign remaining points in order of decreasing density
        for idx in rho_order:
            if self.labels_[idx] == -1:
                # Follow to nearest higher-density point
                self.labels_[idx] = self.labels_[nearest_higher[idx]]

        return self

    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_

    def get_decision_graph_data(self):
        """
        Get data for plotting the decision graph.

        Returns
        -------
        dict with keys:
            'rho': local density values
            'delta': distance to higher-density point
            'gamma': decision score (rho * delta)
            'centers': indices of selected centers
        """
        return {
            'rho': self.rho_,
            'delta': self.delta_,
            'gamma': self.gamma_,
            'centers': self.cluster_centers_
        }
