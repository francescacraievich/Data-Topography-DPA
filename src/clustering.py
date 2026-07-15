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
    # Search the FULL k-NN list to avoid missing nearby higher-g points
    for rank, i in enumerate(sorted_idx):
        if rank == 0:
            # Point with highest g has no point with higher g
            continue

        # Search ALL k-NN neighbors for nearest point with higher g
        for j_pos in range(1, indices.shape[1]):
            j = indices[i, j_pos]
            if g[j] > g[i]:
                delta[i] = distances[i, j_pos]
                nearest_higher[i] = j
                break

    # Set delta for highest-g point to max delta (following DP convention)
    max_g_idx = sorted_idx[0]
    finite_deltas = delta[np.isfinite(delta)]
    if len(finite_deltas) > 0:
        delta[max_g_idx] = np.max(finite_deltas) * 1.1
    else:
        delta[max_g_idx] = np.max(distances[:, -1])

    # Build reverse k-NN map for efficient Condition 2 check
    # reverse_nn[i] = set of points j that have i in j's k_hat neighborhood
    # Uses k_hat (optimal neighborhood size) as defined in the paper
    reverse_nn = defaultdict(set)
    for j in range(n_samples):
        k_j = min(k_hat[j], indices.shape[1] - 1)
        for pos in range(1, k_j + 1):
            reverse_nn[indices[j, pos]].add(j)

    # Find centers using two conditions from Heuristic 1
    centers = []

    for i in range(n_samples):
        # Get optimal neighborhood radius for point i
        k_i = min(k_hat[i], distances.shape[1] - 1)
        r_k_i = distances[i, k_i]

        # Condition 1: delta_i > r_{k_hat_i}
        if delta[i] <= r_k_i:
            continue  # skip Condition 2 check (optimization)

        # Condition 2: i not in neighborhood of any point with higher g
        # Use reverse k-NN map with FULL k-NN for thorough coverage
        # Paper: "i does not belong to NN_j for any j with g_j > g_i"
        cond2 = True
        for j in reverse_nn.get(i, set()):
            if g[j] > g[i]:
                cond2 = False
                break

        if cond2:
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


def find_saddle_points(labels, g, log_density, epsilon, distances, indices, k_hat,
                       centers=None):
    """
    Heuristic 2 (Section 2.1.3): Find saddle points between clusters.

    From the paper: "A point i belonging to cluster c is at the border
    between c and c' if:
    1. Its closest point j in c' is within distance r_{k_hat_i}
    2. i is the closest point to j among those in c"

    The saddle point is the border point with the highest g value.

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

    centers : ndarray or None
        Cluster center indices. Centers are excluded from being border points
        (matching official implementation).

    Returns
    -------
    saddles : dict
        Dictionary {(c1, c2): {'index': saddle_idx, 'g': saddle_g,
                              'log_density': saddle_log_rho, 'epsilon': saddle_eps}}

    border_points : dict
        Dictionary {(c1, c2): [list of border point indices]}
    """
    n_samples = len(labels)
    k_max = indices.shape[1] - 1

    # Build center set for exclusion
    center_set = set(centers) if centers is not None else set()

    saddles = {}
    border_points = defaultdict(list)

    # Step 1: Find border candidates
    # For each non-center point i, find the FIRST (nearest) neighbor within
    # k_hat that belongs to a different cluster.
    # border_dict[i] = {other_cluster: neighbor_index}
    border_dict = {}
    for i in range(n_samples):
        if labels[i] < 0 or i in center_set:
            continue

        c_i = labels[i]
        k_i = min(k_hat[i], k_max)

        for j_pos in range(1, k_i + 1):
            j = indices[i, j_pos]
            if j in center_set:
                continue
            if labels[j] >= 0 and labels[j] != c_i:
                c_j = labels[j]
                if i not in border_dict:
                    border_dict[i] = {}
                border_dict[i][c_j] = j
                break  # only first different-cluster neighbor

    # Step 2: Reciprocal verification
    # For each candidate (i, j) where i is in cluster c and j is in cluster cp,
    # verify from j's side: scan j's full k-NN. If we find i before finding
    # another point from cluster c, the border is confirmed.
    confirmed_borders = defaultdict(list)  # (c, cp) -> [(border_idx, g_value)]

    for i, cluster_neighbors in border_dict.items():
        c_i = labels[i]
        for c_j, j in cluster_neighbors.items():
            # Scan j's full neighborhood to verify reciprocity
            confirmed = False
            for k_pos in range(1, k_max + 1):
                z = indices[j, k_pos]
                if z == i:
                    # Confirmed: i is the nearest point from cluster c_i
                    # as seen from j
                    confirmed = True
                    break
                elif labels[z] == c_i:
                    # Found a closer point from c_i before reaching i
                    # Border (i, j) is rejected
                    break
                # Points from other clusters are skipped

            if confirmed:
                pair = tuple(sorted([c_i, c_j]))
                # Both i (from c) and j (from c') are border candidates
                confirmed_borders[pair].append(i)
                confirmed_borders[pair].append(j)
                border_points[pair].append(i)

    # Select saddle for each cluster pair: highest g among confirmed borders
    for pair, candidates in confirmed_borders.items():
        if len(candidates) == 0:
            continue

        g_values = np.array([g[idx] for idx in candidates])
        best_pos = np.argmax(g_values)
        saddle_idx = candidates[best_pos]

        saddles[pair] = {
            'index': saddle_idx,
            'g': g[saddle_idx],
            'log_density': log_density[saddle_idx],
            'epsilon': epsilon[saddle_idx]
        }

    return saddles, dict(border_points)


def merge_insignificant_peaks(centers, saddles, log_density, epsilon, Z):
    """
    Heuristic 3 (Section 2.1.4): Merge clusters with insignificant peaks.

    From the paper: "A cluster c is merged with a neighbouring cluster c' if:
      (log rho_c - log rho_{cc'}) < Z * (e_c + e_{cc'})
    where rho_c is the density of the center of cluster c."
    This is tested for BOTH clusters in the pair independently (c and c'),
    and the pair is merged if EITHER test fails - matching the official
    reference implementation (src/Pipeline/_DPA.pyx, get_borders), not
    just whichever of the two has the lower peak.

    The paper specifies iterative merging: "Heuristic 3 is checked for all
    clusters c and c' in order of decreasing log rho_{cc'}." At each step,
    the pair with highest border density is merged, then the topography
    is updated before re-evaluating.

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
        Significance threshold (only free parameter of DPA).

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

    # Working copies of border densities and errors
    rho_bord = defaultdict(lambda: defaultdict(lambda: -np.inf))
    rho_bord_err = defaultdict(lambda: defaultdict(lambda: 0.0))

    for (c1, c2), info in saddles.items():
        rho_bord[c1][c2] = info['log_density']
        rho_bord[c2][c1] = info['log_density']
        rho_bord_err[c1][c2] = info['epsilon']
        rho_bord_err[c2][c1] = info['epsilon']

    # Track merges
    merge_target = {}
    merge_history = []
    active = set(range(n_clusters))

    # Iterative merging (paper Algorithm 3):
    # 1. Check all pairs for merge condition
    # 2. Among mergeable pairs, select the one with highest border density
    # 3. Merge lower peak into higher peak
    # 4. Update topography (border inheritance)
    # 5. Repeat until no more merges
    while True:
        best_pair = None
        best_border = -np.inf

        for c in list(active):
            for cp in list(active):
                if cp <= c:
                    continue
                border_d = rho_bord[c][cp]
                if border_d == -np.inf:
                    continue

                border_e = rho_bord_err[c][cp]

                # Test EACH peak independently and merge if EITHER is not
                # significantly higher than the saddle (matches the official
                # reference implementation, src/Pipeline/_DPA.pyx get_borders:
                # `if a1<e1 or a2<e2`) - not just the lower of the two peaks.
                peak_c = log_density[centers[c]]
                peak_cp = log_density[centers[cp]]
                a1 = peak_c - border_d
                a2 = peak_cp - border_d
                e1 = Z * (epsilon[centers[c]] + border_e)
                e2 = Z * (epsilon[centers[cp]] + border_e)

                if a1 < e1 or a2 < e2:
                    if border_d > best_border:
                        best_border = border_d
                        # Merge the lower-density peak into the higher one.
                        if peak_c < peak_cp:
                            lower, higher = c, cp
                        else:
                            lower, higher = cp, c
                        best_pair = (lower, higher)

        if best_pair is None:
            break

        cmin, cmax = best_pair
        merge_target[cmin] = cmax
        merge_history.append((cmin, cmax, {}))
        active.discard(cmin)

        # Topography inheritance: cmax inherits cmin's borders
        for cp in list(active):
            if cp == cmax:
                continue
            border_min_cp = rho_bord[cmin][cp]
            if border_min_cp == -np.inf:
                continue
            # If cmax's border with cp is lower, inherit from cmin
            if rho_bord[cmax][cp] < border_min_cp:
                rho_bord[cmax][cp] = border_min_cp
                rho_bord[cp][cmax] = border_min_cp
                rho_bord_err[cmax][cp] = rho_bord_err[cmin][cp]
                rho_bord_err[cp][cmax] = rho_bord_err[cmin][cp]

        # Clean up cmin borders
        for cp in list(active):
            rho_bord[cmin][cp] = -np.inf
            rho_bord[cp][cmin] = -np.inf

    # Build final cluster mapping
    def find_final(c):
        while c in merge_target:
            c = merge_target[c]
        return c

    merge_map = {}
    new_id = 0
    for old_id in range(n_clusters):
        final = find_final(old_id)
        if final not in merge_map:
            merge_map[final] = new_id
            new_id += 1
        merge_map[old_id] = merge_map[final]

    merged_centers = [centers[i] for i in sorted(active)]

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

        # Step 2: Compute delta (distance to nearest higher-density point).
        # Row-by-row against only the higher-density subset (matching the
        # official reference implementation, DP/DP.py get_decision_graph,
        # which uses scipy.spatial.distance.cdist per point) rather than
        # materializing the full N x N matrix - at N~38k (e.g. the spir2
        # dataset) a dense float64 matrix is ~11GB and doesn't fit in
        # memory, regardless of how few features X has.
        from sklearn.metrics import pairwise_distances

        self.delta_ = np.full(n_samples, np.inf)
        nearest_higher = np.full(n_samples, -1, dtype=int)

        # Sort by density (descending)
        rho_order = np.argsort(-self.rho_)

        # For each point, find nearest point with strictly higher density
        for i in range(1, n_samples):
            idx = rho_order[i]
            # All points processed before this one have higher density
            higher_points = rho_order[:i]
            dists_to_higher = pairwise_distances(
                X[idx:idx + 1], X[higher_points], metric=self.metric)[0]
            min_pos = np.argmin(dists_to_higher)
            self.delta_[idx] = dists_to_higher[min_pos]
            nearest_higher[idx] = higher_points[min_pos]

        # Highest density point: delta = max distance to any point
        top_idx = rho_order[0]
        dists_from_top = pairwise_distances(X[top_idx:top_idx + 1], X, metric=self.metric)[0]
        self.delta_[top_idx] = np.max(dists_from_top)
        nearest_higher[top_idx] = top_idx

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
