"""
Utility functions for DPA clustering.

Helper functions for computing volumes, distances, and other utilities
used throughout the DPA algorithm.

Includes Tangent Distance for image data (MNIST), which provides
invariance to small transformations (translations, rotations, scaling).
"""

import numpy as np
from scipy.special import gamma
from scipy.ndimage import sobel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def compute_knn(X, k_max, metric='euclidean', algorithm='auto'):
    """
    Compute k-NN graph efficiently.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points.
    k_max : int
        Maximum number of neighbors to compute.
    metric : str, default='euclidean'
        Distance metric.
    algorithm : str, default='auto'
        Algorithm for NearestNeighbors ('auto', 'ball_tree', 'kd_tree', 'brute').

    Returns
    -------
    distances : ndarray of shape (n_samples, k_max + 1)
        Distances to k_max nearest neighbors (including self at index 0).
    indices : ndarray of shape (n_samples, k_max + 1)
        Indices of k_max nearest neighbors.
    """
    n_samples = X.shape[0]
    k_actual = min(k_max + 1, n_samples)

    nn = NearestNeighbors(n_neighbors=k_actual, metric=metric, algorithm=algorithm)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    return distances, indices


def unit_ball_volume(d):
    """
    Compute volume of d-dimensional unit ball.

    V_d = pi^(d/2) / Gamma(d/2 + 1)

    Parameters
    ----------
    d : float
        Dimension.

    Returns
    -------
    float
        Volume of unit d-ball.
    """
    return (np.pi ** (d / 2)) / gamma(d / 2 + 1)


def shell_volume(r_outer, r_inner, d):
    """
    Compute volume of spherical shell in d dimensions.

    V = omega_d * (r_outer^d - r_inner^d)

    Parameters
    ----------
    r_outer : float or ndarray
        Outer radius.
    r_inner : float or ndarray
        Inner radius.
    d : float
        Dimension.

    Returns
    -------
    float or ndarray
        Volume of spherical shell.
    """
    omega_d = unit_ball_volume(d)
    return omega_d * (r_outer ** d - r_inner ** d)


def ball_volume(r, d):
    """
    Compute volume of d-dimensional ball with radius r.

    V = omega_d * r^d

    Parameters
    ----------
    r : float or ndarray
        Radius.
    d : float
        Dimension.

    Returns
    -------
    float or ndarray
        Volume of ball.
    """
    omega_d = unit_ball_volume(d)
    return omega_d * (r ** d)


def empirical_cdf(x):
    """
    Compute empirical cumulative distribution function.

    Parameters
    ----------
    x : ndarray
        Data values.

    Returns
    -------
    x_sorted : ndarray
        Sorted x values.
    cdf : ndarray
        Empirical CDF values F(x) = i / n
        Following the paper: Facco et al. (2017), Eq. 5.
    """
    n = len(x)
    x_sorted = np.sort(x)
    cdf = np.arange(1, n + 1) / n
    return x_sorted, cdf


def linear_regression_through_origin(x, y):
    """
    Linear regression through origin: y = slope * x

    Parameters
    ----------
    x : ndarray
        Independent variable.
    y : ndarray
        Dependent variable.

    Returns
    -------
    slope : float
        Estimated slope.
    """
    return np.sum(x * y) / np.sum(x ** 2)


def safe_log(x, eps=1e-300):
    """
    Safe logarithm avoiding log(0).

    Parameters
    ----------
    x : float or ndarray
        Input values.
    eps : float
        Small value to add to avoid log(0).

    Returns
    -------
    float or ndarray
        log(x + eps)
    """
    return np.log(np.maximum(x, eps))


def compute_delta(g, distances, indices):
    """
    Compute delta_i = minimum distance to a point with higher g.

    For cluster center detection in Density Peak clustering.

    Parameters
    ----------
    g : ndarray of shape (n_samples,)
        Error-adjusted log-density g_i = log(rho_i) - epsilon_i.
    distances : ndarray of shape (n_samples, k_max + 1)
        Distances to neighbors.
    indices : ndarray of shape (n_samples, k_max + 1)
        Neighbor indices.

    Returns
    -------
    delta : ndarray of shape (n_samples,)
        Delta values for each point.
    nearest_higher : ndarray of shape (n_samples,)
        Index of nearest point with higher g (-1 if none found in neighbors).
    """
    from scipy.spatial.distance import cdist

    n_samples = len(g)
    delta = np.full(n_samples, np.inf)
    nearest_higher = np.full(n_samples, -1, dtype=int)

    # Sort points by decreasing g
    sorted_idx = np.argsort(-g)

    # For the point with highest g, delta is the max distance to any other point
    # We'll handle this specially

    # For each point, find nearest neighbor with higher g
    for rank, i in enumerate(sorted_idx):
        if rank == 0:
            # Highest g point: delta = max distance (will set later)
            continue

        # Points with higher g are those with rank < current rank
        higher_g_points = sorted_idx[:rank]

        # Check neighbors first (fast path)
        for j_idx in range(1, indices.shape[1]):  # skip self at 0
            j = indices[i, j_idx]
            if g[j] > g[i]:
                delta[i] = distances[i, j_idx]
                nearest_higher[i] = j
                break

        # If no neighbor with higher g found in k-NN, need full search
        if nearest_higher[i] == -1 and len(higher_g_points) > 0:
            # This is rare but can happen
            # For now, use the first higher-g neighbor we can find
            for j in higher_g_points:
                if j in indices[i]:
                    j_pos = np.where(indices[i] == j)[0][0]
                    if delta[i] > distances[i, j_pos]:
                        delta[i] = distances[i, j_pos]
                        nearest_higher[i] = j

    # Set delta for highest-g point to max of all other deltas
    max_delta_idx = sorted_idx[0]
    if np.any(np.isfinite(delta[delta != np.inf])):
        delta[max_delta_idx] = np.max(delta[np.isfinite(delta)])
    else:
        delta[max_delta_idx] = np.max(distances[:, -1])  # max k-th neighbor distance

    return delta, nearest_higher


def assign_to_clusters_by_density(g, nearest_higher, centers):
    """
    Assign points to clusters following density gradient.

    Each non-center point is assigned to the same cluster as its
    nearest neighbor with higher g.

    Parameters
    ----------
    g : ndarray of shape (n_samples,)
        Error-adjusted log-density.
    nearest_higher : ndarray of shape (n_samples,)
        Index of nearest point with higher g.
    centers : ndarray
        Indices of cluster centers.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.
    """
    n_samples = len(g)
    labels = np.full(n_samples, -1, dtype=int)

    # Assign centers to their clusters
    center_set = set(centers)
    for cluster_id, center_idx in enumerate(centers):
        labels[center_idx] = cluster_id

    # Sort by decreasing g and assign following gradient
    sorted_idx = np.argsort(-g)

    for i in sorted_idx:
        if labels[i] >= 0:
            continue  # already assigned (is a center)

        # Follow chain to assigned point
        j = nearest_higher[i]
        if j >= 0 and labels[j] >= 0:
            labels[i] = labels[j]

    return labels


# =============================================================================
# Tangent Distance for Image Data
# =============================================================================

def compute_tangent_vectors(image, image_shape):
    """
    Compute tangent vectors for an image representing infinitesimal transformations.

    Computes gradients that represent the direction of change for:
    1. Horizontal translation
    2. Vertical translation
    3. Rotation
    4. Scaling
    5. Horizontal shear (optional)
    6. Vertical shear (optional)

    Parameters
    ----------
    image : ndarray of shape (n_pixels,)
        Flattened image.
    image_shape : tuple (height, width)
        Original image dimensions.

    Returns
    -------
    tangent_vectors : ndarray of shape (n_transformations, n_pixels)
        Tangent vectors for each transformation.

    Notes
    -----
    Tangent Distance was introduced by Simard et al. (1993) and is
    particularly effective for handwritten digit recognition (MNIST).

    The tangent vectors are computed as partial derivatives of the
    image with respect to transformation parameters:
    - Translation: ∂I/∂x, ∂I/∂y (image gradients)
    - Rotation: -y·∂I/∂x + x·∂I/∂y
    - Scaling: x·∂I/∂x + y·∂I/∂y

    Reference: Simard, LeCun, Denker (1993) "Efficient Pattern Recognition
    Using a New Transformation Distance"
    """
    h, w = image_shape
    img_2d = image.reshape(h, w)

    # Compute image gradients using Sobel filters
    grad_x = sobel(img_2d, axis=1, mode='constant') / 4.0  # normalize
    grad_y = sobel(img_2d, axis=0, mode='constant') / 4.0

    # Create coordinate grids (centered at image center)
    y_coords, x_coords = np.ogrid[:h, :w]
    x_centered = x_coords - w / 2
    y_centered = y_coords - h / 2

    tangent_vectors = []

    # 1. Horizontal translation: ∂I/∂x
    tangent_vectors.append(grad_x.flatten())

    # 2. Vertical translation: ∂I/∂y
    tangent_vectors.append(grad_y.flatten())

    # 3. Rotation: -y·∂I/∂x + x·∂I/∂y
    rotation = (-y_centered * grad_x + x_centered * grad_y).flatten()
    tangent_vectors.append(rotation)

    # 4. Scaling: x·∂I/∂x + y·∂I/∂y
    scaling = (x_centered * grad_x + y_centered * grad_y).flatten()
    tangent_vectors.append(scaling)

    # 5. Horizontal shear: y·∂I/∂x
    h_shear = (y_centered * grad_x).flatten()
    tangent_vectors.append(h_shear)

    # 6. Vertical shear: x·∂I/∂y
    v_shear = (x_centered * grad_y).flatten()
    tangent_vectors.append(v_shear)

    return np.array(tangent_vectors)


def tangent_distance(x1, x2, tangent_vectors_1=None, tangent_vectors_2=None,
                     image_shape=None, mode='two_sided'):
    """
    Compute Tangent Distance between two images.

    The Tangent Distance measures the minimum distance between the
    tangent hyperplanes of two images, providing invariance to small
    transformations.

    Parameters
    ----------
    x1 : ndarray of shape (n_pixels,)
        First flattened image.
    x2 : ndarray of shape (n_pixels,)
        Second flattened image.
    tangent_vectors_1 : ndarray or None
        Precomputed tangent vectors for x1.
    tangent_vectors_2 : ndarray or None
        Precomputed tangent vectors for x2.
    image_shape : tuple (height, width)
        Image dimensions (required if tangent_vectors not provided).
    mode : str, default='two_sided'
        - 'one_sided': Only consider transformations of x2 towards x1
        - 'two_sided': Consider transformations of both images

    Returns
    -------
    distance : float
        Tangent Distance between the two images.

    Notes
    -----
    Two-sided Tangent Distance (default):
        d(x1, x2) = min ||x1 + T1·α - (x2 + T2·β)||²

    Where T1, T2 are matrices of tangent vectors and α, β are
    transformation parameters found by least squares.

    One-sided Tangent Distance:
        d(x1, x2) = min ||x1 - (x2 + T2·β)||²

    The two-sided version is symmetric but more expensive.
    """
    if tangent_vectors_1 is None and image_shape is not None:
        tangent_vectors_1 = compute_tangent_vectors(x1, image_shape)
    if tangent_vectors_2 is None and image_shape is not None:
        tangent_vectors_2 = compute_tangent_vectors(x2, image_shape)

    diff = x1 - x2

    if mode == 'one_sided':
        # Project diff onto tangent space of x2
        if tangent_vectors_2 is not None and len(tangent_vectors_2) > 0:
            # Solve: min ||diff - T2·β||²
            # Solution: β = (T2·T2')^(-1)·T2·diff
            T2 = tangent_vectors_2
            T2T2 = T2 @ T2.T
            try:
                # Add small regularization for numerical stability
                T2T2_reg = T2T2 + 1e-10 * np.eye(T2T2.shape[0])
                beta = np.linalg.solve(T2T2_reg, T2 @ diff)
                projected_diff = diff - T2.T @ beta
            except np.linalg.LinAlgError:
                projected_diff = diff
        else:
            projected_diff = diff

        return np.sqrt(np.sum(projected_diff ** 2))

    else:  # two_sided
        # Combine tangent spaces
        if tangent_vectors_1 is not None and tangent_vectors_2 is not None:
            # T_combined = [T1, -T2] (negated because we're adding to x1 and x2)
            T_combined = np.vstack([tangent_vectors_1, tangent_vectors_2])

            # Solve: min ||diff - T_combined'·params||²
            TTT = T_combined @ T_combined.T
            try:
                TTT_reg = TTT + 1e-10 * np.eye(TTT.shape[0])
                params = np.linalg.solve(TTT_reg, T_combined @ diff)
                projected_diff = diff - T_combined.T @ params
            except np.linalg.LinAlgError:
                projected_diff = diff
        else:
            projected_diff = diff

        return np.sqrt(np.sum(projected_diff ** 2))


def compute_tangent_distance_matrix(X, image_shape, mode='one_sided', n_jobs=1):
    """
    Compute pairwise Tangent Distance matrix for a set of images.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_pixels)
        Flattened images.
    image_shape : tuple (height, width)
        Original image dimensions.
    mode : str, default='one_sided'
        Tangent distance mode ('one_sided' or 'two_sided').
    n_jobs : int, default=1
        Number of parallel jobs (currently not implemented).

    Returns
    -------
    distances : ndarray of shape (n_samples, n_samples)
        Pairwise Tangent Distance matrix.

    Notes
    -----
    This is computationally expensive: O(n²) distance computations,
    each requiring tangent vector computation and linear algebra.

    For large datasets, consider:
    1. Using one-sided mode (faster)
    2. Precomputing tangent vectors
    3. Using approximate methods or subsampling
    """
    n_samples = X.shape[0]

    # Precompute all tangent vectors
    tangent_vectors = [compute_tangent_vectors(X[i], image_shape)
                       for i in range(n_samples)]

    # Compute pairwise distances
    distances = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = tangent_distance(
                X[i], X[j],
                tangent_vectors_1=tangent_vectors[i],
                tangent_vectors_2=tangent_vectors[j],
                mode=mode
            )
            distances[i, j] = d
            distances[j, i] = d

    return distances


def compute_knn_tangent(X, k_max, image_shape, mode='one_sided'):
    """
    Compute k-NN using Tangent Distance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_pixels)
        Flattened images.
    k_max : int
        Maximum number of neighbors.
    image_shape : tuple (height, width)
        Original image dimensions.
    mode : str, default='one_sided'
        Tangent distance mode.

    Returns
    -------
    distances : ndarray of shape (n_samples, k_max + 1)
        Distances to k_max nearest neighbors (including self).
    indices : ndarray of shape (n_samples, k_max + 1)
        Indices of k_max nearest neighbors.

    Notes
    -----
    Uses Tangent Distance for finding neighbors, which is more appropriate
    for image data like MNIST than Euclidean distance.

    Warning: This is slow for large datasets due to O(n²) complexity.
    For production use, consider approximate methods.
    """
    # Compute full distance matrix
    dist_matrix = compute_tangent_distance_matrix(X, image_shape, mode=mode)

    n_samples = X.shape[0]
    k_actual = min(k_max + 1, n_samples)

    # Sort to find k-nearest neighbors
    indices = np.argsort(dist_matrix, axis=1)[:, :k_actual]
    distances = np.array([dist_matrix[i, indices[i]] for i in range(n_samples)])

    return distances, indices


class TangentDistanceMetric:
    """
    Sklearn-compatible Tangent Distance metric for use with DPA.

    This class wraps the tangent distance computation to be compatible
    with sklearn's metric interfaces.

    Parameters
    ----------
    image_shape : tuple (height, width)
        Original image dimensions (e.g., (28, 28) for MNIST).
    mode : str, default='one_sided'
        - 'one_sided': Faster, asymmetric
        - 'two_sided': Symmetric but slower

    Examples
    --------
    >>> from src.utils import TangentDistanceMetric
    >>> from src.dpa import DPA
    >>>
    >>> # For MNIST data
    >>> metric = TangentDistanceMetric(image_shape=(28, 28))
    >>> # Use precomputed distances
    >>> distances, indices = metric.compute_knn(X, k_max=100)
    >>> dpa = DPA(metric='precomputed')
    >>> dpa.fit(X, distances=distances, indices=indices)

    Notes
    -----
    Tangent Distance was shown by d'Errico et al. to significantly
    improve clustering quality on MNIST compared to Euclidean distance.
    The key insight is that digit images lie on a manifold where
    small transformations (rotation, translation) preserve identity.
    """

    def __init__(self, image_shape, mode='one_sided'):
        self.image_shape = image_shape
        self.mode = mode
        self._tangent_cache = {}

    def get_tangent_vectors(self, x, idx=None):
        """Get tangent vectors, using cache if available."""
        if idx is not None and idx in self._tangent_cache:
            return self._tangent_cache[idx]

        tv = compute_tangent_vectors(x, self.image_shape)

        if idx is not None:
            self._tangent_cache[idx] = tv

        return tv

    def __call__(self, x1, x2):
        """Compute tangent distance between two images."""
        return tangent_distance(x1, x2, image_shape=self.image_shape, mode=self.mode)

    def compute_knn(self, X, k_max):
        """
        Compute k-NN using tangent distance.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels)
            Flattened images.
        k_max : int
            Maximum number of neighbors.

        Returns
        -------
        distances : ndarray
            Distances to neighbors.
        indices : ndarray
            Neighbor indices.
        """
        return compute_knn_tangent(X, k_max, self.image_shape, self.mode)

    def clear_cache(self):
        """Clear the tangent vector cache."""
        self._tangent_cache = {}


def _compute_td_for_point(args):
    """Helper function for parallel TD computation."""
    i, X_i, candidate_indices_i, X_all, tangent_vectors_all, mode = args
    tv_i = tangent_vectors_all[i]
    distances = np.zeros(len(candidate_indices_i))

    for j_idx, j in enumerate(candidate_indices_i):
        if i == j:
            distances[j_idx] = 0.0
        else:
            tv_j = tangent_vectors_all[j]
            distances[j_idx] = tangent_distance(
                X_i, X_all[j],
                tangent_vectors_1=tv_i,
                tangent_vectors_2=tv_j,
                mode=mode
            )
    return i, distances


def compute_knn_tangent_efficient(X, k_max, image_shape, mode='one_sided',
                                   k_candidates=None, verbose=True, n_jobs=1):
    """
    Compute k-NN using Tangent Distance with efficient hybrid approach.

    This is the approach used in d'Errico et al. (2021) for MNIST:
    1. Find candidate neighbors using Euclidean distance (fast)
    2. Recompute distances for candidates using Tangent Distance
    3. Reorder neighbors by Tangent Distance

    This reduces complexity from O(n²) to O(n × k_candidates).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_pixels)
        Flattened images.
    k_max : int
        Number of neighbors to return.
    image_shape : tuple (height, width)
        Original image dimensions (e.g., (28, 28) for MNIST).
    mode : str, default='one_sided'
        Tangent distance mode.
    k_candidates : int or None
        Number of Euclidean candidates to consider (default: 2 * k_max).
        Using more candidates improves accuracy but increases computation.
    verbose : bool, default=True
        Print progress information.
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all CPUs.

    Returns
    -------
    distances : ndarray of shape (n_samples, k_max + 1)
        Tangent distances to k_max nearest neighbors (including self).
    indices : ndarray of shape (n_samples, k_max + 1)
        Indices of k_max nearest neighbors.

    Notes
    -----
    This hybrid approach is crucial for scalability:
    - Full Tangent Distance matrix for 60k images: O(60k²) = 3.6 billion computations
    - Hybrid with k_candidates=200: O(60k × 200) = 12 million computations
    - Speedup: ~300x

    Reference: d'Errico et al. (2021), Section 3.2
    "We compute the pairwise distances using the tangent distance"
    """
    from joblib import Parallel, delayed
    import multiprocessing

    n_samples = X.shape[0]
    if k_candidates is None:
        k_candidates = min(2 * k_max, n_samples - 1)

    k_actual = min(k_max + 1, n_samples)
    k_cand_actual = min(k_candidates + 1, n_samples)

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    if verbose:
        print(f"    Computing k-NN with Tangent Distance (hybrid approach)...")
        print(f"    Using {n_jobs} parallel job(s)")
        print(f"    Step 1: Finding {k_cand_actual} Euclidean candidates per point...")

    # Step 1: Find candidate neighbors using Euclidean distance (fast)
    nn_euclidean = NearestNeighbors(n_neighbors=k_cand_actual, metric='euclidean', n_jobs=n_jobs)
    nn_euclidean.fit(X)
    _, candidate_indices = nn_euclidean.kneighbors(X)

    if verbose:
        print(f"    Step 2: Precomputing tangent vectors for {n_samples} images...")

    # Step 2: Precompute all tangent vectors (can be parallelized)
    def compute_tv(i):
        return compute_tangent_vectors(X[i], image_shape)

    if n_jobs > 1:
        tangent_vectors_all = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(compute_tv)(i) for i in range(n_samples)
        )
    else:
        tangent_vectors_all = []
        for i in range(n_samples):
            tv = compute_tangent_vectors(X[i], image_shape)
            tangent_vectors_all.append(tv)
            if verbose and (i + 1) % 10000 == 0:
                print(f"           Processed {i + 1}/{n_samples} images...")

    if verbose:
        print(f"    Step 3: Computing Tangent Distances for candidates...")

    # Step 3: Compute Tangent Distance only for candidates
    td_distances = np.zeros((n_samples, k_cand_actual))

    if n_jobs > 1:
        # Parallel computation
        def compute_td_row(i):
            tv_i = tangent_vectors_all[i]
            distances = np.zeros(k_cand_actual)
            for j_idx, j in enumerate(candidate_indices[i]):
                if i == j:
                    distances[j_idx] = 0.0
                else:
                    tv_j = tangent_vectors_all[j]
                    distances[j_idx] = tangent_distance(
                        X[i], X[j],
                        tangent_vectors_1=tv_i,
                        tangent_vectors_2=tv_j,
                        mode=mode
                    )
            return distances

        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(compute_td_row)(i) for i in range(n_samples)
        )
        for i, distances in enumerate(results):
            td_distances[i] = distances
    else:
        # Sequential computation
        for i in range(n_samples):
            tv_i = tangent_vectors_all[i]
            for j_idx, j in enumerate(candidate_indices[i]):
                if i == j:
                    td_distances[i, j_idx] = 0.0
                else:
                    tv_j = tangent_vectors_all[j]
                    td_distances[i, j_idx] = tangent_distance(
                        X[i], X[j],
                        tangent_vectors_1=tv_i,
                        tangent_vectors_2=tv_j,
                        mode=mode
                    )

            if verbose and (i + 1) % 10000 == 0:
                print(f"           Processed {i + 1}/{n_samples} points...")

    if verbose:
        print(f"    Step 4: Sorting neighbors by Tangent Distance...")

    # Step 4: Sort candidates by Tangent Distance and take top k_actual
    final_distances = np.zeros((n_samples, k_actual))
    final_indices = np.zeros((n_samples, k_actual), dtype=int)

    for i in range(n_samples):
        sorted_order = np.argsort(td_distances[i])[:k_actual]
        final_indices[i] = candidate_indices[i][sorted_order]
        final_distances[i] = td_distances[i][sorted_order]

    if verbose:
        print(f"    Done! Computed Tangent Distance k-NN for {n_samples} images.")

    return final_distances, final_indices
