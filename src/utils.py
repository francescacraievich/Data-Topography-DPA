"""
Utility functions for DPA clustering.

Helper functions for computing volumes, distances, and other utilities
used throughout the DPA algorithm.

Includes Tangent Distance for image data (MNIST), which provides
invariance to small transformations (translations, rotations, scaling).
"""

import os
import numpy as np
from scipy.special import gamma
from scipy.ndimage import sobel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

# Try to import FAISS for faster k-NN
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Try to import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        # Get GPU memory info
        GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        GPU_MEM_GB = 0
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    GPU_MEM_GB = 0


def compute_knn_faiss(X, k_max):
    """
    Compute k-NN using FAISS (10-50x faster than sklearn).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points (must be float32).
    k_max : int
        Maximum number of neighbors.

    Returns
    -------
    distances : ndarray of shape (n_samples, k_max + 1)
        Distances to k_max nearest neighbors (including self).
    indices : ndarray of shape (n_samples, k_max + 1)
        Indices of k_max nearest neighbors.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

    n_samples, n_features = X.shape
    k_actual = min(k_max + 1, n_samples)

    # FAISS requires float32 and C-contiguous arrays
    X_faiss = np.ascontiguousarray(X.astype(np.float32))

    # Create FAISS index (L2 = Euclidean squared)
    index = faiss.IndexFlatL2(n_features)
    index.add(X_faiss)

    # Search for k nearest neighbors
    distances_sq, indices = index.search(X_faiss, k_actual)

    # Convert squared distances to Euclidean distances
    distances = np.sqrt(np.maximum(distances_sq, 0))

    return distances, indices


def compute_knn(X, k_max, metric='euclidean', algorithm='auto', use_faiss='auto'):
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
    use_faiss : str or bool, default='auto'
        Whether to use FAISS for k-NN computation.
        - 'auto': Use FAISS if available and metric is euclidean
        - True: Force FAISS (raises error if not available)
        - False: Use sklearn

    Returns
    -------
    distances : ndarray of shape (n_samples, k_max + 1)
        Distances to k_max nearest neighbors (including self at index 0).
    indices : ndarray of shape (n_samples, k_max + 1)
        Indices of k_max nearest neighbors.
    """
    # Decide whether to use FAISS
    if use_faiss == 'auto':
        use_faiss = FAISS_AVAILABLE and metric == 'euclidean'

    if use_faiss:
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS requested but not available. Install with: pip install faiss-cpu")
        return compute_knn_faiss(X, k_max)

    # Fallback to sklearn
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

def compute_tangent_vectors(image, image_shape, orthonormalize=True):
    """
    Compute tangent vectors for an image representing infinitesimal transformations.

    Computes gradients that represent the direction of change for:
    1. Horizontal translation
    2. Vertical translation
    3. Rotation
    4. Scaling
    5. Axis elongation (diagonal scaling)
    6. Thickness (line thickening/thinning via Laplacian)
    7. Horizontal shear (optional)

    Parameters
    ----------
    image : ndarray of shape (n_pixels,)
        Flattened image.
    image_shape : tuple (height, width)
        Original image dimensions.
    orthonormalize : bool, default=True
        If True, orthonormalize tangent vectors using SVD.
        This is CRITICAL for proper tangent distance computation.

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
    - Thickness: ∇²I (Laplacian - dilation/erosion)

    Reference: Simard, LeCun, Denker (1993) "Efficient Pattern Recognition
    Using a New Transformation Distance"
    """
    from scipy.ndimage import laplace

    h, w = image_shape
    img_2d = image.reshape(h, w)

    # Compute image gradients using Sobel filters
    grad_x = sobel(img_2d, axis=1, mode='constant') / 4.0
    grad_y = sobel(img_2d, axis=0, mode='constant') / 4.0

    # Create coordinate grids (centered at image center)
    y_coords, x_coords = np.ogrid[:h, :w]
    x_centered = (x_coords - w / 2).astype(float)
    y_centered = (y_coords - h / 2).astype(float)

    tangent_vectors = []

    # 1. Horizontal translation: ∂I/∂x
    tangent_vectors.append(grad_x.flatten())

    # 2. Vertical translation: ∂I/∂y
    tangent_vectors.append(grad_y.flatten())

    # 3. Rotation: -y·∂I/∂x + x·∂I/∂y
    rotation = (-y_centered * grad_x + x_centered * grad_y).flatten()
    tangent_vectors.append(rotation)

    # 4. Scaling (isotropic): x·∂I/∂x + y·∂I/∂y
    scaling = (x_centered * grad_x + y_centered * grad_y).flatten()
    tangent_vectors.append(scaling)

    # 5. Axis elongation (anisotropic): x·∂I/∂x - y·∂I/∂y
    # This captures stretching in x direction vs y direction
    elongation = (x_centered * grad_x - y_centered * grad_y).flatten()
    tangent_vectors.append(elongation)

    # 6. Thickness (line thickening): ∇²I (Laplacian)
    # Positive Laplacian = thickening, negative = thinning
    thickness = laplace(img_2d, mode='constant').flatten()
    tangent_vectors.append(thickness)

    # 7. Horizontal shear: y·∂I/∂x
    h_shear = (y_centered * grad_x).flatten()
    tangent_vectors.append(h_shear)

    tv_array = np.array(tangent_vectors)

    if orthonormalize:
        # Orthonormalize using SVD for numerical stability
        # This ensures tangent vectors span an orthonormal subspace
        # which makes the projection well-conditioned
        U, s, Vt = np.linalg.svd(tv_array, full_matrices=False)
        # Keep only directions with significant singular values
        # (removes near-zero directions that cause numerical issues)
        tol = 1e-10 * s[0] if len(s) > 0 else 1e-10
        rank = np.sum(s > tol)
        if rank > 0:
            # Return orthonormalized tangent vectors
            # Each row of Vt[:rank] is an orthonormal basis vector
            tv_array = s[:rank, np.newaxis] * Vt[:rank]

    return tv_array


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
            # Solve: min ||diff - T2'·β||² using SVD-based least squares
            # This handles ill-conditioned/singular matrices properly
            T2 = tangent_vectors_2  # (n_tangent, n_pixels)
            # lstsq solves: min ||T2' @ beta - diff||², i.e., we need T2.T as the matrix
            # Reshape for lstsq: we want min ||A @ x - b||² where A = T2.T, x = beta, b = diff
            beta, residuals, rank, s = np.linalg.lstsq(T2.T, diff, rcond=None)
            projected_diff = diff - T2.T @ beta
        else:
            projected_diff = diff

        return np.sqrt(np.sum(projected_diff ** 2))

    else:  # two_sided
        # Combine tangent spaces
        if tangent_vectors_1 is not None and tangent_vectors_2 is not None:
            T_combined = np.vstack([tangent_vectors_1, tangent_vectors_2])

            # Solve: min ||diff - T_combined'·params||² using SVD-based least squares
            params, residuals, rank, s = np.linalg.lstsq(T_combined.T, diff, rcond=None)
            projected_diff = diff - T_combined.T @ params
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


def compute_full_tangent_distance_matrix(X, image_shape, mode='one_sided',
                                          n_jobs=-1, verbose=True, cache_file=None):
    """
    Compute FULL pairwise Tangent Distance matrix.

    This computes ALL N×N pairwise distances as described in d'Errico et al. (2021)
    Section 3.2: "We compute the pairwise distances using the tangent distance".

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_pixels)
        Flattened images.
    image_shape : tuple (height, width)
        Original image dimensions (e.g., (28, 28) for MNIST).
    mode : str, default='one_sided'
        Tangent distance mode ('one_sided' or 'two_sided').
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all CPUs.
    verbose : bool, default=True
        Print progress information.
    cache_file : str or None
        Path to cache the distance matrix. If file exists, loads from cache.

    Returns
    -------
    dist_matrix : ndarray of shape (n_samples, n_samples)
        Full pairwise Tangent Distance matrix.

    Notes
    -----
    For N=10,000 images: ~50 million distance computations.
    With parallelization, expect ~20-40 minutes on modern CPU.

    Memory requirement: N² × 4 bytes = ~400MB for N=10,000.
    """
    from joblib import Parallel, delayed
    import multiprocessing

    # Check cache first
    if cache_file is not None and os.path.exists(cache_file):
        if verbose:
            print(f"    Loading cached distance matrix from {cache_file}")
        cached = np.load(cache_file)
        return cached['dist_matrix']

    n_samples = X.shape[0]
    n_pairs = n_samples * (n_samples - 1) // 2

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if verbose:
        print(f"    Computing FULL Tangent Distance matrix...")
        print(f"    Samples: {n_samples}, Pairs to compute: {n_pairs:,}")
        print(f"    Using {n_jobs} parallel job(s)")
        print(f"    Step 1: Precomputing tangent vectors...")

    # Step 1: Precompute all tangent vectors
    def compute_tv(i):
        return compute_tangent_vectors(X[i], image_shape)

    if n_jobs > 1:
        tangent_vectors_all = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(compute_tv)(i) for i in range(n_samples)
        )
    else:
        tangent_vectors_all = [compute_tangent_vectors(X[i], image_shape)
                               for i in range(n_samples)]

    if verbose:
        print(f"    Step 2: Computing pairwise Tangent Distances...")

    # Step 2: Compute pairwise distances
    # We compute upper triangle and mirror to lower triangle

    def compute_row_distances(i, X, tangent_vectors_all, mode):
        """Compute distances from point i to all points j > i."""
        tv_i = tangent_vectors_all[i]
        n = len(tangent_vectors_all)
        distances = np.zeros(n - i - 1)

        for j_idx, j in enumerate(range(i + 1, n)):
            tv_j = tangent_vectors_all[j]
            distances[j_idx] = tangent_distance(
                X[i], X[j],
                tangent_vectors_1=tv_i,
                tangent_vectors_2=tv_j,
                mode=mode
            )
        return i, distances

    # Parallel computation of rows
    if n_jobs > 1:
        # Use progress tracking with batches
        batch_size = max(1, n_samples // 100)  # 100 progress updates
        results = []

        for batch_start in range(0, n_samples - 1, batch_size):
            batch_end = min(batch_start + batch_size, n_samples - 1)
            batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(compute_row_distances)(i, X, tangent_vectors_all, mode)
                for i in range(batch_start, batch_end)
            )
            results.extend(batch_results)

            if verbose:
                pct = 100 * batch_end / (n_samples - 1)
                print(f"           Progress: {batch_end}/{n_samples-1} rows ({pct:.0f}%)")

    else:
        # Sequential with progress
        results = []
        for i in range(n_samples - 1):
            result = compute_row_distances(i, X, tangent_vectors_all, mode)
            results.append(result)
            if verbose and (i + 1) % 500 == 0:
                pct = 100 * (i + 1) / (n_samples - 1)
                print(f"           Progress: {i+1}/{n_samples-1} rows ({pct:.0f}%)")

    # Assemble distance matrix
    if verbose:
        print(f"    Step 3: Assembling distance matrix...")

    dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i, distances in results:
        dist_matrix[i, i+1:] = distances
        dist_matrix[i+1:, i] = distances  # Symmetric

    # Save to cache if requested
    if cache_file is not None:
        if verbose:
            print(f"    Saving distance matrix to {cache_file}")
        np.savez_compressed(cache_file, dist_matrix=dist_matrix)

    if verbose:
        mem_mb = dist_matrix.nbytes / (1024 * 1024)
        print(f"    Done! Matrix size: {mem_mb:.1f} MB")

    return dist_matrix


def knn_from_distance_matrix(dist_matrix, k_max):
    """
    Extract k-NN from a precomputed distance matrix.

    Parameters
    ----------
    dist_matrix : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    k_max : int
        Maximum number of neighbors.

    Returns
    -------
    distances : ndarray of shape (n_samples, k_max + 1)
        Distances to k_max nearest neighbors (including self at index 0).
    indices : ndarray of shape (n_samples, k_max + 1)
        Indices of k_max nearest neighbors.
    """
    n_samples = dist_matrix.shape[0]
    k_actual = min(k_max + 1, n_samples)

    # Sort each row to find k-nearest neighbors
    indices = np.argsort(dist_matrix, axis=1)[:, :k_actual]
    distances = np.array([dist_matrix[i, indices[i]] for i in range(n_samples)])

    return distances, indices


def compute_knn_tangent_gpu(X, k_max, image_shape, mode='one_sided',
                            k_candidates=None, verbose=True, batch_size=1000):
    """
    GPU-accelerated k-NN using Tangent Distance.

    Uses PyTorch CUDA for ~10-20x speedup over CPU on compatible GPUs.
    Optimized for GPUs with limited VRAM (~4GB like RTX 500 Ada).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_pixels)
        Flattened images.
    k_max : int
        Number of neighbors to return.
    image_shape : tuple (height, width)
        Original image dimensions.
    mode : str, default='one_sided'
        Tangent distance mode.
    k_candidates : int or None
        Number of Euclidean candidates (default: 2 * k_max).
    verbose : bool, default=True
        Print progress.
    batch_size : int, default=1000
        Batch size for GPU processing. Reduce if running out of VRAM.

    Returns
    -------
    distances, indices : ndarrays
        k-NN distances and indices using Tangent Distance.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")
    if not CUDA_AVAILABLE:
        print("    Warning: CUDA not available, falling back to CPU PyTorch")

    device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')

    n_samples = X.shape[0]
    if k_candidates is None:
        k_candidates = min(2 * k_max, n_samples - 1)

    k_actual = min(k_max + 1, n_samples)
    k_cand_actual = min(k_candidates + 1, n_samples)

    if verbose:
        print(f"    GPU Tangent Distance k-NN (device: {device})")
        if CUDA_AVAILABLE:
            print(f"    GPU Memory: {GPU_MEM_GB:.1f} GB, batch_size={batch_size}")
        print(f"    Step 1: Finding {k_cand_actual} Euclidean candidates...")

    # Step 1: Find Euclidean candidates (use FAISS if available)
    if FAISS_AVAILABLE:
        _, candidate_indices = compute_knn_faiss(X, k_candidates)
    else:
        nn = NearestNeighbors(n_neighbors=k_cand_actual, metric='euclidean')
        nn.fit(X)
        _, candidate_indices = nn.kneighbors(X)

    if verbose:
        print(f"    Step 2: Precomputing tangent vectors on GPU...")

    # Step 2: Compute tangent vectors on GPU
    X_torch = torch.from_numpy(X.astype(np.float32)).to(device)
    h, w = image_shape

    # Sobel kernels for GPU
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

    def compute_tangent_vectors_gpu(images_batch):
        """Compute tangent vectors for a batch of images."""
        batch_n = images_batch.shape[0]
        imgs = images_batch.view(batch_n, 1, h, w)

        # Pad and apply Sobel filters
        padded = torch.nn.functional.pad(imgs, (1, 1, 1, 1), mode='reflect')
        dx = torch.nn.functional.conv2d(padded, sobel_x).view(batch_n, -1)
        dy = torch.nn.functional.conv2d(padded, sobel_y).view(batch_n, -1)

        # Create coordinate grids for transformation tangents
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing='ij'
        )
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        cx, cy = w / 2.0, h / 2.0

        # Tangent vectors: [t_x, t_y, t_rot, t_scale, t_parallel, t_diag, t_thick]
        # Shape: (batch_n, 7, n_pixels)
        n_pixels = h * w
        tangents = torch.zeros(batch_n, 7, n_pixels, device=device)

        # Translation tangents
        tangents[:, 0, :] = dx  # horizontal
        tangents[:, 1, :] = dy  # vertical

        # Rotation tangent
        tangents[:, 2, :] = dx * (-(y_coords - cy)) + dy * (x_coords - cx)

        # Scaling tangent
        tangents[:, 3, :] = dx * (x_coords - cx) + dy * (y_coords - cy)

        # Hyperbolic (parallel)
        tangents[:, 4, :] = dx * (x_coords - cx) - dy * (y_coords - cy)

        # Diagonal hyperbolic
        tangents[:, 5, :] = dx * (y_coords - cy) + dy * (x_coords - cx)

        # Thickening (approximate with Laplacian)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
        tangents[:, 6, :] = torch.nn.functional.conv2d(padded, laplacian).view(batch_n, -1)

        # Normalize
        norms = torch.norm(tangents, dim=2, keepdim=True) + 1e-10
        tangents = tangents / norms

        return tangents

    # Compute all tangent vectors in batches
    all_tangent_vectors = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_tv = compute_tangent_vectors_gpu(X_torch[start:end])
        all_tangent_vectors.append(batch_tv.cpu())
        if verbose and end % 10000 == 0:
            print(f"           Tangent vectors: {end}/{n_samples}")

    # Concatenate and move to GPU for distance computation
    all_tv = torch.cat(all_tangent_vectors, dim=0)  # (n_samples, 7, n_pixels)

    if verbose:
        print(f"    Step 3: Computing Tangent Distances on GPU...")

    # Step 3: Compute Tangent Distances in batches
    td_distances = np.zeros((n_samples, k_cand_actual), dtype=np.float32)

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_size_actual = batch_end - batch_start

        # Get batch data
        X_batch = X_torch[batch_start:batch_end]  # (batch, n_pixels)
        tv_batch = all_tv[batch_start:batch_end].to(device)  # (batch, 7, n_pixels)
        cand_idx_batch = candidate_indices[batch_start:batch_end]  # (batch, k_cand)

        # For each point in batch, compute TD to all its candidates
        for local_i in range(batch_size_actual):
            global_i = batch_start + local_i
            x_i = X_batch[local_i]
            tv_i = tv_batch[local_i] if mode == 'two_sided' else None

            # Get candidate points and their tangent vectors
            cand_indices = cand_idx_batch[local_i]
            X_cand = X_torch[cand_indices]  # (k_cand, n_pixels)
            tv_cand = all_tv[cand_indices].to(device)  # (k_cand, 7, n_pixels)

            # Compute diff vectors
            diffs = x_i.unsqueeze(0) - X_cand  # (k_cand, n_pixels)

            if mode == 'one_sided':
                # Project onto tangent space of each candidate
                # T2 @ T2.T for each candidate
                T2T2 = torch.bmm(tv_cand, tv_cand.transpose(1, 2))  # (k_cand, 7, 7)

                # T2 @ diff for each candidate
                T2_diff = torch.bmm(tv_cand, diffs.unsqueeze(2)).squeeze(2)  # (k_cand, 7)

                # Solve (T2T2 + eps*I) @ beta = T2_diff
                reg = 1e-6 * torch.eye(7, device=device).unsqueeze(0)
                T2T2_reg = T2T2 + reg

                try:
                    beta = torch.linalg.solve(T2T2_reg, T2_diff)  # (k_cand, 7)
                except:
                    beta = torch.zeros(k_cand_actual, 7, device=device)

                # projected_diff = diff - T2.T @ beta
                correction = torch.bmm(tv_cand.transpose(1, 2), beta.unsqueeze(2)).squeeze(2)
                projected_diffs = diffs - correction

                # Compute distances
                distances = torch.norm(projected_diffs, dim=1).cpu().numpy()

            else:  # two_sided
                # Similar but combining both tangent spaces
                tv_i_expanded = tv_i.unsqueeze(0).expand(k_cand_actual, -1, -1)
                T_combined = torch.cat([tv_i_expanded, tv_cand], dim=1)  # (k_cand, 14, n_pixels)

                TTT = torch.bmm(T_combined, T_combined.transpose(1, 2))
                T_diff = torch.bmm(T_combined, diffs.unsqueeze(2)).squeeze(2)

                reg = 1e-6 * torch.eye(14, device=device).unsqueeze(0)
                TTT_reg = TTT + reg

                try:
                    params = torch.linalg.solve(TTT_reg, T_diff)
                except:
                    params = torch.zeros(k_cand_actual, 14, device=device)

                correction = torch.bmm(T_combined.transpose(1, 2), params.unsqueeze(2)).squeeze(2)
                projected_diffs = diffs - correction
                distances = torch.norm(projected_diffs, dim=1).cpu().numpy()

            # Handle self-distance
            self_mask = cand_indices == global_i
            distances[self_mask] = 0.0

            td_distances[global_i] = distances

        if verbose and batch_end % 5000 == 0:
            print(f"           Distances: {batch_end}/{n_samples}")

        # Clear GPU cache periodically
        if CUDA_AVAILABLE and batch_end % 10000 == 0:
            torch.cuda.empty_cache()

    if verbose:
        print(f"    Step 4: Sorting neighbors...")

    # Step 4: Sort and select top k
    final_distances = np.zeros((n_samples, k_actual), dtype=np.float32)
    final_indices = np.zeros((n_samples, k_actual), dtype=np.int64)

    for i in range(n_samples):
        sorted_order = np.argsort(td_distances[i])[:k_actual]
        final_indices[i] = candidate_indices[i][sorted_order]
        final_distances[i] = td_distances[i][sorted_order]

    if verbose:
        print(f"    Done! GPU Tangent Distance k-NN complete.")

    return final_distances, final_indices
