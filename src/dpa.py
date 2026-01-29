"""
Density Peak Advanced (DPA) - Automatic Topography of High-Dimensional Data Sets

Implementation based on:
d'Errico, M., Facco, E., Laio, A., & Rodriguez, A. (2021).
"Automatic topography of high-dimensional data sets by non-parametric density peak clustering"
Information Sciences, 560, 476-492.

Main components:
1. TWO-NN intrinsic dimension estimation
2. PAk (Point Adaptive k-NN) density estimation
3. Automatic cluster center detection (Heuristic 1)
4. Saddle point detection (Heuristic 2)
5. Peak significance testing with Z-score (Heuristic 3)
6. Topographic visualization (dendrogram + network)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from .intrinsic_dimension import TwoNN
from .density import PAk
from .clustering import (
    find_cluster_centers,
    assign_to_clusters,
    find_saddle_points,
    merge_insignificant_peaks,
    identify_halo_points,
    relabel_after_merge
)
from .utils import compute_delta, assign_to_clusters_by_density, compute_knn
from .topography import Topography
from .utils import compute_knn


class DPA(BaseEstimator, ClusterMixin):
    """
    Density Peak Advanced (DPA) clustering algorithm.

    Automatically identifies clusters as density peaks and determines
    cluster boundaries through saddle points in the density landscape.
    Provides a topographic visualization of the clustering structure.

    Parameters
    ----------
    Z : float, default=2.0
        Significance threshold for peak merging. Higher Z = fewer clusters.
        This is the ONLY free parameter in DPA.
        - Z=1.0: ~68% confidence (liberal, more clusters)
        - Z=1.65: ~90% confidence
        - Z=2.0: ~95% confidence (balanced)
        - Z=3.0: ~99.7% confidence (conservative, fewer clusters)

        For high-dimensional data (like MNIST), use lower Z (~1.6) because
        density estimation errors are larger due to sparse sampling.

    d : float or None, default=None
        Intrinsic dimension. If None, estimated automatically using TWO-NN.
        Providing the known dimension can improve results.

    k_max : int, default=100
        Maximum number of neighbors for PAk density estimation.

    D_thr : float, default=23.928
        Likelihood ratio test threshold for PAk.
        Default corresponds to p-value 10^-6.

    metric : str, default='euclidean'
        Distance metric for neighbor computation.

    halo : bool, default=True
        Whether to identify halo points (set to -1 in labels_).
        Halo points have density below the max saddle of their cluster.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels. -1 indicates halo points (if halo=True).

    labels_full_ : ndarray of shape (n_samples,)
        Cluster labels for all points (no halo masking).

    cluster_centers_ : ndarray
        Indices of identified cluster centers (density peaks).

    log_density_ : ndarray of shape (n_samples,)
        Log-density estimate log(rho_i) for each point.

    free_energy_ : ndarray of shape (n_samples,)
        Free energy F_i = -log(rho_i). Physics terminology from paper.
        Peaks = free energy minima, saddles = transition states.

    epsilon_ : ndarray of shape (n_samples,)
        Density error estimate from Fisher Information.

    g_ : ndarray of shape (n_samples,)
        Error-adjusted log-density g_i = log(rho_i) - epsilon_i.

    k_hat_ : ndarray of shape (n_samples,)
        Optimal k selected by PAk for each point.

    d_ : float
        Intrinsic dimension (estimated or provided).

    topography_ : Topography
        Topographic representation of the clustering.

    saddles_ : dict
        Saddle point information between cluster pairs.

    halo_mask_ : ndarray of bool
        True for points identified as halo.

    n_clusters_ : int
        Number of clusters found.

    Notes
    -----
    The algorithm performs the following steps:
    1. Compute k-NN graph
    2. Estimate intrinsic dimension d using TWO-NN
    3. Estimate density using PAk with error bounds
    4. Find cluster centers (Algorithm 1 / Heuristic 1)
    5. Assign points to clusters
    6. Find saddle points (Algorithm 2 / Heuristic 2)
    7. Merge insignificant peaks (Algorithm 3 / Heuristic 3)
    8. Re-assign to merged clusters
    9. Identify halo points
    10. Build topography for visualization

    Examples
    --------
    >>> from src import load_optdigits
    >>> data = load_optdigits()
    >>> X, y = data['data'], data['target']
    >>> dpa = DPA(Z=2.0)
    >>> labels = dpa.fit_predict(X)
    >>> print(f"Found {dpa.n_clusters_} clusters")

    >>> # Visualize topography
    >>> dpa.plot_topography(kind='dendrogram')
    """

    def __init__(self, Z=2.0, d=None, k_max=100, D_thr=23.928,
                 metric='euclidean', halo=True):
        self.Z = Z
        self.d = d
        self.k_max = k_max
        self.D_thr = D_thr
        self.metric = metric
        self.halo = halo

        # Fitted attributes (initialized to None)
        self.labels_ = None
        self.labels_full_ = None
        self.cluster_centers_ = None
        self.log_density_ = None
        self.free_energy_ = None
        self.epsilon_ = None
        self.g_ = None
        self.k_hat_ = None
        self.d_ = None
        self.topography_ = None
        self.saddles_ = None
        self.halo_mask_ = None
        self.n_clusters_ = None

        # Internal storage
        self._distances = None
        self._indices = None
        self._delta = None
        self._nearest_higher = None
        self._merge_history = None

    def fit(self, X, y=None, distances=None, indices=None):
        """
        Fit the DPA clustering model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency.

        distances : ndarray of shape (n_samples, k_max + 1), optional
            Precomputed distances to neighbors. If provided along with indices,
            skip k-NN computation. Useful for custom distance metrics like
            Tangent Distance for MNIST.

        indices : ndarray of shape (n_samples, k_max + 1), optional
            Precomputed neighbor indices. Must be provided if distances is provided.

        Returns
        -------
        self : DPA
            Fitted estimator.

        Notes
        -----
        For MNIST data, use Tangent Distance for better results:

        >>> from src.utils import compute_knn_tangent_efficient
        >>> distances, indices = compute_knn_tangent_efficient(
        ...     X, k_max=100, image_shape=(28, 28)
        ... )
        >>> dpa = DPA(Z=1.6)
        >>> dpa.fit(X, distances=distances, indices=indices)

        Reference: d'Errico et al. (2021), Section 3.2
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        # Step 1: Compute or use precomputed k-NN graph
        if distances is not None and indices is not None:
            # Use precomputed distances (e.g., Tangent Distance)
            self._distances = np.asarray(distances)
            self._indices = np.asarray(indices)
        else:
            # Compute k-NN graph
            k_actual = min(self.k_max, n_samples - 1)
            self._distances, self._indices = compute_knn(X, k_actual, self.metric)

        # Step 2: Estimate intrinsic dimension using TWO-NN
        if self.d is None:
            twonn = TwoNN()
            twonn.fit(X, precomputed_distances=self._distances,
                      precomputed_indices=self._indices)
            self.d_ = twonn.dimension_
        else:
            self.d_ = self.d

        # Step 3: Estimate density using PAk
        pak = PAk(d=self.d_, k_max=self.k_max, D_thr=self.D_thr, metric=self.metric)
        pak.fit(X, distances=self._distances, indices=self._indices)

        self.log_density_ = pak.log_density_
        self.free_energy_ = pak.free_energy_
        self.epsilon_ = pak.epsilon_
        self.k_hat_ = pak.k_hat_
        self.g_ = pak.get_g()

        # Step 4: Find cluster centers (Algorithm 1)
        centers, self._delta, self._nearest_higher = find_cluster_centers(
            self.log_density_, self.epsilon_,
            self._distances, self._indices, self.k_hat_
        )

        if len(centers) == 0:
            # No centers found - assign all to one cluster
            centers = np.array([np.argmax(self.g_)])

        # Step 5: Initial cluster assignment
        labels = assign_to_clusters(self.g_, self._nearest_higher, centers, n_samples)

        # Step 6: Find saddle points (Algorithm 2)
        saddles, border_points = find_saddle_points(
            labels, self.g_, self.log_density_, self.epsilon_,
            self._distances, self._indices, self.k_hat_
        )

        # Step 7: Merge insignificant peaks (Algorithm 3)
        if len(centers) > 1:
            merged_centers, merge_map, self._merge_history = merge_insignificant_peaks(
                centers, saddles, self.log_density_, self.epsilon_, self.Z
            )

            # Step 8: Re-assign labels after merging
            labels = relabel_after_merge(labels, merge_map)

            # Update saddles for merged clusters
            self.saddles_ = self._update_saddles_after_merge(saddles, merge_map)
            self.cluster_centers_ = merged_centers
        else:
            self.cluster_centers_ = centers
            self.saddles_ = saddles
            self._merge_history = []

        self.labels_full_ = labels.copy()
        self.n_clusters_ = len(self.cluster_centers_)

        # Step 9: Identify halo points
        if self.halo and self.n_clusters_ > 1:
            self.halo_mask_, _ = identify_halo_points(
                labels, self.log_density_, self.saddles_, self.cluster_centers_
            )
            # Apply halo masking
            self.labels_ = labels.copy()
            self.labels_[self.halo_mask_] = -1
        else:
            self.halo_mask_ = np.zeros(n_samples, dtype=bool)
            self.labels_ = labels.copy()

        # Step 10: Build topography
        self.topography_ = Topography(
            self.cluster_centers_, self.saddles_,
            self.log_density_, self.labels_full_
        )

        return self

    def _update_saddles_after_merge(self, saddles, merge_map):
        """Update saddle dictionary after cluster merging."""
        new_saddles = {}
        for pair, info in saddles.items():
            c1, c2 = pair
            # Map to new cluster IDs
            new_c1 = merge_map.get(c1, c1)
            new_c2 = merge_map.get(c2, c2)

            if new_c1 == new_c2:
                continue  # clusters were merged

            new_pair = tuple(sorted([new_c1, new_c2]))
            if new_pair not in new_saddles or \
               info['log_density'] > new_saddles[new_pair]['log_density']:
                new_saddles[new_pair] = info

        return new_saddles

    def fit_predict(self, X, y=None, distances=None, indices=None):
        """
        Fit the model and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency.

        distances : ndarray, optional
            Precomputed distances to neighbors.

        indices : ndarray, optional
            Precomputed neighbor indices.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. -1 for halo points if halo=True.
        """
        self.fit(X, distances=distances, indices=indices)
        return self.labels_

    def get_topography_matrix(self):
        """
        Get the topography matrix.

        Returns
        -------
        matrix : ndarray of shape (n_clusters, n_clusters)
            Symmetric matrix where diagonal = peak heights,
            off-diagonal = saddle heights.
        """
        if self.topography_ is None:
            raise ValueError("Must call fit() first")
        return self.topography_.matrix_

    def plot_topography(self, kind='dendrogram', ax=None, **kwargs):
        """
        Plot topographic visualization.

        Parameters
        ----------
        kind : str, default='dendrogram'
            Type of plot: 'dendrogram', 'network', 'matrix', or 'decision'.

        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        **kwargs : dict
            Additional arguments passed to the plotting function.

        Returns
        -------
        result
            Plot-specific return value.
        """
        if self.topography_ is None:
            raise ValueError("Must call fit() first")

        if kind == 'dendrogram':
            return self.topography_.plot_dendrogram(ax=ax, **kwargs)
        elif kind == 'network':
            return self.topography_.plot_network(ax=ax, **kwargs)
        elif kind == 'matrix':
            return self.topography_.plot_topography_matrix(ax=ax, **kwargs)
        elif kind == 'decision':
            return self.topography_.plot_decision_graph(
                self.g_, self._delta, ax=ax, centers=self.cluster_centers_, **kwargs
            )
        else:
            raise ValueError(f"Unknown plot kind: {kind}. "
                             "Use 'dendrogram', 'network', 'matrix', or 'decision'.")

    def get_cluster_info(self):
        """
        Get detailed information about each cluster.

        Returns
        -------
        list of dict
            Information for each cluster including center, size,
            peak density, and connected clusters.
        """
        if self.labels_full_ is None:
            raise ValueError("Must call fit() first")

        info = []
        for c in range(self.n_clusters_):
            cluster_mask = self.labels_full_ == c
            center_idx = self.cluster_centers_[c]

            # Find connected clusters
            connected = []
            for pair, saddle_info in self.saddles_.items():
                if c in pair:
                    other = pair[0] if pair[1] == c else pair[1]
                    connected.append({
                        'cluster': other,
                        'saddle_density': saddle_info['log_density']
                    })

            info.append({
                'cluster_id': c,
                'center_index': center_idx,
                'size': np.sum(cluster_mask),
                'size_core': np.sum(self.labels_ == c),
                'peak_density': self.log_density_[center_idx],
                'peak_free_energy': self.free_energy_[center_idx],
                'connected_clusters': connected
            })

        return info

    def summary(self):
        """
        Get a summary of the clustering results.

        Returns
        -------
        dict
            Summary statistics.
        """
        if self.labels_ is None:
            raise ValueError("Must call fit() first")

        return {
            'n_clusters': self.n_clusters_,
            'n_samples': len(self.labels_),
            'n_halo_points': np.sum(self.halo_mask_),
            'intrinsic_dimension': self.d_,
            'Z': self.Z,
            'cluster_sizes': {c: np.sum(self.labels_full_ == c)
                              for c in range(self.n_clusters_)},
            'k_hat_stats': {
                'min': np.min(self.k_hat_),
                'max': np.max(self.k_hat_),
                'mean': np.mean(self.k_hat_)
            }
        }

    @staticmethod
    def recommended_Z(n_samples, d, data_type='generic'):
        """
        Get recommended Z value for a dataset.

        The Z parameter controls cluster merging sensitivity. This method
        provides guidelines based on the paper's analysis and practical
        considerations for different data scenarios.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.
        d : float
            Intrinsic dimension of the data.
        data_type : str, default='generic'
            Type of data:
            - 'generic': Standard continuous data
            - 'image': Image data (MNIST-like)
            - 'sparse': Sparse high-dimensional data
            - 'dense_manifold': Well-sampled low-dimensional manifold

        Returns
        -------
        dict
            Dictionary with recommended Z and explanation.

        Notes
        -----
        **Why Z needs tuning for high-dimensional data:**

        The Z-score test compares peak heights with saddle heights:
            |log(ρ_peak) - log(ρ_saddle)| / sqrt(ε_peak² + ε_saddle²) > Z

        In high dimensions, density estimation becomes less reliable due to:

        1. **Curse of dimensionality**: The volume of d-ball grows as r^d.
           For d=10, doubling the radius increases volume by 2^10 = 1024x.
           This makes density estimates highly variable.

        2. **Sparse sampling**: Even with many points, high-d space is sparsely
           populated. k̂ (optimal neighbors) tends to be smaller, leading to
           larger ε (error bounds).

        3. **Error propagation**: The Fisher Information formula shows:
           ε = sqrt(4(k̂+2)/((k̂-1)·k̂))
           For k̂=10: ε ≈ 0.73 (log units)
           For k̂=50: ε ≈ 0.30 (log units)

           When ε is large, the Z-score denominator is large, making it
           harder to reach significance. Use lower Z to compensate.

        **Practical recommendations:**

        - **MNIST (d~10-15)**: Z ≈ 1.6 (as in the paper)
          High embedding dimension (784) but lower intrinsic dimension.
          Still, density errors are substantial.

        - **Well-sampled 2D data**: Z ≈ 2.0-3.0
          k̂ is typically large (50-100), errors are small.

        - **Sparse data (n/d < 100)**: Z ≈ 1.0-1.5
          Very uncertain density estimates require liberal merging.

        - **Biological data (single-cell, etc.)**: Z ≈ 1.5-2.0
          Often moderately sparse with meaningful local structure.

        **Critical insight from the paper:**

        The paper emphasizes that Z is the ONLY free parameter in DPA.
        Unlike DBSCAN (ε, min_samples) or spectral clustering (n_clusters),
        Z has a clear statistical interpretation: it's a confidence level
        for declaring two peaks as distinct.

        When in doubt, try multiple Z values (1.0, 1.5, 2.0, 2.5, 3.0)
        and examine the topography. Stable structures persist across Z.
        """
        # Sampling density: points per unit of intrinsic dimension
        sampling_ratio = n_samples / max(d, 1)

        if data_type == 'image':
            # Image data like MNIST
            z_rec = 1.6
            explanation = (
                "Image data typically has high embedding dimension but lower "
                "intrinsic dimension. Density estimation errors are substantial "
                "due to the curse of dimensionality. Z=1.6 is used in the paper "
                "for MNIST."
            )
        elif data_type == 'sparse' or sampling_ratio < 100:
            # Sparse sampling
            z_rec = 1.0 + 0.3 * np.log10(max(sampling_ratio, 10))
            z_rec = max(1.0, min(z_rec, 2.0))
            explanation = (
                f"With {n_samples} samples in d={d:.1f}, sampling ratio is "
                f"{sampling_ratio:.1f} points per dimension. Sparse sampling "
                "leads to large density errors. Use lower Z for reliable merging."
            )
        elif data_type == 'dense_manifold' or sampling_ratio > 1000:
            # Well-sampled
            z_rec = 2.5
            explanation = (
                f"With sampling ratio {sampling_ratio:.1f}, density estimates "
                "are reliable. Higher Z provides conservative clustering that "
                "preserves fine structure."
            )
        else:
            # Generic
            if d <= 5:
                z_rec = 2.0
            elif d <= 10:
                z_rec = 1.8
            elif d <= 20:
                z_rec = 1.6
            else:
                z_rec = 1.4

            explanation = (
                f"For generic data with d={d:.1f}, Z={z_rec:.1f} provides "
                "a reasonable balance. Consider lowering Z if clusters are "
                "over-merged, or raising Z if too fragmented."
            )

        return {
            'recommended_Z': z_rec,
            'explanation': explanation,
            'sampling_ratio': sampling_ratio,
            'd': d,
            'n_samples': n_samples,
            'guidelines': {
                'more_clusters': f"Use Z < {z_rec:.1f} (e.g., {max(1.0, z_rec-0.5):.1f})",
                'fewer_clusters': f"Use Z > {z_rec:.1f} (e.g., {z_rec+0.5:.1f})",
                'test_range': [1.0, 1.5, 2.0, 2.5, 3.0]
            }
        }

    def analyze_Z_sensitivity(self, X, Z_values=None, distances=None, indices=None):
        """
        Analyze clustering stability across different Z values.

        This is useful for understanding the robustness of the clustering
        and identifying stable structures that persist across Z.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to cluster.
        Z_values : array-like, optional
            Z values to test. Default: [1.0, 1.5, 2.0, 2.5, 3.0]
        distances : ndarray, optional
            Precomputed distances (e.g., Tangent Distance).
        indices : ndarray, optional
            Precomputed neighbor indices.

        Returns
        -------
        dict
            Analysis results including:
            - n_clusters per Z value
            - Stability metrics
            - Recommendations

        Examples
        --------
        >>> dpa = DPA()
        >>> analysis = dpa.analyze_Z_sensitivity(X)
        >>> print(f"Stable n_clusters: {analysis['stable_n_clusters']}")
        """
        if Z_values is None:
            Z_values = [1.0, 1.5, 2.0, 2.5, 3.0]

        X = np.asarray(X)
        results = {'Z_values': Z_values, 'n_clusters': [], 'labels': []}

        # Run clustering for each Z
        for Z in Z_values:
            dpa_temp = DPA(Z=Z, d=self.d, k_max=self.k_max,
                           D_thr=self.D_thr, metric=self.metric, halo=self.halo)
            dpa_temp.fit(X, distances=distances, indices=indices)
            results['n_clusters'].append(dpa_temp.n_clusters_)
            results['labels'].append(dpa_temp.labels_.copy())

        # Find plateau (stable region)
        n_clusters_arr = np.array(results['n_clusters'])
        transitions = np.where(np.diff(n_clusters_arr) != 0)[0]

        # Find the longest plateau
        if len(transitions) == 0:
            # All same n_clusters
            stable_n = n_clusters_arr[0]
            stable_range = (Z_values[0], Z_values[-1])
        else:
            plateau_starts = [0] + list(transitions + 1)
            plateau_ends = list(transitions + 1) + [len(Z_values)]
            plateau_lengths = [e - s for s, e in zip(plateau_starts, plateau_ends)]
            longest_idx = np.argmax(plateau_lengths)
            stable_start = plateau_starts[longest_idx]
            stable_end = plateau_ends[longest_idx] - 1
            stable_n = n_clusters_arr[stable_start]
            stable_range = (Z_values[stable_start], Z_values[stable_end])

        results['stable_n_clusters'] = int(stable_n)
        results['stable_Z_range'] = stable_range
        results['recommended_Z'] = np.mean(stable_range)

        # Add interpretation
        if n_clusters_arr[0] == n_clusters_arr[-1]:
            results['interpretation'] = (
                "Clustering is very stable across Z values. "
                "The cluster structure is robust."
            )
        elif n_clusters_arr[0] - n_clusters_arr[-1] <= 2:
            results['interpretation'] = (
                "Clustering is moderately stable. "
                "Minor peaks merge at higher Z but main structure persists."
            )
        else:
            results['interpretation'] = (
                "Clustering shows significant Z-sensitivity. "
                "Consider the physical/domain interpretation of different "
                "granularity levels."
            )

        return results

    def analyze_Z_sensitivity_fast(self, X, Z_values=None, distances=None, indices=None,
                                    y_true=None, verbose=True):
        """
        OPTIMIZED Z sensitivity analysis - computes k-NN and densities only ONCE.

        This is much faster than analyze_Z_sensitivity() because it:
        1. Computes k-NN graph once
        2. Estimates intrinsic dimension once
        3. Computes PAk densities once
        4. Only re-runs Heuristics 1-3 (clustering) for each Z value

        Speedup: ~5x compared to running fit() for each Z value.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to cluster.
        Z_values : array-like, optional
            Z values to test. Default: [1.0, 1.5, 2.0, 2.5, 3.0]
        distances : ndarray, optional
            Precomputed distances (e.g., Tangent Distance).
        indices : ndarray, optional
            Precomputed neighbor indices.
        y_true : array-like, optional
            Ground truth labels for computing ARI/NMI.
        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        dict
            Analysis results including n_clusters, ARI, NMI per Z value.
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        if Z_values is None:
            Z_values = [1.0, 1.5, 2.0, 2.5, 3.0]

        X = np.asarray(X)
        n_samples = X.shape[0]

        if verbose:
            print(f"  Z sensitivity analysis (optimized) for {n_samples} samples...")
            print(f"  Step 1: Computing k-NN and densities (done once)...")

        # ===== STEP 1: Compute everything that doesn't depend on Z (ONCE) =====

        # k-NN graph
        if distances is not None and indices is not None:
            self._distances = np.asarray(distances)
            self._indices = np.asarray(indices)
        else:
            k_actual = min(self.k_max, n_samples - 1)
            self._distances, self._indices = compute_knn(X, k_actual, self.metric)

        # Intrinsic dimension
        if self.d is not None:
            self.d_ = self.d
        else:
            twonn = TwoNN()
            twonn.fit(X, precomputed_distances=self._distances, precomputed_indices=self._indices)
            self.d_ = twonn.dimension_

        # PAk density estimation (doesn't depend on Z!)
        pak = PAk(d=self.d_, k_max=self.k_max, D_thr=self.D_thr)
        pak.fit(X, distances=self._distances, indices=self._indices)

        log_density = pak.log_density_
        epsilon = pak.epsilon_
        k_hat = pak.k_hat_

        # Find cluster centers (Heuristic 1) - doesn't depend on Z
        centers, delta, nearest_higher = find_cluster_centers(
            log_density, epsilon, self._distances, self._indices, k_hat
        )

        # Compute g for cluster assignment
        g = log_density - epsilon

        # Assign all points to initial clusters (Heuristic 2) - doesn't depend on Z
        initial_labels = assign_to_clusters(g, nearest_higher, centers, n_samples)

        # Find all saddle points - doesn't depend on Z
        all_saddles, _ = find_saddle_points(
            initial_labels, g, log_density, epsilon,
            self._distances, self._indices, k_hat
        )

        if verbose:
            print(f"  Intrinsic dimension: {self.d_:.2f}")
            print(f"  Initial clusters (before merge): {len(centers)}")
            print(f"  Step 2: Testing {len(Z_values)} Z values (only merging step)...")

        # ===== STEP 2: Run only Heuristic 3 (merging) for each Z =====
        results = {
            'Z_values': Z_values,
            'n_clusters': [],
            'n_halo': [],
            'labels': []
        }
        if y_true is not None:
            results['ARI'] = []
            results['NMI'] = []

        for Z in Z_values:
            # Heuristic 3: Merge insignificant peaks (THIS depends on Z!)
            merged_centers, merge_map, _ = merge_insignificant_peaks(
                np.array(centers), dict(all_saddles), log_density, epsilon, Z
            )

            # Relabel points after merge
            labels = relabel_after_merge(initial_labels.copy(), merge_map)

            # Halo points
            if self.halo and len(merged_centers) > 1:
                # Update saddles after merge
                new_saddles = {}
                for pair, info in all_saddles.items():
                    c1, c2 = pair
                    new_c1 = merge_map.get(c1, c1)
                    new_c2 = merge_map.get(c2, c2)
                    if new_c1 != new_c2:
                        new_pair = tuple(sorted([new_c1, new_c2]))
                        if new_pair not in new_saddles or \
                           info['log_density'] > new_saddles[new_pair]['log_density']:
                            new_saddles[new_pair] = info

                halo_mask, _ = identify_halo_points(labels, log_density, new_saddles, merged_centers)
                n_halo = np.sum(halo_mask)
            else:
                n_halo = 0

            n_clusters = len(merged_centers)
            results['n_clusters'].append(n_clusters)
            results['n_halo'].append(n_halo)
            results['labels'].append(labels.copy())

            if y_true is not None:
                ari = adjusted_rand_score(y_true, labels)
                nmi = normalized_mutual_info_score(y_true, labels)
                results['ARI'].append(ari)
                results['NMI'].append(nmi)

            if verbose:
                msg = f"    Z={Z:.1f}: {n_clusters} clusters, {n_halo} halo"
                if y_true is not None:
                    msg += f", ARI={ari:.3f}, NMI={nmi:.3f}"
                print(msg)

        return results
