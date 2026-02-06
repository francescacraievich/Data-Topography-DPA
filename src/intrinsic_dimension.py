"""
Intrinsic Dimension Estimation using TWO-NN method.

Based on: Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017).
"Estimating the intrinsic dimension of datasets by a minimal neighborhood information"
Scientific Reports, 7, 12140.

The TWO-NN estimator uses the ratio of distances to the first two nearest neighbors
to estimate the intrinsic dimension of the data manifold.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from .utils import empirical_cdf, linear_regression_through_origin


class TwoNN:
    """
    TWO-NN Intrinsic Dimension Estimator.

    Estimates the intrinsic dimension d of the data manifold using
    the ratio of distances to the first two nearest neighbors.

    The key insight: if data lies uniformly on a d-dimensional manifold,
    the ratio mu = r2/r1 follows a Pareto distribution with shape parameter d.
    This leads to: P(mu > m) = m^(-d), so -log(1 - F(mu)) = d * log(mu).

    Parameters
    ----------
    discard_fraction : float, default=0.1
        Fraction of highest mu values to discard for robustness.
        High mu values often come from boundary points or sparse regions,
        biasing the estimate upward.

    metric : str, default='euclidean'
        Distance metric for neighbor computation.

    Attributes
    ----------
    dimension_ : float
        Estimated intrinsic dimension.

    mu_ : ndarray
        Ratio r2/r1 for each point (before filtering).

    mu_filtered_ : ndarray
        Filtered mu values used for estimation.

    Notes
    -----
    The robust estimation uses linear regression through the origin:
        y = -log(1 - F_emp(mu))  vs  x = log(mu)
    where the slope gives d.

    Discarding the top 10% of mu values protects against:
    - Boundary effects (points near manifold edges)
    - Outliers and sparse regions
    - Finite-sample fluctuations
    """

    def __init__(self, discard_fraction=0.1, metric='euclidean'):
        self.discard_fraction = discard_fraction
        self.metric = metric

        self.dimension_ = None
        self.mu_ = None
        self.mu_filtered_ = None
        self._n_discarded = 0

    def fit(self, X, precomputed_distances=None, precomputed_indices=None):
        """
        Estimate intrinsic dimension from data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points.

        precomputed_distances : ndarray, optional
            Precomputed k-NN distances (at least 3 neighbors including self).

        precomputed_indices : ndarray, optional
            Precomputed k-NN indices.

        Returns
        -------
        self : TwoNN
            Fitted estimator.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        if n_samples < 10:
            raise ValueError(f"Need at least 10 samples, got {n_samples}")

        # Get distances to first two neighbors
        if precomputed_distances is not None and precomputed_indices is not None:
            r1 = precomputed_distances[:, 1]  # distance to 1st neighbor (0 is self)
            r2 = precomputed_distances[:, 2]  # distance to 2nd neighbor
        else:
            nn = NearestNeighbors(n_neighbors=3, metric=self.metric)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            r1 = distances[:, 1]  # distance to 1st neighbor (0 is self)
            r2 = distances[:, 2]  # distance to 2nd neighbor

        # Compute mu = r2 / r1
        # Handle case where r1 = 0 (identical points)
        valid_mask = r1 > 1e-10
        if np.sum(valid_mask) < 10:
            raise ValueError("Too many duplicate points (r1 = 0)")

        mu = np.full(n_samples, np.nan)
        mu[valid_mask] = r2[valid_mask] / r1[valid_mask]

        # Filter out invalid values (mu must be >= 1 by triangle inequality)
        valid_mu = mu[valid_mask & (mu >= 1.0)]

        if len(valid_mu) < 10:
            raise ValueError("Not enough valid mu values for estimation")

        self.mu_ = mu

        # Robust estimation: discard top fraction of mu values
        n_keep = int(len(valid_mu) * (1 - self.discard_fraction))
        n_keep = max(n_keep, 10)  # keep at least 10 points

        # Sort and keep lower values
        sorted_mu = np.sort(valid_mu)
        mu_filtered = sorted_mu[:n_keep]
        self._n_discarded = len(valid_mu) - n_keep

        self.mu_filtered_ = mu_filtered

        # Compute empirical CDF
        # F_emp(mu_i) = (i - 0.5) / n for sorted values
        n = len(mu_filtered)
        cdf = (np.arange(1, n + 1) - 0.5) / n

        # Linear regression through origin: y = d * x
        # where x = log(mu), y = -log(1 - F_emp(mu))
        x = np.log(mu_filtered)
        y = -np.log(1 - cdf)

        # Slope = d = sum(x*y) / sum(x^2)
        d_raw = linear_regression_through_origin(x, y)

        # Round to integer as in official DPA implementation
        # The official twoNN returns int(round(id))
        self.dimension_ = int(round(d_raw))
        self.dimension_raw_ = d_raw  # Keep raw value for diagnostics

        return self

    def fit_transform(self, X, **kwargs):
        """
        Fit and return estimated dimension.

        Parameters
        ----------
        X : ndarray
            Data points.

        Returns
        -------
        float
            Estimated intrinsic dimension.
        """
        self.fit(X, **kwargs)
        return self.dimension_

    def get_diagnostics(self):
        """
        Get diagnostic information for the estimation.

        Returns
        -------
        dict
            Dictionary with:
            - 'dimension': estimated dimension
            - 'n_points': number of points used
            - 'n_discarded': number of points discarded
            - 'mu_min': minimum mu value used
            - 'mu_max': maximum mu value used
            - 'mu_mean': mean mu value used
        """
        if self.mu_filtered_ is None:
            raise ValueError("Must call fit() first")

        return {
            'dimension': self.dimension_,
            'n_points': len(self.mu_filtered_),
            'n_discarded': self._n_discarded,
            'mu_min': np.min(self.mu_filtered_),
            'mu_max': np.max(self.mu_filtered_),
            'mu_mean': np.mean(self.mu_filtered_)
        }


def estimate_intrinsic_dimension(X, discard_fraction=0.1, metric='euclidean'):
    """
    Convenience function to estimate intrinsic dimension.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points.

    discard_fraction : float, default=0.1
        Fraction of highest mu values to discard.

    metric : str, default='euclidean'
        Distance metric.

    Returns
    -------
    float
        Estimated intrinsic dimension.
    """
    estimator = TwoNN(discard_fraction=discard_fraction, metric=metric)
    return estimator.fit_transform(X)


class BlockAnalysis:
    """
    Block Analysis for studying intrinsic dimension across scales.

    This technique analyzes how the TWO-NN dimension estimate varies as a
    function of the number of points N used. By subsampling at different
    scales, we can:

    1. Identify the "soft" or relevant dimensions of the dataset
    2. Detect multi-scale structure (dimension varies with scale)
    3. Assess stability of the dimension estimate
    4. Distinguish noise dimensions from intrinsic structure

    A stable plateau in d(N) indicates the true intrinsic dimension.
    If d increases steadily with N, the manifold may have fractal structure
    or the data may not lie on a well-defined manifold.

    Parameters
    ----------
    n_blocks : int, default=10
        Number of different block sizes to test.

    min_samples : int, default=50
        Minimum number of samples in smallest block.

    n_repetitions : int, default=5
        Number of random subsamples at each block size for error estimation.

    discard_fraction : float, default=0.1
        Fraction of highest mu values to discard in TWO-NN.

    metric : str, default='euclidean'
        Distance metric for neighbor computation.

    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    block_sizes_ : ndarray
        Array of block sizes (N values) tested.

    dimensions_ : ndarray
        Mean dimension estimate at each block size.

    dimensions_std_ : ndarray
        Standard deviation of dimension estimates at each block size.

    plateau_dimension_ : float
        Estimated stable dimension from plateau detection.

    plateau_range_ : tuple
        (start_idx, end_idx) of detected plateau region.

    References
    ----------
    Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017).
    "Estimating the intrinsic dimension of datasets by a minimal
    neighborhood information". Scientific Reports, 7, 12140.

    Notes
    -----
    Block Analysis is particularly useful for:
    - Datasets with noise: small N → high d (noise dominates)
    - Datasets with multi-scale structure: d varies with scale
    - Validating dimension estimates: plateau = reliable estimate

    Example interpretation:
    - d ≈ constant: well-defined manifold dimension
    - d increases then plateaus: noise at small scales
    - d keeps increasing: fractal or ill-defined manifold
    - d decreases: possible overfitting at small scales
    """

    def __init__(self, n_blocks=10, min_samples=50, n_repetitions=5,
                 discard_fraction=0.1, metric='euclidean', random_state=None):
        self.n_blocks = n_blocks
        self.min_samples = min_samples
        self.n_repetitions = n_repetitions
        self.discard_fraction = discard_fraction
        self.metric = metric
        self.random_state = random_state

        # Results
        self.block_sizes_ = None
        self.dimensions_ = None
        self.dimensions_std_ = None
        self.all_dimensions_ = None  # Full matrix for detailed analysis
        self.plateau_dimension_ = None
        self.plateau_range_ = None

    def fit(self, X):
        """
        Perform block analysis on the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points.

        Returns
        -------
        self : BlockAnalysis
            Fitted estimator with results.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        if n_samples < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} samples, got {n_samples}"
            )

        rng = np.random.RandomState(self.random_state)

        # Generate block sizes (logarithmically spaced for better coverage)
        block_sizes = np.unique(np.geomspace(
            self.min_samples, n_samples, self.n_blocks
        ).astype(int))

        # Ensure we include the full dataset
        if block_sizes[-1] != n_samples:
            block_sizes = np.append(block_sizes, n_samples)

        n_sizes = len(block_sizes)

        # Store all dimension estimates
        all_dims = np.zeros((n_sizes, self.n_repetitions))

        twonn = TwoNN(
            discard_fraction=self.discard_fraction,
            metric=self.metric
        )

        for i, size in enumerate(block_sizes):
            for j in range(self.n_repetitions):
                if size >= n_samples:
                    # Use full dataset
                    subset = X
                else:
                    # Random subsample
                    indices = rng.choice(n_samples, size=size, replace=False)
                    subset = X[indices]

                try:
                    twonn.fit(subset)
                    all_dims[i, j] = twonn.dimension_
                except ValueError:
                    # If estimation fails, use NaN
                    all_dims[i, j] = np.nan

        self.block_sizes_ = block_sizes
        self.all_dimensions_ = all_dims
        self.dimensions_ = np.nanmean(all_dims, axis=1)
        self.dimensions_std_ = np.nanstd(all_dims, axis=1)

        # Detect plateau
        self._detect_plateau()

        return self

    def _detect_plateau(self, tolerance=0.3):
        """
        Detect stable plateau in dimension estimates.

        A plateau is a region where dimension estimates are stable
        (variation < tolerance) across multiple consecutive block sizes.

        Parameters
        ----------
        tolerance : float, default=0.3
            Maximum allowed variation (relative to mean) within plateau.
        """
        dims = self.dimensions_
        n = len(dims)

        if n < 3:
            self.plateau_dimension_ = dims[-1]
            self.plateau_range_ = (0, n-1)
            return

        # Find longest stable region
        best_length = 0
        best_start = 0
        best_end = 0

        for start in range(n):
            for end in range(start + 2, n + 1):
                region = dims[start:end]
                mean_d = np.mean(region)

                # Check if variation is within tolerance
                if mean_d > 0:
                    rel_variation = np.std(region) / mean_d
                    if rel_variation < tolerance:
                        length = end - start
                        # Prefer plateaus at larger scales (more reliable)
                        score = length + 0.5 * start
                        if score > best_length + 0.5 * best_start:
                            best_length = length
                            best_start = start
                            best_end = end

        if best_length > 0:
            self.plateau_range_ = (best_start, best_end - 1)
            self.plateau_dimension_ = np.mean(dims[best_start:best_end])
        else:
            # No clear plateau, use large-scale estimate
            self.plateau_range_ = (n-2, n-1)
            self.plateau_dimension_ = np.mean(dims[-2:])

    def get_summary(self):
        """
        Get summary of block analysis results.

        Returns
        -------
        dict
            Dictionary with:
            - 'plateau_dimension': estimated stable dimension
            - 'plateau_range': block indices defining plateau
            - 'min_dimension': minimum dimension across scales
            - 'max_dimension': maximum dimension across scales
            - 'full_data_dimension': dimension from full dataset
            - 'is_stable': whether a clear plateau was found
        """
        if self.dimensions_ is None:
            raise ValueError("Must call fit() first")

        return {
            'plateau_dimension': self.plateau_dimension_,
            'plateau_range': self.plateau_range_,
            'plateau_block_sizes': (
                self.block_sizes_[self.plateau_range_[0]],
                self.block_sizes_[self.plateau_range_[1]]
            ),
            'min_dimension': np.min(self.dimensions_),
            'max_dimension': np.max(self.dimensions_),
            'full_data_dimension': self.dimensions_[-1],
            'is_stable': (self.plateau_range_[1] - self.plateau_range_[0]) >= 2,
            'dimension_range': np.max(self.dimensions_) - np.min(self.dimensions_)
        }

    def plot(self, ax=None, show_plateau=True, show_error=True):
        """
        Plot dimension vs block size with plateau highlighting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        show_plateau : bool, default=True
            Highlight detected plateau region.

        show_error : bool, default=True
            Show error bars (std across repetitions).

        Returns
        -------
        ax : matplotlib.axes.Axes
            The plot axes.
        """
        import matplotlib.pyplot as plt

        if self.dimensions_ is None:
            raise ValueError("Must call fit() first")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot dimension vs block size
        if show_error and self.dimensions_std_ is not None:
            ax.errorbar(
                self.block_sizes_, self.dimensions_,
                yerr=self.dimensions_std_,
                fmt='o-', capsize=3, capthick=1,
                label='Dimension estimate', color='blue', alpha=0.8
            )
        else:
            ax.plot(
                self.block_sizes_, self.dimensions_,
                'o-', label='Dimension estimate', color='blue'
            )

        # Highlight plateau
        if show_plateau and self.plateau_range_ is not None:
            start, end = self.plateau_range_
            ax.axhspan(
                self.plateau_dimension_ - 0.1,
                self.plateau_dimension_ + 0.1,
                alpha=0.2, color='green',
                label=f'Plateau: d = {self.plateau_dimension_:.2f}'
            )
            ax.axhline(
                self.plateau_dimension_,
                color='green', linestyle='--', alpha=0.5
            )

        ax.set_xscale('log')
        ax.set_xlabel('Number of points (N)', fontsize=12)
        ax.set_ylabel('Estimated dimension (d)', fontsize=12)
        ax.set_title('Block Analysis: Dimension vs Scale', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return ax

    def interpret(self):
        """
        Provide interpretation of block analysis results.

        Returns
        -------
        str
            Human-readable interpretation of the results.
        """
        if self.dimensions_ is None:
            raise ValueError("Must call fit() first")

        summary = self.get_summary()

        lines = [
            "=" * 50,
            "BLOCK ANALYSIS INTERPRETATION",
            "=" * 50,
            "",
            f"Detected plateau dimension: d = {summary['plateau_dimension']:.2f}",
            f"Dimension range: {summary['min_dimension']:.2f} - {summary['max_dimension']:.2f}",
            f"Full dataset dimension: {summary['full_data_dimension']:.2f}",
            ""
        ]

        # Stability assessment
        if summary['is_stable']:
            lines.append("✓ STABLE PLATEAU DETECTED")
            lines.append(f"  Plateau spans N = {summary['plateau_block_sizes'][0]} to "
                        f"N = {summary['plateau_block_sizes'][1]}")
            lines.append("  → Dimension estimate is reliable")
        else:
            lines.append("WARNING: NO CLEAR PLATEAU")
            lines.append("  → Dimension estimate may be unreliable")

        lines.append("")

        # Pattern interpretation
        dim_range = summary['dimension_range']
        if dim_range < 0.5:
            lines.append("Pattern: CONSTANT dimension across scales")
            lines.append("  → Data lies on a well-defined manifold")
        elif self.dimensions_[0] < self.dimensions_[-1]:
            if dim_range > 2:
                lines.append("Pattern: STRONGLY INCREASING dimension")
                lines.append("  → Possible fractal structure or")
                lines.append("    noise dominating at small scales")
            else:
                lines.append("Pattern: MODERATELY INCREASING dimension")
                lines.append("  → Some noise at small scales, use large-scale estimate")
        else:
            lines.append("Pattern: DECREASING dimension")
            lines.append("  → Possible overfitting at small scales or")
            lines.append("    multi-scale structure in data")

        lines.extend(["", "=" * 50])

        return "\n".join(lines)


def block_analysis(X, n_blocks=10, min_samples=50, n_repetitions=5,
                   discard_fraction=0.1, metric='euclidean', random_state=None):
    """
    Convenience function for block analysis.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points.

    n_blocks : int, default=10
        Number of different block sizes to test.

    min_samples : int, default=50
        Minimum number of samples in smallest block.

    n_repetitions : int, default=5
        Number of random subsamples at each block size.

    discard_fraction : float, default=0.1
        Fraction of highest mu values to discard in TWO-NN.

    metric : str, default='euclidean'
        Distance metric.

    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    BlockAnalysis
        Fitted block analysis with results.

    Examples
    --------
    >>> from src import load_optdigits
    >>> from src.intrinsic_dimension import block_analysis
    >>>
    >>> # Real Optdigits data (64D)
    >>> data = load_optdigits()
    >>> X = data['data']
    >>>
    >>> ba = block_analysis(X, random_state=42)
    >>> print(f"Plateau dimension: {ba.plateau_dimension_:.2f}")
    >>> print(ba.interpret())
    >>> ba.plot()
    """
    ba = BlockAnalysis(
        n_blocks=n_blocks,
        min_samples=min_samples,
        n_repetitions=n_repetitions,
        discard_fraction=discard_fraction,
        metric=metric,
        random_state=random_state
    )
    ba.fit(X)
    return ba
