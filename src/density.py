"""
PAk (Point Adaptive k-NN) Density Estimator with Systematic Correction.

Based on: Rodriguez, A., d'Errico, M., Facco, E., & Laio, A. (2018).
"Computing the free energy without collective variables"
Journal of Chemical Theory and Computation, 14(3), 1206-1215.

The PAk estimator finds the optimal number of neighbors k for each point
using a Likelihood Ratio Test, and provides error estimates via Fisher Information.

The systematic correction α accounts for linear density gradients:
    log(ρ(r)) = log(ρ_0) + α·r

This 2-parameter model provides more accurate estimates when density
varies within the neighborhood of a point.
"""

import numpy as np
from scipy.special import gamma
from scipy.optimize import minimize_scalar
from sklearn.neighbors import NearestNeighbors
from .utils import unit_ball_volume, compute_knn
from .intrinsic_dimension import TwoNN


class PAk:
    """
    Point Adaptive k-NN (PAk) Density Estimator with Systematic Correction.

    For each point, finds the optimal number of neighbors k_hat using
    a Likelihood Ratio Test. The density is estimated in the intrinsic
    dimension d of the manifold, not the embedding dimension.

    Parameters
    ----------
    d : float or None, default=None
        Intrinsic dimension. If None, estimated using TWO-NN.

    k_max : int, default=100
        Maximum number of neighbors to consider.

    k_min : int, default=4
        Minimum number of neighbors (for numerical stability).

    D_thr : float, default=23.928
        Likelihood ratio test threshold.
        Default corresponds to p-value 10^-6 (chi-squared with 1 dof).

    metric : str, default='euclidean'
        Distance metric for neighbor computation.

    use_alpha_correction : bool, default=True
        Whether to use the systematic correction α for density gradients.
        When True, estimates log(ρ(r)) = log(ρ_0) + α·r, providing more
        accurate estimates in regions with varying density.

    Attributes
    ----------
    log_density_ : ndarray of shape (n_samples,)
        Log-density estimates log(rho_i) at r=0 (center of neighborhood).

    free_energy_ : ndarray of shape (n_samples,)
        Free energy F_i = -log(rho_i). (Physics terminology from paper)

    epsilon_ : ndarray of shape (n_samples,)
        Density error estimates from Fisher Information.

    k_hat_ : ndarray of shape (n_samples,)
        Optimal k for each point.

    alpha_ : ndarray of shape (n_samples,)
        Systematic correction parameter for each point.
        Represents the local density gradient: d(log ρ)/dr.

    d_ : float
        Intrinsic dimension used.

    Notes
    -----
    The log-likelihood function for point i with k neighbors (1-parameter model):
        L_{i,k}(ρ) = k * log(ρ) - ρ * V_{i,k}

    where V_{i,k} is the total volume enclosed by k neighbors.

    With the systematic correction (2-parameter model):
        log(ρ(r)) = log(ρ_0) + α·r

    This accounts for linear density gradients within the neighborhood.
    The 2-parameter Fisher Information matrix gives the error formula:
        Var(log(ρ_0)) = 4*(k_hat + 2) / ((k_hat - 1) * k_hat)

    Reference: d'Errico et al. (2021), Supplementary Information Text S1.
    """

    def __init__(self, d=None, k_max=100, k_min=4, D_thr=23.928, metric='euclidean',
                 use_alpha_correction=True):
        self.d = d
        self.k_max = k_max
        self.k_min = k_min
        self.D_thr = D_thr
        self.metric = metric
        self.use_alpha_correction = use_alpha_correction

        # Fitted attributes
        self.log_density_ = None
        self.free_energy_ = None
        self.epsilon_ = None
        self.k_hat_ = None
        self.alpha_ = None  # Systematic correction parameter
        self.d_ = None
        self.distances_ = None
        self.indices_ = None

    def fit(self, X, distances=None, indices=None):
        """
        Estimate density for all points using PAk.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points.

        distances : ndarray, optional
            Precomputed k-NN distances.

        indices : ndarray, optional
            Precomputed k-NN indices.

        Returns
        -------
        self : PAk
            Fitted estimator.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        # Compute k-NN if not provided
        k_actual = min(self.k_max, n_samples - 1)
        if distances is None or indices is None:
            distances, indices = compute_knn(X, k_actual, self.metric)

        self.distances_ = distances
        self.indices_ = indices

        # Estimate or use provided intrinsic dimension
        if self.d is None:
            twonn = TwoNN()
            twonn.fit(X, precomputed_distances=distances, precomputed_indices=indices)
            self.d_ = twonn.dimension_
        else:
            self.d_ = self.d

        # Compute volume of unit ball in d dimensions
        omega_d = unit_ball_volume(self.d_)

        # Initialize arrays
        n_samples = len(X)
        self.log_density_ = np.zeros(n_samples)
        self.k_hat_ = np.zeros(n_samples, dtype=int)
        self.epsilon_ = np.zeros(n_samples)
        self.alpha_ = np.zeros(n_samples)  # Systematic correction

        # For each point, find optimal k using Likelihood Ratio Test
        for i in range(n_samples):
            k_hat, log_rho, eps, alpha = self._estimate_density_point(
                i, distances[i], self.d_, omega_d
            )
            self.k_hat_[i] = k_hat
            self.log_density_[i] = log_rho
            self.epsilon_[i] = eps
            self.alpha_[i] = alpha

        # Free energy (physics terminology)
        self.free_energy_ = -self.log_density_

        return self

    def _estimate_density_point(self, i, dist_i, d, omega_d):
        """
        Estimate density for a single point using Likelihood Ratio Test.

        Optionally applies the systematic correction α to account for
        density gradients: log(ρ(r)) = log(ρ_0) + α·r

        Parameters
        ----------
        i : int
            Point index.
        dist_i : ndarray
            Distances to neighbors for point i.
        d : float
            Intrinsic dimension.
        omega_d : float
            Volume of unit d-ball.

        Returns
        -------
        k_hat : int
            Optimal number of neighbors.
        log_rho : float
            Log-density estimate at r=0.
        epsilon : float
            Error estimate.
        alpha : float
            Systematic correction parameter (density gradient).
        """
        k_max_actual = len(dist_i) - 1  # -1 because index 0 is self
        k_max_actual = min(k_max_actual, self.k_max)

        # Start with minimum k
        k = self.k_min

        # Previous volume for computing shell volumes
        r_prev = 0.0

        # Cumulative volume
        V_cumulative = 0.0

        # Iterate until LRT threshold exceeded or k_max reached
        while k <= k_max_actual:
            # Distance to k-th neighbor (index k because 0 is self)
            r_k = dist_i[k]

            # Avoid zero distances
            if r_k < 1e-10:
                k += 1
                continue

            # Volume enclosed by k neighbors
            V_k = omega_d * (r_k ** d)

            # MLE density at k
            rho_k = k / V_k if V_k > 0 else 1e10
            log_rho_k = np.log(max(rho_k, 1e-300))

            # If at k_max, stop
            if k >= k_max_actual:
                break

            # Distance to (k+1)-th neighbor
            r_k1 = dist_i[k + 1]
            if r_k1 < 1e-10:
                k += 1
                continue

            # Volume enclosed by k+1 neighbors
            V_k1 = omega_d * (r_k1 ** d)

            # MLE density at k+1
            rho_k1 = (k + 1) / V_k1 if V_k1 > 0 else 1e10
            log_rho_k1 = np.log(max(rho_k1, 1e-300))

            # Likelihood Ratio Test (Eq. 2 in paper)
            # Compare: H0 (same density) vs H1 (different density)
            # D_k = 2 * (L_{k+1}(rho_{k+1}) - L_{k+1}(rho_k))

            # Log-likelihood at k+1 neighbors with density rho_{k+1}
            L_k1_rho_k1 = (k + 1) * log_rho_k1 - rho_k1 * V_k1

            # Log-likelihood at k+1 neighbors with density rho_k
            L_k1_rho_k = (k + 1) * log_rho_k - rho_k * V_k1

            D_k = 2 * (L_k1_rho_k1 - L_k1_rho_k)

            # If D_k exceeds threshold, density is no longer constant
            if D_k >= self.D_thr:
                break

            k += 1

        # Use k as optimal k_hat
        k_hat = k

        # Final density estimate at r=0 (center) with optional α correction
        r_k_hat = dist_i[k_hat]
        V_k_hat = omega_d * (r_k_hat ** d)

        if self.use_alpha_correction and k_hat >= self.k_min:
            # 2-parameter model: log(ρ(r)) = log(ρ_0) + α·r
            # Estimate α using the trend in local density estimates
            log_rho_hat, alpha = self._estimate_with_alpha_correction(
                dist_i[:k_hat+1], d, omega_d
            )
        else:
            # 1-parameter model: constant density
            rho_hat = k_hat / V_k_hat if V_k_hat > 0 else 1e10
            log_rho_hat = np.log(max(rho_hat, 1e-300))
            alpha = 0.0

        # Error estimate from Fisher Information
        # Var(log(rho_0)) = 4*(k_hat + 2) / ((k_hat - 1) * k_hat)
        # This is from the 2-parameter Fisher Information matrix inverse
        # See Supplementary Information Text S1 of the paper
        if k_hat > 1:
            variance = 4.0 * (k_hat + 2) / ((k_hat - 1) * k_hat)
            epsilon = np.sqrt(variance)
        else:
            epsilon = np.inf  # undefined for k=1

        return k_hat, log_rho_hat, epsilon, alpha

    def _estimate_with_alpha_correction(self, distances, d, omega_d):
        """
        Estimate log(ρ_0) and α using the 2-parameter model.

        The model is: log(ρ(r)) = log(ρ_0) + α·r

        This is solved by fitting the observed cumulative counts
        to the expected counts under the 2-parameter model.

        Parameters
        ----------
        distances : ndarray
            Distances to k neighbors (including self at index 0).
        d : float
            Intrinsic dimension.
        omega_d : float
            Volume of unit d-ball.

        Returns
        -------
        log_rho_0 : float
            Log-density at r=0.
        alpha : float
            Density gradient parameter.

        Notes
        -----
        For the 2-parameter model, we use a weighted regression approach.
        The local density at each shell is estimated and then fitted to
        the linear model log(ρ(r)) = log(ρ_0) + α·r.

        This is an approximation that works well when the density variation
        is small within the neighborhood. For strongly varying densities,
        more sophisticated methods could be used.
        """
        # Get distances (skip self at index 0)
        r = distances[1:]  # distances to neighbors 1, 2, ..., k
        k = len(r)

        if k < 2:
            # Cannot fit 2 parameters with less than 2 points
            V = omega_d * (r[-1] ** d) if len(r) > 0 else 1.0
            rho = k / V if V > 0 else 1e10
            return np.log(max(rho, 1e-300)), 0.0

        # Estimate local density at each neighbor shell
        # ρ_j ≈ j / V_j where V_j = ω_d · r_j^d
        log_rho_local = np.zeros(k)
        r_mid = np.zeros(k)

        for j in range(k):
            r_j = r[j]
            if r_j < 1e-10:
                log_rho_local[j] = np.nan
                r_mid[j] = 0
                continue

            V_j = omega_d * (r_j ** d)
            rho_j = (j + 1) / V_j if V_j > 0 else 1e10
            log_rho_local[j] = np.log(max(rho_j, 1e-300))
            r_mid[j] = r_j

        # Remove invalid entries
        valid = ~np.isnan(log_rho_local) & (r_mid > 0)
        if np.sum(valid) < 2:
            # Fall back to 1-parameter estimate
            V = omega_d * (r[-1] ** d)
            rho = k / V if V > 0 else 1e10
            return np.log(max(rho, 1e-300)), 0.0

        r_valid = r_mid[valid]
        log_rho_valid = log_rho_local[valid]

        # Weighted linear regression: log(ρ) = log(ρ_0) + α·r
        # Weight by 1/variance, which scales roughly as j for each shell
        weights = np.arange(1, len(r_valid) + 1)
        weights = weights / np.sum(weights)

        # Weighted least squares
        r_mean = np.average(r_valid, weights=weights)
        log_rho_mean = np.average(log_rho_valid, weights=weights)

        # Covariance for slope
        cov_r_rho = np.sum(weights * (r_valid - r_mean) * (log_rho_valid - log_rho_mean))
        var_r = np.sum(weights * (r_valid - r_mean) ** 2)

        if var_r > 1e-10:
            alpha = cov_r_rho / var_r
            log_rho_0 = log_rho_mean - alpha * r_mean
        else:
            alpha = 0.0
            log_rho_0 = log_rho_mean

        return log_rho_0, alpha

    def get_g(self):
        """
        Get error-adjusted log-density g_i = log(rho_i) - epsilon_i.

        This is the quantity used for cluster center detection.

        Returns
        -------
        g : ndarray of shape (n_samples,)
            Error-adjusted log-density.
        """
        if self.log_density_ is None:
            raise ValueError("Must call fit() first")
        return self.log_density_ - self.epsilon_

    def get_statistics(self):
        """
        Get summary statistics of the density estimation.

        Returns
        -------
        dict
            Dictionary with statistics.
        """
        if self.k_hat_ is None:
            raise ValueError("Must call fit() first")

        stats = {
            'd': self.d_,
            'k_hat_min': np.min(self.k_hat_),
            'k_hat_max': np.max(self.k_hat_),
            'k_hat_mean': np.mean(self.k_hat_),
            'k_hat_median': np.median(self.k_hat_),
            'log_density_min': np.min(self.log_density_),
            'log_density_max': np.max(self.log_density_),
            'epsilon_mean': np.mean(self.epsilon_),
        }

        # Add alpha statistics if using systematic correction
        if self.alpha_ is not None:
            stats['alpha_mean'] = np.mean(self.alpha_)
            stats['alpha_std'] = np.std(self.alpha_)
            stats['alpha_min'] = np.min(self.alpha_)
            stats['alpha_max'] = np.max(self.alpha_)
            # Fraction of points with significant negative alpha
            # (indicating density increase towards the point)
            stats['alpha_negative_fraction'] = np.mean(self.alpha_ < 0)

        return stats


def estimate_density_pak(X, d=None, k_max=100, metric='euclidean'):
    """
    Convenience function to estimate density using PAk.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points.

    d : float or None
        Intrinsic dimension (None = auto-estimate).

    k_max : int
        Maximum number of neighbors.

    metric : str
        Distance metric.

    Returns
    -------
    log_density : ndarray
        Log-density for each point.

    epsilon : ndarray
        Error for each point.

    k_hat : ndarray
        Optimal k for each point.
    """
    pak = PAk(d=d, k_max=k_max, metric=metric)
    pak.fit(X)
    return pak.log_density_, pak.epsilon_, pak.k_hat_
