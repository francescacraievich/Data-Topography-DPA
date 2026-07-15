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
varies within the neighborhood of a point. It is fit by a damped,
step-clamped Newton-Raphson search (not an unregularized linear
regression), matching the official reference implementation
(github.com/mariaderrico/DPA, src/Pipeline/_PAk.pyx, function nrmaxl) -
the damping/clamping is what keeps the fit well-behaved even when k_hat
is small, where an unregularized fit is numerically unstable.

The stopping criterion (Likelihood Ratio Test) compares point i's local
volume estimate against a DIFFERENT nearby point's estimate (i's own
(k+1)-th neighbor), not point i's own estimate as k grows - a two-sample
Poisson-rate-equality test, matching _PAk.pyx's ratio_test.
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

    k_min : int, default=3
        Minimum number of neighbors before the Likelihood Ratio Test is
        first evaluated (matches the official reference implementation,
        which hardcodes this to 3).

    D_thr : float, default=23.928
        Likelihood ratio test threshold.
        Default corresponds to p-value 10^-6 (chi-squared with 1 dof).

    metric : str, default='euclidean'
        Distance metric for neighbor computation.

    use_alpha_correction : bool, default=True
        Whether to refine the log-density estimate with the systematic
        correction α for density gradients (log(ρ(r)) = log(ρ_0) + α·r),
        via a damped Newton-Raphson fit. When False, the plain
        1-parameter MLE log(k_hat / V_k_hat) is used.

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
    The Fisher Information matrix gives the error formula:
        Var(log(ρ_0)) = (4*k_hat + 2) / (k_hat * (k_hat - 1))

    Reference: d'Errico et al. (2021), Supplementary Information Text S1;
    github.com/mariaderrico/DPA.
    """

    def __init__(self, d=None, k_max=100, k_min=3, D_thr=23.928, metric='euclidean',
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

        # Shared cache of cumulative neighbor volumes, keyed by point index
        # then neighbor rank: V_dic[idx][k] = volume enclosing idx's k
        # nearest neighbors. Shared across all points because the LRT below
        # compares point i against a DIFFERENT point j, so j's volumes get
        # cached here too and reused once j is itself processed as "i".
        V_dic = {}

        # For each point, find optimal k using the Likelihood Ratio Test
        for i in range(n_samples):
            k_hat, log_rho, eps, alpha = self._estimate_density_point(
                i, distances, indices, self.d_, omega_d, V_dic
            )
            self.k_hat_[i] = k_hat
            self.log_density_[i] = log_rho
            self.epsilon_[i] = eps
            self.alpha_[i] = alpha

        # Free energy (physics terminology)
        self.free_energy_ = -self.log_density_

        return self

    def _estimate_density_point(self, i, distances, indices, d, omega_d, V_dic):
        """
        Estimate density for point i using the two-sample Likelihood Ratio
        Test (matches the official reference implementation's ratio_test):
        at each candidate k, compares point i's own k-th-neighbor volume
        against the k-th-neighbor volume of i's (k+1)-th nearest neighbor
        j - a test of whether i and the nearby point j have statistically
        compatible local densities - rather than testing i's own
        self-consistency as k grows.

        Parameters
        ----------
        i : int
            Point index.
        distances, indices : ndarray of shape (n_samples, k_max+1)
            Full k-NN distance/index matrices (need point j's own row too).
        d : float
            Intrinsic dimension.
        omega_d : float
            Volume of unit d-ball.
        V_dic : dict
            Shared cache {point_idx: {neighbor_rank: cumulative_volume}},
            reused across all points (population points' entries get
            reused once they're processed as "i" themselves).

        Returns
        -------
        k_hat : int
            Optimal number of neighbors.
        log_rho : float
            Log-density estimate at r=0.
        epsilon : float
            Error estimate.
        alpha : float
            Systematic correction parameter (density gradient), diagnostic
            only - not used by the official implementation but retained
            here since our downstream code inspects it.
        """
        k_max_actual = min(distances.shape[1] - 1, self.k_max)

        def get_volume(idx, k):
            cache = V_dic.setdefault(idx, {})
            v = cache.get(k)
            if v is None:
                r = distances[idx][k]
                v = omega_d * (r ** d) if r > 0 else 0.0
                cache[k] = v
            return v

        # Pre-populate shells 1..k_min so the later Newton-Raphson refinement
        # always has an unbroken cumulative-volume sequence to work with,
        # matching the official implementation's eager seeding of V_dic[i].
        for k0 in range(1, self.k_min + 1):
            get_volume(i, k0)

        k = self.k_min
        D_k = 0.0
        while k < k_max_actual and D_k <= self.D_thr:
            vi = get_volume(i, k)
            j = indices[i][k + 1]
            vj = get_volume(j, k)
            if vi > 0 and vj > 0:
                D_k = -2.0 * k * (np.log(vi) + np.log(vj) - 2.0 * np.log(vi + vj) + np.log(4.0))
            k += 1

        k_hat = k - 1
        r_k_hat = distances[i][k_hat]
        V_k_hat = get_volume(i, k_hat)

        # Plain 1-parameter MLE, used as the starting point for the
        # Newton-Raphson refinement below (and as the result if the
        # refinement is disabled or falls back on a degenerate shell).
        rho_hat = k_hat / V_k_hat if V_k_hat > 0 else 1e10
        log_rho_simple = np.log(max(rho_hat, 1e-300))

        if self.use_alpha_correction and k_hat >= 2:
            log_rho_hat, alpha = self._nrmaxl(log_rho_simple, k_hat, V_dic[i])
        else:
            log_rho_hat, alpha = log_rho_simple, 0.0

        # Error estimate from Fisher Information.
        # Var(log(rho_0)) = (4*k_hat + 2) / (k_hat * (k_hat - 1))
        # See Supplementary Information Text S1 of the paper; matches
        # the official reference implementation's err_densities formula.
        if k_hat > 1:
            variance = (4.0 * k_hat + 2.0) / (k_hat * (k_hat - 1))
            epsilon = np.sqrt(variance)
        else:
            epsilon = np.inf  # undefined for k=1

        return k_hat, log_rho_hat, epsilon, alpha

    def _nrmaxl(self, rinit, kopt, V_dic_i):
        """
        Damped, step-clamped 2-parameter Newton-Raphson refinement of the
        log-density estimate: fits log(ρ) = b + a*(shell index) via the
        Poisson-process likelihood, matching the official reference
        implementation's nrmaxl (src/Pipeline/_PAk.pyx).

        The damping (each step is only 10% of the full Newton step) and
        the step-size ceiling (never move more than 10% of the initial
        estimate per iteration) are what keep this well-behaved at small
        k_hat, where an unregularized fit can diverge to extreme values.

        Parameters
        ----------
        rinit : float
            Initial log-density estimate (the plain 1-parameter MLE).
        kopt : int
            k_hat - number of shells to fit.
        V_dic_i : dict
            Cache of point i's cumulative shell volumes, keyed 1..kopt.

        Returns
        -------
        log_rho_0 : float
            Refined log-density at r=0.
        alpha : float
            Fitted gradient parameter (diagnostic only).
        """
        b = rinit
        a = 0.0
        stepmax = 0.1 * abs(b)

        # Shell (incremental) volumes: vi[0] is the volume of the first
        # neighbor's ball; vi[j] for j>=1 is the volume of the shell
        # between the j-th and (j+1)-th neighbors.
        vi = np.empty(kopt)
        vi[0] = V_dic_i[1]
        degenerate = False
        for j in range(1, kopt):
            vi[j] = V_dic_i[j + 1] - V_dic_i[j]
            if vi[j] < 1e-100:
                degenerate = True

        if degenerate:
            # Matches the official kNN fallback: if any shell has ~zero
            # volume (near-duplicate points), skip the correction.
            return b, a

        ga, gb, cov = self._nr_derivatives(a, b, kopt, vi)
        cov_inv = self._nr_inverse(cov)
        if cov_inv is None:
            return b, a

        func = 100.0
        niter = 0
        fepsilon = np.finfo(float).eps
        while func > 1e-3 and niter < 1000:
            sb = cov_inv[0, 0] * gb + cov_inv[0, 1] * ga
            sa = cov_inv[1, 0] * gb + cov_inv[1, 1] * ga
            niter += 1
            sigma = 0.1
            if abs(sigma * sb) > stepmax:
                sigma = abs(stepmax / sb) if sb != 0 else sigma
            b = b - sigma * sb
            a = a - sigma * sa
            ga, gb, cov = self._nr_derivatives(a, b, kopt, vi)
            cov_inv = self._nr_inverse(cov)
            if cov_inv is None:
                return b, a
            if abs(a) <= fepsilon or abs(b) <= fepsilon:
                func = max(abs(gb), abs(ga))
            else:
                func = max(abs(gb / b), abs(ga / a))

        return b, a

    @staticmethod
    def _nr_derivatives(a, b, kopt, vi):
        """Gradient and Fisher-information (Hessian) of the 2-parameter
        Poisson-process log-likelihood, at shell-count kopt."""
        gb = float(kopt)
        ga = (kopt + 1) * kopt / 2.0
        cov = np.zeros((2, 2))
        for j in range(kopt):
            jf = j + 1
            t = b + a * jf
            s = np.exp(t) if t < 700 else np.inf
            tt = vi[j] * s
            gb -= tt
            ga -= jf * tt
            cov[0, 0] -= tt
            cov[0, 1] -= jf * tt
            cov[1, 1] -= jf * jf * tt
        cov[1, 0] = cov[0, 1]
        return ga, gb, cov

    @staticmethod
    def _nr_inverse(cov):
        """Inverse of the 2x2 covariance matrix, or None if singular."""
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
        if det == 0 or not np.isfinite(det):
            return None
        det_inv = 1.0 / det
        cov_inv = np.array([
            [det_inv * cov[1, 1], -det_inv * cov[0, 1]],
            [-det_inv * cov[1, 0], det_inv * cov[0, 0]],
        ])
        return cov_inv

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
