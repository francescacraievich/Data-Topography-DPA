"""
Unit tests for PAk density estimator using REAL UCI data (Optdigits).
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from src.density import PAk, estimate_density_pak
from src import load_optdigits


# Load real data once for all tests
@pytest.fixture(scope="module")
def optdigits_data():
    """Load Optdigits dataset (real UCI data)."""
    data = load_optdigits()
    X = data['data']
    y = data['target']
    return X, y


@pytest.fixture(scope="module")
def optdigits_subset(optdigits_data):
    """Small subset for fast tests."""
    X, y = optdigits_data
    np.random.seed(42)
    idx = np.random.choice(len(X), size=200, replace=False)
    return X[idx], y[idx]


@pytest.fixture(scope="module")
def digit_regions(optdigits_data):
    """Get two different digit classes for density comparison."""
    X, y = optdigits_data
    # Digit 1 (simpler strokes) vs digit 8 (more complex)
    mask_1 = y == 1
    mask_8 = y == 8
    return X[mask_1][:50], X[mask_8][:50]


class TestPAk:
    """Tests for PAk density estimator on real data."""

    def test_high_density_region(self, digit_regions):
        """Test density estimation on different digit classes."""
        X_digit1, X_digit8 = digit_regions
        X = np.vstack([X_digit1, X_digit8])

        pak = PAk(d=10.0, k_max=50)  # Approximate d for Optdigits
        pak.fit(X)

        # Both regions should have computed densities (finite values)
        density_1 = pak.log_density_[:len(X_digit1)].mean()
        density_8 = pak.log_density_[len(X_digit1):].mean()

        assert np.isfinite(density_1), "Digit 1 density should be finite"
        assert np.isfinite(density_8), "Digit 8 density should be finite"

    def test_k_hat_adapts(self, optdigits_subset):
        """k_hat should be computed for each point on real data."""
        X, y = optdigits_subset

        pak = PAk(d=10.0, k_max=50)
        pak.fit(X)

        # k_hat should be computed
        assert pak.k_hat_ is not None
        assert len(pak.k_hat_) == len(X)
        # k_hat should be within valid range
        assert np.all(pak.k_hat_ >= 3), "k_hat should be >= k_min=3"
        assert np.all(pak.k_hat_ <= 50), "k_hat should be <= k_max"

    def test_epsilon_positive(self, optdigits_subset):
        """Error estimates should be positive on real data."""
        X, y = optdigits_subset

        pak = PAk(d=10.0, k_max=50)
        pak.fit(X)

        assert np.all(pak.epsilon_ > 0), "Epsilon should be positive"

    def test_free_energy_inverse_of_log_density(self, optdigits_subset):
        """Free energy should be negative of log density."""
        X, y = optdigits_subset

        pak = PAk(d=10.0, k_max=30)
        pak.fit(X)

        np.testing.assert_array_almost_equal(
            pak.free_energy_, -pak.log_density_,
            err_msg="Free energy should equal -log_density"
        )

    def test_g_computation(self, optdigits_subset):
        """Test error-adjusted log-density g = log(rho) - epsilon on real data."""
        X, y = optdigits_subset

        pak = PAk(d=10.0, k_max=30)
        pak.fit(X)

        g = pak.get_g()
        expected_g = pak.log_density_ - pak.epsilon_

        np.testing.assert_array_almost_equal(g, expected_g)

    def test_auto_dimension_estimation(self, optdigits_subset):
        """PAk should estimate dimension if not provided on real data."""
        X, y = optdigits_subset

        pak = PAk(d=None, k_max=50)  # d=None triggers TWO-NN
        pak.fit(X)

        assert pak.d_ is not None
        # Optdigits intrinsic dimension is roughly 7-15
        assert 3.0 < pak.d_ < 25.0, f"Estimated d={pak.d_} should be reasonable for digits"

    def test_convenience_function(self, optdigits_subset):
        """Test the convenience function on real data."""
        X, y = optdigits_subset

        log_density, epsilon, k_hat = estimate_density_pak(X, d=10.0, k_max=30)

        assert len(log_density) == len(X)
        assert len(epsilon) == len(X)
        assert len(k_hat) == len(X)

    def test_statistics(self, optdigits_subset):
        """Test statistics output on real data."""
        X, y = optdigits_subset

        pak = PAk(d=10.0, k_max=30)
        pak.fit(X)
        stats = pak.get_statistics()

        assert 'd' in stats
        assert 'k_hat_min' in stats
        assert 'k_hat_max' in stats
        assert 'log_density_min' in stats


class TestPAkFisherInformation:
    """Tests for Fisher Information error estimation."""

    def test_epsilon_formula(self, optdigits_subset):
        """Verify epsilon follows Fisher Information formula on real data."""
        X, y = optdigits_subset

        pak = PAk(d=10.0, k_max=30)
        pak.fit(X)

        # Manually compute expected epsilon using Fisher Information formula
        # (matches the official reference implementation's err_densities:
        # sqrt((4*k_hat+2) / (k_hat*(k_hat-1))))
        for i in range(len(X)):
            k = pak.k_hat_[i]
            if k > 1:
                expected_eps = np.sqrt((4.0 * k + 2.0) / (k * (k - 1)))
                np.testing.assert_almost_equal(
                    pak.epsilon_[i], expected_eps, decimal=5,
                    err_msg=f"Epsilon mismatch at point {i}"
                )

    def test_larger_k_smaller_epsilon(self):
        """Larger k should generally lead to smaller epsilon (formula test)."""
        # epsilon = sqrt((4*k+2) / (k*(k-1)))
        # As k increases, epsilon decreases
        # This is a pure formula test, no data needed

        ks = [10, 20, 50, 100]
        epsilons = [np.sqrt((4.0 * k + 2.0) / (k * (k - 1))) for k in ks]

        for i in range(len(ks) - 1):
            assert epsilons[i] > epsilons[i+1], \
                f"Epsilon should decrease with k: eps({ks[i]})={epsilons[i]:.3f} > eps({ks[i+1]})={epsilons[i+1]:.3f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
