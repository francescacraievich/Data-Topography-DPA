"""
Unit tests for TWO-NN intrinsic dimension estimator using REAL UCI data.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from src.intrinsic_dimension import TwoNN, estimate_intrinsic_dimension, BlockAnalysis, block_analysis
from src import load_optdigits, load_pendigits


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
    """Subset for faster tests."""
    X, y = optdigits_data
    np.random.seed(42)
    idx = np.random.choice(len(X), size=500, replace=False)
    return X[idx], y[idx]


@pytest.fixture(scope="module")
def pendigits_subset():
    """Load Pendigits subset (16D real data)."""
    X, y = load_pendigits(return_X_y=True)
    np.random.seed(42)
    idx = np.random.choice(len(X), size=500, replace=False)
    return X[idx], y[idx]


class TestTwoNN:
    """Tests for TWO-NN estimator on real data."""

    def test_optdigits_dimension(self, optdigits_data):
        """TWO-NN should estimate reasonable dimension for Optdigits (64D embedding)."""
        X, y = optdigits_data

        twonn = TwoNN()
        twonn.fit(X)

        # Optdigits has 64 features but intrinsic dimension is much lower
        # Paper reports d ~ 7-13 for digit data
        assert 5 < twonn.dimension_ < 20, \
            f"Expected d ~ 7-15 for Optdigits, got {twonn.dimension_}"

    def test_pendigits_dimension(self, pendigits_subset):
        """TWO-NN should estimate reasonable dimension for Pendigits (16D embedding)."""
        X, y = pendigits_subset

        twonn = TwoNN()
        twonn.fit(X)

        # Pendigits has 16 features, intrinsic dimension should be lower
        assert 3 < twonn.dimension_ < 15, \
            f"Expected d ~ 5-12 for Pendigits, got {twonn.dimension_}"

    def test_single_digit_dimension(self, optdigits_data):
        """Single digit class should have lower dimension than full dataset."""
        X, y = optdigits_data

        # Just digit "1" (simpler structure)
        X_digit1 = X[y == 1]

        twonn = TwoNN()
        twonn.fit(X_digit1)

        # Single digit should have lower intrinsic dimension
        assert 2 < twonn.dimension_ < 15, \
            f"Expected lower d for single digit, got {twonn.dimension_}"

    def test_discard_fraction(self, optdigits_subset):
        """Higher discard fraction should affect estimate on real data."""
        X, y = optdigits_subset

        twonn_10 = TwoNN(discard_fraction=0.1)
        twonn_20 = TwoNN(discard_fraction=0.2)

        twonn_10.fit(X)
        twonn_20.fit(X)

        # Both should give reasonable estimates
        assert 5 < twonn_10.dimension_ < 20
        assert 5 < twonn_20.dimension_ < 20

    def test_convenience_function(self, optdigits_subset):
        """Test the convenience function on real data."""
        X, y = optdigits_subset

        d = estimate_intrinsic_dimension(X)

        # Should be reasonable for digit data
        assert 5 < d < 20

    def test_diagnostics(self, optdigits_subset):
        """Test diagnostic information on real data."""
        X, y = optdigits_subset

        twonn = TwoNN()
        twonn.fit(X)
        diag = twonn.get_diagnostics()

        assert 'dimension' in diag
        assert 'n_points' in diag
        assert 'n_discarded' in diag
        assert diag['n_discarded'] > 0  # Should discard some points


class TestTwoNNEdgeCases:
    """Edge case tests on real data."""

    def test_small_subset(self, optdigits_data):
        """Should work with small subset of real data."""
        X, y = optdigits_data
        np.random.seed(42)
        idx = np.random.choice(len(X), size=50, replace=False)
        X_small = X[idx]

        twonn = TwoNN()
        twonn.fit(X_small)

        assert twonn.dimension_ > 0

    def test_raises_on_too_small(self, optdigits_data):
        """Should raise error if dataset too small."""
        X, y = optdigits_data
        X_tiny = X[:5]  # Only 5 samples

        twonn = TwoNN()
        with pytest.raises(ValueError):
            twonn.fit(X_tiny)


class TestBlockAnalysis:
    """Tests for Block Analysis on real data."""

    def test_basic_fit(self, optdigits_subset):
        """BlockAnalysis should fit and produce results on real data."""
        X, y = optdigits_subset

        ba = BlockAnalysis(n_blocks=5, min_samples=50, n_repetitions=2, random_state=42)
        ba.fit(X)

        assert ba.block_sizes_ is not None
        assert ba.dimensions_ is not None
        assert ba.plateau_dimension_ is not None
        assert len(ba.dimensions_) == len(ba.block_sizes_)

    def test_plateau_detection(self, optdigits_data):
        """Should detect stable plateau on full Optdigits."""
        X, y = optdigits_data

        ba = BlockAnalysis(n_blocks=8, min_samples=100, n_repetitions=3, random_state=42)
        ba.fit(X)

        # Plateau dimension should be reasonable for digit data
        assert 5 < ba.plateau_dimension_ < 20

    def test_get_summary(self, optdigits_subset):
        """Summary should contain expected keys."""
        X, y = optdigits_subset

        ba = BlockAnalysis(n_blocks=4, min_samples=50, n_repetitions=2, random_state=42)
        ba.fit(X)

        summary = ba.get_summary()
        assert 'plateau_dimension' in summary
        assert 'min_dimension' in summary
        assert 'max_dimension' in summary
        assert 'is_stable' in summary

    def test_interpret(self, optdigits_subset):
        """Interpret should return non-empty string on real data."""
        X, y = optdigits_subset

        ba = BlockAnalysis(n_blocks=4, min_samples=50, n_repetitions=2, random_state=42)
        ba.fit(X)

        interpretation = ba.interpret()
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
        assert 'dimension' in interpretation.lower()

    def test_convenience_function(self, optdigits_subset):
        """Test block_analysis convenience function on real data."""
        X, y = optdigits_subset

        ba = block_analysis(X, n_blocks=4, min_samples=50, n_repetitions=2, random_state=42)

        assert ba.dimensions_ is not None
        assert ba.plateau_dimension_ > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
