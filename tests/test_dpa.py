"""
Integration tests for DPA clustering using REAL UCI data.
"""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '..')

from src import DPA, load_optdigits


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
    idx = np.random.choice(len(X), size=300, replace=False)
    return X[idx], y[idx]


@pytest.fixture(scope="module")
def optdigits_binary(optdigits_data):
    """Binary subset (digits 0 and 1) for simpler tests."""
    X, y = optdigits_data
    mask = (y == 0) | (y == 1)
    return X[mask], y[mask]


class TestDPABasic:
    """Basic DPA tests on real Optdigits data."""

    def test_fit_predict(self, optdigits_subset):
        """Test basic fit_predict workflow on real data."""
        X, y = optdigits_subset

        dpa = DPA(Z=2.0)
        labels = dpa.fit_predict(X)

        assert len(labels) == len(X)
        assert dpa.n_clusters_ >= 1

    def test_finds_clusters_on_real_data(self, optdigits_subset):
        """DPA should find multiple clusters on real digit data."""
        X, y = optdigits_subset

        dpa = DPA(Z=1.5, halo=False)
        labels = dpa.fit_predict(X)

        # Should find multiple clusters (real data has structure)
        assert dpa.n_clusters_ >= 2, f"Should find clusters, got {dpa.n_clusters_}"

        # ARI should be positive (better than random)
        ari = adjusted_rand_score(y, labels)
        assert ari > 0.1, f"ARI={ari:.2f} should be > 0.1 for real digit data"

    def test_attributes_after_fit(self, optdigits_subset):
        """All attributes should be set after fit on real data."""
        X, y = optdigits_subset

        dpa = DPA(Z=2.0)
        dpa.fit(X)

        # Check all attributes
        assert dpa.labels_ is not None
        assert dpa.labels_full_ is not None
        assert dpa.cluster_centers_ is not None
        assert dpa.log_density_ is not None
        assert dpa.free_energy_ is not None
        assert dpa.epsilon_ is not None
        assert dpa.g_ is not None
        assert dpa.k_hat_ is not None
        assert dpa.d_ is not None
        assert dpa.topography_ is not None
        assert dpa.n_clusters_ is not None

    def test_halo_points(self, optdigits_subset):
        """Halo points should be identified correctly on real data."""
        X, y = optdigits_subset

        dpa = DPA(Z=2.0, halo=True)
        dpa.fit(X)

        # Check that halo_mask_ was computed
        assert dpa.halo_mask_ is not None
        assert len(dpa.halo_mask_) == len(X)

        # If halo points exist, they should be marked as -1 in labels_
        if np.sum(dpa.halo_mask_) > 0:
            halo_labels = dpa.labels_[dpa.halo_mask_]
            assert np.all(halo_labels == -1), "Halo points should have label -1"

        assert len(dpa.labels_full_) == len(X)

    def test_no_halo_option(self, optdigits_subset):
        """With halo=False, all points should be assigned."""
        X, y = optdigits_subset

        dpa = DPA(Z=2.0, halo=False)
        dpa.fit(X)

        assert np.all(dpa.labels_ >= 0), "All points should be assigned with halo=False"


class TestDPAZParameter:
    """Tests for Z parameter effect on real data."""

    def test_higher_z_fewer_clusters(self, optdigits_subset):
        """Higher Z should generally lead to fewer clusters (more merging)."""
        X, y = optdigits_subset

        dpa_low = DPA(Z=1.0, halo=False)
        dpa_high = DPA(Z=3.0, halo=False)

        dpa_low.fit(X)
        dpa_high.fit(X)

        # Higher Z should merge more, resulting in fewer or equal clusters
        assert dpa_high.n_clusters_ <= dpa_low.n_clusters_, \
            f"Z=3.0 ({dpa_high.n_clusters_} clusters) should have <= clusters than Z=1.0 ({dpa_low.n_clusters_})"

    def test_z_statistical_interpretation(self, optdigits_binary):
        """Z has statistical meaning - verify on simpler binary classification."""
        X, y = optdigits_binary

        # With binary digits (0 vs 1), should find at least 2 distinct clusters
        dpa = DPA(Z=2.0, halo=False)
        dpa.fit(X)

        assert dpa.n_clusters_ >= 2, "Should find at least 2 clusters for two distinct digits"


class TestDPATopography:
    """Tests for topography features on real data."""

    def test_topography_matrix_shape(self, optdigits_subset):
        """Topography matrix should be n_clusters x n_clusters."""
        X, y = optdigits_subset

        dpa = DPA(Z=2.0)
        dpa.fit(X)

        matrix = dpa.get_topography_matrix()

        assert matrix.shape == (dpa.n_clusters_, dpa.n_clusters_)

    def test_topography_matrix_symmetric(self, optdigits_subset):
        """Topography matrix should be symmetric."""
        X, y = optdigits_subset

        dpa = DPA(Z=2.0)
        dpa.fit(X)

        matrix = dpa.get_topography_matrix()

        # Check symmetry (ignoring -inf)
        for i in range(dpa.n_clusters_):
            for j in range(dpa.n_clusters_):
                if np.isfinite(matrix[i, j]) and np.isfinite(matrix[j, i]):
                    assert matrix[i, j] == matrix[j, i], \
                        f"Matrix not symmetric at ({i},{j})"

    def test_topography_diagonal_is_peaks(self, optdigits_subset):
        """Diagonal should contain peak (center) densities."""
        X, y = optdigits_subset

        dpa = DPA(Z=2.0)
        dpa.fit(X)

        matrix = dpa.get_topography_matrix()

        for c in range(dpa.n_clusters_):
            center_idx = dpa.cluster_centers_[c]
            expected_density = dpa.log_density_[center_idx]
            assert matrix[c, c] == expected_density

    def test_summary(self, optdigits_subset):
        """Test summary method on real data."""
        X, y = optdigits_subset

        dpa = DPA(Z=2.0)
        dpa.fit(X)

        summary = dpa.summary()

        assert 'n_clusters' in summary
        assert 'n_samples' in summary
        assert 'intrinsic_dimension' in summary
        assert summary['n_samples'] == len(X)

    def test_cluster_info(self, optdigits_subset):
        """Test cluster info method on real data."""
        X, y = optdigits_subset

        dpa = DPA(Z=2.0)
        dpa.fit(X)

        info = dpa.get_cluster_info()

        assert len(info) == dpa.n_clusters_
        for c_info in info:
            assert 'cluster_id' in c_info
            assert 'center_index' in c_info
            assert 'size' in c_info
            assert 'peak_density' in c_info
            assert 'peak_free_energy' in c_info


class TestDPAIntrinsicDimension:
    """Tests for intrinsic dimension estimation on real data."""

    def test_dimension_estimation(self, optdigits_data):
        """DPA should estimate reasonable intrinsic dimension for Optdigits."""
        X, y = optdigits_data

        dpa = DPA(Z=2.0)
        dpa.fit(X)

        # Optdigits is 64D but lies on a lower-dimensional manifold
        # Paper reports d ~ 7-13 for digit data
        assert 5 < dpa.d_ < 20, f"Expected d ~ 7-15 for digits, got {dpa.d_}"


class TestDPASklearnAPI:
    """Tests for sklearn API compatibility on real data."""

    def test_estimator_interface(self, optdigits_subset):
        """DPA should follow sklearn estimator interface."""
        X, y = optdigits_subset

        dpa = DPA(Z=2.0)

        # fit should return self
        result = dpa.fit(X)
        assert result is dpa

        # fit_predict should return labels
        dpa2 = DPA(Z=2.0)
        labels = dpa2.fit_predict(X)
        assert isinstance(labels, np.ndarray)

    def test_reproducibility(self, optdigits_subset):
        """Same input should give same output."""
        X, y = optdigits_subset

        dpa1 = DPA(Z=2.0)
        dpa2 = DPA(Z=2.0)

        labels1 = dpa1.fit_predict(X)
        labels2 = dpa2.fit_predict(X)

        np.testing.assert_array_equal(labels1, labels2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
