"""
Dataset utilities for DPA experiments.

Provides easy access to datasets used in the paper:
- Optdigits (UCI): 8x8 handwritten digits
- Pendigits (UCI): 16D pen trajectory data
- MNIST (subset): 28x28 handwritten digits

All datasets are loaded via sklearn and cached automatically.
No manual download required.

Reference datasets from d'Errico et al. (2021):
- Optdigits: 1797 samples, 64 features, 10 classes
- Pendigits: 10992 samples, 16 features, 10 classes
- MNIST: 70000 samples, 784 features, 10 classes
"""

import os

import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler


def load_optdigits(normalize=True, return_X_y=False):
    """
    Load Optdigits dataset (UCI).

    8x8 pixel images of handwritten digits 0-9.
    Used in the DPA paper for clustering experiments.

    Parameters
    ----------
    normalize : bool, default=True
        Whether to normalize features to [0, 1].
    return_X_y : bool, default=False
        If True, return (X, y) tuple instead of dict.

    Returns
    -------
    dict or tuple
        Dataset with keys: 'data', 'target', 'images', 'image_shape', 'description'
        Or (X, y) tuple if return_X_y=True.

    Examples
    --------
    >>> from src.datasets import load_optdigits
    >>> data = load_optdigits()
    >>> print(f"Samples: {data['data'].shape[0]}, Features: {data['data'].shape[1]}")
    >>> X, y = load_optdigits(return_X_y=True)
    """
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target

    if normalize:
        X = X / 16.0  # Original values are 0-16

    if return_X_y:
        return X, y

    return {
        'data': X,
        'target': y,
        'images': X.reshape(-1, 8, 8),
        'image_shape': (8, 8),
        'n_classes': 10,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'description': 'Optdigits: 8x8 handwritten digits from UCI repository'
    }


def load_pendigits(normalize=True, return_X_y=False):
    """
    Load Pendigits dataset (UCI).

    16-dimensional pen trajectory data for handwritten digits.
    Each sample has 8 (x, y) coordinates of pen position.

    Parameters
    ----------
    normalize : bool, default=True
        Whether to standardize features.
    return_X_y : bool, default=False
        If True, return (X, y) tuple.

    Returns
    -------
    dict or tuple
        Dataset dict or (X, y) tuple.

    Examples
    --------
    >>> from src.datasets import load_pendigits
    >>> X, y = load_pendigits(return_X_y=True)
    >>> print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    """
    pendigits = fetch_openml('pendigits', version=1, as_frame=False, parser='auto')
    X = pendigits.data.astype(np.float64)
    y = pendigits.target.astype(int)

    if normalize:
        X = StandardScaler().fit_transform(X)

    if return_X_y:
        return X, y

    return {
        'data': X,
        'target': y,
        'n_classes': 10,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'description': 'Pendigits: 16D pen trajectory data from UCI repository'
    }


def load_mnist_subset(n_samples=5000, normalize=True, return_X_y=False, random_state=42):
    """
    Load MNIST dataset (full or subset).

    28x28 pixel images of handwritten digits.
    Full MNIST has 70,000 samples.

    Parameters
    ----------
    n_samples : int or None, default=5000
        Number of samples to load (stratified by class).
        Use None to load the FULL dataset (70,000 samples).
    normalize : bool, default=True
        Whether to normalize features to [0, 1].
    return_X_y : bool, default=False
        If True, return (X, y) tuple.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict or tuple
        Dataset dict or (X, y) tuple.

    Examples
    --------
    >>> from src.datasets import load_mnist_subset
    >>> X, y = load_mnist_subset(n_samples=1000, return_X_y=True)
    >>> print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    >>> # Load full dataset
    >>> X_full, y_full = load_mnist_subset(n_samples=None, return_X_y=True)

    Notes
    -----
    For best results with DPA on MNIST, use Tangent Distance:
    >>> from src.utils import TangentDistanceMetric
    >>> metric = TangentDistanceMetric(image_shape=(28, 28))
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_full = mnist.data.astype(np.float64)
    y_full = mnist.target.astype(int)

    # Full dataset or subset?
    if n_samples is None or n_samples >= len(X_full):
        # Load full dataset
        X = X_full
        y = y_full
        description = f'MNIST full: {len(X)} samples of 28x28 handwritten digits'
    else:
        # Stratified subsample
        rng = np.random.RandomState(random_state)
        n_per_class = n_samples // 10

        indices = []
        for digit in range(10):
            digit_idx = np.where(y_full == digit)[0]
            selected = rng.choice(digit_idx, size=min(n_per_class, len(digit_idx)), replace=False)
            indices.extend(selected)

        indices = np.array(indices)
        rng.shuffle(indices)

        X = X_full[indices]
        y = y_full[indices]
        description = f'MNIST subset: {len(X)} samples of 28x28 handwritten digits'

    if normalize:
        X = X / 255.0

    if return_X_y:
        return X, y

    return {
        'data': X,
        'target': y,
        'images': X.reshape(-1, 28, 28),
        'image_shape': (28, 28),
        'n_classes': 10,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'description': description
    }


def generate_spir2(n_samples=20000, noise=0.0, random_state=42):
    """
    Generate the SPIR2 two-spirals toy dataset (paper Fig. 3 style).

    Two interleaving Archimedean spirals (r = a + b*theta), offset by pi
    from each other, with Gaussian noise added perpendicular to each arm
    (not radially, so the noise doesn't blur the spiral's angular
    structure). Each arm is one ground-truth cluster. The exact SPIR2
    generating density from the paper isn't published, so this is a
    close visual reconstruction: two clearly non-convex, interleaved
    arms that k-means and other centroid-based methods cannot separate,
    but that a density-based method like DPA can.

    Parameters
    ----------
    n_samples : int, default=20000
        Total number of points (split evenly between the two arms).
    noise : float, default=0.0
        Standard deviation of the Gaussian noise added perpendicular to
        each spiral arm. The paper-like look uses noise in [0.3, 0.5].
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
        Ground-truth arm label, 0 or 1.

    Examples
    --------
    >>> from src.datasets import generate_spir2
    >>> X, y = generate_spir2(n_samples=20000, noise=0.35)
    """
    rng = np.random.RandomState(random_state)
    n_arm0 = n_samples // 2
    n_arm1 = n_samples - n_arm0

    a, b = 0.3, 0.4
    theta_max = 3.0 * np.pi

    def make_arm(n, theta_offset):
        theta = np.sort(rng.uniform(0.0, theta_max, size=n))
        r = a + b * theta
        x = r * np.cos(theta + theta_offset)
        y_ = r * np.sin(theta + theta_offset)

        # Noise perpendicular to the arm's tangent direction, so it
        # thickens the arm without smearing points along its length.
        tangent_angle = theta + theta_offset + np.pi / 2.0
        perp = rng.normal(0.0, noise, size=n)
        x = x + perp * np.cos(tangent_angle)
        y_ = y_ + perp * np.sin(tangent_angle)
        return np.column_stack([x, y_])

    arm0 = make_arm(n_arm0, 0.0)
    arm1 = make_arm(n_arm1, np.pi)

    X = np.vstack([arm0, arm1])
    y = np.concatenate([np.zeros(n_arm0, dtype=int), np.ones(n_arm1, dtype=int)])

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def load_spir2(return_X_y=False):
    """
    Load the official SPIR2 two-spirals dataset used in the paper's
    Fig. 3 (d'Errico et al., 2021).

    Sourced from the official DADApy repository
    (github.com/sissa-data-science/DADApy,
    examples/datasets/Fig2.dat + gt_F2.txt), which is the direct
    successor library to the DPA code from the same research group.
    Identified by cross-referencing the paper's Table 1 dataset
    ordering (CLUS8=Fig.1, SPIR2=Fig.3, AGGR/SPIR3/HORSE=Figs. S2-S4)
    against DADApy's own example notebook, which loads
    'datasets/Fig1.dat' and 'datasets/Fig2.dat' as alternatives for the
    same clustering demo - Fig1.dat is CLUS8, so Fig2.dat is SPIR2.

    38358 points in [-1, 1]^2: two interleaving spiral arms (labels 0
    and 1, ~16500 points each) plus uniform background noise (label -1,
    ~5350 points, matching the paper's use of 'halo'/noise points to
    validate DPA's halo mechanism).

    Falls back to generate_spir2() (a synthetic approximation) if the
    local data/spir2.csv file is not present.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, return (X, y) tuple instead of dict.

    Returns
    -------
    dict or tuple
        Dataset dict (keys: 'data', 'target', ...) or (X, y) tuple.
        y == -1 marks ground-truth background noise points.

    Examples
    --------
    >>> from src.datasets import load_spir2
    >>> X, y = load_spir2(return_X_y=True)
    """
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'spir2.csv'
    )

    if os.path.exists(data_path):
        raw = np.genfromtxt(data_path, delimiter=',', skip_header=1)
        X = raw[:, :2]
        y = raw[:, 2].astype(int)
        description = ('SPIR2 (official, paper Fig. 3): two spirals + background '
                       'noise, from DADApy examples/datasets/Fig2.dat')
    else:
        X, y = generate_spir2(n_samples=20000, noise=0.35)
        description = 'SPIR2 (synthetic approximation - official data/spir2.csv not found)'

    if return_X_y:
        return X, y

    return {
        'data': X,
        'target': y,
        'n_classes': len(np.unique(y[y >= 0])),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'description': description
    }


def get_dataset(name, **kwargs):
    """
    Load dataset by name.

    Parameters
    ----------
    name : str
        Dataset name: 'optdigits', 'pendigits', 'mnist'
    **kwargs
        Additional arguments passed to loader.

    Returns
    -------
    Dataset dict or tuple depending on return_X_y parameter.

    Examples
    --------
    >>> from src.datasets import get_dataset
    >>> X, y = get_dataset('optdigits', return_X_y=True)
    >>> data = get_dataset('mnist', n_samples=1000)
    """
    loaders = {
        'optdigits': load_optdigits,
        'pendigits': load_pendigits,
        'mnist': load_mnist_subset,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    return loaders[name](**kwargs)


# List of available datasets (all REAL UCI data, same as used in the paper)
AVAILABLE_DATASETS = ['optdigits', 'pendigits', 'mnist']
