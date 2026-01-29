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
