"""
DPA - Density Peak Advanced Clustering

Automatic topography of high-dimensional data sets by non-parametric
density peak clustering.

Based on: d'Errico, M., Facco, E., Laio, A., & Rodriguez, A. (2021).
"Automatic topography of high-dimensional data sets by non-parametric
density peak clustering" Information Sciences, 560, 476-492.

Main Components:
- DPA: Main clustering class with sklearn-compatible API
- TwoNN: Intrinsic dimension estimation
- BlockAnalysis: Scale-dependent dimension analysis (d vs N)
- PAk: Point Adaptive k-NN density estimator with systematic correction α
- Topography: Visualization utilities
- TangentDistanceMetric: For image data (MNIST)
"""

from .dpa import DPA
from .intrinsic_dimension import TwoNN, estimate_intrinsic_dimension, BlockAnalysis, block_analysis
from .density import PAk, estimate_density_pak
from .topography import Topography
from .clustering import (
    find_cluster_centers,
    find_saddle_points,
    merge_insignificant_peaks,
    identify_halo_points,
    DensityPeaks
)
from .utils import TangentDistanceMetric, compute_tangent_distance_matrix, compute_knn_tangent_efficient
from .datasets import (
    load_optdigits,
    load_pendigits,
    load_mnist_subset,
    get_dataset,
    AVAILABLE_DATASETS
)

__all__ = [
    'DPA',
    'DensityPeaks',
    'TwoNN',
    'PAk',
    'Topography',
    'BlockAnalysis',
    'TangentDistanceMetric',
    'estimate_intrinsic_dimension',
    'block_analysis',
    'estimate_density_pak',
    'find_cluster_centers',
    'find_saddle_points',
    'merge_insignificant_peaks',
    'identify_halo_points',
    'compute_tangent_distance_matrix',
    'compute_knn_tangent_efficient',
    'load_optdigits',
    'load_pendigits',
    'load_mnist_subset',
    'get_dataset',
    'AVAILABLE_DATASETS'
]

__version__ = '1.0.0'
__author__ = 'Implementation based on d\'Errico et al. (2021)'
