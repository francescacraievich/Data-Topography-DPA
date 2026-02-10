# DPA - Density Peak Advanced Clustering

[![CI](https://github.com/francescacraievich/Data-Topography-DPA/actions/workflows/ci.yml/badge.svg)](https://github.com/francescacraievich/Data-Topography-DPA/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/francescacraievich/Data-Topography-DPA/branch/master/graph/badge.svg)](https://codecov.io/gh/francescacraievich/Data-Topography-DPA)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the DPA algorithm for the **Unsupervised Learning** course.

Based on the paper included in this repository: `Density_Peaks_Advanced.pdf`

> d'Errico, M., Facco, E., Laio, A., & Rodriguez, A. (2021). *"Automatic topography of high-dimensional data sets by non-parametric density peak clustering"*. Information Sciences, 560, 476-492.

## Key Features

- **Automatic cluster detection** - No need to specify k (unlike k-means, GMM, Spectral)
- **Single parameter Z** - Statistical significance threshold with clear interpretation
- **Adaptive density estimation** - PAk with Likelihood Ratio Test
- **Error quantification** - Bounds from Fisher Information
- **Topographic analysis** - Dendrogram, connectivity matrix, decision graph

## Quick Start

```bash
# Clone and install
git clone https://github.com/francescacraievich/Data-Topography-DPA.git
cd Data-Topography-DPA
pip install -r requirements.txt

# Run examples
python examples/example_basic.py          # DPA on UCI datasets
python examples/example_comparison.py     # DPA vs DP vs DBSCAN vs Spectral vs GMM
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src    # with coverage
make test                  # shortcut
```

## Project Structure

```
Data-Topography-DPA/
├── src/
│   ├── dpa.py                   # Main DPA class (sklearn-compatible API)
│   ├── intrinsic_dimension.py   # TWO-NN estimator
│   ├── density.py               # PAk density estimator
│   ├── clustering.py            # Heuristics 1-3 + DensityPeaks baseline
│   ├── topography.py            # Visualizations
│   ├── datasets.py              # UCI dataset loaders
│   └── utils.py                 # Helper functions (Tangent Distance, etc.)
├── tests/                       # Unit tests
├── examples/                    # Example scripts
├── notebooks/                   # Jupyter notebooks
├── plots/                       # Generated figures
└── Density_Peaks_Advanced.pdf   # Reference paper
```

## DPA vs DP (Standard Density Peaks)

| Aspect | DP (2014) | DPA (2021) |
|--------|-----------|------------|
| Center selection | Manual (decision graph) | **Automatic** (Heuristic 1) |
| Density estimation | Fixed k-NN | **Adaptive PAk** (LRT) |
| Error bounds | None | **Fisher Information** |
| Cluster merging | None | **Z-score** (Heuristic 3) |
| Halo points | Manual threshold | **Automatic** (saddle density) |

## Results

Comparison of DPA with other clustering methods. Spectral and GMM receive the true number of clusters (k=10); DPA, DP, and DBSCAN find k automatically.

| Dataset | Metric | DPA | DP | DBSCAN | Spectral* | GMM* |
|---------|--------|-----|-----|--------|-----------|------|
| **Optdigits** (1797, d=9) | NMI | 0.736 | **0.805** | 0.591 | 0.789 | 0.397 |
| | ARI | 0.505 | **0.685** | 0.248 | 0.674 | 0.190 |
| | k | 11 | 17 | 26 | 10 | 10 |
| **Pendigits** (10992, d=6) | NMI | 0.781 | 0.748 | 0.706 | **0.820** | 0.593 |
| | ARI | 0.657 | 0.492 | 0.549 | **0.680** | 0.432 |
| | k | 20 | 54 | 56 | 10 | 10 |
| **MNIST 10k** (10000, d=14) | NMI | **0.664**† | 0.463 | 0.231 | 0.641 | 0.315 |
| | ARI | 0.438† | 0.228 | 0.042 | **0.461** | 0.114 |
| | k | 28† | 50 | 19 | 10 | 10 |

\*Spectral/GMM receive true k=10 as input. †DPA with Tangent Distance (k-NN hybrid). Paper reports NMI=0.84 on full MNIST 60k with TD.

DPA finds more clusters than ground truth by design — it discovers sub-structure within classes (e.g., different handwriting styles for the same digit).

## Datasets

| Dataset | Samples | Features | Classes |
|---------|---------|----------|---------|
| Optdigits | 1,797 | 64 (8x8) | 10 |
| Pendigits | 10,992 | 16 | 10 |
| MNIST | 10,000 | 784 (28x28) | 10 |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Z` | 2.0 | Significance threshold (higher = fewer clusters) |
| `d` | None | Intrinsic dimension (None = auto-estimate via TWO-NN) |
| `k_max` | 100 | Max neighbors for PAk |
| `halo` | True | Mark low-confidence points as halo |

### Z Parameter Guide

| Z | Confidence | Effect |
|---|------------|--------|
| 1.0 | ~68% | More clusters (sensitive) |
| 2.0 | ~95% | Balanced (recommended) |
| 3.0 | ~99.7% | Fewer clusters (conservative) |

For **high-dimensional data** (d > 10): use lower Z (~1.5) due to larger estimation errors.

## Tangent Distance (MNIST)

For image data, DPA uses **Tangent Distance** (Simard et al., 1993) as described in the paper (Section 3.2). The implementation uses a k-NN hybrid approach:

1. Find candidate neighbors using Euclidean distance (fast, k=300)
2. Recompute distances for candidates using one-sided Tangent Distance
3. Reorder and keep top-100 neighbors by TD

This is faster than computing the full N×N TD matrix and gives better clustering results (the Euclidean locality prior prevents spurious inter-cluster connections).

## Evaluation Metrics

- **ARI** (Adjusted Rand Index): Similarity with ground truth, corrected for chance
- **NMI** (Normalized Mutual Information): Information-theoretic measure

## Limitations

1. **High dimensions**: PAk estimation degrades when d > 10-15
2. **Computation**: O(N x k_max x log N) - slower than k-means
3. **Image data**: Tangent Distance recommended for better results

## License

MIT License - see [LICENSE](LICENSE)
