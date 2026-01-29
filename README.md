# DPA - Density Peak Advanced Clustering

[![CI](https://github.com/francescacraievich/Data-Topography-DPA/actions/workflows/ci.yml/badge.svg)](https://github.com/francescacraievich/Data-Topography-DPA/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the DPA algorithm from d'Errico et al. (2021) for the **Unsupervised Learning** course.

> **Paper**: d'Errico, M., Facco, E., Laio, A., & Rodriguez, A. (2021). *"Automatic topography of high-dimensional data sets by non-parametric density peak clustering"*. Information Sciences, 560, 476-492.

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

## Usage

```python
from src import DPA, load_optdigits

# Load data
data = load_optdigits()
X, y = data['data'], data['target']

# Fit DPA
dpa = DPA(Z=2.0)
labels = dpa.fit_predict(X)

# Results
print(f"Clusters found: {dpa.n_clusters_}")
print(f"Intrinsic dimension: {dpa.d_:.2f}")

# Visualization
dpa.plot_topography(kind='dendrogram')
```

## Installation

```bash
# Basic installation
pip install -r requirements.txt

# Development mode (editable)
pip install -e .
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src

# Or use Makefile
make test
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
└── plots/                       # Generated figures
```

## DPA vs DP (Standard Density Peaks)

| Aspect | DP (2014) | DPA (2021) |
|--------|-----------|------------|
| Center selection | Manual (decision graph) | **Automatic** (Heuristic 1) |
| Density estimation | Fixed k-NN | **Adaptive PAk** (LRT) |
| Error bounds | None | **Fisher Information** |
| Cluster merging | None | **Z-score** (Heuristic 3) |
| Halo points | Manual threshold | **Automatic** (saddle density) |

## Datasets

All datasets are from UCI, same as the paper:

| Dataset | Samples | Features | Classes |
|---------|---------|----------|---------|
| Optdigits | 1,797 | 64 (8×8) | 10 |
| Pendigits | 10,992 | 16 | 10 |
| MNIST | 70,000 | 784 (28×28) | 10 |

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

## Key Formulas

| Component | Formula |
|-----------|---------|
| TWO-NN | μ = r₂/r₁ → d = slope of -log(1-F(μ)) vs log(μ) |
| PAk (LRT) | D_k = 2(L_{k+1}(ρ_{k+1}) - L_{k+1}(ρ_k)) < D_thr |
| Fisher Info | ε = √(4(k̂+2) / ((k̂-1)k̂)) |
| Heuristic 3 | Merge if: log(ρ_c) - log(ρ_{cc'}) < Z × √(ε_c² + ε_{cc'}²) |

## Evaluation Metrics

- **ARI** (Adjusted Rand Index): Similarity with ground truth, corrected for chance
- **NMI** (Normalized Mutual Information): Information-theoretic measure

## Limitations

1. **High dimensions**: PAk estimation degrades when d > 10-15
2. **Computation**: O(N × k_max × log N) - slower than k-means
3. **Image data**: Tangent Distance recommended for better results (implemented in `utils.py`)

## CI/CD

GitHub Actions automatically runs:
- Tests on Python 3.9-3.12, Linux/Windows/macOS
- Linting with flake8
- Coverage report
- Example verification

## References

1. d'Errico et al. (2021) - DPA paper
2. Rodriguez & Laio (2014) - Original Density Peaks
3. Facco et al. (2017) - TWO-NN dimension estimation

## License

MIT License
