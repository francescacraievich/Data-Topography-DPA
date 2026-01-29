"""
Comparison of DPA with other clustering methods.

Compares DPA, DP, DBSCAN, Spectral Clustering, and GMM on:
- Optdigits (UCI): 1,797 samples, 64 features
- Pendigits (UCI): 10,992 samples, 16 features
- MNIST: 10,000 samples, 784 features (subset for fair comparison)

Note: DP computes O(n²) distance matrix, so MNIST is limited to 10k.
For paper's 60k MNIST with Tangent Distance, see example_mnist_tangent.py

Reference: d'Errico et al. (2021) Information Sciences
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import DPA, DensityPeaks, load_optdigits, load_pendigits, load_mnist_subset
from src.utils import compute_knn_tangent_efficient


def run_dpa(X_scaled, Z=1.5, distances=None, indices=None):
    """Run DPA clustering."""
    try:
        start = time.time()
        dpa = DPA(Z=Z, halo=False)
        labels = dpa.fit_predict(X_scaled, distances=distances, indices=indices)
        elapsed = time.time() - start
        return {
            'labels': labels,
            'n_clusters': dpa.n_clusters_,
            'd_estimated': dpa.d_,
            'time': elapsed,
            'note': f'Auto k, Z={Z}'
        }
    except Exception as e:
        return {'error': str(e)}


def run_dpa_tangent(X, image_shape, Z=1.6, k_max=100, n_jobs=-1):
    """
    Run DPA with Tangent Distance (as in paper Section 3.2).

    This is the method used by d'Errico et al. for MNIST:
    - Tangent Distance (Simard et al., 1993)
    - Z = 1.6
    - Expected ID ~ 8, NMI ~ 0.84

    Parameters
    ----------
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all CPUs.
    """
    try:
        print("    Computing Tangent Distance k-NN (parallelized)...")
        start_knn = time.time()

        # Use efficient hybrid approach with parallelization
        distances, indices = compute_knn_tangent_efficient(
            X, k_max=k_max, image_shape=image_shape,
            mode='one_sided', k_candidates=200, verbose=True,
            n_jobs=n_jobs
        )
        knn_time = time.time() - start_knn
        print(f"    k-NN computation: {knn_time:.1f}s")

        # Run DPA with precomputed Tangent Distance
        print(f"    Running DPA (Z={Z})...")
        start_dpa = time.time()
        dpa = DPA(Z=Z, halo=False)
        labels = dpa.fit_predict(X, distances=distances, indices=indices)
        dpa_time = time.time() - start_dpa

        total_time = time.time() - start_knn

        return {
            'labels': labels,
            'n_clusters': dpa.n_clusters_,
            'd_estimated': dpa.d_,
            'time': total_time,
            'knn_time': knn_time,
            'dpa_time': dpa_time,
            'note': f'Tangent Distance, Z={Z}'
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def run_dp(X_scaled, k=20, percent=2.0):
    """Run standard Density Peaks."""
    try:
        start = time.time()
        dp = DensityPeaks(k=k, percent=percent)
        labels = dp.fit_predict(X_scaled)
        elapsed = time.time() - start
        return {
            'labels': labels,
            'n_clusters': dp.n_clusters_,
            'time': elapsed,
            'note': f'Manual k ({percent}%)'
        }
    except Exception as e:
        return {'error': str(e)}


def run_dbscan(X_scaled, y_true):
    """Run DBSCAN with eps tuning (parallelized)."""
    try:
        start = time.time()
        best_ari = -1
        best_labels = None
        best_eps = None

        for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            dbscan = DBSCAN(eps=eps, min_samples=5, n_jobs=-1)
            labels = dbscan.fit_predict(X_scaled)
            n_clusters = len(np.unique(labels[labels >= 0]))
            if n_clusters > 1:
                ari = adjusted_rand_score(y_true, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_labels = labels
                    best_eps = eps

        elapsed = time.time() - start
        if best_labels is not None:
            return {
                'labels': best_labels,
                'n_clusters': len(np.unique(best_labels[best_labels >= 0])),
                'time': elapsed,
                'note': f'Best eps={best_eps}'
            }
        else:
            return {'labels': np.zeros(len(X_scaled), dtype=int), 'n_clusters': 1,
                    'time': elapsed, 'note': 'No good eps found'}
    except Exception as e:
        return {'error': str(e)}


def run_spectral_scalable(X_scaled, n_clusters, n_features):
    """Run Spectral Clustering with scalable settings."""
    try:
        start = time.time()

        # For very high dimensional data, reduce dims first
        if n_features > 100:
            pca = PCA(n_components=50, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled

        # Try AMG solver first (fastest), fallback to arpack
        try:
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                n_neighbors=15,
                eigen_solver='amg',
                assign_labels='cluster_qr',
                random_state=42,
                n_jobs=-1
            )
            labels = spectral.fit_predict(X_reduced)
            solver_used = 'amg'
        except Exception:
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                n_neighbors=15,
                eigen_solver='arpack',
                assign_labels='cluster_qr',
                random_state=42,
                n_jobs=-1
            )
            labels = spectral.fit_predict(X_reduced)
            solver_used = 'arpack'

        elapsed = time.time() - start
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'time': elapsed,
            'note': f'Requires k ({solver_used})'
        }
    except Exception as e:
        return {'error': str(e)}


def run_gmm_scalable(X_scaled, n_clusters, n_features):
    """Run GMM with scalable settings."""
    try:
        start = time.time()

        if n_features > 100:
            pca = PCA(n_components=50, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled

        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',
            init_params='k-means++',
            n_init=3,
            random_state=42
        )
        labels = gmm.fit_predict(X_reduced)
        elapsed = time.time() - start

        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'time': elapsed,
            'note': 'Requires k (diag)'
        }
    except Exception as e:
        return {'error': str(e)}


def run_all_methods(X, y_true, n_clusters_true, dataset_name):
    """Run all clustering methods on the FULL dataset."""
    results = {}
    n_samples, n_features = X.shape

    print(f"\n  Running on FULL data: {n_samples:,} samples, {n_features} features")

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine Z based on dimensionality
    Z = 1.0 if n_features > 100 else 1.5

    # 1. DPA
    print(f"  [1/5] DPA (Z={Z})...", end=" ", flush=True)
    results['DPA'] = run_dpa(X_scaled, Z=Z)
    if 'time' in results['DPA']:
        print(f"{results['DPA']['time']:.1f}s, {results['DPA']['n_clusters']} clusters")
    else:
        print(f"ERROR: {results['DPA'].get('error', 'unknown')[:50]}")

    # 2. DP
    print(f"  [2/5] DP...", end=" ", flush=True)
    results['DP'] = run_dp(X_scaled)
    if 'time' in results['DP']:
        print(f"{results['DP']['time']:.1f}s, {results['DP']['n_clusters']} clusters")
    else:
        print(f"ERROR: {results['DP'].get('error', 'unknown')[:50]}")

    # 3. DBSCAN
    print(f"  [3/5] DBSCAN...", end=" ", flush=True)
    results['DBSCAN'] = run_dbscan(X_scaled, y_true)
    if 'time' in results['DBSCAN']:
        print(f"{results['DBSCAN']['time']:.1f}s, {results['DBSCAN']['n_clusters']} clusters")
    else:
        print(f"ERROR: {results['DBSCAN'].get('error', 'unknown')[:50]}")

    # 4. Spectral (scalable)
    print(f"  [4/5] Spectral (scalable)...", end=" ", flush=True)
    results['Spectral'] = run_spectral_scalable(X_scaled, n_clusters_true, n_features)
    if 'time' in results['Spectral']:
        print(f"{results['Spectral']['time']:.1f}s")
    else:
        print(f"ERROR: {results['Spectral'].get('error', 'unknown')[:50]}")

    # 5. GMM (scalable)
    print(f"  [5/5] GMM (scalable)...", end=" ", flush=True)
    results['GMM'] = run_gmm_scalable(X_scaled, n_clusters_true, n_features)
    if 'time' in results['GMM']:
        print(f"{results['GMM']['time']:.1f}s")
    else:
        print(f"ERROR: {results['GMM'].get('error', 'unknown')[:50]}")

    return results, X_scaled


def evaluate_results(y_true, results):
    """Compute evaluation metrics."""
    metrics = {}
    for method, result in results.items():
        if 'error' in result:
            metrics[method] = {'ARI': None, 'NMI': None, 'error': result['error']}
        else:
            labels = result['labels']
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)
            metrics[method] = {
                'ARI': ari,
                'NMI': nmi,
                'n_clusters': result.get('n_clusters', 'N/A'),
                'time': result.get('time', 'N/A'),
                'note': result.get('note', '')
            }
    return metrics


def plot_comparison(X, y_true, results, title, filename=None):
    """Plot clustering results."""
    # Subsample for visualization if too large
    if len(X) > 10000:
        np.random.seed(42)
        idx = np.random.choice(len(X), 10000, replace=False)
        X_plot = X[idx]
        y_plot = y_true[idx]
        results_plot = {}
        for method, res in results.items():
            if 'labels' in res:
                results_plot[method] = {'labels': res['labels'][idx], **{k: v for k, v in res.items() if k != 'labels'}}
            else:
                results_plot[method] = res
    else:
        X_plot = X
        y_plot = y_true
        results_plot = results

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(StandardScaler().fit_transform(X_plot))

    n_methods = len(results) + 1  # +1 for ground truth
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_methods <= n_cols:
        axes = [axes] if n_methods == 1 else list(axes)
    else:
        axes = axes.flatten()

    # Ground truth
    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_plot, cmap='tab10', s=3, alpha=0.6)
    axes[0].set_title(f'Ground Truth ({len(np.unique(y_plot))} classes)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    methods = list(results.keys())
    for idx, method in enumerate(methods):
        ax = axes[idx + 1]
        if method in results_plot and 'labels' in results_plot[method]:
            labels = results_plot[method]['labels']
            mask = labels >= 0
            if np.any(mask):
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=labels[mask],
                          cmap='tab10', s=3, alpha=0.6)
            if np.any(~mask):
                ax.scatter(X_2d[~mask, 0], X_2d[~mask, 1], c='gray', s=2, alpha=0.3)

            n_clus = results_plot[method].get('n_clusters', '?')
            note = results_plot[method].get('note', '')[:25]
            ax.set_title(f'{method}: {n_clus} clusters\n({note})', fontsize=10)
        else:
            ax.set_title(f'{method} (error)')
        ax.set_xlabel('PC1')

    # Hide empty subplots
    for idx in range(len(methods) + 1, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")

    return fig


def print_metrics_table(metrics, methods=None):
    """Print a formatted metrics table."""
    if methods is None:
        methods = list(metrics.keys())

    print(f"\n  {'Method':<15} | {'ARI':>6} | {'NMI':>6} | {'Clusters':>8} | {'Time':>10} | Note")
    print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*10}-+{'-'*25}")
    for method in methods:
        if method in metrics:
            m = metrics[method]
            if m.get('ARI') is not None:
                t = f"{m['time']:.1f}s" if isinstance(m.get('time'), float) else 'N/A'
                print(f"  {method:<15} | {m['ARI']:>6.3f} | {m['NMI']:>6.3f} | {m['n_clusters']:>8} | {t:>10} | {m['note'][:23]}")
            else:
                print(f"  {method:<15} | {'ERR':>6} | {'ERR':>6} | {'-':>8} | {'-':>10} | {m.get('error', '')[:23]}")


if __name__ == "__main__":
    print("=" * 70)
    print("DPA vs Other Methods - Comprehensive Comparison")
    print("=" * 70)
    print("\nThis script compares DPA with DP and other methods on:")
    print("1. Optdigits (UCI): 1,797 samples, 64 features")
    print("2. Pendigits (UCI): 10,992 samples, 16 features")
    print("3. MNIST: 10,000 samples, 784 features (subset for fair comparison)")
    print()

    warnings.filterwarnings('ignore')

    all_metrics = {}
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # =========================================================================
    # Dataset 1: Optdigits (1797 samples, 64 features)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Dataset: Optdigits (UCI) - 1,797 samples, 64 features")
    print("=" * 70)

    data = load_optdigits()
    X, y = data['data'], data['target']

    results, X_scaled = run_all_methods(X, y, 10, 'optdigits')
    metrics = evaluate_results(y, results)
    all_metrics['optdigits'] = metrics

    print_metrics_table(metrics, ['DPA', 'DP', 'DBSCAN', 'Spectral', 'GMM'])

    fig = plot_comparison(X, y, results, 'Optdigits - All Methods on Full Data',
                         os.path.join(plot_dir, 'comparison_optdigits.png'))
    plt.close(fig)

    # =========================================================================
    # Dataset 2: Pendigits (10992 samples, 16 features)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Dataset: Pendigits (UCI) - 10,992 samples, 16 features")
    print("=" * 70)

    data = load_pendigits()
    X, y = data['data'], data['target']

    results, X_scaled = run_all_methods(X, y, 10, 'pendigits')
    metrics = evaluate_results(y, results)
    all_metrics['pendigits'] = metrics

    print_metrics_table(metrics, ['DPA', 'DP', 'DBSCAN', 'Spectral', 'GMM'])

    fig = plot_comparison(X, y, results, 'Pendigits - All Methods on Full Data',
                         os.path.join(plot_dir, 'comparison_pendigits.png'))
    plt.close(fig)

    # =========================================================================
    # Dataset 3: MNIST - 10k subset for fair comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Dataset: MNIST - 10k subset (fair comparison)")
    print("=" * 70)
    print("""
  ALL methods on 10k subset for fair comparison.
  (DP computes O(n²) matrix, cannot handle full 70k MNIST)

  Note: Paper uses DPA + Tangent Distance on 60k, but we use
  10k subset with Euclidean for fair method comparison.
    """)

    # Load MNIST 10k subset
    print("  Loading MNIST (10k subset)...")
    data = load_mnist_subset(n_samples=10000)
    X_mnist, y_mnist = data['data'], data['target']
    print(f"  Data: {X_mnist.shape}")

    # Standardize
    scaler = StandardScaler()
    X_mnist_scaled = scaler.fit_transform(X_mnist)

    # Run all methods on 10k subset
    print("\n  Running all methods on 10k subset...")

    # DPA
    print(f"  [1/5] DPA (Z=1.0)...", end=" ", flush=True)
    result_dpa = run_dpa(X_mnist_scaled, Z=1.0)
    if 'time' in result_dpa:
        print(f"{result_dpa['time']:.1f}s, {result_dpa['n_clusters']} clusters")
    else:
        print(f"ERROR: {result_dpa.get('error', 'unknown')[:50]}")

    # DP
    print(f"  [2/5] DP...", end=" ", flush=True)
    result_dp = run_dp(X_mnist_scaled)
    if 'time' in result_dp:
        print(f"{result_dp['time']:.1f}s, {result_dp['n_clusters']} clusters")
    else:
        print(f"ERROR: {result_dp.get('error', 'unknown')[:50]}")

    # DBSCAN
    print(f"  [3/5] DBSCAN...", end=" ", flush=True)
    result_dbscan = run_dbscan(X_mnist_scaled, y_mnist)
    if 'time' in result_dbscan:
        print(f"{result_dbscan['time']:.1f}s, {result_dbscan['n_clusters']} clusters")
    else:
        print(f"ERROR: {result_dbscan.get('error', 'unknown')[:50]}")

    # Spectral
    print(f"  [4/5] Spectral...", end=" ", flush=True)
    result_spectral = run_spectral_scalable(X_mnist_scaled, 10, 784)
    if 'time' in result_spectral:
        print(f"{result_spectral['time']:.1f}s")
    else:
        print(f"ERROR: {result_spectral.get('error', 'unknown')[:50]}")

    # GMM
    print(f"  [5/5] GMM...", end=" ", flush=True)
    result_gmm = run_gmm_scalable(X_mnist_scaled, 10, 784)
    if 'time' in result_gmm:
        print(f"{result_gmm['time']:.1f}s")
    else:
        print(f"ERROR: {result_gmm.get('error', 'unknown')[:50]}")

    # Evaluate all methods
    mnist_results = {
        'DPA': result_dpa,
        'DP': result_dp,
        'DBSCAN': result_dbscan,
        'Spectral': result_spectral,
        'GMM': result_gmm
    }
    metrics_mnist = evaluate_results(y_mnist, mnist_results)
    all_metrics['mnist'] = metrics_mnist

    print_metrics_table(metrics_mnist, ['DPA', 'DP', 'DBSCAN', 'Spectral', 'GMM'])

    # Save plot
    fig = plot_comparison(X_mnist, y_mnist, mnist_results,
                         'MNIST (10k subset) - All Methods',
                         os.path.join(plot_dir, 'comparison_mnist.png'))
    plt.close(fig)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: All Results")
    print("=" * 70)

    print(f"\n{'Dataset':<20} | {'Method':<15} | {'ARI':>6} | {'NMI':>6} | {'Clusters':>8}")
    print("-" * 75)

    # All datasets
    for dataset in ['optdigits', 'pendigits', 'mnist']:
        if dataset in all_metrics:
            for method in ['DPA', 'DP', 'DBSCAN', 'Spectral', 'GMM']:
                if method in all_metrics[dataset]:
                    m = all_metrics[dataset][method]
                    if m.get('ARI') is not None:
                        ds_name = 'MNIST (10k)' if dataset == 'mnist' else dataset
                        print(f"{ds_name:<20} | {method:<15} | {m['ARI']:>6.3f} | {m['NMI']:>6.3f} | {m['n_clusters']:>8}")
            print()

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
    1. DPA vs DP:
       - DP requires MANUAL center selection from decision graph
       - DPA AUTOMATES this with statistical significance testing
       - DPA uses adaptive PAk density (DP uses fixed k)
       - DP computes O(n²) matrix → limited to ~10k samples

    2. DPA vs Other Methods:
       - Spectral/GMM require knowing k (number of clusters)
       - DPA finds k automatically via Z-score merging
       - DBSCAN needs epsilon tuning

    3. Note on MNIST:
       - Paper uses Tangent Distance on 60k for NMI ~ 0.84
       - Here we use Euclidean on 10k for fair comparison

    Reference: d'Errico et al. (2021), Information Sciences
    """)

    print("=" * 70)
    print("Comparison complete! All plots saved in plots/")
    print("=" * 70)
