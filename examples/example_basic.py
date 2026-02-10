"""
DPA clustering on datasets from the paper (d'Errico et al., 2021).

Paper: "Automatic topography of high-dimensional data sets by
        non-parametric density peak clustering"
        Information Sciences 560 (2021) 476-492

Datasets:
- Optdigits: 1797 samples, 64 features (8x8 pixels)
- Pendigits: 10992 samples, 16 features
- MNIST: 10000 samples with Tangent Distance (Section 3.2)

Z parameter is chosen automatically based on data characteristics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import DPA, TwoNN, load_optdigits, load_pendigits, load_mnist_subset
from src.utils import compute_knn_tangent_fast
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA


def run_dpa(name, X, y_true, Z=None, d_override=None, distances=None, indices=None, paper_ref=None):
    """
    Run DPA clustering on a dataset.

    If Z is None, automatically determines optimal Z using recommended_Z().
    If d_override is provided, uses that intrinsic dimension instead of TWO-NN estimate.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"True classes: {len(np.unique(y_true))}")
    print(f"{'='*60}")

    # Estimate intrinsic dimension
    if d_override is not None:
        # Use provided intrinsic dimension (e.g., from paper or prior knowledge)
        d_est = d_override
        print(f"\nUsing provided intrinsic dimension: {d_est}")
    else:
        # IMPORTANT: TWO-NN should use Euclidean distance to estimate manifold dimension,
        # even when Tangent Distance is used for clustering (as in paper Section 3.2).
        # The intrinsic dimension is a property of the data manifold, not the metric.
        print("\nEstimating intrinsic dimension (TWO-NN)...")
        twonn = TwoNN()
        twonn.fit(X)  # Always use Euclidean for ID estimation
        d_est = twonn.dimension_
        print(f"  Intrinsic dimension: {d_est:.2f}")

    # Get recommended Z if not provided
    if Z is None:
        rec = DPA.recommended_Z(n_samples=X.shape[0], d=d_est)
        Z = rec['recommended_Z']
        print(f"\nAutomatic Z selection:")
        print(f"  Recommended Z: {Z:.2f}")
        print(f"  Rationale: {rec['explanation']}")
    else:
        print(f"\nUsing Z = {Z}")

    # Fit DPA
    print(f"\nFitting DPA (Z={Z:.2f})...")
    start = time.time()
    dpa = DPA(Z=Z, d=d_est, halo=True)

    if distances is not None:
        labels = dpa.fit_predict(X, distances=distances, indices=indices)
    else:
        labels = dpa.fit_predict(X)

    elapsed = time.time() - start

    # Results
    summary = dpa.summary()
    n_clust = summary['n_clusters']
    n_halo = summary['n_halo_points']

    ari = adjusted_rand_score(y_true, dpa.labels_full_)
    nmi = normalized_mutual_info_score(y_true, dpa.labels_full_)

    print(f"\nResults:")
    print(f"  Intrinsic dimension: {d_est:.2f}", end="")
    if paper_ref and 'id' in paper_ref:
        print(f" (paper: {paper_ref['id']})")
    else:
        print()

    print(f"  Clusters: {n_clust}", end="")
    if paper_ref and 'clusters' in paper_ref:
        print(f" (paper: {paper_ref['clusters']})")
    else:
        print()

    print(f"  Halo: {n_halo} ({100*n_halo/len(X):.1f}%)")
    print(f"  ARI: {ari:.3f}")
    print(f"  NMI: {nmi:.3f}", end="")
    if paper_ref and 'nmi' in paper_ref:
        print(f" (paper: {paper_ref['nmi']})")
    else:
        print()
    print(f"  Time: {elapsed:.1f}s")

    # Save plot
    save_plot(name, X, y_true, dpa, labels, Z, summary, ari, nmi, elapsed)

    return {
        'name': name, 'n_samples': X.shape[0], 'n_features': X.shape[1],
        'id': d_est, 'Z': Z, 'n_clusters': n_clust, 'n_halo': n_halo,
        'ari': ari, 'nmi': nmi, 'time': elapsed, 'dpa': dpa
    }


def save_plot(name, X, y_true, dpa, labels, Z, summary, ari, nmi, elapsed):
    """Save visualization plots (combined + separate for presentation)."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    # ==========================================================================
    # SEPARATE PLOTS FOR PRESENTATION
    # ==========================================================================

    # 1. Dendrogram (separate, larger)
    fig_dend, ax_dend = plt.subplots(figsize=(14, 7))
    dpa.plot_topography(kind='dendrogram', ax=ax_dend)
    ax_dend.set_title(f'Cluster Dendrogram - {name}\n({dpa.n_clusters_} clusters, NMI={nmi:.3f})',
                      fontsize=14, fontweight='bold')
    fig_dend.tight_layout()
    fig_dend.savefig(os.path.join(plot_dir, f'{safe_name}_dendrogram.png'), dpi=200,
                     bbox_inches='tight', facecolor='white')
    plt.close(fig_dend)
    print(f"  Dendrogram saved: plots/{safe_name}_dendrogram.png")

    # 2. Decision Graph (separate, larger)
    fig_dec, ax_dec = plt.subplots(figsize=(10, 8))
    dpa.plot_topography(kind='decision', ax=ax_dec)
    ax_dec.set_title(f'Decision Graph - {name}\n({dpa.n_clusters_} density peaks detected)',
                     fontsize=14, fontweight='bold')
    ax_dec.set_xlabel('g = log(ρ) - ε  (error-adjusted log-density)', fontsize=11)
    ax_dec.set_ylabel('δ  (min distance to higher-g point)', fontsize=11)
    fig_dec.tight_layout()
    fig_dec.savefig(os.path.join(plot_dir, f'{safe_name}_decision_graph.png'), dpi=200,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig_dec)
    print(f"  Decision graph saved: plots/{safe_name}_decision_graph.png")

    # 3. Clustering comparison (separate)
    fig_clust, axes_clust = plt.subplots(1, 2, figsize=(14, 6))

    # Ground truth
    axes_clust[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10', s=8, alpha=0.6)
    axes_clust[0].set_title(f'Ground Truth ({len(np.unique(y_true))} classes)', fontsize=12, fontweight='bold')
    axes_clust[0].set_xlabel('PC1')
    axes_clust[0].set_ylabel('PC2')

    # DPA result
    core = labels >= 0
    axes_clust[1].scatter(X_2d[core, 0], X_2d[core, 1], c=labels[core], cmap='tab20', s=8, alpha=0.6)
    if np.any(~core):
        axes_clust[1].scatter(X_2d[~core, 0], X_2d[~core, 1], c='lightgray', s=5, alpha=0.3, label='Halo')
    # Mark centers
    centers_2d = X_2d[dpa.cluster_centers_]
    axes_clust[1].scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=100, marker='*',
                          edgecolors='black', linewidths=1, label='Centers', zorder=5)
    axes_clust[1].set_title(f'DPA (Z={Z:.1f}, {dpa.n_clusters_} clusters)\nNMI={nmi:.3f}, ARI={ari:.3f}',
                            fontsize=12, fontweight='bold')
    axes_clust[1].set_xlabel('PC1')
    axes_clust[1].set_ylabel('PC2')
    axes_clust[1].legend(loc='upper right', fontsize=9)

    fig_clust.suptitle(f'{name}', fontsize=14, fontweight='bold')
    fig_clust.tight_layout()
    fig_clust.savefig(os.path.join(plot_dir, f'{safe_name}_clustering.png'), dpi=200,
                      bbox_inches='tight', facecolor='white')
    plt.close(fig_clust)
    print(f"  Clustering saved: plots/{safe_name}_clustering.png")

    # 4. Density landscape (separate)
    fig_dens, ax_dens = plt.subplots(figsize=(10, 8))
    sc = ax_dens.scatter(X_2d[:, 0], X_2d[:, 1], c=dpa.log_density_, cmap='viridis', s=10, alpha=0.7)
    ax_dens.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=150, marker='*',
                    edgecolors='white', linewidths=2, label='Density Peaks', zorder=5)
    plt.colorbar(sc, ax=ax_dens, label='log(ρ)')
    ax_dens.set_title(f'Density Landscape - {name}', fontsize=14, fontweight='bold')
    ax_dens.set_xlabel('PC1')
    ax_dens.set_ylabel('PC2')
    ax_dens.legend(loc='upper right')
    fig_dens.tight_layout()
    fig_dens.savefig(os.path.join(plot_dir, f'{safe_name}_density.png'), dpi=200,
                     bbox_inches='tight', facecolor='white')
    plt.close(fig_dens)
    print(f"  Density saved: plots/{safe_name}_density.png")

    # ==========================================================================
    # COMBINED PLOT (original, for quick overview)
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'DPA: {name}', fontsize=14, fontweight='bold')

    # Ground truth
    axes[0, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10', s=5, alpha=0.6)
    axes[0, 0].set_title(f'Ground Truth ({len(np.unique(y_true))} classes)')

    # DPA result
    axes[0, 1].scatter(X_2d[core, 0], X_2d[core, 1], c=labels[core], cmap='tab20', s=5, alpha=0.6)
    if np.any(~core):
        axes[0, 1].scatter(X_2d[~core, 0], X_2d[~core, 1], c='gray', s=3, alpha=0.2)
    axes[0, 1].set_title(f'DPA (Z={Z:.2f}, {dpa.n_clusters_} clusters)')

    # Density
    sc = axes[0, 2].scatter(X_2d[:, 0], X_2d[:, 1], c=dpa.log_density_, cmap='hot', s=5)
    plt.colorbar(sc, ax=axes[0, 2], label='log(ρ)')
    axes[0, 2].set_title('Density')

    # Decision graph & Dendrogram
    dpa.plot_topography(kind='decision', ax=axes[1, 0])
    dpa.plot_topography(kind='dendrogram', ax=axes[1, 1])

    # Metrics
    axes[1, 2].axis('off')
    txt = f"Samples: {X.shape[0]:,}\nFeatures: {X.shape[1]}\n\n"
    txt += f"ID: {summary['intrinsic_dimension']:.2f}\nZ: {Z:.2f}\n"
    txt += f"Clusters: {dpa.n_clusters_}\nHalo: {summary['n_halo_points']}\n\n"
    txt += f"ARI: {ari:.3f}\nNMI: {nmi:.3f}\nTime: {elapsed:.1f}s"
    axes[1, 2].text(0.1, 0.9, txt, transform=axes[1, 2].transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace')

    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, f'dpa_{name.lower().replace(" ", "_")}.png'), dpi=150)
    plt.close()
    print(f"  Plot saved: plots/dpa_{name.lower().replace(' ', '_')}.png")


def z_study(X, y_true, name, distances=None, indices=None):
    """
    Z parameter sensitivity study.

    Uses analyze_Z_sensitivity_fast for efficiency.
    """
    print(f"\nZ Sensitivity Study ({name}):")
    print("-" * 50)

    dpa = DPA(halo=True)

    # Use the fast method that computes densities once
    results = dpa.analyze_Z_sensitivity_fast(
        X, Z_values=None,  # Uses default range [1.0, 1.5, 2.0, 2.5, 3.0]
        distances=distances, indices=indices, y_true=y_true, verbose=False
    )

    print(f"\n   Z   | Clusters |  NMI  |  ARI")
    print("  -----|----------|-------|------")
    for i, Z in enumerate(results['Z_values']):
        nmi = results['NMI'][i] if 'NMI' in results else 0
        ari = results['ARI'][i] if 'ARI' in results else 0
        print(f"  {Z:.1f}  |    {results['n_clusters'][i]:>3}   | {nmi:.3f} | {ari:.3f}")

    # Find best Z by NMI
    if 'NMI' in results:
        best_idx = np.argmax(results['NMI'])
        best_Z = results['Z_values'][best_idx]
        best_nmi = results['NMI'][best_idx]
        print(f"\n  Best Z by NMI: {best_Z:.1f} (NMI={best_nmi:.3f})")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("DPA Clustering - d'Errico et al. (2021)")
    print("Z parameter selected automatically based on data")
    print("=" * 60)

    results = []

    # Optdigits
    print("\n>>> Loading Optdigits...")
    data = load_optdigits()
    r = run_dpa("Optdigits", data['data'], data['target'])  # Z automatic
    results.append(r)
    z_study(data['data'], data['target'], "Optdigits")

    # Pendigits
    print("\n>>> Loading Pendigits...")
    data = load_pendigits()
    r = run_dpa("Pendigits", data['data'], data['target'])  # Z automatic
    results.append(r)
    z_study(data['data'], data['target'], "Pendigits")

    # MNIST 10k with Tangent Distance (as in paper Section 3.2)
    # Uses k-NN hybrid: Euclidean candidates + TD reranking (best approach)
    print("\n>>> Loading MNIST 10k (undersampled, as in paper)...")
    print("    Using Tangent Distance (Simard et al., 1993)")
    data = load_mnist_subset(n_samples=10000)
    X_mnist, y_mnist = data['data'], data['target']

    # Cache directory
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'mnist_10000_tangent_knn.npz')

    print("    Computing k-NN with Tangent Distance (hybrid approach)...")
    start = time.time()
    distances, indices = compute_knn_tangent_fast(
        X_mnist, k_max=100, image_shape=(28, 28), mode='one_sided',
        k_candidates=300, verbose=True, cache_file=cache_file
    )
    print(f"    TD k-NN time: {time.time()-start:.1f}s")

    # Paper reference: MNIST 10k -> NMI=0.76, ID=7, Z=1.0
    paper_ref = {'nmi': 0.76, 'id': 7, 'clusters': '>10'}

    r = run_dpa("MNIST 10k", X_mnist, y_mnist,
                distances=distances, indices=indices, paper_ref=paper_ref)
    results.append(r)
    z_study(X_mnist, y_mnist, "MNIST 10k", distances=distances, indices=indices)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<15} | {'N':>6} | {'ID':>5} | {'Z':>4} | {'Clust':>5} | {'NMI':>5} | {'ARI':>5}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<15} | {r['n_samples']:>6} | {r['id']:>5.1f} | {r['Z']:>4.1f} | "
              f"{r['n_clusters']:>5} | {r['nmi']:>5.3f} | {r['ari']:>5.3f}")

    print("\nPaper reference (MNIST 10k): NMI=0.76, ID=7, Z=1.0")
    print("=" * 70)

    # Sources
    print("\nReferences:")
    print("- Official DPA: https://github.com/mariaderrico/DPA")
    print("- Documentation: https://dpaclustering.readthedocs.io/")
    print("- Paper: https://doi.org/10.1016/j.ins.2021.01.010")
