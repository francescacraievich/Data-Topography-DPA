"""
DPA clustering on all datasets from the paper (d'Errico et al., 2021).

Applies DPA to the FULL datasets without reduction:
- Optdigits: 1797 samples, 64 features (8x8 pixels)
- Pendigits: 10992 samples, 16 features
- MNIST: 70000 samples, 784 features (28x28 pixels)

Note: MNIST takes longer due to size. Use example_comparison.py for
a faster comparison with subset.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import DPA, load_optdigits, load_pendigits, load_mnist_subset
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA


def run_dpa_on_dataset(name, X, y_true, Z=2.0, save_plots=True):
    """Run DPA on a dataset and display results."""

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"True classes: {len(np.unique(y_true))}")

    # Fit DPA
    print(f"\nFitting DPA (Z={Z})...")
    start_time = time.time()
    dpa = DPA(Z=Z, halo=True)
    labels = dpa.fit_predict(X)
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f} seconds")

    # Summary
    summary = dpa.summary()
    print(f"\nResults:")
    print(f"  Intrinsic dimension (TWO-NN): {summary['intrinsic_dimension']:.2f}")
    print(f"  Clusters found: {summary['n_clusters']}")
    print(f"  Halo points: {summary['n_halo_points']} ({100*summary['n_halo_points']/len(X):.1f}%)")
    print(f"  k_hat range: [{summary['k_hat_stats']['min']}, {summary['k_hat_stats']['max']}]")

    # Metrics
    ari = adjusted_rand_score(y_true, dpa.labels_full_)
    nmi = normalized_mutual_info_score(y_true, dpa.labels_full_)
    print(f"\nClustering Quality:")
    print(f"  ARI: {ari:.3f}")
    print(f"  NMI: {nmi:.3f}")

    # Cluster details
    print(f"\nTop 5 clusters by size:")
    cluster_info = sorted(dpa.get_cluster_info(), key=lambda x: -x['size'])[:5]
    for info in cluster_info:
        print(f"  C{info['cluster_id']}: {info['size']} points, "
              f"peak density={info['peak_density']:.2f}")

    # Create visualization
    if save_plots:
        print(f"\nCreating visualization...")

        # PCA for 2D projection
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'DPA Clustering: {name}', fontsize=14, fontweight='bold')
        fig.subplots_adjust(hspace=0.35, wspace=0.3)

        # Plot 1: Ground truth
        ax = axes[0, 0]
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10', alpha=0.6, s=5)
        ax.set_title(f'Ground Truth ({len(np.unique(y_true))} classes)', fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        # Plot 2: DPA result
        ax = axes[0, 1]
        core_mask = labels >= 0
        ax.scatter(X_2d[core_mask, 0], X_2d[core_mask, 1],
                   c=labels[core_mask], cmap='tab20', alpha=0.6, s=5)
        halo_mask = labels < 0
        if np.any(halo_mask):
            ax.scatter(X_2d[halo_mask, 0], X_2d[halo_mask, 1],
                       c='gray', alpha=0.2, s=3, label=f'Halo ({np.sum(halo_mask)})')
            ax.legend(fontsize=8)
        ax.set_title(f'DPA (Z={Z}, {dpa.n_clusters_} clusters)', fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        # Plot 3: Density
        ax = axes[0, 2]
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=dpa.log_density_, cmap='hot', s=5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log(ρ)')
        ax.set_title('Density Landscape', fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        # Plot 4: Decision graph
        ax = axes[1, 0]
        dpa.plot_topography(kind='decision', ax=ax)

        # Plot 5: Dendrogram
        ax = axes[1, 1]
        dpa.plot_topography(kind='dendrogram', ax=ax)

        # Plot 6: Metrics summary
        ax = axes[1, 2]
        ax.axis('off')
        metrics_text = (
            f"Dataset: {name}\n"
            f"{'─'*30}\n\n"
            f"Samples: {X.shape[0]:,}\n"
            f"Features: {X.shape[1]}\n"
            f"True classes: {len(np.unique(y_true))}\n\n"
            f"DPA Results (Z={Z}):\n"
            f"{'─'*30}\n\n"
            f"Intrinsic dim: {summary['intrinsic_dimension']:.2f}\n"
            f"Clusters found: {summary['n_clusters']}\n"
            f"Halo points: {summary['n_halo_points']}\n\n"
            f"Quality Metrics:\n"
            f"{'─'*30}\n\n"
            f"ARI: {ari:.3f}\n"
            f"NMI: {nmi:.3f}\n\n"
            f"Time: {elapsed:.1f}s"
        )
        ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save
        plot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'dpa_{name.lower()}.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close(fig)

    return {
        'name': name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_true_classes': len(np.unique(y_true)),
        'intrinsic_dim': summary['intrinsic_dimension'],
        'n_clusters': summary['n_clusters'],
        'n_halo': summary['n_halo_points'],
        'ari': ari,
        'nmi': nmi,
        'time': elapsed
    }


def z_parameter_study(X, y_true, name):
    """Study effect of Z parameter."""
    print(f"\nZ Parameter Study for {name}:")
    print("   Z    | Clusters | Halo pts | ARI   | NMI")
    print("   -----|----------|----------|-------|------")

    results = []
    for Z in [1.0, 1.5, 2.0, 2.5, 3.0]:
        dpa = DPA(Z=Z, halo=True)
        dpa.fit(X)
        ari = adjusted_rand_score(y_true, dpa.labels_full_)
        nmi = normalized_mutual_info_score(y_true, dpa.labels_full_)
        n_halo = np.sum(dpa.halo_mask_)
        print(f"   {Z:.1f}  | {dpa.n_clusters_:^8} | {n_halo:^8} | {ari:.3f} | {nmi:.3f}")
        results.append({'Z': Z, 'clusters': dpa.n_clusters_, 'ari': ari, 'nmi': nmi})

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("DPA (Density Peak Advanced) - Full Dataset Analysis")
    print("Paper: d'Errico et al. (2021) Information Sciences")
    print("=" * 60)

    # Results accumulator
    all_results = []

    # =========================================================================
    # Dataset 1: Optdigits (small, fast)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading Optdigits...")
    data = load_optdigits()
    X_opt, y_opt = data['data'], data['target']

    result = run_dpa_on_dataset("Optdigits", X_opt, y_opt, Z=2.0)
    all_results.append(result)
    z_parameter_study(X_opt, y_opt, "Optdigits")

    # =========================================================================
    # Dataset 2: Pendigits (medium)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading Pendigits...")
    data = load_pendigits()
    X_pen, y_pen = data['data'], data['target']

    result = run_dpa_on_dataset("Pendigits", X_pen, y_pen, Z=2.0)
    all_results.append(result)
    z_parameter_study(X_pen, y_pen, "Pendigits")

    # =========================================================================
    # Dataset 3: MNIST (large - full 70k samples)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading MNIST (full dataset - this may take a while)...")
    print("Note: For faster testing, use load_mnist_subset(n_samples=5000)")

    # Load full MNIST
    data = load_mnist_subset(n_samples=None)  # None = full dataset
    X_mnist, y_mnist = data['data'], data['target']

    # MNIST is large, use slightly lower Z due to high dimensionality
    result = run_dpa_on_dataset("MNIST", X_mnist, y_mnist, Z=1.6)
    all_results.append(result)

    # Z study on full dataset
    print("\n(Z study on full MNIST - this will take a while)")
    z_parameter_study(X_mnist, y_mnist, "MNIST")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY: DPA Results on All Datasets")
    print("=" * 60)
    print(f"\n{'Dataset':<12} | {'Samples':>8} | {'Dim':>4} | {'ID':>5} | {'Clust':>5} | "
          f"{'Halo':>6} | {'ARI':>5} | {'NMI':>5} | {'Time':>6}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['name']:<12} | {r['n_samples']:>8,} | {r['n_features']:>4} | "
              f"{r['intrinsic_dim']:>5.1f} | {r['n_clusters']:>5} | "
              f"{r['n_halo']:>6} | {r['ari']:>5.3f} | {r['nmi']:>5.3f} | "
              f"{r['time']:>5.1f}s")

    print("\n" + "=" * 60)
    print("Analysis complete! Plots saved in plots/ directory.")
    print("=" * 60)

    # Show interpretation
    print("\nInterpretation:")
    print("-" * 40)
    print("- Intrinsic Dimension (ID): Estimated manifold dimension")
    print("  Lower than embedding dimension indicates data lies on manifold")
    print("- Halo points: Low-confidence assignments (density < saddle)")
    print("- ARI/NMI: Higher = better match to ground truth")
    print("\nNote: MNIST benefits from Tangent Distance (not used here)")
    print("      See example_comparison.py for Tangent Distance results")
