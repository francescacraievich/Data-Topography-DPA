"""
DPA topography at Z=1, Z=2, Z=3 on Pendigits — paper-faithful figure.

Reproduces the visual language of Fig. 1E/1F and Fig. 4 in
d'Errico et al. (2021), Information Sciences 560, 476-492:
for each Z, the same clustering is shown through its three
complementary topographic representations (dendrogram, cluster
purity matrix, network), rather than comparing DPA against other
methods on a single Z. This is the property no other clustering
algorithm in the project's comparison offers.

TWO-NN, PAk and Heuristics 1-2 (center detection, saddle detection)
do not depend on Z, so they are computed once and reused; only
Heuristic 3 (merging) is re-run for each Z value.
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import TwoNN, PAk, load_pendigits
from src.utils import compute_knn
from src.clustering import (
    find_cluster_centers,
    assign_to_clusters,
    find_saddle_points,
    merge_insignificant_peaks,
    identify_halo_points,
    relabel_after_merge,
)
from src.topography import Topography
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Consistent color scheme for the 10 digit classes, shared by every
# panel and every Z value (dendrogram leaves, network nodes, ...).
DIGIT_COLORS = {i: plt.get_cmap('tab10')(i) for i in range(10)}

Z_VALUES = [1.0, 2.0, 3.0]


def compute_shared_density(X, k_max=100, metric='euclidean', d=None):
    """
    Run the Z-independent part of the DPA pipeline exactly once:
    k-NN graph, TWO-NN intrinsic dimension, PAk density estimation,
    Heuristic 1 (center detection) and Heuristic 2 (saddle detection).
    """
    n_samples = X.shape[0]
    k_actual = min(k_max, n_samples - 1)
    distances, indices = compute_knn(X, k_actual, metric)

    if d is None:
        twonn = TwoNN()
        twonn.fit(X, precomputed_distances=distances, precomputed_indices=indices)
        d = twonn.dimension_

    pak = PAk(d=d, k_max=k_max)
    pak.fit(X, distances=distances, indices=indices)

    log_density = pak.log_density_
    epsilon = pak.epsilon_
    k_hat = pak.k_hat_
    g = log_density - epsilon

    centers, delta, nearest_higher = find_cluster_centers(
        log_density, epsilon, distances, indices, k_hat
    )
    if len(centers) == 0:
        centers = np.array([np.argmax(g)])

    labels = assign_to_clusters(g, nearest_higher, centers, n_samples)

    # Points with no higher-density neighbor: fall back to nearest assigned neighbor.
    unassigned = np.where(labels < 0)[0]
    for i in unassigned:
        for j_pos in range(1, indices.shape[1]):
            j = indices[i, j_pos]
            if labels[j] >= 0:
                labels[i] = labels[j]
                break

    saddles, _ = find_saddle_points(
        labels, g, log_density, epsilon, distances, indices, k_hat, centers=centers
    )

    return {
        'd': d,
        'log_density': log_density,
        'epsilon': epsilon,
        'centers': centers,
        'initial_labels': labels,
        'saddles': saddles,
    }


def cluster_for_Z(shared, Z, halo=True):
    """Apply Heuristic 3 (merging) for a given Z, reusing the shared density/saddles."""
    centers = shared['centers']
    saddles = shared['saddles']
    log_density = shared['log_density']
    epsilon = shared['epsilon']
    labels = shared['initial_labels'].copy()

    if len(centers) > 1:
        merged_centers, merge_map, _ = merge_insignificant_peaks(
            np.array(centers), dict(saddles), log_density, epsilon, Z
        )
        labels_full = relabel_after_merge(labels, merge_map)

        # Re-key saddles onto the merged cluster ids, keeping the highest
        # saddle when several old pairs collapse onto the same new pair.
        new_saddles = {}
        for pair, info in saddles.items():
            c1, c2 = pair
            new_c1 = merge_map.get(c1, c1)
            new_c2 = merge_map.get(c2, c2)
            if new_c1 != new_c2:
                new_pair = tuple(sorted([new_c1, new_c2]))
                if new_pair not in new_saddles or \
                   info['log_density'] > new_saddles[new_pair]['log_density']:
                    new_saddles[new_pair] = info
        saddles_final = new_saddles
    else:
        merged_centers = centers
        labels_full = labels
        saddles_final = saddles

    n_clusters = len(merged_centers)

    if halo and n_clusters > 1:
        halo_mask, _ = identify_halo_points(labels_full, log_density, saddles_final, merged_centers)
        labels_masked = labels_full.copy()
        labels_masked[halo_mask] = -1
    else:
        halo_mask = np.zeros(len(labels_full), dtype=bool)
        labels_masked = labels_full.copy()

    topography = Topography(merged_centers, saddles_final, log_density, labels_full)

    return {
        'Z': Z,
        'n_clusters': n_clusters,
        'cluster_centers': merged_centers,
        'saddles': saddles_final,
        'labels_full': labels_full,
        'labels_masked': labels_masked,
        'halo_mask': halo_mask,
        'topography': topography,
    }


def save_individual_panels(res, y, plot_dir):
    """Save the three panels for one Z value separately, for use in slides."""
    topo = res['topography']
    Z = res['Z']

    fig_d, ax_d = plt.subplots(figsize=(10, 7))
    topo.plot_dendrogram(ax=ax_d, ground_truth=y, digit_colors=DIGIT_COLORS)
    ax_d.set_title(f'Dendrogram — Z={Z:.1f} ({res["n_clusters"]} clusters)',
                   fontsize=13, fontweight='bold')
    fig_d.tight_layout()
    fig_d.savefig(os.path.join(plot_dir, f'pendigits_z{Z:.1f}_dendrogram.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_d)

    fig_c, ax_c = plt.subplots(figsize=(8, 7))
    topo.plot_confusion_matrix(y, ax=ax_c)
    ax_c.set_title(f'Purity Matrix — Z={Z:.1f}', fontsize=13, fontweight='bold')
    fig_c.tight_layout()
    fig_c.savefig(os.path.join(plot_dir, f'pendigits_z{Z:.1f}_confusion.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_c)

    fig_n, ax_n = plt.subplots(figsize=(9, 9))
    topo.plot_network(ax=ax_n, ground_truth=y, digit_colors=DIGIT_COLORS)
    ax_n.set_title(f'Network — Z={Z:.1f} ({res["n_clusters"]} clusters)',
                   fontsize=13, fontweight='bold')
    fig_n.tight_layout()
    fig_n.savefig(os.path.join(plot_dir, f'pendigits_z{Z:.1f}_network.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_n)


def main():
    print("=" * 70)
    print("DPA Z-sweep on Pendigits - paper-faithful topography (Z=1, 2, 3)")
    print("=" * 70)

    data = load_pendigits()
    X, y = data['data'], data['target']
    print(f"Pendigits: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} ground-truth classes")

    plot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    print("\nComputing k-NN, TWO-NN, PAk, Heuristics 1-2 (once, Z-independent)...")
    t0 = time.time()
    shared = compute_shared_density(X)
    print(f"  Intrinsic dimension d = {shared['d']:.2f}  ({time.time() - t0:.1f}s)")
    print(f"  Raw density peaks (Heuristic 1, before merging): {len(shared['centers'])}")

    results = []
    for Z in Z_VALUES:
        t0 = time.time()
        res = cluster_for_Z(shared, Z)
        res['nmi'] = normalized_mutual_info_score(y, res['labels_full'])
        res['ari'] = adjusted_rand_score(y, res['labels_full'])
        print(f"  Z={Z:.1f}: {res['n_clusters']:3d} clusters, "
              f"NMI={res['nmi']:.3f}, ARI={res['ari']:.3f}  ({time.time() - t0:.2f}s)")
        results.append(res)

    # ==========================================================================
    # Composite 3x3 figure: rows = Z, columns = (dendrogram, confusion, network)
    # ==========================================================================
    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.35)

    for row, res in enumerate(results):
        topo = res['topography']

        ax_dendro = fig.add_subplot(gs[row, 0])
        topo.plot_dendrogram(ax=ax_dendro, ground_truth=y, digit_colors=DIGIT_COLORS)

        ax_conf = fig.add_subplot(gs[row, 1])
        topo.plot_confusion_matrix(y, ax=ax_conf)

        ax_net = fig.add_subplot(gs[row, 2])
        topo.plot_network(ax=ax_net, ground_truth=y, digit_colors=DIGIT_COLORS)

        row_label = (f"Z = {res['Z']:.1f}\n"
                     f"({res['n_clusters']} clusters, NMI = {res['nmi']:.3f})")
        ax_dendro.annotate(row_label, xy=(-0.38, 0.5), xycoords='axes fraction',
                            rotation=90, ha='center', va='center',
                            fontsize=13, fontweight='bold')

    fig.suptitle('DPA Topography at Different Significance Levels — Pendigits',
                 fontsize=17, fontweight='bold', y=0.995)
    fig.savefig(os.path.join(plot_dir, 'z_sweep_pendigits.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nComposite figure saved: plots/z_sweep_pendigits.png")

    # ==========================================================================
    # Individual panels, for direct use in presentation slides
    # ==========================================================================
    for res in results:
        save_individual_panels(res, y, plot_dir)
        print(f"  Individual panels saved for Z={res['Z']:.1f}")

    print("\nSummary:")
    print(f"{'Z':>5} | {'Clusters':>8} | {'NMI':>6} | {'ARI':>6}")
    print("-" * 35)
    for res in results:
        print(f"{res['Z']:>5.1f} | {res['n_clusters']:>8} | {res['nmi']:>6.3f} | {res['ari']:>6.3f}")


if __name__ == "__main__":
    main()
