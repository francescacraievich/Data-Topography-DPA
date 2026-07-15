"""
DPA topography Z-sweep across three datasets: Optdigits, Pendigits, MNIST 10k.

Generalizes z_sweep_optdigits.py / z_sweep_pendigits.py into a single script
so the same paper-faithful topography figure (Fig. 1E/1F, Fig. 4 style:
dendrogram + network + PCA assignment, instead of a multi-method scatter
comparison) can be produced and compared across datasets with different
intrinsic dimension.

For each dataset, TWO-NN (on Euclidean distance, always - dimension is a
property of the manifold, not of whatever metric is used for clustering)
and PAk are computed ONCE and reused; only Heuristic 3 (merging) is re-run
for each Z value.

MNIST 10k uses the cached Tangent Distance k-NN graph if available
(cache/mnist_10000_tangent_knn.npz, built by examples/example_basic.py),
falling back to Euclidean on raw pixels otherwise.
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

from src import TwoNN, PAk, load_optdigits, load_pendigits, load_mnist_subset
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
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(REPO_ROOT, 'plots')
CACHE_DIR = os.path.join(REPO_ROOT, 'cache')

MIN_CLUSTERS = 2
MAX_CLUSTERS = 20


# ==============================================================================
# Dataset loaders: each returns (X, y, distances, indices, note).
# distances/indices are None unless a precomputed neighbor graph should be
# used for clustering (e.g. Tangent Distance for MNIST); TWO-NN always uses
# fresh Euclidean neighbors on X regardless of what is returned here.
# ==============================================================================

def load_optdigits_data():
    data = load_optdigits()
    return data['data'], data['target'], None, None, ''


def load_pendigits_data():
    data = load_pendigits()
    return data['data'], data['target'], None, None, ''


def load_mnist10k_data():
    data = load_mnist_subset(n_samples=10000)
    X, y = data['data'], data['target']
    cache_file = os.path.join(CACHE_DIR, 'mnist_10000_tangent_knn.npz')
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        distances, indices = cached['distances'], cached['indices']
        note = f'Using cached Tangent Distance k-NN ({cache_file})'
    else:
        distances, indices = None, None
        note = ('Tangent Distance cache not found at cache/mnist_10000_tangent_knn.npz - '
                'falling back to Euclidean on raw pixels. Tangent Distance would separate '
                'digit classes better but is too slow to compute from scratch here.')
    return X, y, distances, indices, note


DATASETS = [
    ('optdigits', 'Optdigits', load_optdigits_data, [1.0, 2.5, 5.0], MAX_CLUSTERS, 100),
    # Pendigits: k_max=500 (vs. the 100 default) because k-hat otherwise
    # saturates at the ceiling for ~99% of points, artificially shrinking
    # epsilon and inflating the raw peak count. Z=[2.0, 3.0, 5.0] was swept
    # and picked after fixing the density estimator (src/density.py: damped
    # Newton-Raphson alpha correction, two-point LRT, corrected epsilon
    # formula, matching the official reference implementation) - the old
    # Z=[1.0, 4.0, 10.0] range is poorly calibrated post-fix (Z=10 collapses
    # to a single dominant cluster). [2.0, 3.0, 5.0] gives a clean,
    # non-degenerate 19->8->5 cluster spread with NMI 0.765->0.633->0.474.
    ('pendigits', 'Pendigits', load_pendigits_data, [2.0, 3.0, 5.0], MAX_CLUSTERS, 500),
    ('mnist10k', 'MNIST 10k', load_mnist10k_data, [1.0, 2.0, 4.0], MAX_CLUSTERS, 100),
]


# ==============================================================================
# Z-independent pipeline (k-NN, TWO-NN, PAk, Heuristics 1-2) + Heuristic 3
# ==============================================================================

def compute_shared_density(X, k_max=100, metric='euclidean', d=None,
                           distances=None, indices=None):
    """
    Run the Z-independent part of the DPA pipeline exactly once.

    distances/indices, if given, are the neighbor graph used for PAk and
    Heuristics 1-2 (the clustering metric, e.g. Tangent Distance for MNIST).
    Intrinsic dimension is always estimated with Euclidean TWO-NN directly
    on X, regardless of the clustering metric.
    """
    n_samples = X.shape[0]

    if distances is None or indices is None:
        k_actual = min(k_max, n_samples - 1)
        distances, indices = compute_knn(X, k_actual, metric)
    else:
        k_actual = min(k_max, indices.shape[1] - 1)

    if d is None:
        twonn = TwoNN()
        twonn.fit(X)  # always Euclidean, independent of the clustering metric
        d = twonn.dimension_

    pak = PAk(d=d, k_max=k_actual)
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
    else:
        halo_mask = np.zeros(len(labels_full), dtype=bool)

    topography = Topography(merged_centers, saddles_final, log_density, labels_full)

    return {
        'Z': Z,
        'n_clusters': n_clusters,
        'cluster_centers': merged_centers,
        'saddles': saddles_final,
        'labels_full': labels_full,
        'halo_mask': halo_mask,
        'topography': topography,
    }


def adjust_Z(shared, Z_target, min_clusters=MIN_CLUSTERS, max_clusters=MAX_CLUSTERS,
            max_iters=6, verbose=True):
    """
    Nudge Z away from Z_target if it produces too few/too many clusters,
    printing every probe so the requested Z -> actual Z mapping is visible
    and can be adjusted manually if the automatic nudge isn't satisfying.
    """
    Z = Z_target
    res = None
    for _ in range(max_iters):
        res = cluster_for_Z(shared, Z)
        n = res['n_clusters']
        if verbose:
            print(f"      probe Z={Z:.2f} -> {n} clusters")
        if n < min_clusters:
            Z = max(0.05, Z * 0.6)
        elif n > max_clusters:
            Z = min(50.0, Z * 1.4)
        else:
            return Z, res
    return Z, res


# ==============================================================================
# Plotting
# ==============================================================================

def unique_cluster_colors(n_clusters):
    """
    Assign each cluster a unique color indexed directly by cluster id -
    no graph coloring, no spatial-adjacency reuse, no majority-rule. Every
    cluster gets its own palette slot, so two clusters never share a
    color regardless of where they sit in space.
    """
    if n_clusters <= 0:
        return {}
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_clusters]
    elif n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_clusters]
    else:
        # For 20+ clusters, use HSV evenly spaced for maximum distinction
        colors = plt.cm.hsv(np.linspace(0, 0.9, n_clusters))
    return {i: colors[i] for i in range(n_clusters)}


def plot_scatter(ax, X_2d, res, cluster_colors):
    """
    PCA scatter: every point colored by its cluster id (a distinct color
    per cluster, see unique_cluster_colors - not majority-rule, so
    clusters sharing the same dominant digit still look visually
    distinct). Halo points use the SAME color as their cluster, just
    smaller/fainter - no gray. This keeps cluster membership visible
    even for halo points.
    """
    labels_full = res['labels_full']
    halo_mask = res['halo_mask']
    core_mask = ~halo_mask

    if halo_mask.any():
        halo_colors = [cluster_colors[c] for c in labels_full[halo_mask]]
        ax.scatter(X_2d[halo_mask, 0], X_2d[halo_mask, 1], c=halo_colors,
                   s=8, alpha=0.25, linewidths=0, zorder=1)

    if core_mask.any():
        core_colors = [cluster_colors[c] for c in labels_full[core_mask]]
        ax.scatter(X_2d[core_mask, 0], X_2d[core_mask, 1], c=core_colors,
                   s=20, alpha=0.9, edgecolors='black', linewidths=0.3, zorder=2)

    ax.set_title(f"DPA (Z={res['Z']:.2f}, {res['n_clusters']} clusters)",
                fontsize=11, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')


def save_individual_panels(dataset_key, res, X_2d, y, plot_dir):
    """Save the three panels for one Z value separately, for use in slides."""
    topo = res['topography']
    Z = res['Z']
    scatter_colors = unique_cluster_colors(res['n_clusters'])

    fig_s, ax_s = plt.subplots(figsize=(8, 7))
    plot_scatter(ax_s, X_2d, res, scatter_colors)
    ax_s.set_title(f'PCA Assignment — Z={Z:.2f} ({res["n_clusters"]} clusters)',
                   fontsize=13, fontweight='bold')
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(plot_dir, f'{dataset_key}_z{Z:.1f}_scatter.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_s)

    fig_d, ax_d = plt.subplots(figsize=(10, 7))
    topo.plot_dendrogram(ax=ax_d)
    ax_d.set_title(f'Dendrogram — Z={Z:.2f} ({res["n_clusters"]} clusters)',
                   fontsize=13, fontweight='bold')
    fig_d.tight_layout()
    fig_d.savefig(os.path.join(plot_dir, f'{dataset_key}_z{Z:.1f}_dendrogram.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_d)

    fig_n, ax_n = plt.subplots(figsize=(9, 9))
    topo.plot_network(ax=ax_n, cluster_colors=scatter_colors)
    ax_n.set_title(f'Network — Z={Z:.2f} ({res["n_clusters"]} clusters)',
                   fontsize=13, fontweight='bold')
    fig_n.tight_layout()
    fig_n.savefig(os.path.join(plot_dir, f'{dataset_key}_z{Z:.1f}_network.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_n)


# ==============================================================================
# Per-dataset driver
# ==============================================================================

def process_dataset(key, display_name, loader_fn, Z_values, max_clusters=MAX_CLUSTERS, k_max=100):
    print("=" * 70)
    print(f"Dataset: {display_name}")
    print("=" * 70)

    X, y, distances, indices, note = loader_fn()
    if note:
        print(f"  Note: {note}")
    print(f"  {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} ground-truth classes")

    t0 = time.time()
    shared = compute_shared_density(X, k_max=k_max, distances=distances, indices=indices)
    print(f"  Intrinsic dimension d = {shared['d']:.2f}  (k_max={k_max}, {time.time() - t0:.1f}s)")
    print(f"  Raw density peaks (Heuristic 1, before merging): {len(shared['centers'])}")

    print("  Computing PCA projection...")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    print(f"  Z adjustment (target range [{MIN_CLUSTERS}, {max_clusters}] clusters):")
    results = []
    for Z_target in Z_values:
        final_Z, res = adjust_Z(shared, Z_target, max_clusters=max_clusters)
        res['nmi'] = normalized_mutual_info_score(y, res['labels_full'])
        res['ari'] = adjusted_rand_score(y, res['labels_full'])

        n_halo = int(np.sum(res['halo_mask']))
        n_total = len(res['halo_mask'])
        n_core = n_total - n_halo
        max_saddle = max((info['log_density'] for info in res['saddles'].values()), default=None)
        max_saddle_str = 'n/a (no adjacent pairs)' if max_saddle is None else f'{max_saddle:.2f}'

        print(f"    Z={Z_target:.2f} (used Z={final_Z:.2f}): {res['n_clusters']} clusters | "
              f"core={n_core} ({100 * n_core / n_total:.1f}%) "
              f"halo={n_halo} ({100 * n_halo / n_total:.1f}%) | "
              f"NMI={res['nmi']:.3f} ARI={res['ari']:.3f} | "
              f"max saddle log-density={max_saddle_str}")

        results.append(res)

    # ==========================================================================
    # Composite 3x3 figure: rows = Z, columns = (scatter, dendrogram, network)
    # ==========================================================================
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.35)

    for row, res in enumerate(results):
        topo = res['topography']
        scatter_colors = unique_cluster_colors(res['n_clusters'])

        ax_scatter = fig.add_subplot(gs[row, 0])
        plot_scatter(ax_scatter, X_2d, res, scatter_colors)

        ax_dendro = fig.add_subplot(gs[row, 1])
        topo.plot_dendrogram(ax=ax_dendro)

        ax_net = fig.add_subplot(gs[row, 2])
        topo.plot_network(ax=ax_net, cluster_colors=scatter_colors)

        row_label = (f"Z = {res['Z']:.2f}\n"
                     f"({res['n_clusters']} clusters, NMI = {res['nmi']:.3f})")
        ax_scatter.annotate(row_label, xy=(-0.38, 0.5), xycoords='axes fraction',
                            rotation=90, ha='center', va='center',
                            fontsize=13, fontweight='bold')

    fig.suptitle(f'DPA Topography at Different Significance Levels — {display_name}',
                 fontsize=17, fontweight='bold', y=0.995)
    fig.savefig(os.path.join(PLOT_DIR, f'z_sweep_{key}.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Composite figure saved: plots/z_sweep_{key}.png")

    # ==========================================================================
    # Individual panels, for direct use in presentation slides
    # ==========================================================================
    for res in results:
        save_individual_panels(key, res, X_2d, y, PLOT_DIR)
        print(f"  Individual panels saved for Z={res['Z']:.2f}")

    return results


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    all_results = {}
    for key, display_name, loader_fn, Z_values, max_clusters, k_max in DATASETS:
        t0 = time.time()
        results = process_dataset(key, display_name, loader_fn, Z_values, max_clusters, k_max)
        all_results[key] = results
        print(f"  ({display_name} done in {time.time() - t0:.1f}s)\n")

    print("=" * 70)
    print("SUMMARY (all datasets)")
    print("=" * 70)
    for key, display_name, _, _, _, _ in DATASETS:
        print(f"\n{display_name}:")
        print(f"{'Z':>6} | {'Clusters':>8} | {'NMI':>6} | {'ARI':>6} | {'Halo %':>7}")
        print("-" * 48)
        for res in all_results[key]:
            n_total = len(res['halo_mask'])
            halo_pct = 100 * np.sum(res['halo_mask']) / n_total
            print(f"{res['Z']:>6.2f} | {res['n_clusters']:>8} | {res['nmi']:>6.3f} | "
                  f"{res['ari']:>6.3f} | {halo_pct:>6.1f}%")


if __name__ == "__main__":
    main()
