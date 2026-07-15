"""
DPA topography Z-sweep on SPIR2 (paper Fig. 3 reproduction): two
interleaving, non-convex spirals - the classic case where k-means and
other centroid-based methods fail but a density-based method should
succeed.

Uses the OFFICIAL SPIR2 dataset (data/spir2.csv), sourced from the
DADApy repository (examples/datasets/Fig2.dat + gt_F2.txt) - the same
data used to produce the paper's Fig. 3, not a synthetic approximation.
It includes uniform background noise (ground-truth label -1) in
addition to the two spiral arms (0, 1), which is what DPA's halo
mechanism is meant to reject.

Reuses the shared DPA pipeline (k-NN, TWO-NN, PAk, Heuristics 1-2 run
once; only Heuristic 3 re-run per Z) and the unique-per-cluster color
scheme from z_sweep_topography.py, so results are visually consistent
with the other Z-sweep figures.
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from src import load_spir2
from z_sweep_topography import (
    compute_shared_density,
    cluster_for_Z,
    unique_cluster_colors,
    PLOT_DIR,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

Z_VALUES = [1.0, 2.0, 3.0]
# Highly contrasting pair for the 2-arm ground truth (and any Z-sweep row
# that also happens to land on exactly 2 clusters).
GROUND_TRUTH_COLORS = {0: '#e6194b', 1: '#3182bd'}
NOISE_COLOR = '#c7c7c7'


def cluster_colors_for(n_clusters):
    """2 clusters -> high-contrast red/blue; otherwise the shared unique palette."""
    if n_clusters == 2:
        return dict(GROUND_TRUTH_COLORS)
    return unique_cluster_colors(n_clusters)


def plot_scatter_spir2(ax, X, res, cluster_colors, minimal_axes=False):
    """2D scatter (no PCA needed - data is already 2D): the actual clustering."""
    labels_full = res['labels_full']
    halo_mask = res['halo_mask']
    core_mask = ~halo_mask

    if halo_mask.any():
        halo_colors = [cluster_colors[c] for c in labels_full[halo_mask]]
        ax.scatter(X[halo_mask, 0], X[halo_mask, 1], c=halo_colors,
                   s=1, alpha=0.15, linewidths=0, zorder=1)

    if core_mask.any():
        core_colors = [cluster_colors[c] for c in labels_full[core_mask]]
        ax.scatter(X[core_mask, 0], X[core_mask, 1], c=core_colors,
                   s=4, alpha=0.9, linewidths=0, zorder=2)

    ax.set_title(f"DPA (Z={res['Z']:.1f}, {res['n_clusters']} clusters)",
                fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    if minimal_axes:
        # Composite-grid style: the point is the shape, not the axis scale.
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.tick_params(labelsize=8)


def save_individual_panels(res, X, plot_dir):
    """Save the three panels for one Z value separately, for use in slides."""
    topo = res['topography']
    Z = res['Z']
    cluster_colors = cluster_colors_for(res['n_clusters'])

    fig_s, ax_s = plt.subplots(figsize=(8, 8))
    plot_scatter_spir2(ax_s, X, res, cluster_colors)
    ax_s.set_title(f'SPIR2 Assignment — Z={Z:.1f} ({res["n_clusters"]} clusters)',
                   fontsize=13, fontweight='bold')
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(plot_dir, f'spir2_z{Z:.1f}_scatter.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_s)

    fig_d, ax_d = plt.subplots(figsize=(10, 7))
    topo.plot_dendrogram(ax=ax_d, show_labels=res['n_clusters'] <= 15)
    ax_d.set_title(f'Dendrogram — Z={Z:.1f} ({res["n_clusters"]} clusters)',
                   fontsize=13, fontweight='bold')
    fig_d.tight_layout()
    fig_d.savefig(os.path.join(plot_dir, f'spir2_z{Z:.1f}_dendrogram.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_d)

    fig_n, ax_n = plt.subplots(figsize=(9, 9))
    topo.plot_network(ax=ax_n, cluster_colors=cluster_colors)
    ax_n.set_title(f'Network — Z={Z:.1f} ({res["n_clusters"]} clusters)',
                   fontsize=13, fontweight='bold')
    fig_n.tight_layout()
    fig_n.savefig(os.path.join(plot_dir, f'spir2_z{Z:.1f}_network.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_n)


def main():
    print("=" * 70)
    print("DPA Z-sweep on SPIR2 (two spirals) - cf. paper Fig. 3")
    print("=" * 70)

    X, y = load_spir2(return_X_y=True)
    n_noise = int(np.sum(y == -1))
    print(f"SPIR2 (official, paper Fig. 3): {X.shape[0]} samples, 2 features, "
          f"2 ground-truth arms + {n_noise} background noise points")

    os.makedirs(PLOT_DIR, exist_ok=True)

    # ==========================================================================
    # Ground truth figure (noise in gray, arms in red/blue)
    # ==========================================================================
    fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
    noise_mask = y == -1
    if noise_mask.any():
        ax_gt.scatter(X[noise_mask, 0], X[noise_mask, 1], s=4, alpha=0.4,
                      c=NOISE_COLOR, linewidths=0, zorder=1)
    for label in (0, 1):
        mask = y == label
        ax_gt.scatter(X[mask, 0], X[mask, 1], s=6, alpha=0.6,
                      c=GROUND_TRUTH_COLORS[label], linewidths=0, zorder=2)
    ax_gt.set_title('SPIR2 — Ground Truth', fontsize=14, fontweight='bold')
    ax_gt.set_xlabel('x1')
    ax_gt.set_ylabel('x2')
    ax_gt.set_aspect('equal')
    fig_gt.tight_layout()
    fig_gt.savefig(os.path.join(PLOT_DIR, 'spir2_ground_truth.png'), dpi=300,
                   bbox_inches='tight', facecolor='white')
    plt.close(fig_gt)
    print("Ground truth figure saved: plots/spir2_ground_truth.png")

    # k_max=1000 matches the official DPA reference implementation's
    # default (github.com/mariaderrico/DPA) and its exact SPIR2 comparison
    # script (Z=3.0, k_max=1000 -> 2 clean clusters, ARI=0.963). After
    # fixing the density estimator (damped Newton-Raphson alpha correction,
    # two-point LRT, corrected epsilon formula - see src/density.py) to
    # match that reference implementation, k_max=1000 now also gives us
    # exactly 2 clusters at Z=3.0 (ARI=0.736) - previously, with the old,
    # numerically unstable density estimator, no k_max value could reach
    # a clean 2-cluster result (best was 3 clusters at k_max=500,
    # ARI=0.693). See examples/z_sweep_topography.py for the
    # general-purpose k_max=100 default used on the other three datasets.
    K_MAX = 1000
    print(f"\nComputing k-NN, TWO-NN, PAk, Heuristics 1-2 (once, Z-independent, k_max={K_MAX})...")
    t0 = time.time()
    shared = compute_shared_density(X, k_max=K_MAX)
    print(f"  Intrinsic dimension d = {shared['d']:.2f}  ({time.time() - t0:.1f}s)")
    print(f"  Raw density peaks (Heuristic 1, before merging): {len(shared['centers'])}")

    results = []
    for Z in Z_VALUES:
        t0 = time.time()
        res = cluster_for_Z(shared, Z)

        # Halo-aware comparison: recode DPA's halo points as -1 too, so the
        # metric checks both arm separation AND correct noise rejection -
        # the paper's "NMI_halo" convention for datasets with ground-truth noise.
        labels_masked = res['labels_full'].copy()
        labels_masked[res['halo_mask']] = -1
        res['nmi'] = normalized_mutual_info_score(y, labels_masked)
        res['ari'] = adjusted_rand_score(y, labels_masked)

        if noise_mask.any():
            noise_correctly_halo = np.sum(res['halo_mask'][noise_mask]) / n_noise
            res['noise_halo_rate'] = noise_correctly_halo

        n_halo = int(np.sum(res['halo_mask']))
        n_total = len(res['halo_mask'])
        noise_str = (f" | true-noise correctly halo={100 * res['noise_halo_rate']:.1f}%"
                    if 'noise_halo_rate' in res else '')
        print(f"  Z={Z:.1f}: {res['n_clusters']:3d} clusters | "
              f"core={n_total - n_halo} ({100 * (n_total - n_halo) / n_total:.1f}%) "
              f"halo={n_halo} ({100 * n_halo / n_total:.1f}%) | "
              f"NMI={res['nmi']:.3f} ARI={res['ari']:.3f}{noise_str}  ({time.time() - t0:.2f}s)")
        results.append(res)

    # ==========================================================================
    # Composite 3x3 figure: rows = Z, columns = (scatter, dendrogram, network)
    # constrained_layout, instead of manual hspace/wspace + tight_layout,
    # keeps all three columns aligned on a consistent grid across rows.
    # ==========================================================================
    fig = plt.figure(figsize=(18, 15), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.35, h_pad=0.35, wspace=0.05, hspace=0.05)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Shared normalization across rows so the same population maps to the
    # same node size in every network panel, and so cluster count alone
    # doesn't make one row's dendrogram look more/less "significant" than
    # another's on a different density scale.
    global_max_population = max(
        max(res['topography']._cluster_populations.values()) for res in results
    )

    scatter_axes, dendro_axes, net_axes = [], [], []
    for row, res in enumerate(results):
        topo = res['topography']
        cluster_colors = cluster_colors_for(res['n_clusters'])

        ax_scatter = fig.add_subplot(gs[row, 0])
        plot_scatter_spir2(ax_scatter, X, res, cluster_colors, minimal_axes=True)
        scatter_axes.append(ax_scatter)

        ax_dendro = fig.add_subplot(gs[row, 1])
        ax_dendro.set_facecolor('white')
        dendro_info = topo.plot_dendrogram(ax=ax_dendro, show_labels=res['n_clusters'] <= 15)
        dendro_axes.append((ax_dendro, dendro_info))

        ax_net = fig.add_subplot(gs[row, 2])
        ax_net.set_facecolor('white')
        topo.plot_network(ax=ax_net, cluster_colors=cluster_colors,
                          max_population=global_max_population)
        net_axes.append(ax_net)

        row_label = (f"Z = {res['Z']:.1f}\n"
                     f"{res['n_clusters']} clusters | NMI {res['nmi']:.3f}")
        ax_scatter.annotate(row_label, xy=(-0.15, 0.5), xycoords='axes fraction',
                            rotation=90, ha='center', va='center',
                            fontsize=16, fontweight='bold')

    # Align the dendrogram y-axes on a shared *density* range (not shared
    # *distance* range - each row's distance is relative to its own peak
    # density, see Topography.plot_dendrogram), so branch heights are
    # visually comparable across Z values.
    mpds = [info['max_peak_density'] for _, info in dendro_axes]
    density_bounds = [
        (mpd - ax.get_ylim()[1], mpd - ax.get_ylim()[0])
        for mpd, (ax, info) in zip(mpds, dendro_axes)
    ]
    global_dmin = min(b[0] for b in density_bounds)
    global_dmax = max(b[1] for b in density_bounds)
    for mpd, (ax_dendro, _) in zip(mpds, dendro_axes):
        ax_dendro.set_ylim(mpd - global_dmax, mpd - global_dmin)

    fig.suptitle('DPA Topography at Different Z — Two Spirals (cf. Paper Fig. 3)',
                 fontsize=17, fontweight='bold')
    fig.savefig(os.path.join(PLOT_DIR, 'z_sweep_spir2.png'), dpi=300,
                facecolor='white')
    plt.close(fig)
    print(f"\nComposite figure saved: plots/z_sweep_spir2.png")

    # ==========================================================================
    # Individual panels, for direct use in presentation slides
    # ==========================================================================
    for res in results:
        save_individual_panels(res, X, PLOT_DIR)
        print(f"  Individual panels saved for Z={res['Z']:.1f}")

    print("\nSummary:")
    print(f"{'Z':>5} | {'Clusters':>8} | {'NMI':>6} | {'ARI':>6}")
    print("-" * 35)
    for res in results:
        print(f"{res['Z']:>5.1f} | {res['n_clusters']:>8} | {res['nmi']:>6.3f} | {res['ari']:>6.3f}")


if __name__ == "__main__":
    main()
