"""
DP vs DPA comparison on SPIR2 (two interleaving spirals + background noise).

Runs classic Density Peaks (Rodriguez & Laio, 2014 - src.DensityPeaks,
automated center selection via a `percent` threshold since the paper's
original manual decision-graph selection isn't reproducible headlessly)
and DPA on the SAME full dataset (38,358 points, official Fig. 2 data -
see z_sweep_spir2.py) and plots them side by side.

DensityPeaks.fit() computes delta row-by-row (matching the official
DP/DP.py reference implementation) rather than materializing a full N x N
distance matrix, so it can run on the full spir2 dataset without running
out of memory - previously this would have needed ~11GB just to store the
matrix. See src/clustering.py DensityPeaks for the fix.

DP has no halo/noise-rejection mechanism (every point gets a hard cluster
assignment via gradient ascent to the nearest higher-density neighbor),
unlike DPA. This is deliberately left as-is in the comparison, since it is
one of the two headline improvements DPA makes over DP (the other being
automatic vs. manual center selection).
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from src import load_spir2, DensityPeaks
from z_sweep_topography import compute_shared_density, cluster_for_Z, unique_cluster_colors, PLOT_DIR
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Best of the percent sweep on the full 38,358-point dataset (see
# conversation/session log): 0.01/0.02/0.05/0.1/0.2/0.5/1.0/2.0 -> ARI
# peaks at percent=0.02 (7 clusters, ARI=0.181); NMI peaks a bit higher
# (percent=0.05, 19 clusters, NMI=0.338) but ARI is the standard model-
# selection criterion used everywhere else in this repo (run_dp,
# analyze_Z_sensitivity_fast), so percent=0.02 is used here for consistency.
DP_PERCENT = 0.02
DP_K = 20
DPA_Z = 3.0
GROUND_TRUTH_COLORS = {0: '#e6194b', 1: '#3182bd'}
NOISE_COLOR = '#c7c7c7'


BG_S, BG_ALPHA = 1, 0.15
CORE_S, CORE_ALPHA = 4, 0.9


def _style_axes(ax):
    """Shared per-panel cleanup: equal aspect, white face, minimal frame."""
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_ground_truth(ax, X, y):
    noise_mask = y == -1
    if noise_mask.any():
        ax.scatter(X[noise_mask, 0], X[noise_mask, 1], s=BG_S, alpha=BG_ALPHA,
                   c=NOISE_COLOR, linewidths=0, zorder=1)
    for label in (0, 1):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], s=CORE_S, alpha=CORE_ALPHA,
                   c=GROUND_TRUTH_COLORS[label], linewidths=0, zorder=2)
    ax.set_title('Ground Truth\n(2 arms + noise)', fontsize=13, fontweight='bold')
    _style_axes(ax)


def plot_dpa(ax, X, res, ari, nmi):
    n_clusters = res['n_clusters']
    cluster_colors = (dict(GROUND_TRUTH_COLORS) if n_clusters == 2
                       else unique_cluster_colors(n_clusters))
    labels_full = res['labels_full']
    halo_mask = res['halo_mask']
    core_mask = ~halo_mask

    if halo_mask.any():
        halo_colors = [cluster_colors[c] for c in labels_full[halo_mask]]
        ax.scatter(X[halo_mask, 0], X[halo_mask, 1], c=halo_colors,
                   s=BG_S, alpha=BG_ALPHA, linewidths=0, zorder=1)
    if core_mask.any():
        core_colors = [cluster_colors[c] for c in labels_full[core_mask]]
        ax.scatter(X[core_mask, 0], X[core_mask, 1], c=core_colors,
                   s=CORE_S, alpha=CORE_ALPHA, linewidths=0, zorder=2)

    n_halo_pct = 100 * np.sum(halo_mask) / len(halo_mask)
    # % applied to an isolated raw string, THEN concatenated with +, so a
    # literal '%' elsewhere in the title (e.g. "halo=25%") can never be
    # mistaken for a format directive by the % operator - adjacent string
    # literals (f'...' r'...') concatenate BEFORE % is applied, which
    # previously fed the whole combined string through %-formatting.
    metrics = r'$\mathbf{ARI=%.3f\ \ NMI=%.3f}$' % (ari, nmi)
    ax.set_title(
        f'DPA (Z={DPA_Z:.1f}, {n_clusters} clusters)\n' + metrics +
        f'   halo={n_halo_pct:.0f}%',
        fontsize=13)
    _style_axes(ax)


def plot_dp(ax, X, labels, n_clusters, ari, nmi):
    cluster_colors = unique_cluster_colors(n_clusters)
    colors = [cluster_colors[c] for c in labels]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=CORE_S, alpha=CORE_ALPHA,
              linewidths=0, zorder=2)

    metrics = r'$\mathbf{ARI=%.3f\ \ NMI=%.3f}$' % (ari, nmi)
    ax.set_title(
        f'DP (percent={DP_PERCENT}%, {n_clusters} clusters)\n' + metrics +
        '   no halo/noise rejection',
        fontsize=13)
    _style_axes(ax)


def main():
    print("=" * 70)
    print("DP vs DPA comparison on SPIR2 (full 38,358-point dataset)")
    print("=" * 70)

    X, y = load_spir2(return_X_y=True)
    n_noise = int(np.sum(y == -1))
    print(f"SPIR2: {X.shape[0]} samples, 2 ground-truth arms + {n_noise} noise points")

    os.makedirs(PLOT_DIR, exist_ok=True)

    # ---- DPA (Z=3.0, halo-aware scoring - same convention as z_sweep_spir2.py) ----
    print(f"\nRunning DPA (Z={DPA_Z})...")
    t0 = time.time()
    shared = compute_shared_density(X, k_max=1000)
    res = cluster_for_Z(shared, DPA_Z)
    labels_masked = res['labels_full'].copy()
    labels_masked[res['halo_mask']] = -1
    dpa_ari = adjusted_rand_score(y, labels_masked)
    dpa_nmi = normalized_mutual_info_score(y, labels_masked)
    print(f"  {res['n_clusters']} clusters, ARI={dpa_ari:.3f}, NMI={dpa_nmi:.3f} "
          f"({time.time() - t0:.1f}s)")

    # ---- DP (percent=0.02, best of the sweep - see module docstring) ----
    print(f"\nRunning DP (percent={DP_PERCENT}, k={DP_K})...")
    t0 = time.time()
    dp = DensityPeaks(k=DP_K, percent=DP_PERCENT)
    dp.fit(X)
    dp_ari = adjusted_rand_score(y, dp.labels_)
    dp_nmi = normalized_mutual_info_score(y, dp.labels_)
    print(f"  {dp.n_clusters_} clusters, ARI={dp_ari:.3f}, NMI={dp_nmi:.3f} "
          f"({time.time() - t0:.1f}s)")

    # ---- Comparison figure ----
    # constrained_layout keeps the three equal-aspect panels aligned on one
    # horizontal baseline instead of drifting per-panel with tight_layout.
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.4, h_pad=0.3, wspace=0.05)
    plot_ground_truth(axes[0], X, y)
    plot_dpa(axes[1], X, res, dpa_ari, dpa_nmi)
    plot_dp(axes[2], X, dp.labels_, dp.n_clusters_, dp_ari, dp_nmi)

    fig.suptitle('DP vs DPA on SPIR2 (two spirals + noise, full 38,358-point dataset)',
                fontsize=16, fontweight='bold')
    out_path = os.path.join(PLOT_DIR, 'compare_dp_dpa_spir2.png')
    fig.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)
    print(f"\nComparison figure saved: plots/compare_dp_dpa_spir2.png")

    print("\nSummary:")
    print(f"{'Method':<8} | {'Clusters':>8} | {'ARI':>6} | {'NMI':>6}")
    print("-" * 38)
    print(f"{'DPA':<8} | {res['n_clusters']:>8} | {dpa_ari:>6.3f} | {dpa_nmi:>6.3f}")
    print(f"{'DP':<8} | {dp.n_clusters_:>8} | {dp_ari:>6.3f} | {dp_nmi:>6.3f}")


if __name__ == "__main__":
    main()
