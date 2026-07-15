"""
Diagnostic: does our halo criterion diverge from the official reference
implementation's find_halos (src/Pipeline/_DPA.pyx in
github.com/mariaderrico/DPA) because of a sign/sentinel bug in the
official code?

Official find_halos:
    point i in cluster c is halo if:
        densities[i] < min_rho_bord[c]  AND  min_rho_bord[c] > 0

"min_rho_bord[c]" is actually the MAX border (saddle) log-density of
cluster c (misleading name - it's updated via `if densities[i] >
min_rho_bord[c]: min_rho_bord[c] = densities[i]`, starting from 0.0).
Since `densities` is a LOG-density (can be negative), if every saddle
touching cluster c has a NEGATIVE log-density, min_rho_bord[c] never
rises above its 0.0 initialization - so the `> 0` guard silently
disables halo detection for that cluster entirely, even though it does
have a genuine (negative) border density. This is the same class of bug
we found and fixed in Topography.plot_network (rho_bord=0 used as a
"no saddle" sentinel when the real value can be negative).

Our identify_halo_points (src/clustering.py) avoids this: it uses
-np.inf as the "no border" sentinel and np.isfinite() to test for it,
so a genuine negative border density is NOT mistaken for "no border".

This script checks, for SPIR2 (Z=3.0) and Pendigits (Z=3.0):
  1. Whether any final cluster's max border log-density is negative
     (i.e. would trigger the official code's sentinel bug).
  2. What halo_mask we'd get if we replicated the official condition
     bug-for-bug (`max_border > 0 and log_density[i] < max_border`)
     instead of our correct one (`np.isfinite(max_border) and ...`).
  3. How much this changes core/halo counts and (for spir2, which has a
     known official-code ARI=0.963 benchmark at this exact Z/k_max) ARI/NMI.
"""

import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from src import load_spir2, load_pendigits
from src.clustering import identify_halo_points
from z_sweep_topography import compute_shared_density, cluster_for_Z
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def official_style_halo_mask(labels_full, log_density, cluster_max_border_density):
    """Replicate _DPA.pyx find_halos exactly, including its `> 0` sentinel bug."""
    mask = np.zeros(len(labels_full), dtype=bool)
    for i in range(len(labels_full)):
        c = labels_full[i]
        if c < 0:
            continue
        max_border = cluster_max_border_density.get(c, -np.inf)
        if max_border > 0 and log_density[i] < max_border:
            mask[i] = True
    return mask


def diagnose(name, X, y, Z, k_max, halo_aware_scoring):
    print("=" * 70)
    print(f"{name}  (Z={Z}, k_max={k_max})")
    print("=" * 70)

    t0 = time.time()
    shared = compute_shared_density(X, k_max=k_max)
    res = cluster_for_Z(shared, Z)
    print(f"  shared density + clustering: {time.time() - t0:.1f}s, "
          f"{res['n_clusters']} clusters")

    log_density = shared['log_density']
    labels_full = res['labels_full']
    _, cluster_max_border_density = identify_halo_points(
        labels_full, log_density, res['saddles'], res['cluster_centers']
    )

    print(f"\n  Per-cluster max border log-density:")
    n_negative = 0
    for c in range(res['n_clusters']):
        max_border = cluster_max_border_density.get(c, -np.inf)
        pop = int(np.sum(labels_full == c))
        flag = ""
        if np.isfinite(max_border) and max_border <= 0:
            flag = "  <-- NEGATIVE: official code's '>0' sentinel would silently skip halo detection here"
            n_negative += 1
        elif not np.isfinite(max_border):
            flag = "  (isolated cluster, no border)"
        print(f"    C{c:<3d} pop={pop:6d}  max_border_log_density={max_border:8.3f}{flag}")

    print(f"\n  {n_negative}/{res['n_clusters']} clusters have a NEGATIVE max border "
          f"log-density (would trigger the official sentinel bug)")

    our_halo = res['halo_mask']
    official_halo = official_style_halo_mask(labels_full, log_density, cluster_max_border_density)

    n_total = len(labels_full)
    print(f"\n  Our halo:      {np.sum(our_halo):6d} / {n_total} ({100*np.sum(our_halo)/n_total:.1f}%)")
    print(f"  Official-style halo (bug-for-bug): {np.sum(official_halo):6d} / {n_total} "
          f"({100*np.sum(official_halo)/n_total:.1f}%)")
    print(f"  Difference: {np.sum(our_halo) - np.sum(official_halo):6d} points "
          f"only halo under OUR (correct) criterion")

    if halo_aware_scoring:
        our_labels_masked = labels_full.copy()
        our_labels_masked[our_halo] = -1
        official_labels_masked = labels_full.copy()
        official_labels_masked[official_halo] = -1

        our_ari = adjusted_rand_score(y, our_labels_masked)
        our_nmi = normalized_mutual_info_score(y, our_labels_masked)
        off_ari = adjusted_rand_score(y, official_labels_masked)
        off_nmi = normalized_mutual_info_score(y, official_labels_masked)

        print(f"\n  ARI/NMI with OUR halo criterion:            ARI={our_ari:.3f}  NMI={our_nmi:.3f}")
        print(f"  ARI/NMI with OFFICIAL-style (buggy) halo:   ARI={off_ari:.3f}  NMI={off_nmi:.3f}")
    else:
        our_ari = adjusted_rand_score(y, labels_full)
        our_nmi = normalized_mutual_info_score(y, labels_full)
        print(f"\n  ARI/NMI (non-halo-aware scoring, as used for this dataset): "
              f"ARI={our_ari:.3f}  NMI={our_nmi:.3f}  (unaffected by halo criterion)")

    print()


def main():
    X_spir2, y_spir2 = load_spir2(return_X_y=True)
    diagnose("SPIR2", X_spir2, y_spir2, Z=3.0, k_max=1000, halo_aware_scoring=True)

    X_pen, y_pen = load_pendigits(return_X_y=True)
    diagnose("Pendigits", X_pen, y_pen, Z=3.0, k_max=500, halo_aware_scoring=False)


if __name__ == "__main__":
    main()
