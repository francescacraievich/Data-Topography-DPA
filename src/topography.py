"""
Topography visualization for DPA clustering.

Provides visualization of the density landscape including:
- Topography matrix (peaks and saddles)
- Dendrogram representation (hierarchical clustering style)
- Network representation (MDS layout on saddle-density dissimilarity,
  matching the official DPA reference implementation)
- Decision graph (classic Density Peaks)

Based on: d'Errico, M., Facco, E., Laio, A., & Rodriguez, A. (2021).
"Automatic topography of high-dimensional data sets by non-parametric
density peak clustering" Information Sciences, 560, 476-492.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class Topography:
    """
    Topographic representation of clustering results.

    Creates visualizations showing the "landscape" of the data:
    - Cluster centers as peaks with heights = log(rho)
    - Saddle points as passes between peaks
    - Hierarchical relationships through dendrogram
    - Connectivity through network representation

    Parameters
    ----------
    centers : ndarray
        Indices of cluster centers.

    saddles : dict
        Saddle information {(c1, c2): {'log_density': ..., 'index': ...}}.

    log_density : ndarray
        Log-density for all points.

    labels : ndarray
        Cluster labels for all points.

    Attributes
    ----------
    matrix_ : ndarray of shape (n_clusters, n_clusters)
        Topography matrix where:
        - diagonal[c] = log(rho_c) (peak height)
        - off-diagonal[c,c'] = log(rho_{cc'}) (saddle height)
        - -inf if clusters are not adjacent

    n_clusters_ : int
        Number of clusters.
    """

    def __init__(self, centers, saddles, log_density, labels):
        self.centers = np.asarray(centers)
        self.saddles = saddles
        self.log_density = np.asarray(log_density)
        self.labels = np.asarray(labels)

        self.n_clusters_ = len(centers)
        self.matrix_ = self._build_matrix()
        self._cluster_populations = self._compute_populations()

    def _build_matrix(self):
        """Build the topography matrix."""
        n = self.n_clusters_
        matrix = np.full((n, n), -np.inf)

        # Diagonal: peak heights
        for c in range(n):
            matrix[c, c] = self.log_density[self.centers[c]]

        # Off-diagonal: saddle heights
        for pair, info in self.saddles.items():
            c1, c2 = pair
            if c1 < n and c2 < n:
                saddle_density = info['log_density']
                matrix[c1, c2] = saddle_density
                matrix[c2, c1] = saddle_density

        return matrix

    def _compute_populations(self):
        """Compute cluster populations."""
        populations = {}
        for c in range(self.n_clusters_):
            populations[c] = np.sum(self.labels == c)
        return populations

    def _majority_rule_colors(self, ground_truth, digit_colors=None):
        """
        Assign each cluster the color of its majority ground-truth label.

        Paper Section 2.1.5: "the color is the one corresponding to the
        digit label with higher presence in the cluster".

        Parameters
        ----------
        ground_truth : ndarray of shape (n_samples,)
            True labels for all points (same length/order as self.labels).

        digit_colors : dict, optional
            Mapping {label: color}. If None, built from the 'tab10' colormap
            over the sorted unique labels in ground_truth.

        Returns
        -------
        cluster_colors : dict
            {cluster_id: color}
        digit_colors : dict
            The label -> color mapping used (echoed back for reuse/legends).
        """
        ground_truth = np.asarray(ground_truth)
        unique_labels = np.unique(ground_truth)

        if digit_colors is None:
            cmap = plt.get_cmap('tab10')
            digit_colors = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}

        cluster_colors = {}
        for c in range(self.n_clusters_):
            members = ground_truth[self.labels == c]
            if len(members) == 0:
                cluster_colors[c] = (0.6, 0.6, 0.6, 1.0)
                continue
            values, counts = np.unique(members, return_counts=True)
            majority_label = values[np.argmax(counts)]
            cluster_colors[c] = digit_colors.get(majority_label, (0.6, 0.6, 0.6, 1.0))

        return cluster_colors, digit_colors

    def get_linkage_matrix(self):
        """
        Build linkage matrix for dendrogram.

        Uses distance: d_{cc'} = max(log(rho)) - log(rho_{cc'})
        Higher saddle = smaller distance = merge earlier
        """
        n = self.n_clusters_

        if n < 2:
            return None

        # Maximum and minimum peak heights
        peak_densities = np.diag(self.matrix_)
        max_density = np.max(peak_densities)
        min_density = np.min(peak_densities)

        # Handle edge case where densities are not finite
        if not np.isfinite(max_density):
            max_density = 0.0
        if not np.isfinite(min_density):
            min_density = max_density - 10.0

        # First pass: compute distances for connected clusters
        dist_matrix = np.zeros((n, n))
        max_connected_dist = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                saddle = self.matrix_[i, j]
                if np.isfinite(saddle):
                    # Distance = max density - saddle density
                    # Higher saddle = smaller distance = merge earlier
                    dist = max_density - saddle
                    dist = max(0.001, dist) if np.isfinite(dist) else 1.0
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                    max_connected_dist = max(max_connected_dist, dist)

        # Second pass: set disconnected clusters to merge LAST
        # Use distance larger than any connected pair
        disconnected_dist = max_connected_dist * 1.5 + 1.0 if max_connected_dist > 0 else 10.0

        for i in range(n):
            for j in range(i + 1, n):
                saddle = self.matrix_[i, j]
                if not np.isfinite(saddle):
                    dist_matrix[i, j] = disconnected_dist
                    dist_matrix[j, i] = disconnected_dist

        # Ensure no NaN or inf values
        dist_matrix = np.nan_to_num(dist_matrix, nan=disconnected_dist,
                                     posinf=disconnected_dist, neginf=0.001)

        # Convert to condensed form and compute linkage
        try:
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method='single')
            return Z
        except Exception:
            # Fallback: return None if linkage fails
            return None

    def plot_dendrogram(self, ax=None, show_labels=True, ground_truth=None,
                        digit_colors=None, **kwargs):
        """
        Plot dendrogram showing cluster hierarchy, in the paper's style
        (Fig. 1E / Fig. 4): peaks at the top, branches descending to the
        saddle at which they merge, root (lowest saddle) at the bottom.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates new figure if None.

        show_labels : bool, default=True
            Whether to show cluster labels.

        ground_truth : ndarray of shape (n_samples,), optional
            True labels for all points. If provided, each leaf is colored
            by majority rule (Section 2.1.5): the color of the ground-truth
            label with the highest presence in that cluster.

        digit_colors : dict, optional
            Mapping {label: color} used for majority-rule coloring. Built
            automatically from 'tab10' if not provided.

        **kwargs : dict
            Additional arguments passed to scipy dendrogram.

        Returns
        -------
        dict
            Dendrogram dictionary from scipy (plus 'cluster_colors' and
            'digit_colors' if ground_truth was provided), or None if it
            cannot be plotted.
        """
        if self.n_clusters_ < 2:
            if ax is not None:
                ax.text(0.5, 0.5, f'Only {self.n_clusters_} cluster(s)\nDendrogram requires >= 2',
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title('Dendrogram (N/A)')
            return None

        Z = self.get_linkage_matrix()
        if Z is None:
            if ax is not None:
                ax.text(0.5, 0.5, 'Could not build linkage matrix',
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title('Dendrogram (Error)')
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plain cluster-number labels only - no color, no population.
        if show_labels:
            labels = [f'C{i}' for i in range(self.n_clusters_)]
        else:
            labels = None

        cluster_colors = None
        used_digit_colors = digit_colors
        if ground_truth is not None:
            cluster_colors, used_digit_colors = self._majority_rule_colors(
                ground_truth, digit_colors)

        kwargs.setdefault('orientation', 'bottom')
        # Paper style (Fig. 4): plain black tree.
        kwargs.setdefault('color_threshold', 0)
        kwargs.setdefault('above_threshold_color', 'black')
        if not show_labels:
            # labels=None alone still falls back to scipy's default numeric
            # leaf labels unless no_labels is set - needed to fully clear
            # the leaf row when there are too many clusters to read (e.g.
            # 30+), not just to drop the custom "C0, C1, ..." text.
            kwargs.setdefault('no_labels', True)

        try:
            dendro = dendrogram(Z, ax=ax, labels=labels, **kwargs)

            # Plain, uncolored leaf ticks. Font shrinks as more leaves
            # compete for the same width.
            leaf_fontsize = max(5, min(8, 220 / max(self.n_clusters_, 1)))
            for tick in ax.get_xticklabels():
                tick.set_fontsize(leaf_fontsize)

            # y-axis: show actual log(density) instead of raw merge distance.
            # dist = max_peak_density - saddle_density, so density = max - dist.
            peak_heights = np.diag(self.matrix_)
            finite_peaks = peak_heights[np.isfinite(peak_heights)]
            max_peak_density = np.max(finite_peaks) if len(finite_peaks) > 0 else 0.0

            def _to_density(y, _pos):
                return f'{max_peak_density - y:.1f}'

            ax.yaxis.set_major_formatter(plt.FuncFormatter(_to_density))

            ax.set_xlabel('Cluster')
            ax.set_ylabel('log(ρ)')
            ax.set_title('Cluster Topography Dendrogram')

            if cluster_colors is not None:
                dendro['cluster_colors'] = cluster_colors
                dendro['digit_colors'] = used_digit_colors
            # Exposed so callers can align the y-axis (density) range across
            # several dendrograms plotted at different Z - each has its own
            # raw distance scale (dist = max_peak_density - saddle_density),
            # so a shared *distance* ylim would not correspond to a shared
            # *density* range unless max_peak_density is the same everywhere.
            dendro['max_peak_density'] = max_peak_density

            return dendro
        except Exception as e:
            ax.text(0.5, 0.5, f'Dendrogram error:\n{str(e)[:50]}',
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.set_title('Dendrogram (Error)')
            return None

    def plot_network(self, ax=None, layout='mds', node_scale=1000,
                     edge_scale=2.0, cmap='viridis', ground_truth=None,
                     digit_colors=None, cluster_colors=None, show_edge_labels=False,
                     random_state=42, max_population=None, edge_color='#999999'):
        """
        Plot network representation of the topography, following the
        construction used in the official DPA code
        (Examples/auxplot.py::plots_topography in
        https://github.com/mariaderrico/DPA):

        - Nodes are placed with MDS on the dissimilarity matrix
          d[c, c'] = max(log_rho_peaks) - log_rho_saddle[c, c'], using 0
          (i.e. maximum dissimilarity) for pairs with no confirmed saddle.
          Node position therefore reflects density-topography distance,
          not adjacency in the original PCA/2D scatter.
        - A line segment is drawn for every pair of clusters, with width
          proportional to the saddle density - pairs without a saddle get
          width 0 (invisible) rather than being omitted outright.
        - Node size is proportional to sqrt(population), as in the paper.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        layout : str, default='mds'
            Layout algorithm. 'mds' reproduces the paper (see above);
            'circular' and 'random' are purely cosmetic fallbacks that
            ignore saddle densities.

        node_scale : float, default=1000
            Scale factor for node sizes.

        edge_scale : float, default=2.0
            Scale factor for edge widths.

        cmap : str, default='viridis'
            Colormap for nodes. Ignored if cluster_colors or ground_truth
            is provided.

        ground_truth : ndarray of shape (n_samples,), optional
            True labels for all points. If provided (and cluster_colors is
            not), nodes are colored by majority rule (Section 2.1.5)
            instead of by density.

        digit_colors : dict, optional
            Mapping {label: color} used for majority-rule coloring. Built
            automatically from 'tab10' if not provided.

        cluster_colors : dict, optional
            Explicit mapping {cluster_id: color}. Takes precedence over
            ground_truth/majority-rule and density coloring - use this to
            keep node colors identical to another plot (e.g. a PCA scatter
            using a distinct-per-cluster colormap rather than majority rule).

        show_edge_labels : bool, default=False
            Annotate edges with their saddle log-density, if there are
            fewer than 15 connected edges (otherwise labels would
            overlap/clutter).

        random_state : int, default=42
            Random state for the MDS solver.

        max_population : int or float, optional
            Population used to normalize node sizes, in place of this
            topography's own largest cluster. Pass the same value across
            several plot_network calls (e.g. one per Z in a sweep) so node
            size is comparable across plots instead of each one rescaling
            to its own local maximum.

        edge_color : str, default='#999999'
            Color for the saddle-density edges (light gray by default, so
            they read clearly on a white panel background).

        Returns
        -------
        dict
            Network information: 'pos' ({cluster_id: (x, y)}), 'rho_bord'
            (the saddle-density matrix used for edges/layout), and
            'cluster_colors'/'digit_colors' if colors were computed.
        """
        from matplotlib.collections import LineCollection
        from sklearn.manifold import MDS

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        n = self.n_clusters_

        if n < 2:
            ax.text(0.5, 0.5, f'Only {n} cluster(s)\nNetwork requires >= 2',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Cluster Network (N/A)')
            ax.axis('off')
            return None

        # Rho_bord[c, c']: saddle log-density if a saddle was confirmed
        # between c and c', else NaN. The paper's Rho_bord (a plain,
        # always-positive density) defaults to 0 for "no saddle" - but
        # ours is a log-density and can be negative, so 0 cannot be used
        # as a sentinel (it could rank as "closer" than a genuine but very
        # negative saddle). `connected` tracks which pairs actually have a
        # saddle, independent of the sign of its value.
        rho_bord = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(self.matrix_[i, j]):
                    rho_bord[i, j] = self.matrix_[i, j]
        connected = np.isfinite(rho_bord)

        peak_heights = np.diag(self.matrix_)
        finite_peaks = peak_heights[np.isfinite(peak_heights)]
        f_max = np.max(finite_peaks) if len(finite_peaks) > 0 else 0.0

        # Dissimilarity: d[c,c'] = f_max - saddle_density for connected
        # pairs. Unconnected pairs get a dissimilarity larger than any
        # connected pair's, pushing them away from everything else in the
        # MDS embedding (same sentinel pattern as get_linkage_matrix).
        finite_saddles = rho_bord[connected]
        if finite_saddles.size > 0:
            max_connected_dist = f_max - finite_saddles.min()
        else:
            max_connected_dist = 0.0
        disconnected_dist = max_connected_dist * 1.5 + 1.0 if max_connected_dist > 0 else 10.0

        dissimilarity = np.where(connected, f_max - np.nan_to_num(rho_bord), disconnected_dist)
        np.fill_diagonal(dissimilarity, 0.0)

        if layout == 'mds':
            mds = MDS(n_components=2, dissimilarity='precomputed',
                     random_state=random_state, normalized_stress=False)
            coords = mds.fit_transform(dissimilarity)
            pos = {c: coords[c] for c in range(n)}
        elif layout == 'circular':
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pos = {c: np.array([np.cos(a), np.sin(a)]) for c, a in enumerate(angles)}
        else:
            rng = np.random.RandomState(random_state)
            pos = {c: rng.rand(2) for c in range(n)}

        # Node sizes ~ sqrt(population), as in the paper. Normalizing
        # against max_population (if given) instead of this plot's own max
        # keeps sizes comparable across a set of plots (e.g. a Z sweep).
        populations = np.array([self._cluster_populations[c] for c in range(n)], dtype=float)
        max_pop = max_population if max_population is not None else populations.max()
        max_pop = max_pop if max_pop and max_pop > 0 else 1.0
        node_sizes = np.sqrt(populations / max_pop) * node_scale * 0.6 + 50

        # Node colors: explicit cluster_colors > majority-rule (ground truth) > density
        used_digit_colors = digit_colors
        majority_colors = None
        if cluster_colors is not None:
            node_color = [cluster_colors[c] for c in range(n)]
            node_cmap = None
        elif ground_truth is not None:
            majority_colors, used_digit_colors = self._majority_rule_colors(
                ground_truth, digit_colors)
            node_color = [majority_colors[c] for c in range(n)]
            node_cmap = None
        else:
            node_color = [self.log_density[self.centers[c]] for c in range(n)]
            node_cmap = cmap

        # One segment per connected pair (i<j): width proportional to
        # saddle density. Pairs without a confirmed saddle are skipped
        # outright - a matplotlib LineCollection drawn with linewidth=0
        # still rasterizes a hairline (Agg draws zero-width strokes as 1
        # device pixel), so relying on width alone to hide unconnected
        # pairs would fill the plot with overlapping "invisible" edges
        # once there are many clusters.
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if connected[i, j]]
        weights = np.array([rho_bord[i, j] for i, j in pairs])
        n_connected = len(pairs)

        if len(pairs) > 0:
            min_w, max_w = weights.min(), weights.max()
            if max_w > min_w:
                linewidths = (weights - min_w) / (max_w - min_w) * edge_scale + 0.3
            else:
                linewidths = np.full(len(weights), edge_scale * 0.5 + 0.3)
            segments = [[pos[i], pos[j]] for i, j in pairs]
            lc = LineCollection(segments, linewidths=linewidths, colors=edge_color,
                                alpha=0.7, zorder=1)
            ax.add_collection(lc)

            if show_edge_labels and n_connected < 15:
                for (i, j), w in zip(pairs, weights):
                    mid = (pos[i] + pos[j]) / 2
                    ax.annotate(f'{w:.1f}', mid, fontsize=7, color='dimgray',
                               ha='center', va='center',
                               bbox=dict(facecolor='white', edgecolor='none',
                                         alpha=0.7, pad=0.5))

        # Draw nodes - plain circles, no text labels.
        xs = [pos[c][0] for c in range(n)]
        ys = [pos[c][1] for c in range(n)]
        nodes = ax.scatter(xs, ys, s=node_sizes, c=node_color, cmap=node_cmap,
                           edgecolors='black', linewidths=0.5, zorder=2)

        if node_cmap is not None:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
            cbar.set_label('Log Density', fontsize=9)

        # Add padding to ensure nodes aren't cut off
        x_range = max(xs) - min(xs) if max(xs) != min(xs) else 1.0
        y_range = max(ys) - min(ys) if max(ys) != min(ys) else 1.0
        x_pad = max(0.15 * x_range, 1e-6)
        y_pad = max(0.15 * y_range, 1e-6)
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

        # Title with connection info
        n_possible = n * (n - 1) // 2
        title = f'Cluster Network ({n_connected}/{n_possible} connections)'
        if n_connected == 0:
            title += '\n(Clusters well-separated)'
        ax.set_title(title, fontsize=10)
        ax.axis('off')

        result = {'pos': pos, 'rho_bord': rho_bord}
        if cluster_colors is not None:
            result['cluster_colors'] = cluster_colors
        elif majority_colors is not None:
            result['cluster_colors'] = majority_colors
            result['digit_colors'] = used_digit_colors
        return result

    def plot_decision_graph(self, g, delta, ax=None, centers=None, **kwargs):
        """
        Plot classic Density Peaks decision graph.

        Parameters
        ----------
        g : ndarray
            Error-adjusted log-density (or regular density).

        delta : ndarray
            Delta values (min distance to higher-g point).

        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        centers : ndarray, optional
            Center indices to highlight.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(g, delta, alpha=0.5, **kwargs)

        if centers is not None:
            ax.scatter(g[centers], delta[centers], c='red', s=100,
                       marker='*', label='Centers', zorder=5)
            ax.legend()

        ax.set_xlabel('g = log(ρ) - ε')
        ax.set_ylabel('δ (min distance to higher-g point)')
        ax.set_title('Decision Graph')

        return ax

    def plot_topography_matrix(self, ax=None, cmap='RdYlBu_r', annot=True):
        """
        Plot the topography matrix as a heatmap.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        cmap : str, default='RdYlBu_r'
            Colormap.

        annot : bool, default=True
            Whether to annotate cells with values.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Replace -inf with NaN for visualization
        matrix_viz = self.matrix_.copy()
        matrix_viz[np.isinf(matrix_viz)] = np.nan

        im = ax.imshow(matrix_viz, cmap=cmap)
        plt.colorbar(im, ax=ax, label='Log Density')

        # Add annotations
        if annot:
            for i in range(self.n_clusters_):
                for j in range(self.n_clusters_):
                    val = self.matrix_[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                                fontsize=8)

        ax.set_xticks(range(self.n_clusters_))
        ax.set_yticks(range(self.n_clusters_))
        ax.set_xticklabels([f'C{i}' for i in range(self.n_clusters_)])
        ax.set_yticklabels([f'C{i}' for i in range(self.n_clusters_)])
        ax.set_title('Topography Matrix\n(diagonal=peaks, off-diagonal=saddles)')

        return ax

    def plot_confusion_matrix(self, ground_truth, ax=None, cmap='Greys'):
        """
        Plot the cluster-vs-ground-truth purity matrix, as in the paper's
        Fig. 4: for each cluster, the fraction of its members belonging to
        each ground-truth label. Darker = higher fraction.

        Parameters
        ----------
        ground_truth : ndarray of shape (n_samples,)
            True labels for all points.

        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        cmap : str, default='Greys'
            Colormap (grey palette, as in the paper).

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ground_truth = np.asarray(ground_truth)
        unique_labels = np.unique(ground_truth)
        n_labels = len(unique_labels)

        matrix = np.zeros((self.n_clusters_, n_labels))
        for c in range(self.n_clusters_):
            members = ground_truth[self.labels == c]
            pop = len(members)
            if pop == 0:
                continue
            for j, lab in enumerate(unique_labels):
                matrix[c, j] = np.sum(members == lab) / pop

        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Fraction of cluster', fontsize=9)

        if self.n_clusters_ < 15:
            for c in range(self.n_clusters_):
                for j in range(n_labels):
                    val = matrix[c, j]
                    if val > 0:
                        text_color = 'white' if val > 0.5 else 'black'
                        ax.text(j, c, f'{val:.2f}', ha='center', va='center',
                                fontsize=7, color=text_color)

        ax.set_xticks(range(n_labels))
        ax.set_xticklabels([str(lab) for lab in unique_labels])
        ax.set_yticks(range(self.n_clusters_))
        ax.set_yticklabels([f'C{c} ({self._cluster_populations[c]})'
                            for c in range(self.n_clusters_)], fontsize=7)
        ax.set_xlabel('Ground truth label')
        ax.set_ylabel('Cluster')
        ax.set_title('Cluster Purity Matrix')

        return ax

    def get_merge_sequence(self):
        """
        Get sequence of cluster merges from topography.

        Returns list of tuples (c1, c2, saddle_density) ordered by
        decreasing saddle density (order of merging).
        """
        merges = []
        for pair, info in self.saddles.items():
            c1, c2 = pair
            if c1 < self.n_clusters_ and c2 < self.n_clusters_:
                merges.append((c1, c2, info['log_density']))

        # Sort by decreasing saddle density
        merges.sort(key=lambda x: -x[2])
        return merges

    def summary(self):
        """
        Get summary of topography.

        Returns
        -------
        dict
            Summary statistics.
        """
        peak_heights = np.diag(self.matrix_)
        saddle_heights = self.matrix_[np.triu_indices(self.n_clusters_, k=1)]
        saddle_heights = saddle_heights[np.isfinite(saddle_heights)]

        return {
            'n_clusters': self.n_clusters_,
            'peak_heights': {
                'min': np.min(peak_heights),
                'max': np.max(peak_heights),
                'mean': np.mean(peak_heights)
            },
            'saddle_heights': {
                'n_connections': len(saddle_heights),
                'min': np.min(saddle_heights) if len(saddle_heights) > 0 else None,
                'max': np.max(saddle_heights) if len(saddle_heights) > 0 else None,
                'mean': np.mean(saddle_heights) if len(saddle_heights) > 0 else None
            },
            'populations': self._cluster_populations
        }
