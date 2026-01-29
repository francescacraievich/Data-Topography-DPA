"""
Topography visualization for DPA clustering.

Provides visualization of the density landscape including:
- Topography matrix (peaks and saddles)
- Dendrogram representation (hierarchical clustering style)
- Network representation (Markov State Model style)
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

    def plot_dendrogram(self, ax=None, show_labels=True, **kwargs):
        """
        Plot dendrogram showing cluster hierarchy.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates new figure if None.

        show_labels : bool, default=True
            Whether to show cluster labels.

        **kwargs : dict
            Additional arguments passed to scipy dendrogram.

        Returns
        -------
        dict
            Dendrogram dictionary from scipy, or None if cannot be plotted.
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

        # Create labels with population info
        if show_labels:
            labels = [f'C{i}\n({self._cluster_populations[i]})'
                      for i in range(self.n_clusters_)]
        else:
            labels = None

        try:
            dendro = dendrogram(Z, ax=ax, labels=labels, **kwargs)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Distance (max density - saddle density)')
            ax.set_title('Cluster Topography Dendrogram')
            return dendro
        except Exception as e:
            ax.text(0.5, 0.5, f'Dendrogram error:\n{str(e)[:50]}',
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.set_title('Dendrogram (Error)')
            return None

    def plot_network(self, ax=None, layout='spring', node_scale=1000,
                     edge_scale=2.0, cmap='viridis'):
        """
        Plot network representation (Markov State Model style).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        layout : str, default='spring'
            Layout algorithm ('spring', 'circular', 'random').

        node_scale : float, default=1000
            Scale factor for node sizes.

        edge_scale : float, default=2.0
            Scale factor for edge widths.

        cmap : str, default='viridis'
            Colormap for nodes.

        Returns
        -------
        dict
            Network information.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required for network plot. "
                              "Install with: pip install networkx")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Build graph
        G = nx.Graph()

        # Add nodes
        for c in range(self.n_clusters_):
            G.add_node(c, population=self._cluster_populations[c],
                       density=self.log_density[self.centers[c]])

        # Add edges
        for pair, info in self.saddles.items():
            c1, c2 = pair
            if c1 < self.n_clusters_ and c2 < self.n_clusters_:
                G.add_edge(c1, c2, weight=info['log_density'])

        # Layout - use circular for sparse graphs, spring for dense
        n_edges = G.number_of_edges()
        n_nodes = G.number_of_nodes()

        if n_edges < n_nodes - 1 or layout == 'circular':
            # Sparse graph or explicitly circular - use circular layout
            pos = nx.circular_layout(G)
        elif layout == 'spring':
            pos = nx.spring_layout(G, seed=42, k=2.0)
        else:
            pos = nx.random_layout(G, seed=42)

        # Node sizes based on population (scaled down for better visibility)
        populations = [self._cluster_populations[c] for c in range(self.n_clusters_)]
        max_pop = max(populations) if populations else 1
        node_sizes = [p / max_pop * node_scale * 0.6 + 200 for p in populations]

        # Node colors based on density
        densities = [self.log_density[self.centers[c]] for c in range(self.n_clusters_)]

        # Edge widths based on saddle density
        edges = G.edges()
        if len(edges) > 0:
            weights = [G[u][v]['weight'] for u, v in edges]
            min_w = min(weights)
            max_w = max(weights)
            if max_w > min_w:
                edge_widths = [(w - min_w) / (max_w - min_w) * edge_scale + 0.5
                               for w in weights]
            else:
                edge_widths = [1.0] * len(weights)
        else:
            edge_widths = []

        # Draw edges
        if len(edges) > 0:
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.6,
                                   edge_color='gray')

        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                       node_color=densities, cmap=cmap,
                                       edgecolors='black', linewidths=2)

        # Labels with cluster info
        labels = {c: f"C{c}\n({populations[c]})" for c in range(self.n_clusters_)}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8,
                                font_weight='bold')

        if nodes is not None:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
            cbar.set_label('Log Density', fontsize=9)

        # Add padding to ensure nodes aren't cut off
        x_vals = [p[0] for p in pos.values()]
        y_vals = [p[1] for p in pos.values()]
        padding = 0.45
        ax.set_xlim(min(x_vals) - padding, max(x_vals) + padding)
        ax.set_ylim(min(y_vals) - padding, max(y_vals) + padding)

        # Title with connection info
        n_possible = self.n_clusters_ * (self.n_clusters_ - 1) // 2
        title = f'Cluster Network ({len(edges)}/{n_possible} connections)'
        if len(edges) == 0:
            title += '\n(Clusters well-separated)'
        ax.set_title(title, fontsize=10)
        ax.axis('off')

        return {'graph': G, 'pos': pos}

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
