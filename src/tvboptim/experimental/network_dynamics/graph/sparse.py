"""Sparse graph implementations using JAX BCOO format.

This module provides sparse alternatives to dense graphs for memory efficiency
with large, sparse connectivity matrices.
"""

from typing import Optional, Tuple, Union, List, Sequence

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax.tree_util import register_pytree_node_class

from .base import AbstractGraph


@register_pytree_node_class
class SparseGraph(AbstractGraph):
    """Sparse graph representation using JAX BCOO format.

    Stores only non-zero weights for memory efficiency. Suitable for large
    networks with sparse connectivity (e.g., < 30% density).

    Args:
        weights: Sparse weight matrix (BCOO) or dense array (will be sparsified)
        region_labels: Optional sequence of region labels (list, tuple, or array). If None, defaults to ['Region_0', 'Region_1', ...]
        threshold: Values with absolute value below this are treated as zero

    Example:
        >>> # From dense
        >>> dense_weights = jnp.array([[0, 0.5, 0], [0.3, 0, 0], [0, 0.2, 0]])
        >>> graph = SparseGraph(dense_weights)
        >>>
        >>> # From COO format
        >>> data = jnp.array([0.5, 0.3, 0.2])
        >>> row = jnp.array([0, 1, 2])
        >>> col = jnp.array([1, 0, 1])
        >>> graph = SparseGraph.from_coo(data, row, col, shape=(3, 3))
        >>>
        >>> # From dense graph
        >>> from network_dynamics.graph.base import DenseGraph
        >>> dense_graph = DenseGraph(dense_weights)
        >>> sparse_graph = SparseGraph.from_dense(dense_graph, threshold=1e-10)
    """

    def __init__(self, weights: Union[BCOO, jnp.ndarray], region_labels: Optional[Sequence[str]] = None, threshold: float = 0.0):
        """Initialize sparse graph from BCOO or dense array."""
        if isinstance(weights, BCOO):
            self._weights = weights
        else:
            # Convert dense to sparse
            weights_arr = jnp.asarray(weights)
            # Apply threshold
            if threshold > 0.0:
                weights_arr = jnp.where(jnp.abs(weights_arr) > threshold, weights_arr, 0.0)
            self._weights = BCOO.fromdense(weights_arr)

        # Validate shape
        if self._weights.ndim != 2:
            raise ValueError(f"Weight matrix must be 2D, got {self._weights.ndim}D")

        if self._weights.shape[0] != self._weights.shape[1]:
            raise ValueError(
                f"Weight matrix must be square, got shape {self._weights.shape}"
            )

        self._n_nodes = self._weights.shape[0]

        # Store region labels (auto-generate if not provided)
        if region_labels is None:
            self._region_labels = [f"Region_{i}" for i in range(self._n_nodes)]
        else:
            if len(region_labels) != self._n_nodes:
                raise ValueError(
                    f"Number of region labels ({len(region_labels)}) must match "
                    f"number of nodes ({self._n_nodes})"
                )
            self._region_labels = list(region_labels)

    @classmethod
    def from_dense(cls, graph: 'AbstractGraph', threshold: float = 1e-10):
        """Convert dense graph to sparse.

        Args:
            graph: Dense graph to convert
            threshold: Set values with |weight| < threshold to zero

        Returns:
            SparseGraph with same connectivity (zeroed below threshold)
        """
        weights = graph.weights
        # Apply threshold
        weights_masked = jnp.where(jnp.abs(weights) > threshold, weights, 0.0)
        # Convert to sparse (threshold already applied, so pass 0.0)
        return cls(weights_masked, threshold=0.0)

    @classmethod
    def from_coo(cls, data: jnp.ndarray, row: jnp.ndarray, col: jnp.ndarray,
                 shape: Tuple[int, int]):
        """Create sparse graph from COO format.

        Args:
            data: Non-zero weight values [nnz]
            row: Row indices [nnz]
            col: Column indices [nnz]
            shape: Matrix shape (n_nodes, n_nodes)

        Returns:
            SparseGraph with specified connectivity

        Example:
            >>> # Triangle graph: 0->1, 1->2, 2->0
            >>> data = jnp.array([0.5, 0.3, 0.2])
            >>> row = jnp.array([0, 1, 2])
            >>> col = jnp.array([1, 2, 0])
            >>> graph = SparseGraph.from_coo(data, row, col, shape=(3, 3))
        """
        indices = jnp.stack([row, col], axis=1)
        weights_sparse = BCOO((data, indices), shape=shape)
        return cls(weights_sparse, threshold=0.0)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the network."""
        return self._n_nodes

    @property
    def region_labels(self) -> Sequence[str]:
        """Labels for each node/region in the network."""
        return self._region_labels

    @property
    def weights(self) -> BCOO:
        """Sparse weight matrix in BCOO format."""
        return self._weights

    @property
    def nnz(self) -> int:
        """Number of non-zero elements in weight matrix."""
        return self._weights.nse

    @property
    def sparsity(self) -> float:
        """Fraction of non-zero connections (excluding diagonal)."""
        total_possible = self._n_nodes * (self._n_nodes - 1)
        if total_possible == 0:
            return 0.0

        # Count off-diagonal non-zeros
        row_idx = self._weights.indices[:, 0]
        col_idx = self._weights.indices[:, 1]
        off_diag_nnz = jnp.sum(row_idx != col_idx)

        return float(off_diag_nnz) / total_possible

    @property
    def symmetric(self) -> bool:
        """Check if the graph is symmetric (undirected).

        Note: This converts to dense for comparison. For large sparse graphs,
        this can be memory intensive. Use sparingly.
        """
        dense = self._weights.todense()
        return bool(jnp.allclose(dense, dense.T))

    def todense(self) -> jnp.ndarray:
        """Convert sparse graph to dense array.

        Warning: This creates a full n_nodes x n_nodes array. Use sparingly
        for large sparse graphs.

        Returns:
            Dense weight matrix [n_nodes, n_nodes]
        """
        return self._weights.todense()

    def verify(self, verbose: bool = True) -> bool:
        """Verify graph structure and properties.

        Args:
            verbose: Whether to print verification details

        Returns:
            True if verification passes, False otherwise
        """
        # Check for NaN/Inf in sparse data
        if jnp.any(jnp.isnan(self._weights.data)):
            if verbose:
                print("ERROR: NaN values in sparse weights")
            return False

        if jnp.any(jnp.isinf(self._weights.data)):
            if verbose:
                print("ERROR: Inf values in sparse weights")
            return False

        if verbose:
            density_pct = self.sparsity * 100
            print(f"SparseGraph verification passed:")
            print(f"  Nodes: {self._n_nodes}")
            print(f"  Non-zeros: {self.nnz}")
            print(f"  Density: {density_pct:.3f}%")
            print(f"  Symmetric: {self.symmetric}")

        return True

    @classmethod
    def random(cls,
               n_nodes: int,
               sparsity: float = 0.7,
               symmetric: bool = True,
               weight_dist: str = 'lognormal',
               allow_self_loops: bool = False,
               key: Optional[jax.random.PRNGKey] = None) -> 'SparseGraph':
        """Create a random sparse graph with brain-like connectivity.

        Args:
            n_nodes: Number of nodes in the network
            sparsity: Fraction of connections present (0.7 = 70% dense)
            symmetric: Whether to create undirected (symmetric) connectivity
            weight_dist: Weight distribution ('lognormal', 'uniform', or 'binary')
            allow_self_loops: Whether to allow self-connections (diagonal)
            key: JAX random key (if None, creates one with seed 0)

        Returns:
            SparseGraph with random connectivity

        Example:
            >>> import jax
            >>> key = jax.random.key(42)
            >>> graph = SparseGraph.random(n_nodes=100, sparsity=0.3, key=key)
        """
        if key is None:
            key = jax.random.key(0)

        # Split keys for edge sampling and weights
        key_edges, key_weights = jax.random.split(key)

        # Calculate expected number of edges
        max_edges = n_nodes * (n_nodes - 1) if not allow_self_loops else n_nodes * n_nodes
        if symmetric:
            max_edges = max_edges // 2  # Only upper triangle

        n_edges = int(sparsity * max_edges)

        if symmetric:
            # Sample edges from upper triangle only
            key_row, key_col = jax.random.split(key_edges)

            # Sample random upper triangular edges
            row = jax.random.randint(key_row, (n_edges,), 0, n_nodes)
            col = jax.random.randint(key_col, (n_edges,), 0, n_nodes)

            # Ensure upper triangle (swap if needed) and no self-loops if required
            needs_swap = row > col
            row, col = jnp.where(needs_swap, col, row), jnp.where(needs_swap, row, col)

            if not allow_self_loops:
                # Filter out diagonal entries
                valid = row != col
                row, col = row[valid], col[valid]
                # Resample to get back to desired edge count (approximate)
                actual_n = len(row)
                if actual_n < n_edges:
                    # Add more edges to compensate
                    key_extra = jax.random.fold_in(key_edges, 1)
                    extra_needed = n_edges - actual_n
                    key_row2, key_col2 = jax.random.split(key_extra)
                    extra_row = jax.random.randint(key_row2, (extra_needed,), 0, n_nodes)
                    extra_col = jax.random.randint(key_col2, (extra_needed,), 0, n_nodes)
                    needs_swap2 = extra_row >= extra_col
                    extra_row = jnp.where(needs_swap2, extra_row + 1, extra_row)
                    extra_row = jnp.clip(extra_row, 0, n_nodes - 1)
                    row = jnp.concatenate([row, extra_row])
                    col = jnp.concatenate([col, extra_col])

            # Generate weights for upper triangle edges
            if weight_dist == 'lognormal':
                edge_weights = jax.random.lognormal(key_weights, shape=(len(row),))
            elif weight_dist == 'uniform':
                edge_weights = jax.random.uniform(key_weights, shape=(len(row),))
            elif weight_dist == 'binary':
                edge_weights = jnp.ones(len(row))
            else:
                raise ValueError(f"Unknown weight_dist: {weight_dist}")

            # Create symmetric edges by duplicating (i,j) -> (j,i)
            row_sym = jnp.concatenate([row, col])
            col_sym = jnp.concatenate([col, row])
            weights_sym = jnp.concatenate([edge_weights, edge_weights])

            # Create sparse matrix from COO
            indices = jnp.stack([row_sym, col_sym], axis=1)
            weights_bcoo = BCOO((weights_sym, indices), shape=(n_nodes, n_nodes))

        else:
            # Non-symmetric: sample random edges
            key_row, key_col = jax.random.split(key_edges)
            row = jax.random.randint(key_row, (n_edges,), 0, n_nodes)
            col = jax.random.randint(key_col, (n_edges,), 0, n_nodes)

            if not allow_self_loops:
                # Filter and resample diagonal entries
                valid = row != col
                row, col = row[valid], col[valid]

            # Generate weights
            if weight_dist == 'lognormal':
                edge_weights = jax.random.lognormal(key_weights, shape=(len(row),))
            elif weight_dist == 'uniform':
                edge_weights = jax.random.uniform(key_weights, shape=(len(row),))
            elif weight_dist == 'binary':
                edge_weights = jnp.ones(len(row))
            else:
                raise ValueError(f"Unknown weight_dist: {weight_dist}")

            # Create sparse matrix from COO
            indices = jnp.stack([row, col], axis=1)
            weights_bcoo = BCOO((edge_weights, indices), shape=(n_nodes, n_nodes))

        return cls(weights_bcoo, threshold=0.0)

    def plot(self, log_scale_weights: bool = False, figsize: tuple = (12, 5)):
        """Plot sparse connectivity matrix and weight distribution.

        Args:
            log_scale_weights: If True, log-transform weights before plotting (helps reveal structure)
            figsize: Figure size (width, height)

        Returns:
            fig, axes: Matplotlib figure and axes

        Note:
            Converts sparse matrix to dense for visualization. Zeros are shown as white (background).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Convert to dense for plotting
        weights_dense = self.todense()

        # Prepare weights for plotting
        if log_scale_weights:
            weights_plot = jnp.where(weights_dense > 0, jnp.log10(weights_dense + 1e-10), jnp.nan)
            weight_label = "log10(Weight)"
        else:
            weights_plot = jnp.where(weights_dense > 0, weights_dense, jnp.nan)
            weight_label = "Weight"

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Create custom colormap with white for NaN (zeros)
        cmap = plt.cm.cividis.copy()
        cmap.set_bad('white')

        # Plot connectivity matrix
        im1 = ax1.imshow(weights_plot, cmap=cmap, aspect='auto')
        ax1.set_title('Connectivity Matrix')
        ax1.set_xlabel('Target Node')
        ax1.set_ylabel('Source Node')
        plt.colorbar(im1, ax=ax1, label=weight_label)

        # Plot weight distribution
        weights_nonzero = weights_dense[weights_dense > 0]
        if len(weights_nonzero) > 0:
            if log_scale_weights:
                weights_nonzero = jnp.log10(weights_nonzero + 1e-10)
            ax2.hist(weights_nonzero, bins=50, edgecolor='black', alpha=0.7)
            ax2.set_xlabel(weight_label)
            ax2.set_ylabel('Count')
            ax2.set_title('Weight Distribution')
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'No connections', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Weight Distribution')

        # Main title with graph properties
        fig.suptitle(f'{self.__class__.__name__}: {self.n_nodes} nodes, nnz={self.nnz}, sparsity={self.sparsity:.3f}, symmetric={self.symmetric}',
                     fontsize=12, y=1.02)

        plt.tight_layout()
        return fig, (ax1, ax2)

    def tree_flatten(self):
        """Flatten for JAX PyTree."""
        children = (self._weights,)
        aux_data = (self._n_nodes, self._region_labels)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from JAX PyTree."""
        obj = object.__new__(cls)
        obj._weights = children[0]
        obj._n_nodes = aux_data[0]
        obj._region_labels = aux_data[1]
        return obj

    def __repr__(self) -> str:
        """String representation of sparse graph."""
        return (
            f"SparseGraph("
            f"n_nodes={self._n_nodes}, "
            f"nnz={self.nnz}, "
            f"sparsity={self.sparsity:.3f}, "
            f"symmetric={self.symmetric})"
        )


@register_pytree_node_class
class SparseDelayGraph(SparseGraph):
    """Sparse graph with transmission delays.

    Both weights and delays are stored as sparse BCOO matrices with the same
    sparsity pattern. Where weight is zero (no connection), delay is undefined.

    Args:
        weights: Sparse weights (BCOO) or dense array
        delays: Sparse delays (BCOO) or dense array (same pattern as weights)
        region_labels: Optional sequence of region labels (list, tuple, or array). If None, defaults to ['Region_0', 'Region_1', ...]
        threshold: Sparsity threshold for weights

    Example:
        >>> # From dense
        >>> weights = jnp.array([[0, 0.5, 0], [0.3, 0, 0], [0, 0.2, 0]])
        >>> delays = jnp.array([[0, 10.0, 0], [5.0, 0, 0], [0, 15.0, 0]])
        >>> graph = SparseDelayGraph(weights, delays)
        >>>
        >>> # From dense delay graph
        >>> from network_dynamics.graph.base import DenseDelayGraph
        >>> dense_graph = DenseDelayGraph(weights, delays)
        >>> sparse_graph = SparseDelayGraph.from_dense(dense_graph, threshold=1e-10)
    """

    def __init__(self, weights: Union[BCOO, jnp.ndarray],
                 delays: Union[BCOO, jnp.ndarray],
                 region_labels: Optional[Sequence[str]] = None,
                 threshold: float = 0.0):
        """Initialize sparse delay graph."""
        # Initialize weights via parent
        super().__init__(weights, region_labels=region_labels, threshold=threshold)

        # Handle delays
        if isinstance(delays, BCOO):
            self._delays = delays
        else:
            # Convert delays to sparse with same pattern as weights
            delays_arr = jnp.asarray(delays)

            # Get weight pattern (where weights are non-zero)
            weights_dense = self._weights.todense()
            weights_mask = jnp.abs(weights_dense) > 0.0

            # Apply mask to delays (delays only where weights exist)
            delays_masked = jnp.where(weights_mask, delays_arr, 0.0)
            self._delays = BCOO.fromdense(delays_masked)

        # Verify same shape
        if self._delays.shape != self._weights.shape:
            raise ValueError(
                f"Delays shape {self._delays.shape} doesn't match "
                f"weights shape {self._weights.shape}"
            )

        # Ideally same sparsity pattern, but we'll allow some flexibility
        if self._delays.nse != self._weights.nse:
            print(
                f"Warning: delays ({self._delays.nse} nnz) and "
                f"weights ({self._weights.nse} nnz) have different sparsity patterns. "
                f"This may indicate an issue."
            )

        # Compute max delay from sparse data (keep as array for JAX tracing)
        self._max_delay = jnp.max(self._delays.data) if self._delays.nse > 0 else jnp.array(0.0)

    @classmethod
    def from_dense(cls, graph: 'AbstractGraph', threshold: float = 1e-10):
        """Convert dense delay graph to sparse.

        Applies the same sparsity pattern to both weights and delays.

        Args:
            graph: Dense delay graph to convert
            threshold: Values with |weight| < threshold treated as zero

        Returns:
            SparseDelayGraph with same connectivity pattern for weights and delays
        """
        if not hasattr(graph, 'delays'):
            raise ValueError("Graph must have delays attribute to convert to SparseDelayGraph")

        weights = graph.weights
        delays = graph.delays

        # Apply threshold to weights to get mask
        weights_mask = jnp.abs(weights) > threshold

        # Apply mask to both weights and delays
        weights_masked = jnp.where(weights_mask, weights, 0.0)
        delays_masked = jnp.where(weights_mask, delays, 0.0)

        return cls(weights_masked, delays_masked, threshold=0.0)

    @property
    def delays(self) -> BCOO:
        """Sparse delay matrix in BCOO format (same pattern as weights)."""
        return self._delays

    @property
    def max_delay(self) -> float:
        """Maximum delay across all connections."""
        return self._max_delay

    def verify(self, verbose: bool = True) -> bool:
        """Verify graph structure and delays.

        Args:
            verbose: Whether to print verification details

        Returns:
            True if verification passes, False otherwise
        """
        # Check weights via parent
        if not super().verify(verbose=False):
            return False

        # Check delays for NaN/Inf
        if jnp.any(jnp.isnan(self._delays.data)):
            if verbose:
                print("ERROR: NaN values in sparse delays")
            return False

        if jnp.any(jnp.isinf(self._delays.data)):
            if verbose:
                print("ERROR: Inf values in sparse delays")
            return False

        # Check for negative delays
        if jnp.any(self._delays.data < 0):
            if verbose:
                print("ERROR: Negative delays found")
            return False

        if verbose:
            density_pct = self.sparsity * 100
            print(f"SparseDelayGraph verification passed:")
            print(f"  Nodes: {self._n_nodes}")
            print(f"  Non-zeros: {self.nnz}")
            print(f"  Density: {density_pct:.3f}%")
            print(f"  Max delay: {self._max_delay:.3f}")
            print(f"  Symmetric: {self.symmetric}")

        return True

    @classmethod
    def random(cls,
               n_nodes: int,
               sparsity: float = 0.7,
               symmetric: bool = True,
               weight_dist: str = 'lognormal',
               max_delay: float = 50.0,
               delay_dist: str = 'uniform',
               allow_self_loops: bool = False,
               key: Optional[jax.random.PRNGKey] = None) -> 'SparseDelayGraph':
        """Create a random sparse delay graph with brain-like connectivity.

        Args:
            n_nodes: Number of nodes in the network
            sparsity: Fraction of connections present (0.7 = 70% dense)
            symmetric: Whether to create undirected (symmetric) connectivity
            weight_dist: Weight distribution ('lognormal', 'uniform', or 'binary')
            max_delay: Maximum transmission delay
            delay_dist: Delay distribution ('uniform' or 'constant')
            allow_self_loops: Whether to allow self-connections (diagonal)
            key: JAX random key (if None, creates one with seed 0)

        Returns:
            SparseDelayGraph with random connectivity and delays

        Example:
            >>> import jax
            >>> key = jax.random.key(42)
            >>> graph = SparseDelayGraph.random(n_nodes=100, sparsity=0.3, max_delay=20.0, key=key)
        """
        if key is None:
            key = jax.random.key(0)

        # Split keys for edge sampling, weights, and delays
        key_edges, key_weights, key_delays = jax.random.split(key, 3)

        # Calculate expected number of edges
        max_edges = n_nodes * (n_nodes - 1) if not allow_self_loops else n_nodes * n_nodes
        if symmetric:
            max_edges = max_edges // 2  # Only upper triangle

        n_edges = int(sparsity * max_edges)

        if symmetric:
            # Sample edges from upper triangle only
            key_row, key_col = jax.random.split(key_edges)

            # Sample random upper triangular edges
            row = jax.random.randint(key_row, (n_edges,), 0, n_nodes)
            col = jax.random.randint(key_col, (n_edges,), 0, n_nodes)

            # Ensure upper triangle (swap if needed) and no self-loops if required
            needs_swap = row > col
            row, col = jnp.where(needs_swap, col, row), jnp.where(needs_swap, row, col)

            if not allow_self_loops:
                # Filter out diagonal entries
                valid = row != col
                row, col = row[valid], col[valid]
                # Resample to get back to desired edge count (approximate)
                actual_n = len(row)
                if actual_n < n_edges:
                    # Add more edges to compensate
                    key_extra = jax.random.fold_in(key_edges, 1)
                    extra_needed = n_edges - actual_n
                    key_row2, key_col2 = jax.random.split(key_extra)
                    extra_row = jax.random.randint(key_row2, (extra_needed,), 0, n_nodes)
                    extra_col = jax.random.randint(key_col2, (extra_needed,), 0, n_nodes)
                    needs_swap2 = extra_row >= extra_col
                    extra_row = jnp.where(needs_swap2, extra_row + 1, extra_row)
                    extra_row = jnp.clip(extra_row, 0, n_nodes - 1)
                    row = jnp.concatenate([row, extra_row])
                    col = jnp.concatenate([col, extra_col])

            # Generate weights for upper triangle edges
            if weight_dist == 'lognormal':
                edge_weights = jax.random.lognormal(key_weights, shape=(len(row),))
            elif weight_dist == 'uniform':
                edge_weights = jax.random.uniform(key_weights, shape=(len(row),))
            elif weight_dist == 'binary':
                edge_weights = jnp.ones(len(row))
            else:
                raise ValueError(f"Unknown weight_dist: {weight_dist}")

            # Generate delays for upper triangle edges
            if delay_dist == 'uniform':
                edge_delays = jax.random.uniform(key_delays, shape=(len(row),),
                                                minval=0.0, maxval=max_delay)
            elif delay_dist == 'constant':
                edge_delays = jnp.full((len(row),), max_delay)
            else:
                raise ValueError(f"Unknown delay_dist: {delay_dist}")

            # Create symmetric edges by duplicating (i,j) -> (j,i)
            row_sym = jnp.concatenate([row, col])
            col_sym = jnp.concatenate([col, row])
            weights_sym = jnp.concatenate([edge_weights, edge_weights])
            delays_sym = jnp.concatenate([edge_delays, edge_delays])

            # Create sparse matrices from COO
            indices = jnp.stack([row_sym, col_sym], axis=1)
            weights_bcoo = BCOO((weights_sym, indices), shape=(n_nodes, n_nodes))
            delays_bcoo = BCOO((delays_sym, indices), shape=(n_nodes, n_nodes))

        else:
            # Non-symmetric: sample random edges
            key_row, key_col = jax.random.split(key_edges)
            row = jax.random.randint(key_row, (n_edges,), 0, n_nodes)
            col = jax.random.randint(key_col, (n_edges,), 0, n_nodes)

            if not allow_self_loops:
                # Filter and resample diagonal entries
                valid = row != col
                row, col = row[valid], col[valid]

            # Generate weights
            if weight_dist == 'lognormal':
                edge_weights = jax.random.lognormal(key_weights, shape=(len(row),))
            elif weight_dist == 'uniform':
                edge_weights = jax.random.uniform(key_weights, shape=(len(row),))
            elif weight_dist == 'binary':
                edge_weights = jnp.ones(len(row))
            else:
                raise ValueError(f"Unknown weight_dist: {weight_dist}")

            # Generate delays
            if delay_dist == 'uniform':
                edge_delays = jax.random.uniform(key_delays, shape=(len(row),),
                                                minval=0.0, maxval=max_delay)
            elif delay_dist == 'constant':
                edge_delays = jnp.full((len(row),), max_delay)
            else:
                raise ValueError(f"Unknown delay_dist: {delay_dist}")

            # Create sparse matrices from COO
            indices = jnp.stack([row, col], axis=1)
            weights_bcoo = BCOO((edge_weights, indices), shape=(n_nodes, n_nodes))
            delays_bcoo = BCOO((edge_delays, indices), shape=(n_nodes, n_nodes))

        return cls(weights_bcoo, delays_bcoo, threshold=0.0)

    def plot(self, log_scale_weights: bool = False, figsize: tuple = (12, 10)):
        """Plot sparse connectivity matrix, delays, and their distributions.

        Args:
            log_scale_weights: If True, log-transform weights before plotting (helps reveal structure)
            figsize: Figure size (width, height)

        Returns:
            fig, axes: Matplotlib figure and axes (2x2 grid)

        Note:
            Converts sparse matrices to dense for visualization. Zeros are shown as white (background).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Convert to dense for plotting
        weights_dense = self.todense()
        delays_dense = self._delays.todense()

        # Prepare weights for plotting
        if log_scale_weights:
            weights_plot = jnp.where(weights_dense > 0, jnp.log10(weights_dense + 1e-10), jnp.nan)
            weight_label = "log10(Weight)"
        else:
            weights_plot = jnp.where(weights_dense > 0, weights_dense, jnp.nan)
            weight_label = "Weight"

        # Prepare delays for plotting (show as NaN where zero)
        delays_plot = jnp.where(delays_dense > 0, delays_dense, jnp.nan)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Create custom colormap with white for NaN (zeros)
        cmap_weights = plt.cm.cividis.copy()
        cmap_weights.set_bad('white')
        cmap_delays = plt.cm.cividis_r.copy()
        cmap_delays.set_bad('white')

        # Plot connectivity matrix
        im1 = axes[0, 0].imshow(weights_plot, cmap=cmap_weights, aspect='auto')
        axes[0, 0].set_title('Connectivity Matrix')
        axes[0, 0].set_xlabel('Target Node')
        axes[0, 0].set_ylabel('Source Node')
        plt.colorbar(im1, ax=axes[0, 0], label=weight_label)

        # Plot weight distribution
        weights_nonzero = weights_dense[weights_dense > 0]
        if len(weights_nonzero) > 0:
            if log_scale_weights:
                weights_nonzero = jnp.log10(weights_nonzero + 1e-10)
            axes[0, 1].hist(weights_nonzero, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 1].set_xlabel(weight_label)
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Weight Distribution')
            axes[0, 1].set_yscale('log')
        else:
            axes[0, 1].text(0.5, 0.5, 'No connections', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Weight Distribution')

        # Plot delay matrix
        im2 = axes[1, 0].imshow(delays_plot, cmap=cmap_delays, aspect='auto')
        axes[1, 0].set_title('Transmission Delays')
        axes[1, 0].set_xlabel('Target Node')
        axes[1, 0].set_ylabel('Source Node')
        plt.colorbar(im2, ax=axes[1, 0], label='Delay')

        # Plot delay distribution
        delays_nonzero = delays_dense[delays_dense > 0]
        if len(delays_nonzero) > 0:
            axes[1, 1].hist(delays_nonzero, bins=50, edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Delay')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Delay Distribution')
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'No delays', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Delay Distribution')

        # Main title with graph properties
        fig.suptitle(f'{self.__class__.__name__}: {self.n_nodes} nodes, nnz={self.nnz}, sparsity={self.sparsity:.3f}, max_delay={self._max_delay:.2f}',
                     fontsize=12, y=0.995)

        plt.tight_layout()
        return fig, axes

    def tree_flatten(self):
        """Flatten for JAX PyTree."""
        children = (self._weights, self._delays)
        aux_data = (self._n_nodes, self._region_labels, self._max_delay)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from JAX PyTree."""
        obj = object.__new__(cls)
        obj._weights = children[0]
        obj._delays = children[1]
        obj._n_nodes = aux_data[0]
        obj._region_labels = aux_data[1]
        obj._max_delay = aux_data[2]
        return obj

    def __repr__(self) -> str:
        """String representation of sparse delay graph."""
        return (
            f"SparseDelayGraph("
            f"n_nodes={self._n_nodes}, "
            f"nnz={self.nnz}, "
            f"sparsity={self.sparsity:.3f}, "
            f"max_delay={self._max_delay:.3f})"
        )
