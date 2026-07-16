"""Sparse graph implementations using JAX BCOO format.

This module provides sparse alternatives to dense graphs for memory efficiency
with large, sparse connectivity matrices.
"""

from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from jax.tree_util import register_pytree_node_class

from .base import AbstractGraph


def _concrete_edge_tuples(matrix: BCOO, name: str) -> list[tuple[int, int]]:
    """Return concrete COO indices and reject unsupported sparse layouts."""
    if matrix.indices.ndim != 2 or matrix.indices.shape[1] != 2:
        raise ValueError(
            f"{name} must use scalar 2D COO entries, got indices shape "
            f"{matrix.indices.shape}"
        )
    if matrix.data.ndim != 1:
        raise ValueError(
            f"{name} must use scalar 2D COO entries, got data shape {matrix.data.shape}"
        )

    try:
        indices = np.asarray(jax.device_get(matrix.indices))
    except Exception as exc:
        raise ValueError(
            f"{name} topology indices must be concrete when the graph is constructed"
        ) from exc

    shape = matrix.shape
    edges = [(int(target), int(source)) for target, source in indices]
    if any(
        target < 0 or target >= shape[0] or source < 0 or source >= shape[1]
        for target, source in edges
    ):
        raise ValueError(f"{name} contains an edge index outside shape {shape}")
    if len(set(edges)) != len(edges):
        raise ValueError(
            f"{name} contains duplicate edge indices; graph topology must be unique"
        )
    return edges


def _align_bcoo_data(
    matrix: BCOO,
    expected_edges: list[tuple[int, int]],
    shared_indices: jnp.ndarray,
    name: str,
) -> BCOO:
    """Reindex BCOO data to ``expected_edges`` without making a dense matrix."""
    actual_edges = _concrete_edge_tuples(matrix, name)
    positions = {edge: position for position, edge in enumerate(actual_edges)}
    expected_set = set(expected_edges)
    if len(actual_edges) != len(expected_edges) or positions.keys() != expected_set:
        missing = expected_set - positions.keys()
        extra = positions.keys() - expected_set
        raise ValueError(
            f"{name} sparsity pattern must exactly match weights; "
            f"missing {len(missing)} edge(s), extra {len(extra)} edge(s)"
        )
    order = jnp.asarray(
        [positions[edge] for edge in expected_edges], dtype=shared_indices.dtype
    )
    return BCOO(
        (matrix.data[order], shared_indices),
        shape=matrix.shape,
        unique_indices=True,
    )


def _edge_capacity(n_nodes: int, symmetric: bool, allow_self_loops: bool) -> int:
    if symmetric:
        return (
            n_nodes * (n_nodes + 1) // 2
            if allow_self_loops
            else n_nodes * (n_nodes - 1) // 2
        )
    return n_nodes * n_nodes if allow_self_loops else n_nodes * (n_nodes - 1)


def _sample_unique_ids(key, population: int, sample_size: int) -> np.ndarray:
    """Sample integer IDs without replacement without allocating ``population``.

    JAX's ``choice(..., replace=False)`` permutes the full population. Sparse
    graphs instead draw small batches with replacement and retain first-seen
    unique IDs. Sampling the complement keeps the expected work bounded when
    density is high.
    """
    if sample_size == 0:
        return np.empty((0,), dtype=np.int64)
    if sample_size > population // 2:
        omitted = _sample_unique_ids(key, population, population - sample_size)
        return np.setdiff1d(
            np.arange(population, dtype=np.int64), omitted, assume_unique=True
        )

    selected: list[int] = []
    seen: set[int] = set()
    attempt = 0
    while len(selected) < sample_size:
        remaining = sample_size - len(selected)
        draw_count = max(2 * remaining, 8)
        draw_key = jax.random.fold_in(key, attempt)
        draws = np.asarray(
            jax.random.randint(
                draw_key,
                (draw_count,),
                minval=0,
                maxval=population,
                dtype=jnp.int64 if jax.config.x64_enabled else jnp.int32,
            )
        )
        for value in draws:
            edge_id = int(value)
            if edge_id not in seen:
                seen.add(edge_id)
                selected.append(edge_id)
                if len(selected) == sample_size:
                    break
        attempt += 1
    return np.asarray(selected, dtype=np.int64)


def _decode_edge_ids(
    edge_ids: np.ndarray,
    n_nodes: int,
    symmetric: bool,
    allow_self_loops: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode compact candidate IDs into unique target/source pairs."""
    if edge_ids.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty
    if symmetric and allow_self_loops:
        source = np.floor((np.sqrt(8.0 * edge_ids + 1.0) - 1.0) / 2.0).astype(np.int64)
        target = edge_ids - source * (source + 1) // 2
    elif symmetric:
        source = np.floor((1.0 + np.sqrt(1.0 + 8.0 * edge_ids)) / 2.0).astype(np.int64)
        target = edge_ids - source * (source - 1) // 2
    elif allow_self_loops:
        target, source = np.divmod(edge_ids, n_nodes)
    else:
        target, source = np.divmod(edge_ids, n_nodes - 1)
        source = source + (source >= target)
    return target, source


def _sample_unique_edge_indices(
    key,
    n_nodes: int,
    density: float,
    symmetric: bool,
    allow_self_loops: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Sample one unique topology and return value-sharing indices.

    The returned COO array is in public ``(target, source)`` order.
    ``value_indices`` maps each directed entry to its sampled representative,
    so reciprocal entries of a symmetric graph receive identical values.
    """
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be between 0 and 1, got {density}")

    population = _edge_capacity(n_nodes, symmetric, allow_self_loops)
    n_representatives = int(round(density * population))
    if density > 0 and population > 0:
        n_representatives = max(1, n_representatives)

    edge_ids = _sample_unique_ids(key, population, n_representatives)
    target, source = _decode_edge_ids(edge_ids, n_nodes, symmetric, allow_self_loops)
    representatives = np.arange(n_representatives, dtype=np.int64)

    if symmetric:
        off_diagonal = target != source
        target = np.concatenate([target, source[off_diagonal]])
        source = np.concatenate([source, target[:n_representatives][off_diagonal]])
        representatives = np.concatenate(
            [representatives, representatives[off_diagonal]]
        )

    indices = jnp.asarray(np.stack([target, source], axis=1))
    return indices, jnp.asarray(representatives), n_representatives


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

    def __init__(
        self,
        weights: Union[BCOO, jnp.ndarray],
        region_labels: Optional[Sequence[str]] = None,
        threshold: float = 0.0,
    ):
        """Initialize sparse graph from BCOO or dense array."""
        if isinstance(weights, BCOO):
            self._weights = weights
        else:
            # Convert dense to sparse
            weights_arr = jnp.asarray(weights)
            # Apply threshold
            if threshold > 0.0:
                weights_arr = jnp.where(
                    jnp.abs(weights_arr) > threshold, weights_arr, 0.0
                )
            self._weights = BCOO.fromdense(weights_arr)

        # Validate shape
        if self._weights.ndim != 2:
            raise ValueError(f"Weight matrix must be 2D, got {self._weights.ndim}D")

        if self._weights.shape[0] != self._weights.shape[1]:
            raise ValueError(
                f"Weight matrix must be square, got shape {self._weights.shape}"
            )

        _concrete_edge_tuples(self._weights, "weights")

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
    def from_dense(cls, graph: "AbstractGraph", threshold: float = 1e-10):
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
    def from_coo(
        cls,
        data: jnp.ndarray,
        row: jnp.ndarray,
        col: jnp.ndarray,
        shape: Tuple[int, int],
    ):
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

    @weights.setter
    def weights(self, value: BCOO) -> None:
        """Replace the weight matrix in place, no validation or recompute.

        See DenseGraph.weights.setter for the rationale: cached derived
        fields are introspection-only after prepare(), and the setter is
        deliberately unvalidated so axis-sweep placeholders (e.g. DataAxis)
        can be assigned here before Space resolves them.
        """
        self._weights = value

    @property
    def edge_indices(self) -> jnp.ndarray:
        """Read-only ``[E, 2]`` COO indices in ``(target, source)`` order.

        The order is the graph's public edge order and is preserved by
        coupling preparation. Build scalable edge-shaped parameters in this
        order, preferably via :meth:`gather_edges`.
        """
        return self._weights.indices

    def gather_edges(self, graph_shaped: jnp.ndarray) -> jnp.ndarray:
        """Gather dense ``[target, source]`` values in public edge order."""
        values = jnp.asarray(graph_shaped)
        if values.shape != self._weights.shape:
            raise ValueError(
                "graph_shaped must match the graph shape "
                f"{self._weights.shape}, got {values.shape}"
            )
        return values[self.edge_indices[:, 0], self.edge_indices[:, 1]]

    @property
    def nnz(self) -> int:
        """Number of non-zero elements in weight matrix."""
        return self._weights.nse

    @property
    def density(self) -> float:
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
            density_pct = self.density * 100
            print("SparseGraph verification passed:")
            print(f"  Nodes: {self._n_nodes}")
            print(f"  Non-zeros: {self.nnz}")
            print(f"  Density: {density_pct:.3f}%")
            print(f"  Symmetric: {self.symmetric}")

        return True

    @classmethod
    def random(
        cls,
        n_nodes: int,
        density: float = 1.0,
        symmetric: bool = True,
        weight_dist: str = "lognormal",
        allow_self_loops: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> "SparseGraph":
        """Create a random sparse graph with brain-like connectivity.

        Args:
            n_nodes: Number of nodes in the network
            density: Fraction of connections present (1.0 = fully connected, 0.3 = 30% dense)
            symmetric: Whether to create undirected (symmetric) connectivity
            weight_dist: Weight distribution ('lognormal', 'uniform', or 'binary')
            allow_self_loops: Whether to allow self-connections (diagonal)
            key: JAX random key (if None, creates one with seed 0)

        Returns:
            SparseGraph with random connectivity

        Example:
            >>> import jax
            >>> key = jax.random.key(42)
            >>> graph = SparseGraph.random(n_nodes=100, density=0.3, key=key)
        """
        if key is None:
            key = jax.random.key(0)

        # Split keys for edge sampling and weights
        key_edges, key_weights = jax.random.split(key)
        indices, value_indices, n_values = _sample_unique_edge_indices(
            key_edges, n_nodes, density, symmetric, allow_self_loops
        )

        if weight_dist == "lognormal":
            sampled_weights = jax.random.lognormal(key_weights, shape=(n_values,))
        elif weight_dist == "uniform":
            sampled_weights = jax.random.uniform(key_weights, shape=(n_values,))
        elif weight_dist == "binary":
            sampled_weights = jnp.ones(n_values)
        else:
            raise ValueError(f"Unknown weight_dist: {weight_dist}")

        edge_weights = sampled_weights[value_indices]
        weights_bcoo = BCOO(
            (edge_weights, indices),
            shape=(n_nodes, n_nodes),
            unique_indices=True,
        )
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
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        # Convert to dense for plotting
        weights_dense = self.todense()

        # Prepare weights for plotting
        if log_scale_weights:
            weights_plot = jnp.where(
                weights_dense > 0, jnp.log10(weights_dense + 1e-10), jnp.nan
            )
            weight_label = "log10(Weight)"
        else:
            weights_plot = jnp.where(weights_dense > 0, weights_dense, jnp.nan)
            weight_label = "Weight"

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Create custom colormap with white for NaN (zeros)
        cmap = plt.cm.cividis.copy()
        cmap.set_bad("white")

        # Plot connectivity matrix
        im1 = ax1.imshow(weights_plot, cmap=cmap, aspect="auto")
        ax1.set_title("Connectivity Matrix")
        ax1.set_xlabel("Target Node")
        ax1.set_ylabel("Source Node")
        plt.colorbar(im1, ax=ax1, label=weight_label)

        # Plot weight distribution
        weights_nonzero = weights_dense[weights_dense > 0]
        if len(weights_nonzero) > 0:
            if log_scale_weights:
                weights_nonzero = jnp.log10(weights_nonzero + 1e-10)
            ax2.hist(weights_nonzero, bins=50, edgecolor="black", alpha=0.7)
            ax2.set_xlabel(weight_label)
            ax2.set_ylabel("Count")
            ax2.set_title("Weight Distribution")
            ax2.set_yscale("log")
        else:
            ax2.text(
                0.5,
                0.5,
                "No connections",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Weight Distribution")

        # Main title with graph properties
        fig.suptitle(
            f"{self.__class__.__name__}: {self.n_nodes} nodes, nnz={self.nnz}, density={self.density:.3f}, symmetric={self.symmetric}",
            fontsize=12,
            y=1.02,
        )

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
            f"density={self.density:.3f}, "
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
        max_delay_bound: Optional static bound on the largest representable delay,
            used to size the history buffer instead of ``max(delays)``. Distinct
            from the ``max_delay`` property (which always reports the largest
            *actual* delay). See ``DenseDelayGraph`` for the full rationale.

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

    def __init__(
        self,
        weights: Union[BCOO, jnp.ndarray],
        delays: Union[BCOO, jnp.ndarray],
        region_labels: Optional[Sequence[str]] = None,
        threshold: float = 0.0,
        max_delay_bound: Optional[float] = None,
    ):
        """Initialize sparse delay graph."""
        # Initialize weights via parent
        super().__init__(weights, region_labels=region_labels, threshold=threshold)

        weight_edges = _concrete_edge_tuples(self._weights, "weights")
        shared_indices = self._weights.indices

        if isinstance(delays, BCOO):
            if delays.shape != self._weights.shape:
                raise ValueError(
                    f"Delays shape {delays.shape} doesn't match "
                    f"weights shape {self._weights.shape}"
                )
            self._delays = _align_bcoo_data(
                delays, weight_edges, shared_indices, "delays"
            )
        else:
            delays_arr = jnp.asarray(delays)
            if delays_arr.shape != self._weights.shape:
                raise ValueError(
                    f"Delays shape {delays_arr.shape} doesn't match "
                    f"weights shape {self._weights.shape}"
                )
            delay_data = delays_arr[shared_indices[:, 0], shared_indices[:, 1]]
            self._delays = BCOO(
                (delay_data, shared_indices),
                shape=self._weights.shape,
                unique_indices=True,
            )

        delays_are_concrete = not isinstance(self._delays.data, jax.core.Tracer)

        if max_delay_bound is not None:
            max_delay_bound = float(max_delay_bound)

        # Compute max delay from sparse data (keep as array for JAX tracing).
        # Only computable when delays are concrete; if delays is a tracer,
        # max_delay_bound must be supplied since nothing else can size the
        # (static) history buffer. Mirrors DenseDelayGraph.__init__.
        if delays_are_concrete:
            self._max_delay = (
                jnp.max(self._delays.data) if self._delays.nse > 0 else jnp.array(0.0)
            )
            if max_delay_bound is not None and max_delay_bound < float(self._max_delay):
                raise ValueError(
                    f"max_delay_bound={max_delay_bound} is smaller than the "
                    f"largest delay in the network ({float(self._max_delay)}); the "
                    f"history buffer would be too short to represent it."
                )
        elif max_delay_bound is not None:
            self._max_delay = jnp.array(max_delay_bound)
        else:
            raise ValueError(
                "delays is a JAX tracer (not a concrete array); max_delay_bound "
                "must be supplied so the history buffer has a static size."
            )

        self._max_delay_bound = max_delay_bound

    @classmethod
    def from_dense(cls, graph: "AbstractGraph", threshold: float = 1e-10):
        """Convert dense delay graph to sparse.

        Applies the same sparsity pattern to both weights and delays.

        Args:
            graph: Dense delay graph to convert
            threshold: Values with |weight| < threshold treated as zero

        Returns:
            SparseDelayGraph with same connectivity pattern for weights and delays
        """
        if not hasattr(graph, "delays"):
            raise ValueError(
                "Graph must have delays attribute to convert to SparseDelayGraph"
            )

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

    @delays.setter
    def delays(self, value: BCOO) -> None:
        """Replace the delay matrix in place, no validation or recompute.

        See DenseDelayGraph.delays.setter for the rationale. Mutate a config
        that has already been prepare()'d (buffer already sized); construct a
        fresh SparseDelayGraph via __init__ instead if you need max_delay /
        max_delay_bound to reflect a changed delays array before prepare().
        """
        self._delays = value

    @property
    def max_delay(self) -> float:
        """Maximum delay across all connections."""
        return self._max_delay

    @property
    def max_delay_bound(self) -> Optional[float]:
        """Declared static bound on delays, or None if sized from max(delays)."""
        return self._max_delay_bound

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
            density_pct = self.density * 100
            print("SparseDelayGraph verification passed:")
            print(f"  Nodes: {self._n_nodes}")
            print(f"  Non-zeros: {self.nnz}")
            print(f"  Density: {density_pct:.3f}%")
            print(f"  Max delay: {self._max_delay:.3f}")
            print(f"  Symmetric: {self.symmetric}")

        return True

    @classmethod
    def random(
        cls,
        n_nodes: int,
        density: float = 1.0,
        symmetric: bool = True,
        weight_dist: str = "lognormal",
        max_delay: float = 50.0,
        delay_dist: str = "uniform",
        allow_self_loops: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> "SparseDelayGraph":
        """Create a random sparse delay graph with brain-like connectivity.

        Args:
            n_nodes: Number of nodes in the network
            density: Fraction of connections present (1.0 = fully connected, 0.3 = 30% dense)
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
            >>> graph = SparseDelayGraph.random(n_nodes=100, density=0.3, max_delay=20.0, key=key)
        """
        if key is None:
            key = jax.random.key(0)

        # Split keys for edge sampling, weights, and delays
        key_edges, key_weights, key_delays = jax.random.split(key, 3)
        indices, value_indices, n_values = _sample_unique_edge_indices(
            key_edges, n_nodes, density, symmetric, allow_self_loops
        )

        if weight_dist == "lognormal":
            sampled_weights = jax.random.lognormal(key_weights, shape=(n_values,))
        elif weight_dist == "uniform":
            sampled_weights = jax.random.uniform(key_weights, shape=(n_values,))
        elif weight_dist == "binary":
            sampled_weights = jnp.ones(n_values)
        else:
            raise ValueError(f"Unknown weight_dist: {weight_dist}")

        if delay_dist == "uniform":
            sampled_delays = jax.random.uniform(
                key_delays, shape=(n_values,), minval=0.0, maxval=max_delay
            )
        elif delay_dist == "constant":
            sampled_delays = jnp.full((n_values,), max_delay)
        else:
            raise ValueError(f"Unknown delay_dist: {delay_dist}")

        weights_bcoo = BCOO(
            (sampled_weights[value_indices], indices),
            shape=(n_nodes, n_nodes),
            unique_indices=True,
        )
        delays_bcoo = BCOO(
            (sampled_delays[value_indices], indices),
            shape=(n_nodes, n_nodes),
            unique_indices=True,
        )
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
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        # Convert to dense for plotting
        weights_dense = self.todense()
        delays_dense = self._delays.todense()

        # Prepare weights for plotting
        if log_scale_weights:
            weights_plot = jnp.where(
                weights_dense > 0, jnp.log10(weights_dense + 1e-10), jnp.nan
            )
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
        cmap_weights.set_bad("white")
        cmap_delays = plt.cm.cividis_r.copy()
        cmap_delays.set_bad("white")

        # Plot connectivity matrix
        im1 = axes[0, 0].imshow(weights_plot, cmap=cmap_weights, aspect="auto")
        axes[0, 0].set_title("Connectivity Matrix")
        axes[0, 0].set_xlabel("Target Node")
        axes[0, 0].set_ylabel("Source Node")
        plt.colorbar(im1, ax=axes[0, 0], label=weight_label)

        # Plot weight distribution
        weights_nonzero = weights_dense[weights_dense > 0]
        if len(weights_nonzero) > 0:
            if log_scale_weights:
                weights_nonzero = jnp.log10(weights_nonzero + 1e-10)
            axes[0, 1].hist(weights_nonzero, bins=50, edgecolor="black", alpha=0.7)
            axes[0, 1].set_xlabel(weight_label)
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].set_title("Weight Distribution")
            axes[0, 1].set_yscale("log")
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "No connections",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Weight Distribution")

        # Plot delay matrix
        im2 = axes[1, 0].imshow(delays_plot, cmap=cmap_delays, aspect="auto")
        axes[1, 0].set_title("Transmission Delays")
        axes[1, 0].set_xlabel("Target Node")
        axes[1, 0].set_ylabel("Source Node")
        plt.colorbar(im2, ax=axes[1, 0], label="Delay")

        # Plot delay distribution
        delays_nonzero = delays_dense[delays_dense > 0]
        if len(delays_nonzero) > 0:
            axes[1, 1].hist(delays_nonzero, bins=50, edgecolor="black", alpha=0.7)
            axes[1, 1].set_xlabel("Delay")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("Delay Distribution")
            axes[1, 1].set_yscale("log")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No delays",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Delay Distribution")

        # Main title with graph properties
        fig.suptitle(
            f"{self.__class__.__name__}: {self.n_nodes} nodes, nnz={self.nnz}, density={self.density:.3f}, max_delay={self._max_delay:.2f}",
            fontsize=12,
            y=0.995,
        )

        plt.tight_layout()
        return fig, axes

    def tree_flatten(self):
        """Flatten for JAX PyTree."""
        children = (self._weights, self._delays)
        aux_data = (
            self._n_nodes,
            self._region_labels,
            self._max_delay,
            self._max_delay_bound,
        )
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
        obj._max_delay_bound = aux_data[3]
        return obj

    def __repr__(self) -> str:
        """String representation of sparse delay graph."""
        return (
            f"SparseDelayGraph("
            f"n_nodes={self._n_nodes}, "
            f"nnz={self.nnz}, "
            f"density={self.density:.3f}, "
            f"max_delay={self._max_delay:.3f})"
        )
