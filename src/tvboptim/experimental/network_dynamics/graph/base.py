"""Abstract base class for network topology representations."""

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


class AbstractGraph(ABC):
    """Abstract base class for network topology representations.

    Defines the interface for network topology that can be used with dynamics.
    All graph types (dense, sparse, multilayer, etc.) should inherit from this class.

    Note: Subclasses should store immutable properties (n_nodes, sparsity, symmetric)
    as private fields computed during initialization to avoid issues during pytree
    transformations where weights might not be available.
    """

    @property
    @abstractmethod
    def n_nodes(self) -> int:
        """Number of nodes in the network.

        This should return a stored value (_n_nodes), not computed from weights.
        """
        pass

    @property
    @abstractmethod
    def region_labels(self) -> Sequence[str]:
        """Labels for each node/region in the network.

        This should return a stored sequence of labels (_region_labels).
        If not provided during initialization, defaults to ['Region_0', 'Region_1', ...].
        """
        pass

    @property
    @abstractmethod
    def weights(self) -> jnp.ndarray:
        """Weight matrix [n_nodes, n_nodes].

        Returns the connectivity matrix that can be used directly with
        JAX operations like jnp.matmul() or @. Works for both dense
        and sparse representations.
        """
        pass

    @abstractmethod
    def verify(self, verbose: bool = True) -> bool:
        """Verify graph structure and properties.

        Args:
            verbose: Whether to print verification details

        Returns:
            True if verification passes, False otherwise
        """
        pass

    @property
    @abstractmethod
    def symmetric(self) -> bool:
        """Check if the graph is symmetric (undirected).

        This should return a stored value (_symmetric), not computed from weights.
        """
        pass

    @property
    @abstractmethod
    def sparsity(self) -> float:
        """Calculate the sparsity of the graph (fraction of non-zero connections).

        This should return a stored value (_sparsity), not computed from weights.
        """
        pass

    def __repr__(self) -> str:
        """String representation of the graph."""
        return (
            f"{self.__class__.__name__}("
            f"n_nodes={self.n_nodes}, "
            f"sparsity={self.sparsity:.3f}, "
            f"symmetric={self.symmetric})"
        )


@register_pytree_node_class
class DenseGraph(AbstractGraph):
    """Dense graph representation.

    Standard dense matrix representation of network topology.
    Suitable for most brain networks which are typically 50-90% dense.

    Args:
        weights: Weight matrix [n_nodes, n_nodes]
        region_labels: Optional sequence of region labels (list, tuple, or array). If None, defaults to ['Region_0', 'Region_1', ...]
        symmetric: Whether to treat as symmetric (None = auto-detect)
    """

    def __init__(
        self,
        weights: jnp.ndarray,
        region_labels: Optional[Sequence[str]] = None,
        symmetric: Optional[bool] = None,
    ):
        # Convert to JAX array if needed
        self._weights = jnp.asarray(weights)

        # Validate basic shape requirements
        if self._weights.ndim != 2:
            raise ValueError(f"Weight matrix must be 2D, got {self._weights.ndim}D")

        if self._weights.shape[0] != self._weights.shape[1]:
            raise ValueError(
                f"Weight matrix must be square, got shape {self._weights.shape}"
            )

        # Store n_nodes to avoid accessing weights.shape during pytree transformations
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
            self._region_labels = list(region_labels)  # Ensure it's a list

        # Compute and store symmetric property
        if symmetric is None:
            # Auto-detect symmetry
            self._symmetric = bool(jnp.allclose(self._weights, self._weights.T))
        else:
            self._symmetric = symmetric

        # Compute and store sparsity
        if self._n_nodes <= 1:
            self._sparsity = 0.0
        else:
            total_possible = self._n_nodes * (self._n_nodes - 1)  # Exclude diagonal
            non_zero = jnp.count_nonzero(
                self._weights - jnp.diag(jnp.diag(self._weights))
            )
            self._sparsity = float(non_zero) / total_possible

        # Run verification
        if not self.verify(verbose=False):
            raise ValueError("Graph verification failed")

    @property
    def weights(self) -> jnp.ndarray:
        """Weight matrix [n_nodes, n_nodes]."""
        return self._weights

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the network."""
        return self._n_nodes

    @property
    def region_labels(self) -> Sequence[str]:
        """Labels for each node/region in the network."""
        return self._region_labels

    @property
    def symmetric(self) -> bool:
        """Check if the graph is symmetric (undirected)."""
        return self._symmetric

    @property
    def sparsity(self) -> float:
        """Calculate the sparsity of the graph (fraction of non-zero connections)."""
        return self._sparsity

    def verify(self, verbose: bool = True) -> bool:
        """Verify graph structure and properties.

        Checks:
        - Finite values in weight matrix
        - Consistent shape
        - Symmetric flag consistency (if provided)

        Args:
            verbose: Whether to print verification details

        Returns:
            True if verification passes, False otherwise
        """
        try:
            if verbose:
                print(f"Verifying {self.__class__.__name__}:")
                print(f"  Shape: {self.weights.shape}")
                print(f"  Nodes: {self.n_nodes}")
                print(f"  Sparsity: {self.sparsity:.3f}")

            # Check for finite values
            if not jnp.all(jnp.isfinite(self.weights)):
                if verbose:
                    print("  L ERROR: Weight matrix contains non-finite values")
                return False

            # Verify symmetric flag consistency with actual weights
            actual_symmetric = bool(jnp.allclose(self.weights, self.weights.T))
            if self._symmetric != actual_symmetric:
                if verbose:
                    print(
                        f"  L ERROR: Stored symmetric flag ({self._symmetric}) doesn't match actual symmetry ({actual_symmetric})"
                    )
                return False

            if verbose:
                print(f"  Symmetric: {self.symmetric}")
                print("   Verification passed!")

            return True

        except Exception as e:
            if verbose:
                print(f"  L ERROR: {e}")
            return False

    @classmethod
    def random(
        cls,
        n_nodes: int,
        sparsity: float = 0.7,
        symmetric: bool = True,
        weight_dist: str = "lognormal",
        allow_self_loops: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> "DenseGraph":
        """Create a random graph with brain-like connectivity.

        Args:
            n_nodes: Number of nodes in the network
            sparsity: Fraction of connections present (0.7 = 70% dense)
            symmetric: Whether to create undirected (symmetric) connectivity
            weight_dist: Weight distribution ('lognormal', 'uniform', or 'binary')
            allow_self_loops: Whether to allow self-connections (diagonal)
            key: JAX random key (if None, creates one with seed 0)

        Returns:
            DenseGraph with random connectivity

        Example:
            >>> import jax
            >>> key = jax.random.key(42)
            >>> graph = DenseGraph.random(n_nodes=10, sparsity=0.5, key=key)
        """
        if key is None:
            key = jax.random.key(0)

        # Split keys for connectivity and weights
        key_conn, key_weights = jax.random.split(key)

        # Generate connectivity mask (Erdős-Rényi)
        conn_prob = jax.random.uniform(key_conn, shape=(n_nodes, n_nodes))
        mask = conn_prob < sparsity

        # Remove self-loops if needed
        if not allow_self_loops:
            mask = mask.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(False)

        # Make symmetric if needed
        if symmetric:
            # Use upper triangle and mirror
            mask = jnp.triu(mask, k=1)
            mask = mask | mask.T

        # Generate weights based on distribution
        if weight_dist == "lognormal":
            # Log-normal: mean=0, std=1 in log-space
            weights = jax.random.lognormal(key_weights, shape=(n_nodes, n_nodes))
        elif weight_dist == "uniform":
            weights = jax.random.uniform(key_weights, shape=(n_nodes, n_nodes))
        elif weight_dist == "binary":
            weights = jnp.ones((n_nodes, n_nodes))
        else:
            raise ValueError(
                f"Unknown weight_dist: {weight_dist}. Use 'lognormal', 'uniform', or 'binary'"
            )

        # Make weights symmetric if needed
        if symmetric:
            weights = jnp.triu(weights, k=1)
            weights = weights + weights.T

        # Apply mask
        weights = jnp.where(mask, weights, 0.0)

        return cls(weights, symmetric=symmetric if symmetric else None)

    def tree_flatten(self):
        """Flatten Graph for JAX PyTree."""
        children = (self._weights,)  # Array data
        aux_data = (
            self._n_nodes,
            self._symmetric,
            self._sparsity,
            self._region_labels,
        )  # Static metadata
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct Graph from PyTree data."""
        obj = object.__new__(cls)
        obj._weights = children[0]
        obj._n_nodes = aux_data[0]
        obj._symmetric = aux_data[1]
        obj._sparsity = aux_data[2]
        obj._region_labels = aux_data[3]
        return obj

    def plot(self, log_scale_weights: bool = False, figsize: tuple = (12, 5)):
        """Plot connectivity matrix and weight distribution.

        Args:
            log_scale_weights: If True, log-transform weights before plotting (helps reveal structure)
            figsize: Figure size (width, height)

        Returns:
            fig, axes: Matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        # Prepare weights for plotting
        weights_plot = jnp.array(self.weights)
        if log_scale_weights:
            # Log transform, handling zeros
            weights_plot = jnp.where(
                weights_plot > 0, jnp.log10(weights_plot + 1e-10), 0.0
            )
            weight_label = "log10(Weight)"
        else:
            weight_label = "Weight"

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot connectivity matrix
        im1 = ax1.imshow(weights_plot, cmap="cividis", aspect="auto")
        ax1.set_title("Connectivity Matrix")
        ax1.set_xlabel("Target Node")
        ax1.set_ylabel("Source Node")
        plt.colorbar(im1, ax=ax1, label=weight_label)

        # Plot weight distribution
        weights_nonzero = self.weights[self.weights > 0]
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
            f"{self.__class__.__name__}: {self.n_nodes} nodes, sparsity={self.sparsity:.3f}, symmetric={self.symmetric}",
            fontsize=12,
            y=1.02,
        )

        plt.tight_layout()
        return fig, (ax1, ax2)


@register_pytree_node_class
class DenseDelayGraph(DenseGraph):
    """Dense graph with transmission delays.

    Extends DenseGraph with a delay matrix for modeling transmission delays
    between network nodes. Used for delay differential equations (DDEs).

    Args:
        weights: Weight matrix [n_nodes, n_nodes]
        delays: Delay matrix [n_nodes, n_nodes] in same units as integration time
        region_labels: Optional sequence of region labels (list, tuple, or array). If None, defaults to ['Region_0', 'Region_1', ...]
        symmetric: Whether to treat as symmetric (None = auto-detect)
    """

    def __init__(
        self,
        weights: jnp.ndarray,
        delays: jnp.ndarray,
        region_labels: Optional[Sequence[str]] = None,
        symmetric: Optional[bool] = None,
    ):
        # Process delays first (needed for verify method)
        self._delays = jnp.asarray(delays)

        # Validate delay matrix shape before calling parent
        weights_array = jnp.asarray(weights)
        if self._delays.shape != weights_array.shape:
            raise ValueError(
                f"Delay matrix shape {self._delays.shape} must match weight matrix shape {weights_array.shape}"
            )

        # Compute and store max_delay to avoid accessing delays during pytree transformations
        self._max_delay = float(jnp.max(self._delays))

        # Initialize parent Graph (pass region_labels)
        super().__init__(weights, region_labels=region_labels, symmetric=symmetric)

        # Run additional verification
        if not self.verify(verbose=False):
            raise ValueError("DelayGraph verification failed")

    @property
    def delays(self) -> jnp.ndarray:
        """Delay matrix [n_nodes, n_nodes]."""
        return self._delays

    @property
    def max_delay(self) -> float:
        """Maximum delay in the network."""
        return self._max_delay

    def verify(self, verbose: bool = True) -> bool:
        """Verify delay graph structure.

        Additional checks beyond Graph.verify():
        - Non-negative delays
        - Finite delay values
        - Delay matrix shape consistency
        """
        # First run parent verification
        if not super().verify(verbose):
            return False

        try:
            if verbose:
                print(f"  Delays shape: {self.delays.shape}")
                print(f"  Max delay: {self.max_delay}")

            # Check for non-negative delays
            if jnp.any(self.delays < 0):
                if verbose:
                    print("  L ERROR: Delay matrix contains negative values")
                return False

            # Check for finite delays
            if not jnp.all(jnp.isfinite(self.delays)):
                if verbose:
                    print("  L ERROR: Delay matrix contains non-finite values")
                return False

            if verbose:
                print("   DelayGraph verification passed!")

            return True

        except Exception as e:
            if verbose:
                print(f"  L ERROR: {e}")
            return False

    def __repr__(self) -> str:
        """String representation of the delay graph."""
        return (
            f"{self.__class__.__name__}("
            f"n_nodes={self.n_nodes}, "
            f"sparsity={self.sparsity:.3f}, "
            f"symmetric={self.symmetric}, "
            f"max_delay={self.max_delay:.3f})"
        )

    @classmethod
    def random(
        cls,
        n_nodes: int,
        sparsity: float = 0.7,
        symmetric: bool = True,
        weight_dist: str = "lognormal",
        max_delay: float = 50.0,
        delay_dist: str = "uniform",
        allow_self_loops: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> "DenseDelayGraph":
        """Create a random delay graph with brain-like connectivity.

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
            DenseDelayGraph with random connectivity and delays

        Example:
            >>> import jax
            >>> key = jax.random.key(42)
            >>> graph = DenseDelayGraph.random(n_nodes=10, max_delay=20.0, key=key)
        """
        if key is None:
            key = jax.random.key(0)

        # Split keys for weights and delays
        key_weights, key_delays = jax.random.split(key)

        # Generate connectivity mask and weights using parent method
        key_conn, key_w = jax.random.split(key_weights)
        conn_prob = jax.random.uniform(key_conn, shape=(n_nodes, n_nodes))
        mask = conn_prob < sparsity

        # Remove self-loops if needed
        if not allow_self_loops:
            mask = mask.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(False)

        # Make symmetric if needed
        if symmetric:
            mask = jnp.triu(mask, k=1)
            mask = mask | mask.T

        # Generate weights
        if weight_dist == "lognormal":
            weights = jax.random.lognormal(key_w, shape=(n_nodes, n_nodes))
        elif weight_dist == "uniform":
            weights = jax.random.uniform(key_w, shape=(n_nodes, n_nodes))
        elif weight_dist == "binary":
            weights = jnp.ones((n_nodes, n_nodes))
        else:
            raise ValueError(f"Unknown weight_dist: {weight_dist}")

        # Make weights symmetric if needed
        if symmetric:
            weights = jnp.triu(weights, k=1)
            weights = weights + weights.T

        weights = jnp.where(mask, weights, 0.0)

        # Generate delays with same sparsity pattern
        if delay_dist == "uniform":
            delays = jax.random.uniform(
                key_delays, shape=(n_nodes, n_nodes), minval=0.0, maxval=max_delay
            )
        elif delay_dist == "constant":
            delays = jnp.full((n_nodes, n_nodes), max_delay)
        else:
            raise ValueError(
                f"Unknown delay_dist: {delay_dist}. Use 'uniform' or 'constant'"
            )

        # Make delays symmetric if needed
        if symmetric:
            delays = jnp.triu(delays, k=1)
            delays = delays + delays.T

        # Apply same mask to delays
        delays = jnp.where(mask, delays, 0.0)

        return cls(weights, delays, symmetric=symmetric if symmetric else None)

    def plot(self, log_scale_weights: bool = False, figsize: tuple = (12, 10)):
        """Plot connectivity matrix, delays, and their distributions.

        Args:
            log_scale_weights: If True, log-transform weights before plotting (helps reveal structure)
            figsize: Figure size (width, height)

        Returns:
            fig, axes: Matplotlib figure and axes (2x2 grid)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        # Prepare weights for plotting
        weights_plot = jnp.array(self.weights)
        if log_scale_weights:
            weights_plot = jnp.where(
                weights_plot > 0, jnp.log10(weights_plot + 1e-10), 0.0
            )
            weight_label = "log10(Weight)"
        else:
            weight_label = "Weight"

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot connectivity matrix
        im1 = axes[0, 0].imshow(weights_plot, cmap="cividis", aspect="auto")
        axes[0, 0].set_title("Connectivity Matrix")
        axes[0, 0].set_xlabel("Target Node")
        axes[0, 0].set_ylabel("Source Node")
        plt.colorbar(im1, ax=axes[0, 0], label=weight_label)

        # Plot weight distribution
        weights_nonzero = self.weights[self.weights > 0]
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
        im2 = axes[1, 0].imshow(self.delays, cmap="cividis_r", aspect="auto")
        axes[1, 0].set_title("Transmission Delays")
        axes[1, 0].set_xlabel("Target Node")
        axes[1, 0].set_ylabel("Source Node")
        plt.colorbar(im2, ax=axes[1, 0], label="Delay")

        # Plot delay distribution
        delays_nonzero = self.delays[self.delays > 0]
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
            f"{self.__class__.__name__}: {self.n_nodes} nodes, sparsity={self.sparsity:.3f}, symmetric={self.symmetric}, max_delay={self.max_delay:.2f}",
            fontsize=12,
            y=0.995,
        )

        plt.tight_layout()
        return fig, axes

    def tree_flatten(self):
        """Flatten DenseDelayGraph for JAX PyTree."""
        children = (self._weights, self._delays)  # Array data
        aux_data = (
            self._n_nodes,
            self._symmetric,
            self._sparsity,
            self._max_delay,
            self._region_labels,
        )  # Static metadata
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct DenseDelayGraph from PyTree data."""
        obj = object.__new__(cls)
        obj._weights = children[0]
        obj._delays = children[1]
        obj._n_nodes = aux_data[0]
        obj._symmetric = aux_data[1]
        obj._sparsity = aux_data[2]
        obj._max_delay = aux_data[3]
        obj._region_labels = aux_data[4]
        return obj


# Backward compatibility aliases
Graph = DenseGraph
DelayGraph = DenseDelayGraph
