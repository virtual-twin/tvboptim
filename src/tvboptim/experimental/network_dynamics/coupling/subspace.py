"""Subspace coupling: operates on regional/cluster level.

This module implements coupling that aggregates node states to regional states,
applies coupling at the regional level, then distributes back to nodes.

Use cases:
- Brain networks: 1000s of voxels → 100s of brain regions
- Multi-scale networks: Fine-grain nodes → coarse-grain clusters
- Hierarchical coupling: Different coupling strengths at different scales
"""

from typing import Callable, Optional, Tuple

import jax.numpy as jnp

from ..core.bunch import Bunch
from ..graph.base import AbstractGraph
from .base import AbstractCoupling


class _RegionalNetworkContext:
    """Lightweight Network-like interface for inner coupling preparation.

    Provides minimal interface for coupling.prepare(): graph, dynamics, get_history().
    Aggregates node-level history to regional level on-the-fly.

    Parameters
    ----------
    graph : AbstractGraph
        Regional graph
    dynamics : AbstractDynamics
        For state name resolution
    node_network : Network
        Node-level network for history aggregation
    region_one_hot_normalized : ndarray or BCOO
        Aggregation matrix
    n_regions : int
        Number of regions
    aggregator : callable
        Aggregation function (typically SubspaceCoupling.aggregate)
    """

    def __init__(
        self,
        graph: AbstractGraph,
        dynamics,
        node_network,
        region_one_hot_normalized: jnp.ndarray,
        n_regions: int,
        aggregator: Callable,
    ):
        self.graph = graph
        self.dynamics = dynamics
        self._node_network = node_network
        self._region_one_hot_normalized = region_one_hot_normalized
        self._n_regions = n_regions
        self._aggregator = aggregator

    def get_history(self, dt: float) -> Optional[jnp.ndarray]:
        """Get aggregated regional history buffer.

        Parameters
        ----------
        dt : float
            Integration timestep

        Returns
        -------
        ndarray, shape (n_steps, n_states, n_regions) or None
            Aggregated history if delays present, None otherwise
        """
        # Check if node network has custom history set
        if self._node_network._history is None:
            return self.get_initial_history(dt)
        else:
            return self._extract_regional_history_window(dt)

    def get_initial_history(self, dt: float) -> Optional[jnp.ndarray]:
        """Create initial history from aggregated node initial states.

        Parameters
        ----------
        dt : float
            Integration timestep

        Returns
        -------
        ndarray, shape (n_steps, n_states, n_regions) or None
            Initial regional history or None if no delays
        """
        # Check if regional graph has delays
        if not hasattr(self.graph, "max_delay") or self.graph.max_delay == 0.0:
            return None

        # Regional graph has delays, so we need to create history
        # Even if node network has no delays, we need to initialize regional history
        n_steps = int(jnp.ceil(self.graph.max_delay / dt))

        # Get node-level initial state (not history, just initial state)
        # Aggregate it and broadcast to create regional history
        node_initial_state = self._node_network.initial_state  # [n_states, n_nodes]

        # Create minimal coupling_data for aggregator
        # (contains only the precomputed fields needed for aggregation)
        aggregation_data = Bunch(
            region_one_hot_normalized=self._region_one_hot_normalized
        )

        regional_initial_state = self._aggregator(node_initial_state, aggregation_data)

        # Broadcast to create history buffer
        regional_history = jnp.broadcast_to(
            regional_initial_state[None, :, :],
            (n_steps, regional_initial_state.shape[0], self._n_regions),
        )

        return regional_history

    def _extract_regional_history_window(self, dt: float) -> jnp.ndarray:
        """Extract and aggregate custom node history to regional level.

        Parameters
        ----------
        dt : float
            Integration timestep

        Returns
        -------
        ndarray, shape (n_steps, n_states, n_regions)
            Aggregated regional history window
        """
        from ..utils.history import extract_history_window

        # Create minimal coupling_data for aggregator
        aggregation_data = Bunch(
            region_one_hot_normalized=self._region_one_hot_normalized
        )

        # Define transformation: node states -> regional states
        def aggregate_to_regional(node_state_at_t: jnp.ndarray) -> jnp.ndarray:
            """Aggregate node states [n_states, n_nodes] to [n_states, n_regions]"""
            return self._aggregator(node_state_at_t, aggregation_data)

        # Use shared utility with aggregation transform
        return extract_history_window(
            hist_ts=self._node_network._history.ts,
            hist_ys=self._node_network._history.ys,
            max_delay=self.graph.max_delay,
            dt=dt,
            transform_fn=aggregate_to_regional,
        )


class SubspaceCoupling(AbstractCoupling):
    """Coupling on regional subspace: aggregate nodes → couple regions → distribute.

    Performs three-stage computation:
    1. Aggregate node states to regional states (default: mean)
    2. Apply inner coupling on regional graph
    3. Distribute regional results to nodes (default: broadcast)

    Parameters
    ----------
    inner_coupling : AbstractCoupling
        Coupling applied at regional level
    region_mapping : ndarray, shape (n_nodes,)
        Maps each node to region ID (0 to n_regions-1)
    regional_graph : AbstractGraph
        Regional connectivity (n_nodes must equal n_regions)
    use_sparse : bool, default=True
        Use sparse BCOO format for aggregation (memory efficient)
    **kwargs
        Additional parameters for custom subclasses

    Attributes
    ----------
    n_regions : int
        Number of regions from region_mapping
    N_OUTPUT_STATES : int
        Output coupling dimensions from inner_coupling

    Notes
    -----
    Customization: Override ``aggregate()`` or ``distribute()`` for custom aggregation/
    distribution strategies beyond mean/broadcast.

    Examples
    --------
    Surface-level network with regional delayed coupling:

    >>> region_mapping = jnp.array([...])  # [n_vertices] -> region IDs
    >>> regional_graph = DenseDelayGraph.from_weights_delays(SC, delays)
    >>>
    >>> coupling = SubspaceCoupling(
    ...     inner_coupling=DelayedLinearCoupling(incoming_states='S', G=0.5),
    ...     region_mapping=region_mapping,
    ...     regional_graph=regional_graph
    ... )
    """

    def __init__(
        self,
        inner_coupling: AbstractCoupling,
        region_mapping: jnp.ndarray,
        regional_graph: AbstractGraph,
        use_sparse: bool = True,
        **kwargs,
    ):
        """
        Raises
        ------
        ValueError
            If regional_graph.n_nodes != max(region_mapping) + 1
        """
        # Extract state names from inner coupling
        super().__init__(
            incoming_states=inner_coupling.INCOMING_STATE_NAMES,
            local_states=inner_coupling.LOCAL_STATE_NAMES,
            **kwargs,
        )

        self.inner_coupling = inner_coupling
        self.region_mapping = region_mapping
        self.regional_graph = regional_graph
        self.use_sparse = use_sparse
        self.N_OUTPUT_STATES = inner_coupling.N_OUTPUT_STATES

        # Validate region_mapping
        self.n_regions = int(jnp.max(region_mapping) + 1)
        if self.regional_graph.n_nodes != self.n_regions:
            raise ValueError(
                f"Regional graph has {self.regional_graph.n_nodes} nodes but "
                f"region_mapping defines {self.n_regions} regions. They must match."
            )

    def prepare(self, network, dt: float, t0: float, t1: float) -> Tuple[Bunch, Bunch]:
        """Prepare aggregation matrices, regional context, and inner coupling.

        Parameters
        ----------
        network : Network
            Node-level network instance
        dt : float
            Integration timestep
        t0, t1 : float
            Simulation time window

        Returns
        -------
        coupling_data : Bunch
            Static data with region_one_hot_normalized, inner_data
        coupling_state : Bunch
            Mutable state with inner_state, cached_regional_state

        Raises
        ------
        ValueError
            If len(region_mapping) != network.graph.n_nodes
        """
        # Validate region_mapping size matches network
        if len(self.region_mapping) != network.graph.n_nodes:
            raise ValueError(
                f"region_mapping has {len(self.region_mapping)} nodes but "
                f"network has {network.graph.n_nodes} nodes. They must match."
            )

        # Precompute static data for aggregation/distribution
        n_nodes = network.graph.n_nodes

        if self.use_sparse:
            # Build sparse BCOO aggregation matrix
            import jax.experimental.sparse as jsparse

            # Create indices: one entry per node mapping to its region
            # indices[i] = [i, region_mapping[i]]
            indices = jnp.column_stack(
                [jnp.arange(n_nodes), self.region_mapping]
            )  # [n_nodes, 2]

            # Count nodes per region for normalization
            region_counts = jnp.array(
                [jnp.sum(self.region_mapping == r) for r in range(self.n_regions)]
            )

            # Create normalized values: 1/count[r] for each node in region r
            values = 1.0 / region_counts[self.region_mapping]  # [n_nodes]

            # Create BCOO sparse matrix [n_nodes, n_regions]
            region_one_hot_normalized = jsparse.BCOO(
                (values, indices), shape=(n_nodes, self.n_regions)
            )
        else:
            # Dense version (original)
            region_one_hot = jnp.eye(self.n_regions)[self.region_mapping]
            region_counts = jnp.sum(region_one_hot, axis=0)
            region_one_hot_normalized = region_one_hot / region_counts[None, :]

        # Create regional context for inner coupling preparation
        regional_context = self._create_regional_context(
            network, region_one_hot_normalized
        )

        # Prepare inner coupling with regional context
        inner_data, inner_state = self.inner_coupling.prepare(
            regional_context, dt, t0, t1
        )

        # Precompute initial aggregated regional state for caching
        # This avoids redundant aggregation in first compute() call
        aggregation_data_for_cache = Bunch(
            region_one_hot_normalized=region_one_hot_normalized
        )
        initial_regional_state = self.aggregate(
            network.initial_state, aggregation_data_for_cache
        )

        coupling_data = Bunch(
            n_regions=self.n_regions,
            region_one_hot_normalized=region_one_hot_normalized,  # Normalized for mean aggregation
            region_mapping=self.region_mapping,  # For distribute()
            inner_data=inner_data,
        )

        coupling_state = Bunch(
            inner_state=inner_state,
            cached_regional_state=initial_regional_state,  # Cache to avoid redundant aggregation
        )

        return coupling_data, coupling_state

    def _create_regional_context(self, node_network, region_one_hot_normalized):
        """Create minimal network-like context for inner coupling preparation.

        Parameters
        ----------
        node_network : Network
            Node-level network instance
        region_one_hot_normalized : ndarray or BCOO
            Normalized aggregation matrix

        Returns
        -------
        _RegionalNetworkContext
            Provides graph, dynamics, get_history() interface for inner coupling
        """
        return _RegionalNetworkContext(
            graph=self.regional_graph,
            dynamics=node_network.dynamics,
            node_network=node_network,
            region_one_hot_normalized=region_one_hot_normalized,
            n_regions=self.n_regions,
            aggregator=self.aggregate,
        )

    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        coupling_data: Bunch,
        coupling_state: Bunch,
        params: Bunch,
        graph: AbstractGraph,
    ) -> jnp.ndarray:
        """Compute coupling using cached aggregated state.

        Parameters
        ----------
        t : float
            Current time
        state : ndarray, shape (n_states, n_nodes)
            Not used - cached state from previous update_state() avoids aggregation
        coupling_data : Bunch
            Static regional structures
        coupling_state : Bunch
            Contains cached_regional_state, inner_state
        params : Bunch
            Not used - inner coupling has own params
        graph : AbstractGraph
            Not used - regional_graph used instead

        Returns
        -------
        ndarray, shape (n_coupling_inputs, n_nodes)
            Node-level coupling input
        """
        # 1. Use cached aggregated regional state (computed in previous update_state)
        # This avoids redundant aggregation - the cached state is already aggregated
        regional_state = coupling_state.cached_regional_state

        # 2. Compute coupling at regional level
        regional_coupling = self.inner_coupling.compute(
            t,
            regional_state,
            coupling_data.inner_data,
            coupling_state.inner_state,
            self.inner_coupling.params,  # Use inner coupling's params
            self.regional_graph,
        )

        # 3. Distribute: regional coupling → node coupling
        node_coupling = self.distribute(regional_coupling, coupling_data)

        return node_coupling

    def update_state(
        self, coupling_data: Bunch, coupling_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """Update inner coupling state and cache aggregated state.

        Parameters
        ----------
        coupling_data : Bunch
            Static regional structures
        coupling_state : Bunch
            Current state with inner_state, cached_regional_state
        new_state : ndarray, shape (n_states, n_nodes)
            Network state after integration step

        Returns
        -------
        Bunch
            Updated inner_state and cached_regional_state for next compute()
        """
        # Aggregate new node state to regional state
        # This will be cached for the next compute() call, avoiding redundant aggregation
        regional_state = self.aggregate(new_state, coupling_data)

        # Update inner coupling state with regional state
        new_inner_state = self.inner_coupling.update_state(
            coupling_data.inner_data, coupling_state.inner_state, regional_state
        )

        return Bunch(
            inner_state=new_inner_state,
            cached_regional_state=regional_state,  # Cache for next compute()
        )

    # ========================================================================
    # Customizable Methods (like pre/post pattern)
    # ========================================================================

    def aggregate(self, node_state: jnp.ndarray, coupling_data: Bunch) -> jnp.ndarray:
        """Aggregate node states to regional states (default: mean).

        Override for custom aggregation strategies (sum, weighted, etc).

        Parameters
        ----------
        node_state : ndarray, shape (n_states, n_nodes)
            Node-level states
        coupling_data : Bunch
            Contains region_one_hot_normalized for aggregation

        Returns
        -------
        ndarray, shape (n_states, n_regions)
            Regional states

        Notes
        -----
        Default uses normalized one-hot matrix: node_state @ region_one_hot_normalized
        Supports both dense and sparse (BCOO) matrices.
        """
        # Mean aggregation via single matrix multiply
        # [n_states, n_nodes] @ [n_nodes, n_regions] → [n_states, n_regions]
        # Works for both dense and sparse (BCOO) matrices
        regional_state = node_state @ coupling_data.region_one_hot_normalized

        return regional_state

    def distribute(
        self, regional_coupling: jnp.ndarray, coupling_data: Bunch
    ) -> jnp.ndarray:
        """Distribute regional coupling to nodes (default: broadcast).

        Override for custom distribution strategies (scaled, weighted, etc).

        Parameters
        ----------
        regional_coupling : ndarray, shape (n_coupling_inputs, n_regions)
            Regional coupling values
        coupling_data : Bunch
            Contains region_mapping for distribution

        Returns
        -------
        ndarray, shape (n_coupling_inputs, n_nodes)
            Node-level coupling

        Notes
        -----
        Default broadcasts: each node receives its region's value via indexing.
        """
        # Broadcast: all nodes in region r get regional_coupling[:, r]
        node_coupling = regional_coupling[:, coupling_data.region_mapping]

        return node_coupling

    def describe(self) -> dict:
        """Generate description for network printer.

        Returns
        -------
        dict
            Coupling metadata with regional structure info
        """
        # Check if inner coupling has describe() method
        if hasattr(self.inner_coupling, "describe"):
            inner_desc = self.inner_coupling.describe()
        else:
            # Fallback for couplings without describe()
            inner_desc = {}

        # Determine type from inner coupling class
        from .base import DelayedCoupling

        coupling_type = (
            "delayed"
            if isinstance(self.inner_coupling, DelayedCoupling)
            else "instantaneous"
        )

        # Build description with subspace-specific information
        desc = {
            "class_name": f"Subspace({self.inner_coupling.__class__.__name__})",
            "type": inner_desc.get("type", coupling_type),  # Use inner type or infer
            "incoming_states": inner_desc.get("incoming_states", []),
            "local_states": inner_desc.get("local_states", []),
            "params": inner_desc.get("params", {}),
            "network_form": inner_desc.get("network_form", ""),
            "pre_form": inner_desc.get("pre_form"),
            "post_form": inner_desc.get("post_form"),
            # Subspace-specific fields
            "n_regions": self.n_regions,
            "aggregation": self._get_aggregation_method_name(),
            "distribution": self._get_distribution_method_name(),
            "inner_coupling_name": self.inner_coupling.__class__.__name__,
        }

        # Modify network_form to indicate regional operation
        if desc["network_form"]:
            # Replace node index j with region index r (Unicode subscripts)
            regional_form = (
                desc["network_form"]
                .replace("ⱼ", "ᵣ")
                .replace("wᵢⱼ", "wᵢᵣ")
                .replace("τᵢⱼ", "τᵢᵣ")
            )
            desc["network_form"] = f"[{self.n_regions} regions] {regional_form}"

        # Add regional max_delay if applicable
        if desc["type"] == "delayed" and hasattr(self.regional_graph, "max_delay"):
            desc["max_delay"] = self.regional_graph.max_delay

        return desc

    def _get_aggregation_method_name(self) -> str:
        """Get the name of the aggregation method (for printing)."""
        # Check if user overrode aggregate method
        if type(self).aggregate is not SubspaceCoupling.aggregate:
            return "custom"
        return "mean"

    def _get_distribution_method_name(self) -> str:
        """Get the name of the distribution method (for printing)."""
        # Check if user overrode distribute method
        if type(self).distribute is not SubspaceCoupling.distribute:
            return "custom"
        return "broadcast"
