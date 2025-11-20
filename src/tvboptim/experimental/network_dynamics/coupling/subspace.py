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
    """Minimal context for regional coupling preparation.

    This lightweight class provides only the interface needed by coupling.prepare():
    - graph: Regional graph
    - dynamics: Dynamics model (for state name resolution)
    - get_initial_history(): Aggregated regional history

    This avoids the need to create a full Network instance and deal with
    property overriding issues.
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
        """Initialize regional network context.

        Args:
            graph: Regional graph
            dynamics: Dynamics model
            node_network: Node-level network (for history aggregation)
            region_one_hot_normalized: [n_nodes, n_regions] normalized one-hot encoding
            n_regions: Number of regions
            aggregator: Function to aggregate states (typically SubspaceCoupling.aggregate)
        """
        self.graph = graph
        self.dynamics = dynamics
        self._node_network = node_network
        self._region_one_hot_normalized = region_one_hot_normalized
        self._n_regions = n_regions
        self._aggregator = aggregator

    def get_initial_history(self, dt: float) -> Optional[jnp.ndarray]:
        """Return aggregated regional history.

        Args:
            dt: Integration timestep

        Returns:
            Regional history [n_steps, n_states, n_regions] or None if no delays
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


class SubspaceCoupling(AbstractCoupling):
    """Coupling that operates on a regional/cluster subspace.

    Aggregates node states to regional states, applies inner coupling
    on regional graph, then distributes results back to nodes.

    Notes
    -----
    The coupling operates in three stages:

    1. **Aggregation**: Node states are aggregated to regional states (default: mean)

       $$s_r = \\text{aggregate}(\\{s_i : i \\in \\text{region } r\\})$$

    2. **Regional coupling**: Inner coupling is applied on the regional graph

       $$c_r = f_{\\text{coupling}}(s, \\text{regional_graph})$$

    3. **Distribution**: Regional coupling is distributed back to nodes (default: broadcast)

       $$c_i = \\text{distribute}(c_{\\text{region}[i]})$$

    Users can customize aggregation/distribution by overriding:

    - ``aggregate()``: node_state → regional_state (default: mean)
    - ``distribute()``: regional_coupling → node_coupling (default: broadcast)

    Parameters
    ----------
    inner_coupling : AbstractCoupling
        Coupling to apply at regional level (can be any coupling: Linear, Delayed, custom, etc.)
    region_mapping : jnp.ndarray
        Array with shape ``[n_nodes]`` mapping nodes to region IDs (0 to n_regions-1)
    regional_graph : AbstractGraph
        Graph defining region-to-region connectivity (must have n_nodes == n_regions)
    use_sparse : bool, optional
        If True, use BCOO sparse format for aggregation matrix (default: ``True``)
    **kwargs
        Additional parameters stored in ``self.params`` for custom subclasses

    Attributes
    ----------
    inner_coupling : AbstractCoupling
        Coupling to apply at regional level
    region_mapping : jnp.ndarray
        Array mapping nodes to region IDs
    regional_graph : AbstractGraph
        Graph defining region-to-region connectivity
    n_regions : int
        Number of regions (derived from region_mapping)
    N_OUTPUT_STATES : int
        Number of output coupling states (inherited from inner_coupling)

    Examples
    --------
    >>> # 1000 nodes mapped to 90 brain regions
    >>> region_mapping = jnp.array([...])  # [1000] with values 0-89
    >>> regional_graph = DelayGraph(weights_90x90, delays_90x90)
    >>>
    >>> coupling = SubspaceCoupling(
    ...     inner_coupling=DelayedLinearCoupling(incoming_states='S', G=0.5),
    ...     region_mapping=region_mapping,
    ...     regional_graph=regional_graph
    ... )
    >>>
    >>> # Use in network with mixed coupling
    >>> network = Network(
    ...     dynamics=ReducedWongWang(),
    ...     coupling={
    ...         'instant': LinearCoupling(incoming_states='S', G=0.2),
    ...         'delayed': coupling  # Regional delayed coupling
    ...     },
    ...     graph=node_graph
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
        """Initialize subspace coupling.

        Args:
            inner_coupling: Coupling to apply at regional level
                           Can be any coupling (Linear, Delayed, custom, etc.)
            region_mapping: [n_nodes] array mapping nodes to region IDs (0 to n_regions-1)
                           Example: [0, 0, 1, 1, 2, 2] maps 6 nodes to 3 regions
            regional_graph: Graph defining region-to-region connectivity
                           Must have n_nodes == n_regions
            use_sparse: If True, use BCOO sparse format for aggregation matrix
                       Can improve memory usage for large networks with many regions
            **kwargs: Additional parameters (stored in self.params for custom subclasses)

        Raises:
            ValueError: If regional_graph size doesn't match number of regions
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

    def prepare(self, network, dt: float) -> Tuple[Bunch, Bunch]:
        """Prepare regional structures and inner coupling.

        Args:
            network: Network instance (node-level)
            dt: Integration timestep

        Returns:
            coupling_data: Bunch with regional structures and inner_data
            coupling_state: Bunch with inner_state

        Raises:
            ValueError: If region_mapping size doesn't match network size
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
        inner_data, inner_state = self.inner_coupling.prepare(regional_context, dt)

        coupling_data = Bunch(
            n_regions=self.n_regions,
            region_one_hot_normalized=region_one_hot_normalized,  # Normalized for mean aggregation
            region_mapping=self.region_mapping,  # For distribute()
            inner_data=inner_data,
        )

        coupling_state = Bunch(inner_state=inner_state)

        return coupling_data, coupling_state

    def _create_regional_context(self, node_network, region_one_hot_normalized):
        """Create regional context for inner coupling preparation.

        This creates a lightweight mock that provides only what coupling.prepare()
        needs: graph, dynamics, and get_initial_history().

        Args:
            node_network: Node-level Network instance
            region_one_hot_normalized: Precomputed normalized one-hot encoding

        Returns:
            _RegionalNetworkContext instance with regional graph and aggregated history
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
        """Compute subspace coupling: aggregate → couple → distribute.

        Args:
            t: Current simulation time
            state: Current network state [n_states, n_nodes]
            coupling_data: Precomputed regional structures
            coupling_state: Inner coupling state
            params: Coupling parameters (not used - inner coupling has its own params)
            graph: Node-level graph (not used - regional_graph used instead)

        Returns:
            Node-level coupling input [n_coupling_inputs, n_nodes]
        """
        # 1. Aggregate: node states → regional states
        regional_state = self.aggregate(state, coupling_data)

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
        """Update inner coupling state (e.g., regional history buffer).

        Args:
            coupling_data: Precomputed regional structures
            coupling_state: Current inner coupling state
            new_state: New network state [n_states, n_nodes]

        Returns:
            Updated coupling_state with new inner_state
        """
        # Aggregate new node state to regional state
        regional_state = self.aggregate(new_state, coupling_data)

        # Update inner coupling state with regional state
        new_inner_state = self.inner_coupling.update_state(
            coupling_data.inner_data, coupling_state.inner_state, regional_state
        )

        return Bunch(inner_state=new_inner_state)

    # ========================================================================
    # Customizable Methods (like pre/post pattern)
    # ========================================================================

    def aggregate(self, node_state: jnp.ndarray, coupling_data: Bunch) -> jnp.ndarray:
        """Aggregate node states to regional states. Default: mean.

        Override this method to customize aggregation strategy.

        Args:
            node_state: [n_states, n_nodes] node-level states
            coupling_data: Bunch containing precomputed aggregation data
                          (region_one_hot_normalized, etc.)

        Returns:
            regional_state: [n_states, n_regions] region-level states

        Examples:
            Mean (default): Use normalized region_one_hot_normalized (single matmul)
            Sum: Use unnormalized region_one_hot instead
            Weighted: Add custom aggregation matrix to coupling_data in prepare()

        Note:
            coupling_data is populated in prepare() and contains all precomputed
            static data needed for aggregation. By default, region_one_hot_normalized
            is pre-divided by region counts, so a single matrix multiply computes the mean.
            Supports both dense and BCOO sparse matrices automatically.
        """
        # Mean aggregation via single matrix multiply
        # [n_states, n_nodes] @ [n_nodes, n_regions] → [n_states, n_regions]
        # Works for both dense and sparse (BCOO) matrices
        regional_state = node_state @ coupling_data.region_one_hot_normalized

        return regional_state

    def distribute(
        self, regional_coupling: jnp.ndarray, coupling_data: Bunch
    ) -> jnp.ndarray:
        """Distribute regional coupling to node-level coupling. Default: broadcast.

        Override this method to customize distribution strategy.

        Args:
            regional_coupling: [n_coupling_inputs, n_regions] regional coupling
            coupling_data: Bunch containing precomputed distribution data
                          (region_mapping, region_counts, etc.)

        Returns:
            node_coupling: [n_coupling_inputs, n_nodes] node-level coupling

        Examples:
            Broadcast (default): All nodes in region get same value
            Scaled: Divide by region_counts from coupling_data
            Weighted: Add custom distribution weights to coupling_data in prepare()

        Note:
            coupling_data is populated in prepare() and contains all precomputed
            static data needed for distribution. This design allows flexible
            customization without changing the API.
        """
        # Broadcast: all nodes in region r get regional_coupling[:, r]
        node_coupling = regional_coupling[:, coupling_data.region_mapping]

        return node_coupling

    def describe(self) -> dict:
        """Generate human-readable description of subspace coupling for printing.

        Delegates to inner coupling and adds regional/subspace information.

        Returns:
            Dictionary with coupling description including regional structure
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
