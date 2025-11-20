"""New coupling system with self-contained state management.

This module implements the coupling architecture with:
- prepare(), compute(), update_state() interface
- Bunch-based coupling_data and coupling_state
- Support for multi-coupling networks
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import jax.numpy as jnp

from ..core.bunch import Bunch
from ..graph.base import AbstractGraph

# ============================================================================
# Helper Functions
# ============================================================================


def _get_n_states(state_idx) -> int:
    """Get number of states from index array or scalar."""
    if hasattr(state_idx, "__len__"):
        return len(state_idx)
    elif state_idx is not None:
        return 1
    else:
        return 0


def _ensure_dense(arr):
    """Convert sparse array to dense if needed."""
    return arr.todense() if hasattr(arr, "todense") else arr


def _prepare_bcoo_indices(
    graph, incoming_idx
) -> Tuple[Optional[jnp.ndarray], Optional[tuple], bool]:
    """Prepare BCOO sparse indices for 3D tensor [n_incoming, n_nodes, n_nodes].

    Args:
        graph: Graph instance (checked for sparsity)
        incoming_idx: Incoming state indices

    Returns:
        Tuple of (bcoo_indices, bcoo_shape, use_sparse_incoming):
        - bcoo_indices: [n_batch*nnz, 3] indices for BCOO construction
        - bcoo_shape: (n_incoming, n_nodes, n_nodes) shape tuple
        - use_sparse_incoming: bool flag whether to use sparse optimization
    """
    from ..graph.sparse import SparseDelayGraph, SparseGraph

    if not isinstance(graph, (SparseGraph, SparseDelayGraph)):
        return None, None, False

    n_incoming = _get_n_states(incoming_idx)

    # Only use sparse incoming if we actually have incoming states
    if n_incoming == 0:
        return None, None, False

    nnz = graph.nnz
    n_nodes = graph.n_nodes

    # Build 3D indices [n_batch*nnz, 3] for BCOO [n_incoming, n_nodes, n_nodes]
    batch_indices = jnp.repeat(jnp.arange(n_incoming), nnz)
    spatial_indices = jnp.tile(graph.weights.indices, (n_incoming, 1))
    bcoo_indices = jnp.column_stack([batch_indices, spatial_indices])

    bcoo_shape = (n_incoming, n_nodes, n_nodes)

    return bcoo_indices, bcoo_shape, True


def _sparse_weighted_sum(pre_states, graph, is_bcoo: bool = False):
    """Perform sparse weighted sum: sum over edges of (pre_states * weights).

    Handles both BCOO sparse pre_states and dense pre_states with sparse graph.

    Args:
        pre_states: Either BCOO [n_output, n_nodes, n_nodes] or dense array
        graph: Sparse graph with weights
        is_bcoo: True if pre_states is BCOO, False if dense

    Returns:
        Dense array [n_output, n_nodes] after weighted sum
    """
    import jax.experimental.sparse as jsparse

    indices = graph.weights.indices  # [nnz, 2]
    n_output = pre_states.shape[0]
    nnz = graph.nnz

    if is_bcoo or hasattr(pre_states, "todense"):
        # BCOO path: pre_states.data is flat [n_output * nnz]
        # Reshape pre data: [n_output * nnz] -> [n_output, nnz]
        pre_data_batched = pre_states.data.reshape(n_output, nnz)

        # Element-wise multiply: [n_output, nnz] * [nnz] -> [n_output, nnz]
        weighted_data = pre_data_batched * graph.weights.data[None, :]

        # Sum over edges for each output dimension
        summed_list = []
        for i in range(n_output):
            weighted_bcoo_i = jsparse.BCOO(
                (weighted_data[i], indices), shape=(graph.n_nodes, graph.n_nodes)
            )
            summed_i = jsparse.bcoo_reduce_sum(weighted_bcoo_i, axes=(1,))
            summed_list.append(_ensure_dense(summed_i))

        return jnp.stack(summed_list, axis=0)
    else:
        # Dense pre_states path: extract values at sparse locations
        summed_list = []
        for i in range(n_output):
            # Extract values at sparse edges only
            pre_values = pre_states[i, indices[:, 0], indices[:, 1]]  # [nnz]

            # Element-wise multiply: pre.data * weights.data
            weighted_values = pre_values * graph.weights.data  # [nnz]

            # Reconstruct BCOO and sum over source nodes (axis=1)
            pre_bcoo = jsparse.BCOO(
                (weighted_values, indices), shape=graph.weights.shape
            )
            summed_i_sparse = jsparse.bcoo_reduce_sum(pre_bcoo, axes=(1,))
            summed_list.append(_ensure_dense(summed_i_sparse))

        return jnp.stack(summed_list, axis=0)


# ============================================================================
# Coupling Classes
# ============================================================================


class AbstractCoupling(ABC):
    """Ultra-minimal interface for completely custom coupling implementations.

    Use this base class only when you need full control over coupling computation
    and the standard matrix multiplication patterns don't apply.

    Attributes:
        N_OUTPUT_STATES: Number of output states after coupling
        DEFAULT_PARAMS: Default coupling parameters as a Bunch
    """

    N_OUTPUT_STATES: int = 0
    DEFAULT_PARAMS: Bunch = Bunch()

    def __init__(self, incoming_states=None, local_states=None, **kwargs):
        """Initialize coupling with state names and parameter overrides.

        Args:
            incoming_states: State names from connected nodes (str, tuple, or list)
                            Optional - what states to collect from incoming connections
            local_states: State names from current node (str, tuple, or list)
                         Optional - what states to use from local node
            **kwargs: Parameter overrides for DEFAULT_PARAMS

        Raises:
            ValueError: If neither incoming_states nor local_states is provided,
                       or if unknown parameters given
        """
        # At least one state source must be specified
        if incoming_states is None and local_states is None:
            raise ValueError(
                f"{self.__class__.__name__} requires at least one of 'incoming_states' "
                f"or 'local_states'. A coupling must couple something!"
            )

        # Store state names as instance attributes
        self.INCOMING_STATE_NAMES = (
            incoming_states if incoming_states is not None else []
        )
        self.LOCAL_STATE_NAMES = local_states if local_states is not None else []

        # Create instance parameters by copying defaults and updating with kwargs
        self.params = self.DEFAULT_PARAMS.copy() if self.DEFAULT_PARAMS else Bunch()
        for key, value in kwargs.items():
            if key not in self.DEFAULT_PARAMS:
                raise ValueError(
                    f"Unknown parameter '{key}' for {self.__class__.__name__}. "
                    f"Available parameters: {list(self.DEFAULT_PARAMS.keys())}"
                )
            self.params[key] = value

    @abstractmethod
    def prepare(self, network, dt: float) -> Tuple[Bunch, Bunch]:
        """Prepare coupling for simulation.

        Handles all setup logic that was previously in solve.py:
        - State index computation
        - History buffer initialization (for delays)
        - Parameter validation

        Args:
            network: Network instance with graph, dynamics, initial_state
            dt: Integration timestep

        Returns:
            Tuple of (coupling_data, coupling_state):

            coupling_data: Bunch
                Static precomputed data (stored outside scan carry):
                - incoming_indices: State indices to read from
                - local_indices: Local state indices
                - delay_indices: Delay step indices (for delayed coupling)
                - Other static precomputed data
                NOTE: Does NOT contain weights/delays - access via graph parameter

            coupling_state: Bunch
                Mutable internal state (stored in scan carry):
                - history: Circular buffer for delays (if applicable)
                - [empty Bunch() for instantaneous coupling]
        """
        pass

    @abstractmethod
    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        coupling_data: Bunch,
        coupling_state: Bunch,
        params: Bunch,
        graph: AbstractGraph,
    ) -> jnp.ndarray:
        """Compute coupling input during simulation.

        Replaces the coupling_fun built by networks. Handles all coupling
        computation including delays, matrix multiplication, pre/post transforms.

        Args:
            t: Current simulation time
            state: Current network state [n_states, n_nodes]
            coupling_data: Precomputed static data (indices, etc.)
            coupling_state: Mutable internal state (history buffers, etc.)
            params: Coupling parameters (G, a, b, etc.)
            graph: Network graph for accessing weights/delays

        Returns:
            Coupling input [n_coupling_inputs, n_nodes]
        """
        pass

    @abstractmethod
    def update_state(
        self, coupling_data: Bunch, coupling_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """Update coupling internal state after integration step.

        Handles state-dependent updates like history buffer management.

        Args:
            coupling_data: Precomputed arrays (for reference)
            coupling_state: Current internal state
            new_state: New network state after integration [n_states, n_nodes]

        Returns:
            Updated coupling_state as Bunch
        """
        pass


class InstantaneousCoupling(AbstractCoupling):
    """Base class for coupling without delays (ODE/SDE systems).

    Handles standard weight matrix multiplication pattern:
    1. Extract incoming and local states
    2. Apply pre() transformation
    3. Matrix multiplication with connectivity weights
    4. Apply post() transformation

    Users typically only need to override pre() and/or post() methods.
    """

    def get_mode(self, incoming_idx, local_idx, n_nodes=2) -> str:
        """Detect coupling mode by calling pre() with dummy data.

        This allows us to optimize sparse graphs by knowing ahead of time
        whether the coupling uses vectorized or per-edge mode.

        Args:
            incoming_idx: Indices for incoming states (for sizing dummy data)
            local_idx: Indices for local states (for sizing dummy data)

        Returns:
            'vectorized' if pre() reduces dimensionality (2D output)
            'per_edge' if pre() preserves per-edge structure (3D output)
        """
        # Create dummy data with correct shapes based on actual state selection
        n_incoming = len(incoming_idx) if hasattr(incoming_idx, "__len__") else 1
        n_local = len(local_idx) if hasattr(local_idx, "__len__") else 1

        # Dummy per-edge tensor: [n_incoming, n_nodes, n_nodes] with n_nodes=2
        dummy_incoming = jnp.zeros((n_incoming, n_nodes, n_nodes))
        dummy_local = jnp.zeros((n_local, n_nodes))

        # Call pre() to check output shape
        result = self.pre(dummy_incoming, dummy_local, self.params)

        return "vectorized" if result.ndim == 2 else "per_edge"

    def prepare(self, network, dt: float) -> Tuple[Bunch, Bunch]:
        """Standard preparation for instantaneous coupling.

        Args:
            network: Network instance with graph, dynamics, initial_state
            dt: Integration timestep (not used for instantaneous coupling)

        Returns:
            coupling_data: Bunch with incoming_indices, local_indices, state_indices, mode
            coupling_state: Empty Bunch (no internal state)
        """
        graph = network.graph
        dynamics = network.dynamics

        # Resolve state indices
        incoming_idx = dynamics.name_to_index(self.INCOMING_STATE_NAMES)
        local_idx = dynamics.name_to_index(self.LOCAL_STATE_NAMES)

        # Detect mode for optimization (used to decide sparse strategy)
        mode = self.get_mode(incoming_idx, local_idx, n_nodes=graph.n_nodes)

        # Create state_indices for per-edge coupling support (from v1 system)
        # This allows incoming_states[:, state_indices] to expand to per-edge format
        n_nodes = graph.n_nodes
        state_indices = jnp.arange(n_nodes) * jnp.ones((n_nodes, n_nodes)).astype(int)

        # Store mode as boolean flags (JAX-traceable) instead of string
        is_per_edge = mode == "per_edge"

        # Precompute BCOO indices for sparse incoming states optimization
        bcoo_indices, bcoo_shape, use_sparse_incoming = _prepare_bcoo_indices(
            graph, incoming_idx
        )

        coupling_data = Bunch(
            incoming_indices=incoming_idx,
            local_indices=local_idx,
            state_indices=state_indices,
            is_per_edge=is_per_edge,
            bcoo_indices=bcoo_indices,
            bcoo_shape=bcoo_shape,
            use_sparse_incoming=use_sparse_incoming,
        )
        coupling_state = Bunch()  # Empty for instantaneous coupling

        return coupling_data, coupling_state

    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        coupling_data: Bunch,
        coupling_state: Bunch,
        params: Bunch,
        graph: AbstractGraph,
    ) -> jnp.ndarray:
        """Coupling computation with automatic vectorized vs per-edge dispatch.

        Supports two modes based on pre() output shape:
        1. Vectorized (2D): pre() returns [n_states, n_nodes] → use matmul
        2. Per-edge (3D): pre() returns [n_states, n_nodes, n_nodes] → element-wise multiply + sum

        Both modes support dense and sparse graphs automatically.
        For sparse graphs, per-edge expansion is kept sparse to save memory.

        Args:
            t: Current simulation time
            state: Current network state [n_states, n_nodes]
            coupling_data: Bunch with incoming_indices, local_indices, state_indices
            coupling_state: Empty Bunch (not used)
            params: Coupling parameters
            graph: Network graph for accessing weights (dense or sparse)

        Returns:
            Coupling input [n_coupling_inputs, n_nodes]
        """
        from ..graph.sparse import SparseGraph

        # Extract states
        local_states = state[coupling_data.local_indices]
        incoming_states = state[coupling_data.incoming_indices]

        # Build incoming_states_edge and compute coupling based on graph type
        if coupling_data.use_sparse_incoming:
            # SPARSE INCOMING: Build 3D BCOO to avoid dense materialization
            summed = self._compute_sparse_incoming(
                incoming_states, local_states, params, graph, coupling_data
            )

        elif isinstance(graph, SparseGraph) and coupling_data.is_per_edge:
            # SPARSE FALLBACK: Dense incoming with sparse graph
            incoming_states_edge = incoming_states[:, coupling_data.state_indices]
            pre_states = self.pre(incoming_states_edge, local_states, params)
            summed = _sparse_weighted_sum(pre_states, graph, is_bcoo=False)

        else:
            # DENSE or VECTORIZED
            incoming_states_edge = incoming_states[:, coupling_data.state_indices]
            pre_states = self.pre(incoming_states_edge, local_states, params)

            if pre_states.ndim == 2:
                # Vectorized mode: matmul
                summed = pre_states @ graph.weights
            elif pre_states.ndim == 3:
                # Per-edge mode: element-wise multiply + sum
                summed = jnp.sum(pre_states * graph.weights, axis=2)
            else:
                raise ValueError(
                    f"pre() returned unexpected shape: {pre_states.shape}. "
                    f"Expected 2D or 3D."
                )

        # Apply post-transform
        return self.post(summed, local_states, params)

    def _compute_sparse_incoming(
        self, incoming_states, local_states, params, graph, coupling_data
    ):
        """Compute coupling with sparse incoming states (BCOO optimization)."""
        import jax.experimental.sparse as jsparse

        indices = graph.weights.indices
        n_incoming = incoming_states.shape[0]

        # Extract values at source nodes for each incoming state
        incoming_data_list = []
        for i in range(n_incoming):
            values_at_sources = incoming_states[i, indices[:, 1]]
            incoming_data_list.append(values_at_sources)

        incoming_data_flat = jnp.concatenate(incoming_data_list)

        # Create 3D BCOO [n_incoming, n_nodes, n_nodes]
        incoming_states_edge = jsparse.BCOO(
            (incoming_data_flat, coupling_data.bcoo_indices),
            shape=coupling_data.bcoo_shape,
        )

        # Apply sparse pre-transform
        pre_states = self._sparse_pre(incoming_states_edge, local_states, params)

        # Perform sparse weighted sum
        return _sparse_weighted_sum(pre_states, graph, is_bcoo=True)

    def update_state(
        self, coupling_data: Bunch, coupling_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """No mutable state to update for instantaneous coupling.

        Args:
            coupling_data: Not used
            coupling_state: Empty Bunch
            new_state: Not used

        Returns:
            Unchanged coupling_state (empty Bunch)
        """
        return coupling_state

    def describe(self) -> dict:
        """Generate human-readable description of coupling for printing.

        Uses introspection of pre() and post() methods to infer mathematical form.
        Subclasses can override to provide custom descriptions.

        Returns:
            Dictionary with 'network_form', 'pre_form', 'post_form' keys
        """
        # Get incoming/local state descriptions (no subscripts here)
        incoming = self._format_state_list(self.INCOMING_STATE_NAMES)
        local = self._format_state_list(self.LOCAL_STATE_NAMES)

        # Check if pre() and post() are customized
        pre_form = self._infer_pre_form(incoming, local)
        post_form = self._infer_post_form()

        # Determine which states are used in the network form
        # FastLinearCoupling uses local states (vectorized mode)

        # Get subscripted version for network form
        state_with_subscript = self._format_state_list(
            self.INCOMING_STATE_NAMES if incoming else self.LOCAL_STATE_NAMES,
            with_subscript=True,
        )

        # Build network form with pre/post embedded
        # Use post_form directly if available (already has actual values substituted)
        base_sum = f"Σⱼ wᵢⱼ * {state_with_subscript}"

        if pre_form and post_form:
            # Substitute base into post_form
            network_form = post_form.replace(
                "(...)", f"Σⱼ wᵢⱼ * pre({state_with_subscript})"
            )
        elif pre_form:
            network_form = f"Σⱼ wᵢⱼ * pre({state_with_subscript})"
        elif post_form:
            # Substitute the sum into post form
            network_form = post_form.replace("(...)", base_sum)
        else:
            network_form = base_sum

        return {
            "network_form": network_form,
            "pre_form": pre_form if pre_form else None,
            "post_form": post_form if post_form else None,
        }

    def _format_state_list(self, states, with_subscript: bool = False) -> str:
        """Format state names for display.

        Args:
            states: State names (str, list, or tuple)
            with_subscript: If True, add subscript ⱼ to each state

        Returns:
            Formatted string like 'S', 'Sⱼ', '(y1ⱼ, y2ⱼ)', etc.
        """
        if isinstance(states, str):
            return states + "ⱼ" if with_subscript else states
        elif isinstance(states, (list, tuple)):
            if len(states) == 0:
                return ""
            elif len(states) == 1:
                state = states[0]
                return state + "ⱼ" if with_subscript else state
            else:
                # Multiple states: format as (state1ⱼ, state2ⱼ)
                if with_subscript:
                    formatted = [s + "ⱼ" for s in states]
                else:
                    formatted = states
                return f"({', '.join(formatted)})"
        return str(states)

    def _infer_pre_form(self, incoming: str, local: str) -> str:
        """Infer pre() transformation form by introspecting the method.

        Returns empty string if pre() is the default identity.
        """
        import inspect

        try:
            pre_source = inspect.getsource(self.pre)
            # Check if it's the default identity (returns incoming_states unchanged)
            if "return incoming_states" in pre_source and pre_source.count("\n") <= 5:
                return ""  # Default identity, don't show
        except (AttributeError, OSError, TypeError):
            pass

        # Check if there are meaningful parameters used in pre
        # For now, return empty - subclasses will override if needed
        return ""

    def _infer_post_form(self) -> str:
        """Infer post() transformation form by introspecting the method.

        Returns the post transformation with actual parameter values.
        """
        import inspect

        try:
            post_source = inspect.getsource(self.post)

            # Try to extract the return expression
            lines = post_source.strip().split("\n")
            for line in lines:
                if "return" in line:
                    # Extract expression after return
                    expr = line.split("return", 1)[1].strip()

                    # Substitute parameter values
                    for key, value in self.params.items():
                        # Replace params.key with actual value
                        expr = expr.replace(f"params.{key}", str(value))

                    # Clean up common patterns
                    expr = expr.replace("summed_inputs", "(...)")
                    expr = expr.replace("local_states", "local")

                    return expr
        except (AttributeError, OSError, TypeError, KeyError):
            pass

        # Fallback: check for common parameter patterns
        if "G" in self.params:
            return "G * (...)"

        return ""

    def pre(
        self, incoming_states: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Transform states before matrix multiplication. Default: identity (per-edge).

        Args:
            incoming_states: States from connected nodes in per-edge format
                           Shape: [n_incoming, n_nodes_target, n_nodes_source]
                           where [:, j, k] is state from source node k to target node j
            local_states: States from current node [n_local, n_nodes]
            params: Coupling parameters

        Returns:
            Transformed states - shape determines coupling mode:
            - [n_states, n_nodes]: Vectorized mode (uses matmul)
            - [n_states, n_nodes, n_nodes]: Per-edge mode (uses element-wise multiply + sum)

            Default returns incoming_states unchanged (3D) → per-edge mode
            This matches v1 behavior where default pre() is identity.
        """
        # Default: return incoming_states unchanged (identity)
        # This gives [n_incoming, n_nodes, n_nodes] shape → per-edge mode (like v1)
        return incoming_states

    def _sparse_pre(self, incoming_states, local_states: jnp.ndarray, params: Bunch):
        """Sparse version of pre() using sparsify decorator.

        Default implementation wraps self.pre() with sparsify.
        Subclasses can override this for custom sparse implementations.

        Args:
            incoming_states: BCOO tensor [n_incoming, n_nodes, n_nodes]
            local_states: Dense tensor [n_local, n_nodes]
            params: Coupling parameters

        Returns:
            BCOO tensor [n_output, n_nodes, n_nodes]
        """
        import jax.experimental.sparse as jsparse

        return jsparse.sparsify(self.pre)(incoming_states, local_states, params)

    @abstractmethod
    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Transform coupling after summation. Must be implemented by subclasses.

        Args:
            summed_inputs: Summed coupling inputs [n_coupling_inputs, n_nodes]
            local_states: States from current node [n_local, n_nodes]
            params: Coupling parameters

        Returns:
            Final coupling input [n_coupling_inputs, n_nodes]
        """
        pass


class DelayedCoupling(AbstractCoupling):
    """Base class for coupling with transmission delays (DDE/SDDE systems).

    Handles delayed coupling pattern:
    1. Extract delayed states from history buffer
    2. Apply pre() transformation
    3. Matrix multiplication with connectivity weights
    4. Apply post() transformation
    5. Update history buffer

    Users typically only need to override pre() and/or post() methods.
    """

    def prepare(self, network, dt: float) -> Tuple[Bunch, Bunch]:
        """Standard preparation for delayed coupling.

        Args:
            network: Network instance with graph, dynamics, initial_state
            dt: Integration timestep

        Returns:
            coupling_data: Bunch with indices, delay_indices, state_indices
            coupling_state: Bunch with history buffer
        """
        graph = network.graph
        dynamics = network.dynamics

        # Resolve state indices
        incoming_idx = dynamics.name_to_index(self.INCOMING_STATE_NAMES)
        local_idx = dynamics.name_to_index(self.LOCAL_STATE_NAMES)

        # Setup delay indexing (static)
        delays_dense = _ensure_dense(graph.delays)
        delay_indices = jnp.round(delays_dense / dt).astype(jnp.int32)
        state_indices = jnp.arange(graph.n_nodes)

        # Precompute BCOO indices for sparse incoming states optimization
        bcoo_indices, bcoo_shape, use_sparse_incoming = _prepare_bcoo_indices(
            graph, incoming_idx
        )

        coupling_data = Bunch(
            incoming_indices=incoming_idx,
            local_indices=local_idx,
            delay_indices=delay_indices,
            state_indices=state_indices,
            bcoo_indices=bcoo_indices,
            bcoo_shape=bcoo_shape,
            use_sparse_incoming=use_sparse_incoming,
        )

        # Initialize history buffer using network's get_history
        # This matches v1's network.get_history(dt=dt) behavior
        history_full = network.get_history(dt)  # [max_delay_steps+1, n_states, n_nodes]

        # Extract only incoming states for history buffer
        history = history_full[
            :, incoming_idx, :
        ]  # [max_delay_steps+1, n_incoming, n_nodes]

        coupling_state = Bunch(history=history)

        return coupling_data, coupling_state

    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        coupling_data: Bunch,
        coupling_state: Bunch,
        params: Bunch,
        graph: AbstractGraph,
    ) -> jnp.ndarray:
        """Delayed coupling computation with sparse graph support.

        Delays are always per-edge (heterogeneous delays per connection).
        Exploits same sparsity pattern for weights and delays when using sparse graphs.

        Args:
            t: Current simulation time
            state: Current network state [n_states, n_nodes]
            coupling_data: Bunch with indices, delay_indices
            coupling_state: Bunch with history buffer
            params: Coupling parameters
            graph: Network graph for accessing weights (dense or sparse)

        Returns:
            Coupling input [n_coupling_inputs, n_nodes]
        """
        from ..graph.sparse import SparseDelayGraph

        # Extract delayed states from history using indices from coupling_data
        # delayed_states[i, j, k] = history[delay[j,k], i, k]
        # i.e., state i from node k delayed by delay[j,k] going to node j
        delayed_states = jnp.transpose(
            coupling_state.history[
                coupling_data.delay_indices, :, coupling_data.state_indices
            ],
            (
                2,
                0,
                1,
            ),  # Reorder to [n_incoming, n_nodes_target, n_nodes_source] - matches v1
        )

        # Extract local states
        local_states = state[coupling_data.local_indices]

        # Compute coupling based on graph type
        if coupling_data.use_sparse_incoming:
            # SPARSE DELAYED INCOMING: Build 3D BCOO
            summed = self._compute_sparse_delayed(
                delayed_states, local_states, params, graph, coupling_data
            )

        elif isinstance(graph, SparseDelayGraph):
            # SPARSE FALLBACK: Dense delayed states with sparse graph
            pre_states = self.pre(delayed_states, local_states, params)
            summed = _sparse_weighted_sum(pre_states, graph, is_bcoo=False)

        else:
            # DENSE DELAYED PATH
            pre_states = self.pre(delayed_states, local_states, params)
            summed = jnp.sum(pre_states * graph.weights, axis=2)

        # Apply post-transform
        return self.post(summed, local_states, params)

    def _compute_sparse_delayed(
        self, delayed_states, local_states, params, graph, coupling_data
    ):
        """Compute delayed coupling with sparse delayed states (BCOO optimization)."""
        import jax.experimental.sparse as jsparse

        indices = graph.weights.indices
        n_incoming = coupling_data.bcoo_shape[0]

        # Extract delayed values at edges for each incoming state
        delayed_data_list = []
        for i in range(n_incoming):
            # Get delayed state from source_k to target_j for each edge
            values_at_edges = delayed_states[i, indices[:, 0], indices[:, 1]]
            delayed_data_list.append(values_at_edges)

        delayed_data_flat = jnp.concatenate(delayed_data_list)

        # Create 3D BCOO [n_incoming, n_nodes, n_nodes]
        delayed_states_sparse = jsparse.BCOO(
            (delayed_data_flat, coupling_data.bcoo_indices),
            shape=coupling_data.bcoo_shape,
        )

        # Apply sparse pre-transform
        pre_states = self._sparse_pre(delayed_states_sparse, local_states, params)

        # Perform sparse weighted sum
        return _sparse_weighted_sum(pre_states, graph, is_bcoo=True)

    def update_state(
        self, coupling_data: Bunch, coupling_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """Update history buffer for delayed coupling.

        Args:
            coupling_data: Bunch with incoming_indices (for extracting states)
            coupling_state: Bunch with current history buffer
            new_state: New network state after integration [n_states, n_nodes]

        Returns:
            New Bunch with updated history buffer
        """
        # Update history buffer (circular) - only mutable operation
        new_history = jnp.roll(coupling_state.history, -1, axis=0)
        new_incoming_states = new_state[coupling_data.incoming_indices]
        new_history = new_history.at[0].set(new_incoming_states)

        return Bunch(history=new_history)

    def describe(self) -> dict:
        """Generate human-readable description of delayed coupling for printing.

        Uses introspection of pre() and post() methods to infer mathematical form.
        Subclasses can override to provide custom descriptions.

        Returns:
            Dictionary with 'network_form', 'pre_form', 'post_form' keys
        """
        # Get incoming/local state descriptions (no subscripts here)
        incoming = self._format_state_list(self.INCOMING_STATE_NAMES)
        local = self._format_state_list(self.LOCAL_STATE_NAMES)

        # Check if pre() and post() are customized
        pre_form = self._infer_pre_form(incoming, local)
        post_form = self._infer_post_form()

        # Get subscripted version for network form
        incoming_with_subscript = self._format_state_list(
            self.INCOMING_STATE_NAMES, with_subscript=True
        )

        # Build network form with pre/post embedded and delays
        if pre_form and post_form:
            network_form = f"post(Σⱼ wᵢⱼ * pre({incoming_with_subscript}(t - τᵢⱼ)))"
        elif pre_form:
            network_form = f"Σⱼ wᵢⱼ * pre({incoming_with_subscript}(t - τᵢⱼ))"
        elif post_form:
            network_form = f"post(Σⱼ wᵢⱼ * {incoming_with_subscript}(t - τᵢⱼ))"
        else:
            network_form = f"Σⱼ wᵢⱼ * {incoming_with_subscript}(t - τᵢⱼ)"

        return {
            "network_form": network_form,
            "pre_form": pre_form,
            "post_form": post_form,
        }

    def _format_state_list(self, states, with_subscript: bool = False) -> str:
        """Format state names for display.

        Args:
            states: State names (str, list, or tuple)
            with_subscript: If True, add subscript ⱼ to each state

        Returns:
            Formatted string like 'S', 'Sⱼ', '(y1ⱼ, y2ⱼ)', etc.
        """
        if isinstance(states, str):
            return states + "ⱼ" if with_subscript else states
        elif isinstance(states, (list, tuple)):
            if len(states) == 0:
                return ""
            elif len(states) == 1:
                state = states[0]
                return state + "ⱼ" if with_subscript else state
            else:
                # Multiple states: format as (state1ⱼ, state2ⱼ)
                if with_subscript:
                    formatted = [s + "ⱼ" for s in states]
                else:
                    formatted = states
                return f"({', '.join(formatted)})"
        return str(states)

    def _infer_pre_form(self, incoming: str, local: str) -> str:
        """Infer pre() transformation form by introspecting the method.

        Returns empty string if pre() is the default identity.
        """
        import inspect

        try:
            pre_source = inspect.getsource(self.pre)
            # Check if it's the default identity (returns delayed_states unchanged)
            if "return delayed_states" in pre_source and pre_source.count("\n") <= 5:
                return ""  # Default identity, don't show
        except (AttributeError, OSError, TypeError):
            pass

        # Check if there are meaningful parameters used in pre
        # For now, return empty - subclasses will override if needed
        return ""

    def _infer_post_form(self) -> str:
        """Infer post() transformation form by introspecting the method.

        Returns the post transformation with actual parameter values.
        """
        import inspect

        try:
            post_source = inspect.getsource(self.post)

            # Try to extract the return expression
            lines = post_source.strip().split("\n")
            for line in lines:
                if "return" in line:
                    # Extract expression after return
                    expr = line.split("return", 1)[1].strip()

                    # Substitute parameter values
                    for key, value in self.params.items():
                        # Replace params.key with actual value
                        expr = expr.replace(f"params.{key}", str(value))

                    # Clean up common patterns
                    expr = expr.replace("summed_inputs", "(...)")
                    expr = expr.replace("local_states", "local")

                    return expr
        except (AttributeError, OSError, TypeError, KeyError):
            pass

        # Fallback: check for common parameter patterns
        if "G" in self.params:
            return "G * (...)"

        return ""

    def pre(
        self, delayed_states: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Transform delayed states before matrix multiplication. Default: identity.

        Args:
            delayed_states: Delayed states from history in per-edge format
                          Shape: [n_incoming, n_nodes_target, n_nodes_source]
                          where [:, j, k] is state from source node k to target node j,
                          delayed by delay[j, k]
            local_states: States from current node [n_local, n_nodes]
            params: Coupling parameters

        Returns:
            Transformed states [n_incoming, n_nodes_target, n_nodes_source]
            Note: Delays are always per-edge (3D), unlike instantaneous coupling
            which supports both vectorized (2D) and per-edge (3D) modes.
        """
        return delayed_states

    def _sparse_pre(self, delayed_states, local_states: jnp.ndarray, params: Bunch):
        """Sparse version of pre() using sparsify decorator.

        Default implementation wraps self.pre() with sparsify.
        Subclasses can override this for custom sparse implementations.

        Args:
            delayed_states: BCOO tensor [n_incoming, n_nodes, n_nodes]
            local_states: Dense tensor [n_local, n_nodes]
            params: Coupling parameters

        Returns:
            BCOO tensor [n_output, n_nodes, n_nodes]
        """
        import jax.experimental.sparse as jsparse

        return jsparse.sparsify(self.pre)(delayed_states, local_states, params)

    @abstractmethod
    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Transform coupling after summation. Must be implemented by subclasses.

        Args:
            summed_inputs: Summed coupling inputs [n_coupling_inputs, n_nodes]
            local_states: States from current node [n_local, n_nodes]
            params: Coupling parameters

        Returns:
            Final coupling input [n_coupling_inputs, n_nodes]
        """
        pass
