"""New coupling system with self-contained state management.

This module implements the coupling architecture with:
- prepare(), compute(), update_state() interface
- Bunch-based coupling_data and coupling_state
- Support for multi-coupling networks
"""

import warnings
from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp

from ..core.bunch import Bunch
from ..graph.base import AbstractGraph, delay_steps_bound, effective_max_delay

BufferStrategy = Literal["roll", "circular", "preallocated"]

_PRE_CONTRACT_MIGRATION = (
    "pre() must be elementwise over the final message axes: remove explicit "
    "[:, :, None] / [None, :, :] reshapes and declare PRE_USES_LOCAL or "
    "EDGE_PARAMS when target-local or edge-shaped inputs are used."
)

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


# ============================================================================
# Coupling Classes
# ============================================================================


class AbstractCoupling(ABC):
    """Ultra-minimal interface for completely custom coupling implementations.

    Use this base class only when you need full control over coupling computation
    and the standard matrix multiplication patterns don't apply (e.g., SubspaceCoupling).

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

        State selectors may both be omitted while constructing a coupling for
        a heterogeneous ``SignalRoute``; the route supplies canonical arrays
        instead. Ordinary ``Network`` construction validates that at least one
        selector is present.

        Raises:
            ValueError: If unknown parameters are given.
        """
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
    def prepare(self, network, dt: float, t0: float, t1: float) -> Tuple[Bunch, Bunch]:
        """Prepare coupling for simulation.

        Handles all setup logic that was previously in solve.py:
        - State index computation
        - History buffer initialization (for delays)
        - Parameter validation

        Args:
            network: Network instance with graph, dynamics, initial_state
            dt: Integration timestep
            t0: Simulation start time
            t1: Simulation end time

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

    def precompute(
        self,
        coupling_data: Bunch,
        params: Bunch,
        graph: AbstractGraph,
    ) -> Bunch:
        """Called once per forward pass, inside JIT, before the scan.

        Use this to compute quantities that depend on optimisable parameters
        (e.g. wLRE, wFFI) combined with static graph data (e.g. W), such that
        gradients flow through the parameters while the computation is only
        performed once per simulation call rather than once per integration step.

        The returned Bunch is passed to compute() as the coupling_data argument,
        replacing the static coupling_data from prepare(). It may contain both
        the original static fields and new JAX-traced fields.

        Default implementation is a no-op (returns coupling_data unchanged),
        so existing couplings require no modification.

        Args:
            coupling_data: Static data returned by prepare().
            params: Current coupling parameters (JAX-traced, gradients flow).
            graph: Network graph (weights, delays etc.)

        Returns:
            Updated coupling_data Bunch, potentially with additional
            JAX-traced fields.
        """
        return coupling_data

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


class PrePostCoupling(AbstractCoupling):
    """Base class for couplings with pre/post transform pattern.

    This intermediate class provides common infrastructure for couplings that
    follow the TVB-style pattern:
    1. Extract states (incoming or delayed)
    2. Apply pre() transformation
    3. Matrix multiplication with connectivity weights
    4. Apply post() transformation

    Subclasses must implement:
    - prepare(): Setup coupling data and state
    - compute(): Compute coupling input
    - update_state(): Update internal state after integration
    - _build_network_form(): Build the network form string for describe()

    Users typically only need to override pre() and/or post() methods.
    """

    PRE_USES_LOCAL = False
    EDGE_PARAMS: tuple[str, ...] = ()

    # ========================================================================
    # Shared Helper Methods
    # ========================================================================

    def _validate_edge_param_declarations(self, graph) -> None:
        """Validate public edge-parameter declarations before tracing."""
        graph_shape = tuple(graph.weights.shape)
        n_edges = (
            graph.nnz if hasattr(graph, "nnz") else graph_shape[0] * graph_shape[1]
        )

        if len(set(self.EDGE_PARAMS)) != len(self.EDGE_PARAMS) or not all(
            isinstance(name, str) for name in self.EDGE_PARAMS
        ):
            raise ValueError(
                f"{self.__class__.__name__}.EDGE_PARAMS must contain unique strings"
            )

        for name in self.EDGE_PARAMS:
            if name not in self.params:
                raise ValueError(
                    f"{self.__class__.__name__}.EDGE_PARAMS declares {name!r}, "
                    "but that parameter does not exist"
                )
            shape = tuple(jnp.shape(self.params[name]))
            if shape not in (graph_shape, (n_edges,)):
                raise ValueError(
                    f"Edge parameter {name!r} has shape {shape}; expected the "
                    f"graph-shaped layout {graph_shape} or prepared-edge layout "
                    f"({n_edges},)"
                )

        declared = set(self.EDGE_PARAMS)
        for name, value in self.params.items():
            if name not in declared and tuple(jnp.shape(value)) == graph_shape:
                raise ValueError(
                    f"Parameter {name!r} has graph-shaped layout {graph_shape} but "
                    f"is not declared in {self.__class__.__name__}.EDGE_PARAMS"
                )

    def _normalize_edge_params(self, params: Bunch, graph) -> Bunch:
        """Map declared edge params to the graph's execution representation."""
        from ..graph.sparse import SparseGraph

        graph_shape = tuple(graph.weights.shape)
        n_edges = (
            graph.nnz
            if isinstance(graph, SparseGraph)
            else graph_shape[0] * graph_shape[1]
        )
        aligned = Bunch()
        for name in self.EDGE_PARAMS:
            value = jnp.asarray(params[name])
            if value.shape == graph_shape:
                aligned[name] = (
                    graph.gather_edges(value)
                    if isinstance(graph, SparseGraph)
                    else value
                )
            elif value.shape == (n_edges,):
                aligned[name] = (
                    value
                    if isinstance(graph, SparseGraph)
                    else value.reshape(graph_shape)
                )
            else:
                raise ValueError(
                    f"Edge parameter {name!r} has shape {value.shape}; expected "
                    f"{graph_shape} or ({n_edges},)"
                )
        return aligned

    def _build_pre_params(self, coupling_data: Bunch, params: Bunch) -> Bunch:
        """Overlay normalized edge leaves without capturing live parameters."""
        pre_params = Bunch(params)
        for name, value in coupling_data.aligned_edge_params.items():
            pre_params[name] = value
        return pre_params

    def _validate_pre_contract(self, graph, incoming_idx, local_idx, dtype) -> None:
        """Probe ``pre`` with exact abstract execution shapes and no allocation."""
        from ..graph.sparse import SparseGraph

        self._validate_edge_param_declarations(graph)

        n_incoming = _get_n_states(incoming_idx)
        n_local = _get_n_states(local_idx)
        if self.PRE_USES_LOCAL and n_local == 0:
            raise ValueError(
                f"{self.__class__.__name__}.PRE_USES_LOCAL is True but no "
                "local_states were configured"
            )

        n_target, n_source = graph.weights.shape
        is_sparse = isinstance(graph, SparseGraph)
        is_delayed = isinstance(self, DelayedCoupling)
        uses_edge_path = is_delayed or self.PRE_USES_LOCAL or bool(self.EDGE_PARAMS)
        if uses_edge_path:
            message_shape = (graph.nnz,) if is_sparse else (n_target, n_source)
        else:
            message_shape = (n_source,)

        incoming = jax.ShapeDtypeStruct((n_incoming, *message_shape), dtype)
        if self.PRE_USES_LOCAL:
            local_message_shape = message_shape if is_sparse else (n_target, 1)
            local = jax.ShapeDtypeStruct(
                (n_local, *local_message_shape),
                dtype,
            )
        else:
            local = None

        pre_params = Bunch(self.params)
        for name, value in self.params.items():
            value = jnp.asarray(value)
            pre_params[name] = jax.ShapeDtypeStruct(value.shape, value.dtype)
        for name in self.EDGE_PARAMS:
            value = jnp.asarray(self.params[name])
            normalized_shape = (graph.nnz,) if is_sparse else (n_target, n_source)
            pre_params[name] = jax.ShapeDtypeStruct(normalized_shape, value.dtype)

        try:
            result = jax.eval_shape(self.pre, incoming, local, pre_params)
        except Exception as exc:
            raise ValueError(
                f"{self.__class__.__name__}.pre() violates the coupling contract. "
                f"{_PRE_CONTRACT_MIGRATION}"
            ) from exc

        expected = (self.N_OUTPUT_STATES, *message_shape)
        if not hasattr(result, "shape") or tuple(result.shape) != expected:
            actual = getattr(result, "shape", type(result).__name__)
            raise ValueError(
                f"{self.__class__.__name__}.pre() returned shape {actual}; "
                f"expected {expected}. {_PRE_CONTRACT_MIGRATION}"
            )

    def precompute(
        self,
        coupling_data: Bunch,
        params: Bunch,
        graph: AbstractGraph,
    ) -> Bunch:
        """Align declared edge parameters once per forward pass."""
        coupling_data = coupling_data.copy()
        coupling_data.aligned_edge_params = self._normalize_edge_params(params, graph)
        return coupling_data

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
            # Check if it's the default identity (returns first argument unchanged)
            # Works for both "incoming_states" and "delayed_states" naming
            if (
                "return incoming_states" in pre_source
                or "return delayed_states" in pre_source
            ) and pre_source.count("\n") <= 5:
                return ""  # Default identity, don't show
        except (AttributeError, OSError, TypeError):
            pass

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

    # ========================================================================
    # Describe Infrastructure (Template Method Pattern)
    # ========================================================================

    def describe(self) -> dict:
        """Generate human-readable description of coupling for printing.

        Uses introspection of pre() and post() methods to infer mathematical form.
        Calls _build_network_form() for subclass-specific network form building.

        Returns:
            Dictionary with 'network_form', 'pre_form', 'post_form' keys
        """
        incoming = self._format_state_list(self.INCOMING_STATE_NAMES)
        local = self._format_state_list(self.LOCAL_STATE_NAMES)

        pre_form = self._infer_pre_form(incoming, local)
        post_form = self._infer_post_form()

        network_form = self._build_network_form(pre_form, post_form)

        return {
            "network_form": network_form,
            "pre_form": pre_form if pre_form else None,
            "post_form": post_form if post_form else None,
        }

    @abstractmethod
    def _build_network_form(self, pre_form: str, post_form: str) -> str:
        """Build the network form string for describe().

        Subclasses implement this to provide coupling-specific network form.

        Args:
            pre_form: Inferred pre() form string (empty if identity)
            post_form: Inferred post() form string

        Returns:
            Complete network form string like "G * Σⱼ wᵢⱼ * Sⱼ"
        """
        pass

    # ========================================================================
    # Pre/Post Transform Interface
    # ========================================================================

    def pre(
        self, incoming_states: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Elementwise transform of pre-aligned source messages. Default: identity.

        Args:
            incoming_states: Source values [n_incoming, *message_shape].
            local_states: Aligned target values when PRE_USES_LOCAL, otherwise None.
            params: Coupling parameters

        Returns:
            Transformed messages [N_OUTPUT_STATES, *message_shape].
        """
        return incoming_states

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


class InstantaneousCoupling(PrePostCoupling):
    """Base class for coupling without delays (ODE/SDE systems).

    Handles standard weight matrix multiplication pattern:
    1. Extract incoming and local states
    2. Apply pre() transformation
    3. Matrix multiplication with connectivity weights
    4. Apply post() transformation

    Users typically only need to override pre() and/or post() methods.
    """

    def prepare(self, network, dt: float, t0: float, t1: float) -> Tuple[Bunch, Bunch]:
        """Standard preparation for instantaneous coupling.

        Args:
            network: Network instance with graph, dynamics, initial_state
            dt: Integration timestep (not used for instantaneous coupling)
            t0: Simulation start time (not used for instantaneous coupling)
            t1: Simulation end time (not used for instantaneous coupling)

        Returns:
            coupling_data: Bunch with resolved incoming and local state indices
            coupling_state: Empty Bunch (no internal state)
        """
        del t0, t1  # Unused for instantaneous coupling
        graph = network.graph
        dynamics = network.dynamics

        # Resolve state indices
        incoming_idx = dynamics.name_to_index(self.INCOMING_STATE_NAMES)
        local_idx = dynamics.name_to_index(self.LOCAL_STATE_NAMES)

        self._validate_pre_contract(
            graph,
            incoming_idx,
            local_idx,
            network.initial_state.dtype,
        )

        coupling_data = Bunch(
            incoming_indices=incoming_idx,
            local_indices=local_idx,
        )
        coupling_state = Bunch()  # Empty for instantaneous coupling

        return coupling_data, coupling_state

    def _compute_from_signals(
        self,
        incoming_states: jnp.ndarray,
        local_states: jnp.ndarray,
        coupling_data: Bunch,
        params: Bunch,
        graph: AbstractGraph,
    ) -> jnp.ndarray:
        """Apply this prepared coupling to canonical node signal arrays.

        This private seam contains the established pre/transport/post
        implementation.  The homogeneous ``compute`` adapter below only
        resolves dynamics-state names; heterogeneous routing can supply the
        same ``[Q, N]`` arrays directly without fabricating model state.
        """
        from ..graph.sparse import SparseGraph
        from .transport import _aggregate_nodes, _reduce_edges

        uses_edge_messages = self.PRE_USES_LOCAL or bool(self.EDGE_PARAMS)
        source_mask = coupling_data.get("_source_mask")
        pre_params = (
            self._build_pre_params(coupling_data, params)
            if self.EDGE_PARAMS
            else params
        )

        if isinstance(graph, SparseGraph):
            topology = coupling_data.get("_prepared_topology")
            if topology is None:
                edge_indices = graph.edge_indices
                target_e = edge_indices[:, 0]
                source_e = edge_indices[:, 1]
                n_target = graph.weights.shape[0]
            else:
                target_e = topology.target_e
                source_e = topology.source_e
                n_target = topology.n_target

            source_messages = incoming_states[:, source_e]
            target_messages = local_states[:, target_e] if self.PRE_USES_LOCAL else None
            messages = self.pre(source_messages, target_messages, pre_params)
            if source_mask is not None:
                messages = messages * source_mask[source_e][None, :]
            summed = _reduce_edges(
                messages,
                graph.weights.data,
                target_e,
                n_target,
            )
        elif uses_edge_messages:
            n_target, n_source = graph.weights.shape
            source_messages = jnp.broadcast_to(
                incoming_states[:, None, :],
                (incoming_states.shape[0], n_target, n_source),
            )
            target_messages = local_states[:, :, None] if self.PRE_USES_LOCAL else None
            messages = self.pre(source_messages, target_messages, pre_params)
            if source_mask is not None:
                messages = messages * source_mask[None, None, :]
            summed = jnp.sum(messages * graph.weights[None, :, :], axis=-1)
        else:
            messages = self.pre(incoming_states, None, pre_params)
            if source_mask is not None:
                messages = messages * source_mask[None, :]
            summed = _aggregate_nodes(messages, graph.weights)

        return self.post(summed, local_states, params)

    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        coupling_data: Bunch,
        coupling_state: Bunch,
        params: Bunch,
        graph: AbstractGraph,
    ) -> jnp.ndarray:
        """Transform and transport instantaneous source messages.

        Incoming-only couplings transform node signals before transport. Couplings
        that declare ``PRE_USES_LOCAL`` or ``EDGE_PARAMS`` instead transform
        messages aligned to dense matrix cells or sparse prepared edges.

        Args:
            t: Current simulation time
            state: Current network state [n_states, n_nodes]
            coupling_data: Bunch with resolved state indices and prepared topology
            coupling_state: Empty Bunch (not used)
            params: Coupling parameters
            graph: Network graph for accessing weights (dense or sparse)

        Returns:
            Coupling input [n_coupling_inputs, n_nodes]
        """
        del t, coupling_state
        local_states = state[coupling_data.local_indices]
        incoming_states = state[coupling_data.incoming_indices]
        return self._compute_from_signals(
            incoming_states, local_states, coupling_data, params, graph
        )

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

    def _build_network_form(self, pre_form: str, post_form: str) -> str:
        """Build network form for instantaneous coupling.

        Handles both incoming_states and local_states modes (for FastLinearCoupling).
        """
        incoming = self._format_state_list(self.INCOMING_STATE_NAMES)

        # Determine which states to use: incoming if available, else local
        state_with_subscript = self._format_state_list(
            self.INCOMING_STATE_NAMES if incoming else self.LOCAL_STATE_NAMES,
            with_subscript=True,
        )

        base_sum = f"Σⱼ wᵢⱼ * {state_with_subscript}"

        if pre_form and post_form:
            return post_form.replace("(...)", f"Σⱼ wᵢⱼ * pre({state_with_subscript})")
        elif pre_form:
            return f"Σⱼ wᵢⱼ * pre({state_with_subscript})"
        elif post_form:
            return post_form.replace("(...)", base_sum)
        else:
            return base_sum


class DelayedCoupling(PrePostCoupling):
    """Base class for coupling with transmission delays (DDE/SDDE systems).

    Handles delayed coupling pattern:
    1. Extract delayed states from history buffer
    2. Apply pre() transformation
    3. Matrix multiplication with connectivity weights
    4. Apply post() transformation
    5. Update history buffer

    Users typically only need to override pre() and/or post() methods.

    Parameters
    ----------
    buffer_strategy : {"roll", "circular", "preallocated"}, default "roll"
        Strategy for history buffer management. ``T = max_delay_steps + 1`` is
        the buffer length; sweeping or fitting delays sizes it for the declared
        ``max_delay_bound`` headroom (see the delay graph), which is where these
        strategies pull apart.
        - "roll": ``jnp.roll`` shifts the whole buffer each step, so its
          per-step cost scales with ``T``. Fine for fixed delays at minimal
          headroom; it pays for every extra buffer row a bound adds. Memory: O(T).
        - "circular": Pointer-based circular buffer with modulo indexing. Only
          the write pointer moves, so per-step cost is independent of ``T``.
          Preferred for swept or differentiable delays, where the buffer carries
          headroom. Memory: O(T).
        - "preallocated": Pre-allocates the full trajectory. Best forward-pass
          performance but a larger gradient tape. Memory: O(T + simulation_steps).
    warn_on_delay_clamp : bool, default False
        Read indices are recomputed from ``graph.delays`` on every forward pass
        (in ``precompute()``) and clamped into the history buffer sized at
        `prepare()` time. A delay that has grown past the buffer's declared
        bound (see ``max_delay_bound`` on the delay graph) is silently clamped
        to the oldest/newest available state, which degrades to plausible but
        wrong output rather than failing. Set this to emit a host-side warning
        (via ``jax.debug.callback``) whenever any delay hits the bound, so a
        sweep or gradient step that walked past its declared headroom is
        discoverable. Off by default since the callback has a (small) runtime
        cost.
    history_interpolation : {None, "linear"}, default None
        How the history buffer is read to reconstruct the delayed state.
        - None: nearest-integer gather (the default). Static and no-delay
          simulations pay nothing and the buffer keeps its minimal size.
          ``d/d(delays)`` is zero almost everywhere this way (the gather
          rounds through ``rint``), so delays are sweep-accessible but not
          gradient-accessible.
        - "linear": read with linear interpolation. ``d_real = graph.delays /
          dt`` splits into a floor step ``k`` and fraction ``frac``, and the
          read blends ``(1 - frac) * history[k] + frac * history[k + 1]``.
          Makes ``d/d(delays)`` informative. Works with all three buffer
          strategies; grows the history buffer by one slot to hold the
          ``k + 1`` read at the top of the declared delay range.

        "linear" also enables the solver's stage-time shift, which needs a
        read off the integer grid to express itself: the delays are read at
        ``delays - solver.stage_time_centroid * dt``, undoing the bias that
        freezing the coupling across stages introduces and restoring second
        order in the delayed term. With the nearest read the interpolant
        caps the global order at 1 regardless, so no shift is applied. See
        precompute().

        The enum is left open for future ``"cubic"`` / ``"auto"`` strategies
        that slot in without a signature change.
    **kwargs
        Passed to parent class (incoming_states, local_states, params)
    """

    def __init__(
        self,
        buffer_strategy: BufferStrategy = "roll",
        warn_on_delay_clamp: bool = False,
        history_interpolation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if buffer_strategy not in ("roll", "circular", "preallocated"):
            raise ValueError(
                f"Unknown buffer_strategy: {buffer_strategy}. "
                f"Must be one of: 'roll', 'circular', 'preallocated'"
            )

        if history_interpolation not in (None, "linear"):
            raise ValueError(
                f"Unknown history_interpolation: {history_interpolation!r}. "
                f"Must be one of: None, 'linear'."
            )

        self.buffer_strategy = buffer_strategy
        self.warn_on_delay_clamp = warn_on_delay_clamp
        self.history_interpolation = history_interpolation

    @property
    def _interpolating(self) -> bool:
        """True when the history read blends samples (history_interpolation)."""
        return self.history_interpolation == "linear"

    def prepare(self, network, dt: float, t0: float, t1: float) -> Tuple[Bunch, Bunch]:
        """Standard preparation for delayed coupling.

        Only sizes and allocates the static history buffer. It does NOT
        compute delay read indices (delay_steps / delay_indices): those
        depend on graph.delays, which must stay live for delays to be
        sweep/grad-accessible without a re-prepare(). precompute() rebuilds
        them from the live graph once per forward pass instead.

        Args:
            network: Network instance with graph, dynamics, initial_state
            dt: Integration timestep
            t0: Simulation start time
            t1: Simulation end time

        Returns:
            coupling_data: Bunch with indices, dt, and max_delay_steps
                (delay_steps / delay_indices are added by precompute())
            coupling_state: Bunch with history buffer (and write_idx for non-roll strategies)
        """
        graph = network.graph
        dynamics = network.dynamics

        # Resolve state indices
        incoming_idx = dynamics.name_to_index(self.INCOMING_STATE_NAMES)
        local_idx = dynamics.name_to_index(self.LOCAL_STATE_NAMES)

        self._validate_pre_contract(
            graph,
            incoming_idx,
            local_idx,
            network.initial_state.dtype,
        )

        # Buffer length is governed by the declared bound (max_delay_bound),
        # or max(delays) if none was declared -- see effective_max_delay().
        # precompute() recomputes the actual read indices from the live
        # delays every forward pass, clamped into this buffer.
        # Initialize history buffer using network's get_history
        history_full = network.get_history(
            dt
        )  # [base_history_length, n_states, n_nodes]
        history_init = history_full[
            :, incoming_idx, :
        ]  # [base_history_length, n_incoming, n_nodes]

        return self._prepare_history(
            graph,
            history_init,
            dt,
            t0,
            t1,
            incoming_indices=incoming_idx,
            local_indices=local_idx,
        )

    def _prepare_history(
        self,
        graph: AbstractGraph,
        history_init: jnp.ndarray,
        dt: float,
        t0: float,
        t1: float,
        *,
        incoming_indices,
        local_indices,
    ) -> Tuple[Bunch, Bunch]:
        """Prepare a delayed buffer from an already canonicalized history.

        Homogeneous ``prepare`` resolves model state names before entering this
        seam. A heterogeneous route supplies its packed ``[T, Q_source, N]``
        signal directly, so both paths share buffer sizing and update timing.
        """
        max_delay_steps = delay_steps_bound(effective_max_delay(graph), dt)
        base_history_length = max_delay_steps + 1

        # The read indices below are derived from max_delay_steps, while the
        # buffer's newest physical row is history_init.shape[0] - 1. If the two
        # ever disagree, every delayed read silently shifts in time instead of
        # raising, so pin the invariant here.
        if history_init.shape[0] != base_history_length:
            raise ValueError(
                f"History buffer has {history_init.shape[0]} rows but the delay "
                f"read indices assume {base_history_length} "
                f"(max_delay_steps={max_delay_steps}). get_history() and "
                f"delay_steps_bound() must agree on how many steps cover "
                f"[t0 - max_delay, t0]."
            )

        if self._interpolating:
            # Interpolation reads one slot further into the past (k + 1) at
            # the longest representable delay, so the buffer needs one more
            # row than the integer gather does. Extend by constant-
            # extrapolating from the oldest available sample: the same
            # convention Network._get_initial_history/_extract_history_window
            # already use when padding a too-short history.
            base_history_length += 1
            history_init = jnp.concatenate([history_init[:1], history_init], axis=0)

        # Strategy-specific buffer setup. Read indices are filled in by
        # precompute() (delay_indices for "roll", delay_steps otherwise); dt
        # and max_delay_steps are stashed here since precompute() has no dt
        # argument and needs both to rebuild them from graph.delays.
        if self.buffer_strategy == "roll":
            # Roll: history buffer structure [max_delay_steps+1, n_incoming, n_nodes]
            #   - history[0] = oldest state (max_delay in the past)
            #   - history[-1] = newest state (current)
            coupling_data = Bunch(
                incoming_indices=incoming_indices,
                local_indices=local_indices,
                dt=dt,
                max_delay_steps=max_delay_steps,
            )
            coupling_state = Bunch(history=history_init)

        elif self.buffer_strategy == "circular":
            # Circular buffer with write pointer and modulo indexing
            buffer_size = base_history_length

            coupling_data = Bunch(
                incoming_indices=incoming_indices,
                local_indices=local_indices,
                dt=dt,
                max_delay_steps=max_delay_steps,
                buffer_size=jnp.int32(buffer_size),
            )
            coupling_state = Bunch(
                history=history_init,
                write_idx=jnp.int32(0),
            )

        else:  # preallocated
            # Pre-allocate full simulation buffer
            # Use actual history_init shape (may differ slightly from computed base_history_length)
            actual_history_length = history_init.shape[0]
            n_steps = int(jnp.ceil((t1 - t0) / dt))
            buffer_size = actual_history_length + n_steps
            n_incoming = history_init.shape[1]
            n_nodes = history_init.shape[2]

            history = jnp.zeros(
                (buffer_size, n_incoming, n_nodes), dtype=history_init.dtype
            )
            history = history.at[:actual_history_length].set(history_init)

            coupling_data = Bunch(
                incoming_indices=incoming_indices,
                local_indices=local_indices,
                dt=dt,
                max_delay_steps=max_delay_steps,
                buffer_size=jnp.int32(buffer_size),
            )
            coupling_state = Bunch(
                history=history,
                write_idx=jnp.int32(actual_history_length),
            )

        return coupling_data, coupling_state

    def _warn_delay_clamp(self, is_clamped) -> None:
        """Host callback: warn once a recomputed delay saturated the buffer bound."""
        if bool(is_clamped):
            warnings.warn(
                f"{self.__class__.__name__}: a delay fell outside "
                "[0, max_delay_bound] and was clamped to the "
                "oldest/newest buffered state. Raise max_delay_bound "
                "on the delay graph if this sweep or gradient step "
                "needs the headroom. At the low end, a delay shorter "
                "than stage_time_centroid * dt cannot be read after the "
                "stage-time shift and degrades that edge to first order; "
                "shorten dt if it carries appreciable weight.",
                stacklevel=2,
            )

    def _effective_stage_time_centroid(self, coupling_data) -> float:
        """How far back to shift the delayed read, in units of dt.

        The solver's centroid, except that a coupling which also reads a
        *local* state only gets the shift when the solver re-evaluates the
        coupling at each stage. Freezing such a coupling commits two distinct
        first-order errors: the delay bias the shift removes, and a frozen
        local state the shift cannot touch. Correcting one while the other
        stands is not an improvement. Measured on DelayedKuramotoCoupling under
        Heun at dt=0.2, the shift alone makes the error ~6x *worse* (the two
        errors had been partially cancelling), while leaving the global order
        at 1 either way. With recompute_coupling_per_stage the local state is
        evaluated at each stage's own state, the gather stays frozen at the
        centroid, and the pair reaches order 2 (measured 2.06, 2.03).

        Couplings with no local state (DelayedLinearCoupling) always take the
        shift: the delay bias is their only stage-related error.
        """
        centroid = coupling_data.get("stage_time_centroid", 0.0)
        if not centroid or len(coupling_data.local_indices) == 0:
            return centroid
        if coupling_data.get("recompute_coupling_per_stage", False):
            return centroid
        return 0.0

    def _clamp_out_of_bounds(self, d_real_raw, max_delay_steps, weights):
        """Which edges fall outside the buffer, ignoring edges that carry no weight.

        A zero-weight edge contributes nothing to the coupling sum, so clamping
        it is not a mis-simulation. Masking matters because every connectome
        has a zero diagonal in ``delays``: without the mask the stage-time
        shift pushes ``delays[i, i] - centroid * dt`` negative and the warning
        fires on every run, for edges that do not exist.
        """
        relevant = weights != 0.0
        out = (d_real_raw < 0.0) | (d_real_raw > max_delay_steps)
        return jnp.any(out & relevant)

    def precompute(
        self,
        coupling_data: Bunch,
        params: Bunch,
        graph: AbstractGraph,
    ) -> Bunch:
        """Rebuild delay read indices from the live graph, once per forward pass.

        prepare() only sizes the static history buffer; this recomputes the
        read from the *current* graph.delays every call, so mutating the
        delay leaf (a swept GridAxis, a caller `delays = x * delays`, or a
        jax.grad step) changes the simulation without a re-prepare().

        Two read modes, selected by self.history_interpolation:

        - Off (default): delay_steps = rint(graph.delays / dt), clamped into
          [0, max_delay_steps]. Goes through rint, so d/d(delays) is zero
          almost everywhere -- delays are *sweep*-accessible, not *gradient*
          -accessible, this way.
        - On: d_real = graph.delays / dt splits into floor step k and
          fraction frac = d_real - k; compute() blends history[k] and
          history[k + 1] by frac, so d/d(delays) is informative. k, frac are
          computed from d_real clamped into [0, max_delay_steps] (same bound
          as the integer path), so the blend saturates at frac=0 at the
          boundary rather than reading past the buffer prepare() sized (it
          was grown by one slot precisely to make history[k + 1] valid for
          every k in that clamped range -- see prepare()).

        The stage-time shift (interpolating read only)
        ---------------------------------------------
        The solver evaluates the coupling once per step at the step-start
        point and holds it across every stage. Over a step, a method of order
        >= 2 contributes ``h * c(t_n + centroid * h)``, not ``h * c(t_n)``.
        Since ``c`` is a function of the delayed state, evaluating it a
        fraction ``centroid`` of a step early is *identical* to evaluating it
        on time with every delay lengthened by ``centroid * dt``. Freezing
        therefore adds ``centroid * dt`` to every delay, a deterministic bias
        that survives averaging over noise realizations.

        So subtract it back: read at ``delays - centroid * dt``. This restores
        second-order accuracy in the delayed term at one gather per step, the
        same cost freezing already paid. ``stage_time_centroid`` is 0 for
        Euler (no bias, nothing to undo) and 1/2 for Heun and RK4.

        Only applied under history_interpolation="linear". With the nearest read
        the interpolant is piecewise constant, which caps the global order at
        1 regardless (Bellen & Zennaro's min(p, q) rule), so there is no order
        to recover; rint() would either absorb the half-step shift entirely or
        jump the read by a whole step depending on where the delay sits
        between grid points.

        And only applied when the shift is the *whole* remaining stage error:
        for a coupling that also reads a local state, freezing pins that state
        too, and the shift cannot touch it. See
        _effective_stage_time_centroid(); such couplings need
        ``recompute_coupling_per_stage=True`` on the solver, which pairs the
        frozen (shifted) gather with a per-stage local evaluation and reaches
        order 2.

        Edges with ``delays < centroid * dt`` cannot be shifted: the read
        would land inside the step currently being taken. They clamp to a zero
        delay, which leaves them with the original first-order bias. Since the
        coupling sum mixes edges additively, one such edge with appreciable
        weight pins the whole network back to first order. warn_on_delay_clamp
        surfaces exactly this (and ignores zero-weight edges, so a connectome's
        zero diagonal does not trip it).

        max_delay_steps rounds the declared bound *up* (delay_steps_bound), so
        no delay within the bound ever clamps. Rounding to nearest would pin
        frac=0 on the longest delay whenever its fractional step part is <=
        0.5, truncating it and zeroing its gradient.

        Both modes clamp into the buffer declared at prepare() time
        (max_delay_bound, or max(delays) if none was set): a delay past that
        bound degrades to the oldest/newest available state rather than
        gathering out of range. That's a silent mis-simulation, not a crash
        -- enable warn_on_delay_clamp to surface it.
        """
        coupling_data = super().precompute(coupling_data, params, graph)
        from ..graph.sparse import SparseGraph

        dt = coupling_data.dt
        max_delay_steps = coupling_data.max_delay_steps
        is_sparse = isinstance(graph, SparseGraph)
        delays = graph.delays.data if is_sparse else graph.delays
        weights = graph.weights.data if is_sparse else graph.weights
        coupling_data = coupling_data.copy()

        if self._interpolating:
            centroid = self._effective_stage_time_centroid(coupling_data)
            d_real_raw = delays / dt - centroid

            if self.warn_on_delay_clamp:
                out_of_bounds = self._clamp_out_of_bounds(
                    d_real_raw, max_delay_steps, weights
                )
                jax.debug.callback(self._warn_delay_clamp, out_of_bounds)

            d_real = jnp.clip(d_real_raw, 0.0, float(max_delay_steps))
            k = jnp.floor(d_real).astype(jnp.int32)
            coupling_data.delay_frac = d_real - k

            if self.buffer_strategy == "roll":
                # Buffer grew by one slot in prepare(): newest sits at index
                # max_delay_steps + 1 instead of max_delay_steps. idx_hi is
                # one step further into the past (lower index), verified
                # against a synthetic buffer.
                idx_lo = (max_delay_steps + 1) - k
                coupling_data.delay_indices = idx_lo
                coupling_data.delay_indices_hi = idx_lo - 1
            else:  # circular, preallocated: index resolution needs write_idx,
                # which lives in coupling_state (scan carry), not available
                # here -- deferred to compute(), same split as the
                # non-interpolating path below.
                coupling_data.delay_steps = k

            return coupling_data

        # No stage-time shift on the nearest read: q = 1 caps the global order
        # at 1 anyway, so there is no order to recover (see docstring).
        delay_steps_raw = jnp.rint(delays / dt).astype(jnp.int32)

        if self.warn_on_delay_clamp:
            out_of_bounds = self._clamp_out_of_bounds(
                delay_steps_raw, max_delay_steps, weights
            )
            jax.debug.callback(self._warn_delay_clamp, out_of_bounds)

        delay_steps = jnp.clip(delay_steps_raw, 0, max_delay_steps)

        if self.buffer_strategy == "roll":
            coupling_data.delay_indices = max_delay_steps - delay_steps
        else:  # circular, preallocated
            coupling_data.delay_steps = delay_steps

        return coupling_data

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
            coupling_data: Bunch with indices, delay_indices/delay_steps
            coupling_state: Bunch with history buffer (and write_idx for non-roll)
            params: Coupling parameters
            graph: Network graph for accessing weights (dense or sparse)

        Returns:
            Coupling input [n_coupling_inputs, n_nodes]
        """
        del t
        local_states = state[coupling_data.local_indices]
        return self._compute_from_history(
            local_states,
            coupling_data,
            coupling_state,
            params,
            graph,
        )

    def _compute_from_history(
        self,
        local_states: jnp.ndarray,
        coupling_data: Bunch,
        coupling_state: Bunch,
        params: Bunch,
        graph: AbstractGraph,
    ) -> jnp.ndarray:
        """Transport delayed canonical signals from a prepared route history."""
        from ..graph.sparse import SparseGraph
        from .transport import (
            _gather_history,
            _interpolate_history,
            _reduce_edges,
        )

        # Compute read indices based on buffer strategy. idx_hi (one step
        # further into the past than idx_lo) is only needed under
        # interpolation; same direction (idx_hi = idx_lo - 1) for every
        # strategy -- verified against a synthetic buffer.
        if self.buffer_strategy == "roll":
            # Roll strategy: direct lookup using pre-computed delay_indices
            # history[delay_indices[j,k], :, k] retrieves state from node k
            read_indices = coupling_data.delay_indices
            read_indices_hi = (
                coupling_data.delay_indices_hi if self._interpolating else None
            )

        elif self.buffer_strategy == "circular":
            # Circular: compute read indices with modulo
            # newest_idx = write_idx - 1 (position of most recent state)
            # read_idx = newest_idx - delay_steps, wrapped with modulo
            newest_idx = (coupling_state.write_idx - 1) % coupling_data.buffer_size
            read_indices = (
                newest_idx - coupling_data.delay_steps
            ) % coupling_data.buffer_size
            read_indices_hi = (
                (read_indices - 1) % coupling_data.buffer_size
                if self._interpolating
                else None
            )

        else:  # preallocated
            # Preallocated: no modulo needed, indices always increase
            newest_idx = coupling_state.write_idx - 1
            read_indices = newest_idx - coupling_data.delay_steps
            read_indices_hi = read_indices - 1 if self._interpolating else None

        pre_params = (
            self._build_pre_params(coupling_data, params)
            if self.EDGE_PARAMS
            else params
        )
        source_mask = coupling_data.get("_source_mask")

        if isinstance(graph, SparseGraph):
            topology = coupling_data.get("_prepared_topology")
            if topology is None:
                edge_indices = graph.edge_indices
                target_e = edge_indices[:, 0]
                source_e = edge_indices[:, 1]
                n_target = graph.weights.shape[0]
            else:
                target_e = topology.target_e
                source_e = topology.source_e
                n_target = topology.n_target

            if self._interpolating:
                delayed_states = _interpolate_history(
                    coupling_state.history,
                    read_indices,
                    read_indices_hi,
                    source_e,
                    coupling_data.delay_frac,
                )
            else:
                delayed_states = _gather_history(
                    coupling_state.history,
                    read_indices,
                    source_e,
                )
            target_states = local_states[:, target_e] if self.PRE_USES_LOCAL else None
            messages = self.pre(delayed_states, target_states, pre_params)
            if source_mask is not None:
                messages = messages * source_mask[source_e][None, :]
            summed = _reduce_edges(
                messages,
                graph.weights.data,
                target_e,
                n_target,
            )
        else:
            source = jnp.arange(graph.weights.shape[1])
            if self._interpolating:
                delayed_states = _interpolate_history(
                    coupling_state.history,
                    read_indices,
                    read_indices_hi,
                    source,
                    coupling_data.delay_frac,
                )
            else:
                delayed_states = _gather_history(
                    coupling_state.history,
                    read_indices,
                    source,
                )
            target_states = local_states[:, :, None] if self.PRE_USES_LOCAL else None
            messages = self.pre(delayed_states, target_states, pre_params)
            if source_mask is not None:
                messages = messages * source_mask[None, None, :]
            summed = jnp.sum(messages * graph.weights[None, :, :], axis=-1)

        # Apply post-transform
        return self.post(summed, local_states, params)

    def update_state(
        self, coupling_data: Bunch, coupling_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """Update history buffer for delayed coupling.

        Args:
            coupling_data: Bunch with incoming_indices (for extracting states)
            coupling_state: Bunch with current history buffer (and write_idx for non-roll)
            new_state: New network state after integration [n_states, n_nodes]

        Returns:
            New Bunch with updated history buffer (and write_idx for non-roll)
        """
        new_incoming_states = new_state[coupling_data.incoming_indices]
        return self._update_history_from_signal(
            coupling_data, coupling_state, new_incoming_states
        )

    def _update_history_from_signal(
        self,
        coupling_data: Bunch,
        coupling_state: Bunch,
        transmitted: jnp.ndarray,
    ) -> Bunch:
        """Advance a delayed buffer with one accepted canonical signal."""
        from .transport import _roll_history, _write_history

        if self.buffer_strategy == "roll":
            # Roll strategy: shift buffer and write at end
            # Roll by -1 shifts all states towards index 0 (older end):
            #   [oldest, ..., newest] → [2nd_oldest, ..., newest, oldest]
            # Then overwrite index -1 with new current state
            new_history = _roll_history(
                coupling_state.history,
                transmitted,
            )
            return Bunch(history=new_history)

        elif self.buffer_strategy == "circular":
            # Circular: write at pointer, wrap pointer with modulo
            new_history = _write_history(
                coupling_state.history,
                coupling_state.write_idx,
                transmitted,
            )
            new_write_idx = (coupling_state.write_idx + 1) % coupling_data.buffer_size
            return Bunch(history=new_history, write_idx=new_write_idx)

        else:  # preallocated
            # Preallocated: write at pointer, increment (no wrap)
            new_history = _write_history(
                coupling_state.history,
                coupling_state.write_idx,
                transmitted,
            )
            new_write_idx = coupling_state.write_idx + 1
            return Bunch(history=new_history, write_idx=new_write_idx)

    def _build_network_form(self, pre_form: str, post_form: str) -> str:
        """Build network form for delayed coupling.

        Always uses incoming states with delay notation (t - τᵢⱼ).
        """
        incoming_with_subscript = self._format_state_list(
            self.INCOMING_STATE_NAMES, with_subscript=True
        )

        state_expr = f"{incoming_with_subscript}(t - τᵢⱼ)"

        if pre_form and post_form:
            return f"post(Σⱼ wᵢⱼ * pre({state_expr}))"
        elif pre_form:
            return f"Σⱼ wᵢⱼ * pre({state_expr})"
        elif post_form:
            return f"post(Σⱼ wᵢⱼ * {state_expr})"
        else:
            return f"Σⱼ wᵢⱼ * {state_expr}"
