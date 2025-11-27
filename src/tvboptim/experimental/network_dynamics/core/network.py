"""Unified network class for coupling system.

This module implements a single Network class that handles all equation types
(ODE/DDE/SDE/SDDE) through composition rather than inheritance.
"""

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import jax.numpy as jnp

from ..coupling.base import AbstractCoupling
from ..dynamics.base import AbstractDynamics
from ..graph.base import AbstractGraph
from ..noise.base import AbstractNoise
from ..result import NativeSolution
from .bunch import Bunch

if TYPE_CHECKING:
    from ..external_input.base import AbstractExternalInput


class Network:
    """Unified network class for all equation types (ODE/DDE/SDE/SDDE).

    Network behavior is determined by composition:
    - Coupling type: InstantaneousCoupling (ODE/SDE) vs DelayedCoupling (DDE/SDDE)
    - Noise presence: None (deterministic) vs AbstractNoise (stochastic)

    Supports multiple named coupling inputs and external inputs that are automatically
    mapped to dynamics model's COUPLING_INPUTS and EXTERNAL_INPUTS specifications.

    Args:
        dynamics: Dynamics model (must be AbstractDynamics)
        coupling: Single coupling, dict {name: coupling}, or list [coupling1, ...]
        graph: Network connectivity (AbstractGraph or DelayGraph)
        noise: Optional noise model for stochastic systems
        history: Optional history for warm-starting simulations
        external_input: Optional external inputs (single, dict, or list)

    Example:
        # Single coupling (ODE)
        network = Network(Lorenz(), LinearCoupling(...), graph)

        # Multiple couplings (dict)
        network = Network(
            FlexibleLorenz(),
            {'structural': LinearCoupling(...), 'modulatory': SigmoidCoupling(...)},
            graph
        )

        # Multiple couplings (list - auto-mapped to COUPLING_INPUTS order)
        network = Network(
            FlexibleLorenz(),
            [LinearCoupling(...), SigmoidCoupling(...)],
            graph
        )

        # With noise (SDE)
        network = Network(Lorenz(), LinearCoupling(...), graph, noise=AdditiveNoise(...))

        # With delays (DDE)
        network = Network(Lorenz(), DelayedLinearCoupling(...), delay_graph)

        # With external input
        network = Network(
            WongWang(),
            LinearCoupling(...),
            graph,
            external_input={'I_ext': DataInput(...)}
        )
    """

    def __init__(
        self,
        dynamics: AbstractDynamics,
        coupling: Union[
            AbstractCoupling, Dict[str, AbstractCoupling], List[AbstractCoupling]
        ],
        graph: AbstractGraph,
        noise: Optional[AbstractNoise] = None,
        history: Optional[NativeSolution] = None,
        external_input: Optional[
            Union[
                "AbstractExternalInput",
                Dict[str, "AbstractExternalInput"],
                List["AbstractExternalInput"],
            ]
        ] = None,
    ):
        """Initialize unified network with validation."""
        self.dynamics = dynamics
        self.graph = graph
        self.noise = noise

        # Normalize coupling input to dict format with validation
        self.couplings = self._normalize_couplings(coupling, dynamics)

        # Normalize external input to dict format with validation

        self.externals = self._normalize_external_inputs(external_input or {}, dynamics)

        # Extract max_delay from graph (0.0 for non-DelayGraph)
        self.max_delay = getattr(graph, "max_delay", 0.0)

        # Resolve noise state indices if noise provided
        if self.noise is not None:
            self.noise._state_indices = self.noise._resolve_state_indices(dynamics)

        # Initialize history (validates if provided)
        self._history = None
        if history is not None:
            self.update_history(history)

    def _normalize_couplings(
        self,
        coupling_input: Union[
            AbstractCoupling, Dict[str, AbstractCoupling], List[AbstractCoupling]
        ],
        dynamics: AbstractDynamics,
    ) -> Dict[str, AbstractCoupling]:
        """Convert coupling input to dict format with validation.

        Args:
            coupling_input: Single coupling, dict, or list
            dynamics: Dynamics model for COUPLING_INPUTS validation

        Returns:
            Dict mapping coupling names to coupling objects

        Raises:
            ValueError: If coupling specification doesn't match dynamics COUPLING_INPUTS
        """
        # Case 1: Dict - validate names match COUPLING_INPUTS
        if isinstance(coupling_input, dict):
            # Check for unknown coupling names
            unknown = set(coupling_input.keys()) - set(dynamics.COUPLING_INPUTS.keys())
            if unknown:
                raise ValueError(
                    f"Unknown coupling names: {unknown}. "
                    f"Dynamics {dynamics.__class__.__name__} expects: "
                    f"{list(dynamics.COUPLING_INPUTS.keys())}"
                )
            return coupling_input

        # Case 2: List - map to COUPLING_INPUTS order
        elif isinstance(coupling_input, list):
            if len(coupling_input) != len(dynamics.COUPLING_INPUTS):
                raise ValueError(
                    f"Got {len(coupling_input)} couplings but dynamics "
                    f"{dynamics.__class__.__name__} expects {len(dynamics.COUPLING_INPUTS)}: "
                    f"{list(dynamics.COUPLING_INPUTS.keys())}"
                )
            return {
                name: coup
                for name, coup in zip(dynamics.COUPLING_INPUTS.keys(), coupling_input)
            }

        # Case 3: Single coupling
        else:
            if len(dynamics.COUPLING_INPUTS) == 0:
                raise ValueError(
                    f"Dynamics {dynamics.__class__.__name__} has no COUPLING_INPUTS. "
                    f"Cannot add coupling."
                )
            elif len(dynamics.COUPLING_INPUTS) == 1:
                # Use the only coupling name
                name = list(dynamics.COUPLING_INPUTS.keys())[0]
                return {name: coupling_input}
            else:
                # Multiple coupling inputs - need clarification
                raise ValueError(
                    f"Dynamics {dynamics.__class__.__name__} expects multiple coupling inputs: "
                    f"{list(dynamics.COUPLING_INPUTS.keys())}. "
                    f"Please provide dict {{name: coupling}} or list [coupling1, coupling2, ...]"
                )

    def _normalize_external_inputs(
        self,
        external_input: Union[
            "AbstractExternalInput",
            Dict[str, "AbstractExternalInput"],
            List["AbstractExternalInput"],
        ],
        dynamics: AbstractDynamics,
    ) -> Dict[str, "AbstractExternalInput"]:
        """Convert external input to dict format with validation.

        Args:
            external_input: Single input, dict, or list
            dynamics: Dynamics model for EXTERNAL_INPUTS validation

        Returns:
            Dict mapping input names to input objects

        Raises:
            ValueError: If input specification doesn't match dynamics EXTERNAL_INPUTS
        """

        # Case 1: Dict - validate names match EXTERNAL_INPUTS
        if isinstance(external_input, dict):
            # Check for unknown input names
            unknown = set(external_input.keys()) - set(dynamics.EXTERNAL_INPUTS.keys())
            if unknown:
                raise ValueError(
                    f"Unknown external input names: {unknown}. "
                    f"Dynamics {dynamics.__class__.__name__} expects: "
                    f"{list(dynamics.EXTERNAL_INPUTS.keys())}"
                )
            return external_input

        # Case 2: List - map to EXTERNAL_INPUTS order
        elif isinstance(external_input, list):
            if len(external_input) != len(dynamics.EXTERNAL_INPUTS):
                raise ValueError(
                    f"Got {len(external_input)} external inputs but dynamics "
                    f"{dynamics.__class__.__name__} expects {len(dynamics.EXTERNAL_INPUTS)}: "
                    f"{list(dynamics.EXTERNAL_INPUTS.keys())}"
                )
            return {
                name: inp
                for name, inp in zip(dynamics.EXTERNAL_INPUTS.keys(), external_input)
            }

        # Case 3: Single external input
        else:
            if len(dynamics.EXTERNAL_INPUTS) == 0:
                raise ValueError(
                    f"Dynamics {dynamics.__class__.__name__} has no EXTERNAL_INPUTS. "
                    f"Cannot add external input."
                )
            elif len(dynamics.EXTERNAL_INPUTS) == 1:
                # Use the only input name
                name = list(dynamics.EXTERNAL_INPUTS.keys())[0]
                return {name: external_input}
            else:
                # Multiple external inputs - need clarification
                raise ValueError(
                    f"Dynamics {dynamics.__class__.__name__} expects multiple external inputs: "
                    f"{list(dynamics.EXTERNAL_INPUTS.keys())}. "
                    f"Please provide dict {{name: input}} or list [input1, input2, ...]"
                )

    def update_history(self, solution: Optional[NativeSolution]) -> None:
        """Update internal history from a simulation result.

        Validates that the solution matches network configuration:
        - n_states must match dynamics.STATE_VARIABLES
        - n_nodes must match graph.n_nodes
        - Warns if time coverage < max_delay (will be padded in get_history)

        Args:
            solution: NativeSolution from a previous simulation, or None to clear history

        Raises:
            ValueError: If solution shape doesn't match network configuration
        """
        if solution is None:
            self._history = None
            return

        # Validate shape
        if solution.ys.ndim != 3:
            raise ValueError(
                f"History must be 3D [n_time, n_states, n_nodes], "
                f"got shape {solution.ys.shape}"
            )

        n_states = len(self.dynamics.STATE_NAMES)
        n_nodes = self.graph.n_nodes
        _, hist_states, hist_nodes = solution.ys.shape

        # Error on state/node mismatch
        if hist_states != n_states:
            raise ValueError(
                f"History has {hist_states} states but network expects {n_states} "
                f"(STATE_NAMES: {self.dynamics.STATE_NAMES})"
            )

        if hist_nodes != n_nodes:
            raise ValueError(
                f"History has {hist_nodes} nodes but network has {n_nodes} nodes"
            )

        # Warn if insufficient time coverage
        if self.max_delay > 0.0:
            time_coverage = solution.ts[-1] - solution.ts[0]
            if time_coverage < self.max_delay:
                warnings.warn(
                    f"History covers {time_coverage:.3f}s but network needs "
                    f"{self.max_delay:.3f}s for delays. History will be padded.",
                    UserWarning,
                )

        self._history = solution

    def prepare(self, dt: float, t0: float, t1: float) -> tuple[Bunch, Bunch]:
        """Prepare all couplings for simulation.

        Calls prepare() on each coupling to get coupling_data and coupling_state.

        Args:
            dt: Integration timestep
            t0: Simulation start time
            t1: Simulation end time

        Returns:
            coupling_data_dict: Bunch {name: coupling_data_bunch}
                Precomputed data for each coupling (stored outside scan carry)
            coupling_state_dict: Bunch {name: coupling_state_bunch}
                Internal state for each coupling (stored in scan carry)
        """
        coupling_data_dict = Bunch()
        coupling_state_dict = Bunch()

        for name, coupling in self.couplings.items():
            data, state = coupling.prepare(self, dt, t0, t1)
            coupling_data_dict[name] = data
            coupling_state_dict[name] = state

        return coupling_data_dict, coupling_state_dict

    def prepare_external(self, dt: float) -> tuple[Bunch, Bunch]:
        """Prepare all external inputs for simulation.

        Calls prepare() on each external input to get external_data and external_state.

        Args:
            dt: Integration timestep

        Returns:
            external_data_dict: Bunch {name: external_data_bunch}
                Precomputed data for each external input (stored outside scan carry)
            external_state_dict: Bunch {name: external_state_bunch}
                Internal state for each external input (stored in scan carry)
        """
        external_data_dict = Bunch()
        external_state_dict = Bunch()

        for name, external in self.externals.items():
            data, state = external.prepare(self, dt)
            external_data_dict[name] = data
            external_state_dict[name] = state

        return external_data_dict, external_state_dict

    def compute_coupling_inputs(
        self,
        t: float,
        state: jnp.ndarray,
        coupling_data_dict: Bunch,
        coupling_state_dict: Bunch,
    ) -> Bunch:
        """Compute all coupling inputs and return as Bunch.

        Args:
            t: Current time
            state: Current network state [n_states, n_nodes]
            coupling_data_dict: Precomputed coupling data from prepare()
            coupling_state_dict: Current coupling internal states

        Returns:
            coupling_inputs: Bunch {name: array[n_dims, n_nodes]}
                Named coupling inputs ready for dynamics.dynamics()
                Missing couplings automatically filled with zeros
        """
        coupling_inputs = Bunch()

        for name, n_dims in self.dynamics.COUPLING_INPUTS.items():
            if name not in self.couplings:
                # Missing coupling - use zeros
                coupling_inputs[name] = jnp.zeros((n_dims, self.graph.n_nodes))
            else:
                # Compute coupling
                coupling = self.couplings[name]
                data = coupling_data_dict[name]
                state_data = coupling_state_dict[name]
                coupling_inputs[name] = coupling.compute(
                    t, state, data, state_data, coupling.params, self.graph
                )

        return coupling_inputs

    def compute_external_inputs(
        self,
        t: float,
        state: jnp.ndarray,
        external_data_dict: Bunch,
        external_state_dict: Bunch,
    ) -> Bunch:
        """Compute all external inputs and return as Bunch.

        Args:
            t: Current time
            state: Current network state [n_states, n_nodes]
            external_data_dict: Precomputed external data from prepare_external()
            external_state_dict: Current external internal states

        Returns:
            external_inputs: Bunch {name: array[n_dims, n_nodes]}
                Named external inputs ready for dynamics.dynamics()
                Missing inputs automatically filled with zeros
        """
        external_inputs = Bunch()

        for name, n_dims in self.dynamics.EXTERNAL_INPUTS.items():
            if name not in self.externals:
                # Missing external input - use zeros
                external_inputs[name] = jnp.zeros((n_dims, self.graph.n_nodes))
            else:
                # Compute external input
                external = self.externals[name]
                data = external_data_dict[name]
                state_data = external_state_dict[name]
                external_inputs[name] = external.compute(
                    t, state, data, state_data, external.params
                )

        return external_inputs

    def update_coupling_states(
        self,
        coupling_data_dict: Bunch,
        coupling_state_dict: Bunch,
        new_state: jnp.ndarray,
    ) -> Bunch:
        """Update all coupling internal states after integration step.

        Args:
            coupling_data_dict: Precomputed coupling data from prepare()
            coupling_state_dict: Current coupling internal states
            new_state: New network state after integration [n_states, n_nodes]

        Returns:
            new_coupling_state_dict: Bunch {name: updated_coupling_state_bunch}
        """
        new_states = Bunch()

        for name, coupling in self.couplings.items():
            new_states[name] = coupling.update_state(
                coupling_data_dict[name],
                coupling_state_dict[name],
                new_state,
            )

        return new_states

    def update_external_states(
        self,
        external_data_dict: Bunch,
        external_state_dict: Bunch,
        new_state: jnp.ndarray,
    ) -> Bunch:
        """Update all external input internal states after integration step.

        Args:
            external_data_dict: Precomputed external data from prepare_external()
            external_state_dict: Current external internal states
            new_state: New network state after integration [n_states, n_nodes]

        Returns:
            new_external_state_dict: Bunch {name: updated_external_state_bunch}
        """
        new_states = Bunch()

        for name, external in self.externals.items():
            new_states[name] = external.update_state(
                external_data_dict[name],
                external_state_dict[name],
                new_state,
            )

        return new_states

    @property
    def params(self) -> Bunch:
        """Aggregate parameters from all network components.

        Returns:
            Bunch with flattened structure:
                dynamics: Dynamics parameters
                coupling.{name}: Parameters for each coupling
                noise: Noise parameters (if noise present)
        """
        p = Bunch()
        p.dynamics = self.dynamics.params

        # Couplings as nested dict
        p.coupling = Bunch()
        for name, coupling in self.couplings.items():
            p.coupling[name] = coupling.params

        # External inputs as nested dict
        p.external = Bunch()
        for name, external in self.externals.items():
            p.external[name] = external.params

        # Noise (optional)
        if self.noise is not None:
            p.noise = self.noise.params

        return p

    @property
    def initial_state(self) -> jnp.ndarray:
        """Initial state for all nodes.

        If history is set, returns the last state from the history.
        Otherwise, returns the dynamics' INITIAL_STATE broadcasted to all nodes.

        Returns:
            Initial state array [n_states, n_nodes]
        """
        if self._history is not None:
            # Use the last state from history
            return self._history.ys[-1]  # [n_states, n_nodes]
        else:
            # Use default initial state from dynamics
            ic = jnp.array(self.dynamics.INITIAL_STATE)
            return jnp.broadcast_to(ic[:, None], (ic.shape[0], self.graph.n_nodes))

    def _get_initial_history(self, dt: float) -> Optional[jnp.ndarray]:
        """Get history buffer based on initial state for delayed coupling networks.

        Args:
            dt: Integration timestep

        Returns:
            None if no delays (max_delay == 0.0)
            Otherwise history buffer [n_steps, n_states, n_nodes]
                where n_steps = ceil(max_delay / dt)
        """
        n_steps = max(
            1, int(jnp.ceil(self.max_delay / dt))
        )  # at least 1 step (case: speed = inf)
        return jnp.broadcast_to(
            self.initial_state[None, :, :],
            (n_steps, self.initial_state.shape[0], self.initial_state.shape[1]),
        )

    def _extract_history_window(self, dt: float) -> jnp.ndarray:
        """Extract history window from stored solution.

        Extracts the last max_delay seconds from self._history:
        - Interpolates if dt doesn't match the history's dt
        - Pads with first timestep if history is too short

        Args:
            dt: Integration timestep for the new simulation

        Returns:
            History buffer [n_steps, n_states, n_nodes]
                where n_steps = ceil(max_delay / dt)
        """
        # Calculate required number of steps
        n_steps_needed = int(jnp.rint(self.max_delay / dt)) + 1

        # Get history data
        hist_ts = self._history.ts
        hist_ys = self._history.ys  # [n_time, n_states, n_nodes]

        # Calculate history dt (assume uniform spacing)
        hist_dt = hist_ts[1] - hist_ts[0] if len(hist_ts) > 1 else dt

        # Check if we need interpolation (allow small numerical tolerance)
        needs_interpolation = jnp.abs(hist_dt - dt) > 1e-9

        # Calculate time coverage and check if we need padding
        time_coverage = hist_ts[-1] - hist_ts[0]
        needs_padding = time_coverage < self.max_delay

        if needs_padding:
            # Pad at the beginning with the first timestep
            n_steps_available = len(hist_ts)
            n_steps_to_pad = n_steps_needed - n_steps_available

            if n_steps_to_pad > 0:
                # Repeat first timestep
                first_state = hist_ys[0:1, :, :]  # [1, n_states, n_nodes]
                padding = jnp.tile(first_state, (n_steps_to_pad, 1, 1))

                if needs_interpolation:
                    # Interpolate available data, then pad
                    interpolated = self._interpolate_history(
                        hist_ts, hist_ys, n_steps_needed - n_steps_to_pad
                    )
                    return jnp.concatenate([padding, interpolated], axis=0)
                else:
                    # Just pad and concatenate
                    return jnp.concatenate([padding, hist_ys], axis=0)
            else:
                # We have enough data but warned in update_history
                # Extract what we need
                if needs_interpolation:
                    return self._interpolate_history(hist_ts, hist_ys, n_steps_needed)
                else:
                    return hist_ys[-n_steps_needed:]
        else:
            # Sufficient time coverage - extract last max_delay seconds
            if needs_interpolation:
                # Find the time window we need
                t_start = hist_ts[-1] - self.max_delay
                start_idx = jnp.searchsorted(hist_ts, t_start)

                # Extract and interpolate
                window_ts = hist_ts[start_idx:]
                window_ys = hist_ys[start_idx:]

                return self._interpolate_history(window_ts, window_ys, n_steps_needed)
            else:
                # No interpolation needed, just extract
                return hist_ys[-n_steps_needed:]

    def _interpolate_history(
        self, old_ts: jnp.ndarray, old_ys: jnp.ndarray, n_steps: int
    ) -> jnp.ndarray:
        """Interpolate history to match target dt.

        Args:
            old_ts: Original time points [n_time_old]
            old_ys: Original trajectory [n_time_old, n_states, n_nodes]
            n_steps: Number of steps needed in output

        Returns:
            Interpolated history [n_steps, n_states, n_nodes]
        """
        # Create new time grid
        new_ts = jnp.linspace(old_ts[0], old_ts[-1], n_steps)

        # Reshape for vectorized interpolation: [n_time, n_states * n_nodes]
        n_time_old, n_states, n_nodes = old_ys.shape
        old_ys_flat = old_ys.reshape(n_time_old, -1)  # [n_time_old, n_states*n_nodes]

        # Interpolate each (state, node) combination
        # jnp.interp works on 1D, so we vmap over the second dimension
        def interp_1d(y_values):
            return jnp.interp(new_ts, old_ts, y_values)

        # Vectorize over all state-node combinations
        from jax import vmap

        new_ys_flat = vmap(interp_1d, in_axes=1, out_axes=1)(old_ys_flat)

        # Reshape back to [n_steps, n_states, n_nodes]
        return new_ys_flat.reshape(n_steps, n_states, n_nodes)

    def get_history(self, dt: float) -> Optional[jnp.ndarray]:
        """Get history buffer for delayed coupling networks.

        If no custom history is set, returns initial state-based history.
        If custom history is set, extracts appropriate window from stored solution.

        Args:
            dt: Integration timestep

        Returns:
            None if no delays (max_delay == 0.0)
            Otherwise history buffer [n_steps, n_states, n_nodes]
                where n_steps = ceil(max_delay / dt)
        """
        if self._history is None:
            return self._get_initial_history(dt)
        else:
            return self._extract_history_window(dt)

    def __repr__(self) -> str:
        """Human-readable representation of network configuration."""
        parts = [
            "Network(",
            f"  dynamics={self.dynamics.__class__.__name__}",
            f"  nodes={self.graph.n_nodes}",
            f"  couplings={list(self.couplings.keys())}",
        ]

        if self.externals:
            parts.append(f"  externals={list(self.externals.keys())}")

        if self.max_delay > 0:
            parts.append(f"  max_delay={self.max_delay:.3f}")

        if self.noise is not None:
            parts.append(f"  noise={self.noise.__class__.__name__}")

        return "\n".join(parts) + "\n)"
