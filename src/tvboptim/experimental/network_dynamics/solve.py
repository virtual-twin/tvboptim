"""Solving system for network architecture.

This module provides the prepare-solve pattern for Network with multi-coupling
support. The prepare() function sets up the integration with all coupling state
management, and returns a pure function for execution.
"""

from typing import Callable, Tuple

import diffrax
import jax
import jax.numpy as jnp
from plum import dispatch

from .core.bunch import Bunch
from .core.network import Network
from .result import wrap_native_result
from .solvers.diffrax import DiffraxSolver
from .solvers.native import NativeSolver


def solve(
    network: Network,
    solver: NativeSolver,
    t0: float = 0.0,
    t1: float = 100.0,
    dt: float = 0.1,
):
    """Main entry point for network simulation.

    Args:
        network: Network instance with multi-coupling support
        solver: NativeSolver instance (Euler, Heun, etc.)
        t0: Start time
        t1: End time
        dt: Time step

    Returns:
        Simulation results wrapped in result object

    Example:
        >>> from network_dynamics import Network, solve
        >>> from network_dynamics.solvers import Euler
        >>> from network_dynamics.dynamics import Lorenz
        >>> from network_dynamics.coupling import LinearCoupling
        >>> from network_dynamics.graph import Graph
        >>>
        >>> dynamics = Lorenz()
        >>> coupling = LinearCoupling(incoming_states='x', G=1.0)
        >>> graph = Graph(weights)
        >>> network = Network(dynamics, coupling, graph)
        >>>
        >>> result = solve(network, Euler(), t0=0, t1=10, dt=0.01)
    """
    solve_fn, params = prepare(network, solver, t0=t0, t1=t1, dt=dt)
    return solve_fn(params)


@dispatch
def prepare(
    network: Network,
    solver: NativeSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
) -> Tuple[Callable, Bunch]:
    """Prepare network for native solver integration.

    This function:
    1. Prepares all couplings (creates history buffers if needed)
    2. Builds config structure with params, graph, initial_state, _internal
    3. Pre-generates noise samples if stochastic
    4. Pre-compiles coupling computation and state update closures
    5. Returns pure function for integration

    Args:
        network: Network with multi-coupling
        solver: NativeSolver (Euler, Heun, etc.)
        t0: Start time
        t1: End time
        dt: Time step

    Returns:
        Tuple of (solve_function, config)
        - solve_function: Pure function that takes config and returns results
        - config: Bunch with flattened params, graph, initial_state, _internal
    """
    # Prepare all couplings (creates history buffers, computes indices, etc.)
    coupling_data_dict, coupling_state_dict_init = network.prepare(dt)

    # Prepare all external inputs
    external_data_dict, external_state_dict_init = network.prepare_external(dt)

    # Time array
    time_steps = jnp.arange(t0, t1, dt)

    # Build new config structure
    config = Bunch(
        # Parameters (flattened - no params. prefix)
        dynamics=network.dynamics.params,
        coupling=Bunch(),
        external=Bunch(),
        # Graph (PyTree object)
        graph=network.graph,
        # Initial state
        initial_state=Bunch(
            dynamics=network.initial_state,
            coupling=coupling_state_dict_init,
            external=external_state_dict_init,
        ),
        # Internal (static precomputed data)
        _internal=Bunch(
            coupling=coupling_data_dict,
            external=external_data_dict,
            time=Bunch(t0=t0, t1=t1, dt=dt),
        ),
    )

    # Add coupling params
    for name, coupling in network.couplings.items():
        config.coupling[name] = coupling.params

    # Add external input params
    for name, external in network.externals.items():
        config.external[name] = external.params

    # Add noise params and samples if stochastic
    if network.noise is not None:
        config.noise = network.noise.params
        n_steps = len(time_steps)
        n_nodes = network.graph.n_nodes
        n_noise_states = len(network.noise._state_indices)
        noise_shape = (n_steps, n_noise_states, n_nodes)
        config._internal.noise_samples = network.noise.generate_noise_samples(
            noise_shape
        )

    # =========================================================================
    # PRE-COMPILE COUPLING COMPUTATION CLOSURE
    # =========================================================================
    # Build a list of (name, coupling, data) tuples to avoid dict iteration in scan
    coupling_list = []
    coupling_names_ordered = []
    for name in network.dynamics.COUPLING_INPUTS.keys():
        coupling_names_ordered.append(name)
        if name in network.couplings:
            coupling = network.couplings[name]
            data = coupling_data_dict[name]
            coupling_list.append((name, coupling, data))
        else:
            coupling_list.append((name, None, None))

    n_nodes = network.graph.n_nodes
    graph = network.graph  # Capture graph for closure

    def compute_all_couplings(t, network_state, coupling_state_dict, config):
        """Pre-compiled closure for coupling computation.

        Avoids method calls and dict iterations in scan loop.

        Args:
            config: Config containing coupling parameters to use
        """
        coupling_inputs = Bunch()

        for name, coupling, data in coupling_list:
            if coupling is None:
                # Missing coupling - use zeros
                n_dims = network.dynamics.COUPLING_INPUTS[name]
                coupling_inputs[name] = jnp.zeros((n_dims, n_nodes))
            else:
                # Compute coupling using pre-fetched data and graph
                state_data = coupling_state_dict[name]
                coupling_inputs[name] = coupling.compute(
                    t, network_state, data, state_data, config.coupling[name], graph
                )

        return coupling_inputs

    # =========================================================================
    # PRE-COMPILE COUPLING STATE UPDATE CLOSURE
    # =========================================================================
    # Build list of couplings that need state updates (avoid dict iteration)
    update_list = [
        (name, network.couplings[name], coupling_data_dict[name])
        for name in network.couplings.keys()
    ]

    def update_all_coupling_states(coupling_state_dict, new_network_state):
        """Pre-compiled closure for coupling state updates.

        Avoids method calls and dict iterations in scan loop.
        """
        new_states = Bunch()

        for name, coupling, data in update_list:
            new_states[name] = coupling.update_state(
                data,
                coupling_state_dict[name],
                new_network_state,
            )

        return new_states

    # =========================================================================
    # PRE-COMPILE EXTERNAL INPUT COMPUTATION CLOSURE
    # =========================================================================
    # Build a list of (name, external, data) tuples to avoid dict iteration in scan
    external_list = []
    for name in network.dynamics.EXTERNAL_INPUTS.keys():
        if name in network.externals:
            external_obj = network.externals[name]
            data = external_data_dict[name]
            external_list.append((name, external_obj, data))
        else:
            external_list.append((name, None, None))

    def compute_all_externals(t, network_state, external_state_dict, config):
        """Pre-compiled closure for external input computation.

        Avoids method calls and dict iterations in scan loop.

        Args:
            config: Config containing external input parameters to use
        """
        external_inputs = Bunch()

        for name, external_obj, data in external_list:
            if external_obj is None:
                # Missing external input - use zeros
                n_dims = network.dynamics.EXTERNAL_INPUTS[name]
                external_inputs[name] = jnp.zeros((n_dims, n_nodes))
            else:
                # Compute external input using pre-fetched data
                state_data = external_state_dict[name]
                external_inputs[name] = external_obj.compute(
                    t, network_state, data, state_data, config.external[name]
                )

        return external_inputs

    # =========================================================================
    # PRE-COMPILE EXTERNAL STATE UPDATE CLOSURE
    # =========================================================================
    # Build list of external inputs that need state updates (avoid dict iteration)
    external_update_list = [
        (name, network.externals[name], external_data_dict[name])
        for name in network.externals.keys()
    ]

    def update_all_external_states(external_state_dict, new_network_state):
        """Pre-compiled closure for external state updates.

        Avoids method calls and dict iterations in scan loop.
        """
        new_states = Bunch()

        for name, external_obj, data in external_update_list:
            new_states[name] = external_obj.update_state(
                data,
                external_state_dict[name],
                new_network_state,
            )

        return new_states

    # =========================================================================
    # PRE-COMPILE DYNAMICS + COUPLING COMPUTATION
    # =========================================================================
    # Store dynamics function reference to avoid attribute lookup
    dynamics_fn = network.dynamics.dynamics

    # Pre-allocate solver step function reference
    solver_step = solver.step

    # =========================================================================
    # VARIABLES OF INTEREST - Determine what to record
    # =========================================================================
    voi_indices = network.dynamics.get_variables_of_interest_indices()
    n_states = network.dynamics.N_STATES

    # Split VOI indices into state and auxiliary indices
    state_voi_indices = jnp.array([i for i in voi_indices if i < n_states], dtype=int)
    aux_voi_indices = jnp.array(
        [i - n_states for i in voi_indices if i >= n_states], dtype=int
    )

    # Flag: do we need to record any auxiliaries?
    record_auxiliaries = len(aux_voi_indices) > 0

    def _f(config):
        """Pure integration function."""
        state0 = config.initial_state

        def op(state, inputs):
            """Single integration step.

            Args:
                state: Bunch(dynamics=network_state, coupling=coupling_state_dict, external=external_state_dict)
                inputs: (t, step_idx) for SDE or just t for ODE

            Returns:
                (next_state, output) tuple for scan
            """
            # Unpack inputs
            if network.noise is not None:
                t = inputs[0]
                step_idx = jnp.int32(inputs[1])
            else:
                t = inputs
                step_idx = None

            # Inline dynamics wrapper to avoid extra function creation
            # Note: This is still inside op, but it's the minimal unavoidable closure
            def wrapped_dynamics(t_inner, network_state, params_dynamics):
                # Compute all coupling inputs using pre-compiled closure
                coupling_inputs = compute_all_couplings(
                    t_inner, network_state, state.coupling, config
                )
                # Compute all external inputs using pre-compiled closure
                external_inputs = compute_all_externals(
                    t_inner, network_state, state.external, config
                )
                # Call dynamics with coupling and external inputs
                # Returns (derivatives, auxiliaries) or just derivatives
                return dynamics_fn(
                    t_inner,
                    network_state,
                    params_dynamics,
                    coupling_inputs,
                    external_inputs,
                )

            # Prepare noise sample if stochastic
            noise_sample = jnp.zeros_like(state.dynamics)
            if network.noise is not None:
                # Get pre-generated noise for this timestep
                noise = config._internal.noise_samples[step_idx]

                # Compute diffusion coefficient
                diffusion = network.noise.diffusion(t, state.dynamics, config.noise)

                # Scale noise: g(t,x) * sqrt(dt) * dW
                # The sqrt(dt) factor is essential for SDEs - Brownian increments scale with sqrt(dt)
                scaled_noise = diffusion * jnp.sqrt(dt) * noise

                # Insert into correct state indices
                noise_sample = noise_sample.at[network.noise._state_indices].set(
                    scaled_noise
                )

            # Solver integration step using pre-compiled function references
            # Returns (next_state, auxiliaries)
            next_dynamics_state, auxiliaries = solver_step(
                wrapped_dynamics, t, state.dynamics, dt, config.dynamics, noise_sample
            )

            # Use pre-compiled closure for coupling state updates
            next_coupling_state_dict = update_all_coupling_states(
                state.coupling, next_dynamics_state
            )

            # Use pre-compiled closure for external state updates
            next_external_state_dict = update_all_external_states(
                state.external, next_dynamics_state
            )

            # Build next state Bunch
            next_state = Bunch(
                dynamics=next_dynamics_state,
                coupling=next_coupling_state_dict,
                external=next_external_state_dict,
            )

            # Apply VARIABLES_OF_INTEREST filtering to build output
            # Collect selected state variables
            if len(state_voi_indices) > 0:
                selected_states = next_dynamics_state[state_voi_indices]
            else:
                selected_states = jnp.array([]).reshape(0, next_dynamics_state.shape[1])

            # Collect selected auxiliary variables if needed
            if record_auxiliaries and auxiliaries.size > 0:
                selected_aux = auxiliaries[aux_voi_indices]
                # Concatenate states and auxiliaries
                output = jnp.concatenate([selected_states, selected_aux], axis=0)
            else:
                output = selected_states

            # Return (carry, output)
            return next_state, output

        # Prepare scan inputs
        if network.noise is None:
            # ODE/DDE: just time
            scan_inputs = time_steps
        else:
            # SDE/SDDE: time + step index for noise lookup
            scan_inputs = jnp.stack([time_steps, jnp.arange(len(time_steps))], axis=1)

        # Run integration
        _, res = jax.lax.scan(op, state0, scan_inputs)

        # Wrap result for consistency
        return wrap_native_result(res, t0, t1, dt)

    return _f, config


@dispatch
def prepare(
    network: Network,
    solver: DiffraxSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
) -> Tuple[Callable, Bunch]:
    """Prepare network for Diffrax solver integration.

    Note: Diffrax solvers only support STATELESS couplings and external inputs.
    Stateful couplings (e.g., delayed coupling with history buffers) cannot be
    used with Diffrax solvers because Diffrax controls the integration loop internally.

    Args:
        network: Network instance
        solver: DiffraxSolver instance
        t0: Start time
        t1: End time
        dt: Initial time step (used as dt0 for Diffrax)

    Returns:
        Tuple of (solve_function, config)
        - solve_function: Pure function that takes config and returns Diffrax solution
        - config: Bunch with parameters and precomputed data

    Raises:
        ValueError: If network contains stateful couplings or noise
    """
    # =========================================================================
    # VALIDATION: Check for unsupported features
    # =========================================================================

    # Check for delayed coupling (stateful)
    if network.max_delay > 0.0:
        raise ValueError(
            f"Diffrax solver does not support delayed coupling (max_delay={network.max_delay}). "
            "Delayed couplings require stateful history buffers that cannot be maintained "
            "with Diffrax's internal integration loop. Use NativeSolver instead."
        )

    # Note on solver compatibility with SDEs:
    # We let Diffrax handle any incompatibility errors rather than checking solver types here.
    # Diffrax will raise appropriate errors if a solver doesn't support SDEs.

    # Warn about potential stateful couplings
    # Note: We can't easily detect if update_state() is non-trivial without running it,
    # but delayed coupling is the main stateful case, which we've already checked above.

    # =========================================================================
    # PREPARE COUPLING AND EXTERNAL INPUT DATA
    # =========================================================================

    # Prepare all couplings (get read-only data, ignore state since we can't maintain it)
    coupling_data_dict, _ = network.prepare(dt)

    # Prepare all external inputs (get read-only data, ignore state)
    external_data_dict, _ = network.prepare_external(dt)

    # Build config structure
    config = Bunch(
        # Parameters
        dynamics=network.dynamics.params,
        coupling=Bunch(),
        external=Bunch(),
        # Graph
        graph=network.graph,
        # Initial state [n_states, n_nodes]
        initial_state=network.initial_state,
        # Internal data
        _internal=Bunch(
            coupling=coupling_data_dict,
            external=external_data_dict,
            time=Bunch(t0=t0, t1=t1, dt=dt),
        ),
    )

    # Add coupling params
    for name, coupling in network.couplings.items():
        config.coupling[name] = coupling.params

    # Add external input params
    for name, external in network.externals.items():
        config.external[name] = external.params

    # Add noise params if present
    if network.noise is not None:
        config.noise = network.noise.params

    # =========================================================================
    # PRE-COMPILE COUPLING COMPUTATION CLOSURE
    # =========================================================================

    # Build coupling list for fast iteration (avoid dict lookups in vector field)
    coupling_list = []
    for name in network.dynamics.COUPLING_INPUTS.keys():
        if name in network.couplings:
            coupling = network.couplings[name]
            data = coupling_data_dict[name]
            coupling_list.append((name, coupling, data))
        else:
            coupling_list.append((name, None, None))

    n_nodes = network.graph.n_nodes
    graph = network.graph

    def compute_all_couplings(t, network_state, config):
        """Compute all coupling inputs (stateless - no coupling state).

        Args:
            config: Config containing coupling parameters to use
        """
        coupling_inputs = Bunch()

        for name, coupling, data in coupling_list:
            if coupling is None:
                # Missing coupling - use zeros
                n_dims = network.dynamics.COUPLING_INPUTS[name]
                coupling_inputs[name] = jnp.zeros((n_dims, n_nodes))
            else:
                # Compute coupling (stateless - pass empty state)
                # For stateless couplings, coupling_state should be ignored
                empty_state = Bunch()
                coupling_inputs[name] = coupling.compute(
                    t, network_state, data, empty_state, config.coupling[name], graph
                )

        return coupling_inputs

    # =========================================================================
    # PRE-COMPILE EXTERNAL INPUT COMPUTATION CLOSURE
    # =========================================================================

    # Build external input list
    external_list = []
    for name in network.dynamics.EXTERNAL_INPUTS.keys():
        if name in network.externals:
            external_obj = network.externals[name]
            data = external_data_dict[name]
            external_list.append((name, external_obj, data))
        else:
            external_list.append((name, None, None))

    def compute_all_externals(t, network_state, config):
        """Compute all external inputs (stateless).

        Args:
            config: Config containing external input parameters to use
        """
        external_inputs = Bunch()

        for name, external_obj, data in external_list:
            if external_obj is None:
                # Missing external input - use zeros
                n_dims = network.dynamics.EXTERNAL_INPUTS[name]
                external_inputs[name] = jnp.zeros((n_dims, n_nodes))
            else:
                # Compute external input (stateless - pass empty state)
                empty_state = Bunch()
                external_inputs[name] = external_obj.compute(
                    t, network_state, data, empty_state, config.external[name]
                )

        return external_inputs

    # =========================================================================
    # CREATE DIFFRAX ODETerm WRAPPER (Drift)
    # =========================================================================

    dynamics_fn = network.dynamics.dynamics
    n_states = network.dynamics.N_STATES

    def vector_field(t, y, args):
        """Diffrax-compatible vector field: f(t, y, args) -> dy/dt.

        Args:
            t: Current time
            y: Network state [n_states, n_nodes]
            args: Not used (params are in closure)

        Returns:
            derivatives: [n_states, n_nodes]
        """
        # Compute coupling inputs
        coupling_inputs = compute_all_couplings(t, y, config)

        # Compute external inputs
        external_inputs = compute_all_externals(t, y, config)

        # Call dynamics
        result = dynamics_fn(t, y, config.dynamics, coupling_inputs, external_inputs)

        # Extract derivatives (discard auxiliaries if present)
        if isinstance(result, tuple):
            derivatives, _ = result
        else:
            derivatives = result

        return derivatives

    # =========================================================================
    # CREATE DIFFRAX ControlTerm WRAPPER (Diffusion) if noise present
    # =========================================================================

    diffusion_term = None
    if network.noise is not None:
        # Get noise configuration
        noise_state_indices = network.noise._state_indices
        n_noise_states = len(noise_state_indices)

        def diffusion_vector_field(t, y, args):
            """Diffusion coefficient g(t, y) for ControlTerm.

            Args:
                t: Current time
                y: Network state [n_states, n_nodes]
                args: Not used (params are in closure)

            Returns:
                Diffusion matrix [n_states, n_nodes, n_brownian]
                where n_brownian = n_noise_states * n_nodes
            """
            # Compute diffusion coefficients using noise model
            g_raw = network.noise.diffusion(t, y, config.noise)

            # Handle different return types from diffusion():
            # - Scalar (e.g., AdditiveNoise with constant sigma)
            # - Array [n_noise_states, n_nodes] (e.g., MultiplicativeNoise)

            # Ensure g has shape [n_noise_states, n_nodes]
            if jnp.ndim(g_raw) == 0:
                # Scalar - broadcast to all noise states and nodes
                g = jnp.full((n_noise_states, n_nodes), g_raw)
            elif jnp.ndim(g_raw) == 1:
                # 1D array - could be per-state or per-node, assume broadcasting needed
                g = jnp.broadcast_to(g_raw[..., None], (n_noise_states, n_nodes))
            else:
                # Already [n_noise_states, n_nodes]
                g = g_raw

            # Build full diffusion matrix that maps Brownian motion to state space
            # Shape: [n_states, n_nodes, n_brownian]
            # where n_brownian = n_noise_states * n_nodes
            n_brownian = n_noise_states * n_nodes

            # Initialize with zeros
            diffusion_matrix = jnp.zeros((n_states, n_nodes, n_brownian))

            # Fill in the diagonal blocks for states that receive noise
            for i, state_idx in enumerate(noise_state_indices):
                for j in range(n_nodes):
                    # This Brownian dimension corresponds to state_idx, node j
                    brownian_idx = i * n_nodes + j
                    # Set the diffusion coefficient
                    diffusion_matrix = diffusion_matrix.at[
                        state_idx, j, brownian_idx
                    ].set(g[i, j])

            return diffusion_matrix

        # Create Brownian motion
        n_brownian = n_noise_states * n_nodes
        brownian_motion = diffrax.VirtualBrownianTree(
            t0=t0,
            t1=t1,
            tol=dt * 0.01,  # Brownian tree tolerance (finer than dt)
            shape=(n_brownian,),
            key=network.noise.key,
        )

        # Create diffusion term
        diffusion_term = diffrax.ControlTerm(diffusion_vector_field, brownian_motion)

    # =========================================================================
    # CREATE SOLVE FUNCTION
    # =========================================================================

    def _f(config):
        """Pure integration function using Diffrax."""
        # Create drift term (deterministic dynamics)
        drift_term = diffrax.ODETerm(vector_field)

        # Combine terms
        if diffusion_term is not None:
            # SDE: combine drift and diffusion
            terms = diffrax.MultiTerm(drift_term, diffusion_term)
        else:
            # ODE: just drift
            terms = drift_term

        # Solve using diffrax with 2D state [n_states, n_nodes]
        solution = diffrax.diffeqsolve(
            terms,
            solver.solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=config.initial_state,
            saveat=solver.saveat,
            stepsize_controller=solver.stepsize_controller,
            max_steps=solver.max_steps,
            **solver.diffrax_kwargs,
        )

        # NOTE: Diffrax may pad solution arrays with inf when max_steps is specified.
        # Users should filter finite values in post-processing if needed:
        #   finite_mask = jnp.isfinite(solution.ts)
        #   solution_filtered = solution.ts[finite_mask], solution.ys[finite_mask]

        return solution

    return _f, config
