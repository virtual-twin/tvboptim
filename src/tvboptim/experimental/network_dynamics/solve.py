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
from .dynamics.base import AbstractDynamics
from .result import DiffraxSolution, wrap_native_result
from .solvers.diffrax import DiffraxSolver
from .solvers.native import NativeSolver


def _make_diffusion_matrix_fn(diffusion_fn, state_indices, n_states, n_nodes):
    """Build a vectorized diffusion-matrix closure for Diffrax SDE integration.

    Returns ``compute_diffusion_matrix(t, y, noise_params) -> [n_states, n_nodes, n_brownian]``
    where ``n_brownian = len(state_indices) * n_nodes``. The Brownian vector is laid
    out so that index ``i * n_nodes + j`` drives state ``state_indices[i]`` on node ``j``.

    Accepts a diffusion callable returning either a scalar, a per-noise-state 1-D
    array, or a full ``[n_noise_states, n_nodes]`` array; broadcasts accordingly.
    """
    state_indices_arr = jnp.asarray(state_indices)
    n_noise_states = len(state_indices)
    n_brownian = n_noise_states * n_nodes
    i_idx = jnp.arange(n_noise_states)[:, None]
    j_idx = jnp.arange(n_nodes)[None, :]
    brownian_idx = i_idx * n_nodes + j_idx
    state_idx_b = state_indices_arr[:, None]

    def compute_diffusion_matrix(t, y, noise_params):
        g_raw = diffusion_fn(t, y, noise_params)
        if jnp.ndim(g_raw) == 0:
            g = jnp.full((n_noise_states, n_nodes), g_raw)
        elif jnp.ndim(g_raw) == 1:
            g = jnp.broadcast_to(g_raw[..., None], (n_noise_states, n_nodes))
        else:
            g = g_raw
        diffusion_matrix = jnp.zeros((n_states, n_nodes, n_brownian))
        return diffusion_matrix.at[state_idx_b, j_idx, brownian_idx].set(g)

    return compute_diffusion_matrix


_PREPARE_DOC = """Prepare a dynamics model for simulation.

Transforms a model into a JAX-compiled simulation function and a corresponding
configuration PyTree. Dispatches on the first two arguments via ``plum``:

==========================  ================  ======================================
model                       solver            supports
==========================  ================  ======================================
``Network``                 ``NativeSolver``  full feature set: delays, noise,
                                              external inputs, auxiliaries, VOI
``Network``                 ``DiffraxSolver`` stateless couplings only; no delays,
                                              no auxiliaries, no VOI filtering
``AbstractDynamics``        ``NativeSolver``  uncoupled multi-node with optional
                                              noise and external inputs
``AbstractDynamics``        ``DiffraxSolver`` uncoupled multi-node with optional
                                              noise and external inputs, no VOI
==========================  ================  ======================================

Parameters
----------
model : Network or AbstractDynamics
    Either a full ``Network`` (dynamics + couplings + graph + optional noise/
    externals) or a bare ``AbstractDynamics`` for uncoupled simulation.
solver : NativeSolver or DiffraxSolver
    Integration method. ``NativeSolver`` (Euler, Heun) uses fixed-step
    ``jax.lax.scan`` and supports every feature. ``DiffraxSolver`` wraps
    ``diffrax.diffeqsolve`` for adaptive stepping but is restricted to
    stateless couplings (see Diffrax limitations below).
t0 : float, optional
    Start time. Default ``0.0``.
t1 : float, optional
    End time. Default ``1.0``. For native solvers ``t1`` is included in the
    save grid; for Diffrax the save grid is governed by ``solver.saveat``.
dt : float, optional
    Time step. Default ``0.1``. Fixed step for ``NativeSolver``; initial step
    ``dt0`` for ``DiffraxSolver``'s adaptive controller.
n_nodes : int, optional
    **Bare-dynamics dispatch only.** Number of uncoupled nodes. Default ``1``.
    Passing this with a ``Network`` raises a dispatch error.
noise : AbstractNoise, optional
    **Bare-dynamics dispatch only.** Stochastic process. For ``Network``,
    noise is attached to the network itself.
externals : dict, optional
    **Bare-dynamics dispatch only.** Mapping ``{name: AbstractExternalInput}``.
    For ``Network``, externals live on the network.

Returns
-------
solve_function : Callable
    Pure JAX function ``solve_function(config) -> result``. JIT-safe and
    compatible with ``jax.grad``, ``jax.vmap``, and ``jax.pmap``.
config : Bunch
    Configuration PyTree. Keys depend on dispatch but always include
    ``dynamics`` (params), ``initial_state``, and ``_internal`` (precomputed
    static data). Network dispatches additionally carry ``coupling``,
    ``external``, ``graph``, and — if stochastic — ``noise`` with pre-generated
    noise samples under ``_internal.noise_samples`` (native path only).

    The returned ``result`` is a ``NativeSolution`` (native solvers) or a
    ``DiffraxSolution`` (Diffrax). Both expose ``.ts``, ``.ys``,
    ``.variable_names``, and ``.dt``.

Raises
------
ValueError
    If ``DiffraxSolver`` is used with a network whose ``max_delay > 0``.
    Diffrax cannot maintain history buffers across its internal loop.

Notes
-----
**Native solver time grid.** Native solvers scan over
``arange(t0, t1, dt)`` and emit the post-step state on each iteration, so
the returned ``result.ts`` is the half-open grid ``(t0, t1]``:
``[t0 + dt, t0 + 2*dt, ..., t1]`` with ``(t1 - t0) / dt`` samples. The
initial state at ``t0`` is *not* included; ``t1`` *is*. ``t1 - t0`` must be
an integer multiple of ``dt`` for the grid to land exactly on ``t1``.
Diffrax uses ``solver.saveat`` instead.

**Diffrax limitations.** The Diffrax dispatches are experimental and
intentionally narrower than the native ones:

- Delayed couplings are rejected (``max_delay > 0`` raises ``ValueError``).
- Auxiliary outputs from ``dynamics_fn`` are discarded — Diffrax vector
  fields must be pure ``dy/dt``.
- ``VARIABLES_OF_INTEREST`` is **ignored**. The returned ``ys`` always has
  shape ``[n_time, N_STATES, n_nodes]`` and ``variable_names`` always equals
  ``STATE_NAMES``. Use ``NativeSolver`` if you need VOI filtering or
  auxiliaries.
- ``effective_save_dt`` is inferred as ``saveat.ts[1] - saveat.ts[0]`` and
  is only meaningful for uniform ``SaveAt(ts=...)``. Downstream monitors
  that rely on a scalar ``dt`` will be wrong for non-uniform save grids.

**Noise sample caching (native path).** Noise increments are pre-generated
at prepare time using ``network.noise.key``. Reassigning the key on the
noise object after ``prepare`` does *not* change the samples used by
``solve_function``; re-run ``prepare`` to reseed.

**Preparation steps (native).**

1. Prepare all couplings, building history buffers for delays if needed.
2. Prepare external inputs.
3. Pre-generate noise samples if stochastic (one per timestep).
4. Pre-compile coupling/external compute and state-update closures to
   avoid dict iteration inside the scan.
5. Return a pure function wrapping ``jax.lax.scan``.

**Preparation steps (Diffrax).**

1. Validate no delays.
2. Build stateless coupling/external data.
3. Construct the ``diffrax.ODETerm`` drift and, if stochastic, a
   ``ControlTerm`` over a ``VirtualBrownianTree``.
4. Return a pure function wrapping ``diffrax.diffeqsolve``.

**Solver selection.** Use ``NativeSolver`` for anything with delays, noise
interacting with history, VOI filtering, or auxiliary recording. Use
``DiffraxSolver`` for stiff stateless systems where adaptive stepping pays
off.

Examples
--------
Network with native solver (all features):

>>> from tvboptim.experimental.network_dynamics import Network, prepare
>>> from tvboptim.experimental.network_dynamics.dynamics import ReducedWongWang
>>> from tvboptim.experimental.network_dynamics.solvers import Euler
>>> from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
>>> from tvboptim.experimental.network_dynamics.graph import DenseGraph
>>> import jax.numpy as jnp
>>> network = Network(ReducedWongWang(),
...                   LinearCoupling(incoming_states='S', G=1.0),
...                   DenseGraph(jnp.ones((68, 68))))
>>> model_fn, config = prepare(network, Euler(), t0=0, t1=100, dt=0.1)
>>> result = model_fn(config)

Network with Diffrax (no delays, no VOI, no auxiliaries):

>>> import diffrax
>>> from tvboptim.experimental.network_dynamics.solvers import DiffraxSolver
>>> solver = DiffraxSolver(diffrax.Tsit5(),
...                        saveat=diffrax.SaveAt(ts=jnp.arange(0, 100, 0.1)))
>>> model_fn, config = prepare(network, solver, t0=0, t1=100, dt=0.1)

Bare dynamics, uncoupled multi-node:

>>> from tvboptim.experimental.network_dynamics.dynamics import JansenRit
>>> from tvboptim.experimental.network_dynamics.solvers import Heun
>>> model_fn, config = prepare(JansenRit(), Heun(), t0=0, t1=1.0, dt=1e-3,
...                            n_nodes=3)

Modifying parameters between runs (config is a PyTree):

>>> import copy
>>> cfg2 = copy.deepcopy(config)
>>> cfg2.dynamics.G = 2.5
>>> result2 = model_fn(cfg2)

See Also
--------
solve : Thin wrapper that calls ``prepare`` and executes immediately.
Network : Network dynamics container.
NativeSolver : Fixed-step integrators (Euler, Heun).
DiffraxSolver : Adaptive-step integrators via Diffrax.
"""


def solve(
    model,
    solver,
    t0: float = 0.0,
    t1: float = 100.0,
    dt: float = 0.1,
    **kwargs,
):
    """Main entry point for simulation.

    Accepts either a Network or a bare AbstractDynamics instance.
    Dispatches to the appropriate prepare() overload via plum.

    Args:
        model: Network or AbstractDynamics instance
        solver: NativeSolver or DiffraxSolver instance
        t0: Start time
        t1: End time (inclusive for native solvers — see note on time grid)
        dt: Time step
        **kwargs: Additional arguments forwarded to prepare()
            (e.g. n_nodes for bare dynamics)

    Returns:
        Simulation results wrapped in result object

    Notes:
        Native solvers use the half-open scan grid ``arange(t0, t1, dt)`` and
        emit the post-step state on each iteration, so the returned save grid
        is ``(t0, t1]``: ``result.ts = [t0 + dt, t0 + 2*dt, ..., t1]``, with the
        initial state at ``t0`` excluded and the endpoint ``t1`` included.
        The number of saved samples is ``(t1 - t0) / dt``. ``t1 - t0`` must be
        an integer multiple of ``dt`` for the grid to land exactly on ``t1``.

    Examples:
        >>> # With Network
        >>> result = solve(network, Euler(), t0=0, t1=10, dt=0.01)

        >>> # With bare dynamics (single node)
        >>> result = solve(JansenRit(), Heun(), t0=0, t1=1.0, dt=0.001)

        >>> # With bare dynamics (multi-node uncoupled)
        >>> result = solve(JansenRit(), Heun(), t0=0, t1=1.0, dt=0.001, n_nodes=3)
    """
    solve_fn, params = prepare(model, solver, t0=t0, t1=t1, dt=dt, **kwargs)
    return solve_fn(params)


@dispatch
def prepare(
    network: Network,
    solver: NativeSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
) -> Tuple[Callable, Bunch]:
    """Compile a model into a pure JAX solve function and a config PyTree.

    Builds per-dispatch data (coupling buffers, noise samples, external
    inputs) and returns ``(solve_fn, config)`` where ``solve_fn(config)``
    runs the integration. Dispatches on the first two arguments via
    ``plum``: ``Network``/``AbstractDynamics`` paired with
    ``NativeSolver``/``DiffraxSolver``.

    Parameters
    ----------
    t0, t1, dt : float
        Integration interval and step size. ``dt`` is the fixed step for
        native solvers and the initial step for Diffrax.

    Returns
    -------
    (Callable, Bunch)
        Pure solve function and its runtime configuration PyTree.

    See ``help(prepare)`` or ``prepare.__doc__`` for the full reference,
    including per-dispatch parameters (``n_nodes``, ``noise``, ``externals``
    for bare dynamics) and Diffrax limitations (no delays, no auxiliaries,
    no VOI filtering).
    """
    # Prepare all couplings (creates history buffers, computes indices, etc.)
    coupling_data_dict, coupling_state_dict_init = network.prepare(dt, t0, t1)

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
    for name, coupling in network.coupling.items():
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
        if name in network.coupling:
            coupling = network.coupling[name]
            data = coupling_data_dict[name]
            coupling_list.append((name, coupling, data))
        else:
            coupling_list.append((name, None, None))

    n_nodes = network.graph.n_nodes

    def compute_all_couplings(
        t, network_state, coupling_state_dict, config, coupling_data_dict
    ):
        """Pre-compiled closure for coupling computation.

        Avoids method calls and dict iterations in scan loop.

        Args:
            config: Config containing coupling parameters to use
            coupling_data_dict: Per-coupling data (enriched by precompute if applicable)
        """
        coupling_inputs = Bunch()

        for name, coupling, _ in coupling_list:
            if coupling is None:
                # Missing coupling - use zeros
                n_dims = network.dynamics.COUPLING_INPUTS[name]
                coupling_inputs[name] = jnp.zeros((n_dims, n_nodes))
            else:
                # Compute coupling using enriched data and graph
                data = coupling_data_dict[name]
                state_data = coupling_state_dict[name]
                coupling_inputs[name] = coupling.compute(
                    t,
                    network_state,
                    data,
                    state_data,
                    config.coupling[name],
                    config.graph,
                )

        return coupling_inputs

    # =========================================================================
    # PRE-COMPILE COUPLING STATE UPDATE CLOSURE
    # =========================================================================
    # Build list of couplings that need state updates (avoid dict iteration)
    update_list = [
        (name, network.coupling[name], coupling_data_dict[name])
        for name in network.coupling.keys()
    ]

    def update_all_coupling_states(
        coupling_state_dict, new_network_state, coupling_data_dict
    ):
        """Pre-compiled closure for coupling state updates.

        Avoids method calls and dict iterations in scan loop.
        """
        new_states = Bunch()

        for name, coupling, _ in update_list:
            data = coupling_data_dict[name]
            new_states[name] = coupling.update_state(
                data,
                coupling_state_dict[name],
                new_network_state,
            )

        return new_states

    def precompute_all_couplings(config):
        """Call precompute() for each coupling. Runs inside JIT, once per forward pass."""
        enriched = {}
        for name, coupling, static_data in coupling_list:
            if coupling is not None:
                enriched[name] = coupling.precompute(
                    static_data, config.coupling[name], config.graph
                )
            else:
                enriched[name] = None
        return enriched

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

    # Variable names that label axis 1 of the output trajectory.
    # Ordering mirrors the concatenation below: selected states, then selected auxiliaries.
    _all_variable_names = network.dynamics.all_variable_names
    variable_names = tuple(
        _all_variable_names[i] for i in voi_indices if i < n_states
    ) + tuple(_all_variable_names[i] for i in voi_indices if i >= n_states)

    def _f(config):
        """Pure integration function."""
        state0 = config.initial_state

        # Run precompute() for all couplings once before the scan.
        # This allows parameter-dependent quantities (e.g. W_eff = W * wLRE) to
        # be computed with gradient flow while avoiding per-step redundancy.
        enriched = precompute_all_couplings(config)

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
                    t_inner, network_state, state.coupling, config, enriched
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
                state.coupling, next_dynamics_state, enriched
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
        return wrap_native_result(res, t0, t1, dt, variable_names=variable_names)

    return _f, config


@dispatch
def prepare(
    network: Network,
    solver: DiffraxSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
) -> Tuple[Callable, Bunch]:
    """Compile a model into a pure JAX solve function and a config PyTree.

    Builds per-dispatch data (coupling buffers, noise samples, external
    inputs) and returns ``(solve_fn, config)`` where ``solve_fn(config)``
    runs the integration. Dispatches on the first two arguments via
    ``plum``: ``Network``/``AbstractDynamics`` paired with
    ``NativeSolver``/``DiffraxSolver``.

    Parameters
    ----------
    t0, t1, dt : float
        Integration interval and step size. ``dt`` is the fixed step for
        native solvers and the initial step for Diffrax.

    Returns
    -------
    (Callable, Bunch)
        Pure solve function and its runtime configuration PyTree.

    See ``help(prepare)`` or ``prepare.__doc__`` for the full reference,
    including per-dispatch parameters (``n_nodes``, ``noise``, ``externals``
    for bare dynamics) and Diffrax limitations (no delays, no auxiliaries,
    no VOI filtering).
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
    coupling_data_dict, _ = network.prepare(dt, t0, t1)

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
    for name, coupling in network.coupling.items():
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
        if name in network.coupling:
            coupling = network.coupling[name]
            data = coupling_data_dict[name]
            coupling_list.append((name, coupling, data))
        else:
            coupling_list.append((name, None, None))

    n_nodes = network.graph.n_nodes

    def compute_all_couplings(t, network_state, config, coupling_data_dict):
        """Compute all coupling inputs (stateless - no coupling state).

        Args:
            config: Config containing coupling parameters to use
            coupling_data_dict: Per-coupling data (enriched by precompute if applicable)
        """
        coupling_inputs = Bunch()

        for name, coupling, _ in coupling_list:
            if coupling is None:
                # Missing coupling - use zeros
                n_dims = network.dynamics.COUPLING_INPUTS[name]
                coupling_inputs[name] = jnp.zeros((n_dims, n_nodes))
            else:
                # Compute coupling (stateless - pass empty state)
                # For stateless couplings, coupling_state should be ignored
                data = coupling_data_dict[name]
                empty_state = Bunch()
                coupling_inputs[name] = coupling.compute(
                    t,
                    network_state,
                    data,
                    empty_state,
                    config.coupling[name],
                    config.graph,
                )

        return coupling_inputs

    def precompute_all_couplings(config):
        """Call precompute() for each coupling. Runs inside JIT, once per forward pass."""
        enriched = {}
        for name, coupling, static_data in coupling_list:
            if coupling is not None:
                enriched[name] = coupling.precompute(
                    static_data, config.coupling[name], config.graph
                )
            else:
                enriched[name] = None
        return enriched

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
    # DYNAMICS REFERENCES (captured statically for closures)
    # =========================================================================

    dynamics_fn = network.dynamics.dynamics
    n_states = network.dynamics.N_STATES

    # =========================================================================
    # DIFFUSION HELPER (outside _f — captures only static objects)
    # =========================================================================

    has_noise = network.noise is not None
    brownian_motion = None
    if has_noise:
        noise_state_indices = network.noise._state_indices
        n_noise_states = len(noise_state_indices)
        n_brownian = n_noise_states * n_nodes

        compute_diffusion_matrix = _make_diffusion_matrix_fn(
            network.noise.diffusion, noise_state_indices, n_states, n_nodes
        )

        # Brownian motion is static (depends only on t0, t1, dt, key)
        brownian_motion = diffrax.VirtualBrownianTree(
            t0=t0,
            t1=t1,
            tol=dt * 0.01,
            shape=(n_brownian,),
            key=network.noise.key,
        )

    # =========================================================================
    # COMPUTE EFFECTIVE SAVE DT FROM SAVEAT
    # =========================================================================
    # Infer the effective output dt from saveat.ts if available (ts-based saveat).
    # This is a Python-level computation done once at prepare time, so it is
    # concrete and usable as static data in monitors (int/round patterns, etc.).
    # None signals that dt cannot be determined; monitors must handle this case.
    effective_save_dt = None
    saveat_ts = getattr(getattr(solver.saveat, "subs", solver.saveat), "ts", None)
    if saveat_ts is not None and len(saveat_ts) > 1:
        effective_save_dt = float(saveat_ts[1] - saveat_ts[0])

    # =========================================================================
    # CREATE SOLVE FUNCTION
    # =========================================================================

    def _f(config):
        """Pure integration function using Diffrax."""
        # Run precompute() for all couplings once before the solve.
        # This allows parameter-dependent quantities (e.g. W_eff = W * wLRE) to
        # be computed with gradient flow while avoiding per-step redundancy.
        enriched = precompute_all_couplings(config)

        def vector_field(t, y, args):
            """Diffrax-compatible vector field: f(t, y, args) -> dy/dt."""
            # Compute coupling inputs using enriched (precomputed) data
            coupling_inputs = compute_all_couplings(t, y, config, enriched)

            # Compute external inputs
            external_inputs = compute_all_externals(t, y, config)

            # Call dynamics
            result = dynamics_fn(
                t, y, config.dynamics, coupling_inputs, external_inputs
            )

            # Extract derivatives (discard auxiliaries if present)
            if isinstance(result, tuple):
                derivatives, _ = result
            else:
                derivatives = result

            return derivatives

        # Create drift term (deterministic dynamics)
        drift_term = diffrax.ODETerm(vector_field)

        # Combine terms
        if has_noise:
            # SDE: build diffusion term inside _f so it uses runtime config
            def diffusion_vector_field(t, y, args):
                return compute_diffusion_matrix(t, y, config.noise)

            diffusion_term = diffrax.ControlTerm(
                diffusion_vector_field, brownian_motion
            )
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

        return DiffraxSolution(
            solution,
            dt=effective_save_dt,
            variable_names=tuple(network.dynamics.STATE_NAMES),
        )

    return _f, config


@dispatch
def prepare(
    dynamics: AbstractDynamics,
    solver: NativeSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
    n_nodes: int = 1,
    noise=None,
    externals=None,
) -> Tuple[Callable, Bunch]:
    """Compile a model into a pure JAX solve function and a config PyTree.

    Builds per-dispatch data (coupling buffers, noise samples, external
    inputs) and returns ``(solve_fn, config)`` where ``solve_fn(config)``
    runs the integration. Dispatches on the first two arguments via
    ``plum``: ``Network``/``AbstractDynamics`` paired with
    ``NativeSolver``/``DiffraxSolver``.

    Parameters
    ----------
    t0, t1, dt : float
        Integration interval and step size. ``dt`` is the fixed step for
        native solvers and the initial step for Diffrax.

    Returns
    -------
    (Callable, Bunch)
        Pure solve function and its runtime configuration PyTree.

    See ``help(prepare)`` or ``prepare.__doc__`` for the full reference,
    including per-dispatch parameters (``n_nodes``, ``noise``, ``externals``
    for bare dynamics) and Diffrax limitations (no delays, no auxiliaries,
    no VOI filtering).
    """
    # Initial state [N_STATES, n_nodes]
    state0 = dynamics.get_default_initial_state(n_nodes)

    # Zero coupling (always — bare dynamics has no coupling)
    zero_coupling = Bunch()
    for name, n_dims in dynamics.COUPLING_INPUTS.items():
        zero_coupling[name] = jnp.zeros((n_dims, n_nodes))

    # Time array
    time_steps = jnp.arange(t0, t1, dt)
    n_steps = len(time_steps)

    # Config
    config = Bunch(
        dynamics=dynamics.params,
        initial_state=state0,
        _internal=Bunch(time=Bunch(t0=t0, t1=t1, dt=dt)),
    )

    # ---- Noise setup ----
    has_noise = noise is not None
    if has_noise:
        noise._state_indices = noise._resolve_state_indices(dynamics)
        noise_state_indices = noise._state_indices
        config.noise = noise.params
        n_noise_states = len(noise_state_indices)
        noise_shape = (n_steps, n_noise_states, n_nodes)
        config._internal.noise_samples = noise.generate_noise_samples(noise_shape)
        noise_diffusion = noise.diffusion

    # ---- External inputs setup ----
    has_externals = externals is not None and len(externals) > 0
    if has_externals:
        config.external = Bunch()
        external_list = []
        external_data_dict = Bunch()
        external_state_dict_init = Bunch()

        for name in dynamics.EXTERNAL_INPUTS.keys():
            if name in externals:
                ext_obj = externals[name]
                config.external[name] = ext_obj.params
                # Pass None as network — parametric externals don't use it
                ext_data, ext_state = ext_obj.prepare(None, dt)
                external_data_dict[name] = ext_data
                external_state_dict_init[name] = ext_state
                external_list.append((name, ext_obj, ext_data))
            else:
                external_list.append((name, None, None))

        config._internal.external = external_data_dict
        config.initial_state = Bunch(
            dynamics=state0,
            external=external_state_dict_init,
        )

        def compute_all_externals(t, state, external_state_dict, config):
            external_inputs = Bunch()
            for name, ext_obj, data in external_list:
                if ext_obj is None:
                    n_dims = dynamics.EXTERNAL_INPUTS[name]
                    external_inputs[name] = jnp.zeros((n_dims, n_nodes))
                else:
                    state_data = external_state_dict[name]
                    external_inputs[name] = ext_obj.compute(
                        t, state, data, state_data, config.external[name]
                    )
            return external_inputs

        def update_all_external_states(external_state_dict, new_state):
            new_states = Bunch()
            for name, ext_obj, data in external_list:
                if ext_obj is not None:
                    new_states[name] = ext_obj.update_state(
                        data, external_state_dict[name], new_state
                    )
            return new_states
    else:
        # Zero external inputs
        zero_external = Bunch()
        for name, n_dims in dynamics.EXTERNAL_INPUTS.items():
            zero_external[name] = jnp.zeros((n_dims, n_nodes))

    # References
    dynamics_fn = dynamics.dynamics
    solver_step = solver.step
    n_states = dynamics.N_STATES

    # VOI filtering
    voi_indices = dynamics.get_variables_of_interest_indices()
    state_voi_indices = jnp.array([i for i in voi_indices if i < n_states], dtype=int)
    aux_voi_indices = jnp.array(
        [i - n_states for i in voi_indices if i >= n_states], dtype=int
    )
    record_auxiliaries = len(aux_voi_indices) > 0

    # Variable names matching the output layout: selected states, then selected auxiliaries.
    _all_variable_names = dynamics.all_variable_names
    variable_names = tuple(
        _all_variable_names[i] for i in voi_indices if i < n_states
    ) + tuple(_all_variable_names[i] for i in voi_indices if i >= n_states)

    def _f(config):
        """Pure integration function for bare dynamics."""

        def op(carry, scan_input):
            t = scan_input[0]
            step_idx = scan_input[1].astype(int)

            # Unpack carry
            if has_externals:
                state = carry.dynamics
                external_state_dict = carry.external
            else:
                state = carry

            def wrapped_dynamics(t_inner, s, params):
                if has_externals:
                    ext_inputs = compute_all_externals(
                        t_inner, s, external_state_dict, config
                    )
                else:
                    ext_inputs = zero_external
                return dynamics_fn(t_inner, s, params, zero_coupling, ext_inputs)

            # Noise
            if has_noise:
                noise_raw = config._internal.noise_samples[step_idx]
                diffusion = noise_diffusion(t, state, config.noise)
                scaled_noise = diffusion * jnp.sqrt(dt) * noise_raw
                noise_sample = jnp.zeros_like(state)
                noise_sample = noise_sample.at[noise_state_indices].set(scaled_noise)
            else:
                noise_sample = jnp.zeros_like(state)

            next_state, auxiliaries = solver_step(
                wrapped_dynamics, t, state, dt, config.dynamics, noise_sample
            )

            # VOI filtering
            if len(state_voi_indices) > 0:
                selected_states = next_state[state_voi_indices]
            else:
                selected_states = jnp.array([]).reshape(0, next_state.shape[1])

            if record_auxiliaries and auxiliaries.size > 0:
                selected_aux = auxiliaries[aux_voi_indices]
                output = jnp.concatenate([selected_states, selected_aux], axis=0)
            else:
                output = selected_states

            # Update carry
            if has_externals:
                next_external = update_all_external_states(
                    external_state_dict, next_state
                )
                next_carry = Bunch(dynamics=next_state, external=next_external)
            else:
                next_carry = next_state

            return next_carry, output

        scan_inputs = jnp.stack(
            [time_steps, jnp.arange(n_steps, dtype=time_steps.dtype)], axis=1
        )
        _, res = jax.lax.scan(op, config.initial_state, scan_inputs)
        return wrap_native_result(res, t0, t1, dt, variable_names=variable_names)

    return _f, config


@dispatch
def prepare(
    dynamics: AbstractDynamics,
    solver: DiffraxSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
    n_nodes: int = 1,
    noise=None,
    externals=None,
) -> Tuple[Callable, Bunch]:
    """Compile a model into a pure JAX solve function and a config PyTree.

    Builds per-dispatch data (coupling buffers, noise samples, external
    inputs) and returns ``(solve_fn, config)`` where ``solve_fn(config)``
    runs the integration. Dispatches on the first two arguments via
    ``plum``: ``Network``/``AbstractDynamics`` paired with
    ``NativeSolver``/``DiffraxSolver``.

    Parameters
    ----------
    t0, t1, dt : float
        Integration interval and step size. ``dt`` is the fixed step for
        native solvers and the initial step for Diffrax.

    Returns
    -------
    (Callable, Bunch)
        Pure solve function and its runtime configuration PyTree.

    See ``help(prepare)`` or ``prepare.__doc__`` for the full reference,
    including per-dispatch parameters (``n_nodes``, ``noise``, ``externals``
    for bare dynamics) and Diffrax limitations (no delays, no auxiliaries,
    no VOI filtering).
    """
    # Initial state [N_STATES, n_nodes]
    state0 = dynamics.get_default_initial_state(n_nodes)

    # Zero coupling (always)
    zero_coupling = Bunch()
    for name, n_dims in dynamics.COUPLING_INPUTS.items():
        zero_coupling[name] = jnp.zeros((n_dims, n_nodes))

    # Config
    config = Bunch(
        dynamics=dynamics.params,
        initial_state=state0,
        _internal=Bunch(time=Bunch(t0=t0, t1=t1, dt=dt)),
    )

    # References
    dynamics_fn = dynamics.dynamics
    n_states = dynamics.N_STATES

    # ---- Noise setup ----
    has_noise = noise is not None
    brownian_motion = None
    if has_noise:
        noise._state_indices = noise._resolve_state_indices(dynamics)
        noise_state_indices = noise._state_indices
        config.noise = noise.params
        n_noise_states = len(noise_state_indices)
        n_brownian = n_noise_states * n_nodes

        compute_diffusion_matrix = _make_diffusion_matrix_fn(
            noise.diffusion, noise_state_indices, n_states, n_nodes
        )

        brownian_motion = diffrax.VirtualBrownianTree(
            t0=t0,
            t1=t1,
            tol=dt * 0.01,
            shape=(n_brownian,),
            key=noise.key,
        )

    # ---- External inputs setup ----
    has_externals = externals is not None and len(externals) > 0
    if has_externals:
        config.external = Bunch()
        external_list = []
        external_data_dict = Bunch()

        for name in dynamics.EXTERNAL_INPUTS.keys():
            if name in externals:
                ext_obj = externals[name]
                config.external[name] = ext_obj.params
                ext_data, _ = ext_obj.prepare(None, dt)
                external_data_dict[name] = ext_data
                external_list.append((name, ext_obj, ext_data))
            else:
                external_list.append((name, None, None))

        config._internal.external = external_data_dict

        def compute_all_externals(t, state, config):
            external_inputs = Bunch()
            for name, ext_obj, data in external_list:
                if ext_obj is None:
                    n_dims = dynamics.EXTERNAL_INPUTS[name]
                    external_inputs[name] = jnp.zeros((n_dims, n_nodes))
                else:
                    empty_state = Bunch()
                    external_inputs[name] = ext_obj.compute(
                        t, state, data, empty_state, config.external[name]
                    )
            return external_inputs
    else:
        zero_external = Bunch()
        for name, n_dims in dynamics.EXTERNAL_INPUTS.items():
            zero_external[name] = jnp.zeros((n_dims, n_nodes))

    # Effective save dt from saveat
    effective_save_dt = None
    saveat_ts = getattr(getattr(solver.saveat, "subs", solver.saveat), "ts", None)
    if saveat_ts is not None and len(saveat_ts) > 1:
        effective_save_dt = float(saveat_ts[1] - saveat_ts[0])

    def _f(config):
        """Pure integration function using Diffrax for bare dynamics."""

        def vector_field(t, y, args):
            if has_externals:
                ext_inputs = compute_all_externals(t, y, config)
            else:
                ext_inputs = zero_external
            result = dynamics_fn(t, y, config.dynamics, zero_coupling, ext_inputs)
            if isinstance(result, tuple):
                derivatives, _ = result
            else:
                derivatives = result
            return derivatives

        drift_term = diffrax.ODETerm(vector_field)

        if has_noise:

            def diffusion_vector_field(t, y, args):
                return compute_diffusion_matrix(t, y, config.noise)

            diffusion_term = diffrax.ControlTerm(
                diffusion_vector_field, brownian_motion
            )
            terms = diffrax.MultiTerm(drift_term, diffusion_term)
        else:
            terms = drift_term

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

        return DiffraxSolution(
            solution,
            dt=effective_save_dt,
            variable_names=tuple(dynamics.STATE_NAMES),
        )

    return _f, config


prepare.__doc__ = _PREPARE_DOC
