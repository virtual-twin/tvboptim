"""Native solvers for network system.

This module provides native solver implementations that work with the
multi-coupling architecture. These solvers work with wrapped dynamics
functions that already include coupling computation.
"""

from typing import Callable

import jax.numpy as jnp

from ..core.bunch import Bunch
from .base import AbstractSolver


class NativeSolver(AbstractSolver):
    """Base class for  Network Dynamics native solvers.

    Solvers integrate only the dynamics state. Coupling state management
    happens outside the solver in the scan loop.

    The dynamics_fn passed to step() should already have coupling computation
    bundled in via closure, so the solver only needs to worry about time
    integration of the dynamics state.
    """

    def __init__(
        self,
        block_size: int | None = None,
        recompute_coupling_per_stage: bool = False,
        tbptt_window: int | None = None,
    ):
        """
        Args:
            recompute_coupling_per_stage: If False (default), coupling
                inputs are evaluated once per step at ``(t, y_n)`` and held
                constant across all solver stages (Heun's
                predictor/corrector, RK4's k1..k4). If True, coupling is
                recomputed at each stage's own ``(time, state)``, costing
                ``n_stages`` coupling evaluations per step. Only the
                Network+NativeSolver path reads this; bare dynamics has no
                coupling and single-stage Euler is unaffected.

                For delayed coupling the two settings are bit-exact: the
                history buffer is a step-level carry that does not depend
                on the stage state, so the default just avoids redundant
                gathers. They differ only for instantaneous
                (state-dependent) coupling, where freezing replaces
                ``c(t)`` by the constant ``c(t_n)`` and pins the coupling
                component of the solution to first order regardless of the
                base method's order. Set True to recover full method order
                when the coupling varies fast relative to ``dt``.

                External inputs are always evaluated per stage regardless
                of this flag.
            block_size: The granularity of the nested block scan, and the
                one block unit for the streaming features. If None (default),
                the integration runs as a single ``jax.lax.scan`` and every
                step's carry is saved for the backward pass (the monolithic
                path; no blocking). If an int ``K``, the scan is split into an
                outer scan over blocks of ``K`` steps each wrapped in
                ``jax.checkpoint``, with an inner scan running the steps inside
                a block. The backward pass then only retains block-boundary
                carries and recomputes inner activations on demand. Trades
                roughly 1.3-1.7x gradient wall time (one extra forward
                recompute, added to an already backward-dominated cost) for
                ``O(n_steps/K + K)`` backward memory instead of ``O(n_steps)``.
                Peak memory is U-shaped in ``K``: small ``K`` inflates the
                outer block-boundary tape (``n_steps/K`` term), large ``K``
                inflates the per-block inner tape (``K`` term). The minimum
                sits near ``K = sqrt(n_steps)``. Has no effect on the
                forward-only path.

                This same block is the unit that carries per-block streaming
                noise and the online ``reduce`` accumulator when those per-call
                features are used (so blocking *is* checkpointing; the
                checkpoint grain is ``block_size``). ``block_size`` was formerly
                named ``checkpoint_every``; once streaming noise is active a
                blocked run reseeds the noise relative to the monolithic path
                (see the Phase 4 plan).

                The memory model assumes a per-step carry whose size does
                not grow with ``n_steps``. This holds for the ``roll`` and
                ``circular`` delayed-coupling buffer strategies (history
                buffer size = ``max_delay_steps + 1``), but **not** for
                ``preallocated`` (history buffer grows linearly with
                ``n_steps``). Blocking still works correctly with
                ``preallocated``, but the practical memory win is much
                smaller because the carry itself dominates.
            tbptt_window: If None (default), the gradient is exact over the
                whole rollout. If an int ``W``, the integration runs as an
                outer scan over windows of ``W`` steps with the carry
                gradient severed (``jax.lax.stop_gradient``) at each window
                boundary: truncated backpropagation through time. Credit is
                assigned only within a window, so the backward ``T`` in the
                ``exp(T·Lambda)`` sensitivity growth becomes ``W·dt`` instead
                of the full horizon. The forward rollout is unchanged and
                bit-exact; only the backward pass differs.

                ``tbptt_window`` and ``block_size`` are independent and
                nest. ``block_size`` is a *memory* granularity (how big a
                block to rematerialize on the backward pass, set by available
                RAM); ``tbptt_window`` is the *gradient horizon* (how far back
                credit is assigned, set by the slowest timescale of interest).
                The two block sizes are not the same number. When both are set,
                ``block_size`` tiles ``tbptt_window`` (each window is
                rematerialized as ``W / block_size`` blocks); the clean case is
                ``W % block_size == 0`` and ``n_steps % W == 0``, with
                remainders handled by a tail scan.

                Severing the carry bounds the gradient horizon but does **not**
                by itself bound activation memory: the full output is still
                stacked and the loss taped over the whole rollout. For a long
                rollout that needs both a bounded horizon and bounded memory,
                set both knobs (``tbptt_window`` for the horizon,
                ``block_size`` for memory within it). ``tbptt_window`` set
                with ``block_size=None`` is a gradient-stability fix only.

                TBPTT biases toward fast timescales: any credit assignment
                longer than ``W·dt`` is zeroed, so too short a window can null
                the gradient of genuinely slow parameters. The window is a real
                hyperparameter. For SDEs it is well-defined because noise is
                fixed per call (common random numbers), so each window is a
                deterministic map given its noise.
        """
        self.block_size = block_size
        self.recompute_coupling_per_stage = recompute_coupling_per_stage
        self.tbptt_window = tbptt_window

    def step(
        self,
        dynamics_fn: Callable,
        t: float,
        state: jnp.ndarray,
        dt: float,
        params: Bunch,
        noise_sample: jnp.ndarray = 0.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Single integration step for dynamics state.

        Args:
            dynamics_fn: Dynamics function (t, state, params) -> (derivatives, auxiliaries)
                        Coupling is already computed inside via closure
            t: Current time
            state: Network state array [n_states, n_nodes]
            dt: Time step
            params: Parameters (params.dynamics, etc.)
            noise_sample: Scaled noise for SDEs [n_states, n_nodes]
                         Default 0.0 for deterministic systems

        Returns:
            Tuple of (next_state, auxiliaries):
                - next_state: Updated network state array [n_states, n_nodes]
                - auxiliaries: Auxiliary variables array [n_auxiliaries, n_nodes]
                              Empty array if no auxiliaries
        """
        raise NotImplementedError("Subclasses must implement step()")


class Euler(NativeSolver):
    """Euler method

    For ODEs: Standard Euler when noise_sample=0
    For SDEs: Euler-Maruyama when noise_sample provided
    """

    def step(
        self,
        dynamics_fn: Callable,
        t: float,
        state: jnp.ndarray,
        dt: float,
        params: Bunch,
        noise_sample: jnp.ndarray = 0.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Euler integration step: y_{n+1} = y_n + dt * f(t, y_n) + noise.

        Args:
            dynamics_fn: Dynamics function (t, state, params) -> (derivatives, auxiliaries)
            t: Current time
            state: Current state [n_states, n_nodes]
            dt: Time step
            params: Parameters
            noise_sample: Pre-scaled noise increment

        Returns:
            Tuple of (next_state, auxiliaries):
                - next_state: Next state [n_states, n_nodes]
                - auxiliaries: Auxiliary variables at current time [n_auxiliaries, n_nodes]
        """
        # Compute derivatives and auxiliaries (coupling already inside dynamics_fn)
        result = dynamics_fn(t, state, params)

        # Handle both formats: just derivatives or (derivatives, auxiliaries)
        if isinstance(result, tuple):
            derivatives, auxiliaries = result
        else:
            derivatives = result
            auxiliaries = jnp.array([])  # Empty array for no auxiliaries

        # Euler step for state integration
        next_state = state + dt * derivatives + noise_sample

        return next_state, auxiliaries


class Heun(NativeSolver):
    """Heun's method (improved Euler).

    Two-stage method with predictor-corrector structure.
    For SDEs: Stochastic Heun method.

    Note: Whether the coupling input is held constant across the two stages
    or recomputed per stage is decided outside ``step`` by the
    ``recompute_coupling_per_stage`` flag (see ``NativeSolver``); the
    ``dynamics_fn`` passed here already encodes that choice via closure.
    """

    def step(
        self,
        dynamics_fn: Callable,
        t: float,
        state: jnp.ndarray,
        dt: float,
        params: Bunch,
        noise_sample: jnp.ndarray = 0.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Heun integration step with predictor-corrector.

        Args:
            dynamics_fn: Dynamics function (t, state, params) -> (derivatives, auxiliaries)
            t: Current time
            state: Current state [n_states, n_nodes]
            dt: Time step
            params: Parameters
            noise_sample: Pre-scaled noise increment

        Returns:
            Tuple of (next_state, auxiliaries):
                - next_state: Next state [n_states, n_nodes]
                - auxiliaries: Auxiliary variables from FIRST evaluation [n_auxiliaries, n_nodes]
                              (represents valid state at current time t)
        """
        # First evaluation: k1 = f(t, y_n)
        result1 = dynamics_fn(t, state, params)

        # Handle both formats and extract first evaluation results
        if isinstance(result1, tuple):
            k1, auxiliaries = result1  # Keep auxiliaries from first evaluation
        else:
            k1 = result1
            auxiliaries = jnp.array([])

        # Predictor step
        y_pred = state + dt * k1 + noise_sample

        # Second evaluation: k2 = f(t + dt, y_pred)
        result2 = dynamics_fn(t + dt, y_pred, params)

        # Extract k2 (discard auxiliaries from second evaluation)
        if isinstance(result2, tuple):
            k2 = result2[0]
        else:
            k2 = result2

        # Corrector: average drift, apply noise once
        next_state = state + dt * 0.5 * (k1 + k2) + noise_sample

        return next_state, auxiliaries


class RungeKutta4(NativeSolver):
    """Classical 4th order Runge-Kutta method (RK4).

    Four-stage explicit method with high accuracy for smooth ODEs.
    For SDEs: Applies noise at the end of the RK4 step.

    Note: Whether the coupling input is held constant across the four stages
    or recomputed per stage is decided outside ``step`` by the
    ``recompute_coupling_per_stage`` flag (see ``NativeSolver``); the
    ``dynamics_fn`` passed here already encodes that choice via closure.
    """

    def step(
        self,
        dynamics_fn: Callable,
        t: float,
        state: jnp.ndarray,
        dt: float,
        params: Bunch,
        noise_sample: jnp.ndarray = 0.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """RK4 integration step with four evaluations.

        Args:
            dynamics_fn: Dynamics function (t, state, params) -> (derivatives, auxiliaries)
            t: Current time
            state: Current state [n_states, n_nodes]
            dt: Time step
            params: Parameters
            noise_sample: Pre-scaled noise increment (applied at end of step)

        Returns:
            Tuple of (next_state, auxiliaries):
                - next_state: Next state [n_states, n_nodes]
                - auxiliaries: Auxiliary variables from FIRST evaluation [n_auxiliaries, n_nodes]
                              (represents valid state at current time t)
        """
        # First evaluation: k1 = f(t, y_n)
        result1 = dynamics_fn(t, state, params)

        # Handle both formats and extract first evaluation results
        if isinstance(result1, tuple):
            k1, auxiliaries = result1  # Keep auxiliaries from first evaluation
        else:
            k1 = result1
            auxiliaries = jnp.array([])

        # Second evaluation: k2 = f(t + dt/2, y_n + dt/2 * k1)
        result2 = dynamics_fn(t + 0.5 * dt, state + 0.5 * dt * k1, params)
        k2 = result2[0] if isinstance(result2, tuple) else result2

        # Third evaluation: k3 = f(t + dt/2, y_n + dt/2 * k2)
        result3 = dynamics_fn(t + 0.5 * dt, state + 0.5 * dt * k2, params)
        k3 = result3[0] if isinstance(result3, tuple) else result3

        # Fourth evaluation: k4 = f(t + dt, y_n + dt * k3)
        result4 = dynamics_fn(t + dt, state + dt * k3, params)
        k4 = result4[0] if isinstance(result4, tuple) else result4

        # RK4 combination: y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4) + noise
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) + noise_sample

        return next_state, auxiliaries


class BoundedSolver(NativeSolver):
    """Wrapper that enforces hard bounds on solver output via clipping.

    Wraps any native solver and clips the output state to specified bounds.
    Useful for ensuring states remain in valid ranges (e.g., firing rates ≥ 0).

    The bounds support flexible broadcasting:
    - Scalar: same bound for all states/nodes
    - [n_states]: different bounds per state variable
    - [n_states, n_nodes]: different bounds per state per node

    Use -jnp.inf or jnp.inf to disable clipping for specific states/nodes.

    Args:
        base_solver: The underlying solver to wrap
        low: Lower bound(s) for state clipping (default: -inf, no clipping)
        high: Upper bound(s) for state clipping (default: +inf, no clipping)

    Example:
        # Ensure all states stay in [0, 1]
        solver = BoundedSolver(Euler(), low=0.0, high=1.0)

        # Different bounds per state variable
        solver = BoundedSolver(
            Heun(),
            low=jnp.array([0.0, -5.0]),  # state 0: ≥0, state 1: ≥-5
            high=jnp.array([1.0, 5.0])   # state 0: ≤1, state 1: ≤5
        )
    """

    def __init__(
        self,
        base_solver: NativeSolver,
        low: float | jnp.ndarray = -jnp.inf,
        high: float | jnp.ndarray = jnp.inf,
    ):
        # Deliberately skip NativeSolver.__init__: block_size,
        # recompute_coupling_per_stage and tbptt_window are delegated to
        # base_solver via the properties below so that wrapping a solver does
        # not silently lose those settings.
        self.base_solver = base_solver
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        self.low = low[:, None] if low.ndim == 1 else low
        self.high = high[:, None] if high.ndim == 1 else high

    @property
    def block_size(self):
        return self.base_solver.block_size

    @property
    def recompute_coupling_per_stage(self):
        return self.base_solver.recompute_coupling_per_stage

    @property
    def tbptt_window(self):
        return self.base_solver.tbptt_window

    def step(
        self,
        dynamics_fn: Callable,
        t: float,
        state: jnp.ndarray,
        dt: float,
        params: Bunch,
        noise_sample: jnp.ndarray = 0.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Integration step with state clipping.

        Delegates to base solver, then clips output state to bounds.
        Input state is not clipped (already clipped from previous iteration).

        Args:
            dynamics_fn: Dynamics function (t, state, params) -> (derivatives, auxiliaries)
            t: Current time
            state: Current state [n_states, n_nodes]
            dt: Time step
            params: Parameters
            noise_sample: Pre-scaled noise increment

        Returns:
            Tuple of (next_state, auxiliaries):
                - next_state: Clipped next state [n_states, n_nodes]
                - auxiliaries: Auxiliary variables [n_auxiliaries, n_nodes]
        """
        # Delegate to wrapped solver
        next_state, auxiliaries = self.base_solver.step(
            dynamics_fn, t, state, dt, params, noise_sample
        )

        # Clip output to bounds
        next_state = jnp.clip(next_state, self.low, self.high)

        return next_state, auxiliaries
