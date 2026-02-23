import jax.numpy as jnp
from ..solvers import Heun
from .. import prepare


def lyapunov(network, solver=None, t=1000.0, n=10, d0=1e-9, dt=0.1, t0=0.0):
    """
    Estimate the maximum Lyapunov exponent for a network.

    Uses Benettin's rescaling algorithm: two nearby trajectories are
    simulated for n segments of length t ms; their divergence rate is
    averaged to estimate the MLE.

    Parameters
    ----------
    network : Network
    solver : solver instance, optional  (default: Heun())
    t : float   — segment duration in ms
    n : int     — number of rescaling steps
    d0 : float  — initial perturbation magnitude
    dt : float  — integration timestep in ms
    t0 : float  — simulation start time

    Returns
    -------
    float — estimated maximum Lyapunov exponent (1/ms)

    Notes
    -----
    For networks with delayed coupling the delay history is held fixed
    across segments (good approximation when t >> max_delay). Run a
    warmup via network.update_history(result) before calling this
    function to initialise the delay buffer from a settled trajectory.
    """
    if solver is None:
        solver = Heun()
    solve_fn, config = prepare(network, solver, t0=t0, t1=t0 + t, dt=dt)
    return _lyapunov(solve_fn, config, t=t, n=n, d0=d0)


def _lyapunov(solve_fn, config, t, n=10, d0=1e-9):
    """Low-level MLE estimation using a pre-built solve_fn / config."""
    u1 = config.initial_state.dynamics        # [n_states, n_nodes]
    D  = u1.size
    u2 = u1 + d0 / jnp.sqrt(D)

    log_sum = jnp.zeros(())
    for _ in range(n):
        config.initial_state.dynamics = u1
        new_u1 = solve_fn(config).ys[-1]

        config.initial_state.dynamics = u2
        new_u2 = solve_fn(config).ys[-1]

        d = _ldist(new_u1, new_u2)
        log_sum = log_sum + jnp.log(d / d0)

        u1 = new_u1
        u2 = _lrescale(new_u1, new_u2, d / d0)

    return float(log_sum / (n * t))


def _ldist(u1, u2):
    return jnp.sqrt(jnp.sum((u1 - u2) ** 2))


def _lrescale(u1, u2, a):
    return u1 + (u2 - u1) / a
