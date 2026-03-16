import jax
import jax.numpy as jnp
from ..solvers import Heun
from ..core.bunch import Bunch
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
    u_init = config.initial_state.dynamics     # [n_states, n_nodes]
    D = u_init.size
    u2_init = u_init + d0 / jnp.sqrt(D)

    def _solve_from(u):
        new_is = Bunch({**config.initial_state, 'dynamics': u})
        new_cfg = Bunch({**config, 'initial_state': new_is})
        return solve_fn(new_cfg).ys[-1]

    @jax.jit
    def _scan(u1, u2):
        def _step(carry, _):
            u1, u2, log_sum = carry
            new_u1 = _solve_from(u1)
            new_u2 = _solve_from(u2)
            d = jnp.sqrt(jnp.sum((new_u1 - new_u2) ** 2))
            log_sum = log_sum + jnp.log(d / d0)
            u2 = new_u1 + (new_u2 - new_u1) / (d / d0)
            return (new_u1, u2, log_sum), None

        init = (u1, u2, jnp.zeros(()))
        (_, _, log_sum), _ = jax.lax.scan(_step, init, None, length=n)
        return log_sum

    log_sum = _scan(u_init, u2_init)
    return float(log_sum / (n * t))


def lyapunov_spectrum(network, solver=None, t=1000.0, n=10, k=None,
                      dt=0.1, t0=0.0, mode="jvp", d0=1e-9):
    """
    Estimate the Lyapunov spectrum of a network.

    Parameters
    ----------
    network : Network
    solver : solver instance, optional  (default: Heun())
    t : float   — segment duration in ms
    n : int     — number of rescaling steps
    k : int     — number of exponents to compute (default: all D)
    dt : float  — integration timestep in ms
    t0 : float  — simulation start time
    mode : str  — "jvp" (default) uses tangent-space propagation via
                  jax.linearize; exact and efficient for differentiable
                  systems. "sim" uses finite-difference with d0-scaled
                  perturbations; works for non-differentiable systems.
    d0 : float  — perturbation magnitude (only used when mode="sim")

    Returns
    -------
    jnp.ndarray — top k Lyapunov exponents sorted descending (1/ms)
    """
    if solver is None:
        solver = Heun()
    solve_fn, config = prepare(network, solver, t0=t0, t1=t0 + t, dt=dt)
    if mode == "jvp":
        return _lyapunov_spectrum_jvp(solve_fn, config, t=t, n=n, k=k)
    elif mode == "sim":
        return _lyapunov_spectrum_sim(solve_fn, config, t=t, n=n, k=k, d0=d0)
    else:
        raise ValueError(f"Unknown mode {mode!r}, expected 'jvp' or 'sim'")


def _lyapunov_spectrum_sim(solve_fn, config, t, n=10, k=None, d0=1e-9):
    """Spectrum via finite-difference (k+1 trajectories per step)."""
    u_init = config.initial_state.dynamics
    shape = u_init.shape
    D = u_init.size
    if k is None:
        k = D
    Q_init = jnp.eye(D, k)  # [D, k] — first k columns of identity

    def _solve_flat(u_flat):
        u = u_flat.reshape(shape)
        new_is = Bunch({**config.initial_state, 'dynamics': u})
        new_cfg = Bunch({**config, 'initial_state': new_is})
        return solve_fn(new_cfg).ys[-1].flatten()

    @jax.jit
    def _scan(u_flat, Q):
        def _step(carry, _):
            u_flat, Q, log_sum = carry
            new_u = _solve_flat(u_flat)
            perturbed = u_flat + d0 * Q.T                   # [k, D]
            new_perturbed = jax.vmap(_solve_flat)(perturbed) # [k, D]
            delta = (new_perturbed - new_u) / d0            # [k, D]
            Q_new, R = jnp.linalg.qr(delta.T)              # [D, k], [k, k]
            log_sum = log_sum + jnp.log(jnp.abs(jnp.diag(R)))
            return (new_u, Q_new, log_sum), None

        init = (u_flat, Q, jnp.zeros(k))
        (_, _, log_sum), _ = jax.lax.scan(_step, init, None, length=n)
        return log_sum

    log_sum = _scan(u_init.flatten(), Q_init)
    return jnp.sort(log_sum / (n * t))[::-1]


def _lyapunov_spectrum_jvp(solve_fn, config, t, n=10, k=None):
    """Spectrum via tangent-space propagation (linearize + vmap)."""
    u_init = config.initial_state.dynamics
    shape = u_init.shape
    D = u_init.size
    if k is None:
        k = D
    Q_init = jnp.eye(D, k)  # [D, k]

    def _solve_flat(u_flat):
        u = u_flat.reshape(shape)
        new_is = Bunch({**config.initial_state, 'dynamics': u})
        new_cfg = Bunch({**config, 'initial_state': new_is})
        return solve_fn(new_cfg).ys[-1].flatten()

    @jax.jit
    def _scan(u_flat, Q):
        def _step(carry, _):
            u_flat, Q, log_sum = carry
            new_u, jvp_fn = jax.linearize(_solve_flat, u_flat)
            tangents = jax.vmap(jvp_fn)(Q.T)       # [k, D]
            Q_new, R = jnp.linalg.qr(tangents.T)   # [D, k], [k, k]
            log_sum = log_sum + jnp.log(jnp.abs(jnp.diag(R)))
            return (new_u, Q_new, log_sum), None

        init = (u_flat, Q, jnp.zeros(k))
        (_, _, log_sum), _ = jax.lax.scan(_step, init, None, length=n)
        return log_sum

    log_sum = _scan(u_init.flatten(), Q_init)
    return jnp.sort(log_sum / (n * t))[::-1]
