"""Adiabatic parameter scan: a network bifurcation diagram.

Slowly ramp one parameter from ``low`` to ``high`` (and optionally back down to
catch hysteresis), carrying the settled state forward between steps, and record
summary statistics of an observed network signal at each value. This traces a
bifurcation-diagram-like picture of how the network's activity changes with the
swept parameter.

The swept parameter is addressed with a lens ``accessor`` applied through
``eqx.tree_at`` (e.g. ``lambda c: c.coupling.instant.G``), so any nested config
field can be scanned without the function knowing its name.
"""

from dataclasses import dataclass
from typing import Callable, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .. import prepare
from ..core.bunch import Bunch
from ..solvers import Heun


@dataclass
class AdiabaticScanResult:
    """Result of an :func:`adiabatic_scan`.

    Attributes
    ----------
    p : jax.Array
        The swept parameter values, in scan order. Length ``2*n`` when
        ``bothways`` (up then down), else ``n``.
    n_up : int
        Number of values in the ascending branch. ``p[:n_up]`` is the up-branch
        and ``p[n_up:]`` the down-branch.
    stats : Bunch of str -> jax.Array
        One array per recorded statistic, stacked along the scan axis and
        reachable by attribute (``stats.mean``) or key (``stats["mean"]``).
        Shape is ``[len(p)]`` for scalar reducers and ``[len(p), ...]`` for
        vector-valued reducers (e.g. ``[len(p), n_nodes]`` for a per-node
        statistic).
    """

    p: jax.Array
    n_up: int
    stats: Bunch


def _default_observe(result):
    """Observe the first variable across all nodes -> [n_time, n_nodes]."""
    return result.ys[:, 0, :]


def _network_mean(arr):
    """Mean over time per node, then averaged across nodes."""
    return jnp.mean(arr, axis=0).mean()


def _network_min(arr):
    """Mean over time per node, then the minimum across nodes."""
    return jnp.mean(arr, axis=0).min()


def _network_max(arr):
    """Mean over time per node, then the maximum across nodes."""
    return jnp.mean(arr, axis=0).max()


_DEFAULT_STATISTICS = {
    "mean": _network_mean,
    "min": _network_min,
    "max": _network_max,
}


def adiabatic_scan(
    network,
    solver=None,
    *,
    accessor: Callable,
    low: float,
    high: float,
    n: int,
    t: float = 2000.0,
    skip: float = 1000.0,
    dt: float = 1.0,
    t0: float = 0.0,
    bothways: bool = True,
    observe: Callable = None,
    statistics: Dict[str, Callable] = None,
) -> AdiabaticScanResult:
    """Ramp one parameter and record network statistics (bifurcation diagram).

    Parameters
    ----------
    network : Network
    solver : solver instance, optional  (default: Heun())
    accessor : callable
        Lens onto the swept leaf, used as ``eqx.tree_at(accessor, config, value)``.
        Example: ``lambda c: c.coupling.instant.G``.
    low, high : float
        Bounds of the swept parameter.
    n : int
        Number of values per branch.
    t : float
        Simulation duration per step in ms.
    skip : float
        Transient duration in ms discarded before computing statistics.
    dt : float
        Integration timestep in ms.
    t0 : float
        Simulation start time.
    bothways : bool
        If True, scan up then back down to expose hysteresis.
    observe : callable, optional
        ``result -> [n_time, n_nodes]`` signal to summarise. Defaults to the
        first variable across all nodes.
    statistics : dict of str -> callable, optional
        Maps a name to a reducer ``[n_time, n_nodes] -> scalar or array``.
        Vector-valued reducers (e.g. a per-node ``[n_nodes]`` statistic) are
        supported as long as the output shape is the same at every scan point.
        Defaults to mean/min/max of the per-node temporal mean across the
        network.

    Returns
    -------
    AdiabaticScanResult

    Notes
    -----
    The settled state is carried forward between steps (the slow, adiabatic
    ramp). For networks with delayed coupling the delay history buffer is not
    carried, so this is only exact for instantaneous (non-delayed) coupling.
    """
    if solver is None:
        solver = Heun()
    if observe is None:
        observe = _default_observe
    if statistics is None:
        statistics = _DEFAULT_STATISTICS

    solve_fn, config = prepare(network, solver, t0=t0, t1=t0 + t, dt=dt)

    p_up = jnp.linspace(low, high, n)
    if bothways:
        p = jnp.concatenate([p_up, p_up[::-1]])
    else:
        p = p_up

    n_states = config.initial_state.dynamics.shape[0]
    init_state = config.initial_state.dynamics

    # The save grid is deterministic in (t0, t1, dt), so the post-transient
    # window is the same at every step. Resolve it once to static integer
    # indices (a host-side computation) so the scan body has no data-dependent
    # shapes and stays jittable.
    probe_ts = np.asarray(solve_fn(config).ts)
    keep = jnp.asarray(np.flatnonzero(probe_ts > (t0 + skip)))

    def step(state, value):
        cfg = eqx.tree_at(accessor, config, value)
        cfg = eqx.tree_at(lambda c: c.initial_state.dynamics, cfg, state)
        result = solve_fn(cfg)

        # statistics over the post-transient window. Reducer outputs stay on the
        # JAX side; lax.scan stacks them along the scan axis, so array-valued
        # reducers like a per-node median ([n_nodes]) are supported.
        arr = observe(result)[keep]
        outs = {name: fn(arr) for name, fn in statistics.items()}

        # carry the settled state forward (the slow, adiabatic ramp)
        new_state = result.ys[-1][:n_states]
        return new_state, outs

    # The whole sweep is one jitted scan: a single XLA compilation, the carry
    # expressed natively, and the function is vmap-able over a batch of `p`
    # arrays to explore several ranges in parallel.
    _, stacked = jax.jit(lambda s, ps: jax.lax.scan(step, s, ps))(init_state, p)

    return AdiabaticScanResult(p=p, n_up=n, stats=Bunch(stacked))
