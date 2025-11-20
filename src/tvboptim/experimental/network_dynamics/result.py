"""Result types for  Network Dynamics native solvers.

This module provides Diffrax-like solution objects for native  Network Dynamics solvers,
ensuring consistent API across solver types.
"""

import jax.numpy as jnp
from jax import tree_util


@tree_util.register_pytree_node_class
class NativeSolution:
    """Solution object for Network Dynamics native solvers.

    Provides the same interface as Diffrax solutions (.ys, .ts) while being
    a proper JAX PyTree for compatibility with JAX transformations.

    Attributes:
        ts: Time points, shape [n_time]
        ys: Trajectory data, shape [n_time, n_states, n_nodes]
        dt: Time step (optional), stored as static auxiliary data
    """

    def __init__(self, ts: jnp.ndarray, ys: jnp.ndarray, dt: float = None):
        """Initialize native solution.

        Args:
            ts: Time points, shape [n_time]
            ys: Trajectory array, shape [n_time, n_states, n_nodes]
            dt: Time step (optional), stored as static auxiliary data for JIT compatibility
        """
        self.ts = ts
        self.ys = ys
        self.dt = dt

    @property
    def time(self):
        return self.ts

    @property
    def data(self):
        return self.ys

    def tree_flatten(self):
        """JAX PyTree flatten for transformations."""
        children = (self.ts, self.ys)
        aux_data = {"dt": self.dt}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """JAX PyTree unflatten for transformations."""
        ts, ys = children
        dt = aux_data.get("dt") if aux_data else None
        return cls(ts, ys, dt=dt)

    def __repr__(self):
        """String representation."""
        return f"NativeSolution(shape={self.ys.shape}, t=[{self.ts[0]:.2f}, {self.ts[-1]:.2f}])"


def wrap_native_result(
    trajectory: jnp.ndarray, t0: float, t1: float, dt: float
) -> NativeSolution:
    """Wrap native solver trajectory in solution object.

    Args:
        trajectory: Trajectory array from native solver, shape [n_time, n_states, n_nodes]
        t0: Start time
        t1: End time
        dt: Time step

    Returns:
        NativeSolution with .ys and .ts attributes like Diffrax
    """
    n_steps = trajectory.shape[0]
    ts = jnp.linspace(t0, t1, n_steps)
    return NativeSolution(ts=ts, ys=trajectory, dt=dt)
