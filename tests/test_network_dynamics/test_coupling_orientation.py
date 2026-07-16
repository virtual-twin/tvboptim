"""Regression tests for coupling orientation on directed graphs."""

import jax.numpy as jnp
import numpy as np

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.coupling import (
    FastLinearCoupling,
    LinearCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import Linear
from tvboptim.experimental.network_dynamics.graph import DenseGraph


def _compute(coupling, graph, state):
    network = Network(Linear(), {"instant": coupling}, graph)
    coupling_data, coupling_state = coupling.prepare(network, dt=0.1, t0=0.0, t1=1.0)
    return coupling.compute(
        0.0,
        state,
        coupling_data,
        coupling_state,
        coupling.params,
        graph,
    )


def test_fast_linear_uses_target_source_orientation_on_directed_graph():
    """Fast and per-edge linear coupling both reduce the source axis."""
    weights = jnp.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 0.0],
        ]
    )
    state = jnp.array([[5.0, 7.0, 11.0]])
    graph = DenseGraph(weights)

    linear = _compute(LinearCoupling(incoming_states="x"), graph, state)
    fast = _compute(FastLinearCoupling(local_states="x"), graph, state)

    expected = jnp.array([[14.0, 33.0, 5.0]])  # state @ weights.T
    np.testing.assert_array_equal(linear, expected)
    np.testing.assert_array_equal(fast, expected)
    np.testing.assert_array_equal(fast, linear)
