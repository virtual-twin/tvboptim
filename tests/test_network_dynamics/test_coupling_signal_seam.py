"""Behavior-preserving signal-array seam for PrePostCoupling routes."""

import jax
import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
from tvboptim.experimental.network_dynamics.graph import DenseGraph, SparseGraph

WEIGHTS = jnp.array(
    [
        [0.0, 0.3, 0.0, 0.8],
        [0.2, 0.0, 0.5, 0.0],
        [0.0, 0.7, 0.0, 0.1],
        [0.6, 0.0, 0.4, 0.0],
    ]
)
STATE = jnp.array(
    [
        [0.2, -0.4, 0.8, 0.1],
        [0.5, 0.3, -0.2, 0.7],
    ]
)


class TwoChannelSigmoid(InstantaneousCoupling):
    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(gain=0.6, midpoint=0.1, slope=1.3)

    def pre(self, incoming_states, local_states, params):
        del local_states
        difference = incoming_states[0:1] - incoming_states[1:2]
        return jax.nn.sigmoid(params.slope * (difference - params.midpoint))

    def post(self, summed_inputs, local_states, params):
        del local_states
        return params.gain * summed_inputs


class LocalDifference(InstantaneousCoupling):
    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(gain=0.4)
    PRE_USES_LOCAL = True

    def pre(self, incoming_states, local_states, params):
        del params
        return incoming_states - local_states

    def post(self, summed_inputs, local_states, params):
        del local_states
        return params.gain * summed_inputs


@pytest.mark.parametrize("graph_type", [DenseGraph, SparseGraph])
def test_multichannel_state_adapter_matches_direct_signal_kernel(graph_type):
    graph = graph_type(WEIGHTS)
    coupling = TwoChannelSigmoid(incoming_states=("x", "y"))
    data = Bunch(
        incoming_indices=jnp.array([0, 1]),
        local_indices=jnp.array([], dtype=jnp.int32),
    )

    via_state = coupling.compute(0.0, STATE, data, Bunch(), coupling.params, graph)
    via_signals = coupling._compute_from_signals(
        STATE,
        STATE[:0],
        data,
        coupling.params,
        graph,
    )
    assert jnp.array_equal(via_signals, via_state)


@pytest.mark.parametrize("graph_type", [DenseGraph, SparseGraph])
def test_local_state_adapter_matches_direct_signal_kernel(graph_type):
    graph = graph_type(WEIGHTS)
    coupling = LocalDifference(incoming_states="x", local_states="y")
    data = Bunch(
        incoming_indices=jnp.array([0]),
        local_indices=jnp.array([1]),
    )

    via_state = coupling.compute(0.0, STATE, data, Bunch(), coupling.params, graph)
    via_signals = coupling._compute_from_signals(
        STATE[0:1],
        STATE[1:2],
        data,
        coupling.params,
        graph,
    )
    assert jnp.array_equal(via_signals, via_state)


@pytest.mark.parametrize("graph_type", [DenseGraph, SparseGraph])
def test_direct_signal_kernel_preserves_jit_and_gradients(graph_type):
    graph = graph_type(WEIGHTS)
    coupling = TwoChannelSigmoid(incoming_states=("x", "y"))
    data = Bunch(
        incoming_indices=jnp.array([0, 1]),
        local_indices=jnp.array([], dtype=jnp.int32),
    )

    def loss(signal, gain):
        params = coupling.params.copy()
        params.gain = gain
        result = coupling._compute_from_signals(signal, signal[:0], data, params, graph)
        return jnp.square(result).sum()

    value, gradients = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(
        STATE, coupling.params.gain
    )
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradients[0]))
    assert jnp.isfinite(gradients[1])
