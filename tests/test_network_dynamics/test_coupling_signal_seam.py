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
    coupling = TwoChannelSigmoid(source=("x", "y"))
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
    coupling = LocalDifference(source="x", local="y")
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
    coupling = TwoChannelSigmoid(source=("x", "y"))
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


# The canonical source/local spelling and its deprecated incoming_states/
# local_states aliases must construct identically. Listed as (kwargs, legacy)
# selector pairs so new coupling shapes extend one list.
_ALIAS_CASES = [
    ({"source": "x"}, {"incoming_states": "x"}),
    ({"source": ("x", "y")}, {"incoming_states": ("x", "y")}),
    ({"source": "x", "local": "y"}, {"incoming_states": "x", "local_states": "y"}),
]


@pytest.mark.parametrize("new_kwargs, legacy_kwargs", _ALIAS_CASES)
def test_source_local_spelling_matches_deprecated_aliases(new_kwargs, legacy_kwargs):
    with pytest.warns(DeprecationWarning, match=r"(incoming_states|local_states)="):
        legacy = TwoChannelSigmoid(**legacy_kwargs)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # the new spelling must not warn
        current = TwoChannelSigmoid(**new_kwargs)

    assert current.INCOMING_STATE_NAMES == legacy.INCOMING_STATE_NAMES
    assert current.LOCAL_STATE_NAMES == legacy.LOCAL_STATE_NAMES


@pytest.mark.parametrize("graph_type", [DenseGraph, SparseGraph])
def test_deprecated_alias_produces_identical_results(graph_type):
    graph = graph_type(WEIGHTS)
    data = Bunch(
        incoming_indices=jnp.array([0, 1]),
        local_indices=jnp.array([], dtype=jnp.int32),
    )
    current = TwoChannelSigmoid(source=("x", "y"))
    with pytest.warns(DeprecationWarning):
        legacy = TwoChannelSigmoid(incoming_states=("x", "y"))

    via_new = current.compute(0.0, STATE, data, Bunch(), current.params, graph)
    via_legacy = legacy.compute(0.0, STATE, data, Bunch(), legacy.params, graph)
    assert jnp.array_equal(via_new, via_legacy)


def test_mixing_name_and_deprecated_alias_raises():
    with pytest.raises(ValueError, match="not both"):
        TwoChannelSigmoid(source="x", incoming_states="x")
