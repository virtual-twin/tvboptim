"""Elementwise ``pre`` and declared edge-parameter contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling.base import (
    DelayedCoupling,
    InstantaneousCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import Linear
from tvboptim.experimental.network_dynamics.graph import (
    DenseDelayGraph,
    DenseGraph,
    SparseGraph,
)

WEIGHTS = jnp.array(
    [
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [5.0, 0.0, 0.0],
    ]
)
EDGE_VALUES = jnp.array(
    [
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0],
    ]
)


class EdgeScaledCoupling(InstantaneousCoupling):
    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(edge=1.0, gain=2.0)
    EDGE_PARAMS = ("edge",)

    def pre(self, incoming_states, local_states, params):
        assert local_states is None
        return params.edge * incoming_states

    def post(self, summed_inputs, local_states, params):
        return params.gain * summed_inputs


class DelayedEdgeScaledCoupling(DelayedCoupling):
    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(edge=1.0)
    EDGE_PARAMS = ("edge",)

    def pre(self, incoming_states, local_states, params):
        return params.edge * incoming_states

    def post(self, summed_inputs, local_states, params):
        return summed_inputs


def _prepare_data(coupling, graph):
    slot = "delayed" if isinstance(coupling, DelayedCoupling) else "instant"
    network = Network(Linear(), {slot: coupling}, graph)
    data, state = coupling.prepare(network, dt=0.1, t0=0.0, t1=0.2)
    return data, state


def _prepare_runtime(coupling, graph):
    network = Network(Linear(), {"instant": coupling}, graph)
    data, state = network.prepare(dt=0.1, t0=0.0, t1=0.2)
    return data.instant, state.instant


def _compute_instantaneous(coupling, graph, state, params=None):
    data, coupling_state = _prepare_runtime(coupling, graph)
    params = coupling.params if params is None else params
    data = coupling.precompute(data, params, graph)
    return coupling.compute(
        0.0,
        state,
        data,
        coupling_state,
        params,
        graph,
    )


@pytest.mark.parametrize("layout", ["graph", "prepared_e"])
def test_dense_edge_params_normalize_to_graph_shape(layout):
    graph = DenseGraph(WEIGHTS)
    value = EDGE_VALUES if layout == "graph" else graph.gather_edges(EDGE_VALUES)
    coupling = EdgeScaledCoupling(incoming_states="x", edge=value)
    data, _ = _prepare_data(coupling, graph)

    enriched = coupling.precompute(data, coupling.params, graph)
    np.testing.assert_array_equal(enriched.aligned_edge_params.edge, EDGE_VALUES)
    np.testing.assert_array_equal(
        coupling._build_pre_params(enriched, coupling.params).edge,
        EDGE_VALUES,
    )
    assert (
        coupling._build_pre_params(enriched, coupling.params).gain
        is coupling.params.gain
    )


@pytest.mark.parametrize("layout", ["graph", "prepared_e"])
def test_sparse_edge_params_normalize_to_public_prepared_order(layout):
    graph = SparseGraph(WEIGHTS)
    prepared_e = graph.gather_edges(EDGE_VALUES)
    value = EDGE_VALUES if layout == "graph" else prepared_e
    coupling = EdgeScaledCoupling(incoming_states="x", edge=value)
    data, _ = _prepare_data(coupling, graph)

    enriched = coupling.precompute(data, coupling.params, graph)
    np.testing.assert_array_equal(enriched.aligned_edge_params.edge, prepared_e)
    np.testing.assert_array_equal(
        enriched.aligned_edge_params.edge,
        EDGE_VALUES[graph.edge_indices[:, 0], graph.edge_indices[:, 1]],
    )


def test_edge_param_normalization_is_jittable_and_differentiable():
    graph = SparseGraph(WEIGHTS)
    coupling = EdgeScaledCoupling(
        incoming_states="x",
        edge=graph.gather_edges(EDGE_VALUES),
    )
    data, _ = _prepare_data(coupling, graph)

    def loss(edge):
        params = Bunch(edge=edge, gain=coupling.params.gain)
        enriched = coupling.precompute(data, params, graph)
        return jnp.sum(enriched.aligned_edge_params.edge**2)

    edge = coupling.params.edge
    np.testing.assert_allclose(jax.jit(loss)(edge), loss(edge))
    np.testing.assert_allclose(jax.grad(loss)(edge), 2.0 * edge)
    mapped = jax.vmap(loss)(jnp.stack([edge, 2.0 * edge]))
    np.testing.assert_allclose(mapped, jnp.array([loss(edge), 4.0 * loss(edge)]))


def test_delayed_precompute_extends_edge_alignment():
    graph = DenseDelayGraph(WEIGHTS, jnp.zeros_like(WEIGHTS))
    coupling = DelayedEdgeScaledCoupling(
        incoming_states="x",
        edge=graph.gather_edges(EDGE_VALUES),
    )
    data, _ = _prepare_data(coupling, graph)

    enriched = coupling.precompute(data, coupling.params, graph)
    np.testing.assert_array_equal(enriched.aligned_edge_params.edge, EDGE_VALUES)
    assert enriched.delay_indices.shape == WEIGHTS.shape


class LocalElementwiseCoupling(InstantaneousCoupling):
    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch()
    PRE_USES_LOCAL = True

    def pre(self, incoming_states, local_states, params):
        return incoming_states - local_states

    def post(self, summed_inputs, local_states, params):
        return summed_inputs


class LegacyReshapeCoupling(InstantaneousCoupling):
    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch()

    def pre(self, incoming_states, local_states, params):
        return incoming_states[:, :, None]

    def post(self, summed_inputs, local_states, params):
        return summed_inputs


class LegacyIndexReshapeCoupling(LegacyReshapeCoupling):
    def pre(self, incoming_states, local_states, params):
        return incoming_states[0][None, :, :]


class UndeclaredEdgeCoupling(LegacyReshapeCoupling):
    DEFAULT_PARAMS = Bunch(edge=1.0)

    def pre(self, incoming_states, local_states, params):
        return incoming_states


class MissingEdgeCoupling(UndeclaredEdgeCoupling):
    DEFAULT_PARAMS = Bunch()
    EDGE_PARAMS = ("missing",)


def test_prepare_probe_accepts_aligned_local_elementwise_pre():
    graph = DenseGraph(WEIGHTS)
    coupling = LocalElementwiseCoupling(incoming_states="x", local_states="x")
    data, _ = _prepare_data(coupling, graph)
    assert data.incoming_indices.shape == data.local_indices.shape == (1,)


def test_prepare_probe_rejects_legacy_explicit_reshape_with_migration_message():
    graph = DenseGraph(WEIGHTS)
    coupling = LegacyReshapeCoupling(incoming_states="x")

    with pytest.raises(ValueError, match=r"elementwise.*PRE_USES_LOCAL.*EDGE_PARAMS"):
        _prepare_data(coupling, graph)


def test_prepare_probe_wraps_errors_with_migration_message():
    graph = SparseGraph(WEIGHTS)
    coupling = LegacyIndexReshapeCoupling(incoming_states="x")

    with pytest.raises(ValueError, match="violates the coupling contract"):
        _prepare_data(coupling, graph)


def test_prepare_rejects_undeclared_graph_shaped_param():
    graph = DenseGraph(WEIGHTS)
    coupling = UndeclaredEdgeCoupling(incoming_states="x", edge=EDGE_VALUES)

    with pytest.raises(ValueError, match="not declared.*EDGE_PARAMS"):
        _prepare_data(coupling, graph)


def test_prepare_rejects_missing_or_unsupported_declared_edge_param():
    graph = SparseGraph(WEIGHTS)
    missing = MissingEdgeCoupling(incoming_states="x")
    with pytest.raises(ValueError, match="does not exist"):
        _prepare_data(missing, graph)

    invalid = EdgeScaledCoupling(incoming_states="x", edge=jnp.ones(4))
    with pytest.raises(ValueError, match=r"expected.*\(3, 3\).*\(3,\)"):
        _prepare_data(invalid, graph)


def test_prepared_e_fixture_is_constructed_without_private_indices():
    graph = SparseGraph(
        BCOO.fromdense(WEIGHTS),
    )
    prepared_e = graph.gather_edges(EDGE_VALUES)
    coupling = EdgeScaledCoupling(incoming_states="x", edge=prepared_e)
    data, _ = _prepare_data(coupling, graph)

    actual = coupling.precompute(data, coupling.params, graph)
    np.testing.assert_array_equal(actual.aligned_edge_params.edge, prepared_e)


@pytest.mark.parametrize("representation", ["dense", "sparse"])
@pytest.mark.parametrize("layout", ["graph", "prepared_e"])
def test_instantaneous_edge_params_execute_in_both_public_layouts(
    representation, layout
):
    graph = DenseGraph(WEIGHTS) if representation == "dense" else SparseGraph(WEIGHTS)
    edge = EDGE_VALUES if layout == "graph" else graph.gather_edges(EDGE_VALUES)
    coupling = EdgeScaledCoupling(incoming_states="x", edge=edge, gain=2.0)
    state = jnp.array([[2.0, 3.0, 7.0]])

    actual = _compute_instantaneous(coupling, graph, state)
    expected = (
        2.0
        * jnp.sum(
            WEIGHTS * EDGE_VALUES * state[0][None, :],
            axis=1,
            keepdims=True,
        ).T
    )

    np.testing.assert_allclose(actual, expected)


def test_sparse_prepared_edge_param_and_state_gradients_match_dense():
    dense_graph = DenseGraph(WEIGHTS)
    sparse_graph = SparseGraph(WEIGHTS)
    dense_coupling = EdgeScaledCoupling(
        incoming_states="x",
        edge=EDGE_VALUES,
        gain=2.0,
    )
    sparse_edge = sparse_graph.gather_edges(EDGE_VALUES)
    sparse_coupling = EdgeScaledCoupling(
        incoming_states="x",
        edge=sparse_edge,
        gain=2.0,
    )
    dense_data, dense_state = _prepare_runtime(dense_coupling, dense_graph)
    sparse_data, sparse_state = _prepare_runtime(sparse_coupling, sparse_graph)

    def dense_loss(edge, state):
        params = Bunch(edge=edge, gain=2.0)
        data = dense_coupling.precompute(dense_data, params, dense_graph)
        result = dense_coupling.compute(
            0.0, state, data, dense_state, params, dense_graph
        )
        return jnp.sum(result**2)

    def sparse_loss(edge, state):
        params = Bunch(edge=edge, gain=2.0)
        data = sparse_coupling.precompute(sparse_data, params, sparse_graph)
        result = sparse_coupling.compute(
            0.0, state, data, sparse_state, params, sparse_graph
        )
        return jnp.sum(result**2)

    state = jnp.array([[2.0, 3.0, 7.0]])
    dense_value, dense_grad = jax.jit(jax.value_and_grad(dense_loss, argnums=(0, 1)))(
        EDGE_VALUES, state
    )
    sparse_value, sparse_grad = jax.jit(
        jax.value_and_grad(sparse_loss, argnums=(0, 1))
    )(sparse_edge, state)

    np.testing.assert_allclose(sparse_value, dense_value)
    np.testing.assert_allclose(
        sparse_grad[0],
        sparse_graph.gather_edges(dense_grad[0]),
    )
    np.testing.assert_allclose(sparse_grad[1], dense_grad[1])
