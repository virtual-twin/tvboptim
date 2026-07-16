"""Integrated instantaneous dense/sparse message-passing contracts."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import Linear
from tvboptim.experimental.network_dynamics.graph import DenseGraph, SparseGraph

from .coupling_message_passing_oracle import coupling_oracle


class ThreeChannelLocalCoupling(InstantaneousCoupling):
    """Exercise Q>1 over one topology without coupling-specific transport."""

    N_OUTPUT_STATES = 3
    DEFAULT_PARAMS = Bunch()
    PRE_USES_LOCAL = True

    def pre(self, incoming_states, local_states, params):
        difference = incoming_states - local_states
        return jnp.concatenate(
            [difference, incoming_states + local_states, jnp.sin(difference)],
            axis=0,
        )

    def post(self, summed_inputs, local_states, params):
        return summed_inputs


WEIGHTS = np.array(
    [
        [0.0, 2.0, 0.0, 0.0],
        [3.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 5.0],
    ],
    dtype=np.float32,
)
STATE = np.array([[0.2, 1.1, -0.7, 0.4]], dtype=np.float32)


def _sparse_graph(edge_indices):
    edge_indices = jnp.asarray(edge_indices)
    data = jnp.asarray(WEIGHTS)[edge_indices[:, 0], edge_indices[:, 1]]
    return SparseGraph(
        BCOO(
            (data, edge_indices),
            shape=WEIGHTS.shape,
            unique_indices=True,
        )
    )


def _compute(graph):
    coupling = ThreeChannelLocalCoupling(
        incoming_states="x",
        local_states="x",
    )
    network = Network(Linear(), {"instant": coupling}, graph)
    data, coupling_state = network.prepare(dt=0.1, t0=0.0, t1=0.2)
    return network.compute_coupling_inputs(
        0.0,
        jnp.asarray(STATE),
        data,
        coupling_state,
    ).instant


def _oracle(edge_indices=None):
    def pre(source, target):
        difference = source - target
        return np.concatenate(
            [difference, source + target, np.sin(difference)],
            axis=0,
        )

    return coupling_oracle(
        STATE,
        WEIGHTS,
        pre=pre,
        post=lambda summed, _local: summed,
        target_local=STATE,
        edge_indices=edge_indices,
    )


def test_q3_local_messages_match_dense_oracle_and_sparse_edge_orders():
    edge_indices = np.argwhere(WEIGHTS != 0.0)
    reversed_indices = edge_indices[::-1].copy()

    dense = _compute(DenseGraph(jnp.asarray(WEIGHTS)))
    sparse = _compute(_sparse_graph(edge_indices))
    reordered = _compute(_sparse_graph(reversed_indices))

    np.testing.assert_allclose(dense, _oracle(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(sparse, _oracle(edge_indices), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        reordered,
        _oracle(reversed_indices),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(sparse, dense, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(reordered, dense, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(dense[:, 2], jnp.zeros(3))


def test_genuinely_empty_sparse_graph_returns_zero_for_every_channel():
    empty_indices = jnp.empty((0, 2), dtype=jnp.int32)
    graph = SparseGraph(
        BCOO(
            (jnp.empty((0,), dtype=jnp.float32), empty_indices),
            shape=WEIGHTS.shape,
            unique_indices=True,
        )
    )

    actual = _compute(graph)

    assert actual.shape == (3, 4)
    np.testing.assert_array_equal(actual, jnp.zeros_like(actual))


def test_q3_weight_and_state_gradients_sparse_match_dense_under_jit():
    edge_indices = np.argwhere(WEIGHTS != 0.0)
    dense_graph = DenseGraph(jnp.asarray(WEIGHTS))
    sparse_graph = _sparse_graph(edge_indices)
    dense_coupling = ThreeChannelLocalCoupling(
        incoming_states="x",
        local_states="x",
    )
    sparse_coupling = ThreeChannelLocalCoupling(
        incoming_states="x",
        local_states="x",
    )
    dense_network = Network(Linear(), {"instant": dense_coupling}, dense_graph)
    sparse_network = Network(Linear(), {"instant": sparse_coupling}, sparse_graph)
    dense_data, dense_coupling_state = dense_network.prepare(0.1, 0.0, 0.2)
    sparse_data, sparse_coupling_state = sparse_network.prepare(0.1, 0.0, 0.2)
    dense_aux = dense_graph.tree_flatten()[1]
    sparse_aux = sparse_graph.tree_flatten()[1]
    indices = sparse_graph.edge_indices

    def dense_loss(weights, state):
        graph = DenseGraph.tree_unflatten(dense_aux, (weights,))
        data = dense_coupling.precompute(
            dense_data.instant,
            dense_coupling.params,
            graph,
        )
        result = dense_coupling.compute(
            0.0,
            state,
            data,
            dense_coupling_state.instant,
            dense_coupling.params,
            graph,
        )
        return jnp.sum(result**2)

    def sparse_loss(weights_e, state):
        weights = BCOO(
            (weights_e, indices),
            shape=WEIGHTS.shape,
            unique_indices=True,
        )
        graph = SparseGraph.tree_unflatten(sparse_aux, (weights,))
        data = sparse_coupling.precompute(
            sparse_data.instant,
            sparse_coupling.params,
            graph,
        )
        result = sparse_coupling.compute(
            0.0,
            state,
            data,
            sparse_coupling_state.instant,
            sparse_coupling.params,
            graph,
        )
        return jnp.sum(result**2)

    state = jnp.asarray(STATE)
    dense_value, dense_grad = jax.jit(jax.value_and_grad(dense_loss, argnums=(0, 1)))(
        dense_graph.weights, state
    )
    sparse_value, sparse_grad = jax.jit(
        jax.value_and_grad(sparse_loss, argnums=(0, 1))
    )(sparse_graph.weights.data, state)

    np.testing.assert_allclose(sparse_value, dense_value, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        sparse_grad[0],
        dense_grad[0][indices[:, 0], indices[:, 1]],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        sparse_grad[1],
        dense_grad[1],
        rtol=1e-6,
        atol=1e-7,
    )
