"""Stable edge-order and sparse weight/delay alignment contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from tvboptim.experimental.network_dynamics.graph import (
    DenseGraph,
    SparseDelayGraph,
    SparseGraph,
)


def _bcoo(data, indices, shape=(3, 3)):
    return BCOO((jnp.asarray(data, dtype=float), jnp.asarray(indices)), shape=shape)


def test_dense_gather_edges_is_target_major_and_transform_safe():
    graph = DenseGraph(jnp.eye(3))
    graph_shaped = jnp.arange(9).reshape(3, 3)
    expected = np.arange(9)

    np.testing.assert_array_equal(graph.gather_edges(graph_shaped), expected)
    np.testing.assert_array_equal(jax.jit(graph.gather_edges)(graph_shaped), expected)
    np.testing.assert_array_equal(
        jax.vmap(graph.gather_edges)(jnp.stack([graph_shaped, graph_shaped + 10])),
        np.stack([expected, expected + 10]),
    )

    with pytest.raises(ValueError, match="must match the graph shape"):
        graph.gather_edges(jnp.ones((3, 2)))


def test_sparse_edge_order_is_public_and_gather_uses_it():
    indices = jnp.array([[2, 0], [0, 1], [1, 2]])
    graph = SparseGraph(_bcoo([2.0, 3.0, 4.0], indices))
    graph_shaped = jnp.arange(9).reshape(3, 3)
    expected = np.array([6, 1, 5])

    np.testing.assert_array_equal(graph.edge_indices, indices)
    np.testing.assert_array_equal(graph.gather_edges(graph_shaped), expected)
    np.testing.assert_array_equal(jax.jit(graph.gather_edges)(graph_shaped), expected)
    np.testing.assert_array_equal(
        jax.vmap(graph.gather_edges)(jnp.stack([graph_shaped, graph_shaped + 10])),
        np.stack([expected, expected + 10]),
    )

    with pytest.raises(AttributeError):
        graph.edge_indices = jnp.zeros_like(indices)
    with pytest.raises(ValueError, match="must match the graph shape"):
        graph.gather_edges(jnp.ones((3, 2)))


def test_sparse_graph_rejects_duplicate_topology_entries():
    weights = _bcoo(
        [1.0, 2.0, 3.0],
        [[0, 1], [0, 1], [2, 0]],
    )
    with pytest.raises(ValueError, match="duplicate edge indices"):
        SparseGraph(weights)


def test_dense_delays_are_gathered_without_densifying_or_dropping_zero(
    monkeypatch,
):
    weights = _bcoo(
        [2.0, 3.0, 4.0],
        [[2, 0], [0, 1], [1, 2]],
    )
    delays = jnp.array(
        [
            [9.0, 1.5, 9.0],
            [9.0, 9.0, 3.5],
            [0.0, 9.0, 9.0],
        ]
    )

    def fail_if_densified(_self):
        raise AssertionError("SparseDelayGraph construction must stay O(E)")

    monkeypatch.setattr(BCOO, "todense", fail_if_densified)
    graph = SparseDelayGraph(weights, delays)

    assert graph.weights.indices is graph.delays.indices
    assert graph.delays.nse == graph.weights.nse == 3
    np.testing.assert_array_equal(graph.delays.data, [0.0, 1.5, 3.5])


def test_sparse_delays_are_reindexed_to_the_weight_edge_order():
    weight_indices = jnp.array([[2, 0], [0, 1], [1, 2]])
    weights = _bcoo([2.0, 3.0, 4.0], weight_indices)
    delays = _bcoo(
        [3.5, 0.0, 1.5],
        [[1, 2], [2, 0], [0, 1]],
    )

    graph = SparseDelayGraph(weights, delays)

    assert graph.weights.indices is graph.delays.indices
    np.testing.assert_array_equal(graph.edge_indices, weight_indices)
    np.testing.assert_array_equal(graph.delays.data, [0.0, 1.5, 3.5])


@pytest.mark.parametrize(
    "delay_indices, message",
    [
        ([[2, 0], [0, 1]], "exactly match weights"),
        ([[2, 0], [0, 1], [0, 2]], "exactly match weights"),
        ([[2, 0], [2, 0], [1, 2]], "duplicate edge indices"),
    ],
)
def test_sparse_delay_constructor_rejects_invalid_delay_topology(
    delay_indices, message
):
    weights = _bcoo(
        [2.0, 3.0, 4.0],
        [[2, 0], [0, 1], [1, 2]],
    )
    delays = _bcoo([0.0] * len(delay_indices), delay_indices)

    with pytest.raises(ValueError, match=message):
        SparseDelayGraph(weights, delays)


@pytest.mark.parametrize("graph_type", [SparseGraph, SparseDelayGraph])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("allow_self_loops", [False, True])
def test_random_sparse_graphs_have_unique_exact_topologies(
    graph_type, symmetric, allow_self_loops
):
    kwargs = dict(
        n_nodes=5,
        density=1.0,
        symmetric=symmetric,
        allow_self_loops=allow_self_loops,
        weight_dist="binary",
        key=jax.random.key(7),
    )
    if graph_type is SparseDelayGraph:
        kwargs.update(delay_dist="constant", max_delay=2.0)
    graph = graph_type.random(**kwargs)

    indices = np.asarray(graph.edge_indices)
    expected_edges = 25 if allow_self_loops else 20
    assert len(indices) == expected_edges
    assert len(set(map(tuple, indices))) == expected_edges
    if not allow_self_loops:
        assert np.all(indices[:, 0] != indices[:, 1])

    dense_weights = np.asarray(graph.weights.todense())
    if symmetric:
        np.testing.assert_array_equal(dense_weights, dense_weights.T)

    if graph_type is SparseDelayGraph:
        assert graph.weights.indices is graph.delays.indices
        np.testing.assert_array_equal(graph.delays.indices, indices)
        if symmetric:
            dense_delays = np.asarray(graph.delays.todense())
            np.testing.assert_array_equal(dense_delays, dense_delays.T)


@pytest.mark.parametrize("symmetric, expected_edges", [(False, 10), (True, 10)])
def test_random_sparse_density_controls_the_exact_unique_edge_count(
    symmetric, expected_edges
):
    graph = SparseGraph.random(
        n_nodes=6,
        density=0.35,
        symmetric=symmetric,
        allow_self_loops=False,
        key=jax.random.key(2),
    )

    assert graph.nnz == expected_edges
    assert len(set(map(tuple, np.asarray(graph.edge_indices)))) == expected_edges


@pytest.mark.parametrize("graph_type", [SparseGraph, SparseDelayGraph])
@pytest.mark.parametrize("n_nodes, density", [(4, 0.0), (1, 0.5)])
def test_random_sparse_graph_can_be_genuinely_empty(graph_type, n_nodes, density):
    graph = graph_type.random(
        n_nodes=n_nodes,
        density=density,
        symmetric=False,
        allow_self_loops=False,
        key=jax.random.key(3),
    )

    assert graph.edge_indices.shape == (0, 2)
    assert graph.weights.data.shape == (0,)
    if graph_type is SparseDelayGraph:
        assert graph.weights.indices is graph.delays.indices
        assert graph.delays.data.shape == (0,)


@pytest.mark.parametrize("density", [-0.01, 1.01])
def test_random_sparse_density_must_be_a_probability(density):
    with pytest.raises(ValueError, match="between 0 and 1"):
        SparseGraph.random(3, density=density)
