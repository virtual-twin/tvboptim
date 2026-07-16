"""Array-only message transport orientation and gradient contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.experimental.network_dynamics.coupling.transport import (
    _aggregate_nodes,
    _reduce_edges,
)

MESSAGES = jnp.array(
    [
        [2.0, 3.0, 5.0],
        [7.0, 11.0, 13.0],
        [17.0, 19.0, 23.0],
    ]
)
WEIGHTS = jnp.array(
    [
        [0.0, 2.0, 3.0],
        [5.0, 0.0, 7.0],
    ]
)
TARGET_E = jnp.array([0, 0, 1, 1])
SOURCE_E = jnp.array([1, 2, 0, 2])
WEIGHTS_E = WEIGHTS[TARGET_E, SOURCE_E]


def test_aggregate_nodes_uses_target_source_orientation_rectangular():
    actual = _aggregate_nodes(MESSAGES, WEIGHTS)
    expected = np.einsum("qs,ts->qt", np.asarray(MESSAGES), np.asarray(WEIGHTS))

    assert actual.shape == (3, 2)
    np.testing.assert_array_equal(actual, expected)


def test_reduce_edges_matches_rectangular_dense_oracle():
    edge_messages = MESSAGES[:, SOURCE_E]
    actual = _reduce_edges(edge_messages, WEIGHTS_E, TARGET_E, n_target=2)
    expected = _aggregate_nodes(MESSAGES, WEIGHTS)

    assert actual.shape == (3, 2)
    np.testing.assert_array_equal(actual, expected)


def test_edge_reduce_jit_vmap_and_gradients_keep_numerical_data_live():
    edge_messages = MESSAGES[:, SOURCE_E]

    def loss(messages, weights):
        return _reduce_edges(messages, weights, TARGET_E, n_target=2).sum()

    eager = loss(edge_messages, WEIGHTS_E)
    compiled = jax.jit(loss)(edge_messages, WEIGHTS_E)
    np.testing.assert_array_equal(compiled, eager)

    weight_batches = jnp.stack([WEIGHTS_E, 2.0 * WEIGHTS_E])
    mapped = jax.vmap(
        lambda weights: _reduce_edges(
            edge_messages,
            weights,
            TARGET_E,
            n_target=2,
        )
    )(weight_batches)
    np.testing.assert_array_equal(mapped[0], _aggregate_nodes(MESSAGES, WEIGHTS))
    np.testing.assert_array_equal(mapped[1], 2.0 * mapped[0])

    grad_messages, grad_weights = jax.grad(loss, argnums=(0, 1))(
        edge_messages,
        WEIGHTS_E,
    )
    np.testing.assert_array_equal(
        grad_messages,
        jnp.broadcast_to(WEIGHTS_E, edge_messages.shape),
    )
    np.testing.assert_array_equal(grad_weights, edge_messages.sum(axis=0))
