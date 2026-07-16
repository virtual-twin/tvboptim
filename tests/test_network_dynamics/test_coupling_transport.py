"""Array-only message transport orientation and gradient contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.experimental.network_dynamics.coupling.transport import (
    _aggregate_nodes,
    _gather_history,
    _interpolate_history,
    _reduce_edges,
    _roll_history,
    _write_history,
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


def test_history_gather_is_array_only_channel_major_and_rectangular_ready():
    history = jnp.arange(5 * 3 * 3, dtype=jnp.float32).reshape(5, 3, 3)
    read_dense = jnp.array([[3, 2, 1], [0, 1, 2]])
    source = jnp.arange(3)

    dense = _gather_history(history, read_dense, source)
    expected = np.empty((3, 2, 3), dtype=np.float32)
    for channel in range(3):
        for target in range(2):
            for source_node in range(3):
                expected[channel, target, source_node] = history[
                    read_dense[target, source_node], channel, source_node
                ]

    assert dense.shape == (3, 2, 3)
    np.testing.assert_array_equal(dense, expected)

    target_e = jnp.array([1, 0, 1, 0])
    source_e = jnp.array([2, 1, 0, 2])
    read_e = read_dense[target_e, source_e]
    edges = _gather_history(history, read_e, source_e)
    np.testing.assert_array_equal(edges, dense[:, target_e, source_e])


def test_interpolated_history_and_updates_accept_supplied_signals():
    history = jnp.arange(5 * 3 * 3, dtype=jnp.float32).reshape(5, 3, 3)
    read_lo = jnp.array([3, 2, 1, 2])
    read_hi = read_lo - 1
    source_e = jnp.array([2, 1, 0, 2])
    fraction = jnp.array([0.0, 0.25, 0.5, 0.75])

    actual = jax.jit(_interpolate_history)(
        history,
        read_lo,
        read_hi,
        source_e,
        fraction,
    )
    lo = _gather_history(history, read_lo, source_e)
    hi = _gather_history(history, read_hi, source_e)
    np.testing.assert_array_equal(
        actual,
        (1.0 - fraction[None, :]) * lo + fraction[None, :] * hi,
    )

    transmitted = jnp.arange(9, dtype=history.dtype).reshape(3, 3) + 100.0
    rolled = _roll_history(history, transmitted)
    np.testing.assert_array_equal(rolled[:-1], history[1:])
    np.testing.assert_array_equal(rolled[-1], transmitted)

    written = _write_history(history, jnp.int32(2), transmitted)
    np.testing.assert_array_equal(written[2], transmitted)
    np.testing.assert_array_equal(written[:2], history[:2])
    np.testing.assert_array_equal(written[3:], history[3:])
