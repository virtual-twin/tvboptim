"""Executable references for shared-graph heterogeneous dynamics.

These tests deliberately do not use the future public heterogeneous API.  They
freeze the numerical contracts that implementation code must reproduce:
segmented state, graph-order signal packing, one transport per route, and the
existing frozen/per-stage coupling policies.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

A_NODES = jnp.array([0, 2, 5])
B_NODES = jnp.array([1, 3, 4])

# Directed convention: weights[target, source].  The deliberately asymmetric
# values make source/target reversal and wrong group ordering observable.
WEIGHTS = jnp.array(
    [
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.7],
        [0.3, 0.0, 0.0, 0.0, 0.4, 0.0],
        [0.0, 0.5, 0.0, 0.6, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.0, 0.3, 0.0, 0.0],
    ]
)


def _pack(a_signal, b_signal):
    """Pack equally wide group signals into the public graph-node order."""
    assert a_signal.shape[0] == b_signal.shape[0]
    packed = jnp.zeros((a_signal.shape[0], WEIGHTS.shape[0]))
    packed = packed.at[:, A_NODES].set(a_signal)
    return packed.at[:, B_NODES].set(b_signal)


def _dense_reduce(messages, weights=WEIGHTS):
    return messages @ weights.T


def _sparse_reduce(messages, weights=WEIGHTS):
    target_e, source_e = jnp.nonzero(weights)
    edge_messages = messages[:, source_e]
    weighted = edge_messages.T * weights[target_e, source_e, None]
    return jax.ops.segment_sum(weighted, target_e, num_segments=weights.shape[0]).T


def _linear_route(state, gain):
    # Group A exposes its only state; group B exposes a declared derived signal.
    signal = _pack(state["a"], state["b"][0:1] - 0.25 * state["b"][1:2])
    return gain * _dense_reduce(signal)


def _vector_field(state, params, frozen_coupling=None):
    incoming = (
        _linear_route(state, params["gain"])
        if frozen_coupling is None
        else frozen_coupling
    )
    incoming_a = incoming[:, A_NODES]
    incoming_b = incoming[:, B_NODES]
    return {
        "a": params["alpha"] * state["a"] + incoming_a,
        "b": jnp.stack(
            [
                params["beta"] * state["b"][0] + 0.7 * incoming_b[0],
                -params["gamma"] * state["b"][1] + 0.2 * incoming_b[0],
            ]
        ),
    }


def _tree_add_scaled(x, dx, scale):
    return jax.tree.map(lambda a, da: a + scale * da, x, dx)


def _heun_step(state, params, dt, recompute_coupling_per_stage):
    frozen = None
    if not recompute_coupling_per_stage:
        frozen = _linear_route(state, params["gain"])
    k1 = _vector_field(state, params, frozen)
    predictor = _tree_add_scaled(state, k1, dt)
    k2 = _vector_field(predictor, params, frozen)
    return jax.tree.map(lambda y, d1, d2: y + 0.5 * dt * (d1 + d2), state, k1, k2)


@pytest.fixture
def segmented_state():
    return {
        "a": jnp.array([[0.2, -0.3, 0.7]]),
        "b": jnp.array([[0.1, 0.8, -0.4], [0.5, -0.2, 0.3]]),
    }


@pytest.fixture
def params():
    return {
        "alpha": jnp.array(-0.4),
        "beta": jnp.array(0.3),
        "gamma": jnp.array(0.2),
        "gain": jnp.array(0.6),
    }


def test_interleaved_pack_is_in_graph_order(segmented_state):
    packed = _pack(
        segmented_state["a"],
        segmented_state["b"][0:1] - 0.25 * segmented_state["b"][1:2],
    )
    expected = jnp.array([[0.2, -0.025, -0.3, 0.85, -0.475, 0.7]])
    assert jnp.allclose(packed, expected)


def test_dense_and_sparse_transport_match_with_gradients(segmented_state, params):
    signal = _pack(segmented_state["a"], segmented_state["b"][0:1])
    assert jnp.allclose(_dense_reduce(signal), _sparse_reduce(signal))

    def loss(gain):
        return jnp.square(gain * _sparse_reduce(signal)).sum()

    tangent = jax.jvp(loss, (params["gain"],), (jnp.array(1.0),))[1]
    reverse = jax.grad(loss)(params["gain"])
    assert jnp.allclose(tangent, reverse)


def test_two_source_channels_are_transformed_before_transport(segmented_state):
    # This is the SigmoidalJansenRit shape contract: two raw source channels
    # become one nonlinear message channel before graph reduction.
    raw = _pack(
        jnp.concatenate([segmented_state["a"], -segmented_state["a"]]),
        segmented_state["b"],
    )

    def sigmoid_pre(x):
        return jax.nn.sigmoid(1.3 * (x[0:1] - x[1:2] - 0.2))

    correct = _dense_reduce(sigmoid_pre(raw))
    wrong = sigmoid_pre(_dense_reduce(raw))
    assert correct.shape == (1, WEIGHTS.shape[0])
    assert not jnp.allclose(correct, wrong)


def test_local_paired_difference_uses_target_edge_values(segmented_state):
    signal = _pack(segmented_state["a"], segmented_state["b"][0:1])
    target_e, source_e = jnp.nonzero(WEIGHTS)
    messages = signal[:, source_e] - signal[:, target_e]
    expected = jax.ops.segment_sum(
        messages.T * WEIGHTS[target_e, source_e, None],
        target_e,
        num_segments=WEIGHTS.shape[0],
    ).T

    manual = jnp.zeros_like(expected)
    for target, source in zip(target_e.tolist(), source_e.tolist()):
        manual = manual.at[:, target].add(
            WEIGHTS[target, source] * (signal[:, source] - signal[:, target])
        )
    assert jnp.allclose(expected, manual)


@pytest.mark.parametrize("recompute", [False, True])
def test_heun_stage_policy_is_jittable_and_differentiable(
    segmented_state, params, recompute
):
    step = jax.jit(partial(_heun_step, dt=0.05, recompute_coupling_per_stage=recompute))
    result = step(segmented_state, params)
    assert result["a"].shape == segmented_state["a"].shape
    assert result["b"].shape == segmented_state["b"].shape

    def loss(gain):
        varied = dict(params, gain=gain)
        out = step(segmented_state, varied)
        return sum(jnp.square(x).sum() for x in out.values())

    value, tangent = jax.jvp(loss, (params["gain"],), (jnp.array(1.0),))
    reverse = jax.grad(loss)(params["gain"])
    assert jnp.isfinite(value)
    assert jnp.allclose(tangent, reverse, rtol=2e-5, atol=2e-6)


def test_frozen_and_per_stage_policies_are_observably_different(
    segmented_state, params
):
    frozen = _heun_step(segmented_state, params, 0.2, False)
    recomputed = _heun_step(segmented_state, params, 0.2, True)
    differences = jax.tree.leaves(
        jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), frozen, recomputed)
    )
    assert max(float(x) for x in differences) > 1e-6
