"""Instantaneous shared-graph SignalRoute execution contracts."""

import jax
import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics import (
    Bunch,
    DenseGraph,
    DynamicsGroup,
    HeterogeneousNetwork,
    SignalRoute,
    SparseGraph,
    prepare,
)
from tvboptim.experimental.network_dynamics.coupling import (
    DifferenceCoupling,
    LinearCoupling,
)
from tvboptim.experimental.network_dynamics.coupling.base import (
    InstantaneousCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.solvers import Euler, Heun

A_NODES = jnp.array([0, 2, 5])
B_NODES = jnp.array([1, 3, 4])
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
A0 = jnp.array([[0.2, -0.3, 0.7]])
B0 = jnp.array([[0.1, 0.8, -0.4], [0.5, -0.2, 0.3]])


class OneState(AbstractDynamics):
    STATE_NAMES = ("x",)
    INITIAL_STATE = (0.0,)
    DEFAULT_PARAMS = Bunch(alpha=-0.4)
    COUPLING_INPUTS = {"drive": 1}

    def dynamics(self, t, state, params, coupling, external):
        del t, external
        return params.alpha * state + coupling.drive


class TwoState(AbstractDynamics):
    STATE_NAMES = ("u", "v")
    INITIAL_STATE = (0.0, 0.0)
    DEFAULT_PARAMS = Bunch(beta=0.3, gamma=0.2)
    COUPLING_INPUTS = {"drive": 1}

    def dynamics(self, t, state, params, coupling, external):
        del t, external
        return jnp.stack(
            [
                params.beta * state[0] + 0.7 * coupling.drive[0],
                -params.gamma * state[1] + 0.2 * coupling.drive[0],
            ]
        )


def derived_b(state, params):
    return state[0:1] - params.scale * state[1:2]


def _network(graph_type=DenseGraph):
    return HeterogeneousNetwork(
        graph=graph_type(WEIGHTS),
        groups={
            "a": DynamicsGroup(OneState(), A_NODES, initial_state=A0),
            "b": DynamicsGroup(TwoState(), B_NODES, initial_state=B0),
        },
        routes={
            "activity": SignalRoute(
                source={"a": "x", "b": derived_b},
                source_params={"b": Bunch(scale=0.25)},
                coupling=LinearCoupling(G=0.6),
                target={"a": "drive", "b": "drive"},
            )
        },
    )


def _route(state, gain, scale, weights=WEIGHTS):
    signal = jnp.zeros((1, weights.shape[0]))
    signal = signal.at[:, A_NODES].set(state["a"])
    signal = signal.at[:, B_NODES].set(state["b"][0:1] - scale * state["b"][1:2])
    return gain * signal @ weights.T


def _field(state, alpha, beta, gamma, gain, scale, weights=WEIGHTS, frozen=None):
    routed = _route(state, gain, scale, weights) if frozen is None else frozen
    return {
        "a": alpha * state["a"] + routed[:, A_NODES],
        "b": jnp.stack(
            [
                beta * state["b"][0] + 0.7 * routed[0, B_NODES],
                -gamma * state["b"][1] + 0.2 * routed[0, B_NODES],
            ]
        ),
    }


@pytest.mark.parametrize("graph_type", [DenseGraph, SparseGraph])
def test_interleaved_route_matches_one_transport_euler_oracle(graph_type):
    solve_fn, config = prepare(_network(graph_type), Euler(), t1=0.1, dt=0.1)
    result = solve_fn(config)

    state = {"a": A0, "b": B0}
    derivative = _field(state, -0.4, 0.3, 0.2, 0.6, 0.25)
    expected = jax.tree.map(lambda y, dy: y + 0.1 * dy, state, derivative)
    assert jnp.allclose(result.ys.a[0], expected["a"])
    assert jnp.allclose(result.ys.b[0], expected["b"])


@pytest.mark.parametrize("recompute", [False, True])
def test_heun_route_policy_matches_explicit_oracle(recompute):
    solver = Heun(recompute_coupling_per_stage=recompute)
    solve_fn, config = prepare(_network(), solver, t1=0.1, dt=0.1)
    result = solve_fn(config)

    state = {"a": A0, "b": B0}
    frozen = None if recompute else _route(state, 0.6, 0.25)
    k1 = _field(state, -0.4, 0.3, 0.2, 0.6, 0.25, frozen=frozen)
    predictor = jax.tree.map(lambda y, dy: y + 0.1 * dy, state, k1)
    k2 = _field(predictor, -0.4, 0.3, 0.2, 0.6, 0.25, frozen=frozen)
    expected = jax.tree.map(lambda y, d1, d2: y + 0.05 * (d1 + d2), state, k1, k2)
    assert jnp.allclose(result.ys.a[0], expected["a"])
    assert jnp.allclose(result.ys.b[0], expected["b"])


def test_route_parameters_graph_and_initial_state_remain_live_under_jit_and_ad():
    solve_fn, config = prepare(_network(), Heun(), t1=0.2, dt=0.1)

    def loss(cfg):
        result = solve_fn(cfg)
        return sum(jnp.square(values).sum() for values in result.ys.values())

    value, gradient = jax.jit(jax.value_and_grad(loss))(config)
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient.routes.activity.coupling.G)
    assert jnp.isfinite(gradient.routes.activity.source_params.b.scale)
    assert jnp.isfinite(gradient.groups.a.dynamics.alpha)
    assert jnp.all(jnp.isfinite(gradient.graph.weights))
    assert jnp.all(jnp.isfinite(gradient.initial_state.b))


class TwoChannelPre(InstantaneousCoupling):
    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(slope=1.3)

    def pre(self, incoming_states, local_states, params):
        del local_states
        return jax.nn.sigmoid(
            params.slope * (incoming_states[0:1] - incoming_states[1:2])
        )

    def post(self, summed_inputs, local_states, params):
        del local_states, params
        return summed_inputs


def test_multichannel_pre_is_applied_before_transport():
    network = HeterogeneousNetwork(
        graph=DenseGraph(WEIGHTS),
        groups={
            "a": DynamicsGroup(TwoState(), A_NODES, initial_state=B0),
            "b": DynamicsGroup(TwoState(), B_NODES, initial_state=B0),
        },
        routes={
            "two": SignalRoute(
                source={"a": ("u", "v"), "b": ("u", "v")},
                coupling=TwoChannelPre(slope=1.3),
                target={"a": "drive", "b": "drive"},
            )
        },
    )
    solve_fn, config = prepare(network, Euler(), t1=0.1, dt=0.1)
    result = solve_fn(config)

    packed = jnp.zeros((2, 6))
    packed = packed.at[:, A_NODES].set(B0)
    packed = packed.at[:, B_NODES].set(B0)
    routed = jax.nn.sigmoid(1.3 * (packed[0:1] - packed[1:2])) @ WEIGHTS.T
    expected_u = B0[0] + 0.1 * (0.3 * B0[0] + 0.7 * routed[0, A_NODES])
    assert jnp.allclose(result.ys.a[0, 0], expected_u)


def test_omitted_source_groups_emit_no_messages_for_nonlinear_pre():
    network = HeterogeneousNetwork(
        graph=DenseGraph(WEIGHTS),
        groups={
            "a": DynamicsGroup(TwoState(), A_NODES, initial_state=B0),
            "b": DynamicsGroup(TwoState(), B_NODES, initial_state=B0),
        },
        routes={
            "subset": SignalRoute(
                source={"a": ("u", "v")},
                coupling=TwoChannelPre(slope=1.3),
                target={"a": "drive", "b": "drive"},
            )
        },
    )
    solve_fn, config = prepare(network, Euler(), t1=0.1, dt=0.1)
    result = solve_fn(config)

    messages = jnp.zeros((1, 6))
    messages = messages.at[:, A_NODES].set(jax.nn.sigmoid(1.3 * (B0[0:1] - B0[1:2])))
    routed = messages @ WEIGHTS.T
    expected_b_u = B0[0] + 0.1 * (0.3 * B0[0] + 0.7 * routed[0, B_NODES])
    assert jnp.allclose(result.ys.b[0, 0], expected_b_u)


@pytest.mark.parametrize("graph_type", [DenseGraph, SparseGraph])
def test_local_paired_route_uses_target_values_before_reduction(graph_type):
    state_a = A0
    state_b = B0[0:1]
    network = HeterogeneousNetwork(
        graph=graph_type(WEIGHTS),
        groups={
            "a": DynamicsGroup(OneState(alpha=0.0), A_NODES, initial_state=state_a),
            "b": DynamicsGroup(OneState(alpha=0.0), B_NODES, initial_state=state_b),
        },
        routes={
            "difference": SignalRoute(
                source={"a": "x", "b": "x"},
                local={"a": "x", "b": "x"},
                coupling=DifferenceCoupling(G=0.4),
                target={"a": "drive", "b": "drive"},
            )
        },
    )
    solve_fn, config = prepare(network, Euler(), t1=0.1, dt=0.1)
    result = solve_fn(config)

    packed = jnp.zeros((1, 6))
    packed = packed.at[:, A_NODES].set(state_a)
    packed = packed.at[:, B_NODES].set(state_b)
    expected_route = 0.4 * jnp.sum(
        (packed[:, None, :] - packed[:, :, None]) * WEIGHTS[None, :, :],
        axis=-1,
    )
    assert jnp.allclose(result.ys.a[0], state_a + 0.1 * expected_route[:, A_NODES])
    assert jnp.allclose(result.ys.b[0], state_b + 0.1 * expected_route[:, B_NODES])


def scale_conversion(signal, params):
    return params.gain * signal


def test_routes_accumulate_after_live_target_conversion():
    network = HeterogeneousNetwork(
        graph=DenseGraph(WEIGHTS),
        groups={
            "a": DynamicsGroup(OneState(alpha=0.0), A_NODES, initial_state=A0),
            "b": DynamicsGroup(OneState(alpha=0.0), B_NODES, initial_state=B0[0:1]),
        },
        routes={
            "converted": SignalRoute(
                source={"a": "x", "b": "x"},
                coupling=LinearCoupling(G=0.2),
                target={"a": ("drive", scale_conversion)},
                target_params={"a": Bunch(gain=1.5)},
            ),
            "direct": SignalRoute(
                source={"a": "x", "b": "x"},
                coupling=LinearCoupling(G=0.3),
                target={"a": "drive"},
            ),
        },
    )
    solve_fn, config = prepare(network, Euler(), t1=0.1, dt=0.1)
    result = solve_fn(config)

    packed = jnp.zeros((1, 6))
    packed = packed.at[:, A_NODES].set(A0)
    packed = packed.at[:, B_NODES].set(B0[0:1])
    expected_drive = (1.5 * 0.2 + 0.3) * packed @ WEIGHTS.T
    assert jnp.allclose(result.ys.a[0], A0 + 0.1 * expected_drive[:, A_NODES])
    assert jnp.array_equal(result.ys.b[0], B0[0:1])

    def loss(gain):
        changed = config.copy()
        changed.routes.converted.target_params.a.gain = gain
        return jnp.square(solve_fn(changed).ys.a).sum()

    assert jnp.isfinite(jax.jit(jax.grad(loss))(jnp.array(1.5)))


def test_readout_missing_params_error_points_at_source_params():
    # derived_b reads params.scale, but source_params is omitted, so the group's
    # params bunch is empty and the readout probe fails at prepare().
    network = HeterogeneousNetwork(
        graph=DenseGraph(WEIGHTS),
        groups={
            "a": DynamicsGroup(OneState(), A_NODES, initial_state=A0),
            "b": DynamicsGroup(TwoState(), B_NODES, initial_state=B0),
        },
        routes={
            "activity": SignalRoute(
                source={"a": "x", "b": derived_b},
                coupling=LinearCoupling(G=0.6),
                target={"a": "drive", "b": "drive"},
            )
        },
    )
    with pytest.raises(ValueError, match=r"source_params\['b'\] are empty"):
        prepare(network, Euler(), t1=0.1, dt=0.1)


def test_conversion_missing_params_error_points_at_target_params():
    # scale_conversion reads params.gain, but target_params is omitted.
    network = HeterogeneousNetwork(
        graph=DenseGraph(WEIGHTS),
        groups={
            "a": DynamicsGroup(OneState(alpha=0.0), A_NODES, initial_state=A0),
            "b": DynamicsGroup(OneState(alpha=0.0), B_NODES, initial_state=B0[0:1]),
        },
        routes={
            "converted": SignalRoute(
                source={"a": "x", "b": "x"},
                coupling=LinearCoupling(G=0.2),
                target={"a": ("drive", scale_conversion)},
            )
        },
    )
    with pytest.raises(ValueError, match=r"target_params\['a'\] are empty"):
        prepare(network, Euler(), t1=0.1, dt=0.1)


def scaled_local(state, params):
    return params.scale * state[0:1]


def test_local_params_feed_receive_only_callable_local_readout():
    # "b" is a target (so DifferenceCoupling requires a local readout for it) but
    # not a source. Its local readout is a callable needing params. Before
    # local_params existed those params had nowhere legal to live: source_params
    # rejects "b" (not a source). local_params is their home.
    state_a = A0
    state_b = B0[0:1]
    scale = 0.5
    network = HeterogeneousNetwork(
        graph=DenseGraph(WEIGHTS),
        groups={
            "a": DynamicsGroup(OneState(alpha=0.0), A_NODES, initial_state=state_a),
            "b": DynamicsGroup(OneState(alpha=0.0), B_NODES, initial_state=state_b),
        },
        routes={
            "difference": SignalRoute(
                source={"a": "x"},  # b does not send
                local={"a": "x", "b": scaled_local},  # b still receives, needs local
                coupling=DifferenceCoupling(G=0.4),
                target={"a": "drive", "b": "drive"},
                local_params={"b": Bunch(scale=scale)},
            )
        },
    )
    solve_fn, config = prepare(network, Euler(), t1=0.1, dt=0.1)
    result = solve_fn(config)

    # Only "a" contributes to the transported signal; source_mask zeroes b.
    source = jnp.zeros((1, 6)).at[:, A_NODES].set(state_a)
    mask = jnp.zeros((6,)).at[A_NODES].set(1.0)
    local = (
        jnp.zeros((1, 6)).at[:, A_NODES].set(state_a).at[:, B_NODES].set(scale * state_b)
    )
    difference = (source[:, None, :] - local[:, :, None]) * mask[None, None, :]
    expected_route = 0.4 * jnp.sum(difference * WEIGHTS[None, :, :], axis=-1)
    assert jnp.allclose(result.ys.a[0], state_a + 0.1 * expected_route[:, A_NODES])
    assert jnp.allclose(result.ys.b[0], state_b + 0.1 * expected_route[:, B_NODES])

    # local_params is a live, differentiable leaf.
    def loss(scale_value):
        changed = config.copy()
        changed.routes.difference.local_params.b.scale = scale_value
        return jnp.square(solve_fn(changed).ys.b).sum()

    assert jnp.isfinite(jax.jit(jax.grad(loss))(jnp.array(scale)))


def test_receive_only_local_params_rejected_from_source_params():
    # A target-only group's local params belong in local_params; source_params
    # rejects it because it is not a source.
    common = dict(
        source={"a": "x"},
        local={"a": "x", "b": scaled_local},
        coupling=DifferenceCoupling(G=0.4),
        target={"a": "drive", "b": "drive"},
    )
    with pytest.raises(ValueError, match=r"source=\['b'\]"):
        SignalRoute(**common, source_params={"b": Bunch(scale=0.5)})

    # The same params are accepted under local_params.
    route = SignalRoute(**common, local_params={"b": Bunch(scale=0.5)})
    assert route.local_params == {"b": Bunch(scale=0.5)}
