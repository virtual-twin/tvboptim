"""Delayed canonical-signal routing on heterogeneous shared graphs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from tvboptim.experimental.network_dynamics import (
    DynamicsGroup,
    HeterogeneousNetwork,
    Network,
    SignalRoute,
    prepare,
)
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedDifferenceCoupling,
    DelayedLinearCoupling,
    DelayedSigmoidalJansenRit,
    LinearCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    Generic2dOscillator,
    JansenRit,
    Linear,
)
from tvboptim.experimental.network_dynamics.external_input import ConstantInput
from tvboptim.experimental.network_dynamics.graph import (
    DenseDelayGraph,
    DenseGraph,
    SparseDelayGraph,
)
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import Euler, Heun

WEIGHTS = jnp.array(
    [
        [0.0, 0.4, 0.0, 0.2],
        [0.3, 0.0, 0.5, 0.0],
        [0.0, 0.7, 0.0, 0.6],
        [0.8, 0.0, 0.1, 0.0],
    ],
    dtype=jnp.float32,
)
DELAYS = jnp.array(
    [
        [0.0, 0.10, 0.0, 0.25],
        [0.15, 0.0, 0.30, 0.0],
        [0.0, 0.20, 0.0, 0.35],
        [0.05, 0.0, 0.25, 0.0],
    ],
    dtype=jnp.float32,
)


def _dense_graph(delays=DELAYS):
    return DenseDelayGraph(WEIGHTS, delays, max_delay_bound=0.5)


def _sparse_graph():
    indices = jnp.asarray(np.argwhere(np.asarray(WEIGHTS) != 0.0))
    weights = BCOO(
        (WEIGHTS[indices[:, 0], indices[:, 1]], indices),
        shape=WEIGHTS.shape,
        indices_sorted=True,
        unique_indices=True,
    )
    delays = BCOO(
        (DELAYS[indices[:, 0], indices[:, 1]], indices),
        shape=DELAYS.shape,
        indices_sorted=True,
        unique_indices=True,
    )
    return SparseDelayGraph(weights, delays, max_delay_bound=0.5)


def _one_group(graph, coupling):
    return HeterogeneousNetwork(
        graph=graph,
        groups={"all": DynamicsGroup(Linear(gamma=-0.4), [0, 1, 2, 3])},
        routes={
            "delayed": SignalRoute(
                source={"all": "x"},
                coupling=coupling,
                target={"all": "delayed"},
            )
        },
    )


def _split_groups(graph, coupling):
    local = (
        {"left": "x", "right": "x"}
        if isinstance(coupling, DelayedDifferenceCoupling)
        else None
    )
    return HeterogeneousNetwork(
        graph=graph,
        groups={
            "left": DynamicsGroup(Linear(gamma=-0.4), [0, 2]),
            "right": DynamicsGroup(Linear(gamma=-0.4), [1, 3]),
        },
        routes={
            "delayed": SignalRoute(
                source={"left": "x", "right": "x"},
                local=local,
                coupling=coupling,
                target={"left": "delayed", "right": "delayed"},
            )
        },
    )


@pytest.mark.parametrize("buffer_strategy", ["roll", "circular", "preallocated"])
@pytest.mark.parametrize("interpolation", [None, "linear"])
def test_one_group_delayed_route_matches_ordinary_network(
    buffer_strategy, interpolation
):
    graph = _dense_graph()
    grouped_coupling = DelayedLinearCoupling(
        G=0.3,
        buffer_strategy=buffer_strategy,
        history_interpolation=interpolation,
    )
    ordinary_coupling = DelayedLinearCoupling(
        incoming_states="x",
        G=0.3,
        buffer_strategy=buffer_strategy,
        history_interpolation=interpolation,
    )
    solver = Heun(recompute_coupling_per_stage=True)
    grouped_fn, grouped_config = prepare(
        _one_group(graph, grouped_coupling), solver, t0=0.0, t1=0.6, dt=0.1
    )
    ordinary_fn, ordinary_config = prepare(
        Network(Linear(gamma=-0.4), {"delayed": ordinary_coupling}, graph),
        solver,
        t0=0.0,
        t1=0.6,
        dt=0.1,
    )

    grouped = jax.jit(grouped_fn)(grouped_config)
    ordinary = jax.jit(ordinary_fn)(ordinary_config)
    assert jnp.array_equal(grouped.groups.all.ys, ordinary.ys)


@pytest.mark.parametrize("coupling_type", [DelayedLinearCoupling, DelayedDifferenceCoupling])
def test_split_group_dense_and_sparse_delayed_routes_match(coupling_type):
    kwargs = dict(G=0.2, history_interpolation="linear", buffer_strategy="circular")
    dense_fn, dense_config = prepare(
        _split_groups(_dense_graph(), coupling_type(**kwargs)),
        Heun(recompute_coupling_per_stage=True),
        t0=0.0,
        t1=0.6,
        dt=0.1,
    )
    sparse_fn, sparse_config = prepare(
        _split_groups(_sparse_graph(), coupling_type(**kwargs)),
        Heun(recompute_coupling_per_stage=True),
        t0=0.0,
        t1=0.6,
        dt=0.1,
    )
    dense = dense_fn(dense_config)
    sparse = sparse_fn(sparse_config)

    assert jnp.allclose(dense.groups.left.ys, sparse.groups.left.ys, atol=2e-7)
    assert jnp.allclose(dense.groups.right.ys, sparse.groups.right.ys, atol=2e-7)


def test_zero_delay_route_matches_instantaneous_route():
    graph = DenseDelayGraph(WEIGHTS, jnp.zeros_like(DELAYS))
    delayed_fn, delayed_config = prepare(
        _split_groups(graph, DelayedLinearCoupling(G=0.3)),
        Heun(),
        t0=0.0,
        t1=0.6,
        dt=0.1,
    )
    instantaneous = HeterogeneousNetwork(
        graph=DenseGraph(WEIGHTS),
        groups={
            "left": DynamicsGroup(Linear(gamma=-0.4), [0, 2]),
            "right": DynamicsGroup(Linear(gamma=-0.4), [1, 3]),
        },
        routes={
            "instant": SignalRoute(
                source={"left": "x", "right": "x"},
                coupling=LinearCoupling(G=0.3),
                target={"left": "delayed", "right": "delayed"},
            )
        },
    )
    instant_fn, instant_config = prepare(
        instantaneous, Heun(), t0=0.0, t1=0.6, dt=0.1
    )

    delayed = delayed_fn(delayed_config)
    instant = instant_fn(instant_config)
    assert jnp.array_equal(delayed.groups.left.ys, instant.groups.left.ys)
    assert jnp.array_equal(delayed.groups.right.ys, instant.groups.right.ys)


def test_live_warm_history_matches_ordinary_path_and_is_differentiable():
    graph = _dense_graph()
    grouped_fn, grouped_config = prepare(
        _one_group(
            graph,
            DelayedLinearCoupling(G=0.3, history_interpolation="linear"),
        ),
        Euler(),
        t0=0.0,
        t1=0.4,
        dt=0.1,
    )
    ordinary_fn, ordinary_config = prepare(
        Network(
            Linear(gamma=-0.4),
            {
                "delayed": DelayedLinearCoupling(
                    incoming_states="x", G=0.3, history_interpolation="linear"
                )
            },
            graph,
        ),
        Euler(),
        t0=0.0,
        t1=0.4,
        dt=0.1,
    )
    shape = grouped_config.routes.delayed.history.shape
    warm_history = jnp.arange(
        np.prod(shape), dtype=grouped_config.routes.delayed.history.dtype
    ).reshape(shape) / 20
    grouped_config.routes.delayed.history = warm_history
    ordinary_config.initial_state.coupling.delayed.history = warm_history

    grouped = grouped_fn(grouped_config)
    ordinary = ordinary_fn(ordinary_config)
    assert jnp.array_equal(grouped.groups.all.ys, ordinary.ys)

    def history_loss(history):
        varied = eqx.tree_at(
            lambda config: config.routes.delayed.history,
            grouped_config,
            history,
        )
        return jnp.square(grouped_fn(varied).groups.all.ys).sum()

    gradient = jax.jit(jax.grad(history_loss))(warm_history)
    assert gradient.shape == warm_history.shape
    assert jnp.any(gradient != 0.0)


def test_update_history_continues_a_grouped_delayed_simulation():
    network = _split_groups(
        _dense_graph(),
        DelayedLinearCoupling(
            G=0.2, history_interpolation="linear", buffer_strategy="circular"
        ),
    )
    first_fn, first_config = prepare(
        network, Euler(), t0=0.0, t1=0.7, dt=0.1
    )
    first = first_fn(first_config)

    network.update_history(first)
    continuation_fn, continuation_config = prepare(
        network, Euler(), t0=0.7, t1=1.2, dt=0.1
    )
    continuation = continuation_fn(continuation_config)

    reference_fn, reference_config = prepare(
        _split_groups(
            _dense_graph(),
            DelayedLinearCoupling(
                G=0.2,
                history_interpolation="linear",
                buffer_strategy="circular",
            ),
        ),
        Euler(),
        t0=0.0,
        t1=1.2,
        dt=0.1,
    )
    reference = reference_fn(reference_config)

    assert jnp.allclose(continuation.ts, reference.ts[-5:], atol=1e-7)
    assert jnp.allclose(
        continuation.groups.left.ys, reference.groups.left.ys[-5:], atol=1e-8
    )
    assert jnp.allclose(
        continuation.groups.right.ys, reference.groups.right.ys[-5:], atol=1e-8
    )


def test_warm_start_recomputes_callable_route_history():
    def scaled_readout(state, params):
        return params["scale"] * state[0:1]

    def make_network():
        return HeterogeneousNetwork(
            graph=_dense_graph(),
            groups={
                "left": DynamicsGroup(Linear(gamma=-0.4), [0, 2]),
                "right": DynamicsGroup(Linear(gamma=-0.4), [1, 3]),
            },
            routes={
                "delayed": SignalRoute(
                    source={"left": scaled_readout, "right": scaled_readout},
                    source_params={
                        "left": {"scale": 0.7},
                        "right": {"scale": 0.7},
                    },
                    coupling=DelayedLinearCoupling(
                        G=0.2, history_interpolation="linear"
                    ),
                    target={"left": "delayed", "right": "delayed"},
                )
            },
        )

    network = make_network()
    first_fn, first_config = prepare(
        network, Euler(), t0=0.0, t1=0.75, dt=0.125
    )
    first = first_fn(first_config)
    network.update_history(first)
    continuation_fn, continuation_config = prepare(
        network, Euler(), t0=0.75, t1=1.25, dt=0.125
    )
    continuation = continuation_fn(continuation_config)

    reference_fn, reference_config = prepare(
        make_network(), Euler(), t0=0.0, t1=1.25, dt=0.125
    )
    reference = reference_fn(reference_config)
    assert jnp.allclose(
        continuation.groups.left.ys, reference.groups.left.ys[-4:], atol=1e-8
    )
    assert jnp.allclose(
        continuation.groups.right.ys, reference.groups.right.ys[-4:], atol=1e-8
    )


def test_fractional_delay_gradients_match_dense_and_sparse():
    kwargs = dict(G=0.2, history_interpolation="linear", buffer_strategy="circular")
    dense_fn, dense_config = prepare(
        _split_groups(_dense_graph(), DelayedLinearCoupling(**kwargs)),
        Euler(),
        t0=0.0,
        t1=0.5,
        dt=0.1,
    )
    sparse_fn, sparse_config = prepare(
        _split_groups(_sparse_graph(), DelayedLinearCoupling(**kwargs)),
        Euler(),
        t0=0.0,
        t1=0.5,
        dt=0.1,
    )

    def dense_loss(delays):
        varied = eqx.tree_at(lambda config: config.graph.delays, dense_config, delays)
        result = dense_fn(varied)
        return sum(jnp.square(group.ys).sum() for group in result.groups.values())

    def sparse_loss(delays):
        varied = eqx.tree_at(
            lambda config: config.graph.delays.data, sparse_config, delays
        )
        result = sparse_fn(varied)
        return sum(jnp.square(group.ys).sum() for group in result.groups.values())

    dense_grad = jax.jit(jax.grad(dense_loss))(dense_config.graph.delays)
    sparse_grad = jax.jit(jax.grad(sparse_loss))(sparse_config.graph.delays.data)
    indices = sparse_config.graph.delays.indices
    assert jnp.allclose(
        sparse_grad,
        dense_grad[indices[:, 0], indices[:, 1]],
        rtol=2e-5,
        atol=2e-7,
    )
    assert jnp.any(sparse_grad != 0.0)


def test_delayed_route_supports_checkpointing_vmap_and_reverse_mode():
    network = _split_groups(
        _dense_graph(),
        DelayedLinearCoupling(
            G=0.2, history_interpolation="linear", buffer_strategy="circular"
        ),
    )
    plain_fn, plain_config = prepare(
        network, Heun(), t0=0.0, t1=0.6, dt=0.1
    )
    blocked_fn, blocked_config = prepare(
        network,
        Heun(block_size=2, grad_horizon=4),
        t0=0.0,
        t1=0.6,
        dt=0.1,
    )
    plain = plain_fn(plain_config)
    blocked = blocked_fn(blocked_config)
    assert jnp.array_equal(plain.groups.left.ys, blocked.groups.left.ys)
    assert jnp.array_equal(plain.groups.right.ys, blocked.groups.right.ys)

    def loss(gain):
        varied = plain_config.copy()
        varied.routes.delayed.coupling.G = gain
        result = plain_fn(varied)
        return sum(jnp.square(group.ys).sum() for group in result.groups.values())

    value, reverse = jax.jit(jax.value_and_grad(loss))(jnp.array(0.2))
    _, forward = jax.jvp(loss, (jnp.array(0.2),), (jnp.array(1.0),))
    assert jnp.isfinite(value)
    assert jnp.allclose(forward, reverse, rtol=2e-5, atol=2e-7)

    def with_gain(gain, base):
        return eqx.tree_at(
            lambda config: config.routes.delayed.coupling.G, base, gain
        )

    configs = jax.vmap(with_gain, in_axes=(0, None))(
        jnp.array([0.1, 0.2, 0.3]), plain_config
    )
    batched = jax.jit(jax.vmap(plain_fn))(configs)
    assert batched.groups.left.ys.shape == (3, 6, 1, 2)


def test_delays_noise_and_external_inputs_share_one_grouped_scan_carry():
    network = HeterogeneousNetwork(
        graph=_dense_graph(),
        groups={
            "left": DynamicsGroup(
                Generic2dOscillator(),
                [0, 2],
                noise=AdditiveNoise(
                    sigma=0.01, apply_to="V", key=jax.random.key(3)
                ),
                external_input={"stimulus": ConstantInput(amplitude=0.2)},
            ),
            "right": DynamicsGroup(Generic2dOscillator(), [1, 3]),
        },
        routes={
            "delayed": SignalRoute(
                source={"left": "V", "right": "V"},
                coupling=DelayedLinearCoupling(
                    G=0.2,
                    history_interpolation="linear",
                    buffer_strategy="circular",
                ),
                target={"left": "delayed", "right": "delayed"},
            )
        },
    )
    simulate, config = prepare(
        network, Heun(block_size=2), t0=0.0, t1=0.6, dt=0.1
    )
    first = jax.jit(simulate)(config)
    replay = jax.jit(simulate)(config)
    assert jnp.array_equal(first.groups.left.ys, replay.groups.left.ys)
    assert jnp.array_equal(first.groups.right.ys, replay.groups.right.ys)
    assert jnp.all(jnp.isfinite(first.groups.left.ys))


def test_multichannel_jansen_rit_route_history_stores_only_source_channels():
    weights = jnp.array([[0.0, 0.2], [0.3, 0.0]])
    delays = jnp.array([[0.0, 0.1], [0.2, 0.0]])
    graph = DenseDelayGraph(weights, delays)
    grouped_fn, grouped_config = prepare(
        HeterogeneousNetwork(
            graph=graph,
            groups={"jr": DynamicsGroup(JansenRit(), [0, 1])},
            routes={
                "delayed": SignalRoute(
                    source={"jr": ("y1", "y2")},
                    coupling=DelayedSigmoidalJansenRit(G=0.4),
                    target={"jr": "delayed"},
                )
            },
        ),
        Euler(),
        t0=0.0,
        t1=0.3,
        dt=0.1,
    )
    ordinary_fn, ordinary_config = prepare(
        Network(
            JansenRit(),
            {
                "delayed": DelayedSigmoidalJansenRit(
                    incoming_states=("y1", "y2"), G=0.4
                )
            },
            graph,
        ),
        Euler(),
        t0=0.0,
        t1=0.3,
        dt=0.1,
    )

    assert grouped_config.routes.delayed.history.shape[1:] == (2, 2)
    grouped = grouped_fn(grouped_config)
    ordinary = ordinary_fn(ordinary_config)
    assert jnp.allclose(grouped.groups.jr.ys, ordinary.ys, rtol=1e-6, atol=1e-8)


def test_delayed_route_rejects_graph_without_delays():
    with pytest.raises(ValueError, match="requires a graph with delays"):
        prepare(
            _one_group(DenseGraph(WEIGHTS), DelayedLinearCoupling(G=0.2)),
            Euler(),
            t1=0.2,
            dt=0.1,
        )


def test_direct_warm_history_validates_route_shape():
    simulate, config = prepare(
        _one_group(_dense_graph(), DelayedLinearCoupling(G=0.2)),
        Euler(),
        t1=0.2,
        dt=0.1,
    )
    config.routes.delayed.history = config.routes.delayed.history[:-1]
    with pytest.raises(ValueError, match="history for route 'delayed'.*expected"):
        simulate(config)
