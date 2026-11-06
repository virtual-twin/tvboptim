"""Coupling output channels: one per source state, and checked against inputs."""

import copy
import pickle

import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics import (
    Bunch,
    HeterogeneousNetwork,
    Network,
    NodeGroup,
    SignalRoute,
    prepare,
)
from tvboptim.experimental.network_dynamics.coupling import (
    PER_SOURCE,
    DelayedDifferenceCoupling,
    DelayedKuramotoCoupling,
    DelayedLinearCoupling,
    DelayedSumLinearCoupling,
    DifferenceCoupling,
    KuramotoCoupling,
    LinearCoupling,
    SumLinearCoupling,
)
from tvboptim.experimental.network_dynamics.coupling.linear import (
    DelayedSigmoidCoupling,
    SigmoidCoupling,
    TanhCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun

N_NODES = 4
WEIGHTS = jnp.array(
    [
        [0.0, 0.4, 0.1, 0.0],
        [0.3, 0.0, 0.0, 0.5],
        [0.2, 0.6, 0.0, 0.1],
        [0.0, 0.2, 0.7, 0.0],
    ]
)
DELAYS = jnp.array(
    [
        [0.0, 0.3, 0.2, 0.0],
        [0.3, 0.0, 0.0, 0.4],
        [0.2, 0.1, 0.0, 0.3],
        [0.0, 0.4, 0.3, 0.0],
    ]
)


class Pair(AbstractDynamics):
    """Two leaky states; channel ``i`` of ``c`` drives state ``i``."""

    STATE_NAMES = ("x", "y")
    INITIAL_STATE = (0.3, -0.2)
    DEFAULT_PARAMS = Bunch(k=-0.5)
    COUPLING_INPUTS = {"c": 2}

    def dynamics(self, t, state, params, coupling, external):
        return params.k * state + jnp.tanh(coupling.c)


class PairSplit(Pair):
    """``Pair`` with one input per state."""

    COUPLING_INPUTS = {"cx": 1, "cy": 1}

    def dynamics(self, t, state, params, coupling, external):
        c = jnp.concatenate([coupling.cx, coupling.cy])
        return params.k * state + jnp.tanh(c)


class PairShared(Pair):
    """``Pair`` driven by the sum of both channels."""

    def dynamics(self, t, state, params, coupling, external):
        return params.k * state + jnp.tanh(coupling.c.sum(axis=0, keepdims=True))


class PairSummed(Pair):
    """``Pair`` with a single input, the sum over states."""

    COUPLING_INPUTS = {"c": 1}


# (coupling class, uses local states, delayed)
ELEMENTWISE = [
    (LinearCoupling, False, False),
    (DifferenceCoupling, True, False),
    (SigmoidCoupling, False, False),
    (TanhCoupling, False, False),
    (KuramotoCoupling, True, False),
    (DelayedLinearCoupling, False, True),
    (DelayedDifferenceCoupling, True, True),
    (DelayedSigmoidCoupling, False, True),
    (DelayedKuramotoCoupling, True, True),
]
ELEMENTWISE_IDS = [cls.__name__ for cls, _, _ in ELEMENTWISE]


def _graph(delayed):
    return DenseDelayGraph(WEIGHTS, DELAYS) if delayed else DenseGraph(WEIGHTS)


def _coupling(cls, uses_local, names):
    return cls(source=names, local=names) if uses_local else cls(source=names)


def _solve(network):
    solve_fn, config = prepare(network, Heun(), t1=2.0, dt=0.05)
    return solve_fn(config).ys


@pytest.mark.parametrize("cls,uses_local,delayed", ELEMENTWISE, ids=ELEMENTWISE_IDS)
def test_multi_source_equals_one_coupling_per_source(cls, uses_local, delayed):
    graph = _graph(delayed)
    joint = Network(Pair(), {"c": _coupling(cls, uses_local, ("x", "y"))}, graph)
    split = Network(
        PairSplit(),
        {
            "cx": _coupling(cls, uses_local, "x"),
            "cy": _coupling(cls, uses_local, "y"),
        },
        graph,
    )
    assert cls.N_OUTPUT_STATES is PER_SOURCE
    assert joint.coupling["c"].n_output_channels(2) == 2
    assert jnp.allclose(_solve(joint), _solve(split), atol=1e-6)


@pytest.mark.parametrize(
    "sum_cls,linear_cls,delayed",
    [
        (SumLinearCoupling, LinearCoupling, False),
        (DelayedSumLinearCoupling, DelayedLinearCoupling, True),
    ],
    ids=["instant", "delayed"],
)
def test_sum_linear_equals_summed_linear_channels(sum_cls, linear_cls, delayed):
    graph = _graph(delayed)
    summed = Network(
        PairSummed(), {"c": sum_cls(source=("x", "y"), G=0.8, b=0.1)}, graph
    )
    channels = Network(
        PairShared(), {"c": linear_cls(source=("x", "y"), G=0.8, b=0.05)}, graph
    )
    assert jnp.allclose(_solve(summed), _solve(channels), atol=1e-6)


class SumWithoutDeclaration(LinearCoupling):
    def pre(self, incoming_states, local_states, params):
        return incoming_states.sum(axis=0, keepdims=True)


@pytest.mark.parametrize(
    "dynamics,coupling,match",
    [
        (Pair(), LinearCoupling(source="x"), "outputs 1 channel.*expects 2"),
        (
            Pair(),
            SumLinearCoupling(source=("x", "y")),
            "fixed N_OUTPUT_STATES = 1",
        ),
        (
            PairSummed(),
            LinearCoupling(source=("x", "y")),
            "outputs 2 channel.*SumLinearCoupling",
        ),
    ],
    ids=["too-few-sources", "fixed-width", "too-many-sources"],
)
def test_network_rejects_channel_mismatch(dynamics, coupling, match):
    with pytest.raises(ValueError, match=match):
        Network(dynamics, {"c": coupling}, DenseGraph(WEIGHTS))


def test_combining_pre_must_declare_its_channels():
    network = Network(
        Pair(), {"c": SumWithoutDeclaration(source=("x", "y"))}, DenseGraph(WEIGHTS)
    )
    with pytest.raises(ValueError, match="returned 1 channels for 2 source states"):
        prepare(network, Heun(), t1=0.1, dt=0.05)


def _route_network(readout):
    return HeterogeneousNetwork(
        graph=DenseGraph(WEIGHTS),
        groups={"all": NodeGroup(Pair(), jnp.arange(N_NODES))},
        routes={
            "r": SignalRoute(
                source={"all": readout},
                coupling=LinearCoupling(),
                target={"all": "c"},
            )
        },
    )


def test_route_channels_follow_source_width():
    _solve(_route_network(("x", "y")))


@pytest.mark.parametrize(
    "readout,at_prepare",
    [("x", False), (lambda state, params: state[0:1], True)],
    ids=["names", "callable"],
)
def test_route_rejects_channel_mismatch(readout, at_prepare):
    match = "outputs 1 channels but 'all'.c expects 2"
    if at_prepare:
        network = _route_network(readout)
        with pytest.raises(ValueError, match=match):
            _solve(network)
    else:
        with pytest.raises(ValueError, match=match):
            _route_network(readout)


def test_per_source_marker_survives_copy_and_pickle():
    for clone in (copy.copy, copy.deepcopy, lambda m: pickle.loads(pickle.dumps(m))):
        assert clone(PER_SOURCE) is PER_SOURCE
