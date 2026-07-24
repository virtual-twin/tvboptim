"""Input-dependent source readouts: ``readout(state, params, inputs)``.

A source readout may build its transmitted signal from what the node *receives*
(its coupling inputs), not only from its state -- the relay / short-term-plasticity
case. ``inputs`` is the group's coupling-input ``Bunch``, populated by routes
declared earlier.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tvboptim.experimental.network_dynamics import (
    DynamicsGroup,
    HeterogeneousNetwork,
    SignalRoute,
    prepare,
)
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedLinearCoupling,
    LinearCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun


class Const(AbstractDynamics):
    """Holds its initial value: a steady source (no coupling inputs)."""

    STATE_NAMES = ("y",)
    INITIAL_STATE = (1.0,)
    DEFAULT_PARAMS = Bunch()
    COUPLING_INPUTS = {}

    def dynamics(self, t, state, params, coupling, external):
        return jnp.zeros_like(state)


class Node(AbstractDynamics):
    """Leaky node that relaxes toward its incoming rate ``r_in``."""

    STATE_NAMES = ("v",)
    INITIAL_STATE = (0.0,)
    DEFAULT_PARAMS = Bunch()
    COUPLING_INPUTS = {"r_in": 1}

    def dynamics(self, t, state, params, coupling, external):
        return jnp.array([-state[0] + coupling.r_in[0]])


def _emit_r_in_times_v(state, params, inputs):
    """Transmitted signal = r_in * v -- depends on the node's incoming coupling."""
    return (inputs["r_in"][0] * state[0])[None]


def _graph():
    # a(0) -> relay(1) -> sink(2); weights[target, source]
    return DenseGraph(weights=jnp.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]))


def _groups():
    return {
        "a": DynamicsGroup(Const(), nodes=jnp.array([0])),
        "relay": DynamicsGroup(Node(), nodes=jnp.array([1])),
        "sink": DynamicsGroup(Node(), nodes=jnp.array([2])),
    }


def test_readout_reads_incoming_coupling():
    # relay.r_in settles to 1 (from a); relay.v settles to 1; it emits r_in*v -> 1.
    # If the readout could not see r_in, r_in*v would be 0*v = 0 and sink stays 0.
    net = HeterogeneousNetwork(
        graph=_graph(),
        groups=_groups(),
        routes={
            "drive": SignalRoute(source={"a": "y"}, coupling=LinearCoupling(), target={"relay": "r_in"}),
            "transmit": SignalRoute(source={"relay": _emit_r_in_times_v}, coupling=LinearCoupling(), target={"sink": "r_in"}),
        },
    )
    simulate, config = prepare(net, Heun(block_size=100), t0=0.0, t1=40.0, dt=0.1)
    sink = np.asarray(simulate(config).groups["sink"].sel("v")).squeeze()
    assert np.all(np.isfinite(sink))
    assert sink[-1] == pytest.approx(1.0, abs=1e-2)  # r_in(1) * v(1) transmitted


def test_two_arg_readout_still_dispatched_by_arity():
    net = HeterogeneousNetwork(
        graph=_graph(),
        groups=_groups(),
        routes={
            "drive": SignalRoute(
                source={"a": lambda state, params: state[0:1]},  # plain 2-arg readout
                coupling=LinearCoupling(),
                target={"relay": "r_in"},
            ),
        },
    )
    simulate, config = prepare(net, Heun(block_size=100), t0=0.0, t1=5.0, dt=0.1)
    assert np.all(np.isfinite(np.asarray(simulate(config).groups["relay"].sel("v"))))


def test_consumer_declared_before_producer_raises():
    # "transmit" reads relay.r_in but "drive" (which produces it) is declared after.
    net = HeterogeneousNetwork(
        graph=_graph(),
        groups=_groups(),
        routes={
            "transmit": SignalRoute(source={"relay": _emit_r_in_times_v}, coupling=LinearCoupling(), target={"sink": "r_in"}),
            "drive": SignalRoute(source={"a": "y"}, coupling=LinearCoupling(), target={"relay": "r_in"}),
        },
    )
    with pytest.raises(ValueError, match="no earlier route feeds"):
        prepare(net, Heun(block_size=100), t0=0.0, t1=1.0, dt=0.1)


def test_input_dependent_readout_on_delayed_route_raises():
    delays = jnp.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    net = HeterogeneousNetwork(
        graph=DenseDelayGraph(
            weights=jnp.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            delays=delays,
            max_delay_bound=2.0,
        ),
        groups=_groups(),
        routes={
            "drive": SignalRoute(source={"a": "y"}, coupling=LinearCoupling(), target={"relay": "r_in"}),
            "transmit": SignalRoute(source={"relay": _emit_r_in_times_v}, coupling=DelayedLinearCoupling(), target={"sink": "r_in"}),
        },
    )
    with pytest.raises(NotImplementedError, match="delayed"):
        prepare(net, Heun(block_size=100), t0=0.0, t1=1.0, dt=0.1)
