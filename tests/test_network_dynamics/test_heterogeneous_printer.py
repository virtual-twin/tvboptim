import jax.numpy as jnp

from tvboptim.experimental.network_dynamics import (
    Bunch,
    DenseGraph,
    HeterogeneousNetwork,
    Network,
    NodeGroup,
    SignalRoute,
)
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.utils import format_network


class PrinterDynamics(AbstractDynamics):
    STATE_NAMES = ("x",)
    INITIAL_STATE = (0.0,)
    DEFAULT_PARAMS = Bunch(decay=0.1)
    COUPLING_INPUTS = {"drive": 1}

    def dynamics(self, t, state, params, coupling, external):
        del t, external
        return -params.decay * state + coupling.drive


def test_print_network_renders_heterogeneous_network():
    graph = DenseGraph(jnp.array([[0.0, 1.0], [1.0, 0.0]]))
    network = HeterogeneousNetwork(
        graph=graph,
        groups={
            "left": NodeGroup(PrinterDynamics(), [0]),
            "right": NodeGroup(PrinterDynamics(), [1]),
        },
        routes={
            "fast": SignalRoute(
                source={"left": "x", "right": "x"},
                coupling=LinearCoupling(G=0.2),
                target={"left": "drive", "right": "drive"},
            )
        },
    )

    rendered = format_network(network)
    assert "left (PrinterDynamics)" in rendered
    assert "right (PrinterDynamics)" in rendered
    assert "fast (LinearCoupling, instantaneous)" in rendered
    assert "Source: left=x, right=x" in rendered
    assert "Target: left=drive, right=drive" in rendered


def test_print_network_still_renders_homogeneous_network():
    graph = DenseGraph(jnp.array([[0.0, 1.0], [1.0, 0.0]]))
    network = Network(PrinterDynamics(), LinearCoupling(source="x"), graph)
    rendered = format_network(network)
    assert "Network Dynamics Network System" in rendered
    assert "PrinterDynamics" in rendered
