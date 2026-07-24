"""Construction and static validation for heterogeneous networks."""

import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics import (
    Bunch,
    DynamicsGroup,
    HeterogeneousNetwork,
    Network,
    SignalRoute,
)
from tvboptim.experimental.network_dynamics.coupling import (
    DifferenceCoupling,
    LinearCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import JansenRit, Linear
from tvboptim.experimental.network_dynamics.graph import DenseGraph


def _graph(n_nodes=6):
    return DenseGraph(jnp.eye(n_nodes))


def _groups():
    return {
        "a": DynamicsGroup(Linear(gamma=-0.2), nodes=[0, 2, 5]),
        "b": DynamicsGroup(JansenRit(), nodes=[1, 3, 4]),
    }


def test_interleaved_partition_is_normalized_and_order_is_canonical():
    network = HeterogeneousNetwork(graph=_graph(), groups=Bunch(_groups()))
    assert network.group_names == ("a", "b")
    assert network.group_nodes == {"a": (0, 2, 5), "b": (1, 3, 4)}
    assert network.initial_state_for("a").shape == (1, 3)
    assert network.initial_state_for("b").shape == (6, 3)


def test_public_representations_summarize_static_structure():
    groups = _groups()
    route = SignalRoute(
        source={"a": "x"},
        coupling=LinearCoupling(G=0.5),
        target={"a": "instant"},
    )
    network = HeterogeneousNetwork(
        graph=_graph(), groups=groups, routes={"activity": route}
    )

    assert "dynamics=Linear" in repr(groups["a"])
    assert "sources=['a']" in repr(route)
    assert "coupling=LinearCoupling" in repr(route)
    assert "n_nodes=6" in repr(network)
    assert "a:Linear" in repr(network)
    assert "routes=['activity']" in repr(network)


def test_public_mapping_arguments_fail_with_specific_errors():
    with pytest.raises(TypeError, match="external_input must be a mapping"):
        DynamicsGroup(Linear(), [0], external_input=[])
    with pytest.raises(TypeError, match=r"SignalRoute\.source must be a mapping"):
        SignalRoute(source=[], coupling=LinearCoupling(), target={})
    with pytest.raises(TypeError, match="groups must be a mapping"):
        HeterogeneousNetwork(graph=_graph(), groups=[])
    with pytest.raises(TypeError, match="routes must be a mapping"):
        HeterogeneousNetwork(graph=_graph(), groups=_groups(), routes=[])


def test_boolean_mask_and_custom_initial_state():
    initial = jnp.array([[0.2, 0.3, 0.4]])
    groups = {
        "a": DynamicsGroup(
            Linear(),
            nodes=jnp.array([True, False, True, False, False, True]),
            initial_state=initial,
        ),
        "b": DynamicsGroup(
            JansenRit(),
            nodes=jnp.array([False, True, False, True, True, False]),
        ),
    }
    network = HeterogeneousNetwork(graph=_graph(), groups=groups)
    assert network.group_nodes["a"] == (0, 2, 5)
    assert network.initial_state_for("a") is initial


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        (
            {
                "a": DynamicsGroup(Linear(), [0, 1, 2]),
                "b": DynamicsGroup(JansenRit(), [2, 3, 4, 5]),
            },
            "overlap",
        ),
        (
            {
                "a": DynamicsGroup(Linear(), [0, 1]),
                "b": DynamicsGroup(JansenRit(), [3, 4, 5]),
            },
            "missing",
        ),
        (
            {
                "a": DynamicsGroup(Linear(), [0, 0, 1]),
                "b": DynamicsGroup(JansenRit(), [2, 3, 4, 5]),
            },
            "duplicate",
        ),
        (
            {
                "a": DynamicsGroup(Linear(), [0, 1, 6]),
                "b": DynamicsGroup(JansenRit(), [2, 3, 4, 5]),
            },
            "outside",
        ),
    ],
)
def test_invalid_partitions_are_rejected(groups, match):
    with pytest.raises(ValueError, match=match):
        HeterogeneousNetwork(graph=_graph(), groups=groups)


def test_signal_route_accepts_named_and_multichannel_sources():
    route = SignalRoute(
        source={"a": "x", "b": ("y1", "y2")},
        coupling=LinearCoupling(G=0.5),
        target={"a": "instant", "b": "instant"},
    )
    # Known widths differ, so the complete network catches the mismatch.
    with pytest.raises(ValueError, match="inconsistent"):
        HeterogeneousNetwork(
            graph=_graph(), groups=_groups(), routes={"activity": route}
        )


def test_callable_source_defers_shape_but_target_names_are_validated():
    route = SignalRoute(
        source={"a": "x", "b": lambda state, params: state[1:2] - state[2:3]},
        coupling=LinearCoupling(G=0.5),
        target={"a": "instant", "b": "instant"},
    )
    network = HeterogeneousNetwork(
        graph=_graph(), groups=_groups(), routes={"activity": route}
    )
    assert network.route_names == ("activity",)


def test_local_readout_contract_is_explicit():
    with pytest.raises(ValueError, match="requires local readouts"):
        SignalRoute(
            source={"a": "x"},
            coupling=DifferenceCoupling(G=0.5),
            target={"a": "instant"},
        )

    route = SignalRoute(
        source={"a": "x"},
        local={"a": "x"},
        coupling=DifferenceCoupling(G=0.5),
        target={"a": "instant"},
    )
    assert route.local["a"] == ("x",)


def test_route_rejects_unknown_groups_states_and_inputs():
    unknown_group = SignalRoute(
        source={"missing": "x"},
        coupling=LinearCoupling(),
        target={"a": "instant"},
    )
    with pytest.raises(ValueError, match="unknown groups"):
        HeterogeneousNetwork(
            graph=_graph(), groups=_groups(), routes={"route": unknown_group}
        )

    unknown_state = SignalRoute(
        source={"a": "not_a_state"},
        coupling=LinearCoupling(),
        target={"a": "instant"},
    )
    with pytest.raises(ValueError, match="unknown states"):
        HeterogeneousNetwork(
            graph=_graph(), groups=_groups(), routes={"route": unknown_state}
        )

    unknown_input = SignalRoute(
        source={"a": "x"},
        coupling=LinearCoupling(),
        target={"a": "not_an_input"},
    )
    with pytest.raises(ValueError, match="unknown input"):
        HeterogeneousNetwork(
            graph=_graph(), groups=_groups(), routes={"route": unknown_input}
        )


def test_selector_free_coupling_is_route_only():
    with pytest.raises(ValueError, match="Selector-free"):
        Network(Linear(), {"instant": LinearCoupling(G=0.5)}, _graph())


def test_initial_state_shape_is_group_local():
    groups = _groups()
    groups["a"] = DynamicsGroup(Linear(), [0, 2, 5], initial_state=jnp.zeros((1, 6)))
    with pytest.raises(ValueError, match="initial_state"):
        HeterogeneousNetwork(graph=_graph(), groups=groups)
