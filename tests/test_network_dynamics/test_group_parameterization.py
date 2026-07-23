"""Same-model DynamicsGroups as a groupwise parameterization contract."""

import jax
import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics import (
    DenseDelayGraph,
    DenseGraph,
    DynamicsGroup,
    HeterogeneousNetwork,
    Network,
    SignalRoute,
    prepare,
)
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedLinearCoupling,
    LinearCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import Linear
from tvboptim.experimental.network_dynamics.solvers import Heun


@pytest.mark.parametrize("n_groups", [1, 2, 4, 8])
@pytest.mark.parametrize("delayed", [False, True])
def test_group_scalars_match_equivalent_node_local_parameter_and_gradient(
    n_groups, delayed
):
    n_nodes = 16
    weights = jnp.arange(n_nodes * n_nodes, dtype=jnp.float32).reshape(n_nodes, n_nodes)
    weights = 1e-5 * weights.at[jnp.diag_indices(n_nodes)].set(0.0)
    if delayed:
        delays = jnp.linspace(0.0, 0.2, n_nodes * n_nodes).reshape(n_nodes, n_nodes)
        delays = delays.at[jnp.diag_indices(n_nodes)].set(0.0)
        graph = DenseDelayGraph(weights, delays, max_delay_bound=0.2)
        coupling_name = "delayed"
        ordinary_coupling = DelayedLinearCoupling(
            incoming_states="x", G=0.2, history_interpolation="linear"
        )
        grouped_coupling = DelayedLinearCoupling(G=0.2, history_interpolation="linear")
    else:
        graph = DenseGraph(weights)
        coupling_name = "instant"
        ordinary_coupling = LinearCoupling(incoming_states="x", G=0.2)
        grouped_coupling = LinearCoupling(G=0.2)
    initial = jnp.linspace(0.01, 0.05, n_nodes)[None, :]
    owner = jnp.arange(n_nodes) % n_groups

    ordinary = Network(
        Linear(gamma=-1.0),
        {coupling_name: ordinary_coupling},
        graph,
    )
    ordinary_solve, ordinary_config = prepare(ordinary, Heun(), t1=0.3, dt=0.1)
    ordinary_config.initial_state.dynamics = initial

    groups = {}
    source = {}
    target = {}
    nodes_by_group = {}
    for index in range(n_groups):
        name = f"g{index}"
        nodes = jnp.arange(index, n_nodes, n_groups)
        nodes_by_group[name] = nodes
        groups[name] = DynamicsGroup(
            Linear(gamma=-1.0), nodes, initial_state=initial[:, nodes]
        )
        source[name] = "x"
        target[name] = coupling_name
    grouped = HeterogeneousNetwork(
        graph=graph,
        groups=groups,
        routes={
            "activity": SignalRoute(
                source=source,
                coupling=grouped_coupling,
                target=target,
            )
        },
    )
    grouped_solve, grouped_config = prepare(grouped, Heun(), t1=0.3, dt=0.1)
    if delayed:
        ordinary_config.initial_state.coupling.delayed.history = (
            grouped_config.routes.activity.history
        )

    def ordinary_trajectory(gamma):
        config = ordinary_config.copy()
        config.dynamics.gamma = gamma[owner]
        return ordinary_solve(config).ys

    def grouped_trajectory(gamma):
        config = grouped_config.copy()
        for index in range(n_groups):
            config.groups[f"g{index}"].dynamics.gamma = gamma[index]
        trajectories = grouped_solve(config).ys
        packed = jnp.zeros((3, 1, n_nodes))
        for name, nodes in nodes_by_group.items():
            packed = packed.at[:, :, nodes].set(trajectories[name])
        return packed

    gamma = jnp.linspace(-1.0, -1.3, n_groups)
    ordinary_value = ordinary_trajectory(gamma)
    grouped_value = grouped_trajectory(gamma)
    assert jnp.allclose(grouped_value, ordinary_value, rtol=1e-6, atol=1e-7)

    ordinary_gradient = jax.grad(
        lambda value: jnp.square(ordinary_trajectory(value)).sum()
    )(gamma)
    grouped_gradient = jax.grad(
        lambda value: jnp.square(grouped_trajectory(value)).sum()
    )(gamma)
    assert jnp.allclose(grouped_gradient, ordinary_gradient, rtol=1e-5, atol=1e-7)
