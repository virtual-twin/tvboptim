"""Uncoupled segmented execution before heterogeneous signal routing lands."""

import diffrax
import jax
import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics import (
    DynamicsGroup,
    GroupedSolution,
    HeterogeneousNetwork,
    Network,
    prepare,
    solve,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import JansenRit, Linear
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import (
    BoundedSolver,
    DiffraxSolver,
    Euler,
    Heun,
    RungeKutta4,
)


def _network(*, order="ab", routes=None, noise=False):
    groups = {
        "a": DynamicsGroup(
            Linear(gamma=-0.3),
            [0, 2, 5],
            initial_state=jnp.array([[0.2, -0.1, 0.5]]),
            noise=AdditiveNoise(sigma=0.01) if noise else None,
        ),
        "b": DynamicsGroup(
            JansenRit(mu=0.18),
            [1, 3, 4],
        ),
    }
    if order == "ba":
        groups = {"b": groups["b"], "a": groups["a"]}
    return HeterogeneousNetwork(
        graph=DenseGraph(jnp.zeros((6, 6))),
        groups=groups,
        routes=routes,
    )


@pytest.mark.parametrize("solver", [Euler(), Heun(), RungeKutta4()])
def test_grouped_trajectories_match_independent_bare_dynamics(solver):
    network = _network()
    result = solve(network, solver, t0=0.0, t1=0.3, dt=0.1)

    expected_a_fn, expected_a_cfg = prepare(
        network.groups["a"].dynamics,
        solver,
        t0=0.0,
        t1=0.3,
        dt=0.1,
        n_nodes=3,
    )
    expected_a_cfg.initial_state = network.initial_state_for("a")
    expected_a = expected_a_fn(expected_a_cfg)
    expected_b = solve(
        network.groups["b"].dynamics,
        solver,
        t0=0.0,
        t1=0.3,
        dt=0.1,
        n_nodes=3,
    )

    assert isinstance(result, GroupedSolution)
    assert jnp.allclose(result.groups.a.ys, expected_a.ys, rtol=1e-6, atol=1e-7)
    assert jnp.allclose(result.groups.b.ys, expected_b.ys, rtol=1e-6, atol=1e-7)


def test_prepare_exposes_live_named_group_parameters_and_initial_states():
    simulate, config = prepare(_network(), Heun(), t0=0.0, t1=0.4, dt=0.1)
    baseline = simulate(config)

    varied = config.copy()
    varied.groups.a.dynamics.gamma = jnp.array(-0.8)
    varied.initial_state.a = 2.0 * varied.initial_state.a
    changed = simulate(varied)

    assert config.groups.a.dynamics.gamma == -0.3
    assert not jnp.allclose(changed.ys.a, baseline.ys.a)
    assert jnp.array_equal(changed.ys.b, baseline.ys.b)


def test_grouped_solve_supports_jit_jvp_grad_and_vmap():
    simulate, config = prepare(_network(), Heun(), t0=0.0, t1=0.4, dt=0.1)

    def loss(gamma):
        varied = config.copy()
        varied.groups.a.dynamics.gamma = gamma
        result = simulate(varied)
        return jnp.square(result.ys.a).mean()

    compiled = jax.jit(jax.value_and_grad(loss))
    value, reverse = compiled(jnp.array(-0.3))
    _, forward = jax.jvp(loss, (jnp.array(-0.3),), (jnp.array(1.0),))
    batched = jax.vmap(loss)(jnp.array([-0.2, -0.3, -0.4]))

    assert jnp.isfinite(value)
    assert jnp.allclose(forward, reverse, rtol=2e-5, atol=2e-6)
    assert batched.shape == (3,)
    assert jnp.all(jnp.isfinite(batched))


def test_group_order_and_blocked_scan_do_not_change_results():
    ordinary = solve(_network(order="ab"), Heun(), t0=0.0, t1=0.6, dt=0.1)
    reordered = solve(_network(order="ba"), Heun(), t0=0.0, t1=0.6, dt=0.1)
    blocked = solve(_network(), Heun(block_size=2), t0=0.0, t1=0.6, dt=0.1)
    assert jnp.array_equal(ordinary.ys.a, reordered.ys.a)
    assert jnp.array_equal(ordinary.ys.b, reordered.ys.b)
    assert jnp.array_equal(ordinary.ys.a, blocked.ys.a)
    assert jnp.array_equal(ordinary.ys.b, blocked.ys.b)


def test_one_group_matches_zero_coupling_network():
    dynamics = Linear(gamma=-0.4)
    graph = DenseGraph(jnp.zeros((4, 4)))
    heterogeneous = HeterogeneousNetwork(
        graph=graph,
        groups={"all": DynamicsGroup(dynamics, [0, 1, 2, 3])},
    )
    homogeneous = Network(Linear(gamma=-0.4), {}, graph)
    grouped = solve(heterogeneous, Heun(), t0=0.0, t1=0.5, dt=0.1)
    flat = solve(homogeneous, Heun(), t0=0.0, t1=0.5, dt=0.1)
    assert jnp.allclose(grouped.ys["all"], flat.ys, rtol=1e-6, atol=1e-7)


def test_remaining_unimplemented_execution_features_fail_explicitly():
    with pytest.raises(NotImplementedError, match="native fixed-step"):
        prepare(_network(), DiffraxSolver(diffrax.Euler()), t1=0.2, dt=0.1)
    with pytest.raises(NotImplementedError, match="per-group bounds"):
        prepare(
            _network(),
            BoundedSolver(Euler(), low=0.0, high=1.0),
            t1=0.2,
            dt=0.1,
        )
