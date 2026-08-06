"""Native solver arithmetic over segmented PyTree state."""

import jax
import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics.solvers import (
    Euler,
    Heun,
    RungeKutta4,
)

SOLVERS = (Euler, Heun, RungeKutta4)


def _tree_dynamics(t, state, params):
    return {
        "one": params["one"] * state["one"] + t,
        "two": params["two"] * state["two"] - 0.5 * t,
    }


def _array_dynamics(t, state, param):
    return param * state + t


@pytest.mark.parametrize("solver_type", SOLVERS)
def test_segmented_step_matches_independent_array_steps(solver_type):
    state = {
        "one": jnp.array([[0.2, -0.1, 0.7]]),
        "two": jnp.array([[0.3, 0.8], [-0.4, 0.2]]),
    }
    params = {"one": jnp.array(-0.3), "two": jnp.array(0.15)}
    noise = {
        "one": jnp.array([[0.01, -0.02, 0.03]]),
        "two": jnp.array([[0.02, 0.01], [-0.01, 0.04]]),
    }
    solver = solver_type()

    actual, _ = solver.step(_tree_dynamics, 0.4, state, 0.05, params, noise)
    expected = {}
    for name in state:
        time_factor = 1.0 if name == "one" else -0.5

        def leaf_dynamics(t, leaf_state, param):
            return param * leaf_state + time_factor * t

        expected[name], _ = solver.step(
            leaf_dynamics,
            0.4,
            state[name],
            0.05,
            params[name],
            noise[name],
        )

    assert jax.tree.structure(actual) == jax.tree.structure(state)
    for name in state:
        assert jnp.array_equal(actual[name], expected[name])


@pytest.mark.parametrize("solver_type", SOLVERS)
def test_segmented_deterministic_default_is_jittable_and_differentiable(solver_type):
    state = {
        "a": jnp.array([[0.2, -0.1]]),
        "b": jnp.array([[0.3], [0.7]]),
    }

    def loss(scale):
        params = {"one": scale, "two": -0.5 * scale}
        renamed = {"one": state["a"], "two": state["b"]}
        next_state, _ = solver_type().step(_tree_dynamics, 0.1, renamed, 0.05, params)
        return sum(jnp.square(x).sum() for x in next_state.values())

    compiled = jax.jit(jax.value_and_grad(loss))
    value, gradient = compiled(jnp.array(0.4))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


@pytest.mark.parametrize("solver_type", SOLVERS)
def test_plain_array_expression_remains_bit_exact(solver_type):
    state = jnp.array([[0.2, -0.4], [0.1, 0.7]])
    param = jnp.array(0.3)
    noise = jnp.array([[0.01, -0.02], [0.02, 0.03]])
    dt = 0.05
    t = 0.2

    actual, _ = solver_type().step(_array_dynamics, t, state, dt, param, noise)

    k1 = _array_dynamics(t, state, param)
    if solver_type is Euler:
        expected = state + dt * k1 + noise
    elif solver_type is Heun:
        predictor = state + dt * k1 + noise
        k2 = _array_dynamics(t + dt, predictor, param)
        expected = state + dt * 0.5 * (k1 + k2) + noise
    else:
        k2 = _array_dynamics(t + 0.5 * dt, state + 0.5 * dt * k1, param)
        k3 = _array_dynamics(t + 0.5 * dt, state + 0.5 * dt * k2, param)
        k4 = _array_dynamics(t + dt, state + dt * k3, param)
        expected = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) + noise

    assert jnp.array_equal(actual, expected)


def test_nonmatching_noise_tree_is_rejected():
    state = {"a": jnp.ones((1, 2)), "b": jnp.ones((2, 1))}
    params = {"one": 0.1, "two": 0.2}
    renamed = {"one": state["a"], "two": state["b"]}
    with pytest.raises(ValueError, match="noise_sample"):
        Euler().step(
            _tree_dynamics,
            0.0,
            renamed,
            0.1,
            params,
            {"one": jnp.zeros((1, 2))},
        )
