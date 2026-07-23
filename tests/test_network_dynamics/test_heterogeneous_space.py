"""Space and ParallelExecution contracts for heterogeneous configurations."""

import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.execution import ParallelExecution
from tvboptim.experimental.network_dynamics import (
    Bunch,
    DenseDelayGraph,
    DynamicsGroup,
    HeterogeneousNetwork,
    SignalRoute,
    prepare,
)
from tvboptim.experimental.network_dynamics.coupling import DelayedLinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.external_input import ConstantInput
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import Euler
from tvboptim.types import DataAxis, Space


class DrivenLinear(AbstractDynamics):
    STATE_NAMES = ("x",)
    INITIAL_STATE = (0.1,)
    DEFAULT_PARAMS = Bunch(decay=0.2)
    COUPLING_INPUTS = {"drive": 1}
    EXTERNAL_INPUTS = {"stimulus": 1}

    def dynamics(self, t, state, params, coupling, external):
        del t
        return -params.decay * state + coupling.drive + external.stimulus


WEIGHTS = jnp.array(
    [
        [0.0, 0.3, 0.0, 0.1],
        [0.2, 0.0, 0.4, 0.0],
        [0.0, 0.5, 0.0, 0.2],
        [0.1, 0.0, 0.3, 0.0],
    ]
)
DELAYS = jnp.where(WEIGHTS != 0.0, 0.2, 0.0)


def _setup():
    network = HeterogeneousNetwork(
        graph=DenseDelayGraph(WEIGHTS, DELAYS),
        groups={
            "a": DynamicsGroup(
                DrivenLinear(decay=0.15),
                nodes=(0, 2),
                noise=AdditiveNoise(sigma=0.01, key=jax.random.key(7)),
                initial_state=jnp.array([[0.2, -0.1]]),
            ),
            "b": DynamicsGroup(
                DrivenLinear(decay=0.25),
                nodes=(1, 3),
                external_input={"stimulus": ConstantInput(amplitude=0.05)},
                initial_state=jnp.array([[0.1, 0.3]]),
            ),
        },
        routes={
            "activity": SignalRoute(
                source={"a": "x", "b": "x"},
                coupling=DelayedLinearCoupling(G=0.4),
                target={"a": "drive", "b": "drive"},
            )
        },
    )
    return prepare(network, Euler(), t1=0.4, dt=0.1)


def _run_parallel(model, space, *, n_vmap):
    execution = ParallelExecution(
        model,
        space,
        n_vmap=n_vmap,
        n_pmap=1,
    )
    result = execution.run()
    actual = np.asarray([np.asarray(value) for value in result])
    return actual, execution


def test_product_space_sweeps_group_and_route_parameters():
    solve_fn, config = _setup()
    swept = config.copy()
    swept.groups.a.dynamics.decay = DataAxis(jnp.array([0.1, 0.3]))
    swept.routes.activity.coupling.G = DataAxis(jnp.array([0.2, 0.6]))

    def observe(cfg):
        solution = solve_fn(cfg)
        return jnp.array(
            [
                cfg.groups.a.dynamics.decay,
                cfg.routes.activity.coupling.G,
                solution.ys.a[-1].mean(),
                solution.ys.b[-1].mean(),
            ]
        )

    actual, execution = _run_parallel(observe, Space(swept, mode="product"), n_vmap=2)
    assert actual.shape == (4, 4)
    assert len(jax.tree.leaves(execution.diff_state)) == 2
    assert set(map(tuple, actual[:, :2])) == {
        (np.float32(decay), np.float32(gain))
        for decay in (0.1, 0.3)
        for gain in (0.2, 0.6)
    }

    for decay, gain, *observed in actual:
        manual = config.copy()
        manual.groups.a.dynamics.decay = decay
        manual.routes.activity.coupling.G = gain
        expected = solve_fn(manual)
        np.testing.assert_allclose(
            observed,
            [expected.ys.a[-1].mean(), expected.ys.b[-1].mean()],
            rtol=1e-6,
            atol=1e-6,
        )


def test_aligned_space_sweeps_graph_history_noise_and_external_input():
    solve_fn, config = _setup()
    swept = config.copy()
    group = "case"
    weight_cases = jnp.stack([config.graph.weights, config.graph.weights * 1.4])
    delay_cases = jnp.stack([config.graph.delays, config.graph.delays * 0.5])
    history_cases = jnp.stack(
        [
            config.routes.activity.history,
            jnp.full_like(config.routes.activity.history, 0.35),
        ]
    )
    sigma_cases = jnp.array([0.0, 0.03])
    stimulus_cases = jnp.array([0.0, 0.2])

    swept.graph.weights = DataAxis(weight_cases, group=group)
    swept.graph.delays = DataAxis(delay_cases, group=group)
    swept.routes.activity.history = DataAxis(history_cases, group=group)
    swept.groups.a.noise.sigma = DataAxis(sigma_cases, group=group)
    swept.groups.b.external.stimulus.amplitude = DataAxis(stimulus_cases, group=group)

    def observe(cfg):
        solution = solve_fn(cfg)
        return jnp.array(
            [
                cfg.graph.weights[0, 1],
                cfg.graph.delays[0, 1],
                cfg.routes.activity.history[0, 0, 0],
                cfg.groups.a.noise.sigma,
                cfg.groups.b.external.stimulus.amplitude,
                solution.ys.a[-1].mean(),
                solution.ys.b[-1].mean(),
            ]
        )

    actual, execution = _run_parallel(observe, Space(swept, mode="zip"), n_vmap=2)
    assert actual.shape == (2, 7)
    assert len(jax.tree.leaves(execution.diff_state)) == 5

    for index, row in enumerate(actual):
        manual = config.copy()
        manual.graph.weights = weight_cases[index]
        manual.graph.delays = delay_cases[index]
        manual.routes.activity.history = history_cases[index]
        manual.groups.a.noise.sigma = sigma_cases[index]
        manual.groups.b.external.stimulus.amplitude = stimulus_cases[index]
        expected = solve_fn(manual)

        np.testing.assert_allclose(
            row[:5],
            [
                weight_cases[index, 0, 1],
                delay_cases[index, 0, 1],
                history_cases[index, 0, 0, 0],
                sigma_cases[index],
                stimulus_cases[index],
            ],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            row[5:],
            [expected.ys.a[-1].mean(), expected.ys.b[-1].mean()],
            rtol=1e-6,
            atol=1e-6,
        )
