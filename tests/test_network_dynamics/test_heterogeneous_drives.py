"""Per-group stochastic and external driving for heterogeneous networks."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics import (
    DynamicsGroup,
    HeterogeneousNetwork,
    prepare,
)
from tvboptim.experimental.network_dynamics.core import Bunch
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    Generic2dOscillator,
    Linear,
)
from tvboptim.experimental.network_dynamics.external_input import (
    AbstractExternalInput,
    ConstantInput,
    DataInput,
)
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.noise import (
    AdditiveNoise,
    MultiplicativeNoise,
)
from tvboptim.experimental.network_dynamics.solvers import Euler, Heun, RungeKutta4


class CountingInput(AbstractExternalInput):
    """Small stateful input used to audit accepted-step updates."""

    DEFAULT_PARAMS = Bunch(amplitude=0.25)

    def __init__(self, n_nodes, **kwargs):
        self.n_nodes = n_nodes
        super().__init__(**kwargs)

    def prepare(self, network, dt):
        if network is not None:
            assert network.graph.n_nodes == self.n_nodes
        return Bunch(dt=dt), Bunch(count=jnp.zeros((1, self.n_nodes)))

    def compute(self, t, state, input_data, input_state, params):
        del t, state, input_data
        return params.amplitude * input_state.count

    def update_state(self, input_data, input_state, new_state):
        del input_data, new_state
        return Bunch(count=input_state.count + 1.0)


def _graph(n_nodes=4):
    return DenseGraph(jnp.zeros((n_nodes, n_nodes)))


def _driven_network(external, *, noise=None):
    initial = jnp.array([[0.2, -0.1], [0.05, 0.3]])
    return HeterogeneousNetwork(
        graph=_graph(),
        groups={
            "driven": DynamicsGroup(
                Generic2dOscillator(),
                [0, 2],
                initial_state=initial,
                noise=noise,
                external_input={"stimulus": external},
            ),
            "other": DynamicsGroup(
                Linear(gamma=-0.4),
                [1, 3],
                initial_state=jnp.array([[0.4, -0.2]]),
            ),
        },
    )


@pytest.mark.parametrize("solver", [Euler(), Heun(), RungeKutta4()])
def test_group_external_matches_independent_bare_dynamics(solver):
    group_external = CountingInput(2, amplitude=0.4)
    network = _driven_network(group_external)
    simulate, config = prepare(network, solver, t0=0.0, t1=0.5, dt=0.1)
    actual = simulate(config)

    bare_external = CountingInput(2, amplitude=0.4)
    bare_fn, bare_config = prepare(
        Generic2dOscillator(),
        solver,
        t0=0.0,
        t1=0.5,
        dt=0.1,
        n_nodes=2,
        externals={"stimulus": bare_external},
    )
    bare_config.initial_state.dynamics = config.initial_state.driven
    expected = bare_fn(bare_config)

    assert jnp.array_equal(actual.groups.driven.ys, expected.ys)


def test_group_data_input_uses_group_local_node_space():
    times = jnp.array([0.0, 0.2, 0.4])
    data = jnp.array([[0.0, 1.0], [0.5, -0.5], [1.0, 0.0]])
    simulate, config = prepare(
        _driven_network(DataInput(times, data)),
        Heun(),
        t0=0.0,
        t1=0.4,
        dt=0.1,
    )
    result = jax.jit(simulate)(config)
    assert result.groups.driven.ys.shape == (4, 2, 2)
    assert jnp.all(jnp.isfinite(result.groups.driven.ys))


def test_external_parameters_are_live_and_differentiable_after_prepare():
    simulate, config = prepare(
        _driven_network(ConstantInput(amplitude=0.3)),
        Heun(),
        t0=0.0,
        t1=0.5,
        dt=0.1,
    )

    def loss(amplitude):
        varied = config.copy()
        varied.groups.driven.external.stimulus.amplitude = amplitude
        return jnp.square(simulate(varied).groups.driven.ys).sum()

    value, reverse = jax.jit(jax.value_and_grad(loss))(jnp.array(0.3))
    _, forward = jax.jvp(loss, (jnp.array(0.3),), (jnp.array(1.0),))
    assert jnp.isfinite(value)
    assert jnp.isfinite(reverse)
    assert jnp.allclose(forward, reverse, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("noise_type", [AdditiveNoise, MultiplicativeNoise])
def test_group_noise_matches_independent_bare_dynamics(noise_type):
    key = jax.random.key(7)
    group_noise = noise_type(sigma=0.03, apply_to=["V"], key=key)
    network = _driven_network(ConstantInput(amplitude=0.2), noise=group_noise)
    simulate, config = prepare(network, Heun(), t0=0.0, t1=0.5, dt=0.1)
    actual = simulate(config)

    bare_noise = noise_type(sigma=0.03, apply_to=["V"], key=key)
    bare_fn, bare_config = prepare(
        Generic2dOscillator(),
        Heun(),
        t0=0.0,
        t1=0.5,
        dt=0.1,
        n_nodes=2,
        noise=bare_noise,
        externals={"stimulus": ConstantInput(amplitude=0.2)},
    )
    bare_config.initial_state.dynamics = config.initial_state.driven
    expected = bare_fn(bare_config)

    assert jnp.array_equal(actual.groups.driven.ys, expected.ys)


def test_group_noise_key_injection_vmap_and_grad_are_live():
    network = _driven_network(
        ConstantInput(amplitude=0.2),
        noise=AdditiveNoise(sigma=0.03, key=jax.random.key(3)),
    )
    simulate, config = prepare(network, Heun(), t0=0.0, t1=0.5, dt=0.1)
    baseline = simulate(config)

    samples = jax.random.normal(config.groups.driven.noise.key, (5, 2, 2))
    injected = eqx.tree_at(
        lambda c: c._internal.noise_samples.driven,
        config,
        samples,
        is_leaf=lambda value: value is None,
    )
    assert jnp.array_equal(
        baseline.groups.driven.ys, simulate(injected).groups.driven.ys
    )

    def with_seed(seed, base):
        return eqx.tree_at(
            lambda c: c.groups.driven.noise.key, base, jax.random.key(seed)
        )

    batched_config = jax.vmap(with_seed, in_axes=(0, None))(jnp.arange(3), config)
    batched = jax.jit(jax.vmap(simulate))(batched_config)
    assert batched.groups.driven.ys.shape == (3, 5, 2, 2)
    assert jnp.any(batched.groups.driven.ys[0] != batched.groups.driven.ys[1])

    def loss(sigma):
        varied = config.copy()
        varied.groups.driven.noise.sigma = sigma
        return jnp.square(simulate(varied).groups.driven.ys).sum()

    value, gradient = jax.jit(jax.value_and_grad(loss))(jnp.array(0.03))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_differently_shaped_group_noise_trees_match_independent_solves():
    generic_initial = jnp.array([[0.2, -0.1], [0.05, 0.3]])
    linear_initial = jnp.array([[0.4, -0.2]])
    generic_noise = AdditiveNoise(
        sigma=0.03, apply_to=["V"], key=jax.random.key(5)
    )
    linear_noise = AdditiveNoise(sigma=0.02, key=jax.random.key(8))
    network = HeterogeneousNetwork(
        graph=_graph(),
        groups={
            "generic": DynamicsGroup(
                Generic2dOscillator(),
                [0, 2],
                initial_state=generic_initial,
                noise=generic_noise,
            ),
            "linear": DynamicsGroup(
                Linear(gamma=-0.4),
                [1, 3],
                initial_state=linear_initial,
                noise=linear_noise,
            ),
        },
    )
    simulate, config = prepare(network, Heun(), t0=0.0, t1=0.5, dt=0.1)
    actual = simulate(config)

    for name, dynamics, initial, noise in (
        (
            "generic",
            Generic2dOscillator(),
            generic_initial,
            AdditiveNoise(sigma=0.03, apply_to=["V"], key=jax.random.key(5)),
        ),
        (
            "linear",
            Linear(gamma=-0.4),
            linear_initial,
            AdditiveNoise(sigma=0.02, key=jax.random.key(8)),
        ),
    ):
        bare_fn, bare_config = prepare(
            dynamics,
            Heun(),
            t0=0.0,
            t1=0.5,
            dt=0.1,
            n_nodes=2,
            noise=noise,
        )
        bare_config.initial_state = initial
        expected = bare_fn(bare_config)
        assert jnp.array_equal(actual.groups[name].ys, expected.ys)


def test_blocked_group_noise_streams_deterministically():
    network = _driven_network(
        ConstantInput(amplitude=0.2),
        noise=AdditiveNoise(sigma=0.03, key=jax.random.key(3)),
    )
    simulate, config = prepare(
        network, Heun(block_size=2), t0=0.0, t1=0.5, dt=0.1
    )
    first = jax.jit(simulate)(config)
    replay = jax.jit(simulate)(config)
    varied = config.copy()
    varied.groups.driven.noise.key = jax.random.key(4)
    changed = jax.jit(simulate)(varied)

    assert jnp.array_equal(first.groups.driven.ys, replay.groups.driven.ys)
    assert jnp.any(first.groups.driven.ys != changed.groups.driven.ys)
