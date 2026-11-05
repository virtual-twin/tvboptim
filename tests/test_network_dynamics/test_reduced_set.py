"""ReducedSetFitzHughNagumo: TVB parity of derived values and gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tvb.simulator import models as tvb_models

from tvboptim.experimental.network_dynamics import Bunch, Network, prepare
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    ReducedSetFitzHughNagumo,
)
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun

DISTRIBUTIONS = [(0.0, 0.35), (0.2, 0.6), (-0.1, 0.15)]
DIST_IDS = [f"mu={mu}-sigma={sigma}" for mu, sigma in DISTRIBUTIONS]


class ModeSumCoupling(LinearCoupling):
    """Sums the xi modes before transport, as TVB sums coupling over modes."""

    def pre(self, incoming_states, local_states, params):
        return incoming_states.sum(axis=0, keepdims=True)


def _tvb(mu, sigma):
    model = tvb_models.ReducedSetFitzHughNagumo(
        mu=np.array([mu]), sigma=np.array([sigma])
    )
    model.configure()
    return model


def _flatten(tvb_array):
    """TVB ``(4, nodes, modes)`` to tvboptim ``(12, nodes)``."""
    n_vars, n_nodes, n_modes = tvb_array.shape
    return tvb_array.transpose(0, 2, 1).reshape(n_vars * n_modes, n_nodes)


@pytest.mark.parametrize("mu,sigma", DISTRIBUTIONS, ids=DIST_IDS)
def test_derived_values_match_tvb(mu, sigma):
    reference = _tvb(mu, sigma)
    model = ReducedSetFitzHughNagumo(mu=mu, sigma=sigma)
    derived = model.resolve_params(model.params)
    for name in model.DERIVED_PARAMS:
        expected = getattr(reference, name)
        actual = np.asarray(derived[name])
        if expected.shape == (1, 3):  # TVB row vectors are (3, 1) columns here
            actual = actual.T
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mu,sigma", DISTRIBUTIONS, ids=DIST_IDS)
def test_dynamics_match_tvb_dfun(mu, sigma):
    reference = _tvb(mu, sigma)
    model = ReducedSetFitzHughNagumo(mu=mu, sigma=sigma)
    rng = np.random.default_rng(0)
    state = rng.normal(size=(4, 5, 3))
    coupling = 0.1 * rng.normal(size=(2, 5, 3))

    expected = _flatten(reference.dfun(state, coupling))
    actual = model.dynamics(
        0.0,
        jnp.asarray(_flatten(state)),
        model.resolve_params(model.params),
        Bunch(
            instant=jnp.asarray(coupling[0].sum(axis=1))[None],
            delayed=jnp.zeros((1, 5)),
        ),
        Bunch(),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ["mu", "sigma"])
def test_population_parameters_receive_gradients(name):
    n_nodes = 4
    network = Network(
        dynamics=ReducedSetFitzHughNagumo(nv=300, nu=300),
        coupling={"instant": ModeSumCoupling(source=("xi1", "xi2", "xi3"), G=0.05)},
        graph=DenseGraph(jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)),
    )
    solve_fn, config = prepare(network, Heun(), t1=5.0, dt=0.05)

    def loss(value):
        varied = config.copy()
        varied.dynamics[name] = value
        return jnp.square(solve_fn(varied).ys[:, :3]).mean()

    grad = jax.grad(loss)(jnp.asarray(config.dynamics[name]))
    assert jnp.isfinite(grad) and grad != 0.0


@pytest.mark.parametrize("name", ["mu", "sigma"])
def test_node_varying_distribution_is_rejected(name):
    model = ReducedSetFitzHughNagumo(**{name: jnp.array([0.3, 0.4])})
    with pytest.raises(ValueError, match=f"'{name}' must be a scalar"):
        model.resolve_params(model.params)


@pytest.mark.parametrize("kwargs", [{"nv": 1000}, {"nu": 0}, {"nv": 30.0}])
def test_sample_counts_are_validated(kwargs):
    with pytest.raises(ValueError, match="divisible by 3"):
        ReducedSetFitzHughNagumo(**kwargs)
