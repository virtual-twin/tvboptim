"""Copying prepared states, eagerly and inside traced code."""

import copy

import jax
import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import Heun

COPIES = {"deepcopy": copy.deepcopy, "method": lambda s: s.copy()}


@pytest.fixture(scope="module")
def prepared():
    network = Network(
        dynamics=ReducedWongWang(),
        coupling={"instant": LinearCoupling(source="S", G=0.5)},
        graph=DenseGraph(jnp.ones((4, 4))),
        noise=AdditiveNoise(sigma=0.01, apply_to="S"),
    )
    return prepare(network, Heun(), t0=0.0, t1=10.0, dt=1.0)


@pytest.mark.parametrize("copy_fn", COPIES.values(), ids=COPIES.keys())
def test_copy_inside_jit_with_typed_key(prepared, copy_fn):
    sim, state = prepared
    assert jax.dtypes.issubdtype(state.noise.key.dtype, jax.dtypes.prng_key)

    def loss(G):
        s = copy_fn(state)
        s.coupling.instant.G = G
        return sim(s).ys.sum()

    expected = loss(0.5)
    assert jnp.allclose(jax.jit(loss)(0.5), expected)
    assert jnp.isfinite(jax.jit(jax.grad(loss))(0.5))


@pytest.mark.parametrize("copy_fn", COPIES.values(), ids=COPIES.keys())
def test_copy_isolates_containers(prepared, copy_fn):
    _, state = prepared
    original = state.coupling.instant.G
    clone = copy_fn(state)
    clone.coupling.instant.G = 2.0
    clone.noise.extra = 1.0
    assert state.coupling.instant.G is original
    assert "extra" not in state.noise
