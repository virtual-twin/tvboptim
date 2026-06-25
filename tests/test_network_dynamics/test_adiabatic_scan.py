"""Test the adiabatic_scan analysis helper across model/branch combinations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.analysis import (
    AdiabaticScanResult,
    adiabatic_scan,
)
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    JansenRit,
    ReducedWongWang,
)
from tvboptim.experimental.network_dynamics.graph import DenseGraph

# (model class, coupling state variable) — kept generic/parametrized
MODELS = [
    (ReducedWongWang, "S"),
    (JansenRit, "y1"),
]


def _build_network(model_class, coupling_var, n_nodes=4, seed=0):
    graph = DenseGraph.random(n_nodes=n_nodes, key=jax.random.PRNGKey(seed))
    coupling = LinearCoupling(incoming_states=coupling_var, G=0.1)
    return Network(
        dynamics=model_class(),
        coupling={"instant": coupling},
        graph=graph,
    )


@pytest.mark.parametrize("model_class,coupling_var", MODELS)
@pytest.mark.parametrize("bothways", [True, False])
def test_adiabatic_scan_shapes_and_finiteness(model_class, coupling_var, bothways):
    network = _build_network(model_class, coupling_var)
    n = 4
    res = adiabatic_scan(
        network,
        accessor=lambda c: c.coupling.instant.G,
        low=0.0,
        high=0.5,
        n=n,
        t=20.0,
        skip=10.0,
        dt=1.0,
        bothways=bothways,
    )

    assert isinstance(res, AdiabaticScanResult)
    assert res.n_up == n
    expected_len = 2 * n if bothways else n
    assert len(res.p) == expected_len

    # default statistics present, aligned with p, and finite
    assert set(res.stats) == {"mean", "min", "max"}
    # Bunch allows attribute access alongside key access
    np.testing.assert_array_equal(res.stats.mean, res.stats["mean"])
    for arr in res.stats.values():
        assert arr.shape == (expected_len,)
        assert np.all(np.isfinite(arr))

    # min <= mean <= max at every scan point
    assert np.all(res.stats["min"] <= res.stats["mean"] + 1e-8)
    assert np.all(res.stats["mean"] <= res.stats["max"] + 1e-8)

    # ascending branch sweeps the requested bounds
    np.testing.assert_allclose(res.p[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(res.p[n - 1], 0.5, atol=1e-12)


def test_adiabatic_scan_custom_observe_and_statistics():
    network = _build_network(ReducedWongWang, "S")
    res = adiabatic_scan(
        network,
        accessor=lambda c: c.coupling.instant.G,
        low=0.0,
        high=0.3,
        n=3,
        t=20.0,
        skip=10.0,
        dt=1.0,
        bothways=False,
        observe=lambda result: result.ys[:, 0, :],
        statistics={"std": lambda arr: jnp.std(arr.mean(axis=0))},
    )
    assert set(res.stats) == {"std"}
    assert res.stats["std"].shape == (3,)
    assert np.all(np.isfinite(res.stats["std"]))


def test_adiabatic_scan_vector_valued_statistic():
    n_nodes = 4
    network = _build_network(ReducedWongWang, "S", n_nodes=n_nodes)
    n = 3
    res = adiabatic_scan(
        network,
        accessor=lambda c: c.coupling.instant.G,
        low=0.0,
        high=0.3,
        n=n,
        t=20.0,
        skip=10.0,
        dt=1.0,
        bothways=False,
        # per-node temporal mean -> [n_nodes], stacked to [len(p), n_nodes]
        statistics={"per_node": lambda arr: arr.mean(axis=0)},
    )
    assert set(res.stats) == {"per_node"}
    assert res.stats["per_node"].shape == (n, n_nodes)
    assert np.all(np.isfinite(res.stats["per_node"]))
