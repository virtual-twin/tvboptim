"""Nested sparse-node/delayed-regional Subspace integration regression."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedLinearCoupling,
    SubspaceCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import Linear
from tvboptim.experimental.network_dynamics.graph import (
    DenseDelayGraph,
    SparseGraph,
)
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.solve import prepare
from tvboptim.experimental.network_dynamics.solvers import Euler

EXPECTED = np.array(
    [
        [[0.235, 0.385, 0.002, 0.452, 0.523, 0.073]],
        [[0.17165, 0.28415, 0.0235, 0.361, 0.40625, 0.06875]],
        [[0.1171875, 0.2015625, 0.036825, 0.28995, 0.3042875, 0.0511625]],
        [
            [
                0.070568125,
                0.133849375,
                0.02880675,
                0.2186505,
                0.214679625,
                0.024835875,
            ]
        ],
    ],
    dtype=np.float64,
)


def test_sparse_node_dense_delayed_regional_subspace_matches_fixed_reference():
    node_indices = jnp.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
    node_graph = SparseGraph(
        BCOO(
            (
                jnp.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                node_indices,
            ),
            shape=(6, 6),
            unique_indices=True,
        )
    )
    regional_graph = DenseDelayGraph(
        weights=jnp.array(
            [
                [0.0, 0.5, 0.0],
                [0.3, 0.0, 0.2],
                [0.4, 0.0, 0.0],
            ]
        ),
        delays=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        max_delay_bound=2.0,
    )
    inner = DelayedLinearCoupling(
        incoming_states="x",
        G=0.4,
        b=-0.05,
        history_interpolation="linear",
        buffer_strategy="circular",
    )
    coupling = SubspaceCoupling(
        inner_coupling=inner,
        region_mapping=jnp.array([0, 0, 1, 1, 2, 2]),
        regional_graph=regional_graph,
        use_sparse=True,
    )
    history = NativeSolution(
        ts=jnp.array([-2.0, -1.0, 0.0]),
        ys=jnp.array(
            [
                [[0.1, 0.3, -0.2, 0.4, 0.5, -0.1]],
                [[0.2, 0.4, -0.1, 0.5, 0.6, 0.0]],
                [[0.3, 0.5, 0.0, 0.6, 0.7, 0.1]],
            ]
        ),
        variable_names=("x",),
    )
    network = Network(
        Linear(gamma=-0.25),
        {"delayed": coupling},
        node_graph,
        history=history,
    )
    solve_fn, config = prepare(network, Euler(), t0=0.0, t1=4.0, dt=1.0)

    result = jax.jit(solve_fn)(config)
    assert isinstance(config.graph, SparseGraph)
    assert isinstance(coupling.regional_graph, DenseDelayGraph)
    np.testing.assert_allclose(result.ys, EXPECTED, rtol=1e-14, atol=1e-14)
