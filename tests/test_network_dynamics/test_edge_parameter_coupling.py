"""Two-channel edge-parameter coupling integration contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jax.extend import core as jax_core

jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling.base import (
    InstantaneousCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import Linear
from tvboptim.experimental.network_dynamics.graph import DenseGraph, SparseGraph

WEIGHTS = jnp.array(
    [
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [5.0, 0.0, 0.0],
    ],
    dtype=jnp.float64,
)
WLRE = jnp.array(
    [
        [0.5, 0.7, 0.9],
        [1.1, 1.3, 1.5],
        [1.7, 1.9, 2.1],
    ],
    dtype=jnp.float64,
)
WFFI = jnp.array(
    [
        [2.2, 2.0, 1.8],
        [1.6, 1.4, 1.2],
        [1.0, 0.8, 0.6],
    ],
    dtype=jnp.float64,
)
STATE = jnp.array([[2.0, 3.0, 7.0]], dtype=jnp.float64)


class TwoInputLinear(Linear):
    COUPLING_INPUTS = {"coupling": 2}


class EIBLinearCoupling(InstantaneousCoupling):
    """Documented dual-weight EIB coupling using declared edge parameters."""

    N_OUTPUT_STATES = 2
    DEFAULT_PARAMS = Bunch(wLRE=1.0, wFFI=1.0)
    EDGE_PARAMS = ("wLRE", "wFFI")

    def pre(self, incoming_states, local_states, params):
        source = incoming_states[0]
        return jnp.stack(
            [source * params.wLRE, source * params.wFFI],
            axis=0,
        )

    def post(self, summed_inputs, local_states, params):
        return summed_inputs


def _prepare(coupling, graph):
    network = Network(TwoInputLinear(), {"coupling": coupling}, graph)
    data, state = network.prepare(dt=0.1, t0=0.0, t1=0.2)
    return data.coupling, state.coupling


def _compute(coupling, graph, data, coupling_state, params, state=STATE):
    enriched = coupling.precompute(data, params, graph)
    return coupling.compute(
        0.0,
        state,
        enriched,
        coupling_state,
        params,
        graph,
    )


@pytest.mark.parametrize("representation", ["dense", "sparse"])
@pytest.mark.parametrize("layout", ["graph", "prepared_e"])
def test_eib_edge_params_match_declared_order_oracle(representation, layout):
    graph = DenseGraph(WEIGHTS) if representation == "dense" else SparseGraph(WEIGHTS)
    w_lre = WLRE if layout == "graph" else graph.gather_edges(WLRE)
    w_ffi = WFFI if layout == "graph" else graph.gather_edges(WFFI)
    coupling = EIBLinearCoupling(
        incoming_states="x",
        wLRE=w_lre,
        wFFI=w_ffi,
    )
    data, coupling_state = _prepare(coupling, graph)

    actual = _compute(
        coupling,
        graph,
        data,
        coupling_state,
        coupling.params,
    )
    source = STATE[0][None, :]
    expected = jnp.stack(
        [
            jnp.sum(WEIGHTS * WLRE * source, axis=1),
            jnp.sum(WEIGHTS * WFFI * source, axis=1),
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_prepared_e_edge_param_and_signal_gradients_match_dense():
    dense_graph = DenseGraph(WEIGHTS)
    sparse_graph = SparseGraph(WEIGHTS)
    dense = EIBLinearCoupling(
        incoming_states="x",
        wLRE=WLRE,
        wFFI=WFFI,
    )
    sparse = EIBLinearCoupling(
        incoming_states="x",
        wLRE=sparse_graph.gather_edges(WLRE),
        wFFI=sparse_graph.gather_edges(WFFI),
    )
    dense_data, dense_state = _prepare(dense, dense_graph)
    sparse_data, sparse_state = _prepare(sparse, sparse_graph)

    def dense_loss(w_lre, w_ffi, signal):
        params = Bunch(wLRE=w_lre, wFFI=w_ffi)
        output = _compute(dense, dense_graph, dense_data, dense_state, params, signal)
        return jnp.sum(output**2)

    def sparse_loss(w_lre_e, w_ffi_e, signal):
        params = Bunch(wLRE=w_lre_e, wFFI=w_ffi_e)
        output = _compute(
            sparse,
            sparse_graph,
            sparse_data,
            sparse_state,
            params,
            signal,
        )
        return jnp.sum(output**2)

    dense_grad = jax.jit(jax.grad(dense_loss, argnums=(0, 1, 2)))(
        WLRE,
        WFFI,
        STATE,
    )
    sparse_grad = jax.jit(jax.grad(sparse_loss, argnums=(0, 1, 2)))(
        sparse.params.wLRE,
        sparse.params.wFFI,
        STATE,
    )

    np.testing.assert_allclose(
        sparse_grad[0],
        sparse_graph.gather_edges(dense_grad[0]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sparse_grad[1],
        sparse_graph.gather_edges(dense_grad[1]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(sparse_grad[2], dense_grad[2], rtol=1e-12, atol=1e-12)


def test_prepared_e_eib_large_sparse_jaxpr_has_no_node_squared_array():
    n_nodes = 4096
    degree = 4
    target = jnp.repeat(jnp.arange(n_nodes), degree)
    offset = jnp.tile(jnp.arange(1, degree + 1), n_nodes)
    source = (target + offset) % n_nodes
    indices = jnp.stack([target, source], axis=1)
    n_edges = indices.shape[0]
    weights = BCOO(
        (jnp.linspace(0.1, 1.0, n_edges), indices),
        shape=(n_nodes, n_nodes),
        unique_indices=True,
    )
    graph = SparseGraph(weights)
    coupling = EIBLinearCoupling(
        incoming_states="x",
        wLRE=jnp.linspace(0.5, 1.5, n_edges),
        wFFI=jnp.linspace(1.5, 0.5, n_edges),
    )
    data, coupling_state = _prepare(coupling, graph)
    graph_aux = graph.tree_flatten()[1]

    def sparse_eib(signal, weights_e, w_lre_e, w_ffi_e):
        runtime_weights = BCOO(
            (weights_e, indices),
            shape=(n_nodes, n_nodes),
            unique_indices=True,
        )
        runtime_graph = SparseGraph.tree_unflatten(graph_aux, (runtime_weights,))
        params = Bunch(wLRE=w_lre_e, wFFI=w_ffi_e)
        return _compute(
            coupling,
            runtime_graph,
            data,
            coupling_state,
            params,
            signal,
        )

    closed = jax.make_jaxpr(sparse_eib)(
        jnp.linspace(-1.0, 1.0, n_nodes)[None, :],
        graph.weights.data,
        coupling.params.wLRE,
        coupling.params.wFFI,
    )
    shapes = []
    seen = set()

    def inspect(var):
        shape = getattr(getattr(var, "aval", None), "shape", None)
        if shape is not None:
            shapes.append(tuple(int(size) for size in shape))

    def visit(value):
        if isinstance(value, jax_core.ClosedJaxpr):
            visit(value.jaxpr)
        elif isinstance(value, jax_core.Jaxpr):
            if id(value) in seen:
                return
            seen.add(id(value))
            for var in (*value.constvars, *value.invars, *value.outvars):
                inspect(var)
            for equation in value.eqns:
                for var in (*equation.invars, *equation.outvars):
                    inspect(var)
                visit(equation.params)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(closed)
    assert (n_nodes, n_nodes) not in shapes
    assert max(np.prod(shape, dtype=np.int64) for shape in shapes) < n_nodes**2
