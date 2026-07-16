"""Delayed sparse message passing, buffer, and delay-gradient contracts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jax.extend import core as jax_core

jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling import DelayedLinearCoupling
from tvboptim.experimental.network_dynamics.coupling.base import DelayedCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import Linear
from tvboptim.experimental.network_dynamics.graph import (
    DenseDelayGraph,
    SparseDelayGraph,
)

WEIGHTS = jnp.array(
    [
        [0.0, 0.5, 0.0],
        [0.7, 0.0, 0.2],
        [0.4, 0.0, 0.0],
    ],
    dtype=jnp.float64,
)
DELAYS = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [0.35, 0.0, 1.25],
        [0.65, 0.0, 0.0],
    ],
    dtype=jnp.float64,
)
EDGE_PARAM = jnp.array(
    [
        [1.0, 1.5, 2.0],
        [2.5, 3.0, 3.5],
        [4.0, 4.5, 5.0],
    ],
    dtype=jnp.float64,
)
CURRENT = jnp.array([[0.2, -0.7, 1.1]], dtype=jnp.float64)
BUFFER_STRATEGIES = ("roll", "circular", "preallocated")


class DelayedEdgeScaledCoupling(DelayedCoupling):
    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(edge=1.0)
    EDGE_PARAMS = ("edge",)

    def pre(self, incoming_states, local_states, params):
        return params.edge * incoming_states

    def post(self, summed_inputs, local_states, params):
        return summed_inputs


def _sparse_graph():
    sorted_indices = np.argwhere(np.asarray(WEIGHTS) != 0.0)
    order = np.array([2, 0, 3, 1])
    indices = jnp.asarray(sorted_indices[order])
    weights_e = WEIGHTS[indices[:, 0], indices[:, 1]]
    delays_e = DELAYS[indices[:, 0], indices[:, 1]]
    weights = BCOO(
        (weights_e, indices),
        shape=WEIGHTS.shape,
        indices_sorted=False,
        unique_indices=True,
    )
    delays = BCOO(
        (delays_e, indices),
        shape=DELAYS.shape,
        indices_sorted=False,
        unique_indices=True,
    )
    return SparseDelayGraph(weights, delays, max_delay_bound=2.0)


def _graph(sparse):
    if sparse:
        return _sparse_graph()
    return DenseDelayGraph(WEIGHTS, DELAYS, max_delay_bound=2.0)


def _prepare(coupling, graph, interpolate):
    network = Network(Linear(), {"delayed": coupling}, graph)
    all_data, all_state = network.prepare(dt=1.0, t0=0.0, t1=2.0)
    data = all_data.delayed
    coupling_state = all_state.delayed

    initial_rows = data.max_delay_steps + 1 + int(interpolate)
    row = jnp.arange(coupling_state.history.shape[0], dtype=jnp.float64)
    source = jnp.arange(graph.n_nodes, dtype=jnp.float64)
    history = (
        0.6 * (row[:, None, None] - (initial_rows - 1))
        + 0.4 * source[None, None, :]
        + 0.1
    )
    coupling_state = Bunch({**coupling_state, "history": history})
    return data, coupling_state


def _compute(coupling, graph, data, coupling_state):
    enriched = coupling.precompute(data, coupling.params, graph)
    return coupling.compute(
        0.0,
        CURRENT,
        enriched,
        coupling_state,
        coupling.params,
        graph,
    )


@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("buffer_strategy", BUFFER_STRATEGIES)
def test_sparse_delayed_matches_dense_for_every_history_strategy(
    buffer_strategy, interpolate
):
    outputs = {}
    for sparse in (False, True):
        graph = _graph(sparse)
        coupling = DelayedLinearCoupling(
            incoming_states="x",
            G=0.7,
            b=-0.2,
            buffer_strategy=buffer_strategy,
            history_interpolation="linear" if interpolate else None,
        )
        data, coupling_state = _prepare(coupling, graph, interpolate)
        outputs[sparse] = _compute(coupling, graph, data, coupling_state)

    np.testing.assert_allclose(outputs[True], outputs[False], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("buffer_strategy", BUFFER_STRATEGIES)
def test_sparse_delay_gradients_match_dense_gathered_at_prepared_edges(
    buffer_strategy, interpolate
):
    dense_graph = _graph(False)
    sparse_graph = _graph(True)
    dense_coupling = DelayedLinearCoupling(
        incoming_states="x",
        G=0.7,
        buffer_strategy=buffer_strategy,
        history_interpolation="linear" if interpolate else None,
    )
    sparse_coupling = DelayedLinearCoupling(
        incoming_states="x",
        G=0.7,
        buffer_strategy=buffer_strategy,
        history_interpolation="linear" if interpolate else None,
    )
    dense_data, dense_state = _prepare(dense_coupling, dense_graph, interpolate)
    sparse_data, sparse_state = _prepare(sparse_coupling, sparse_graph, interpolate)

    def dense_loss(delays):
        graph = eqx.tree_at(lambda g: g.delays, dense_graph, delays)
        output = _compute(dense_coupling, graph, dense_data, dense_state)
        return jnp.sum(output**2)

    def sparse_loss(delays_e):
        graph = eqx.tree_at(lambda g: g.delays.data, sparse_graph, delays_e)
        output = _compute(sparse_coupling, graph, sparse_data, sparse_state)
        return jnp.sum(output**2)

    dense_grad = jax.jit(jax.grad(dense_loss))(DELAYS)
    sparse_grad = jax.jit(jax.grad(sparse_loss))(sparse_graph.delays.data)
    expected_sparse_grad = sparse_graph.gather_edges(dense_grad)

    np.testing.assert_allclose(
        sparse_grad,
        expected_sparse_grad,
        rtol=1e-12,
        atol=1e-12,
    )
    assert bool(jnp.all(jnp.isfinite(sparse_grad)))
    zero_delay_edges = sparse_graph.delays.data == 0.0
    assert bool(jnp.any(zero_delay_edges))
    np.testing.assert_allclose(
        sparse_grad[zero_delay_edges],
        expected_sparse_grad[zero_delay_edges],
        rtol=1e-12,
        atol=1e-12,
    )
    if interpolate:
        assert bool(jnp.any(jnp.abs(sparse_grad) > 0.0))
    else:
        np.testing.assert_array_equal(sparse_grad, jnp.zeros_like(sparse_grad))


@pytest.mark.parametrize("representation", ["dense", "sparse"])
@pytest.mark.parametrize("layout", ["graph", "prepared_e"])
def test_delayed_edge_params_execute_in_both_public_layouts(representation, layout):
    sparse = representation == "sparse"
    graph = _graph(sparse)
    edge = EDGE_PARAM if layout == "graph" else graph.gather_edges(EDGE_PARAM)
    coupling = DelayedEdgeScaledCoupling(
        incoming_states="x",
        edge=edge,
        history_interpolation="linear",
    )
    data, coupling_state = _prepare(coupling, graph, interpolate=True)
    actual = _compute(coupling, graph, data, coupling_state)

    dense_graph = _graph(False)
    dense_coupling = DelayedEdgeScaledCoupling(
        incoming_states="x",
        edge=EDGE_PARAM,
        history_interpolation="linear",
    )
    dense_data, dense_state = _prepare(
        dense_coupling,
        dense_graph,
        interpolate=True,
    )
    expected = _compute(dense_coupling, dense_graph, dense_data, dense_state)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("interpolate", [False, True])
def test_genuinely_empty_sparse_delayed_graph_returns_zero(interpolate):
    indices = jnp.empty((0, 2), dtype=jnp.int32)
    data = jnp.empty((0,), dtype=jnp.float64)
    weights = BCOO((data, indices), shape=(3, 3), unique_indices=True)
    delays = BCOO((data, indices), shape=(3, 3), unique_indices=True)
    graph = SparseDelayGraph(weights, delays)
    coupling = DelayedLinearCoupling(
        incoming_states="x",
        G=1.0,
        b=0.0,
        history_interpolation="linear" if interpolate else None,
    )
    prepared, coupling_state = _prepare(coupling, graph, interpolate)

    actual = _compute(coupling, graph, prepared, coupling_state)
    np.testing.assert_array_equal(actual, jnp.zeros((1, 3), dtype=actual.dtype))


def test_sparse_delayed_jaxpr_has_no_node_squared_array():
    n_nodes = 257
    degree = 4
    target = jnp.repeat(jnp.arange(n_nodes), degree)
    offset = jnp.tile(jnp.arange(1, degree + 1), n_nodes)
    source = (target + offset) % n_nodes
    indices = jnp.stack([target, source], axis=1)
    n_edges = indices.shape[0]
    weights = BCOO(
        (jnp.ones(n_edges), indices),
        shape=(n_nodes, n_nodes),
        unique_indices=True,
    )
    delays = BCOO(
        ((jnp.arange(n_edges) % 9) * 0.025, indices),
        shape=(n_nodes, n_nodes),
        unique_indices=True,
    )
    graph = SparseDelayGraph(weights, delays, max_delay_bound=0.25)
    coupling = DelayedLinearCoupling(
        incoming_states="x",
        history_interpolation="linear",
        buffer_strategy="circular",
        warn_on_delay_clamp=True,
    )
    network = Network(Linear(), {"delayed": coupling}, graph)
    all_data, all_state = network.prepare(dt=0.05, t0=0.0, t1=0.1)
    prepared = all_data.delayed
    coupling_state = all_state.delayed
    graph_aux = graph.tree_flatten()[1]

    def sparse_delayed(state, history, weights_e, delays_e):
        runtime_weights = BCOO(
            (weights_e, indices),
            shape=(n_nodes, n_nodes),
            unique_indices=True,
        )
        runtime_delays = BCOO(
            (delays_e, indices),
            shape=(n_nodes, n_nodes),
            unique_indices=True,
        )
        runtime_graph = SparseDelayGraph.tree_unflatten(
            graph_aux,
            (runtime_weights, runtime_delays),
        )
        runtime_state = Bunch({**coupling_state, "history": history})
        enriched = coupling.precompute(prepared, coupling.params, runtime_graph)
        return coupling.compute(
            0.0,
            state,
            enriched,
            runtime_state,
            coupling.params,
            runtime_graph,
        )

    closed = jax.make_jaxpr(sparse_delayed)(
        jnp.zeros((1, n_nodes)),
        coupling_state.history,
        graph.weights.data,
        graph.delays.data,
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
