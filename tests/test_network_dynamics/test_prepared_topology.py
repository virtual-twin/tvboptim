"""Prepared sparse topology lifecycle and Space batching contracts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from tvboptim.execution import ParallelExecution
from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedLinearCoupling,
    LinearCoupling,
    SubspaceCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    Linear,
    ReducedWongWang,
)
from tvboptim.experimental.network_dynamics.graph import (
    DenseGraph,
    SparseDelayGraph,
    SparseGraph,
)
from tvboptim.experimental.network_dynamics.solve import prepare
from tvboptim.experimental.network_dynamics.solvers import Euler
from tvboptim.types import DataAxis, Space

INDICES = jnp.array([[0, 1], [1, 2], [2, 0]])
WEIGHTS = jnp.array([1.0, 2.0, 3.0])
DELAYS = jnp.array([0.0, 0.1, 0.2])


def _bcoo(data, indices=INDICES):
    return BCOO(
        (jnp.asarray(data), jnp.asarray(indices)),
        shape=(3, 3),
        unique_indices=True,
    )


def _instant_setup(*, two_couplings=False):
    graph = SparseGraph(_bcoo(WEIGHTS))
    couplings = {"instant": LinearCoupling(incoming_states="x")}
    if two_couplings:
        couplings["delayed"] = LinearCoupling(incoming_states="x")
    network = Network(Linear(), couplings, graph)
    solve_fn, config = prepare(network, Euler(), t1=0.2, dt=0.1)
    return network, solve_fn, config


def _delayed_setup():
    graph = SparseDelayGraph(_bcoo(WEIGHTS), _bcoo(DELAYS))
    coupling = DelayedLinearCoupling(incoming_states="x")
    network = Network(Linear(), {"delayed": coupling}, graph)
    solve_fn, config = prepare(network, Euler(), t1=0.3, dt=0.1)
    return network, solve_fn, config


def _reordered(matrix):
    return BCOO(
        (matrix.data[::-1], matrix.indices[::-1]),
        shape=matrix.shape,
        unique_indices=True,
    )


def test_prepare_builds_one_public_order_topology_for_all_couplings():
    network, _solve_fn, config = _instant_setup(two_couplings=True)
    instant = config._internal.coupling.instant._prepared_topology
    second = config._internal.coupling.delayed._prepared_topology

    assert instant is second
    assert instant.representation == "sparse"
    assert instant.n_source == instant.n_target == 3
    assert instant.n_edges == 3
    assert instant.edge_indices is network.graph.edge_indices
    np.testing.assert_array_equal(instant.edge_indices, INDICES)
    np.testing.assert_array_equal(instant.target_e, INDICES[:, 0])
    np.testing.assert_array_equal(instant.source_e, INDICES[:, 1])

    snapshot = config.copy()
    assert snapshot.graph is not config.graph
    assert snapshot.graph.edge_indices is config.graph.edge_indices
    assert (
        snapshot._internal.coupling.instant._prepared_topology.edge_indices
        is instant.edge_indices
    )


def test_dense_prepared_topology_does_not_materialize_coo_indices():
    graph = DenseGraph(jnp.eye(3))
    network = Network(Linear(), {"instant": LinearCoupling(incoming_states="x")}, graph)
    _solve_fn, config = prepare(network, Euler(), t1=0.2, dt=0.1)
    topology = config._internal.coupling.instant._prepared_topology

    assert topology.representation == "dense"
    assert topology.edge_indices is None
    assert topology.n_edges == 9


def test_same_size_weight_reorder_fails_eager_and_jit():
    _network, solve_fn, config = _instant_setup()
    bad = config.copy()
    bad.graph.weights = _reordered(config.graph.weights)

    with pytest.raises(Exception, match="Graph topology changed after prepare"):
        solve_fn(bad)

    compiled = jax.jit(lambda cfg: solve_fn(cfg).ys)
    np.asarray(compiled(config))
    with pytest.raises(Exception, match="Graph topology changed after prepare"):
        np.asarray(compiled(bad))


def test_changed_edge_count_fails_with_reprepare_guidance():
    _network, solve_fn, config = _instant_setup()
    bad = config.copy()
    bad.graph.weights = _bcoo(WEIGHTS[:2], INDICES[:2])

    with pytest.raises(ValueError, match=r"prepare\(\) again.*indices shape"):
        solve_fn(bad)


@pytest.mark.parametrize("change", ["delay_only", "weights_and_delays"])
def test_delayed_topology_mutation_is_checked_against_prepared_order(change):
    _network, solve_fn, config = _delayed_setup()
    bad = config.copy()
    bad.graph.delays = _reordered(config.graph.delays)
    if change == "weights_and_delays":
        bad.graph.weights = _reordered(config.graph.weights)

    with pytest.raises(Exception, match="Graph topology changed after prepare"):
        solve_fn(bad)


def test_data_only_weight_and_delay_replacement_remains_live():
    _network, solve_fn, config = _delayed_setup()
    baseline = np.asarray(solve_fn(config).ys)

    changed = eqx.tree_at(
        lambda cfg: (cfg.graph.weights.data, cfg.graph.delays.data),
        config,
        (config.graph.weights.data * 1.5, config.graph.delays.data * 0.5),
    )
    changed_result = np.asarray(solve_fn(changed).ys)
    assert np.all(np.isfinite(changed_result))
    assert not np.array_equal(changed_result, baseline)

    assigned = config.copy()
    assigned.graph.delays = BCOO(
        (config.graph.delays.data * 0.25, config.graph.delays.indices),
        shape=config.graph.delays.shape,
        unique_indices=True,
    )
    assert np.all(np.isfinite(np.asarray(solve_fn(assigned).ys)))


def test_weight_data_gradient_remains_live_through_topology_check():
    _network, solve_fn, config = _instant_setup()

    def loss(weight_data):
        changed = eqx.tree_at(lambda cfg: cfg.graph.weights.data, config, weight_data)
        return jnp.sum(solve_fn(changed).ys)

    gradient = jax.grad(loss)(config.graph.weights.data)
    assert gradient.shape == WEIGHTS.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_network_compute_coupling_inputs_enforces_prepared_topology():
    network, _solve_fn, _config = _instant_setup()
    coupling_data, coupling_state = network.prepare(0.1, 0.0, 0.2)
    network.graph.weights = _reordered(network.graph.weights)

    with pytest.raises(Exception, match="Graph topology changed after prepare"):
        network.compute_coupling_inputs(
            0.0,
            jnp.zeros((1, 3)),
            coupling_data,
            coupling_state,
        )


def test_subspace_inner_graph_has_its_own_prepared_topology_check():
    regional_indices = jnp.array([[0, 1], [1, 0]])
    regional_weights = BCOO((jnp.array([1.0, 2.0]), regional_indices), shape=(2, 2))
    regional_delays = BCOO((jnp.array([0.1, 0.2]), regional_indices), shape=(2, 2))
    regional_graph = SparseDelayGraph(regional_weights, regional_delays)
    coupling = SubspaceCoupling(
        inner_coupling=DelayedLinearCoupling(incoming_states="S"),
        region_mapping=jnp.array([0, 0, 1, 1]),
        regional_graph=regional_graph,
    )
    network = Network(
        ReducedWongWang(),
        {"delayed": coupling},
        DenseGraph(jnp.zeros((4, 4))),
    )
    solve_fn, config = prepare(network, Euler(), t1=0.2, dt=0.1)

    coupling.regional_graph.weights = _reordered(regional_graph.weights)
    coupling.regional_graph.delays = _reordered(regional_graph.delays)

    with pytest.raises(Exception, match="Graph topology changed after prepare"):
        solve_fn(config)


def test_space_parallel_execution_batches_only_sparse_weight_data():
    _network, solve_fn, config = _instant_setup()
    weight_axis = DataAxis(
        jnp.stack([config.graph.weights.data, config.graph.weights.data * 2.0])
    )
    swept = eqx.tree_at(lambda cfg: cfg.graph.weights.data, config, weight_axis)
    space = Space(swept)

    assert space.axis_state.graph.weights.indices is None
    assert space.static_state.graph.weights.data is None
    assert space.static_state.graph.weights.indices.shape == INDICES.shape
    assert (
        space.static_state._internal.coupling.instant._prepared_topology.edge_indices.shape
        == INDICES.shape
    )

    execution = ParallelExecution(
        lambda cfg: solve_fn(cfg).ys[-1].sum(),
        space,
        n_vmap=1,
        n_pmap=1,
    )
    mapped_leaves = jax.tree.leaves(execution.diff_state)
    assert len(mapped_leaves) == 1
    assert mapped_leaves[0].shape == (1, 2, 3)

    results = np.asarray([float(value) for value in execution.run()])
    assert results.shape == (2,)
    assert results[0] != results[1]


def test_space_parallel_execution_supports_aligned_weight_and_delay_data():
    _network, solve_fn, config = _delayed_setup()
    weight_axis = DataAxis(
        jnp.stack([config.graph.weights.data, config.graph.weights.data * 1.5]),
        group="edge-data",
    )
    delay_axis = DataAxis(
        jnp.stack([config.graph.delays.data, config.graph.delays.data * 0.5]),
        group="edge-data",
    )
    swept = eqx.tree_at(
        lambda cfg: (cfg.graph.weights.data, cfg.graph.delays.data),
        config,
        (weight_axis, delay_axis),
    )
    space = Space(swept)
    execution = ParallelExecution(
        lambda cfg: solve_fn(cfg).ys[-1].sum(),
        space,
        n_vmap=1,
        n_pmap=1,
    )

    assert len(jax.tree.leaves(execution.diff_state)) == 2
    results = np.asarray([float(value) for value in execution.run()])
    assert results.shape == (2,)
    assert np.all(np.isfinite(results))


@pytest.mark.parametrize("axis_location", ["weights", "delays", "prepared"])
def test_space_rejects_axes_on_current_or_prepared_indices(axis_location):
    index_axis = DataAxis(jnp.stack([INDICES, INDICES[::-1]]))
    if axis_location == "delays":
        _network, _solve_fn, config = _delayed_setup()
        invalid = eqx.tree_at(lambda cfg: cfg.graph.delays.indices, config, index_axis)
    else:
        _network, _solve_fn, config = _instant_setup()
        if axis_location == "weights":
            invalid = eqx.tree_at(
                lambda cfg: cfg.graph.weights.indices, config, index_axis
            )
        else:
            invalid = eqx.tree_at(
                lambda cfg: (
                    cfg._internal.coupling.instant._prepared_topology.edge_indices
                ),
                config,
                index_axis,
            )

    with pytest.raises(ValueError, match="not current or prepared topology indices"):
        Space(invalid)
