"""Lowering-level performance invariants for heterogeneous routing."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.extend import core as jax_core

from tvboptim.experimental.network_dynamics import (
    Bunch,
    DenseGraph,
    DynamicsGroup,
    HeterogeneousNetwork,
    SignalRoute,
    SparseGraph,
    prepare,
)
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.solvers import Heun


class DrivenLinear(AbstractDynamics):
    STATE_NAMES = ("x",)
    INITIAL_STATE = (0.1,)
    DEFAULT_PARAMS = Bunch(decay=0.2)
    COUPLING_INPUTS = {"drive": 1}

    def dynamics(self, t, state, params, coupling, external):
        del t, external
        return -params.decay * state + coupling.drive


def _network(n_nodes, n_groups, sparse=False):
    row = jnp.arange(n_nodes, dtype=int)
    col = (row + 1) % n_nodes
    graph = (
        SparseGraph.from_coo(jnp.ones(n_nodes), row, col, (n_nodes, n_nodes))
        if sparse
        else DenseGraph(jnp.eye(n_nodes, k=1) + jnp.eye(n_nodes, k=1 - n_nodes))
    )
    groups = {}
    source = {}
    target = {}
    for index in range(n_groups):
        name = f"g{index}"
        nodes = tuple(range(index, n_nodes, n_groups))
        groups[name] = DynamicsGroup(DrivenLinear(), nodes)
        source[name] = "x"
        target[name] = "drive"
    return HeterogeneousNetwork(
        graph=graph,
        groups=groups,
        routes={
            "activity": SignalRoute(
                source=source,
                coupling=LinearCoupling(G=0.3),
                target=target,
            )
        },
    )


def _nested_jaxprs(value, active=None):
    """Yield encountered Jaxprs without following representation cycles.

    JAX versions differ in how nested and closed Jaxprs are exposed. In
    particular, their runtime type checks can overlap, and wrapper references
    may be cyclic. Check the concrete Jaxpr case first and guard only the
    active traversal path so a Jaxpr used at two genuine call sites is still
    counted twice.
    """
    if active is None:
        active = set()
    value_id = id(value)
    if value_id in active:
        return

    traversable = isinstance(
        value, (jax_core.Jaxpr, jax_core.ClosedJaxpr, Mapping, tuple, list)
    )
    if not traversable:
        return

    active.add(value_id)
    try:
        # This order matters on JAX versions where the two runtime type checks
        # overlap: treating a Jaxpr as a ClosedJaxpr can follow .jaxpr to self.
        if isinstance(value, jax_core.Jaxpr):
            yield value
            for equation in value.eqns:
                yield from _nested_jaxprs(equation.params, active)
        elif isinstance(value, jax_core.ClosedJaxpr):
            yield from _nested_jaxprs(value.jaxpr, active)
        elif isinstance(value, Mapping):
            for child in value.values():
                yield from _nested_jaxprs(child, active)
        else:
            for child in value:
                yield from _nested_jaxprs(child, active)
    finally:
        active.remove(value_id)


def _all_jaxprs(closed):
    return tuple(_nested_jaxprs(closed))


def _primitive_count(closed, primitive_name):
    return sum(
        equation.primitive.name == primitive_name
        for jaxpr in _all_jaxprs(closed)
        for equation in jaxpr.eqns
    )


def _equation_count(closed):
    return sum(len(jaxpr.eqns) for jaxpr in _all_jaxprs(closed))


def _trace(n_nodes, n_groups, *, sparse=False, per_stage=False):
    solve_fn, config = prepare(
        _network(n_nodes, n_groups, sparse=sparse),
        Heun(recompute_coupling_per_stage=per_stage),
        t1=0.2,
        dt=0.1,
    )
    return jax.make_jaxpr(solve_fn)(config)


def test_one_route_lowers_to_one_transport_not_group_pair_transports():
    frozen = _trace(16, 4, per_stage=False)
    per_stage = _trace(16, 4, per_stage=True)
    assert _primitive_count(frozen, "dot_general") == 1
    assert _primitive_count(per_stage, "dot_general") == 2


def test_compile_structure_does_not_grow_with_nodes_per_group():
    small = _trace(16, 4)
    large = _trace(64, 4)
    assert _equation_count(small) == _equation_count(large)
    assert _primitive_count(small, "dot_general") == _primitive_count(
        large, "dot_general"
    )


def test_sparse_route_has_no_square_node_intermediate():
    n_nodes = 31
    closed = _trace(n_nodes, 4, sparse=True)
    for jaxpr in _all_jaxprs(closed):
        for variable in (*jaxpr.invars, *jaxpr.outvars):
            shape = getattr(getattr(variable, "aval", None), "shape", ())
            assert shape != (n_nodes, n_nodes)
