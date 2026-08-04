"""Graph-order observation and reduction contracts for heterogeneous networks."""

import warnings
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest
from jax.extend import core as jax_core

from tvboptim.execution import ParallelExecution
from tvboptim.experimental.network_dynamics import (
    Bunch,
    DenseGraph,
    DynamicsGroup,
    GroupObservation,
    HeterogeneousNetwork,
    Readout,
    SignalRoute,
    prepare,
    solve,
)
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.solvers import Euler
from tvboptim.observations.observation import compute_fc, welford_cov
from tvboptim.observations.tvb_monitors import (
    FirstOrderVolterraHRFKernel,
    HRFBold,
    SubSampling,
    streaming_hrf_bold,
)
from tvboptim.types import DataAxis, Space


class RecordedDynamics(AbstractDynamics):
    STATE_NAMES = ("x", "y")
    AUXILIARY_NAMES = ("sum",)
    VARIABLES_OF_INTEREST = ("y", "sum")
    INITIAL_STATE = (0.0, 0.0)
    DEFAULT_PARAMS = Bunch(rate=0.2)

    def dynamics(self, t, state, params, coupling, external):
        del t, coupling, external
        derivatives = params.rate * jnp.stack((state[1], -state[0]))
        return derivatives, state[0:1] + state[1:2]


class CoupledRecordedDynamics(RecordedDynamics):
    COUPLING_INPUTS = {"drive": 1}


def _network():
    weights = jnp.zeros((4, 4))
    return HeterogeneousNetwork(
        graph=DenseGraph(weights),
        groups={
            "a": DynamicsGroup(
                RecordedDynamics(),
                [0, 2],
                initial_state=jnp.array([[0.2, 0.7], [1.0, -0.3]]),
            ),
            "b": DynamicsGroup(
                RecordedDynamics(rate=-0.15),
                [1, 3],
                initial_state=jnp.array([[-0.4, 0.9], [0.5, 0.1]]),
            ),
        },
    )


def _observe_y(**kwargs):
    return GroupObservation({"a": "y", "b": "y"}, channels=("activity",), **kwargs)


def test_observation_projects_to_graph_order_matching_to_graph():
    network = _network()
    grouped = solve(network, Euler(), t1=0.5, dt=0.1)
    observed = solve(network, Euler(), t1=0.5, dt=0.1, observe=_observe_y())

    assert observed.variable_names == ("activity",)
    assert jnp.allclose(observed.sel("activity"), grouped.to_graph("y"))


def test_unobserved_nodes_take_fill_value_without_reduce():
    observation = GroupObservation({"a": "y"}, channels=("activity",), fill_value=-7.0)
    observed = solve(_network(), Euler(), t1=0.2, dt=0.1, observe=observation)
    assert jnp.all(observed.ys[:, 0, jnp.array([1, 3])] == -7.0)


def test_observation_readout_sees_recorded_not_state():
    def recorded_sum(recorded, params):
        del params
        return recorded[1:2]

    network = _network()
    grouped = solve(network, Euler(), t1=0.3, dt=0.1)
    observation = GroupObservation(
        {"a": recorded_sum, "b": recorded_sum}, channels=("sum",)
    )
    observed = solve(network, Euler(), t1=0.3, dt=0.1, observe=observation)
    assert jnp.allclose(observed.sel("sum"), grouped.to_graph("sum"))


def test_wrapped_readout_space_mismatches_name_both_spaces():
    def first(values, params):
        del params
        return values[0:1]

    observation = GroupObservation(
        {
            "a": Readout(first, name="first", space="state"),
            "b": "y",
        },
        channels=("activity",),
    )
    with pytest.raises(ValueError, match=r"space='state'.*reads 'recorded'"):
        prepare(_network(), Euler(), observe=observation)

    route_network = HeterogeneousNetwork(
        graph=DenseGraph(jnp.zeros((2, 2))),
        groups={
            "a": DynamicsGroup(CoupledRecordedDynamics(), [0]),
            "b": DynamicsGroup(CoupledRecordedDynamics(), [1]),
        },
        routes={
            "bad": SignalRoute(
                source={"a": Readout(first, space="recorded")},
                coupling=LinearCoupling(),
                target={"a": "drive"},
            )
        },
    )
    with pytest.raises(ValueError, match=r"space='recorded'.*reads 'state'"):
        prepare(route_network, Euler())


def test_observation_readout_shape_error_names_params_mapping():
    def needs_gain(recorded, params):
        return params.gain * recorded[0:1]

    observation = GroupObservation({"a": needs_gain, "b": "y"}, channels=("activity",))
    with pytest.raises(ValueError, match=r"params\['a'\].*empty"):
        prepare(_network(), Euler(), observe=observation)


def test_observation_channels_match_width_and_are_validated():
    with pytest.raises(ValueError, match="unique"):
        GroupObservation({"a": ("y", "sum")}, channels=("x", "x"))
    with pytest.raises(ValueError, match="non-empty"):
        GroupObservation({"a": "y"}, channels=("",))

    observation = GroupObservation(
        {"a": ("y", "sum"), "b": ("y", "sum")}, channels=("only_one",)
    )
    with pytest.raises(ValueError, match=r"width Q=2"):
        prepare(_network(), Euler(), observe=observation)


def test_reduce_requires_observation_and_exhaustive_coverage_by_default():
    with pytest.raises(ValueError, match="GroupObservation"):
        prepare(_network(), Euler(block_size=2), reduce=welford_cov())

    partial = GroupObservation({"a": "y"}, channels=("activity",))
    with pytest.raises(ValueError, match=r"uncovered nodes \[1, 3\].*groups \['b'\]"):
        prepare(
            _network(),
            Euler(block_size=2),
            reduce=welford_cov(),
            observe=partial,
        )


def test_allow_partial_coverage_opt_in_permits_reduction():
    observation = GroupObservation(
        {"a": "y"},
        channels=("activity",),
        allow_partial_coverage=True,
    )
    result = solve(
        _network(),
        Euler(block_size=2),
        t1=0.4,
        dt=0.1,
        observe=observation,
        reduce=welford_cov(),
    )
    assert result.shape == (4, 4)


def test_welford_cov_matches_posthoc_observed_trajectory():
    network = _network()
    observation = _observe_y()
    trajectory = solve(
        network, Euler(block_size=2), t1=1.0, dt=0.1, observe=observation
    )
    reduced = solve(
        network,
        Euler(block_size=2),
        t1=1.0,
        dt=0.1,
        observe=observation,
        reduce=welford_cov(),
    )
    assert jnp.allclose(reduced, compute_fc(trajectory), atol=1e-5)


def test_reduce_without_block_size_warns_but_remains_valid():
    with pytest.warns(UserWarning, match="full trajectory is materialized"):
        solve(
            _network(),
            Euler(),
            t1=0.4,
            dt=0.1,
            observe=_observe_y(),
            reduce=welford_cov(),
        )


def _nested_jaxprs(value, active=None):
    if active is None:
        active = set()
    value_id = id(value)
    if value_id in active:
        return
    if not isinstance(
        value, (jax_core.Jaxpr, jax_core.ClosedJaxpr, Mapping, tuple, list)
    ):
        return
    active.add(value_id)
    try:
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


def _jaxpr_shapes(closed):
    shapes = set()
    for jaxpr in _nested_jaxprs(closed):
        variables = [*jaxpr.constvars, *jaxpr.invars, *jaxpr.outvars]
        for equation in jaxpr.eqns:
            variables.extend(equation.invars)
            variables.extend(equation.outvars)
        for variable in variables:
            shape = getattr(getattr(variable, "aval", None), "shape", None)
            if shape is not None:
                shapes.add(tuple(shape))
    return shapes


def test_reduce_bounds_trajectory_memory_only_with_block_size():
    target_shape = (6, 1, 4)
    blocked_fn, blocked_config = prepare(
        _network(),
        Euler(block_size=2),
        t1=0.6,
        dt=0.1,
        observe=_observe_y(),
        reduce=welford_cov(),
    )
    blocked_shapes = _jaxpr_shapes(jax.make_jaxpr(blocked_fn)(blocked_config))
    assert target_shape not in blocked_shapes

    with pytest.warns(UserWarning, match="full trajectory"):
        full_fn, full_config = prepare(
            _network(),
            Euler(),
            t1=0.6,
            dt=0.1,
            observe=_observe_y(),
            reduce=welford_cov(),
        )
    full_shapes = _jaxpr_shapes(jax.make_jaxpr(full_fn)(full_config))
    assert target_shape in full_shapes


def test_streaming_hrf_bold_runs_on_heterogeneous_observation():
    dt = 0.1
    monitor = HRFBold(
        period=0.2,
        downsample_period=0.1,
        downsample=SubSampling(period=0.1),
        kernel=FirstOrderVolterraHRFKernel(duration=0.4),
    )
    trajectory = solve(
        _network(),
        Euler(block_size=2),
        t1=0.4,
        dt=dt,
        observe=_observe_y(),
    )
    reduced = solve(
        _network(),
        Euler(block_size=2),
        t1=0.4,
        dt=dt,
        observe=_observe_y(),
        reduce=streaming_hrf_bold(monitor, dt),
    )
    reference = monitor(trajectory)
    assert reduced.shape == reference.ys.shape
    assert jnp.allclose(reduced, reference.ys, atol=1e-6)


def test_observation_params_are_live_differentiable_and_vmappable():
    def scaled(recorded, params):
        return params.gain * recorded[0:1]

    observation = GroupObservation(
        {"a": scaled, "b": scaled},
        params={"a": Bunch(gain=1.0), "b": Bunch(gain=1.0)},
        channels=("scaled",),
    )
    solve_fn, config = prepare(_network(), Euler(), t1=0.3, dt=0.1, observe=observation)

    def total(gain):
        local = config.copy()
        local.observation.a.gain = gain
        local.observation.b.gain = gain
        return jnp.sum(solve_fn(local).ys)

    gradient = jax.grad(total)(jnp.array(1.0))
    batched = jax.vmap(total)(jnp.array([0.5, 1.0, 1.5]))
    assert jnp.isfinite(gradient)
    assert batched.shape == (3,)
    assert jnp.allclose(batched[1], 2.0 * batched[0])


def test_observation_params_work_with_space():
    def scaled(recorded, params):
        return params.gain * recorded[0:1]

    observation = GroupObservation(
        {"a": scaled, "b": "y"},
        params={"a": Bunch(gain=1.0)},
        channels=("scaled",),
    )
    solve_fn, config = prepare(_network(), Euler(), t1=0.2, dt=0.1, observe=observation)
    swept = config.copy()
    swept.observation.a.gain = DataAxis(jnp.array([0.5, 1.5]))
    execution = ParallelExecution(
        lambda cfg: solve_fn(cfg).ys[-1, 0, 0],
        Space(swept, mode="product"),
        n_vmap=2,
        n_pmap=1,
    )
    values = jnp.asarray(execution.run())
    assert values.shape == (2,)
    assert jnp.allclose(values[1], 3.0 * values[0])


def test_readout_constructor_validation():
    with pytest.raises(TypeError, match="callable"):
        Readout(3)
    with pytest.raises(ValueError, match="space"):
        Readout(lambda values, params: values, space="raw")
    with pytest.raises(ValueError, match="non-empty"):
        Readout(lambda values, params: values, name="")


def test_reduce_warning_can_be_promoted_after_imports():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        prepare(_network(), Euler(), observe=_observe_y())
