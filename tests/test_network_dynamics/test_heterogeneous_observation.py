"""Graph-order observation and reduction contracts for heterogeneous networks."""

import warnings
from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.extend import core as jax_core

from tvboptim.execution import ParallelExecution
from tvboptim.experimental.network_dynamics import (
    Bunch,
    DenseGraph,
    GroupObservation,
    HeterogeneousNetwork,
    NodeGroup,
    Readout,
    SignalRoute,
    prepare,
    solve,
)
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import Euler
from tvboptim.observations.observation import compute_fc, welford_cov
from tvboptim.observations.tvb_monitors import (
    BalloonWindkesselBold,
    FirstOrderVolterraHRFKernel,
    HRFBold,
    HRFKernel,
    SubSampling,
    TemporalAverage,
    streaming_hrf_bold,
)
from tvboptim.types import (
    DataAxis,
    Parameter,
    Space,
    combine_state,
    partition_state,
)


class VoiDynamics(AbstractDynamics):
    STATE_NAMES = ("x", "y")
    AUXILIARY_NAMES = ("sum",)
    VARIABLES_OF_INTEREST = ("y", "sum")
    INITIAL_STATE = (0.0, 0.0)
    DEFAULT_PARAMS = Bunch(rate=0.2)

    def dynamics(self, t, state, params, coupling, external):
        del t, coupling, external
        derivatives = params.rate * jnp.stack((state[1], -state[0]))
        return derivatives, state[0:1] + state[1:2]


class CoupledVoiDynamics(VoiDynamics):
    COUPLING_INPUTS = {"drive": 1}


class CurrentSampleHRFKernel(HRFKernel):
    """Two-tap identity kernel for transparent heterogeneous tests."""

    duration: float = eqx.field(static=True, default=0.2)

    def __call__(self, t, downsample_dt):
        del downsample_dt
        return jnp.zeros_like(t).at[0].set(1.0)


def _network(*, noise=False):
    weights = jnp.zeros((4, 4))
    return HeterogeneousNetwork(
        graph=DenseGraph(weights),
        groups={
            "a": NodeGroup(
                VoiDynamics(),
                [0, 2],
                initial_state=jnp.array([[0.2, 0.7], [1.0, -0.3]]),
                noise=(
                    AdditiveNoise(sigma=0.03, key=jax.random.key(7)) if noise else None
                ),
            ),
            "b": NodeGroup(
                VoiDynamics(rate=-0.15),
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


def test_observation_readout_sees_voi_not_state():
    def voi_sum(voi, params):
        del params
        return voi[1:2]

    network = _network()
    grouped = solve(network, Euler(), t1=0.3, dt=0.1)
    observation = GroupObservation({"a": voi_sum, "b": voi_sum}, channels=("sum",))
    observed = solve(network, Euler(), t1=0.3, dt=0.1, observe=observation)
    assert jnp.allclose(observed.sel("sum"), grouped.to_graph("sum"))


def test_wrapped_readout_reads_mismatches_name_both_inputs():
    def first(values, params):
        del params
        return values[0:1]

    observation = GroupObservation(
        {
            "a": Readout(first, name="first", reads="state"),
            "b": "y",
        },
        channels=("activity",),
    )
    with pytest.raises(ValueError, match=r"reads='state'.*provides reads='voi'"):
        prepare(_network(), Euler(), observe=observation)

    route_network = HeterogeneousNetwork(
        graph=DenseGraph(jnp.zeros((2, 2))),
        groups={
            "a": NodeGroup(CoupledVoiDynamics(), [0]),
            "b": NodeGroup(CoupledVoiDynamics(), [1]),
        },
        routes={
            "bad": SignalRoute(
                source={"a": Readout(first, reads="voi")},
                coupling=LinearCoupling(),
                target={"a": "drive"},
            )
        },
    )
    with pytest.raises(ValueError, match=r"reads='voi'.*provides reads='state'"):
        prepare(route_network, Euler())


def test_observation_readout_shape_error_names_params_mapping():
    def needs_gain(voi, params):
        return params.gain * voi[0:1]

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
    def scaled(voi, params):
        return params.gain * voi[0:1]

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
    def scaled(voi, params):
        return params.gain * voi[0:1]

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
    with pytest.raises(ValueError, match="reads"):
        Readout(lambda values, params: values, reads="raw")
    with pytest.raises(ValueError, match="non-empty"):
        Readout(lambda values, params: values, name="")


def test_reduce_warning_can_be_promoted_after_imports():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        prepare(_network(), Euler(), observe=_observe_y())


@pytest.mark.parametrize(
    "monitor",
    [
        SubSampling(period=0.2, voi=0),
        TemporalAverage(period=0.2, voi=0),
        BalloonWindkesselBold(period=0.2, dt_bw=0.1, voi=0),
        HRFBold(
            k_1=1.0,
            V_0=1.0,
            period=0.2,
            downsample_period=0.1,
            voi=0,
            kernel=CurrentSampleHRFKernel(),
        ),
    ],
)
def test_group_observation_then_temporal_monitor_matches_posthoc(monitor):
    observation = _observe_y()
    kwargs = dict(t0=1.0, t1=1.9, dt=0.1)
    projected = solve(_network(), Euler(block_size=4), observe=observation, **kwargs)
    expected = monitor(projected)
    actual = solve(
        _network(),
        Euler(block_size=4),
        observe=(observation, monitor),
        **kwargs,
    )

    assert jnp.allclose(actual.ys, expected.ys, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(actual.ts, expected.ts)
    assert actual.dt == expected.dt == 0.2
    assert actual.variable_names == expected.variable_names


@pytest.mark.parametrize("inject_noise", [False, True])
def test_heterogeneous_temporal_monitor_matches_posthoc_with_noise(inject_noise):
    observation = _observe_y()
    monitor = SubSampling(period=0.2)
    kwargs = dict(t1=0.8, dt=0.1)
    projected_fn, projected_config = prepare(
        _network(noise=True), Euler(block_size=4), observe=observation, **kwargs
    )
    observed_fn, observed_config = prepare(
        _network(noise=True),
        Euler(block_size=4),
        observe=(observation, monitor),
        **kwargs,
    )
    if inject_noise:
        samples = jax.random.normal(jax.random.key(11), (8, 2, 2))
        projected_config._internal.noise_samples.a = samples
        observed_config._internal.noise_samples.a = samples

    expected = monitor(projected_fn(projected_config))
    actual = observed_fn(observed_config)
    assert jnp.array_equal(actual.ys, expected.ys)
    assert jnp.array_equal(actual.ts, expected.ts)


def test_temporal_monitor_selects_common_readout_channels_by_name_order():
    observation = GroupObservation(
        {"a": ("y", "sum"), "b": ("y", "sum")},
        channels=("activity", "total"),
    )
    monitor = SubSampling(period=0.2, voi=1)
    projected = solve(
        _network(), Euler(block_size=4), t1=0.8, dt=0.1, observe=observation
    )
    actual = solve(
        _network(),
        Euler(block_size=4),
        t1=0.8,
        dt=0.1,
        observe=(observation, monitor),
    )

    assert jnp.array_equal(actual.ys, monitor(projected).ys)
    assert actual.variable_names == ("total",)


def test_heterogeneous_temporal_observe_validates_tuple_roles():
    observation = _observe_y()
    monitor = SubSampling(period=0.2)

    with pytest.raises(TypeError, match="GroupObservation, temporal_monitor"):
        prepare(_network(), Euler(block_size=2), observe=(observation,))
    with pytest.raises(TypeError, match="first.*GroupObservation"):
        prepare(_network(), Euler(block_size=2), observe=(monitor, observation))
    with pytest.raises(TypeError, match="second.*temporal monitor"):
        prepare(_network(), Euler(block_size=2), observe=(observation, None))
    with pytest.raises(TypeError, match="Heterogeneous observe must"):
        prepare(_network(), Euler(block_size=2), observe=monitor)
    with pytest.raises(TypeError, match="Unsupported temporal observation"):
        prepare(
            _network(),
            Euler(block_size=2),
            observe=(observation, object()),
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        prepare(
            _network(),
            Euler(block_size=2),
            observe=(observation, monitor),
            reduce=welford_cov(),
        )


def test_temporal_monitor_requires_coverage_unless_fill_is_explicitly_allowed():
    partial = GroupObservation({"a": "y"}, channels=("activity",))
    monitor = SubSampling(period=0.2)
    with pytest.raises(ValueError, match=r"temporal monitor.*uncovered nodes"):
        prepare(
            _network(),
            Euler(block_size=2),
            observe=(partial, monitor),
        )

    allowed = GroupObservation(
        {"a": "y"},
        channels=("activity",),
        fill_value=-7.0,
        allow_partial_coverage=True,
    )
    result = solve(
        _network(),
        Euler(block_size=2),
        t1=0.4,
        dt=0.1,
        observe=(allowed, monitor),
    )
    assert jnp.all(result.ys[:, 0, jnp.array([1, 3])] == -7.0)


def _scaled_observation():
    def scaled(voi, params):
        return params.gain * voi[0:1]

    return GroupObservation(
        {"a": scaled, "b": scaled},
        params={"a": Bunch(gain=1.0), "b": Bunch(gain=1.0)},
        channels=("activity",),
    )


def _identity_hrf(*, k_1=1.0):
    return HRFBold(
        k_1=k_1,
        V_0=1.0,
        period=0.2,
        downsample_period=0.1,
        voi=0,
        kernel=CurrentSampleHRFKernel(),
    )


def test_heterogeneous_network_readout_and_monitor_gradients_match_posthoc():
    observation = _scaled_observation()
    monitor = _identity_hrf()
    kwargs = dict(t1=0.8, dt=0.1)
    projected_fn, projected_config = prepare(
        _network(), Euler(block_size=2), observe=observation, **kwargs
    )
    observed_fn, observed_config = prepare(
        _network(),
        Euler(block_size=2),
        observe=(observation, monitor),
        **kwargs,
    )

    def posthoc_loss(rate, gain, scaling):
        current = projected_config.copy()
        current.groups.a.dynamics.rate = rate
        current.observation.a.gain = gain
        return jnp.sum(_identity_hrf(k_1=scaling)(projected_fn(current)).ys ** 2)

    def observed_loss(rate, gain, scaling):
        current = observed_config.copy()
        current.groups.a.dynamics.rate = rate
        current.observation.a.gain = gain
        current.monitor.k_1 = scaling
        return jnp.sum(observed_fn(current).ys ** 2)

    arguments = (jnp.array(0.25), jnp.array(1.2), jnp.array(0.8))
    expected = jax.value_and_grad(posthoc_loss, argnums=(0, 1, 2))(*arguments)
    actual = jax.value_and_grad(observed_loss, argnums=(0, 1, 2))(*arguments)
    assert jnp.allclose(actual[0], expected[0])
    assert all(
        jnp.allclose(actual_grad, expected_grad)
        for actual_grad, expected_grad in zip(actual[1], expected[1])
    )


def test_heterogeneous_temporal_parameter_namespaces_partition_independently():
    model, config = prepare(
        _network(),
        Euler(block_size=2),
        t1=0.8,
        dt=0.1,
        observe=(_scaled_observation(), _identity_hrf()),
    )
    config.groups.a.dynamics.rate = Parameter(config.groups.a.dynamics.rate)
    config.observation.a.gain = Parameter(config.observation.a.gain)
    config.monitor.k_1 = Parameter(config.monitor.k_1)
    parameters, fixed = partition_state(config)

    gradients = jax.grad(
        lambda current: jnp.sum(model(combine_state(current, fixed)).ys ** 2)
    )(parameters)

    assert jnp.isfinite(gradients.groups.a.dynamics.rate.value)
    assert jnp.isfinite(gradients.observation.a.gain.value)
    assert jnp.isfinite(gradients.monitor.k_1.value)
    assert gradients.groups.a.dynamics.rate.value != 0.0
    assert gradients.observation.a.gain.value != 0.0
    assert gradients.monitor.k_1.value != 0.0


def test_heterogeneous_readout_and_monitor_params_work_with_space():
    model, config = prepare(
        _network(),
        Euler(block_size=2),
        t1=0.4,
        dt=0.1,
        observe=(_scaled_observation(), _identity_hrf()),
    )
    swept = config.copy()
    case = "case"
    swept.observation.a.gain = DataAxis(jnp.array([0.5, 1.5]), group=case)
    swept.monitor.k_1 = DataAxis(jnp.array([0.8, 1.2]), group=case)
    execution = ParallelExecution(
        lambda current: model(current).ys[-1, 0, 0],
        Space(swept, mode="zip"),
        n_vmap=2,
        n_pmap=1,
    )
    values = jnp.asarray(execution.run())

    expected = []
    for gain, scaling in zip((0.5, 1.5), (0.8, 1.2)):
        current = config.copy()
        current.observation.a.gain = gain
        current.monitor.k_1 = scaling
        expected.append(model(current).ys[-1, 0, 0])
    assert jnp.allclose(values, jnp.asarray(expected))


def test_heterogeneous_temporal_jaxpr_has_only_block_sized_common_signal():
    model, config = prepare(
        _network(),
        Euler(block_size=2),
        t1=0.8,
        dt=0.1,
        observe=(_observe_y(), SubSampling(period=0.2)),
    )
    shapes = _jaxpr_shapes(jax.make_jaxpr(model)(config))

    assert (8, 1, 4) not in shapes
    assert (2, 1, 4) in shapes
    assert model(config).ys.shape == (4, 1, 4)
