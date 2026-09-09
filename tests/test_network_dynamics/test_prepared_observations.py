"""Prepared temporal observations executed within native solver blocks."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.extend import core as jax_core

from tvboptim.execution import ParallelExecution
from tvboptim.experimental.network_dynamics import Network, prepare, solve
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.core.observation import (
    PreparedObservation,
    SimulationGrid,
    prepare_observation,
)
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.external_input import ConstantInput
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.solvers import DiffraxSolver, Euler
from tvboptim.observations.observation import welford_cov
from tvboptim.observations.tvb_monitors import (
    BalloonWindkesselBold,
    HRFBold,
    HRFKernel,
    SubSampling,
    TemporalAverage,
)
from tvboptim.types import (
    DataAxis,
    Parameter,
    Space,
    combine_state,
    partition_state,
)


class RampDynamics(AbstractDynamics):
    """Two exactly linear channels that make sample selection transparent."""

    STATE_NAMES = ("slow", "fast")
    INITIAL_STATE = (0.0, 10.0)
    DEFAULT_PARAMS = Bunch(rate=1.0)

    def dynamics(self, t, state, params, coupling, external):
        del t, coupling, external
        rates = jnp.asarray((params.rate, 2.0 * params.rate), dtype=state.dtype)
        return jnp.broadcast_to(rates[:, None], state.shape)


class Float64AuxDynamics(AbstractDynamics):
    """Float32 carry with a deliberately promoted selected auxiliary."""

    STATE_NAMES = ("state",)
    AUXILIARY_NAMES = ("drive",)
    VARIABLES_OF_INTEREST = ("drive",)
    INITIAL_STATE = (jnp.float32(0.25),)

    def dynamics(self, t, state, params, coupling, external):
        del t, params, coupling, external
        return jnp.zeros_like(state), (state + 0.5).astype(jnp.float64)


class ExternalAuxDynamics(AbstractDynamics):
    """Expose an external input as an auxiliary beside a float32 state."""

    STATE_NAMES = ("state",)
    AUXILIARY_NAMES = ("drive",)
    VARIABLES_OF_INTEREST = ("state", "drive")
    INITIAL_STATE = (jnp.float32(0.25),)
    EXTERNAL_INPUTS = {"stimulus": 1}

    def dynamics(self, t, state, params, coupling, external):
        del t, params, coupling
        return jnp.zeros_like(state), external.stimulus


class CouplingAuxDynamics(AbstractDynamics):
    """Expose a coupling input as an auxiliary beside a float32 state."""

    STATE_NAMES = ("state",)
    AUXILIARY_NAMES = ("drive",)
    VARIABLES_OF_INTEREST = ("state", "drive")
    INITIAL_STATE = (jnp.float32(0.25),)
    COUPLING_INPUTS = {"drive": 1}

    def dynamics(self, t, state, params, coupling, external):
        del t, params, external
        return jnp.zeros_like(state), coupling.drive


class PreviousSampleHRFKernel(HRFKernel):
    """Two-tap kernel whose output makes causal alignment explicit."""

    duration: float = eqx.field(static=True, default=4.0)

    def __call__(self, t, downsample_dt):
        del downsample_dt
        return jnp.zeros_like(t).at[1].set(1.0)


def _posthoc_and_online(*, block_size=4, t0=10.0, t1=12.75, dt=0.25):
    dynamics = RampDynamics()
    solver = Euler(block_size=block_size)
    monitor = SubSampling(period=0.5, voi=1)
    raw = solve(dynamics, solver, t0=t0, t1=t1, dt=dt, n_nodes=3)
    return monitor(raw), solve(
        dynamics,
        solver,
        t0=t0,
        t1=t1,
        dt=dt,
        n_nodes=3,
        observe=monitor,
    )


def test_subsampling_preparation_resolves_static_execution_recipe():
    prepared = prepare_observation(
        SubSampling(period=0.5, voi=1),
        SimulationGrid(t0=10.0, dt=0.25, n_steps=11),
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
        ("slow", "fast"),
    )

    assert isinstance(prepared, PreparedObservation)
    assert prepared.data.stride == 2
    assert prepared.state0 is None
    assert prepared.params == {}
    assert prepared.period == 0.5
    assert prepared.first_sample_offset == 0.5
    assert prepared.variable_names == ("fast",)


def test_online_subsampling_matches_posthoc_through_blocks_and_tail():
    expected, actual = _posthoc_and_online()

    assert actual.ys.shape == (5, 1, 3)
    assert actual.variable_names == ("fast",)
    assert actual.dt == 0.5
    assert jnp.array_equal(actual.ys, expected.ys)
    assert jnp.allclose(actual.ts, expected.ts)


def test_network_native_path_supports_prepared_subsampling():
    network = Network(
        dynamics=RampDynamics(),
        coupling={},
        graph=DenseGraph(jnp.zeros((3, 3))),
    )
    monitor = SubSampling(period=0.5, voi=1)
    raw = solve(network, Euler(block_size=4), t1=2.75, dt=0.25)
    actual = solve(
        network,
        Euler(block_size=4),
        t1=2.75,
        dt=0.25,
        observe=monitor,
    )
    expected = monitor(raw)

    assert jnp.array_equal(actual.ys, expected.ys)
    assert jnp.allclose(actual.ts, expected.ts)
    assert actual.variable_names == ("fast",)


@pytest.mark.parametrize(
    ("period", "expected_values", "expected_times"),
    [
        (4.0, [2.5, 6.5], [102.5, 106.5]),
        (3.0, [2.0, 5.0, 8.0], [102.0, 105.0, 108.0]),
        (1.0, list(range(1, 10)), list(range(101, 110))),
    ],
)
def test_posthoc_temporal_average_uses_complete_integer_windows(
    period, expected_values, expected_times
):
    values = jnp.arange(1.0, 10.0).reshape(9, 1, 1)
    solution = NativeSolution(
        ts=jnp.arange(101.0, 110.0),
        ys=values,
        dt=1.0,
        variable_names=("signal",),
    )

    result = TemporalAverage(period=period)(solution)

    assert jnp.allclose(result.ys[:, 0, 0], jnp.asarray(expected_values))
    assert jnp.allclose(result.ts, jnp.asarray(expected_times))
    assert result.variable_names == ("signal",)


def test_online_temporal_average_matches_corrected_posthoc_blocks_and_tail():
    dynamics = RampDynamics()
    solver = Euler(block_size=4)
    monitor = TemporalAverage(period=0.5, voi=1)
    raw = solve(dynamics, solver, t0=10.0, t1=12.75, dt=0.25, n_nodes=3)
    expected = monitor(raw)
    actual = solve(
        dynamics,
        solver,
        t0=10.0,
        t1=12.75,
        dt=0.25,
        n_nodes=3,
        observe=monitor,
    )

    assert actual.ys.shape == (5, 1, 3)
    assert jnp.array_equal(actual.ys, expected.ys)
    assert jnp.allclose(actual.ts, expected.ts)
    assert jnp.allclose(actual.ts, 10.375 + jnp.arange(5) * 0.5)
    assert actual.variable_names == ("fast",)


def test_temporal_average_exact_gradient_matches_posthoc():
    monitor = TemporalAverage(period=0.5, voi=1)
    raw_fn, raw_config = prepare(
        RampDynamics(), Euler(block_size=4), t1=2.75, dt=0.25, n_nodes=2
    )
    observed_fn, observed_config = prepare(
        RampDynamics(),
        Euler(block_size=4),
        t1=2.75,
        dt=0.25,
        n_nodes=2,
        observe=monitor,
    )

    def loss(model, config, rate, posthoc):
        current = config.copy()
        current.dynamics = config.dynamics.copy()
        current.dynamics.rate = rate
        result = model(current)
        values = monitor(result).ys if posthoc else result.ys
        return jnp.sum(values**2)

    rate = jnp.array(1.3)
    raw_value, raw_grad = jax.value_and_grad(
        lambda value: loss(raw_fn, raw_config, value, True)
    )(rate)
    observed_value, observed_grad = jax.value_and_grad(
        lambda value: loss(observed_fn, observed_config, value, False)
    )(rate)
    assert jnp.allclose(observed_value, raw_value)
    assert jnp.allclose(observed_grad, raw_grad)


def test_temporal_average_drops_an_incomplete_short_run():
    result = solve(
        RampDynamics(),
        Euler(block_size=4),
        t1=0.5,
        dt=0.25,
        n_nodes=2,
        observe=TemporalAverage(period=1.0, voi=1),
    )
    assert result.ys.shape == (0, 1, 2)
    assert result.ts.shape == (0,)


@pytest.mark.parametrize("n_steps", [0, 1])
def test_short_run_returns_safe_empty_solution(n_steps):
    result = solve(
        RampDynamics(),
        Euler(block_size=2),
        t1=n_steps * 0.25,
        dt=0.25,
        n_nodes=3,
        observe=SubSampling(period=0.5, voi=1),
    )

    assert result.ys.shape == (0, 1, 3)
    assert result.ts.shape == (0,)
    assert "t=[]" in repr(result)


def test_observe_without_block_size_warns_and_runs_as_one_chunk():
    with pytest.warns(UserWarning, match="block-based memory management"):
        result = solve(
            RampDynamics(),
            Euler(),
            t0=5.0,
            t1=6.25,
            dt=0.25,
            n_nodes=2,
            observe=SubSampling(period=0.5),
        )

    assert result.ys.shape == (2, 2, 2)
    assert jnp.allclose(result.ts, jnp.array([5.5, 6.0]))


def test_aligned_truncation_preserves_forward_values_with_and_without_blocks():
    monitor = SubSampling(period=0.5, voi=1)
    kwargs = dict(t1=2.75, dt=0.25, n_nodes=2, observe=monitor)
    exact = solve(RampDynamics(), Euler(block_size=4), **kwargs)
    truncated_blocked = solve(
        RampDynamics(), Euler(block_size=4, grad_horizon=8), **kwargs
    )
    with pytest.warns(UserWarning, match="block-based memory management"):
        truncated_unblocked = solve(RampDynamics(), Euler(grad_horizon=4), **kwargs)

    assert jnp.array_equal(truncated_blocked.ys, exact.ys)
    assert jnp.array_equal(truncated_unblocked.ys, exact.ys)
    assert jnp.array_equal(truncated_blocked.ts, exact.ts)
    assert jnp.array_equal(truncated_unblocked.ts, exact.ts)


@pytest.mark.parametrize("monitor_kind", ["bw", "hrf"])
def test_stateful_observation_truncation_matches_detached_combined_carry_reference(
    monitor_kind,
):
    from tvboptim.experimental.network_dynamics.solve import _run_observed_scan

    if monitor_kind == "bw":
        monitor = BalloonWindkesselBold(period=4.0, dt_bw=1.0, voi=0)
        parameter_name = "tauo"
        parameter_value = jnp.array(0.98)
    else:
        monitor = HRFBold(
            period=4.0,
            downsample=SubSampling(period=1.0, voi=0),
            kernel=PreviousSampleHRFKernel(),
        )
        parameter_name = "V_0"
        parameter_value = jnp.array(0.02)

    n_steps = 8
    window_size = 4
    solver = Euler(block_size=4, grad_horizon=window_size)
    scan_inputs = jnp.arange(n_steps, dtype=jnp.float32)
    simulation_state0 = jnp.full((1, 1), 0.2, dtype=jnp.float32)
    prepared = prepare_observation(
        monitor,
        SimulationGrid(t0=0.0, dt=1.0, n_steps=n_steps),
        jax.ShapeDtypeStruct((1, 1), jnp.float32),
        ("drive",),
    )

    def run(rate, parameter, explicit_reference):
        params = prepared.params.copy()
        setattr(params, parameter_name, parameter)

        def op(state, _time):
            next_state = state + rate
            return next_state, next_state

        if not explicit_reference:
            _, chunks = _run_observed_scan(
                op,
                simulation_state0,
                scan_inputs,
                n_steps,
                solver,
                prepared,
                params,
                window_size=window_size,
            )
            return jnp.sum(chunks**2), chunks

        observation_state = (
            prepared.state0
            if prepared.initialize is None
            else prepared.initialize(prepared.data, prepared.state0, params)
        )
        carry = (simulation_state0, observation_state)
        output_chunks = []
        for start in range(0, n_steps, window_size):
            # This is the intended truncated-BPTT definition: both the neural
            # and monitor histories are detached at the same window boundary.
            simulation_state, observation_state = jax.lax.stop_gradient(carry)
            simulation_state, raw = jax.lax.scan(
                op,
                simulation_state,
                scan_inputs[start : start + window_size],
            )
            observation_state, chunk = prepared.update(
                prepared.data,
                observation_state,
                raw,
                params,
            )
            carry = (simulation_state, observation_state)
            output_chunks.append(chunk)
        chunks = jnp.concatenate(output_chunks)
        return jnp.sum(chunks**2), chunks

    arguments = (jnp.array(0.03), parameter_value)
    actual = jax.value_and_grad(
        lambda rate, parameter: run(rate, parameter, False),
        argnums=(0, 1),
        has_aux=True,
    )(*arguments)
    expected = jax.value_and_grad(
        lambda rate, parameter: run(rate, parameter, True),
        argnums=(0, 1),
        has_aux=True,
    )(*arguments)

    (actual_loss, actual_chunks), actual_gradient = actual
    (expected_loss, expected_chunks), expected_gradient = expected
    assert jnp.array_equal(actual_chunks, expected_chunks)
    assert jnp.allclose(actual_loss, expected_loss)
    assert jnp.allclose(actual_gradient[0], expected_gradient[0])
    assert jnp.allclose(actual_gradient[1], expected_gradient[1])


def test_online_value_and_exact_gradient_match_posthoc_sampling():
    monitor = SubSampling(period=0.5, voi=1)
    raw_fn, raw_config = prepare(
        RampDynamics(), Euler(block_size=4), t1=2.75, dt=0.25, n_nodes=2
    )
    observed_fn, observed_config = prepare(
        RampDynamics(),
        Euler(block_size=4),
        t1=2.75,
        dt=0.25,
        n_nodes=2,
        observe=monitor,
    )

    # Use the same squared loss on both paths while leaving sampling post-hoc
    # only for the raw model.
    def raw_loss(rate):
        current = raw_config.copy()
        current.dynamics = raw_config.dynamics.copy()
        current.dynamics.rate = rate
        return jnp.sum(monitor(raw_fn(current)).ys ** 2)

    def observed_loss(rate):
        current = observed_config.copy()
        current.dynamics = observed_config.dynamics.copy()
        current.dynamics.rate = rate
        return jnp.sum(observed_fn(current).ys ** 2)

    rate = jnp.array(1.3)
    assert jnp.allclose(observed_loss(rate), raw_loss(rate))
    assert jnp.allclose(jax.grad(observed_loss)(rate), jax.grad(raw_loss)(rate))


def test_prepared_subsampling_is_jittable_and_vmappable():
    model, config = prepare(
        RampDynamics(),
        Euler(block_size=4),
        t1=2.0,
        dt=0.25,
        n_nodes=2,
        observe=SubSampling(period=0.5, voi=1),
    )

    compiled = jax.jit(model)(config)

    def total(rate):
        current = config.copy()
        current.dynamics = config.dynamics.copy()
        current.dynamics.rate = rate
        return model(current).ys.sum()

    totals = jax.vmap(total)(jnp.array([0.5, 1.0, 1.5]))
    assert compiled.ys.shape == (4, 1, 2)
    assert totals.shape == (3,)
    assert jnp.allclose(totals[1] - totals[0], totals[2] - totals[1])
    assert totals[0] < totals[1] < totals[2]


@pytest.mark.parametrize("injected", [False, True])
def test_stochastic_online_sampling_matches_same_raw_block_grid(injected):
    noise = AdditiveNoise(sigma=0.2, key=jax.random.key(17))
    monitor = SubSampling(period=0.5, voi=1)
    solver = Euler(block_size=4)
    kwargs = dict(
        t1=2.75,
        dt=0.25,
        n_nodes=2,
        noise=noise,
    )
    raw_fn, raw_config = prepare(RampDynamics(), solver, **kwargs)
    observed_fn, observed_config = prepare(
        RampDynamics(), solver, observe=monitor, **kwargs
    )
    if injected:
        samples = jnp.arange(44, dtype=jnp.float32).reshape(11, 2, 2) / 100.0
        raw_config._internal.noise_samples = samples
        observed_config._internal.noise_samples = samples

    expected = monitor(raw_fn(raw_config))
    actual = observed_fn(observed_config)

    assert jnp.array_equal(actual.ys, expected.ys)
    assert jnp.allclose(actual.ts, expected.ts)

    def loss(model, config, rate, posthoc):
        current = config.copy()
        current.dynamics = config.dynamics.copy()
        current.dynamics.rate = rate
        result = model(current)
        values = monitor(result).ys if posthoc else result.ys
        return jnp.mean(values**2)

    rate = jnp.array(1.1)
    raw_grad = jax.grad(lambda value: loss(raw_fn, raw_config, value, True))(rate)
    observed_grad = jax.grad(
        lambda value: loss(observed_fn, observed_config, value, False)
    )(rate)
    assert jnp.allclose(observed_grad, raw_grad)


def test_partitioned_network_parameter_gradient_matches_posthoc_sampling():
    weights = (jnp.ones((3, 3)) - jnp.eye(3)) / 2.0
    network = Network(
        dynamics=ReducedWongWang(),
        coupling={"instant": LinearCoupling(source="S", G=0.1)},
        graph=DenseGraph(weights),
    )
    solver = Euler(block_size=4)
    monitor = SubSampling(period=0.2, voi=0)
    raw_fn, raw_config = prepare(network, solver, t1=2.0, dt=0.1)
    observed_fn, observed_config = prepare(
        network, solver, t1=2.0, dt=0.1, observe=monitor
    )
    raw_config.coupling.instant.G = Parameter(0.1)
    observed_config.coupling.instant.G = Parameter(0.1)
    raw_parameters, raw_fixed = partition_state(raw_config)
    observed_parameters, observed_fixed = partition_state(observed_config)

    def raw_loss(parameters):
        result = raw_fn(combine_state(parameters, raw_fixed))
        return jnp.mean(monitor(result).ys ** 2)

    def observed_loss(parameters):
        result = observed_fn(combine_state(parameters, observed_fixed))
        return jnp.mean(result.ys**2)

    raw_value, raw_grad = jax.value_and_grad(raw_loss)(raw_parameters)
    observed_value, observed_grad = jax.value_and_grad(observed_loss)(
        observed_parameters
    )

    assert jnp.allclose(observed_value, raw_value)
    assert jnp.allclose(
        observed_grad.coupling.instant.G.value,
        raw_grad.coupling.instant.G.value,
    )


def _nested_jaxprs(value, active=None):
    if active is None:
        active = set()
    value_id = id(value)
    if value_id in active:
        return
    if not isinstance(value, (jax_core.Jaxpr, jax_core.ClosedJaxpr, dict, tuple, list)):
        return
    active.add(value_id)
    try:
        if isinstance(value, jax_core.Jaxpr):
            yield value
            for equation in value.eqns:
                yield from _nested_jaxprs(equation.params, active)
        elif isinstance(value, jax_core.ClosedJaxpr):
            yield from _nested_jaxprs(value.jaxpr, active)
        elif isinstance(value, dict):
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


def test_observation_carry_does_not_contain_full_raw_trajectory():
    model, config = prepare(
        RampDynamics(),
        Euler(block_size=4),
        t1=3.0,
        dt=0.25,
        n_nodes=3,
        observe=SubSampling(period=0.5, voi=1),
    )
    shapes = _jaxpr_shapes(jax.make_jaxpr(model)(config))

    assert (12, 2, 3) not in shapes
    assert (4, 2, 3) in shapes
    assert model(config).ys.shape == (6, 1, 3)


def test_regular_block_boundary_must_align():
    with pytest.raises(ValueError, match="block_size=3"):
        prepare(
            RampDynamics(),
            Euler(block_size=3),
            t1=2.0,
            dt=0.25,
            observe=SubSampling(period=0.5),
        )


def test_unblocked_truncation_boundary_must_align():
    with pytest.warns(UserWarning, match="block-based memory management"):
        with pytest.raises(ValueError, match="grad_horizon=3"):
            prepare(
                RampDynamics(),
                Euler(grad_horizon=3),
                t1=2.0,
                dt=0.25,
                observe=SubSampling(period=0.5),
            )


def test_invalid_grid_selection_and_unsupported_combinations_fail_at_prepare():
    with pytest.raises(ValueError, match="integer multiple"):
        prepare(
            RampDynamics(),
            Euler(block_size=4),
            dt=0.25,
            observe=SubSampling(period=0.6),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        prepare(
            RampDynamics(),
            Euler(block_size=4),
            dt=0.25,
            observe=SubSampling(period=0.5, voi=slice(0, 0)),
        )
    with pytest.raises(TypeError, match="Unsupported temporal observation"):
        prepare(RampDynamics(), Euler(block_size=4), observe=object())
    with pytest.raises(ValueError, match="cannot be combined"):
        prepare(
            RampDynamics(),
            Euler(block_size=4),
            observe=SubSampling(period=0.4),
            reduce=welford_cov(),
        )


def test_diffrax_rejects_temporal_observation_explicitly():
    solver = DiffraxSolver(diffrax.Euler())
    with pytest.raises(ValueError, match="only supported by NativeSolver"):
        prepare(
            RampDynamics(),
            solver,
            observe=SubSampling(period=0.5),
        )


def test_posthoc_subsampling_now_preserves_selected_variable_names():
    solution = NativeSolution(
        ts=jnp.arange(1.0, 5.0),
        ys=jnp.zeros((4, 2, 3)),
        dt=1.0,
        variable_names=("first", "second"),
    )
    result = SubSampling(period=2.0, voi=1)(solution)
    assert result.variable_names == ("second",)


def test_balloon_windkessel_preparation_has_four_vector_carry_and_live_params():
    monitor = BalloonWindkesselBold(period=8.0, dt_bw=1.0, voi=1, TE=0.05)
    prepared = prepare_observation(
        monitor,
        SimulationGrid(t0=10.0, dt=1.0, n_steps=19),
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
        ("slow", "fast"),
    )

    assert tuple(value.shape for value in prepared.state0) == ((3,),) * 4
    assert prepared.params.TE == 0.05
    assert prepared.params.Eo == monitor.Eo
    assert prepared.data.repeat == 1
    assert prepared.data.decimate == 1
    assert prepared.data.save_every == 8
    assert prepared.period == 8.0
    assert prepared.first_sample_offset == 8.0
    assert prepared.variable_names == ("BOLD(fast)",)


def test_balloon_windkessel_resampling_uses_repetition_and_window_end_decimation():
    from tvboptim.observations.tvb_monitors.bold import _resample_bw_input

    values = jnp.arange(1.0, 5.0).reshape(4, 1)
    assert jnp.array_equal(
        _resample_bw_input(values[:2], repeat=2, decimate=1)[:, 0],
        jnp.array([1.0, 1.0, 2.0, 2.0]),
    )
    assert jnp.array_equal(
        _resample_bw_input(values, repeat=1, decimate=2)[:, 0],
        jnp.array([2.0, 4.0]),
    )


@pytest.mark.parametrize(
    "monitor",
    [
        BalloonWindkesselBold(period=4.0, dt_bw=2.0, voi=1),
        BalloonWindkesselBold(
            period=8.0,
            dt_bw=1.0,
            voi=1,
            downsample=SubSampling(period=4.0),
        ),
        BalloonWindkesselBold(
            period=8.0,
            dt_bw=1.0,
            voi=1,
            downsample=TemporalAverage(period=2.0, voi=[1, 0]),
        ),
    ],
)
def test_online_balloon_windkessel_matches_posthoc_for_supported_input_grids(
    monitor,
):
    kwargs = dict(t0=10.0, t1=29.0, dt=1.0, n_nodes=3)
    raw = solve(RampDynamics(), Euler(block_size=8), **kwargs)
    expected = monitor(raw)
    actual = solve(RampDynamics(), Euler(block_size=8), observe=monitor, **kwargs)

    assert actual.ys.shape == expected.ys.shape
    assert jnp.allclose(actual.ys, expected.ys)
    assert jnp.allclose(actual.ts, expected.ts)
    assert (
        jnp.array_equal(actual.ts, jnp.arange(18.0, 30.0, 8.0))
        if (monitor.period == 8.0)
        else jnp.array_equal(actual.ts, jnp.arange(14.0, 30.0, 4.0))
    )
    assert actual.variable_names == expected.variable_names


def test_balloon_windkessel_supports_per_node_parameters():
    monitor = BalloonWindkesselBold(
        period=4.0,
        dt_bw=1.0,
        voi=1,
        Eo=jnp.array([0.3, 0.4, 0.5]),
        tauo=jnp.array([0.8, 0.9, 1.0]),
    )
    raw = solve(RampDynamics(), Euler(block_size=8), t1=16.0, dt=1.0, n_nodes=3)
    expected = monitor(raw)
    actual = solve(
        RampDynamics(),
        Euler(block_size=8),
        t1=16.0,
        dt=1.0,
        n_nodes=3,
        observe=monitor,
    )

    assert jnp.allclose(actual.ys, expected.ys)
    assert actual.ys.shape == (4, 1, 3)


def test_balloon_windkessel_neural_and_monitor_gradients_match_posthoc():
    monitor = BalloonWindkesselBold(period=4.0, dt_bw=2.0, voi=1)
    kwargs = dict(t1=16.0, dt=1.0, n_nodes=2)
    raw_fn, raw_config = prepare(RampDynamics(), Euler(block_size=8), **kwargs)
    observed_fn, observed_config = prepare(
        RampDynamics(), Euler(block_size=8), observe=monitor, **kwargs
    )

    def raw_loss(rate, echo_time):
        current = raw_config.copy()
        current.dynamics.rate = rate
        posthoc = BalloonWindkesselBold(period=4.0, dt_bw=2.0, voi=1, TE=echo_time)
        return jnp.sum(posthoc(raw_fn(current)).ys ** 2)

    def observed_loss(rate, echo_time):
        current = observed_config.copy()
        current.dynamics.rate = rate
        current.monitor.TE = echo_time
        return jnp.sum(observed_fn(current).ys ** 2)

    arguments = (jnp.array(1.3), jnp.array(0.05))
    expected = jax.value_and_grad(raw_loss, argnums=(0, 1))(*arguments)
    actual = jax.value_and_grad(observed_loss, argnums=(0, 1))(*arguments)
    assert jnp.allclose(actual[0], expected[0])
    assert jnp.allclose(actual[1][0], expected[1][0])
    assert jnp.allclose(actual[1][1], expected[1][1])


def test_balloon_windkessel_monitor_parameter_partitions_and_differentiates():
    model, config = prepare(
        RampDynamics(),
        Euler(block_size=8),
        t1=16.0,
        dt=1.0,
        n_nodes=2,
        observe=BalloonWindkesselBold(period=4.0, dt_bw=1.0, voi=1),
    )
    config.monitor.tauo = Parameter(config.monitor.tauo)
    parameters, fixed = partition_state(config)

    gradient = jax.grad(
        lambda current: jnp.sum(model(combine_state(current, fixed)).ys ** 2)
    )(parameters)

    assert jnp.isfinite(gradient.monitor.tauo.value)
    assert gradient.monitor.tauo.value != 0.0


@pytest.mark.parametrize("as_network", [False, True])
def test_balloon_windkessel_uses_actual_float64_auxiliary_dtype(as_network):
    with jax.enable_x64():
        dynamics = Float64AuxDynamics()
        model = (
            Network(
                dynamics=dynamics,
                coupling={},
                graph=DenseGraph(jnp.zeros((2, 2), dtype=jnp.float32)),
            )
            if as_network
            else dynamics
        )
        kwargs = {} if as_network else {"n_nodes": 2}
        monitor = BalloonWindkesselBold(period=4.0, dt_bw=1.0, voi=0)
        solver = Euler(block_size=4)
        raw = solve(model, solver, t1=10.0, dt=1.0, **kwargs)
        observed_fn, config = prepare(
            model,
            solver,
            t1=10.0,
            dt=1.0,
            observe=monitor,
            **kwargs,
        )

        expected = monitor(raw)
        actual = jax.jit(observed_fn)(config)
        assert raw.ys.dtype == jnp.float64
        assert actual.ys.dtype == expected.ys.dtype == jnp.float64
        assert jnp.allclose(actual.ys, expected.ys)
        assert jnp.array_equal(actual.ts, expected.ts)


@pytest.mark.parametrize("as_network", [False, True])
def test_balloon_windkessel_probes_float64_external_auxiliary(as_network):
    with jax.enable_x64():
        dynamics = ExternalAuxDynamics()
        external = ConstantInput(amplitude=jnp.array(0.75, dtype=jnp.float64))
        if as_network:
            model = Network(
                dynamics=dynamics,
                coupling={},
                graph=DenseGraph(jnp.zeros((2, 2), dtype=jnp.float32)),
                external_input={"stimulus": external},
            )
            kwargs = {}
        else:
            model = dynamics
            kwargs = {"n_nodes": 2, "externals": {"stimulus": external}}

        solver = Euler(block_size=4)
        monitor = BalloonWindkesselBold(period=4.0, dt_bw=1.0, voi=1)
        raw_fn, raw_config = prepare(model, solver, t1=8.0, dt=1.0, **kwargs)
        observed_fn, observed_config = prepare(
            model,
            solver,
            t1=8.0,
            dt=1.0,
            observe=monitor,
            **kwargs,
        )

        expected = monitor(raw_fn(raw_config))
        actual = jax.jit(observed_fn)(observed_config)
        assert raw_config.initial_state.dynamics.dtype == jnp.float32
        assert raw_fn(raw_config).ys.shape[1] == 2
        assert actual.ys.dtype == expected.ys.dtype == jnp.float64
        assert jnp.allclose(actual.ys, expected.ys)
        assert jnp.array_equal(actual.ts, expected.ts)

        def loss(model_fn, config, amplitude, posthoc):
            current = config.copy()
            current.external = config.external.copy()
            current.external.stimulus = config.external.stimulus.copy()
            current.external.stimulus.amplitude = amplitude
            result = model_fn(current)
            values = monitor(result).ys if posthoc else result.ys
            return jnp.sum(values**2)

        amplitude = jnp.array(0.75, dtype=jnp.float64)
        expected_value, expected_gradient = jax.value_and_grad(
            lambda value: loss(raw_fn, raw_config, value, True)
        )(amplitude)
        actual_value, actual_gradient = jax.value_and_grad(
            lambda value: loss(observed_fn, observed_config, value, False)
        )(amplitude)
        assert jnp.allclose(actual_value, expected_value)
        assert jnp.allclose(actual_gradient, expected_gradient)


def test_balloon_windkessel_probes_float64_coupling_auxiliary():
    with jax.enable_x64():
        gain = jnp.array(0.5, dtype=jnp.float64)
        network = Network(
            dynamics=CouplingAuxDynamics(),
            coupling={"drive": LinearCoupling(source="state", G=gain)},
            graph=DenseGraph(jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)),
        )
        solver = Euler(block_size=4)
        monitor = BalloonWindkesselBold(period=4.0, dt_bw=1.0, voi=1)
        raw_fn, raw_config = prepare(network, solver, t1=8.0, dt=1.0)
        observed_fn, observed_config = prepare(
            network,
            solver,
            t1=8.0,
            dt=1.0,
            observe=monitor,
        )

        expected = monitor(raw_fn(raw_config))
        actual = jax.jit(observed_fn)(observed_config)
        assert raw_config.initial_state.dynamics.dtype == jnp.float32
        assert raw_fn(raw_config).ys.shape[1] == 2
        assert actual.ys.dtype == expected.ys.dtype == jnp.float64
        assert jnp.allclose(actual.ys, expected.ys)

        def loss(model_fn, config, value, posthoc):
            current = config.copy()
            current.coupling = config.coupling.copy()
            current.coupling.drive = config.coupling.drive.copy()
            current.coupling.drive.G = value
            result = model_fn(current)
            values = monitor(result).ys if posthoc else result.ys
            return jnp.sum(values**2)

        expected_value, expected_gradient = jax.value_and_grad(
            lambda value: loss(raw_fn, raw_config, value, True)
        )(gain)
        actual_value, actual_gradient = jax.value_and_grad(
            lambda value: loss(observed_fn, observed_config, value, False)
        )(gain)
        assert jnp.allclose(actual_value, expected_value)
        assert jnp.allclose(actual_gradient, expected_gradient)


@pytest.mark.parametrize("as_network", [False, True])
def test_hrf_uses_actual_float64_auxiliary_dtype(as_network):
    with jax.enable_x64():
        dynamics = Float64AuxDynamics()
        model = (
            Network(
                dynamics=dynamics,
                coupling={},
                graph=DenseGraph(jnp.zeros((2, 2), dtype=jnp.float32)),
            )
            if as_network
            else dynamics
        )
        kwargs = {} if as_network else {"n_nodes": 2}
        monitor = HRFBold(
            period=4.0,
            downsample=SubSampling(period=1.0, voi=0),
            kernel=PreviousSampleHRFKernel(),
        )
        solver = Euler(block_size=4)
        raw = solve(model, solver, t1=8.0, dt=1.0, **kwargs)
        observed_fn, config = prepare(
            model,
            solver,
            t1=8.0,
            dt=1.0,
            observe=monitor,
            **kwargs,
        )

        expected = monitor(raw)
        actual = jax.jit(observed_fn)(config)
        assert raw.ys.dtype == jnp.float64
        assert actual.ys.dtype == expected.ys.dtype == jnp.float64
        assert jnp.allclose(actual.ys, expected.ys)
        assert jnp.array_equal(actual.ts, expected.ts)


def test_balloon_windkessel_promotes_carry_for_live_float64_parameter_gradient():
    with jax.enable_x64():
        dynamics = RampDynamics(
            INITIAL_STATE=(jnp.float32(0.0), jnp.float32(10.0)),
            rate=jnp.float32(1.0),
        )
        monitor = BalloonWindkesselBold(period=4.0, dt_bw=1.0, voi=1)
        kwargs = dict(t1=10.0, dt=1.0, n_nodes=2)
        raw_fn, raw_config = prepare(dynamics, Euler(block_size=4), **kwargs)
        observed_fn, observed_config = prepare(
            dynamics, Euler(block_size=4), observe=monitor, **kwargs
        )
        assert observed_config.initial_state.dtype == jnp.float32
        assert observed_fn(observed_config).ys.dtype == jnp.float32

        tauo = jnp.asarray(0.98, dtype=jnp.float64)

        def posthoc_loss(value):
            current_monitor = BalloonWindkesselBold(
                period=4.0, dt_bw=1.0, voi=1, tauo=value
            )
            return jnp.sum(current_monitor(raw_fn(raw_config)).ys ** 2)

        observed_config.monitor.tauo = Parameter(tauo)
        parameters, fixed = partition_state(observed_config)

        def prepared_loss(current):
            return jnp.sum(observed_fn(combine_state(current, fixed)).ys ** 2)

        expected_value, expected_gradient = jax.value_and_grad(posthoc_loss)(tauo)
        actual_value, gradient = jax.value_and_grad(prepared_loss)(parameters)
        assert actual_value.dtype == jnp.float64
        assert gradient.monitor.tauo.value.dtype == jnp.float64
        assert jnp.allclose(actual_value, expected_value)
        assert jnp.allclose(gradient.monitor.tauo.value, expected_gradient)


def test_balloon_windkessel_float64_per_node_parameter_supports_space_sweep():
    with jax.enable_x64():
        dynamics = RampDynamics(
            INITIAL_STATE=(jnp.float32(0.0), jnp.float32(10.0)),
            rate=jnp.float32(1.0),
        )
        kwargs = dict(t1=8.0, dt=1.0, n_nodes=2)
        raw_fn, raw_config = prepare(dynamics, Euler(block_size=4), **kwargs)
        model, config = prepare(
            dynamics,
            Euler(block_size=4),
            observe=BalloonWindkesselBold(period=4.0, dt_bw=1.0, voi=1),
            **kwargs,
        )
        assert config.initial_state.dtype == jnp.float32
        assert model(config).ys.dtype == jnp.float32
        tauo_cases = jnp.asarray(
            [[0.8, 0.9], [1.0, 1.1]],
            dtype=jnp.float64,
        )
        swept = config.copy()
        swept.monitor.tauo = DataAxis(tauo_cases)
        values = jnp.asarray(
            ParallelExecution(
                lambda current: model(current).ys,
                Space(swept),
                n_vmap=1,
                n_pmap=1,
            ).run()
        )

        raw = raw_fn(raw_config)
        expected = [
            BalloonWindkesselBold(
                period=4.0,
                dt_bw=1.0,
                voi=1,
                tauo=tauo_case,
            )(raw).ys
            for tauo_case in tauo_cases
        ]
        assert values.dtype == jnp.float64
        assert jnp.allclose(values, jnp.stack(expected))


@pytest.mark.parametrize(
    ("monitor", "message"),
    [
        (
            BalloonWindkesselBold(period=4.0, dt_bw=1.5, voi=1),
            "integer multiple",
        ),
        (
            BalloonWindkesselBold(
                period=4.0,
                dt_bw=1.0,
                voi=0,
                downsample=SubSampling(period=3.0),
            ),
            "BOLD period",
        ),
        (
            BalloonWindkesselBold(period=4.0, dt_bw=1.0),
            "exactly one input channel",
        ),
    ],
)
def test_balloon_windkessel_rejects_invalid_nested_grids_and_selection(
    monitor, message
):
    with pytest.raises(ValueError, match=message):
        prepare(
            RampDynamics(),
            Euler(block_size=8),
            t1=16.0,
            dt=1.0,
            n_nodes=2,
            observe=monitor,
        )


def test_balloon_windkessel_jaxpr_has_no_full_raw_trajectory_carry():
    model, config = prepare(
        RampDynamics(),
        Euler(block_size=8),
        t1=24.0,
        dt=1.0,
        n_nodes=3,
        observe=BalloonWindkesselBold(period=8.0, dt_bw=1.0, voi=1),
    )
    shapes = _jaxpr_shapes(jax.make_jaxpr(model)(config))

    assert (24, 2, 3) not in shapes
    assert model(config).ys.shape == (3, 1, 3)


def test_balloon_windkessel_supports_empty_posthoc_and_online_results():
    monitor = BalloonWindkesselBold(period=4.0, dt_bw=1.0, voi=1)
    empty = NativeSolution(
        ts=jnp.empty((0,)),
        ys=jnp.empty((0, 2, 3)),
        dt=1.0,
        variable_names=("slow", "fast"),
    )

    posthoc = monitor(empty)
    online = solve(
        RampDynamics(),
        Euler(block_size=8),
        t1=0.0,
        dt=1.0,
        n_nodes=3,
        observe=monitor,
    )

    assert posthoc.ys.shape == online.ys.shape == (0, 1, 3)
    assert posthoc.ts.shape == online.ts.shape == (0,)
    assert posthoc.variable_names == online.variable_names == ("BOLD(fast)",)


def test_hrf_preparation_resolves_default_averaging_and_history_shape():
    prepared = prepare_observation(
        HRFBold(
            period=4.0,
            downsample_period=2.0,
            voi=1,
            kernel=PreviousSampleHRFKernel(),
        ),
        SimulationGrid(t0=10.0, dt=1.0, n_steps=9),
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
        ("slow", "fast"),
    )

    assert prepared.state0.shape == (2, 1, 3)
    assert prepared.state0.dtype == prepared.data.hrf.dtype
    assert jnp.array_equal(
        prepared.state0, jnp.zeros((2, 1, 3), dtype=prepared.state0.dtype)
    )
    assert prepared.data.final_stride == 2
    assert prepared.params == {"k_1": 5.6, "V_0": 0.02}
    assert prepared.period == 4.0
    assert prepared.first_sample_offset == 4.0
    assert prepared.variable_names == ("BOLD(fast)",)


@pytest.mark.parametrize(
    "monitor",
    [
        HRFBold(
            period=4.0,
            downsample_period=2.0,
            voi=1,
            kernel=PreviousSampleHRFKernel(),
        ),
        HRFBold(
            period=4.0,
            downsample=SubSampling(period=2.0, voi=[1, 0]),
            voi=0,
            kernel=PreviousSampleHRFKernel(),
        ),
    ],
)
def test_online_hrf_matches_posthoc_with_averaging_and_subsampling(monitor):
    kwargs = dict(t0=10.0, t1=29.0, dt=1.0, n_nodes=3)
    raw = solve(RampDynamics(), Euler(block_size=8), **kwargs)
    expected = monitor(raw)
    actual = solve(RampDynamics(), Euler(block_size=8), observe=monitor, **kwargs)

    assert jnp.allclose(actual.ys, expected.ys)
    assert jnp.array_equal(actual.ts, jnp.array([14.0, 18.0, 22.0, 26.0]))
    assert actual.dt == expected.dt == 4.0
    assert actual.variable_names == expected.variable_names


def test_hrf_causal_alignment_and_nonzero_history_across_blocks():
    history = jnp.full((2, 1, 2), 7.0)
    monitor = HRFBold(
        k_1=1.0,
        V_0=1.0,
        period=1.0,
        downsample=SubSampling(period=1.0, voi=1),
        kernel=PreviousSampleHRFKernel(),
        history=history,
    )
    raw = solve(RampDynamics(), Euler(block_size=2), t1=4.0, dt=1.0, n_nodes=2)
    posthoc = monitor(raw)
    online = solve(
        RampDynamics(),
        Euler(block_size=2),
        t1=4.0,
        dt=1.0,
        n_nodes=2,
        observe=monitor,
    )

    expected = jnp.array([6.0, 11.0, 13.0, 15.0])
    assert jnp.allclose(posthoc.ys[:, 0, 0], expected)
    assert jnp.allclose(online.ys, posthoc.ys)
    assert jnp.array_equal(online.ts, jnp.arange(1.0, 5.0))


def test_posthoc_hrf_impulse_has_one_sample_causal_delay():
    monitor = HRFBold(
        k_1=1.0,
        V_0=1.0,
        period=1.0,
        downsample=SubSampling(period=1.0),
        kernel=PreviousSampleHRFKernel(),
    )
    impulse = NativeSolution(
        ts=jnp.arange(1.0, 5.0),
        ys=jnp.array([1.0, 0.0, 0.0, 0.0]).reshape(4, 1, 1),
        dt=1.0,
        variable_names=("impulse",),
    )

    result = monitor(impulse)

    assert jnp.allclose(result.ys[:, 0, 0], jnp.array([-1.0, 0.0, -1.0, -1.0]))
    assert jnp.array_equal(result.ts, impulse.ts)


def test_endpoint_hrf_convolution_matches_full_valid_convolution():
    from tvboptim.observations.tvb_monitors.bold import (
        _convolve_hrf,
        _sample_valid_hrf,
    )

    signal = jnp.arange(66.0).reshape(11, 2, 3) / 10.0
    kernel = jnp.array([0.1, 0.4, -0.2, 0.3])

    expected = _convolve_hrf(signal, kernel, mode="valid")[2::2]
    actual = _sample_valid_hrf(signal, kernel, final_stride=2)

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("final_stride", "expect_direct"),
    [(1, False), (64, True)],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hybrid_valid_hrf_matches_explicit_convolution_and_gradients(
    final_stride, expect_direct, dtype
):
    from tvboptim.observations.tvb_monitors.bold import (
        _prefer_direct_hrf,
        _valid_hrf_samples,
    )

    with jax.enable_x64():
        signal = jnp.linspace(-0.4, 0.8, 257 * 2, dtype=dtype).reshape(257, 1, 2)
        kernel = jnp.linspace(0.01, 0.03, 64, dtype=dtype)

        def explicit(values):
            channels = []
            for channel in range(values.shape[1]):
                nodes = [
                    jnp.convolve(values[:, channel, node], kernel, mode="valid")
                    for node in range(values.shape[2])
                ]
                channels.append(jnp.stack(nodes, axis=1))
            return jnp.stack(channels, axis=1)[final_stride::final_stride]

        assert (
            _prefer_direct_hrf(signal.shape[0], kernel.shape[0], final_stride)
            is expect_direct
        )
        actual = _valid_hrf_samples(signal, kernel, final_stride)
        expected = explicit(signal)
        tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == jnp.float32 else {}
        assert jnp.allclose(actual, expected, **tolerance)

        actual_gradient = jax.grad(
            lambda values: jnp.sum(
                _valid_hrf_samples(values, kernel, final_stride) ** 2
            )
        )(signal)
        expected_gradient = jax.grad(lambda values: jnp.sum(explicit(values) ** 2))(
            signal
        )
        assert jnp.allclose(actual_gradient, expected_gradient, **tolerance)


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        (jnp.array([[[5.0, 6.0]]]), jnp.array([0.0, 0.0, 0.0, 5.0])),
        (
            jnp.array(
                [
                    [[1.0, 1.0]],
                    [[2.0, 2.0]],
                    [[3.0, 3.0]],
                    [[4.0, 4.0]],
                    [[5.0, 5.0]],
                ]
            ),
            jnp.array([2.0, 3.0, 4.0, 5.0]),
        ),
    ],
)
def test_hrf_history_is_front_padded_or_suffix_trimmed(history, expected):
    prepared = prepare_observation(
        HRFBold(
            period=1.0,
            downsample=SubSampling(period=1.0, voi=1),
            kernel=PreviousSampleHRFKernel(),
            history=history,
        ),
        SimulationGrid(t0=0.0, dt=1.0, n_steps=4),
        jax.ShapeDtypeStruct((2, 2), jnp.float32),
        ("slow", "fast"),
    )

    assert jnp.array_equal(prepared.state0[:, 0, 0], expected)


def test_hrf_solution_history_is_downsampled_selected_and_normalized():
    history = NativeSolution(
        ts=jnp.arange(1.0, 4.0),
        ys=jnp.array(
            [
                [[1.0, 1.0], [10.0, 12.0]],
                [[3.0, 3.0], [20.0, 22.0]],
                [[5.0, 5.0], [30.0, 32.0]],
            ]
        ),
        dt=1.0,
        variable_names=("slow", "fast"),
    )
    monitor = HRFBold(
        period=4.0,
        downsample_period=2.0,
        voi=1,
        kernel=PreviousSampleHRFKernel(),
        history=history,
    )
    prepared = prepare_observation(
        monitor,
        SimulationGrid(t0=0.0, dt=1.0, n_steps=4),
        jax.ShapeDtypeStruct((2, 2), jnp.float32),
        ("slow", "fast"),
    )

    assert jnp.array_equal(
        prepared.state0[:, 0, :], jnp.array([[0.0, 0.0], [15.0, 17.0]])
    )


def test_hrf_live_scaling_and_neural_gradients_match_posthoc():
    monitor = HRFBold(
        period=4.0,
        downsample_period=2.0,
        voi=1,
        kernel=PreviousSampleHRFKernel(),
    )
    kwargs = dict(t1=16.0, dt=1.0, n_nodes=2)
    raw_fn, raw_config = prepare(RampDynamics(), Euler(block_size=8), **kwargs)
    observed_fn, observed_config = prepare(
        RampDynamics(), Euler(block_size=8), observe=monitor, **kwargs
    )

    def raw_loss(rate, scaling):
        current = raw_config.copy()
        current.dynamics.rate = rate
        posthoc = HRFBold(
            k_1=scaling,
            period=4.0,
            downsample_period=2.0,
            voi=1,
            kernel=PreviousSampleHRFKernel(),
        )
        return jnp.sum(posthoc(raw_fn(current)).ys ** 2)

    def observed_loss(rate, scaling):
        current = observed_config.copy()
        current.dynamics.rate = rate
        current.monitor.k_1 = scaling
        return jnp.sum(observed_fn(current).ys ** 2)

    arguments = (jnp.array(1.3), jnp.array(4.2))
    expected = jax.value_and_grad(raw_loss, argnums=(0, 1))(*arguments)
    actual = jax.value_and_grad(observed_loss, argnums=(0, 1))(*arguments)
    assert jnp.allclose(actual[0], expected[0])
    assert jnp.allclose(actual[1][0], expected[1][0])
    assert jnp.allclose(actual[1][1], expected[1][1])


def test_hrf_scaling_parameter_partitions_and_differentiates():
    model, config = prepare(
        RampDynamics(),
        Euler(block_size=8),
        t1=16.0,
        dt=1.0,
        n_nodes=2,
        observe=HRFBold(
            period=4.0,
            downsample_period=2.0,
            voi=1,
            kernel=PreviousSampleHRFKernel(),
        ),
    )
    config.monitor.V_0 = Parameter(config.monitor.V_0)
    parameters, fixed = partition_state(config)

    gradient = jax.grad(
        lambda current: jnp.sum(model(combine_state(current, fixed)).ys ** 2)
    )(parameters)

    assert jnp.isfinite(gradient.monitor.V_0.value)
    assert gradient.monitor.V_0.value != 0.0


@pytest.mark.parametrize("mode", ["same", "full"])
def test_prepared_hrf_rejects_noncausal_convolution_modes(mode):
    with pytest.raises(ValueError, match="convolution_mode='valid'"):
        prepare(
            RampDynamics(),
            Euler(block_size=8),
            t1=16.0,
            dt=1.0,
            observe=HRFBold(
                period=4.0,
                downsample_period=2.0,
                voi=1,
                kernel=PreviousSampleHRFKernel(),
                convolution_mode=mode,
            ),
        )


def test_prepared_hrf_rejects_invalid_grid_and_history_shape():
    with pytest.raises(ValueError, match="integer multiple"):
        prepare(
            RampDynamics(),
            Euler(block_size=8),
            dt=1.0,
            observe=HRFBold(
                period=4.0,
                downsample_period=1.5,
                voi=1,
                kernel=PreviousSampleHRFKernel(),
            ),
        )
    with pytest.raises(ValueError, match="channel/node shape"):
        prepare(
            RampDynamics(),
            Euler(block_size=8),
            dt=1.0,
            n_nodes=2,
            observe=HRFBold(
                period=4.0,
                downsample_period=2.0,
                voi=1,
                kernel=PreviousSampleHRFKernel(),
                history=jnp.zeros((2, 1, 3)),
            ),
        )


def test_hrf_zero_output_tail_and_empty_run_are_safe():
    monitor = HRFBold(
        period=4.0,
        downsample_period=2.0,
        voi=1,
        kernel=PreviousSampleHRFKernel(),
    )
    short = solve(
        RampDynamics(),
        Euler(block_size=4),
        t1=3.0,
        dt=1.0,
        n_nodes=2,
        observe=monitor,
    )
    empty = solve(
        RampDynamics(),
        Euler(block_size=4),
        t1=0.0,
        dt=1.0,
        n_nodes=2,
        observe=monitor,
    )

    assert short.ys.shape == empty.ys.shape == (0, 1, 2)
    assert short.ts.shape == empty.ts.shape == (0,)


def test_default_hrf_constructor_runs_as_prepared_observation():
    result = solve(
        RampDynamics(),
        Euler(block_size=250),
        t1=1000.0,
        dt=4.0,
        n_nodes=1,
        observe=HRFBold(voi=1),
    )

    assert result.ys.shape == (1, 1, 1)
    assert result.ts[0] == 1000.0
    assert result.variable_names == ("BOLD(fast)",)


def test_hrf_jaxpr_carries_kernel_history_without_full_raw_trajectory():
    model, config = prepare(
        RampDynamics(),
        Euler(block_size=8),
        t1=24.0,
        dt=1.0,
        n_nodes=3,
        observe=HRFBold(
            period=4.0,
            downsample_period=2.0,
            voi=1,
            kernel=PreviousSampleHRFKernel(),
        ),
    )
    shapes = _jaxpr_shapes(jax.make_jaxpr(model)(config))

    assert (24, 2, 3) not in shapes
    assert (2, 1, 3) in shapes
    assert model(config).ys.shape == (6, 1, 3)


def test_reducer_factory_and_prepare_keyword_are_deprecated():
    with pytest.warns(DeprecationWarning, match="welford_cov is deprecated"):
        reducer = welford_cov()
    with pytest.warns(DeprecationWarning, match="reduce= is deprecated"):
        prepare(
            RampDynamics(),
            Euler(block_size=4),
            t1=1.0,
            dt=0.25,
            reduce=reducer,
        )
