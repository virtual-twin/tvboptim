"""Method preparation, the per-sample adapter, and output label rules."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from docs.advanced.low_pass_monitor import LowPass
from tvboptim.experimental.network_dynamics import prepare, solve
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.solvers import Euler
from tvboptim.observations import (
    JointObservation,
    PreparedObservation,
    SampledMonitor,
    SimulationGrid,
    StreamingMonitor,
    apply_observation,
)
from tvboptim.observations.tvb_monitors import (
    AbstractMonitor,
    BalloonWindkesselBold,
    HRFBold,
    SubSampling,
    TemporalAverage,
)


class SineDynamics(AbstractDynamics):
    STATE_NAMES = ("signal",)
    INITIAL_STATE = (0.0,)
    DEFAULT_PARAMS = Bunch()

    def dynamics(self, t, state, params, coupling, external):
        del state, params, coupling, external
        return jnp.cos(0.7 * t) * jnp.ones((1, 1))


class ExplicitLowPass(StreamingMonitor):
    """The LowPass recurrence written directly as a block recipe."""

    def __init__(self, period=10.0, tau=50.0):
        self.period = period
        self.tau = tau

    def prepare(self, grid, sample, variable_names):
        stride = round(self.period / grid.dt)
        dt = grid.dt

        def init(params):
            return jnp.zeros(sample.shape, jnp.result_type(sample.dtype, params.tau))

        def update(state, block, params):
            decay = jnp.exp(-dt / params.tau)

            def step(filtered, value):
                filtered = decay * filtered + (1.0 - decay) * value
                return filtered, filtered

            state, filtered = jax.lax.scan(step, state, block)
            return state, filtered[stride - 1 :: stride]

        return PreparedObservation(
            params=Bunch(tau=self.tau),
            init=init,
            update=update,
            output=grid.output(self.period, variable_names=variable_names),
        )


class RunningVariance(SampledMonitor):
    """Exponentially weighted variance with a mean/variance/weight carry."""

    period: float = eqx.field(static=True, default=4.0)
    bias_correction: bool = eqx.field(static=True, default=False)
    rate: object = 0.3

    def init(self, sample_spec):
        dtype = jnp.result_type(sample_spec.dtype, self.rate)
        zeros = jnp.zeros(sample_spec.shape, dtype)
        return zeros, zeros, jnp.zeros((), dtype)

    def step(self, state, sample, dt):
        del dt
        mean, variance, weight = state
        delta = sample - mean
        mean = mean + self.rate * delta
        variance = (1.0 - self.rate) * (variance + self.rate * delta**2)
        weight = (1.0 - self.rate) * weight + self.rate
        value = variance / weight if self.bias_correction else variance
        return (mean, variance, weight), value


def _low_pass_reference(values, dt, tau):
    decay = np.exp(-dt / tau)
    filtered = np.zeros_like(values[0])
    out = []
    for value in values:
        filtered = decay * filtered + (1.0 - decay) * value
        out.append(filtered)
    return np.asarray(out)


def _variance_reference(values, rate, bias_correction):
    mean = np.zeros_like(values[0])
    variance = np.zeros_like(values[0])
    out = []
    for n, value in enumerate(values, start=1):
        delta = value - mean
        mean = mean + rate * delta
        variance = (1.0 - rate) * (variance + rate * delta**2)
        weight = 1.0 - (1.0 - rate) ** n
        out.append(variance / weight if bias_correction else variance)
    return np.asarray(out)


MONITORS = {
    "sampled_low_pass": (
        lambda: LowPass(period=4.0, tau=3.0),
        lambda values, dt: _low_pass_reference(values, dt, 3.0),
    ),
    "explicit_low_pass": (
        lambda: ExplicitLowPass(period=4.0, tau=3.0),
        lambda values, dt: _low_pass_reference(values, dt, 3.0),
    ),
    "running_variance": (
        lambda: RunningVariance(period=4.0, rate=0.3),
        lambda values, dt: _variance_reference(values, 0.3, False),
    ),
    "running_variance_corrected": (
        lambda: RunningVariance(period=4.0, bias_correction=True, rate=0.3),
        lambda values, dt: _variance_reference(values, 0.3, True),
    ),
}


@pytest.mark.parametrize("n_steps", [16, 19])
@pytest.mark.parametrize("centered", [False, True], ids=["raw", "centered"])
@pytest.mark.parametrize("name", MONITORS)
def test_authored_monitor_matches_reference_online_and_posthoc(name, centered, n_steps):
    build, reference = MONITORS[name]
    monitor = build()
    observe = (
        JointObservation(
            preprocess=TemporalAverage(period=2.0), outputs={"out": monitor}
        )
        if centered
        else monitor
    )
    kwargs = dict(t0=0.0, t1=float(n_steps), dt=1.0)
    raw = solve(SineDynamics(), Euler(), **kwargs)
    online = solve(SineDynamics(), Euler(block_size=8), observe=observe, **kwargs)
    posthoc = observe(raw)
    if centered:
        online, posthoc = online["out"], posthoc["out"]

    values = np.asarray(raw.ys)
    labels = np.asarray(raw.ts)
    if centered:
        n = len(values) // 2
        values = values[: 2 * n].reshape(n, 2, *values.shape[1:]).mean(axis=1)
        labels = labels[: 2 * n].reshape(n, 2).mean(axis=1)
    stride = 2 if centered else 4
    expected = reference(values, 2.0 if centered else 1.0)[stride - 1 :: stride]
    expected_ts = labels[stride - 1 :: stride]

    for result in (online, posthoc):
        np.testing.assert_allclose(result.ys, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(result.ts, expected_ts)
        assert result.variable_names == ("signal",)
        assert result.dt == 4.0


def test_sampled_monitor_publishes_live_fields_and_differentiates_them():
    model, config = prepare(
        SineDynamics(),
        Euler(block_size=8),
        t1=16.0,
        dt=1.0,
        observe=RunningVariance(period=4.0, bias_correction=True, rate=0.3),
    )
    assert isinstance(config.monitor, Bunch)
    assert config.monitor == {"rate": 0.3}

    raw = solve(SineDynamics(), Euler(), t1=16.0, dt=1.0)

    def online(rate):
        current = config.copy()
        current.monitor["rate"] = rate
        return jnp.sum(model(current).ys)

    def posthoc(rate):
        monitor = RunningVariance(period=4.0, bias_correction=True, rate=rate)
        return jnp.sum(monitor(raw).ys)

    rate = jnp.asarray(0.4)
    assert jnp.allclose(jax.jit(online)(rate), posthoc(rate))
    assert jnp.allclose(jax.jit(jax.grad(online))(rate), jax.grad(posthoc)(rate))


def test_sampled_monitor_state_follows_live_parameter_dtype():
    solution = NativeSolution(
        ts=jnp.arange(1.0, 9.0),
        ys=jnp.ones((8, 1, 2), dtype=jnp.float32),
        dt=1.0,
        variable_names=None,
    )
    with jax.enable_x64():
        result = LowPass(period=4.0, tau=jnp.asarray(3.0, jnp.float64))(solution)
    assert result.ys.dtype == jnp.float64


def test_sampled_monitor_rebuilds_without_calling_the_constructor():
    calls = []

    class Counted(SampledMonitor):
        period: float = eqx.field(static=True)
        gain: object

        def __init__(self, period, gain):
            calls.append(gain)
            self.period = period
            self.gain = gain

        def init(self, sample_spec):
            return None

        def step(self, state, sample, dt):
            return state, self.gain * sample

    model, config = prepare(
        SineDynamics(), Euler(block_size=4), t1=8.0, dt=1.0, observe=Counted(2.0, 2.0)
    )
    current = config.copy()
    current.monitor["gain"] = 5.0
    result = model(current)
    raw = solve(SineDynamics(), Euler(), t1=8.0, dt=1.0)

    assert calls == [2.0]
    assert jnp.allclose(result.ys, 5.0 * raw.ys[1::2])


def test_sampled_monitor_rejects_non_jax_live_fields():
    class Labelled(SampledMonitor):
        period: float = eqx.field(static=True, default=1.0)
        label: str = "filtered"

        def init(self, sample_spec):
            return None

        def step(self, state, sample, dt):
            return state, sample

    with pytest.raises(TypeError, match=r"Labelled\.label is a live field"):
        solve(SineDynamics(), Euler(), t1=4.0, dt=1.0, observe=Labelled())


def test_sampled_monitor_rejects_candidates_that_change_the_sample_shape():
    class Summed(SampledMonitor):
        period: float = eqx.field(static=True, default=1.0)

        def init(self, sample_spec):
            return None

        def step(self, state, sample, dt):
            return state, jnp.concatenate([sample, sample])

    solution = NativeSolution(
        ts=jnp.arange(1.0, 5.0), ys=jnp.zeros((4, 1, 2)), dt=1.0, variable_names=None
    )
    with pytest.raises(ValueError, match="update failed abstract validation") as info:
        Summed()(solution)
    assert "shaped like the input sample" in str(info.value.__cause__)


def _doubled_recipe(prepared):
    """Return ``prepared`` with every emitted value doubled."""

    def update(state, block, params):
        state, values = prepared.update(state, block, params)
        return state, 2.0 * values

    return PreparedObservation(
        params=prepared.params,
        init=prepared.init,
        update=update,
        output=prepared.output,
        alignment_steps=prepared.alignment_steps,
    )


class DoubledSubSampling(SubSampling):
    """A built-in subclass that customizes preparation through ``super``."""

    def prepare(self, grid, sample, variable_names):
        return _doubled_recipe(super().prepare(grid, sample, variable_names))


class PostprocessedSubSampling(SubSampling):
    """A built-in subclass that only customizes post-hoc calls."""

    def __call__(self, sol, t_offset=0.0):
        result = super().__call__(sol, t_offset=t_offset)
        return NativeSolution(
            ts=result.ts, ys=2.0 * result.ys, dt=result.dt, variable_names=None
        )


class MethodOnly:
    """A monitor that only implements the preparation method."""

    def prepare(self, grid, sample, variable_names):
        return PreparedObservation(
            params=Bunch(),
            init=lambda params: None,
            update=lambda state, block, params: (state, 2.0 * block),
            output=grid.output(grid.dt, variable_names=variable_names),
        )


class ForeignGain:
    """A post-hoc-only type standing in for code the author cannot modify."""

    def __init__(self, gain):
        self.gain = gain

    def __call__(self, solution):
        return self.gain * solution.ys


class StreamingGain(StreamingMonitor):
    """Adapt ``ForeignGain`` for streaming without modifying or registering it."""

    def __init__(self, foreign):
        self.foreign = foreign

    def prepare(self, grid, sample, variable_names):
        del sample
        return PreparedObservation(
            params=Bunch(gain=self.foreign.gain),
            init=lambda params: None,
            update=lambda state, block, params: (state, params.gain * block),
            output=grid.output(grid.dt, variable_names=variable_names),
        )


POST_HOC_CALLS = []


class PostHocOnly(AbstractMonitor):
    """An existing-style custom monitor implementing only ``__call__``."""

    def __init__(self, voi=None, period=1.0):
        self.voi = self._normalize_voi(voi)
        self.period = period

    def __call__(self, sol):
        POST_HOC_CALLS.append(sol)
        return NativeSolution(
            ts=sol.ts, ys=sol.ys[:, self.voi, :], dt=sol.dt, variable_names=None
        )


def _raw():
    return NativeSolution(
        ts=jnp.arange(1.0, 5.0),
        ys=jnp.arange(4.0).reshape(4, 1, 1),
        dt=1.0,
        variable_names=("signal",),
    )


def _native(monitor, n_steps=4):
    return solve(
        SineDynamics(), Euler(block_size=2), t1=float(n_steps), dt=1.0, observe=monitor
    )


def _raw_sine(n_steps=4):
    return solve(SineDynamics(), Euler(), t1=float(n_steps), dt=1.0)


METHOD_RESOLVED = {
    "method_only": (MethodOnly, lambda ys: 2.0 * ys),
    "builtin_super_prepare": (
        lambda: DoubledSubSampling(period=2.0),
        lambda ys: 2.0 * ys[1::2],
    ),
    "foreign_wrapper": (
        lambda: StreamingGain(ForeignGain(3.0)),
        lambda ys: 3.0 * ys,
    ),
}


@pytest.mark.parametrize("name", METHOD_RESOLVED)
def test_prepare_method_is_selected_natively_and_posthoc(name):
    build, expected = METHOD_RESOLVED[name]
    raw = _raw_sine()
    for result in (_native(build()), apply_observation(build(), raw)):
        assert jnp.allclose(result.ys, expected(raw.ys))


def test_foreign_object_keeps_its_own_call_when_wrapped():
    foreign = ForeignGain(3.0)
    wrapped = StreamingGain(foreign)
    assert jnp.allclose(wrapped(_raw()).ys, foreign(_raw()))


def test_call_override_changes_posthoc_but_native_uses_inherited_recipe():
    monitor = PostprocessedSubSampling(period=2.0)
    raw = _raw_sine()
    inherited = SubSampling(period=2.0)(raw).ys

    assert jnp.allclose(monitor(raw).ys, 2.0 * inherited)
    assert jnp.allclose(monitor(sol=raw).ys, 2.0 * inherited)
    assert jnp.allclose(_native(monitor).ys, inherited)
    assert jnp.allclose(apply_observation(monitor, raw).ys, inherited)


def test_posthoc_only_monitor_keeps_working_and_is_rejected_natively():
    monitor = PostHocOnly(voi=0)
    assert monitor.voi == slice(0, 1)
    assert jnp.array_equal(monitor(_raw()).ys, _raw().ys)
    POST_HOC_CALLS.clear()

    with pytest.raises(TypeError, match="PostHocOnly has no streaming preparation"):
        _native(monitor)
    assert POST_HOC_CALLS == []


STREAMING_KINDS = {
    "sub_sampling": (lambda: SubSampling(), True),
    "temporal_average": (lambda: TemporalAverage(), True),
    "hrf_bold": (lambda: HRFBold(), True),
    "balloon_windkessel": (lambda: BalloonWindkesselBold(), True),
    "method_only": (MethodOnly, True),
    "streaming_subclass": (lambda: ExplicitLowPass(), True),
    "sampled_subclass": (lambda: LowPass(), True),
    "foreign_wrapper": (lambda: StreamingGain(ForeignGain(1.0)), True),
    "joint": (lambda: JointObservation(outputs={"x": SubSampling()}), True),
    "post_hoc_only": (PostHocOnly, False),
    "unrelated": (object, False),
}


@pytest.mark.parametrize("name", STREAMING_KINDS)
def test_streaming_isinstance_is_structural(name):
    build, streaming = STREAMING_KINDS[name]
    monitor = build()
    assert isinstance(monitor, StreamingMonitor) is streaming
    assert isinstance(monitor, SampledMonitor) is (name == "sampled_subclass")


def test_structural_recognition_adds_no_methods():
    assert not hasattr(MethodOnly(), "__call__")
    assert StreamingMonitor not in type(SubSampling()).__mro__


class Counting:
    calls = 0

    def prepare(self, grid, sample, variable_names):
        type(self).calls += 1
        return MethodOnly().prepare(grid, sample, variable_names)


class NotCallable:
    prepare = 3


class WrongResult:
    def prepare(self, grid, sample, variable_names):
        return {}


class Failing:
    def prepare(self, grid, sample, variable_names):
        raise RuntimeError("broken preparation")


@pytest.mark.parametrize(
    "monitor", [Counting, TemporalAverage], ids=lambda c: c.__name__
)
def test_class_instead_of_instance_is_rejected_before_preparing(monitor):
    Counting.calls = 0
    with pytest.raises(TypeError, match=rf"{monitor.__name__}\(\.\.\.\), not the"):
        apply_observation(monitor, _raw())
    assert Counting.calls == 0


@pytest.mark.parametrize(
    ("monitor", "error", "message"),
    [
        (object(), TypeError, "object has no streaming preparation method"),
        (NotCallable(), TypeError, "NotCallable has no streaming preparation"),
        (
            WrongResult(),
            TypeError,
            r"WrongResult\.prepare\(\.\.\.\) must return PreparedObservation; "
            "returned dict",
        ),
        (Failing(), RuntimeError, "broken preparation"),
    ],
    ids=["missing", "not_callable", "wrong_result", "method_failure"],
)
def test_preparation_gateway_errors(monitor, error, message):
    with pytest.raises(error, match=message):
        apply_observation(monitor, _raw())
    with pytest.raises(error, match=message):
        _native(monitor)


@pytest.mark.parametrize(
    ("first", "period", "label", "expected"),
    [
        (None, 4.0, "last_input", 4.0),
        (None, 4.0, "window_center", 2.5),
        (None, 4.0, "period_end", 4.0),
        (None, 1.0, "last_input", 1.0),
        (1.5, 8.0, "last_input", 7.5),
        (1.5, 8.0, "window_center", 4.5),
        (1.5, 8.0, "period_end", 8.0),
        (1.5, 2.0, "last_input", 1.5),
    ],
)
def test_grid_output_label_rules(first, period, label, expected):
    dt = 1.0 if first is None else 2.0
    grid = SimulationGrid(t0=0.0, dt=dt, n_steps=16, first_sample_offset=first)
    output = grid.output(period, label=label, variable_names=("x",))
    assert output.period == period
    assert output.first_sample_offset == expected
    assert output.variable_names == ("x",)


@pytest.mark.parametrize(
    ("period", "label", "message"),
    [
        (3.0, "last_input", "integer multiple"),
        (4.0, "end", "Output label must be"),
    ],
)
def test_grid_output_rejects_invalid_period_or_label(period, label, message):
    grid = SimulationGrid(t0=0.0, dt=2.0, n_steps=8)
    with pytest.raises(ValueError, match=message):
        grid.output(period, label=label)


class NestedSettings(eqx.Module):
    """Static metadata held inside a nested module's static field."""

    gains: list = eqx.field(static=True)


def test_sampled_monitor_snapshots_settings_at_preparation():
    class Scaled(SampledMonitor):
        period: float = eqx.field(static=True, default=1.0)
        settings: dict = eqx.field(static=True, default=None)
        nested: NestedSettings = eqx.field(static=True, default=None)
        offset: object = None

        def init(self, sample_spec):
            return None

        def step(self, state, sample, dt):
            factor = self.settings["factor"] * self.nested.gains[0]
            return state, factor * sample + self.offset[0]

    settings = {"factor": 2.0}
    gains = [1.0]
    offset = np.zeros(1, dtype=np.float32)
    model, config = prepare(
        SineDynamics(),
        Euler(block_size=4),
        t1=4.0,
        dt=1.0,
        observe=Scaled(
            settings=settings, nested=NestedSettings(gains=gains), offset=offset
        ),
    )
    expected = model(config).ys
    settings["factor"] = 5.0
    gains[0] = 7.0
    offset[0] = 3.0

    assert jnp.allclose(model(config).ys, expected)
    assert jnp.allclose(jax.jit(model)(config).ys, expected)
