"""Tests for HRFBold and BalloonWindkesselBold monitors.

Checks output shape, period, and timestamp correctness using dummy/random
NativeSolution inputs.
"""

import unittest

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pytest

from tvboptim.experimental.network_dynamics import Bunch, prepare, solve
from tvboptim.experimental.network_dynamics.dynamics import AbstractDynamics
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.solvers import Euler
from tvboptim.observations.tvb_monitors import (
    BalloonWindkesselBold,
    FirstOrderVolterraHRFKernel,
    HRFBold,
    SubSampling,
    TemporalAverage,
    streaming_hrf_bold,
)


def make_sol(T_ms, dt_ms, n_nodes, n_states=1, seed=0):
    """Create a random NativeSolution with firing-rate-like values.

    Args:
        T_ms: Total duration in ms
        dt_ms: Time step in ms
        n_nodes: Number of nodes
        n_states: Number of state variables
        seed: Random seed (not used, just zeros for determinism)

    Returns:
        NativeSolution with shape [T, n_states, n_nodes]
    """
    import jax

    n_steps = int(round(T_ms / dt_ms))
    key = jax.random.PRNGKey(seed)
    # Firing rates in Hz — positive values around 10 Hz
    ys = jax.random.uniform(
        key, shape=(n_steps, n_states, n_nodes), minval=0.0, maxval=20.0
    )
    ts = jnp.arange(n_steps) * dt_ms
    return NativeSolution(ts=ts, ys=ys, dt=dt_ms)


class ConstantDrive(AbstractDynamics):
    STATE_NAMES = ("signal",)
    INITIAL_STATE = (0.0,)
    DEFAULT_PARAMS = Bunch(rate=1.0)

    def dynamics(self, t, state, params, coupling, external):
        del t, state, coupling, external
        return jnp.asarray([[params.rate]])


class CustomTemporalAverage:
    period = 4.0

    def __call__(self, solution):
        return TemporalAverage(period=self.period)(solution)


class DoubledTemporalAverage(TemporalAverage):
    def __call__(self, solution):
        averaged = super().__call__(solution)
        return NativeSolution(
            ts=averaged.ts,
            ys=2.0 * averaged.ys,
            dt=averaged.dt,
            variable_names=averaged.variable_names,
        )


class DoubledSubSampling(SubSampling):
    def __call__(self, solution):
        sampled = super().__call__(solution)
        return NativeSolution(
            ts=sampled.ts,
            ys=2.0 * sampled.ys,
            dt=sampled.dt,
            variable_names=sampled.variable_names,
        )


class TestHRFBoldOutputShape(unittest.TestCase):
    def setUp(self):
        # 20 s simulation at 1 ms dt, 5 nodes, 1 state variable
        self.sol = make_sol(T_ms=20_000, dt_ms=1.0, n_nodes=5)

    def test_output_shape(self):
        monitor = HRFBold(period=1000.0, voi=0)
        result = monitor(self.sol)
        # [T_bold, 1, N]
        self.assertEqual(result.ys.ndim, 3)
        self.assertEqual(result.ys.shape[1], 1)
        self.assertEqual(result.ys.shape[2], 5)

    def test_output_time_steps_match_period(self):
        period = 1000.0
        monitor = HRFBold(period=period, voi=0)
        result = monitor(self.sol)
        T_bold = result.ys.shape[0]
        self.assertEqual(len(result.ts), T_bold)

    def test_output_dt(self):
        period = 1000.0
        monitor = HRFBold(period=period, voi=0)
        result = monitor(self.sol)
        self.assertAlmostEqual(float(result.dt), period)

    def test_longer_period_fewer_samples(self):
        monitor_fast = HRFBold(period=500.0, voi=0)
        monitor_slow = HRFBold(period=1000.0, voi=0)
        result_fast = monitor_fast(self.sol)
        result_slow = monitor_slow(self.sol)
        self.assertGreater(result_fast.ys.shape[0], result_slow.ys.shape[0])

    def test_multi_node_preserved(self):
        for n_nodes in [1, 5, 10]:
            sol = make_sol(T_ms=20_000, dt_ms=1.0, n_nodes=n_nodes)
            monitor = HRFBold(period=1000.0, voi=0)
            result = monitor(sol)
            self.assertEqual(result.ys.shape[2], n_nodes)


class TestHRFBoldTimestamps(unittest.TestCase):
    def setUp(self):
        self.sol = make_sol(T_ms=10_000, dt_ms=1.0, n_nodes=3)

    def test_timestamps_monotonically_increasing(self):
        monitor = HRFBold(period=1000.0, voi=0)
        result = monitor(self.sol)
        diffs = jnp.diff(result.ts)
        self.assertTrue(jnp.all(diffs > 0))

    def test_timestamp_spacing_matches_period(self):
        period = 1000.0
        monitor = HRFBold(period=period, voi=0)
        result = monitor(self.sol)
        if len(result.ts) > 1:
            diffs = jnp.diff(result.ts)
            for d in diffs:
                self.assertAlmostEqual(float(d), period, places=1)


class TestBalloonWindkesselBoldOutputShape(unittest.TestCase):
    def setUp(self):
        # 20 s simulation at 1 ms dt, 5 nodes
        self.sol = make_sol(T_ms=20_000, dt_ms=1.0, n_nodes=5)

    def test_output_shape(self):
        monitor = BalloonWindkesselBold(period=2000.0, dt_bw=1.0, voi=0)
        result = monitor(self.sol)
        # [T_bold, 1, N]
        self.assertEqual(result.ys.ndim, 3)
        self.assertEqual(result.ys.shape[1], 1)
        self.assertEqual(result.ys.shape[2], 5)

    def test_output_time_steps_match_period(self):
        period = 2000.0
        monitor = BalloonWindkesselBold(period=period, dt_bw=1.0, voi=0)
        result = monitor(self.sol)
        T_bold = result.ys.shape[0]
        self.assertEqual(len(result.ts), T_bold)

    def test_output_dt(self):
        period = 2000.0
        monitor = BalloonWindkesselBold(period=period, dt_bw=1.0, voi=0)
        result = monitor(self.sol)
        self.assertAlmostEqual(float(result.dt), period)

    def test_longer_period_fewer_samples(self):
        monitor_fast = BalloonWindkesselBold(period=1000.0, dt_bw=1.0, voi=0)
        monitor_slow = BalloonWindkesselBold(period=2000.0, dt_bw=1.0, voi=0)
        result_fast = monitor_fast(self.sol)
        result_slow = monitor_slow(self.sol)
        self.assertGreater(result_fast.ys.shape[0], result_slow.ys.shape[0])

    def test_multi_node_preserved(self):
        for n_nodes in [1, 5, 10]:
            sol = make_sol(T_ms=20_000, dt_ms=1.0, n_nodes=n_nodes)
            monitor = BalloonWindkesselBold(period=2000.0, dt_bw=1.0, voi=0)
            result = monitor(sol)
            self.assertEqual(result.ys.shape[2], n_nodes)

    def test_legacy_positional_voi_and_downsample_order_is_preserved(self):
        solution = NativeSolution(
            ts=jnp.arange(1.0, 9.0),
            ys=jnp.arange(16.0).reshape(8, 2, 1) / 100.0,
            dt=1.0,
            variable_names=("first", "second"),
        )
        prefix = (2.0, 1.0, 0.65, 0.41, 0.98, 0.32, 0.4, 0.04, 0.04)

        positional_voi = BalloonWindkesselBold(*prefix, 1)
        keyword_voi = BalloonWindkesselBold(*prefix, voi=1)
        self.assertEqual(positional_voi.voi, keyword_voi.voi)
        self.assertEqual(positional_voi.k3, keyword_voi.k3)
        self.assertEqual(positional_voi.k3, 1.0)
        self.assertTrue(
            jnp.array_equal(positional_voi(solution).ys, keyword_voi(solution).ys)
        )

        downsample = SubSampling(period=1.0)
        positional_downsample = BalloonWindkesselBold(*prefix, 1, downsample)
        keyword_downsample = BalloonWindkesselBold(
            *prefix, voi=1, downsample=downsample
        )
        self.assertIs(positional_downsample.downsample, downsample)
        self.assertTrue(
            jnp.array_equal(
                positional_downsample(solution).ys,
                keyword_downsample(solution).ys,
            )
        )

        overridden = BalloonWindkesselBold(*prefix, voi=1, k3=0.25)
        self.assertEqual(overridden.k3, 0.25)
        from tvboptim.observations.tvb_monitors.bold import _bw_params, _bw_signal

        state = (
            jnp.array([0.0]),
            jnp.array([1.0]),
            jnp.array([0.8]),
            jnp.array([0.7]),
        )
        self.assertFalse(
            jnp.array_equal(
                _bw_signal(state, _bw_params(overridden)),
                _bw_signal(state, _bw_params(keyword_voi)),
            )
        )


class TestBalloonWindkesselBoldTimestamps(unittest.TestCase):
    def setUp(self):
        self.sol = make_sol(T_ms=20_000, dt_ms=1.0, n_nodes=3)

    def test_timestamps_monotonically_increasing(self):
        monitor = BalloonWindkesselBold(period=2000.0, dt_bw=1.0, voi=0)
        result = monitor(self.sol)
        diffs = jnp.diff(result.ts)
        self.assertTrue(jnp.all(diffs > 0))

    def test_timestamp_spacing_matches_period(self):
        period = 2000.0
        monitor = BalloonWindkesselBold(period=period, dt_bw=1.0, voi=0)
        result = monitor(self.sol)
        if len(result.ts) > 1:
            diffs = jnp.diff(result.ts)
            for d in diffs:
                self.assertAlmostEqual(float(d), period, places=1)


class TestDeprecatedBoldAlias(unittest.TestCase):
    def test_bold_alias_warns(self):
        import warnings

        from tvboptim.observations.tvb_monitors import Bold

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            b = Bold(period=1000.0)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertIsInstance(b, HRFBold)


class TestStreamingHrfBold(unittest.TestCase):
    """The block-level streaming HRF-BOLD reducer matches the post-hoc HRFBold.

    Streaming requires a SubSampling downsample (uniform, streamable) and a
    block_size / n_steps that are multiples of the BOLD period in raw steps.
    Because a blocked SDE run streams (reseeds) its noise, the equivalence
    reference is the post-hoc monitor applied to the SAME streamed trajectory
    (matched per-block seeding).
    """

    def _net(self):
        import jax

        from tvboptim.experimental.network_dynamics import Network
        from tvboptim.experimental.network_dynamics.coupling import (
            DelayedLinearCoupling,
        )
        from tvboptim.experimental.network_dynamics.dynamics.tvb import (
            ReducedWongWang,
        )
        from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
        from tvboptim.experimental.network_dynamics.noise import AdditiveNoise

        k = jax.random.PRNGKey(7)
        wk, dk = jax.random.split(k)
        n = 4
        w = jax.random.uniform(wk, (n, n)) * 0.5
        d = jax.random.uniform(dk, (n, n)) * 5.0
        return Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": DelayedLinearCoupling(source="S", G=0.1)},
            graph=DenseDelayGraph(weights=w, delays=d),
            noise=AdditiveNoise(sigma=1e-3, key=jax.random.key(0)),
        )

    def test_matches_posthoc_hrfbold(self):
        from tvboptim.experimental.network_dynamics import solve
        from tvboptim.experimental.network_dynamics.solvers import Heun
        from tvboptim.observations.tvb_monitors import (
            HRFBold,
            SubSampling,
            streaming_hrf_bold,
        )

        net = self._net()
        dt = 0.1
        # period/dt = (200/40)*(40/0.1) = 5*400 = 2000 raw steps per block.
        mon = HRFBold(
            period=200.0,
            downsample_period=40.0,
            downsample=SubSampling(period=40.0),
        )
        t1 = 800.0  # 8000 steps, a multiple of 2000
        # Streaming reducer.
        bold = solve(
            net,
            Heun(block_size=2000),
            t0=0.0,
            t1=t1,
            dt=dt,
            reduce=streaming_hrf_bold(mon, dt),
        )
        # Post-hoc on the SAME streamed trajectory (matched per-block seeding).
        ref = mon(solve(net, Heun(block_size=2000), t0=0.0, t1=t1, dt=dt))
        self.assertEqual(bold.shape, ref.ys.shape)
        self.assertTrue(
            jnp.allclose(bold, ref.ys, atol=1e-5),
            f"max diff {jnp.max(jnp.abs(bold - ref.ys))}",
        )

    def test_misaligned_block_size_rejected(self):
        from tvboptim.experimental.network_dynamics import solve
        from tvboptim.experimental.network_dynamics.solvers import Heun
        from tvboptim.observations.tvb_monitors import (
            HRFBold,
            SubSampling,
            streaming_hrf_bold,
        )

        net = self._net()
        dt = 0.1
        mon = HRFBold(
            period=200.0, downsample_period=40.0, downsample=SubSampling(period=40.0)
        )
        # block_size=1500 is not a multiple of period/dt=2000 -> assert at trace.
        with self.assertRaises(AssertionError):
            solve(
                net,
                Heun(block_size=1500),
                t0=0.0,
                t1=600.0,
                dt=dt,
                reduce=streaming_hrf_bold(mon, dt),
            )

    def test_factory_is_deprecated(self):
        from tvboptim.observations.tvb_monitors import (
            HRFBold,
            SubSampling,
            streaming_hrf_bold,
        )

        monitor = HRFBold(
            period=200.0,
            downsample=SubSampling(period=40.0),
        )
        with self.assertWarnsRegex(DeprecationWarning, "streaming_hrf_bold"):
            streaming_hrf_bold(monitor, dt=0.1)


@pytest.mark.parametrize(
    ("downsample", "downsample_scale"),
    [
        pytest.param(CustomTemporalAverage(), 1.0, id="callable"),
        pytest.param(DoubledTemporalAverage(period=4.0), 2.0, id="built-in-subclass"),
    ],
)
def test_valid_hrf_preserves_custom_callable_downsampler_posthoc_only(
    downsample, downsample_scale
):
    kernel = FirstOrderVolterraHRFKernel(duration=20.0)
    monitor = HRFBold(
        period=1000.0,
        downsample=downsample,
        kernel=kernel,
    )
    base = jnp.linspace(0.25, 1.25, 2000).reshape(2000, 1, 1)

    def solution(scale):
        return NativeSolution(
            ts=jnp.arange(1.0, 2001.0),
            ys=base * scale,
            dt=1.0,
            variable_names=("signal",),
        )

    def independent_reference(scale):
        downsampled = TemporalAverage(period=4.0)(solution(scale))
        n_kernel = int(jnp.ceil(kernel.duration / downsampled.dt))
        hrf = kernel(jnp.linspace(0.0, kernel.duration, n_kernel), downsampled.dt)
        history = jnp.zeros((n_kernel, 1, 1), dtype=downsampled.ys.dtype)
        signal = jnp.concatenate([history, downsample_scale * downsampled.ys], axis=0)
        convolved = jsp.signal.fftconvolve(signal[:, 0, 0], hrf, mode="valid")[
            :, None, None
        ]
        stride = int(monitor.period / downsampled.dt)
        return monitor.k_1 * monitor.V_0 * (convolved[stride::stride] - 1.0)

    result = monitor(solution(jnp.asarray(1.0)))
    expected = independent_reference(jnp.asarray(1.0))
    assert jnp.allclose(result.ys, expected, rtol=1e-5, atol=1e-7)
    assert jnp.array_equal(result.ts, jnp.asarray([1000.0, 2000.0]))
    assert result.dt == 1000.0
    assert result.variable_names == ("BOLD(signal)",)

    actual_gradient = jax.grad(lambda scale: monitor(solution(scale)).ys.sum())(
        jnp.asarray(1.0)
    )
    expected_gradient = jax.grad(lambda scale: independent_reference(scale).sum())(
        jnp.asarray(1.0)
    )
    assert jnp.allclose(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-7)

    with pytest.raises(ValueError, match="Prepared HRFBold downsample"):
        prepare(
            ConstantDrive(),
            Euler(block_size=1000),
            t1=2000.0,
            dt=1.0,
            observe=monitor,
        )


@pytest.mark.parametrize("downsampler", [DoubledTemporalAverage, DoubledSubSampling])
@pytest.mark.parametrize("n_steps", [0, 1, 4003])
def test_bw_preserves_custom_downsampler_subclasses(downsampler, n_steps):
    downsample = downsampler(period=4.0, voi=1)
    drive = jnp.linspace(0.05, 0.15, n_steps)[:, None] * jnp.asarray([1.0, 1.5])
    values = jnp.stack([jnp.zeros_like(drive), drive], axis=1)
    names = ("unused", "drive") if n_steps else None

    def observed(scale, vo, origin, offset):
        solution = NativeSolution(
            ts=origin + jnp.arange(1, n_steps + 1),
            ys=values * scale,
            dt=1.0,
            variable_names=names,
        )
        monitor = BalloonWindkesselBold(period=1000.0, vo=vo, downsample=downsample)
        return monitor(solution, t_offset=offset)

    def reference(scale, vo):
        # Resolve completed four-step windows independently of monitor dispatch.
        windows = drive[: n_steps // 4 * 4].reshape((-1, 4, 2))
        sampled = (
            windows.mean(axis=1)
            if downsampler is DoubledTemporalAverage
            else windows[:, -1, :]
        )
        firing_rates = jnp.repeat(2.0 * scale * sampled, 4, axis=0)

        def step(state, rate):
            s, f, v, q = state
            next_state = (
                s + 0.001 * (rate - s / 0.65 - (f - 1.0) / 0.41),
                f + 0.001 * s,
                v + 0.001 * (f - v ** (1.0 / 0.32)) / 0.98,
                q
                + 0.001
                * (f * (1.0 - 0.6 ** (1.0 / f)) / 0.4 - v ** (1.0 / 0.32 - 1.0) * q)
                / 0.98,
            )
            _, _, next_v, next_q = next_state
            bold = vo * (
                (4.3 * 40.3 * 0.4 * 0.04) * (1.0 - next_q)
                + (25.0 * 0.4 * 0.04) * (1.0 - next_q / next_v)
                + (1.0 - next_v)
            )
            return next_state, bold

        state0 = (jnp.zeros(2), jnp.ones(2), jnp.ones(2), jnp.ones(2))
        _, bold = jax.lax.scan(step, state0, firing_rates)
        return bold[999::1000, None, :]

    scale, vo = jnp.asarray(1.0), jnp.asarray(0.04)
    origin, offset = jnp.asarray(37.0), jnp.asarray(7.0)
    result = jax.jit(observed)(scale, vo, origin, offset)
    expected = reference(scale, vo)
    assert result.ys.shape == (n_steps // 1000, 1, 2)
    assert jnp.allclose(result.ys, expected, rtol=1e-4, atol=1e-7)
    assert jnp.array_equal(
        result.ts, origin + offset + jnp.arange(1, n_steps // 1000 + 1) * 1000.0
    )
    assert result.dt == 1000.0
    assert result.variable_names == (None if names is None else ("BOLD(drive)",))

    actual_gradients = jax.grad(
        lambda scale, vo: observed(scale, vo, origin, offset).ys.sum(), argnums=(0, 1)
    )(scale, vo)
    expected_gradients = jax.grad(
        lambda scale, vo: reference(scale, vo).sum(), argnums=(0, 1)
    )(scale, vo)
    assert jnp.allclose(
        jnp.asarray(actual_gradients),
        jnp.asarray(expected_gradients),
        rtol=1e-4,
        atol=1e-7,
    )
    if n_steps >= 1000:
        assert all(float(gradient) != 0.0 for gradient in actual_gradients)

    with pytest.raises(ValueError, match="Prepared BalloonWindkesselBold downsample"):
        prepare(
            ConstantDrive(),
            Euler(block_size=1000),
            t1=float(n_steps),
            dt=1.0,
            observe=BalloonWindkesselBold(period=1000.0, downsample=downsample),
        )


@pytest.mark.parametrize("mode", ["same", "full"])
def test_hrf_specialized_posthoc_convolution_modes_remain_available(mode):
    kernel = FirstOrderVolterraHRFKernel(duration=20.0)
    solution = NativeSolution(
        ts=jnp.arange(1.0, 41.0),
        ys=jnp.linspace(0.1, 1.1, 40).reshape(40, 1, 1),
        dt=1.0,
        variable_names=("signal",),
    )
    monitor = HRFBold(
        period=8.0,
        downsample=SubSampling(period=4.0),
        kernel=kernel,
        convolution_mode=mode,
    )
    result = monitor(solution)

    downsampled = SubSampling(period=4.0)(solution)
    n_kernel = int(jnp.ceil(kernel.duration / downsampled.dt))
    hrf = kernel(jnp.linspace(0.0, kernel.duration, n_kernel), downsampled.dt)
    history = jnp.zeros((n_kernel, 1, 1), dtype=downsampled.ys.dtype)
    signal = jnp.concatenate([history, downsampled.ys], axis=0)
    convolved = jsp.signal.fftconvolve(signal[:, 0, 0], hrf, mode=mode)[:, None, None]
    expected = monitor.k_1 * monitor.V_0 * (convolved[2::2] - 1.0)

    assert jnp.allclose(result.ys, expected)
    assert jnp.array_equal(
        result.ts, (jnp.arange(expected.shape[0]) + 1) * monitor.period
    )
    assert result.variable_names == ("BOLD(signal)",)


def test_streaming_hrf_resolves_solution_valued_warm_history():
    kernel = FirstOrderVolterraHRFKernel(duration=16.0)
    history = NativeSolution(
        ts=jnp.arange(-19.0, 1.0),
        ys=jnp.linspace(0.2, 1.2, 20).reshape(20, 1, 1),
        dt=1.0,
        variable_names=("signal",),
    )
    downsample = SubSampling(period=4.0)
    array_history = downsample(history).ys
    solution_monitor = HRFBold(
        period=8.0,
        downsample=downsample,
        kernel=kernel,
        history=history,
    )
    array_monitor = HRFBold(
        period=8.0,
        downsample=downsample,
        kernel=kernel,
        history=array_history,
    )
    solver = Euler(block_size=8)
    kwargs = dict(t1=16.0, dt=1.0)

    raw = solve(ConstantDrive(), solver, **kwargs)
    posthoc_solution = solution_monitor(raw)
    posthoc_array = array_monitor(raw)
    with pytest.warns(DeprecationWarning):
        reduced_solution = solve(
            ConstantDrive(),
            solver,
            reduce=streaming_hrf_bold(solution_monitor, 1.0),
            **kwargs,
        )
    with pytest.warns(DeprecationWarning):
        reduced_array = solve(
            ConstantDrive(),
            solver,
            reduce=streaming_hrf_bold(array_monitor, 1.0),
            **kwargs,
        )

    assert posthoc_solution.ys.shape == (2, 1, 1)
    assert jnp.allclose(posthoc_solution.ys, posthoc_array.ys)
    assert jnp.array_equal(posthoc_solution.ts, posthoc_array.ts)
    assert jnp.allclose(reduced_solution, posthoc_solution.ys)
    assert jnp.allclose(reduced_array, posthoc_solution.ys)


if __name__ == "__main__":
    unittest.main()
