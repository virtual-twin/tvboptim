"""Tests for HRFBold and BalloonWindkesselBold monitors.

Checks output shape, period, and timestamp correctness using dummy/random
NativeSolution inputs.
"""

import unittest

import jax.numpy as jnp

from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.observations.tvb_monitors import BalloonWindkesselBold, HRFBold


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
            coupling={"delayed": DelayedLinearCoupling(incoming_states="S", G=0.1)},
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


if __name__ == "__main__":
    unittest.main()
