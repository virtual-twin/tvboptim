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


if __name__ == "__main__":
    unittest.main()
