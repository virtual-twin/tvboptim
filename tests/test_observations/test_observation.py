"""Tests for FC / FCD observations and distribution distances.

Covers compute_fc, compute_fcd, fcd_distribution, wasserstein_1d, ks_distance,
fc_corr, rmse. Parametrised over input variants (raw 2D/3D/4D arrays and
NativeSolution) where the function's surface accepts more than one shape.
"""

import unittest

import jax
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.observations import (
    compute_fc,
    compute_fcd,
    fc_corr,
    fcd_distribution,
    ks_distance,
    rmse,
    wasserstein_1d,
)


T = 200
N = 6
SEED = 0


def _make_bold(t=T, n=N, n_vars=1, n_modes=1, seed=SEED):
    """Random BOLD-like signal in several shapes for parametrised inputs.

    Returns a dict with all four surfaces compute_fc / compute_fcd accept:
    raw 2D, 3D, 4D arrays, and a NativeSolution wrapping the 3D form.
    """
    key = jax.random.PRNGKey(seed)
    arr_3d = jax.random.normal(key, shape=(t, n_vars, n))
    arr_2d = arr_3d[:, 0, :]
    arr_4d = arr_3d[..., jnp.newaxis].repeat(n_modes, axis=-1)
    sol = NativeSolution(ts=jnp.arange(t).astype(float), ys=arr_3d, dt=1.0)
    return {"2d": arr_2d, "3d": arr_3d, "4d": arr_4d, "solution": sol}


class TestComputeFC(unittest.TestCase):
    def setUp(self):
        self.inputs = _make_bold()

    def test_shape_and_zero_diagonal(self):
        for name, ts in self.inputs.items():
            with self.subTest(input=name):
                fc = compute_fc(ts)
                self.assertEqual(fc.shape, (N, N))
                self.assertTrue(jnp.allclose(jnp.diag(fc), 0.0))

    def test_symmetric(self):
        for name, ts in self.inputs.items():
            with self.subTest(input=name):
                fc = compute_fc(ts)
                self.assertTrue(jnp.allclose(fc, fc.T, atol=1e-6))

    def test_skip_t_changes_result(self):
        ts = self.inputs["2d"]
        fc_full = compute_fc(ts)
        fc_skipped = compute_fc(ts, skip_t=T // 2)
        self.assertFalse(jnp.allclose(fc_full, fc_skipped))

    def test_perfectly_correlated_signals(self):
        signal = jax.random.normal(jax.random.PRNGKey(SEED), shape=(T,))
        ts = jnp.broadcast_to(signal[:, None], (T, N))
        fc = compute_fc(ts)
        off_diag = fc[jnp.triu_indices(N, k=1)]
        self.assertTrue(jnp.allclose(off_diag, 1.0, atol=1e-5))


class TestComputeFCD(unittest.TestCase):
    t_window = 30
    step = 5

    def setUp(self):
        self.inputs = _make_bold()
        self.expected_n_windows = (T - self.t_window) // self.step + 1

    def test_shapes(self):
        for name, ts in self.inputs.items():
            with self.subTest(input=name):
                fcd, fcs = compute_fcd(ts, self.t_window, self.step)
                self.assertEqual(fcd.shape, (self.expected_n_windows,) * 2)
                self.assertEqual(fcs.shape, (self.expected_n_windows, N, N))

    def test_fcd_diagonal_is_one(self):
        # A window's per-pair FC vector is perfectly correlated with itself.
        fcd, _ = compute_fcd(self.inputs["2d"], self.t_window, self.step)
        self.assertTrue(jnp.allclose(jnp.diag(fcd), 1.0, atol=1e-5))

    def test_fcd_symmetric(self):
        fcd, _ = compute_fcd(self.inputs["2d"], self.t_window, self.step)
        self.assertTrue(jnp.allclose(fcd, fcd.T, atol=1e-6))

    def test_skip_t_reduces_n_windows(self):
        skip = 50
        fcd, _ = compute_fcd(self.inputs["2d"], self.t_window, self.step, skip_t=skip)
        expected = (T - skip - self.t_window) // self.step + 1
        self.assertEqual(fcd.shape, (expected, expected))

    def test_all_input_surfaces_agree(self):
        ref, _ = compute_fcd(self.inputs["2d"], self.t_window, self.step)
        for name in ("3d", "4d", "solution"):
            with self.subTest(input=name):
                fcd, _ = compute_fcd(self.inputs[name], self.t_window, self.step)
                self.assertTrue(jnp.allclose(fcd, ref, atol=1e-6))


class TestFCDDistribution(unittest.TestCase):
    def setUp(self):
        self.inputs = _make_bold()
        self.fcd, _ = compute_fcd(self.inputs["2d"], 30, 5)

    def test_default_grid_length(self):
        density = fcd_distribution(self.fcd)
        self.assertEqual(density.shape, (100,))

    def test_custom_grid_length(self):
        for nbins in [25, 50, 200]:
            with self.subTest(nbins=nbins):
                midpoints = jnp.linspace(-0.99, 0.99, nbins)
                density = fcd_distribution(self.fcd, midpoints)
                self.assertEqual(density.shape, (nbins,))

    def test_normalised_density_integrates_to_one(self):
        midpoints = jnp.linspace(-0.99, 0.99, 200)
        density = fcd_distribution(self.fcd, midpoints, normalize=True)
        dx = midpoints[1] - midpoints[0]
        integral = jnp.sum(density) * dx
        self.assertAlmostEqual(float(integral), 1.0, places=5)

    def test_density_is_non_negative(self):
        density = fcd_distribution(self.fcd)
        self.assertTrue(jnp.all(density >= 0))


class TestDistributionDistances(unittest.TestCase):
    def setUp(self):
        self.x = jnp.linspace(-1.0, 1.0, 200)
        self.p = jnp.exp(-(self.x ** 2) / 0.1)
        self.q = jnp.exp(-((self.x - 0.3) ** 2) / 0.1)

    def test_w1_self_distance_is_zero(self):
        self.assertAlmostEqual(float(wasserstein_1d(self.p, self.p, self.x)), 0.0, places=6)

    def test_w1_symmetric(self):
        self.assertAlmostEqual(
            float(wasserstein_1d(self.p, self.q, self.x)),
            float(wasserstein_1d(self.q, self.p, self.x)),
            places=6,
        )

    def test_w1_increases_with_shift(self):
        shifts = [0.05, 0.15, 0.30, 0.50]
        dists = [
            float(wasserstein_1d(self.p, jnp.exp(-((self.x - s) ** 2) / 0.1), self.x))
            for s in shifts
        ]
        for a, b in zip(dists, dists[1:]):
            self.assertLess(a, b)

    def test_w1_invariant_to_input_scale(self):
        d_norm = wasserstein_1d(self.p, self.q, self.x)
        d_scaled = wasserstein_1d(self.p * 7.0, self.q * 0.1, self.x)
        self.assertAlmostEqual(float(d_norm), float(d_scaled), places=6)

    def test_ks_self_distance_is_zero(self):
        self.assertAlmostEqual(float(ks_distance(self.p, self.p)), 0.0, places=6)

    def test_ks_bounded_in_unit_interval(self):
        d = float(ks_distance(self.p, self.q))
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)

    def test_ks_invariant_to_input_scale(self):
        d_norm = ks_distance(self.p, self.q)
        d_scaled = ks_distance(self.p * 7.0, self.q * 0.1)
        self.assertAlmostEqual(float(d_norm), float(d_scaled), places=6)


class TestFCCorrAndRMSE(unittest.TestCase):
    def setUp(self):
        key = jax.random.PRNGKey(SEED)
        self.fc1 = jax.random.normal(key, shape=(N, N))
        self.fc2 = jax.random.normal(jax.random.PRNGKey(SEED + 1), shape=(N, N))

    def test_fc_corr_self_is_one(self):
        self.assertAlmostEqual(float(fc_corr(self.fc1, self.fc1)), 1.0, places=5)

    def test_fc_corr_bounded(self):
        r = float(fc_corr(self.fc1, self.fc2))
        self.assertGreaterEqual(r, -1.0)
        self.assertLessEqual(r, 1.0)

    def test_rmse_self_is_zero(self):
        self.assertAlmostEqual(float(rmse(self.fc1, self.fc1)), 0.0, places=6)

    def test_rmse_matches_formula(self):
        expected = float(jnp.sqrt(jnp.mean((self.fc1 - self.fc2) ** 2)))
        self.assertAlmostEqual(float(rmse(self.fc1, self.fc2)), expected, places=6)


if __name__ == "__main__":
    unittest.main()
