"""Tests for differentiable (interpolated) delays and DenseDelayGraph(max_delay).

Covers the two features added in this change:

* ``DelayedCoupling(interpolate_delays=True)`` — reads delayed states by linear
  interpolation between the two bracketing history steps instead of snapping to
  the nearest step, making the coupling differentiable w.r.t. the continuous
  delay (and hence w.r.t. conduction speed, since ``delay = length / speed``).
* ``DenseDelayGraph(max_delay=...)`` — sizes the (static) history buffer
  explicitly so the ``delays`` matrix may be a JAX tracer, enabling
  gradient-based optimisation while the buffer length stays static.

The whole path (graph construction + ``prepare`` + ``solve``) is exercised under
both bare ``jax.grad`` and ``jax.jit``.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import optax

from tvboptim.experimental.network_dynamics import Network, prepare, solve
from tvboptim.experimental.network_dynamics.coupling import DelayedLinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import Generic2dOscillator
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
from tvboptim.experimental.network_dynamics.solvers import Heun

DT = 0.5
T1 = 40.0


def _osc():
    return Generic2dOscillator(a=-1.5, b=-15.0, d=0.015, tau=4.0, INITIAL_STATE=(0.1, 0.1))


def _conn(n, seed):
    """Random weights + symmetric tract-length matrix of size n."""
    rng = np.random.default_rng(seed)
    w = np.where(np.eye(n), 0.0, rng.uniform(0.0, 0.1, (n, n)))
    length = rng.uniform(10.0, 80.0, (n, n))
    length = 0.5 * (length + length.T)
    np.fill_diagonal(length, 0.0)
    return jnp.asarray(w), jnp.asarray(length)


def _net_from_graph(graph, interpolate, G=0.3):
    coup = DelayedLinearCoupling(incoming_states="V", G=G, interpolate_delays=interpolate)
    return Network(_osc(), {"delayed": coup}, graph)


def _run_graph(graph, interpolate, G=0.3):
    return solve(_net_from_graph(graph, interpolate, G), Heun(), t0=0.0, t1=T1, dt=DT).ys


def _run(weights, delays, interpolate, max_delay, G=0.3):
    return _run_graph(DenseDelayGraph(weights, delays, max_delay=max_delay), interpolate, G)


class TestForward(unittest.TestCase):
    """Forward simulation with interpolation across network sizes."""

    def test_finite_and_shaped(self):
        for n in (2, 8, 32):
            with self.subTest(n=n):
                w, length = _conn(n, seed=n)
                delays = length / 3.0
                md = float(jnp.max(delays))
                ys = _run(w, delays, interpolate=True, max_delay=md)
                self.assertEqual(ys.shape[-1], n)
                self.assertTrue(bool(jnp.all(jnp.isfinite(ys))))

    def test_interpolation_is_bounded_correction(self):
        for n in (2, 8, 32):
            with self.subTest(n=n):
                w, length = _conn(n, seed=100 + n)
                delays = length / 3.0
                md = float(jnp.max(delays))
                snap = _run(w, delays, interpolate=False, max_delay=md)
                interp = _run(w, delays, interpolate=True, max_delay=md)
                self.assertTrue(bool(jnp.all(jnp.isfinite(interp))))
                diff = float(jnp.max(jnp.abs(interp - snap)))
                scale = float(jnp.mean(jnp.abs(snap))) + 1e-9
                self.assertLess(diff, 5.0 * scale)  # a bounded correction

    def test_interpolation_is_active_off_grid(self):
        """Half-step-offset delays (frac==0.5) must differ from nearest-step."""
        w, length = _conn(8, seed=42)
        snapped = jnp.round((length / 3.0) / DT)
        delays = (snapped + 0.5) * DT  # exactly half a step off the grid
        md = float(jnp.max(delays))
        snap = _run(w, delays, interpolate=False, max_delay=md)
        interp = _run(w, delays, interpolate=True, max_delay=md)
        self.assertGreater(float(jnp.max(jnp.abs(interp - snap))), 0.0)

    def test_reduces_to_snapping_on_grid(self):
        """When every delay is an exact multiple of dt, frac==0 -> identical to snap."""
        w, length = _conn(8, seed=7)
        delays = jnp.round((length / 3.0) / DT) * DT  # exact grid points
        md = float(jnp.max(delays))
        snap = _run(w, delays, interpolate=False, max_delay=md)
        interp = _run(w, delays, interpolate=True, max_delay=md)
        np.testing.assert_allclose(np.asarray(interp), np.asarray(snap), rtol=0, atol=1e-12)


class TestConstructorGuards(unittest.TestCase):
    def test_interpolate_requires_roll(self):
        for strategy in ("circular", "preallocated"):
            with self.subTest(strategy=strategy):
                with self.assertRaises(ValueError):
                    DelayedLinearCoupling(
                        incoming_states="V",
                        buffer_strategy=strategy,
                        interpolate_delays=True,
                    )

    def test_non_roll_without_interpolate_allowed(self):
        for strategy in ("roll", "circular", "preallocated"):
            with self.subTest(strategy=strategy):
                DelayedLinearCoupling(incoming_states="V", buffer_strategy=strategy)

    def test_roll_with_interpolate_allowed(self):
        DelayedLinearCoupling(
            incoming_states="V", buffer_strategy="roll", interpolate_delays=True
        )


class TestMaxDelay(unittest.TestCase):
    def test_default_derived_from_delays(self):
        w, length = _conn(6, seed=3)
        delays = length / 3.0
        graph = DenseDelayGraph(w, delays)
        self.assertAlmostEqual(graph.max_delay, float(jnp.max(delays)), places=10)

    def test_explicit_overrides_and_sizes_buffer(self):
        w, length = _conn(6, seed=3)
        delays = length / 3.0
        explicit = float(jnp.max(delays)) * 2.0
        graph = DenseDelayGraph(w, delays, max_delay=explicit)
        self.assertAlmostEqual(graph.max_delay, explicit, places=10)
        # An explicit (larger-than-needed) max_delay still simulates correctly.
        ys = _run(w, delays, interpolate=True, max_delay=explicit)
        self.assertTrue(bool(jnp.all(jnp.isfinite(ys))))

    def test_tracer_delays_require_max_delay(self):
        w, length = _conn(6, seed=3)
        # Without max_delay, a tracer ``delays`` cannot size the static buffer.
        with self.assertRaises(Exception):
            jax.grad(lambda v: jnp.sum(DenseDelayGraph(w, length / v).delays))(jnp.asarray(3.0))
        # With max_delay set, it is fine and differentiable.
        md = float(jnp.max(length)) / 2.0
        g = jax.grad(lambda v: jnp.sum(DenseDelayGraph(w, length / v, max_delay=md).delays))(
            jnp.asarray(3.0)
        )
        self.assertTrue(np.isfinite(float(g)))
        self.assertNotEqual(float(g), 0.0)


class TestWithDelays(unittest.TestCase):
    def test_replaces_delays_preserving_static_structure(self):
        w, length = _conn(6, seed=9)
        g = DenseDelayGraph(w, length / 3.0, max_delay=50.0)
        g2 = g.with_delays(length / 5.0)
        np.testing.assert_allclose(np.asarray(g2.delays), np.asarray(length / 5.0))
        self.assertEqual(g2.max_delay, 50.0)  # static buffer bound preserved
        self.assertEqual(g2.n_nodes, g.n_nodes)
        # original graph is unchanged (immutable update)
        np.testing.assert_allclose(np.asarray(g.delays), np.asarray(length / 3.0))

    def test_with_delays_is_jit_safe(self):
        w, length = _conn(6, seed=9)
        g = DenseDelayGraph(w, length / 3.0, max_delay=50.0)
        # rebuilding via with_delays inside jit must not trigger __init__/verify
        delays = jax.jit(lambda v: g.with_delays(length / v).delays)(jnp.asarray(4.0))
        np.testing.assert_allclose(np.asarray(delays), np.asarray(length / 4.0))


class TestDifferentiability(unittest.TestCase):
    """Gradient of a loss w.r.t. conduction speed, validated against finite diff."""

    def setUp(self):
        self.w, self.length = _conn(6, seed=0)
        self.max_delay = float(jnp.max(self.length)) / 2.0  # valid for speed >= 2
        self.true_speed = 3.0
        # Build the graph once (concrete) to fix the static buffer via max_delay,
        # then vary the delays with with_delays() — never reconstructed under jit.
        self.graph = DenseDelayGraph(
            self.w, self.length / self.true_speed, max_delay=self.max_delay
        )
        self.target = _run_graph(self.graph, True, G=0.4)

    def _loss(self, speed, interpolate):
        g = self.graph.with_delays(self.length / speed)
        return jnp.mean((_run_graph(g, interpolate, G=0.4) - self.target) ** 2)

    def test_grad_matches_finite_difference(self):
        s0 = 4.0
        ad = float(jax.grad(lambda s: self._loss(s, True))(jnp.asarray(s0)))
        fd = float((self._loss(s0 + 1e-3, True) - self._loss(s0 - 1e-3, True)) / 2e-3)
        self.assertNotEqual(ad, 0.0)
        self.assertLess(abs(ad - fd) / (abs(fd) + 1e-30), 1e-3)

    def test_snap_gives_zero_gradient(self):
        ad = float(jax.grad(lambda s: self._loss(s, False))(jnp.asarray(4.0)))
        self.assertLess(abs(ad), 1e-12)  # nearest-step is piecewise constant in speed

    def test_jit_grad_matches_finite_difference(self):
        s0 = 4.0
        ad = float(jax.jit(jax.grad(lambda s: self._loss(s, True)))(jnp.asarray(s0)))
        fd = float((self._loss(s0 + 1e-3, True) - self._loss(s0 - 1e-3, True)) / 2e-3)
        self.assertLess(abs(ad - fd) / (abs(fd) + 1e-30), 1e-3)

    def test_jitted_optimization_recovers_speed(self):
        loss = lambda s: self._loss(s, True)
        grad = jax.jit(jax.grad(loss))
        v = jnp.asarray(4.0)
        opt = optax.adam(0.1)
        st = opt.init(v)
        l0 = float(loss(v))
        for _ in range(60):
            updates, st = opt.update(grad(v), st)
            v = optax.apply_updates(v, updates)
        self.assertLess(float(loss(v)), l0)
        # genuinely recovered, not merely nudged toward the target
        self.assertLess(abs(float(v) - self.true_speed), 0.2)


class TestHeterogeneousDelayEquation(unittest.TestCase):
    """delays = offset + L / speed_per_source_node -> heterogeneous, multi-parameter."""

    def setUp(self):
        self.w, self.length = _conn(6, seed=11)
        self.n = 6
        self.max_delay = float(jnp.max(self.length)) / 1.5
        self.graph = DenseDelayGraph(self.w, self.length / 3.0, max_delay=self.max_delay)

    def _delays(self, theta):
        return theta["offset"] + self.length / theta["speed"][None, :]

    def _loss(self, theta, target):
        ys = _run_graph(self.graph.with_delays(self._delays(theta)), True, G=0.4)
        return jnp.mean((ys - target) ** 2)

    def test_grad_flows_through_vector_speed_and_offset(self):
        true = {
            "speed": jnp.full(self.n, 3.0).at[: self.n // 2].set(2.0),
            "offset": jnp.asarray(0.5),
        }
        target = _run(self.w, self._delays(true), True, self.max_delay, G=0.4)
        th0 = {"speed": jnp.full(self.n, 3.0), "offset": jnp.asarray(0.0)}
        g = jax.jit(jax.grad(lambda t: self._loss(t, target)))(th0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(g["speed"]))))
        self.assertGreater(float(jnp.max(jnp.abs(g["speed"]))), 0.0)
        self.assertTrue(np.isfinite(float(g["offset"])))


if __name__ == "__main__":
    unittest.main()
