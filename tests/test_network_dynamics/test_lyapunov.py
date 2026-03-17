"""Tests for Lyapunov exponent computation.

Uses a linear diagonal system dx_i/dt = a_i * x_i where the Lyapunov
exponents are exactly the rates a_i (eigenvalues of the system matrix).
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.analysis import lyapunov, lyapunov_spectrum
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun


# ---------------------------------------------------------------------------
# Test dynamics: linear diagonal ODE with known Lyapunov exponents
# ---------------------------------------------------------------------------

class LinearDiagonal(AbstractDynamics):
    """dx_i/dt = a_i * x_i.  Lyapunov exponents equal the rates a_i."""

    STATE_NAMES = ("x0", "x1", "x2")
    INITIAL_STATE = (1.0, 1.0, 1.0)
    DEFAULT_PARAMS = Bunch(rates=jnp.array([-0.5, -1.0, -2.0]))

    def dynamics(self, t, state, params, coupling, external):
        return params.rates[:, None] * state


def _make_network(rates):
    """Build a single-node Network with the given linear diagonal rates."""
    return Network(
        dynamics=LinearDiagonal(rates=rates),
        graph=DenseGraph(jnp.zeros((1, 1))),
        coupling={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMaxExponent(unittest.TestCase):
    """Test maximum Lyapunov exponent (single-trajectory method)."""

    def setUp(self):
        self.solver = Heun()
        self.kw = dict(t=10.0, n=50, dt=0.01)

    def test_stable_system(self):
        """Max exponent of a stable linear system matches the largest rate."""
        network = _make_network(jnp.array([-0.5, -1.0, -2.0]))
        l_max = lyapunov(network, self.solver, **self.kw)
        np.testing.assert_allclose(l_max, -0.5, atol=0.01)

    def test_unstable_system(self):
        """Max exponent of an unstable system via spectrum k=1.

        The two-trajectory lyapunov() method uses absolute perturbations
        (d0), which lose float64 precision when the reference state grows
        unboundedly. The spectrum method (tangent-space JVP) does not have
        this limitation.
        """
        network = _make_network(jnp.array([0.5, -1.0, -2.0]))
        spectrum = lyapunov_spectrum(
            network, self.solver, k=1, mode="jvp", **self.kw
        )
        self.assertGreater(float(spectrum[0]), 0.0)
        np.testing.assert_allclose(float(spectrum[0]), 0.5, atol=0.01)

    def test_marginal_system(self):
        """Max exponent near zero for a marginally stable system."""
        network = _make_network(jnp.array([0.0, -1.0, -2.0]))
        l_max = lyapunov(network, self.solver, **self.kw)
        np.testing.assert_allclose(l_max, 0.0, atol=0.01)


class TestSpectrum(unittest.TestCase):
    """Test full and partial Lyapunov spectrum computation."""

    def setUp(self):
        self.rates = jnp.array([-0.5, -1.0, -2.0])
        self.expected = jnp.sort(self.rates)[::-1]  # [-0.5, -1.0, -2.0]
        self.network = _make_network(self.rates)
        self.solver = Heun()
        self.kw = dict(t=10.0, n=50, dt=0.01)

    def test_jvp_matches_eigenvalues(self):
        """JVP spectrum matches analytic eigenvalues."""
        spectrum = lyapunov_spectrum(
            self.network, self.solver, mode="jvp", **self.kw
        )
        np.testing.assert_allclose(spectrum, self.expected, atol=0.01)

    def test_sim_matches_eigenvalues(self):
        """Sim spectrum matches analytic eigenvalues."""
        spectrum = lyapunov_spectrum(
            self.network, self.solver, mode="sim", **self.kw
        )
        np.testing.assert_allclose(spectrum, self.expected, atol=0.01)

    def test_jvp_sim_consistency(self):
        """JVP and sim modes produce consistent spectra."""
        spectrum_jvp = lyapunov_spectrum(
            self.network, self.solver, mode="jvp", **self.kw
        )
        spectrum_sim = lyapunov_spectrum(
            self.network, self.solver, mode="sim", **self.kw
        )
        np.testing.assert_allclose(spectrum_jvp, spectrum_sim, atol=0.01)

    def test_partial_spectrum_k2(self):
        """k=2 returns the top 2 exponents matching the full spectrum."""
        full = lyapunov_spectrum(
            self.network, self.solver, mode="jvp", **self.kw
        )
        partial = lyapunov_spectrum(
            self.network, self.solver, k=2, mode="jvp", **self.kw
        )
        self.assertEqual(partial.shape[0], 2)
        np.testing.assert_allclose(partial, full[:2], atol=0.01)

    def test_partial_spectrum_k1_vs_max(self):
        """k=1 spectrum should approximate the single-trajectory max exponent."""
        l_max = lyapunov(self.network, self.solver, **self.kw)
        spectrum_k1 = lyapunov_spectrum(
            self.network, self.solver, k=1, mode="jvp", **self.kw
        )
        np.testing.assert_allclose(float(spectrum_k1[0]), l_max, atol=0.02)

    def test_spectrum_sum_equals_trace(self):
        """Sum of all exponents equals the trace of the system matrix."""
        spectrum = lyapunov_spectrum(
            self.network, self.solver, mode="jvp", **self.kw
        )
        np.testing.assert_allclose(
            float(spectrum.sum()), float(self.rates.sum()), atol=0.01
        )

    def test_spectrum_with_positive_exponents(self):
        """Spectrum correctly identifies positive and negative exponents."""
        rates = jnp.array([0.3, -0.1, -1.5])
        network = _make_network(rates)
        spectrum = lyapunov_spectrum(
            network, self.solver, mode="jvp", **self.kw
        )
        expected = jnp.sort(rates)[::-1]
        np.testing.assert_allclose(spectrum, expected, atol=0.01)
        self.assertGreater(float(spectrum[0]), 0.0)
        self.assertLess(float(spectrum[-1]), 0.0)

    def test_invalid_mode_raises(self):
        """Invalid mode keyword raises ValueError."""
        with self.assertRaises(ValueError):
            lyapunov_spectrum(
                self.network, self.solver, mode="invalid", dt=0.01
            )


if __name__ == "__main__":
    unittest.main()
