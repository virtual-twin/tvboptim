"""Tests for tvboptim.analysis.identifiability (Phase 1: loss Hessian).

The two analytic cases pin down correctness without a full simulation:

* a quadratic loss with a known Hessian -- eigenvalues must match exactly;
* a deliberately degenerate loss that depends only on ``a + b`` -- the
  analysis must report one zero eigenvalue and identify ``a - b`` as the
  flat (non-identifiable) direction.
"""

import unittest
import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tvboptim.analysis import (
    analyze_identifiability,
    eigendecompose_curvature,
    fisher_information,
    loss_hessian,
)
from tvboptim.analysis.identifiability import IdentifiabilityResult
from tvboptim.types import Parameter


class TestQuadraticHessian(unittest.TestCase):
    """L = 0.5 x^T A x  =>  Hessian = A, eigenvalues = eig(A)."""

    def test_diagonal_quadratic(self):
        A = jnp.diag(jnp.array([1.0, 2.0, 5.0]))

        def loss(state):
            x = state["x"].value
            return 0.5 * x @ A @ x

        state = {"x": Parameter(jnp.array([0.3, -0.7, 1.1]))}
        H, theta0, labels, grad_norm = loss_hessian(loss, state)

        np.testing.assert_allclose(np.asarray(H), np.asarray(A), atol=1e-9)
        self.assertEqual(labels, ["x[0]", "x[1]", "x[2]"])
        self.assertEqual(theta0.shape, (3,))
        # Pure quadratic: gradient A@x is non-zero away from the origin.
        self.assertGreater(grad_norm, 0.0)

    def test_dense_symmetric_quadratic_eigenvalues(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((4, 4))
        A = jnp.asarray(M @ M.T + 4.0 * np.eye(4))  # symmetric positive definite

        def loss(state):
            x = state["x"].value
            return 0.5 * x @ A @ x

        state = {"x": Parameter(jnp.zeros(4))}  # at the optimum
        res = analyze_identifiability(loss, state)

        expected = np.sort(np.linalg.eigvalsh(np.asarray(A)))
        np.testing.assert_allclose(
            np.sort(np.asarray(res.eigenvalues)), expected, atol=1e-8
        )
        self.assertEqual(res.rank(), 4)
        self.assertIsInstance(res, IdentifiabilityResult)


class TestDegenerateDirection(unittest.TestCase):
    """L = (a + b - 1)^2 depends only on a+b => a-b is non-identifiable."""

    @staticmethod
    def _loss(state):
        a = state["a"].value
        b = state["b"].value
        return (a + b - 1.0) ** 2

    def test_flat_direction_detected(self):
        # a + b = 1 -> sits exactly at the bottom of the degenerate ridge.
        state = {"a": Parameter(0.3), "b": Parameter(0.7)}
        res = analyze_identifiability(self._loss, state)

        # Hessian = 2 * [[1,1],[1,1]] -> eigenvalues {0, 4}.
        evals = np.sort(np.asarray(res.eigenvalues))
        self.assertAlmostEqual(evals[0], 0.0, places=7)
        self.assertAlmostEqual(evals[1], 4.0, places=7)

        # One flat direction -> rank 1, condition number blows up.
        self.assertEqual(res.rank(), 1)
        cond = res.condition_number()
        self.assertTrue(np.isinf(cond) or cond > 1e6)

        # Flattest direction is a - b: raising a while lowering b is free.
        flat = res.sloppy_directions(1)[0]
        self.assertAlmostEqual(flat["eigenvalue"], 0.0, places=7)
        ia, ib = res.labels.index("a"), res.labels.index("b")
        vec = np.array([res.eigenvectors[ia, 0], res.eigenvectors[ib, 0]])
        vec /= np.linalg.norm(vec)
        cos = abs(vec @ (np.array([1.0, -1.0]) / np.sqrt(2.0)))
        self.assertAlmostEqual(cos, 1.0, places=6)

    def test_labels_track_dict_keys(self):
        state = {"a": Parameter(0.3), "b": Parameter(0.7)}
        _, _, labels, _ = loss_hessian(self._loss, state)
        self.assertEqual(sorted(labels), ["a", "b"])

    def test_summary_runs(self):
        state = {"a": Parameter(0.3), "b": Parameter(0.7)}
        res = analyze_identifiability(self._loss, state)
        text = res.summary()
        self.assertIn("Identifiability analysis", text)
        self.assertIn("condition number", text)


class TestGradientWarning(unittest.TestCase):
    """analyze_identifiability warns when the point is not an optimum."""

    def test_warns_away_from_optimum(self):
        def loss(state):
            return jnp.sum(state["x"].value ** 2)

        state = {"x": Parameter(jnp.array([3.0]))}  # gradient = 6, not optimal
        with self.assertWarns(UserWarning):
            analyze_identifiability(loss, state)

    def test_no_warn_at_optimum(self):
        def loss(state):
            return jnp.sum(state["x"].value ** 2)

        state = {"x": Parameter(jnp.array([0.0]))}  # gradient = 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning -> test failure
            analyze_identifiability(loss, state)

    def test_warns_when_sigma_given_without_model(self):
        def loss(state):
            return jnp.sum(state["x"].value ** 2)

        state = {"x": Parameter(jnp.array([0.0]))}
        with self.assertWarns(UserWarning):
            analyze_identifiability(loss, state, sigma=0.5)


class TestEdgeCases(unittest.TestCase):
    def test_no_parameters_raises(self):
        def loss(state):
            return jnp.sum(state["x"] ** 2)

        with self.assertRaises(ValueError):
            loss_hessian(loss, {"x": jnp.array([1.0])})

    def test_eigendecompose_curvature_directly(self):
        # eigh of [[2,0],[0,8]] -> eigenvalues [2, 8].
        res = eigendecompose_curvature(
            jnp.array([[2.0, 0.0], [0.0, 8.0]]),
            labels=["p", "q"], theta0=jnp.zeros(2), kind="hessian",
        )
        np.testing.assert_allclose(np.asarray(res.eigenvalues), [2.0, 8.0])
        self.assertEqual(res.condition_number(), 4.0)


class TestFisherInformation(unittest.TestCase):
    """Gauss-Newton FIM from a vector-valued model output."""

    @staticmethod
    def _linear_problem(seed=1):
        """A linear residual r(x) = M x - d and the matching scalar loss."""
        rng = np.random.default_rng(seed)
        M = jnp.asarray(rng.standard_normal((6, 3)))
        d = jnp.asarray(rng.standard_normal(6))

        def residual(state):
            return M @ state["x"].value - d

        def loss(state):
            return 0.5 * jnp.sum(residual(state) ** 2)

        state = {"x": Parameter(jnp.array([0.5, -1.0, 2.0]))}
        return M, d, residual, loss, state

    def test_fim_is_jtj(self):
        M, _, residual, _, state = self._linear_problem()
        FIM, theta0, labels = fisher_information(residual, state)
        np.testing.assert_allclose(
            np.asarray(FIM), np.asarray(M).T @ np.asarray(M), atol=1e-9
        )
        self.assertEqual(labels, ["x[0]", "x[1]", "x[2]"])
        self.assertEqual(theta0.shape, (3,))

    def test_fim_equals_loss_hessian_for_linear_model(self):
        # For a linear model the residual-curvature term vanishes, so the
        # Gauss-Newton FIM and the loss Hessian of 0.5||r||^2 are identical.
        _, _, residual, loss, state = self._linear_problem()
        H, _, _, _ = loss_hessian(loss, state)
        FIM, _, _ = fisher_information(residual, state)
        np.testing.assert_allclose(np.asarray(H), np.asarray(FIM), atol=1e-9)

    def test_sigma_scaling(self):
        _, _, residual, _, state = self._linear_problem()
        FIM1, _, _ = fisher_information(residual, state, sigma=1.0)
        FIM2, _, _ = fisher_information(residual, state, sigma=2.0)
        np.testing.assert_allclose(
            np.asarray(FIM2), np.asarray(FIM1) / 4.0, atol=1e-9
        )

    def test_per_observation_sigma(self):
        M, _, residual, _, state = self._linear_problem()
        sig = jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        FIM, _, _ = fisher_information(residual, state, sigma=sig)
        Mw = np.asarray(M) / np.asarray(sig)[:, None]
        np.testing.assert_allclose(np.asarray(FIM), Mw.T @ Mw, atol=1e-9)

    def test_fwd_and_rev_modes_agree(self):
        _, _, residual, _, state = self._linear_problem()
        FIM_fwd, _, _ = fisher_information(residual, state, mode="fwd")
        FIM_rev, _, _ = fisher_information(residual, state, mode="rev")
        np.testing.assert_allclose(
            np.asarray(FIM_fwd), np.asarray(FIM_rev), atol=1e-9
        )

    def test_scalar_model_warns(self):
        _, _, _, _, state = self._linear_problem()

        def scalar_model(s):
            return jnp.sum(s["x"].value ** 2)

        with self.assertWarns(UserWarning):
            fisher_information(scalar_model, state)

    def test_degenerate_model_is_rank_deficient(self):
        # Every output depends only on a + b -> a - b is non-identifiable.
        def model(state):
            s = state["a"].value + state["b"].value
            return jnp.array([s, 2.0 * s, 3.0 * s])

        state = {"a": Parameter(0.3), "b": Parameter(0.7)}
        FIM, theta0, labels = fisher_information(model, state)
        res = eigendecompose_curvature(FIM, labels, theta0, kind="fisher")

        self.assertEqual(res.rank(), 1)
        flat = res.sloppy_directions(1)[0]
        ia, ib = res.labels.index("a"), res.labels.index("b")
        vec = np.array([res.eigenvectors[ia, 0], res.eigenvectors[ib, 0]])
        vec /= np.linalg.norm(vec)
        cos = abs(vec @ (np.array([1.0, -1.0]) / np.sqrt(2.0)))
        self.assertAlmostEqual(cos, 1.0, places=6)

    def test_analyze_identifiability_with_model(self):
        _, _, residual, loss, state = self._linear_problem()
        # warn_gradient_tol high: this fixture is not at an optimum, and this
        # test exercises the model= path, not the gradient check.
        res = analyze_identifiability(
            loss, state, model=residual, warn_gradient_tol=1e9
        )

        self.assertEqual(res.kind, "fisher")
        self.assertIn("Fisher information", res.summary())
        FIM, _, _ = fisher_information(residual, state)
        np.testing.assert_allclose(
            np.sort(np.asarray(res.eigenvalues)),
            np.sort(np.linalg.eigvalsh(np.asarray(FIM))),
            atol=1e-9,
        )


class TestStimulationExampleIntegration(unittest.TestCase):
    """End-to-end on a real single-node forward model.

    The AD Gauss-Newton Fisher information must agree with an independent
    finite-difference loss Hessian, and must expose the amplitude /
    excitability degeneracy ridge that the stimulation example is built on.
    """

    def test_fisher_matches_finite_difference(self):
        from jax.flatten_util import ravel_pytree

        from tvboptim.analysis import eigendecompose_curvature, fisher_information
        from tvboptim.experimental.network_dynamics import Network, prepare
        from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
        from tvboptim.experimental.network_dynamics.dynamics.tvb import (
            Generic2dOscillator,
        )
        from tvboptim.experimental.network_dynamics.external_input import PulseInput
        from tvboptim.experimental.network_dynamics.graph import DenseGraph
        from tvboptim.experimental.network_dynamics.solvers import Heun
        from tvboptim.types import combine_state, partition_state

        T1, DT, ONSET, DURATION = 150.0, 0.2, 10.0, 1.0
        TRUE_AMP, TRUE_EXC = 0.4, 0.1
        dyn = dict(a=-1.5, b=-15.0, c=0.0, d=0.015, e=3.0, f=1.0, tau=4.0)

        def build(amp, exc):
            return Network(
                dynamics=Generic2dOscillator(
                    **dyn, I=exc, VARIABLES_OF_INTEREST=("V",)
                ),
                coupling={"instant": LinearCoupling(incoming_states="V", G=0.0)},
                graph=DenseGraph(jnp.zeros((1, 1))),
                external_input={
                    "stimulus": PulseInput(
                        onset=ONSET, duration=DURATION, amplitude=amp
                    )
                },
            )

        sf, cfg = prepare(build(TRUE_AMP, TRUE_EXC), Heun(), t0=0.0, t1=T1, dt=DT)
        full = sf(cfg)
        obs_idx = jnp.arange(0, len(full.ts), 15)
        # Noiseless target -> residual is exactly 0 at the true parameters, so
        # the Gauss-Newton FIM equals the loss Hessian there (no residual term).
        target = full.ys[obs_idx, 0, 0]

        cfg.external.stimulus.amplitude = Parameter(TRUE_AMP)
        cfg.dynamics.I = Parameter(TRUE_EXC)

        def model(c):
            return sf(c).ys[obs_idx, 0, 0]

        # AD Gauss-Newton Fisher information.
        FIM, theta0, labels = fisher_information(model, cfg)
        res = eigendecompose_curvature(FIM, labels, theta0, kind="fisher")

        # Independent finite-difference loss Hessian (no autodiff).
        diff0, static = partition_state(cfg)
        theta_fd, unravel = ravel_pytree(diff0)

        def loss_theta(theta):
            pred = sf(combine_state(unravel(theta), static)).ys[obs_idx, 0, 0]
            return float(jnp.mean((pred - target) ** 2))

        h = 1e-3
        t = np.asarray(theta_fd)
        ea, eb = np.array([h, 0.0]), np.array([0.0, h])
        haa = (loss_theta(t + ea) - 2 * loss_theta(t) + loss_theta(t - ea)) / h**2
        hbb = (loss_theta(t + eb) - 2 * loss_theta(t) + loss_theta(t - eb)) / h**2
        hab = (
            loss_theta(t + ea + eb)
            - loss_theta(t + ea - eb)
            - loss_theta(t - ea + eb)
            + loss_theta(t - ea - eb)
        ) / (4 * h**2)
        H_fd = np.array([[haa, hab], [hab, hbb]])
        _, evecs_fd = np.linalg.eigh(H_fd)

        # The FIM and the FD loss Hessian must share the flat direction.
        v_fim = np.asarray(res.eigenvectors[:, 0])
        v_fd = evecs_fd[:, 0]
        cos = abs(
            v_fim @ v_fd / (np.linalg.norm(v_fim) * np.linalg.norm(v_fd))
        )
        self.assertGreater(cos, 0.95)

        # The amplitude / excitability degeneracy is a genuine sloppy ridge.
        self.assertGreater(res.condition_number(), 50.0)
        self.assertEqual(res.labels, ["dynamics.I", "external.stimulus.amplitude"])


if __name__ == "__main__":
    unittest.main()
