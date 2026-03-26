"""Test bare dynamics simulation (no network/coupling)."""

import unittest

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import solve
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    Generic2dOscillator,
    JansenRit,
    Kuramoto,
    ReducedWongWang,
)
from tvboptim.experimental.network_dynamics.external_input.parametric import SineInput
from tvboptim.experimental.network_dynamics.noise.gaussian import AdditiveNoise
from tvboptim.experimental.network_dynamics.solve import prepare
from tvboptim.experimental.network_dynamics.solvers import Euler, Heun
from tvboptim.experimental.network_dynamics.solvers.diffrax import DiffraxSolver


class TestBareDynamics(unittest.TestCase):
    """Test solve/prepare with bare AbstractDynamics (no Network)."""

    def setUp(self):
        self.t0 = 0.0
        self.t1 = 10.0
        self.dt = 0.01
        self.n_timesteps = int((self.t1 - self.t0) / self.dt)

        self.models = [
            ("rww", ReducedWongWang, 1),
            ("jr", JansenRit, 6),
            ("g2do", Generic2dOscillator, 2),
            ("kuramoto", Kuramoto, 1),
        ]

    def test_native_solver_single_node(self):
        """Test each model with native solvers, single node."""
        for model_name, model_class, n_states in self.models:
            for solver_name, solver in [("euler", Euler()), ("heun", Heun())]:
                with self.subTest(model=model_name, solver=solver_name):
                    result = solve(
                        model_class(), solver, t0=self.t0, t1=self.t1, dt=self.dt
                    )

                    self.assertEqual(result.ts.shape, (self.n_timesteps,))
                    self.assertEqual(result.ys.shape, (self.n_timesteps, n_states, 1))
                    self.assertFalse(jnp.isnan(result.ys).any())
                    self.assertFalse(jnp.isinf(result.ys).any())

    def test_native_solver_multi_node(self):
        """Test each model with native solver, multiple uncoupled nodes."""
        n_nodes = 4
        for model_name, model_class, n_states in self.models:
            with self.subTest(model=model_name):
                result = solve(
                    model_class(),
                    Heun(),
                    t0=self.t0,
                    t1=self.t1,
                    dt=self.dt,
                    n_nodes=n_nodes,
                )

                self.assertEqual(result.ys.shape, (self.n_timesteps, n_states, n_nodes))
                self.assertFalse(jnp.isnan(result.ys).any())

    def test_diffrax_solver_single_node(self):
        """Test each model with Diffrax solver, single node."""
        saveat = diffrax.SaveAt(ts=jnp.arange(self.t0, self.t1, self.dt))
        solver = DiffraxSolver(diffrax.Tsit5(), saveat=saveat)

        for model_name, model_class, n_states in self.models:
            with self.subTest(model=model_name):
                result = solve(
                    model_class(), solver, t0=self.t0, t1=self.t1, dt=self.dt
                )

                self.assertEqual(result.ts.shape, (self.n_timesteps,))
                self.assertEqual(result.ys.shape, (self.n_timesteps, n_states, 1))
                self.assertFalse(jnp.isnan(result.ys).any())
                self.assertFalse(jnp.isinf(result.ys).any())

    def test_prepare_and_jit(self):
        """Test that prepare + JIT gives same results as solve."""
        for model_name, model_class, n_states in self.models:
            with self.subTest(model=model_name):
                dynamics = model_class()

                solve_fn, config = prepare(
                    dynamics, Heun(), t0=self.t0, t1=self.t1, dt=self.dt
                )

                result_eager = solve_fn(config)
                result_jit = jax.jit(solve_fn)(config)

                np.testing.assert_allclose(
                    result_eager.ys,
                    result_jit.ys,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"JIT mismatch for {model_name}",
                )

    def test_gradient_through_dynamics_params(self):
        """Test that gradients flow through dynamics parameters."""
        for model_name, model_class, n_states in self.models:
            with self.subTest(model=model_name):
                dynamics = model_class()
                solve_fn, config = prepare(
                    dynamics, Heun(), t0=self.t0, t1=self.t1, dt=self.dt
                )

                def loss(config):
                    result = solve_fn(config)
                    return jnp.mean(result.ys[:, 0, :])

                grad = jax.grad(loss)(config)

                # Check that at least some gradient is non-zero
                grad_leaves = jax.tree.leaves(grad.dynamics)
                has_nonzero = any(jnp.any(g != 0.0) for g in grad_leaves)
                self.assertTrue(
                    has_nonzero,
                    msg=f"All gradients are zero for {model_name}",
                )


class TestBareDynamicsWithNoise(unittest.TestCase):
    """Test bare dynamics with noise."""

    def setUp(self):
        self.t0 = 0.0
        self.t1 = 10.0
        self.dt = 0.01
        self.n_timesteps = int((self.t1 - self.t0) / self.dt)
        self.noise = AdditiveNoise(sigma=0.01)

    def test_native_solver_noise(self):
        """Noise produces different trajectories from deterministic."""
        for model_cls in [JansenRit, ReducedWongWang, Generic2dOscillator]:
            with self.subTest(model=model_cls.__name__):
                det = solve(model_cls(), Heun(), t0=self.t0, t1=self.t1, dt=self.dt)
                stoch = solve(
                    model_cls(),
                    Heun(),
                    t0=self.t0,
                    t1=self.t1,
                    dt=self.dt,
                    noise=self.noise,
                )
                self.assertEqual(det.ys.shape, stoch.ys.shape)
                self.assertFalse(jnp.isnan(stoch.ys).any())
                self.assertGreater(
                    jnp.abs(det.ys - stoch.ys).max(),
                    0.0,
                    msg=f"Noise had no effect on {model_cls.__name__}",
                )

    def test_diffrax_solver_noise(self):
        """Noise works with Diffrax solver."""
        saveat = diffrax.SaveAt(ts=jnp.arange(self.t0, self.t1, self.dt))
        solver = DiffraxSolver(
            diffrax.Euler(),
            saveat=saveat,
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        det = solve(JansenRit(), solver, t0=self.t0, t1=self.t1, dt=self.dt)
        stoch = solve(
            JansenRit(),
            solver,
            t0=self.t0,
            t1=self.t1,
            dt=self.dt,
            noise=self.noise,
        )
        self.assertEqual(det.ys.shape, stoch.ys.shape)
        self.assertFalse(jnp.isnan(stoch.ys).any())
        self.assertGreater(jnp.abs(det.ys - stoch.ys).max(), 0.0)

    def test_noise_multi_node(self):
        """Noise works with multiple uncoupled nodes."""
        result = solve(
            JansenRit(),
            Heun(),
            t0=self.t0,
            t1=self.t1,
            dt=self.dt,
            noise=self.noise,
            n_nodes=3,
        )
        self.assertEqual(result.ys.shape, (self.n_timesteps, 6, 3))
        self.assertFalse(jnp.isnan(result.ys).any())

    def test_noise_apply_to_subset(self):
        """Noise applied to a subset of states only affects those states."""
        noise_subset = AdditiveNoise(sigma=1.0, apply_to=[0])
        det = solve(JansenRit(), Heun(), t0=self.t0, t1=self.t1, dt=self.dt)
        stoch = solve(
            JansenRit(),
            Heun(),
            t0=self.t0,
            t1=self.t1,
            dt=self.dt,
            noise=noise_subset,
        )
        # First state should differ
        self.assertGreater(jnp.abs(det.ys[:, 0, :] - stoch.ys[:, 0, :]).max(), 0.0)


class TestBareDynamicsWithExternals(unittest.TestCase):
    """Test bare dynamics with external inputs."""

    def setUp(self):
        self.t0 = 0.0
        self.t1 = 10.0
        self.dt = 0.01
        self.n_timesteps = int((self.t1 - self.t0) / self.dt)

    def test_native_solver_external(self):
        """External input affects trajectory (Generic2dOscillator has stimulus)."""
        ext = SineInput(amplitude=1.0, frequency=0.1)
        det = solve(
            Generic2dOscillator(),
            Heun(),
            t0=self.t0,
            t1=self.t1,
            dt=self.dt,
        )
        driven = solve(
            Generic2dOscillator(),
            Heun(),
            t0=self.t0,
            t1=self.t1,
            dt=self.dt,
            externals={"stimulus": ext},
        )
        self.assertEqual(det.ys.shape, driven.ys.shape)
        self.assertFalse(jnp.isnan(driven.ys).any())
        self.assertGreater(
            jnp.abs(det.ys - driven.ys).max(),
            0.0,
            msg="External input had no effect",
        )

    def test_diffrax_solver_external(self):
        """External input works with Diffrax solver."""
        saveat = diffrax.SaveAt(ts=jnp.arange(self.t0, self.t1, self.dt))
        solver = DiffraxSolver(diffrax.Tsit5(), saveat=saveat)
        ext = SineInput(amplitude=1.0, frequency=0.1)

        det = solve(Generic2dOscillator(), solver, t0=self.t0, t1=self.t1, dt=self.dt)
        driven = solve(
            Generic2dOscillator(),
            solver,
            t0=self.t0,
            t1=self.t1,
            dt=self.dt,
            externals={"stimulus": ext},
        )
        self.assertFalse(jnp.isnan(driven.ys).any())
        self.assertGreater(jnp.abs(det.ys - driven.ys).max(), 0.0)

    def test_noise_and_external_combined(self):
        """Noise and external inputs work together."""
        noise = AdditiveNoise(sigma=0.01)
        ext = SineInput(amplitude=0.5, frequency=0.1)
        result = solve(
            Generic2dOscillator(),
            Heun(),
            t0=self.t0,
            t1=self.t1,
            dt=self.dt,
            noise=noise,
            externals={"stimulus": ext},
        )
        self.assertEqual(result.ys.shape, (self.n_timesteps, 2, 1))
        self.assertFalse(jnp.isnan(result.ys).any())


if __name__ == "__main__":
    unittest.main()
