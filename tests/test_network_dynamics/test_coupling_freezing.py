"""Tests for the coupling-freezing default on multi-stage native solvers.

The default (``recompute_coupling_per_stage=False``) holds the coupling input
constant across solver stages. For delayed coupling this is bit-identical to
per-stage evaluation; for instantaneous coupling the two diverge. See
``docs/advanced/coupling_freezing.qmd``.
"""

import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network, solve
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedLinearCoupling,
    LinearCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import Generic2dOscillator
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import (
    BoundedSolver,
    Euler,
    Heun,
    NativeSolver,
    RungeKutta4,
)

# Multi-stage solvers are the only ones the flag affects.
MULTISTAGE = [Heun, RungeKutta4]
N = 20


def _oscillator():
    return Generic2dOscillator(
        a=-1.5, b=-15.0, d=0.015, tau=4.0, INITIAL_STATE=(0.1, 0.1)
    )


def _delay_network(noise=None):
    graph = DenseDelayGraph.random(N, max_delay=40.0, key=jax.random.key(0))
    return Network(
        _oscillator(),
        {"delayed": DelayedLinearCoupling(incoming_states="V", G=0.05)},
        graph,
        noise=noise,
    )


def _instant_network():
    graph = DenseGraph.random(N, key=jax.random.key(0))
    return Network(
        _oscillator(),
        {"instant": LinearCoupling(incoming_states="V", G=0.1)},
        graph,
    )


class TestCouplingFreezing(unittest.TestCase):
    def test_default_is_frozen(self):
        """Coupling freezing is the default; Euler still carries the flag."""
        for solver_cls in MULTISTAGE + [Euler]:
            with self.subTest(solver=solver_cls.__name__):
                self.assertFalse(solver_cls().recompute_coupling_per_stage)

    def test_delayed_coupling_is_bit_identical(self):
        """For delays, frozen and per-stage agree exactly (with and without noise)."""
        cases = [
            ("noise_free", None),
            ("with_noise", AdditiveNoise(sigma=0.02, key=jax.random.key(7))),
        ]
        for solver_cls in MULTISTAGE:
            for label, noise in cases:
                with self.subTest(solver=solver_cls.__name__, case=label):
                    net = _delay_network(noise=noise)
                    yf = solve(
                        net,
                        solver_cls(recompute_coupling_per_stage=False),
                        t1=200.0,
                        dt=0.5,
                    ).ys
                    yp = solve(
                        net,
                        solver_cls(recompute_coupling_per_stage=True),
                        t1=200.0,
                        dt=0.5,
                    ).ys
                    self.assertEqual(float(jnp.max(jnp.abs(yf - yp))), 0.0)

    def test_instantaneous_coupling_diverges(self):
        """For instantaneous coupling, freezing changes the result."""
        for solver_cls in MULTISTAGE:
            with self.subTest(solver=solver_cls.__name__):
                net = _instant_network()
                yf = solve(
                    net,
                    solver_cls(recompute_coupling_per_stage=False),
                    t1=80.0,
                    dt=0.5,
                ).ys
                yp = solve(
                    net,
                    solver_cls(recompute_coupling_per_stage=True),
                    t1=80.0,
                    dt=0.5,
                ).ys
                self.assertGreater(float(jnp.max(jnp.abs(yf - yp))), 1e-6)

    def test_frozen_pins_instantaneous_to_first_order(self):
        """Freezing instantaneous coupling drops Heun from order 2 to order 1."""
        net = _instant_network()

        def final(dt, recompute):
            solver = Heun(recompute_coupling_per_stage=recompute)
            return solve(net, solver, t1=10.0, dt=dt).ys[-1]

        reference = final(0.005, recompute=True)
        dts = [0.5, 0.25, 0.125]
        err_frozen = [
            float(jnp.max(jnp.abs(final(dt, False) - reference))) for dt in dts
        ]
        err_perstage = [
            float(jnp.max(jnp.abs(final(dt, True) - reference))) for dt in dts
        ]
        # Empirical convergence order from the finest pair.
        from math import log2

        order_frozen = log2(err_frozen[-2] / err_frozen[-1])
        order_perstage = log2(err_perstage[-2] / err_perstage[-1])
        self.assertAlmostEqual(order_frozen, 1.0, delta=0.4)
        self.assertGreater(order_perstage, 1.5)

    def test_bounded_solver_delegates_flag(self):
        """BoundedSolver exposes the wrapped solver's flag."""
        inner = Heun(recompute_coupling_per_stage=True)
        self.assertTrue(BoundedSolver(inner).recompute_coupling_per_stage)
        self.assertFalse(BoundedSolver(Heun()).recompute_coupling_per_stage)

    def test_native_solver_isinstance(self):
        """The exported NativeSolver is the concrete base of the step solvers."""
        for solver_cls in [Euler, Heun, RungeKutta4]:
            with self.subTest(solver=solver_cls.__name__):
                self.assertIsInstance(solver_cls(), NativeSolver)


if __name__ == "__main__":
    unittest.main()
