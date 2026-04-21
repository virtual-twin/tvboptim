"""Tests for OptaxOptimizer on a toy quadratic loss.

These tests target the chunked-scan path, the value_fn forwarding, the
callback contract, has_aux handling, and the fwd-mode branch without
spinning up a full Network — the network-based tests already cover
integration.
"""

import unittest

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from tvboptim.optim import OptaxOptimizer
from tvboptim.types import Parameter


def _toy_state(x0: float = 2.0):
    return {"x": Parameter(x0)}


def _quadratic_loss(state):
    return jnp.sum(state["x"].value ** 2)


class TestChunkedEquivalence(unittest.TestCase):
    """chunk_size=None, 1, mid, and max_steps must yield the same trajectory."""

    def test_final_state_matches_across_chunking(self):
        max_steps = 12
        finals = []
        for chunk in (None, 1, 4, max_steps):
            state = _toy_state(x0=2.0)
            opt = OptaxOptimizer(loss=_quadratic_loss, optimizer=optax.sgd(0.1))
            final, _ = opt.run(state, max_steps=max_steps, chunk_size=chunk)
            finals.append(float(final["x"].value))
        for f in finals[1:]:
            self.assertAlmostEqual(f, finals[0], places=10)


class TestValueFnForwardsTrialParams(unittest.TestCase):
    """Regression test: value_fn must evaluate the loss at its argument `p`,
    not at the currently-held diff_state. Line-search optimizers rely on this.
    """

    def test_value_fn_evaluates_at_trial_params(self):
        captured_loss = []

        def probing_optimizer(learning_rate):
            sgd = optax.sgd(learning_rate)

            def init_fn(params):
                return sgd.init(params)

            def update_fn(
                grads, state, params=None, *, value=None, grad=None, value_fn=None
            ):
                # Probe value_fn at params shifted by +3.0
                trial = jax.tree.map(lambda p: p + 3.0, params)
                captured_loss.append(float(value_fn(trial)))
                return sgd.update(grads, state, params)

            return optax.GradientTransformationExtraArgs(init_fn, update_fn)

        state = _toy_state(x0=2.0)
        opt = OptaxOptimizer(loss=_quadratic_loss, optimizer=probing_optimizer(0.1))
        opt.run(state, max_steps=1)

        # loss at x = 2.0 + 3.0 = 5.0  →  25.0
        # If value_fn ignored its argument and used x=2.0, we'd see 4.0.
        self.assertAlmostEqual(captured_loss[0], 25.0, places=6)


class TestCallbackContractChunked(unittest.TestCase):
    def test_callback_fires_once_per_chunk_with_last_step_index(self):
        calls = []

        def cb(step, diff, static, fit, aux, loss, grads):
            calls.append(int(step))
            return False, diff, static

        state = _toy_state()
        opt = OptaxOptimizer(
            loss=_quadratic_loss, optimizer=optax.sgd(0.01), callback=cb
        )
        opt.run(state, max_steps=10, chunk_size=3)

        # ceil(10/3) = 4 chunks; reported step = last step in each chunk.
        self.assertEqual(calls, [2, 5, 8, 9])

    def test_callback_stop_halts_before_next_chunk(self):
        calls = []

        def cb(step, diff, static, fit, aux, loss, grads):
            calls.append(int(step))
            return int(step) >= 2, diff, static

        state = _toy_state()
        opt = OptaxOptimizer(
            loss=_quadratic_loss, optimizer=optax.sgd(0.01), callback=cb
        )
        opt.run(state, max_steps=10, chunk_size=3)

        # Stop requested on first callback → no further chunks run.
        self.assertEqual(calls, [2])


class TestHasAux(unittest.TestCase):
    def _loss_with_aux(self, state):
        x = state["x"].value
        return jnp.sum(x**2), {"abs_x": jnp.abs(x)}

    def test_aux_flows_per_step(self):
        seen = []

        def cb(step, diff, static, fit, aux, loss, grads):
            seen.append(aux)
            return False, diff, static

        state = _toy_state(x0=2.0)
        opt = OptaxOptimizer(
            loss=self._loss_with_aux,
            optimizer=optax.sgd(0.1),
            callback=cb,
            has_aux=True,
        )
        opt.run(state, max_steps=3)

        self.assertEqual(len(seen), 3)
        for a in seen:
            self.assertIn("abs_x", a)

    def test_aux_flows_chunked(self):
        seen = []

        def cb(step, diff, static, fit, aux, loss, grads):
            seen.append(aux)
            return False, diff, static

        state = _toy_state(x0=2.0)
        opt = OptaxOptimizer(
            loss=self._loss_with_aux,
            optimizer=optax.sgd(0.1),
            callback=cb,
            has_aux=True,
        )
        opt.run(state, max_steps=6, chunk_size=2)

        self.assertEqual(len(seen), 3)
        for a in seen:
            self.assertIn("abs_x", a)


class TestFwdMode(unittest.TestCase):
    def test_fwd_mode_smoke(self):
        state = _toy_state(x0=2.0)
        opt = OptaxOptimizer(loss=_quadratic_loss, optimizer=optax.sgd(0.1))
        final, _ = opt.run(state, max_steps=5, mode="fwd")

        x_final = float(final["x"].value)
        self.assertTrue(jnp.isfinite(x_final))
        # Gradient descent on x^2 from x0=2 must move toward 0.
        self.assertLess(abs(x_final), 2.0)


if __name__ == "__main__":
    unittest.main()
