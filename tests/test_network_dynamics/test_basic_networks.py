"""Test basic network creation and simulation across model/coupling/noise combinations."""

import unittest
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

# Enable float64 for better numerical precision
jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network, solve
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedLinearCoupling,
    LinearCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    JansenRit,
    ReducedWongWang,
)
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solve import prepare
from tvboptim.experimental.network_dynamics.solvers import Heun


class TestBasicNetworks(unittest.TestCase):
    """Test all combinations of model × coupling × noise."""

    def setUp(self):
        """Set up test parameters."""
        self.n_nodes = 5
        self.t0 = 0.0
        self.t1 = 100.0
        self.dt = 0.1
        self.n_timesteps = int(
            (self.t1 - self.t0) / self.dt
        )  # arange doesn't include endpoint
        self.base_key = jax.random.PRNGKey(42)

        # Use low noise and weak coupling for numerical stability
        self.sigma = 1e-5
        self.coupling_G = 0.1

    def test_network_configurations(self):
        """Test all combinations of model + coupling + noise."""

        # Define model configurations
        models = [
            ("rww", ReducedWongWang, "S", 1),  # (name, class, coupling_var, n_states)
            ("jr", JansenRit, "y1", 6),  # JR has 6 states
        ]

        # Define coupling types
        coupling_types = [
            ("linear", False),  # (name, uses_delay)
            ("delayed", True),
        ]

        # Define noise levels
        noise_configs = [
            ("no_noise", None),
            ("with_noise", self.sigma),
        ]

        # Test all combinations
        key = self.base_key
        for (
            (model_name, model_class, coupling_var, n_states),
            (coupling_name, with_delay),
            (noise_name, noise_sigma),
        ) in product(models, coupling_types, noise_configs):
            with self.subTest(
                model=model_name, coupling=coupling_name, noise=noise_name
            ):
                # Split key for reproducibility
                key, graph_key, delay_key = jax.random.split(key, 3)

                # 1. CREATE NETWORK
                # Create random graph
                graph = DenseGraph.random(n_nodes=self.n_nodes, key=graph_key)

                # Add delays if needed
                if with_delay:
                    delays = (
                        jax.random.uniform(delay_key, (self.n_nodes, self.n_nodes))
                        * 50.0
                    )
                    graph = DenseDelayGraph(weights=graph.weights, delays=delays)
                    coupling = DelayedLinearCoupling(
                        incoming_states=coupling_var, G=self.coupling_G
                    )
                else:
                    coupling = LinearCoupling(
                        incoming_states=coupling_var, G=self.coupling_G
                    )

                # Create dynamics
                dynamics = model_class()

                # Add noise if needed
                noise = (
                    AdditiveNoise(sigma=noise_sigma)
                    if noise_sigma is not None
                    else None
                )

                # Assemble network
                network = Network(
                    dynamics=dynamics,
                    coupling={"instant": coupling},
                    graph=graph,
                    noise=noise,
                )

                # 2. SOLVE WITH HEUN
                result = solve(network, Heun(), t0=self.t0, t1=self.t1, dt=self.dt)

                # 3. PREPARE AND JIT
                solve_fn, state = prepare(
                    network, Heun(), t0=self.t0, t1=self.t1, dt=self.dt
                )
                solve_fn_jit = jax.jit(solve_fn)

                # Test non-jit version
                result_no_jit = solve_fn(state)

                # Test jit version
                result_jit = solve_fn_jit(state)

                # 4. CHECK JIT GIVES SAME OUTPUT (within numerical precision)
                np.testing.assert_allclose(
                    result_no_jit.ys,
                    result_jit.ys,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"JIT and non-JIT outputs differ significantly for {model_name}/{coupling_name}/{noise_name}",
                )

                # 5. CHECK OUTPUT SHAPES
                # result.ys should be [n_timesteps, n_states, n_nodes]
                self.assertEqual(
                    result.ys.shape,
                    (self.n_timesteps, n_states, self.n_nodes),
                    msg=f"Incorrect output shape for {model_name}/{coupling_name}/{noise_name}",
                )

                # result.ts should be [n_timesteps]
                self.assertEqual(
                    result.ts.shape,
                    (self.n_timesteps,),
                    msg=f"Incorrect time shape for {model_name}/{coupling_name}/{noise_name}",
                )

                # 6. CHECK NO NANS/INFS
                self.assertFalse(
                    jnp.isnan(result.ys).any(),
                    msg=f"NaN values found in output for {model_name}/{coupling_name}/{noise_name}",
                )
                self.assertFalse(
                    jnp.isinf(result.ys).any(),
                    msg=f"Inf values found in output for {model_name}/{coupling_name}/{noise_name}",
                )

                # 7. CHECK TIME ARRAY
                # Native solvers emit post-step state, so the save grid is
                # (t0, t1] with ts[0] == t0 + dt and ts[-1] == t1.
                self.assertAlmostEqual(
                    result.ts[0],
                    self.t0 + self.dt,
                    places=10,
                    msg=f"Start time incorrect for {model_name}/{coupling_name}/{noise_name}",
                )
                self.assertAlmostEqual(
                    result.ts[-1],
                    self.t1,
                    places=10,
                    msg=f"End time incorrect for {model_name}/{coupling_name}/{noise_name}",
                )

    def test_native_solver_time_grid(self):
        """Native solvers save on the half-open grid (t0, t1]."""
        # Simple network with no delays / no noise
        weights = jnp.ones((2, 2)) - jnp.eye(2)
        graph = DenseGraph(weights)
        network = Network(
            dynamics=ReducedWongWang(),
            coupling={"instant": LinearCoupling(incoming_states="S", G=0.0)},
            graph=graph,
        )

        t0, t1, dt = 0.0, 12.0, 2.0
        result = solve(network, Heun(), t0=t0, t1=t1, dt=dt)

        expected_n = int(round((t1 - t0) / dt))
        self.assertEqual(result.ts.shape[0], expected_n)
        self.assertEqual(result.ys.shape[0], expected_n)

        # Endpoint included; initial t0 excluded.
        self.assertAlmostEqual(float(result.ts[0]), t0 + dt, places=10)
        self.assertAlmostEqual(float(result.ts[-1]), t1, places=10)

        # Exact dt spacing (no drift from linspace-style endpoint distribution).
        diffs = jnp.diff(result.ts)
        self.assertTrue(jnp.allclose(diffs, dt, atol=1e-10))

        # Nonzero t0 offset also lands on t1 exactly.
        t0b = 5.0
        result_b = solve(network, Heun(), t0=t0b, t1=t0b + 10.0, dt=dt)
        self.assertAlmostEqual(float(result_b.ts[0]), t0b + dt, places=10)
        self.assertAlmostEqual(float(result_b.ts[-1]), t0b + 10.0, places=10)

    def test_noise_key_drives_in_scan_sampling(self):
        """config.noise.key controls in-scan noise generation.

        Same key → identical trajectory; different key → divergent trajectory;
        vmap over keys yields a batched run without re-preparing.
        """
        weights = jnp.ones((3, 3)) - jnp.eye(3)
        graph = DenseGraph(weights)
        network = Network(
            dynamics=ReducedWongWang(),
            coupling={"instant": LinearCoupling(incoming_states="S", G=0.0)},
            graph=graph,
            noise=AdditiveNoise(sigma=1e-3, key=jax.random.key(0)),
        )

        solve_fn, cfg = prepare(network, Heun(), t0=0.0, t1=5.0, dt=0.1)

        # The lazy path is the default; injection slot is None.
        self.assertIsNone(cfg._internal.noise_samples)

        # Same key → deterministic.
        r_a = solve_fn(cfg)
        r_b = solve_fn(cfg)
        self.assertTrue(jnp.allclose(r_a.ys, r_b.ys))

        # Different key → divergent (replace via tree_at rather than mutation).
        import equinox as eqx

        cfg_alt = eqx.tree_at(lambda c: c.noise.key, cfg, jax.random.key(1))
        r_alt = solve_fn(cfg_alt)
        self.assertFalse(jnp.allclose(r_a.ys, r_alt.ys))

        # vmap over keys: no wrapper, no re-prepare.
        def with_seed(seed, base):
            return eqx.tree_at(lambda c: c.noise.key, base, jax.random.key(seed))

        seeds = jnp.arange(4)
        cfgs = jax.vmap(with_seed, in_axes=(0, None))(seeds, cfg)
        ys = jax.vmap(solve_fn)(cfgs).ys
        self.assertEqual(ys.shape[0], 4)
        # At least one pair of seeds must diverge.
        self.assertTrue(jnp.any(ys[0] != ys[1]))

    def test_noise_injection_matches_key_path(self):
        """Injected noise tensor reproduces the key-driven trajectory.

        Guards the NumPyro / replay contract: when the injection slot is
        populated with the same Gaussian tensor that the in-call PRNG
        would have produced, both paths must yield bit-identical outputs.
        """
        import equinox as eqx

        weights = jnp.ones((3, 3)) - jnp.eye(3)
        graph = DenseGraph(weights)
        network = Network(
            dynamics=ReducedWongWang(),
            coupling={"instant": LinearCoupling(incoming_states="S", G=0.0)},
            graph=graph,
            noise=AdditiveNoise(sigma=1e-3, key=jax.random.key(7)),
        )

        solve_fn, cfg = prepare(network, Heun(), t0=0.0, t1=5.0, dt=0.1)

        # Key-driven baseline.
        r_key = solve_fn(cfg)

        # Pre-sample the exact tensor the in-call PRNG would have used,
        # inject it, and run again.
        n_steps = int(round((5.0 - 0.0) / 0.1))
        n_noise_states = len(network.noise._state_indices)
        n_nodes = network.graph.n_nodes
        samples = jax.random.normal(
            cfg.noise.key, (n_steps, n_noise_states, n_nodes)
        )
        cfg_inj = eqx.tree_at(
            lambda c: c._internal.noise_samples,
            cfg,
            samples,
            is_leaf=lambda x: x is None,
        )
        r_inj = solve_fn(cfg_inj)

        self.assertTrue(jnp.array_equal(r_key.ys, r_inj.ys))

        # Sanity: an unrelated injected tensor diverges from the key path.
        other = jax.random.normal(
            jax.random.key(99), (n_steps, n_noise_states, n_nodes)
        )
        cfg_other = eqx.tree_at(
            lambda c: c._internal.noise_samples,
            cfg,
            other,
            is_leaf=lambda x: x is None,
        )
        r_other = solve_fn(cfg_other)
        self.assertFalse(jnp.allclose(r_key.ys, r_other.ys))

    def test_diffrax_noise_key_swap_no_reprepare(self):
        """Diffrax path: config.noise.key controls the VirtualBrownianTree.

        After the refactor the BrownianTree is built per call from
        config.noise.key, so swapping the key in the config (no re-prepare)
        must change the realisation. Same key must replay; vmap over keys
        must produce a batched output.
        """
        import diffrax
        import equinox as eqx

        from tvboptim.experimental.network_dynamics.solvers import DiffraxSolver

        weights = jnp.ones((3, 3)) - jnp.eye(3)
        graph = DenseGraph(weights)
        network = Network(
            dynamics=ReducedWongWang(),
            coupling={"instant": LinearCoupling(incoming_states="S", G=0.0)},
            graph=graph,
            noise=AdditiveNoise(sigma=1e-3, key=jax.random.key(0)),
        )

        t0, t1, dt = 0.0, 5.0, 0.1
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0 + dt, t1, 50))
        solver = DiffraxSolver(solver=diffrax.Heun(), saveat=saveat)

        solve_fn, cfg = prepare(network, solver, t0=t0, t1=t1, dt=dt)

        # No injection slot on the Diffrax dispatch.
        self.assertFalse(hasattr(cfg._internal, "noise_samples"))

        # Same key → identical trajectory.
        r_a = solve_fn(cfg)
        r_b = solve_fn(cfg)
        self.assertTrue(jnp.allclose(r_a.ys, r_b.ys, equal_nan=True))

        # Different key → divergent trajectory, no re-prepare.
        cfg_alt = eqx.tree_at(lambda c: c.noise.key, cfg, jax.random.key(1))
        r_alt = solve_fn(cfg_alt)
        finite = jnp.isfinite(r_a.ys) & jnp.isfinite(r_alt.ys)
        self.assertTrue(jnp.any(r_a.ys[finite] != r_alt.ys[finite]))

        # vmap over keys.
        def with_seed(seed, base):
            return eqx.tree_at(lambda c: c.noise.key, base, jax.random.key(seed))

        seeds = jnp.arange(4)
        cfgs = jax.vmap(with_seed, in_axes=(0, None))(seeds, cfg)
        ys = jax.vmap(solve_fn)(cfgs).ys
        self.assertEqual(ys.shape[0], 4)
        self.assertTrue(jnp.any(ys[0] != ys[1]))

    def test_gradient_computation(self):
        """Test that gradients can be computed through the model."""

        # Test both models
        models = [
            ("rww", ReducedWongWang, "S"),
            ("jr", JansenRit, "y1"),
        ]

        for model_name, model_class, coupling_var in models:
            with self.subTest(model=model_name):
                # Create a simple network (no noise, no delay for gradient test)
                key = jax.random.PRNGKey(123)
                graph = DenseGraph.random(n_nodes=self.n_nodes, key=key)
                coupling = LinearCoupling(
                    incoming_states=coupling_var, G=self.coupling_G
                )
                dynamics = model_class()

                network = Network(
                    dynamics=dynamics,
                    coupling={"instant": coupling},
                    graph=graph,
                    noise=None,  # No noise for gradient test
                )

                # Prepare model
                solve_fn, state = prepare(
                    network, Heun(), t0=self.t0, t1=self.t1, dt=self.dt
                )

                # Create wrapper function for gradient test
                def model_grad(G):
                    """Wrapper that takes coupling strength G and returns mean activity."""
                    # Create new state with updated G (avoid mutation)
                    from tvboptim.experimental.network_dynamics.core.bunch import Bunch

                    updated_coupling_params = Bunch(state.coupling.instant)
                    updated_coupling_params["G"] = G

                    updated_coupling = Bunch(state.coupling)
                    updated_coupling["instant"] = updated_coupling_params

                    updated_state = Bunch(state)
                    updated_state["coupling"] = updated_coupling

                    # Run model
                    result = solve_fn(updated_state)

                    # Return mean activity of first state variable as loss
                    # result.ys shape: [n_timesteps, n_states, n_nodes]
                    # Take first state variable (index 0)
                    first_state = result.ys[:, 0, :]  # [n_timesteps, n_nodes]

                    # Return mean activity
                    return jnp.mean(first_state)

                # Test gradients at G=0.1
                # check_grads checks first and second order derivatives
                # Use looser tolerance for numerical gradients with long simulations
                try:
                    check_grads(
                        model_grad,
                        (self.coupling_G,),
                        order=1,
                        modes=["rev"],
                        atol=1e-2,
                        rtol=1e-2,
                    )
                except Exception as e:
                    self.fail(f"Gradient check failed for {model_name}: {str(e)}")


class TestCheckpointedScan(unittest.TestCase):
    """Verify that the block-checkpointed scan path matches the unchecked
    single-scan path bit-exactly on forward and to numerical precision on
    backward, for both divisor and non-divisor block sizes.
    """

    def _build_dde_network(self):
        key = jax.random.PRNGKey(7)
        weights_key, delay_key = jax.random.split(key)
        n_nodes = 4
        weights = jax.random.uniform(weights_key, (n_nodes, n_nodes)) * 0.5
        delays = jax.random.uniform(delay_key, (n_nodes, n_nodes)) * 5.0
        graph = DenseDelayGraph(weights=weights, delays=delays)
        coupling = DelayedLinearCoupling(incoming_states="S", G=0.1)
        return Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=graph,
            noise=AdditiveNoise(sigma=1e-4, key=jax.random.key(0)),
        )

    def _run(self, checkpoint_every, t1=20.0, dt=0.1):
        network = self._build_dde_network()
        solve_fn, cfg = prepare(
            network,
            Heun(checkpoint_every=checkpoint_every),
            t0=0.0,
            t1=t1,
            dt=dt,
        )
        return solve_fn, cfg

    def test_forward_bitexact_divisor(self):
        """Block size that divides n_steps must reproduce the single-scan
        output exactly — the scan body is identical, only the loop nesting
        changes."""
        solve_none, cfg = self._run(None)
        solve_ckpt, _ = self._run(20)  # 200 steps / 20 = 10 blocks
        r_none = solve_none(cfg)
        r_ckpt = solve_ckpt(cfg)
        self.assertTrue(jnp.array_equal(r_none.ys, r_ckpt.ys))
        self.assertTrue(jnp.array_equal(r_none.ts, r_ckpt.ts))

    def test_forward_bitexact_with_tail(self):
        """Non-divisor block size exercises the main+tail split. Still
        bit-exact: tail is a plain scan over the remainder."""
        solve_none, cfg = self._run(None)
        solve_ckpt, _ = self._run(13)  # 200 % 13 == 5
        r_none = solve_none(cfg)
        r_ckpt = solve_ckpt(cfg)
        self.assertTrue(jnp.array_equal(r_none.ys, r_ckpt.ys))

    def test_gradient_matches_baseline(self):
        """Gradient through the checkpointed scan must match the
        unckecpointed gradient to numerical precision, for both divisor and
        non-divisor block sizes."""

        def make_loss(checkpoint_every):
            solve_fn, cfg = self._run(checkpoint_every)

            def loss(G):
                import equinox as eqx

                cfg2 = eqx.tree_at(lambda c: c.coupling.delayed.G, cfg, G)
                return jnp.mean(solve_fn(cfg2).ys[:, 0, :])

            return jax.value_and_grad(loss)

        G = jnp.asarray(0.1)
        v_none, g_none = make_loss(None)(G)
        v_div, g_div = make_loss(20)(G)
        v_tail, g_tail = make_loss(13)(G)

        # Forward path is bit-exact, so the scalar loss must agree exactly.
        self.assertTrue(jnp.array_equal(v_none, v_div))
        self.assertTrue(jnp.array_equal(v_none, v_tail))

        # Gradient is computed via different traces (rematerialised vs.
        # saved activations) so floating-point rounding can differ very
        # slightly. Tight tolerance covers this.
        self.assertTrue(
            jnp.allclose(g_none, g_div, rtol=1e-10, atol=1e-12),
            f"divisor grad mismatch: none={g_none}, ckpt={g_div}",
        )
        self.assertTrue(
            jnp.allclose(g_none, g_tail, rtol=1e-10, atol=1e-12),
            f"tail grad mismatch: none={g_none}, ckpt={g_tail}",
        )

    def test_default_is_none(self):
        """Sentinel: the default constructor must not enable checkpointing.
        Guards the no-perf-regression contract for existing call sites."""
        self.assertIsNone(Heun().checkpoint_every)

    def test_bare_dynamics_dispatch(self):
        """Bare-dynamics+native path also branches on checkpoint_every."""
        from tvboptim.experimental.network_dynamics.dynamics.tvb import (
            ReducedWongWang,
        )

        dyn = ReducedWongWang()
        solve_none, cfg = prepare(
            dyn, Heun(), t0=0.0, t1=10.0, dt=0.1, n_nodes=3
        )
        solve_ckpt, _ = prepare(
            dyn,
            Heun(checkpoint_every=11),  # non-divisor of 100 steps
            t0=0.0,
            t1=10.0,
            dt=0.1,
            n_nodes=3,
        )
        r_none = solve_none(cfg)
        r_ckpt = solve_ckpt(cfg)
        self.assertTrue(jnp.array_equal(r_none.ys, r_ckpt.ys))


class TestPrepareIsolation(unittest.TestCase):
    """prepare() must return a config whose container is disconnected from the source.

    Regression test: previously, ``config.dynamics`` aliased the Bunch held
    inside ``network.dynamics``, so user assignments like
    ``params.dynamics.w = GridAxis(...)`` leaked into the network and
    contaminated subsequent prepare() calls (causing jit() to choke on the
    leftover non-array value).
    """

    def _build_cases(self):
        n_nodes = 4
        weights = np.random.RandomState(0).rand(n_nodes, n_nodes)
        delays = np.random.RandomState(1).rand(n_nodes, n_nodes)

        return [
            (
                "network_instant",
                Network(
                    dynamics=ReducedWongWang(),
                    coupling={"instant": LinearCoupling(incoming_states=["S"], G=0.3)},
                    graph=DenseGraph(weights),
                ),
            ),
            (
                "network_delayed",
                Network(
                    dynamics=ReducedWongWang(),
                    coupling={
                        "delayed": DelayedLinearCoupling(
                            incoming_states=["S"], G=0.3
                        )
                    },
                    graph=DenseDelayGraph(weights, delays),
                ),
            ),
            (
                "network_noise",
                Network(
                    dynamics=ReducedWongWang(),
                    coupling={"instant": LinearCoupling(incoming_states=["S"], G=0.3)},
                    graph=DenseGraph(weights),
                    noise=AdditiveNoise(sigma=1.0, apply_to="S", key=jax.random.key(0)),
                ),
            ),
        ]

    def test_config_mutation_does_not_leak(self):
        """Mutating cfg.dynamics must not affect the source or later prepare() calls."""
        sentinel = "SENTINEL_NOT_AN_ARRAY"

        for name, network in self._build_cases():
            with self.subTest(case=name):
                _, cfg1 = prepare(network, Heun(), t0=0.0, t1=1.0, dt=0.1)
                original_w = cfg1.dynamics.w

                # Reproduce the failure mode from the minimal example: assign
                # a non-array value onto a param attribute.
                cfg1.dynamics.w = sentinel

                # Source network must be untouched.
                self.assertNotEqual(
                    network.dynamics.params.w,
                    sentinel,
                    f"{name}: mutation leaked into network.dynamics.params",
                )

                # A second prepare() must reflect the original network state.
                _, cfg2 = prepare(network, Heun(), t0=0.0, t1=1.0, dt=0.1)
                self.assertNotEqual(
                    cfg2.dynamics.w,
                    sentinel,
                    f"{name}: re-prepare returned mutated value",
                )
                self.assertTrue(
                    jnp.array_equal(jnp.asarray(cfg2.dynamics.w),
                                    jnp.asarray(original_w)),
                    f"{name}: re-prepare did not restore original w",
                )

                # The fresh config must itself be jit-able — i.e. params are
                # arrays, not leftover sentinels.
                solve_fn, cfg3 = prepare(network, Heun(), t0=0.0, t1=1.0, dt=0.1)
                jax.block_until_ready(jax.jit(solve_fn)(cfg3))


if __name__ == "__main__":
    unittest.main()
