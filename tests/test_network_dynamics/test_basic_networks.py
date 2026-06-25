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
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun


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
        samples = jax.random.normal(cfg.noise.key, (n_steps, n_noise_states, n_nodes))
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

    The network is deterministic (no noise) so ``block_size`` is pure
    rematerialization here: it does NOT stream noise, so the blocked result is
    bit-exact to the monolithic one. The streaming-noise behaviour that
    ``block_size`` also triggers for an SDE network (which reseeds, so it is
    deliberately not bit-exact to monolithic) is covered by ``TestStreamingNoise``.
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
            noise=None,
        )

    def _run(self, block_size, t1=20.0, dt=0.1):
        network = self._build_dde_network()
        solve_fn, cfg = prepare(
            network,
            Heun(block_size=block_size),
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

        def make_loss(block_size):
            solve_fn, cfg = self._run(block_size)

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
        self.assertIsNone(Heun().block_size)

    def test_bare_dynamics_dispatch(self):
        """Bare-dynamics+native path also branches on block_size."""
        from tvboptim.experimental.network_dynamics.dynamics.tvb import (
            ReducedWongWang,
        )

        dyn = ReducedWongWang()
        solve_none, cfg = prepare(dyn, Heun(), t0=0.0, t1=10.0, dt=0.1, n_nodes=3)
        solve_ckpt, _ = prepare(
            dyn,
            Heun(block_size=11),  # non-divisor of 100 steps
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
                        "delayed": DelayedLinearCoupling(incoming_states=["S"], G=0.3)
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
                    jnp.array_equal(
                        jnp.asarray(cfg2.dynamics.w), jnp.asarray(original_w)
                    ),
                    f"{name}: re-prepare did not restore original w",
                )

                # The fresh config must itself be jit-able — i.e. params are
                # arrays, not leftover sentinels.
                solve_fn, cfg3 = prepare(network, Heun(), t0=0.0, t1=1.0, dt=0.1)
                jax.block_until_ready(jax.jit(solve_fn)(cfg3))


class TestSolveHelpers(unittest.TestCase):
    """Pin the contract of the Phase 1 module-level helpers extracted from the
    duplicated native dispatch bodies. These are covered transitively by the
    bit-exact integration suite; this class documents the helpers directly so a
    future refactor of one cannot silently drift from the inline logic it
    replaced.
    """

    DYNAMICS = [JansenRit(), ReducedWongWang()]

    def test_split_voi_matches_inline_reference(self):
        from tvboptim.experimental.network_dynamics.solve import _split_voi

        for dyn in self.DYNAMICS:
            with self.subTest(dynamics=type(dyn).__name__):
                # Reference: the exact inline logic the helper replaced.
                voi = dyn.get_variables_of_interest_indices()
                n_states = dyn.N_STATES
                names = dyn.all_variable_names
                ref_state = jnp.array([i for i in voi if i < n_states], dtype=int)
                ref_aux = jnp.array(
                    [i - n_states for i in voi if i >= n_states], dtype=int
                )
                ref_record = len(ref_aux) > 0
                ref_names = tuple(names[i] for i in voi if i < n_states) + tuple(
                    names[i] for i in voi if i >= n_states
                )

                state_idx, aux_idx, record, var_names = _split_voi(dyn)
                self.assertTrue(jnp.array_equal(state_idx, ref_state))
                self.assertTrue(jnp.array_equal(aux_idx, ref_aux))
                self.assertEqual(record, ref_record)
                self.assertEqual(var_names, ref_names)
                # Labels match the number of recorded rows.
                self.assertEqual(len(var_names), len(ref_state) + len(ref_aux))

    def test_materialize_noise_draw_and_injection(self):
        from tvboptim.experimental.network_dynamics.solve import _materialize_noise

        shape = (5, 2, 3)
        key = jax.random.key(0)
        # Default provider: single fused draw, reproducible from the key.
        drawn = _materialize_noise(key, None, shape)
        self.assertEqual(drawn.shape, shape)
        self.assertTrue(jnp.array_equal(drawn, jax.random.normal(key, shape)))
        # Injection: passed through verbatim, ignoring the key.
        injected = jnp.ones(shape)
        out = _materialize_noise(key, injected, shape)
        self.assertIs(out, injected)

    def test_assemble_output_layout(self):
        from tvboptim.experimental.network_dynamics.solve import _assemble_output

        n_nodes = 3
        next_state = jnp.arange(4 * n_nodes, dtype=float).reshape(4, n_nodes)
        aux = jnp.arange(2 * n_nodes, dtype=float).reshape(2, n_nodes) + 100.0

        # States only.
        out = _assemble_output(
            next_state, aux, jnp.array([0, 2]), jnp.array([], dtype=int), False
        )
        self.assertTrue(jnp.array_equal(out, next_state[jnp.array([0, 2])]))

        # States followed by selected auxiliaries.
        out = _assemble_output(next_state, aux, jnp.array([1]), jnp.array([0]), True)
        expected = jnp.concatenate(
            [next_state[jnp.array([1])], aux[jnp.array([0])]], axis=0
        )
        self.assertTrue(jnp.array_equal(out, expected))


class TestTruncatedScan(unittest.TestCase):
    """Verify the truncated-BPTT windowed scan (`_truncated_scan`).

    Forward must stay bit-exact to a plain single scan (``stop_gradient`` is the
    identity on the forward); the backward must (a) match an independent
    hand-rolled windowed reference, (b) reduce to the full exact gradient in the
    degenerate single-window case, and (c) be invariant to the nested memory
    knob ``block_size``. A network-level case checks the wiring through
    ``prepare`` / ``run_scan`` and the ``BoundedSolver`` delegation.

    The combinator is tested with a toy linear recurrence so the reference is
    independent and fully controlled, mirroring ``TestCheckpointedScan`` for the
    network-level checks.
    """

    N = 20  # toy rollout length
    C0 = jnp.asarray(0.3)

    @staticmethod
    def _make_op(a):
        # Toy step: carry' = a * carry + x, output = carry. ``a`` is the
        # closed-over parameter we differentiate (stands in for theta).
        def op(c, x):
            nc = a * c + x
            return nc, nc

        return op

    def _xs(self):
        return jnp.arange(self.N, dtype=float) + 1.0

    def _full(self, a):
        _, ys = jax.lax.scan(self._make_op(a), self.C0, self._xs())
        return jnp.sum(ys)

    def _trunc(self, a, window, block_size):
        from tvboptim.experimental.network_dynamics.solve import _truncated_scan

        _, ys = _truncated_scan(
            self._make_op(a), self.C0, self._xs(), self.N, window, block_size
        )
        return jnp.sum(ys)

    def _ref_trunc(self, a, window):
        # Independent reference: an unrolled Python loop over windows that
        # severs the carry gradient at each window entry, exactly the truncated
        # estimator. No reshape/tail machinery, so it cannot share a bug with
        # `_truncated_scan`.
        op = self._make_op(a)
        xs = self._xs()
        c = self.C0
        outs = []
        i = 0
        while i < self.N:
            j = min(i + window, self.N)
            c = jax.lax.stop_gradient(c)
            c, block = jax.lax.scan(op, c, xs[i:j])
            outs.append(block)
            i = j
        return jnp.sum(jnp.concatenate(outs, axis=0))

    def test_forward_bitexact(self):
        """Truncation does not touch the forward value, for divisor,
        non-divisor and degenerate window sizes."""
        for window in (5, 7, self.N + 3):
            with self.subTest(window=window):
                self.assertTrue(
                    jnp.array_equal(self._full(0.5), self._trunc(0.5, window, None))
                )

    def test_gradient_matches_reference(self):
        """Truncated gradient equals the unrolled windowed reference, and is a
        genuine truncation (differs from the full gradient)."""
        g_full = jax.grad(self._full)(0.5)
        for window in (5, 7):
            with self.subTest(window=window):
                g_trunc = jax.grad(lambda a: self._trunc(a, window, None))(0.5)
                g_ref = jax.grad(lambda a: self._ref_trunc(a, window))(0.5)
                self.assertTrue(
                    jnp.allclose(g_trunc, g_ref, rtol=1e-10, atol=1e-12),
                    f"window={window}: {g_trunc} vs ref {g_ref}",
                )
                # The truncation is real: short windows drop cross-window credit.
                self.assertFalse(jnp.allclose(g_trunc, g_full, rtol=1e-6))

    def test_degenerate_window_equals_full(self):
        """A single window (window >= n_steps) recovers the full exact gradient:
        severing the leaf initial carry does not affect the parameter gradient."""
        g_full = jax.grad(self._full)(0.5)
        for window in (self.N, self.N + 5):
            with self.subTest(window=window):
                g = jax.grad(lambda a: self._trunc(a, window, None))(0.5)
                self.assertTrue(jnp.allclose(g_full, g, rtol=1e-10, atol=1e-12))

    def test_gradient_invariant_to_block_size(self):
        """Within a fixed window, subdividing into checkpoint sub-blocks
        rematerializes activations but must not change the gradient, including
        non-divisor sub-blocks and a sub-block larger than the window."""
        window = 10
        g_none = jax.grad(lambda a: self._trunc(a, window, None))(0.5)
        for ce in (5, 3, 25):  # divisor, non-divisor, larger-than-window
            with self.subTest(block_size=ce):
                g = jax.grad(lambda a: self._trunc(a, window, ce))(0.5)
                self.assertTrue(
                    jnp.allclose(g_none, g, rtol=1e-10, atol=1e-12),
                    f"block_size={ce}: {g} vs {g_none}",
                )

    def _build_sde_network(self):
        key = jax.random.PRNGKey(11)
        wkey, dkey = jax.random.split(key)
        n_nodes = 4
        weights = jax.random.uniform(wkey, (n_nodes, n_nodes)) * 0.5
        delays = jax.random.uniform(dkey, (n_nodes, n_nodes)) * 5.0
        graph = DenseDelayGraph(weights=weights, delays=delays)
        coupling = DelayedLinearCoupling(incoming_states="S", G=0.1)
        return Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=graph,
            noise=AdditiveNoise(sigma=1e-4, key=jax.random.key(0)),
        )

    def test_network_forward_bitexact(self):
        """Through prepare/run_scan, the truncated path's forward trajectory is
        bit-exact to the non-truncated path (truncation changes only gradients),
        for a divisor and a non-divisor window of the 200-step rollout."""
        network = self._build_sde_network()
        solve_full, cfg = prepare(network, Heun(), t0=0.0, t1=20.0, dt=0.1)
        r_full = solve_full(cfg)
        for window in (20, 13):  # 200 % 20 == 0, 200 % 13 == 5
            with self.subTest(window=window):
                solve_trunc, _ = prepare(
                    network, Heun(grad_horizon=window), t0=0.0, t1=20.0, dt=0.1
                )
                self.assertTrue(jnp.array_equal(r_full.ys, solve_trunc(cfg).ys))

    def test_default_and_bounded_delegation(self):
        """Default constructor leaves truncation off; BoundedSolver forwards the
        knob from its base solver."""
        self.assertIsNone(Heun().grad_horizon)
        self.assertEqual(Heun(grad_horizon=50).grad_horizon, 50)
        self.assertEqual(
            BoundedSolver(Heun(grad_horizon=50), low=0.0, high=1.0).grad_horizon,
            50,
        )


class TestReduce(unittest.TestCase):
    """Verify the block-level reduce (fold) path.

    The fold folds each block's stacked outputs into an accumulator carried in
    the scan instead of stacking the whole trajectory. Checked at two levels: a
    toy running-sum reducer through ``run_scan`` (the independent reference is
    the sum over the plainly-stacked trajectory), and a network-level online
    ``welford_cov`` FC pinned against the post-hoc ``compute_fc``. The fold must
    match for divisor / non-divisor / degenerate blocks, compose with
    ``grad_horizon``, and be invariant (value and gradient) to ``block_size``.
    """

    N = 20  # toy rollout length
    C0 = jnp.asarray(0.3)

    @staticmethod
    def _make_op(a):
        # Toy step: carry' = a * carry + x, output = carry (matches
        # TestTruncatedScan so the reference is independent and controlled).
        def op(c, x):
            nc = a * c + x
            return nc, nc

        return op

    def _xs(self):
        return jnp.arange(self.N, dtype=float) + 1.0

    # Toy reducer update: acc is the scalar running sum of all step outputs.
    _UPDATE = staticmethod(lambda acc, block: acc + jnp.sum(block))

    def _stacked_sum(self, a):
        _, ys = jax.lax.scan(self._make_op(a), self.C0, self._xs())
        return jnp.sum(ys)

    def _fold_sum(self, a, block_size, window=None):
        from tvboptim.experimental.network_dynamics.solve import run_scan

        solver = Heun(block_size=block_size, grad_horizon=window)
        carry, _ = run_scan(
            self._make_op(a),
            self.C0,
            self._xs(),
            self.N,
            solver,
            fold=(jnp.asarray(0.0), self._UPDATE),
        )
        return carry[1]  # the accumulator threaded in the (state, acc) carry

    def test_fold_equals_stacked_sum(self):
        """Folded accumulator equals the sum over the plainly-stacked
        trajectory, for divisor / non-divisor / degenerate blocks, with and
        without a truncation window (the forward value is identical)."""
        ref = self._stacked_sum(0.5)
        for block_size in (5, 7, self.N, 4):
            # Windows are None or multiples of block_size (so no snapping fires;
            # snapping is covered in TestStreamingNoise).
            for window in (None, 2 * block_size, 3 * block_size):
                with self.subTest(block_size=block_size, window=window):
                    got = self._fold_sum(0.5, block_size, window)
                    self.assertTrue(
                        jnp.allclose(got, ref, rtol=1e-12, atol=1e-12),
                        f"bs={block_size}, w={window}: {got} vs {ref}",
                    )

    def test_fold_value_and_grad_invariant_to_block_size(self):
        """Without truncation the fold is the exact gradient (checkpoint
        rematerialization), so both value and gradient are invariant to
        ``block_size``, including non-divisor and larger-than-rollout blocks."""
        v_ref = self._stacked_sum(0.5)
        g_ref = jax.grad(self._stacked_sum)(0.5)
        for block_size in (5, 7, 4, self.N):
            with self.subTest(block_size=block_size):
                v = self._fold_sum(0.5, block_size, None)
                g = jax.grad(lambda a: self._fold_sum(a, block_size, None))(0.5)
                self.assertTrue(jnp.allclose(v, v_ref, rtol=1e-12, atol=1e-12))
                self.assertTrue(
                    jnp.allclose(g, g_ref, rtol=1e-10, atol=1e-12),
                    f"bs={block_size}: grad {g} vs {g_ref}",
                )

    def _build_sde_network(self):
        key = jax.random.PRNGKey(11)
        wkey, dkey = jax.random.split(key)
        n_nodes = 5
        weights = jax.random.uniform(wkey, (n_nodes, n_nodes)) * 0.5
        delays = jax.random.uniform(dkey, (n_nodes, n_nodes)) * 5.0
        graph = DenseDelayGraph(weights=weights, delays=delays)
        coupling = DelayedLinearCoupling(incoming_states="S", G=0.1)
        return Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=graph,
            noise=AdditiveNoise(sigma=1e-3, key=jax.random.key(0)),
        )

    def test_welford_matches_compute_fc(self):
        """Online ``welford_cov`` over a blocked (streaming-noise) run equals the
        post-hoc ``compute_fc`` on the matching streamed trajectory, for a
        divisor and a non-divisor block size. The reference uses the SAME
        ``block_size`` (hence the same per-block seeding), not the monolithic
        global draw: blocked mode streams noise and reseeds, so the only valid
        comparison is online-fold vs stack-then-post-hoc at matched seeding."""
        from tvboptim.observations import compute_fc, welford_cov

        net = self._build_sde_network()
        for block_size in (50, 37):  # 300 steps: divisor / non-divisor
            with self.subTest(block_size=block_size):
                # Stacked trajectory with this block_size (streamed noise) -> FC.
                fc_ref = compute_fc(
                    solve(net, Heun(block_size=block_size), t0=0.0, t1=30.0, dt=0.1),
                    s_var=0,
                )
                # Online welford over the same streamed run.
                fc = solve(
                    net,
                    Heun(block_size=block_size),
                    t0=0.0,
                    t1=30.0,
                    dt=0.1,
                    reduce=welford_cov(s_var=0),
                )
                self.assertEqual(fc.shape, fc_ref.shape)
                self.assertTrue(
                    jnp.allclose(fc, fc_ref, atol=1e-4),
                    f"bs={block_size}: max diff {jnp.max(jnp.abs(fc - fc_ref))}",
                )

    def test_welford_monolithic_equals_compute_fc(self):
        """``block_size=None`` with a reducer folds the whole stacked trajectory
        once (the degenerate single-block / post-hoc case) and equals
        ``compute_fc``."""
        from tvboptim.observations import compute_fc, welford_cov

        net = self._build_sde_network()
        solve_full, cfg = prepare(net, Heun(), t0=0.0, t1=30.0, dt=0.1)
        fc_ref = compute_fc(solve_full(cfg), s_var=0)
        solve_fc, _ = prepare(
            net, Heun(), t0=0.0, t1=30.0, dt=0.1, reduce=welford_cov(s_var=0)
        )
        self.assertTrue(jnp.allclose(solve_fc(cfg), fc_ref, atol=1e-4))

    def test_welford_differentiable_and_tbptt_invariant(self):
        """The online FC is differentiable wrt a coupling gain, and (for a fixed
        ``block_size``, hence fixed noise) its forward value is invariant to
        ``grad_horizon`` snapped to a multiple of ``block_size`` -- truncation
        changes only the gradient, not the streamed realisation."""
        import equinox as eqx

        from tvboptim.observations import welford_cov

        net = self._build_sde_network()

        # Differentiable wrt G on a blocked streaming run.
        solve_fc, cfg = prepare(
            net,
            Heun(block_size=50),
            t0=0.0,
            t1=30.0,
            dt=0.1,
            reduce=welford_cov(s_var=0),
        )

        def loss(G):
            c = eqx.tree_at(lambda c: c.coupling.delayed.G, cfg, G)
            return jnp.sum(solve_fc(c) ** 2)

        v, g = jax.value_and_grad(loss)(jnp.asarray(0.1))
        self.assertTrue(jnp.isfinite(g))

        # Forward FC invariant to a truncation window (multiple of block_size).
        fc_base = solve(
            net,
            Heun(block_size=50),
            t0=0.0,
            t1=30.0,
            dt=0.1,
            reduce=welford_cov(s_var=0),
        )
        for window in (100, 150, 300):  # multiples of block_size=50
            with self.subTest(window=window):
                fc_w = solve(
                    net,
                    Heun(block_size=50, grad_horizon=window),
                    t0=0.0,
                    t1=30.0,
                    dt=0.1,
                    reduce=welford_cov(s_var=0),
                )
                self.assertTrue(jnp.array_equal(fc_base, fc_w))

    def test_diffrax_rejects_reduce(self):
        """``reduce`` is native-only; the Diffrax dispatch raises a clear error
        rather than a bare TypeError on an unexpected keyword."""
        import diffrax

        from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
        from tvboptim.experimental.network_dynamics.solvers.diffrax import (
            DiffraxSolver,
        )
        from tvboptim.observations import welford_cov

        n_nodes = 3
        net = Network(
            dynamics=ReducedWongWang(),
            coupling={"instant": LinearCoupling(incoming_states="S", G=0.1)},
            graph=DenseGraph(
                jax.random.uniform(jax.random.PRNGKey(1), (n_nodes, n_nodes))
            ),
        )
        with self.assertRaises(ValueError):
            prepare(
                net,
                DiffraxSolver(solver=diffrax.Heun()),
                t0=0.0,
                t1=5.0,
                dt=0.1,
                reduce=welford_cov(),
            )


class TestStreamingNoise(unittest.TestCase):
    """Per-block streaming noise (``fold_in``) under ``block_size``.

    Streaming activates for an SDE network with ``block_size`` set and no
    injected tensor. The realisation is a pure function of
    ``(key, absolute_block_idx)`` and the block grain, so it is deterministic,
    invariant to the truncation window, and matches an independent reference
    that folds the noise in the same way. It deliberately reseeds relative to
    the monolithic global draw (documented).
    """

    T1 = 30.0
    DT = 0.1  # 300 steps

    def _build(self):
        key = jax.random.PRNGKey(11)
        wkey, dkey = jax.random.split(key)
        n_nodes = 5
        weights = jax.random.uniform(wkey, (n_nodes, n_nodes)) * 0.5
        delays = jax.random.uniform(dkey, (n_nodes, n_nodes)) * 5.0
        graph = DenseDelayGraph(weights=weights, delays=delays)
        coupling = DelayedLinearCoupling(incoming_states="S", G=0.1)
        return Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=graph,
            noise=AdditiveNoise(sigma=1e-3, key=jax.random.key(0)),
        )

    def test_reseed_and_determinism(self):
        """Blocked streaming reseeds vs monolithic; is deterministic for a fixed
        block_size; and a different block grain gives a different realisation."""
        net = self._build()
        kw = dict(t0=0.0, t1=self.T1, dt=self.DT)
        mono = solve(net, Heun(), **kw).ys
        a = solve(net, Heun(block_size=50), **kw).ys
        b = solve(net, Heun(block_size=50), **kw).ys
        c = solve(net, Heun(block_size=37), **kw).ys
        self.assertFalse(jnp.allclose(mono, a))  # reseed vs global draw
        self.assertTrue(jnp.array_equal(a, b))  # deterministic
        self.assertFalse(jnp.allclose(a, c))  # block grain changes realisation

    def test_matches_matched_seeding_reference(self):
        """Streaming forward equals an independent reference that builds the full
        noise tensor by the same per-block ``fold_in`` and injects it into a
        monolithic run, for a divisor and a non-divisor block size (exercising
        the tail block)."""
        net = self._build()
        n_steps = len(jnp.arange(0.0, self.T1, self.DT))
        n_noise = len(net.noise._state_indices)
        n_nodes = net.graph.n_nodes
        key = net.noise.key
        for block_size in (50, 37):
            with self.subTest(block_size=block_size):
                strm = solve(
                    net, Heun(block_size=block_size), t0=0.0, t1=self.T1, dt=self.DT
                ).ys
                # Independent reference: per-block fold_in chunks concatenated.
                n_blocks = n_steps // block_size
                rem = n_steps - n_blocks * block_size
                chunks = [
                    jax.random.normal(
                        jax.random.fold_in(key, i), (block_size, n_noise, n_nodes)
                    )
                    for i in range(n_blocks)
                ]
                if rem:
                    chunks.append(
                        jax.random.normal(
                            jax.random.fold_in(key, n_blocks), (rem, n_noise, n_nodes)
                        )
                    )
                full_noise = jnp.concatenate(chunks, axis=0)
                solve_fn, cfg = prepare(net, Heun(), t0=0.0, t1=self.T1, dt=self.DT)
                cfg._internal.noise_samples = full_noise
                ref = solve_fn(cfg).ys
                self.assertTrue(
                    jnp.allclose(strm, ref, atol=1e-5),
                    f"bs={block_size}: max diff {jnp.max(jnp.abs(strm - ref))}",
                )

    def test_forward_invariant_to_grad_horizon(self):
        """For a fixed block_size the streamed forward trajectory is bit-exact
        across truncation windows (multiples of block_size) -- the absolute
        block grid does not depend on how windows tile it."""
        net = self._build()
        kw = dict(t0=0.0, t1=self.T1, dt=self.DT)
        base = solve(net, Heun(block_size=50), **kw).ys
        for window in (100, 150, 300):
            with self.subTest(window=window):
                ys = solve(net, Heun(block_size=50, grad_horizon=window), **kw).ys
                self.assertTrue(jnp.array_equal(base, ys))

    def test_non_multiple_window_snaps_with_warning(self):
        """A grad_horizon that is not a multiple of block_size is snapped to the
        nearest multiple with a warning, rather than silently accepted."""
        net = self._build()
        with self.assertWarns(UserWarning):
            solve(
                net,
                Heun(block_size=50, grad_horizon=120),
                t0=0.0,
                t1=self.T1,
                dt=self.DT,
            )

    def test_statistical_sanity(self):
        """The streamed increments are standard normal: mean ~ 0, variance ~ 1
        across blocks."""
        from tvboptim.experimental.network_dynamics.solve import _streaming_noise_gen

        gen = _streaming_noise_gen(jax.random.key(0), (2, 8))
        sample = jnp.concatenate([gen(i, 64) for i in range(60)], axis=0)
        self.assertLess(abs(float(jnp.mean(sample))), 0.02)
        self.assertLess(abs(float(jnp.var(sample)) - 1.0), 0.05)


if __name__ == "__main__":
    unittest.main()
