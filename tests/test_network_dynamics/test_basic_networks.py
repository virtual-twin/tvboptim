"""Test basic network creation and simulation across model/coupling/noise combinations."""

import unittest
import warnings
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

# Enable float64 for better numerical precision
jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network, solve
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedKuramotoCoupling,
    DelayedLinearCoupling,
    LinearCoupling,
    SubspaceCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    JansenRit,
    Kuramoto,
    ReducedWongWang,
)
from tvboptim.experimental.network_dynamics.graph import (
    DenseDelayGraph,
    DenseGraph,
    DenseLengthGraph,
    SparseDelayGraph,
    delay_steps_bound,
    effective_max_delay,
)
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solve import prepare
from tvboptim.experimental.network_dynamics.solvers import (
    BoundedSolver,
    Euler,
    Heun,
    RungeKutta4,
)


class _NoShiftHeun(Heun):
    """Heun with the stage-time shift disabled (``stage_time_centroid = 0``).

    Lets a test isolate the interpolating read from the shift that
    DelayedCoupling.precompute() otherwise applies. Also stands in for the
    pre-shift behaviour when checking that the shift is what changed.
    """

    stage_time_centroid = 0.0


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


class TestDelaySweepAccessibility(unittest.TestCase):
    """Delay sweep accessibility: delays must be mutable on an
    already-`prepare()`d config, with no re-`prepare()`, so a `GridAxis` /
    `ParallelExecution` sweep (or a caller-side `delays = x * delays`) can
    vary them against one compiled buffer."""

    BUFFER_STRATEGIES = ("roll", "circular", "preallocated")

    def _build(self, max_delay_bound=None, buffer_strategy="roll"):
        key = jax.random.PRNGKey(3)
        weights_key, delay_key = jax.random.split(key)
        n_nodes = 4
        weights = jax.random.uniform(weights_key, (n_nodes, n_nodes)) * 0.5
        delays = jax.random.uniform(delay_key, (n_nodes, n_nodes)) * 5.0
        graph = DenseDelayGraph(
            weights=weights, delays=delays, max_delay_bound=max_delay_bound
        )
        coupling = DelayedLinearCoupling(
            incoming_states="S", G=0.2, buffer_strategy=buffer_strategy
        )
        network = Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=graph,
            noise=None,
        )
        solve_fn, cfg = prepare(network, Heun(), t0=0.0, t1=10.0, dt=0.1)
        return solve_fn, cfg, delays

    def test_mutating_delay_leaf_changes_output_without_reprepare(self):
        """Replacing config.graph.delays (same compiled solve_fn, no
        re-prepare) must change the trajectory -- this is the core Phase 1
        deliverable: delays read from precompute(), not frozen at prepare().
        Checked for every buffer strategy: all three route delay_steps
        through precompute() now, not just the default ("roll")."""
        import equinox as eqx

        for strategy in self.BUFFER_STRATEGIES:
            with self.subTest(buffer_strategy=strategy):
                solve_fn, cfg, original_delays = self._build(
                    max_delay_bound=10.0, buffer_strategy=strategy
                )
                baseline = solve_fn(cfg).ys

                new_delays = original_delays * 0.1  # much shorter, same bound
                cfg2 = eqx.tree_at(lambda c: c.graph.delays, cfg, new_delays)
                mutated = solve_fn(cfg2).ys

                self.assertFalse(jnp.array_equal(baseline, mutated))

    def test_mutating_delay_leaf_back_to_original_reproduces_baseline(self):
        """Round-tripping the leaf (mutate away, then back) must reproduce the
        original output exactly -- precompute() is a pure function of the
        live graph, not order-dependent on prior calls."""
        import equinox as eqx

        solve_fn, cfg, original_delays = self._build(max_delay_bound=10.0)
        baseline = solve_fn(cfg).ys

        cfg2 = eqx.tree_at(lambda c: c.graph.delays, cfg, original_delays * 0.1)
        solve_fn(cfg2)  # exercise the mutated path first
        cfg3 = eqx.tree_at(lambda c: c.graph.delays, cfg2, original_delays)
        roundtrip = solve_fn(cfg3).ys

        self.assertTrue(jnp.array_equal(baseline, roundtrip))

    def test_grid_axis_sweep_over_delay_scale(self):
        """The documented sweep pattern: a scalar closing over the delay leaf
        (`delays = x * delays`), vmapped via jax.vmap as a stand-in for
        GridAxis/ParallelExecution, produces distinct trajectories per delay
        scale with a single compiled solve_fn."""
        import equinox as eqx

        solve_fn, cfg, original_delays = self._build(max_delay_bound=15.0)

        def run_at_scale(scale):
            scaled_cfg = eqx.tree_at(
                lambda c: c.graph.delays, cfg, scale * original_delays
            )
            return solve_fn(scaled_cfg).ys

        scales = jnp.array([0.5, 1.0, 2.0])
        results = jax.vmap(run_at_scale)(scales)

        self.assertFalse(jnp.array_equal(results[0], results[1]))
        self.assertFalse(jnp.array_equal(results[1], results[2]))

    def test_delay_beyond_bound_clamps_instead_of_erroring(self):
        """A delay pushed past max_delay_bound must clamp into the buffer
        (degraded but defined output) rather than gather out of range, for
        every buffer strategy."""
        import equinox as eqx

        for strategy in self.BUFFER_STRATEGIES:
            with self.subTest(buffer_strategy=strategy):
                solve_fn, cfg, original_delays = self._build(
                    max_delay_bound=6.0, buffer_strategy=strategy
                )
                oversized_delays = original_delays + 20.0  # exceeds the bound
                cfg2 = eqx.tree_at(lambda c: c.graph.delays, cfg, oversized_delays)

                result = solve_fn(cfg2)
                self.assertTrue(bool(jnp.all(jnp.isfinite(result.ys))))

    def test_warn_on_delay_clamp_emits_warning(self):
        """warn_on_delay_clamp=True surfaces the silent-clamp case above as a
        UserWarning instead of a plausible-but-wrong trajectory."""
        import equinox as eqx

        key = jax.random.PRNGKey(3)
        weights_key, delay_key = jax.random.split(key)
        n_nodes = 4
        weights = jax.random.uniform(weights_key, (n_nodes, n_nodes)) * 0.5
        delays = jax.random.uniform(delay_key, (n_nodes, n_nodes)) * 5.0
        graph = DenseDelayGraph(weights=weights, delays=delays, max_delay_bound=6.0)
        coupling = DelayedLinearCoupling(
            incoming_states="S", G=0.2, warn_on_delay_clamp=True
        )
        network = Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=graph,
            noise=None,
        )
        solve_fn, cfg = prepare(network, Heun(), t0=0.0, t1=10.0, dt=0.1)
        cfg2 = eqx.tree_at(lambda c: c.graph.delays, cfg, delays + 20.0)

        with self.assertWarns(UserWarning):
            result = solve_fn(cfg2)
            jax.block_until_ready(result.ys)


class TestDelayInterpolation(unittest.TestCase):
    """Interpolating history read makes d/d(delays) informative instead of
    zero-almost-everywhere, and
    must agree with the nearest-integer read at frac=0 and across
    all three buffer strategies. A wrong idx_hi direction would not crash,
    just quietly bias the blend."""

    BUFFER_STRATEGIES = ("roll", "circular", "preallocated")

    def _build(
        self,
        delays,
        max_delay_bound=8.0,
        buffer_strategy="roll",
        interpolate=True,
        warn_on_delay_clamp=False,
        dt=0.1,
        solver=None,
    ):
        key = jax.random.PRNGKey(7)
        n_nodes = delays.shape[0]
        weights = jax.random.uniform(key, (n_nodes, n_nodes)) * 0.4
        graph = DenseDelayGraph(
            weights=weights, delays=delays, max_delay_bound=max_delay_bound
        )
        coupling = DelayedLinearCoupling(
            incoming_states="S",
            G=0.2,
            buffer_strategy=buffer_strategy,
            history_interpolation="linear" if interpolate else None,
            warn_on_delay_clamp=warn_on_delay_clamp,
        )
        network = Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=graph,
            noise=None,
        )
        solve_fn, cfg = prepare(network, solver or Heun(), t0=0.0, t1=3.0, dt=dt)
        return solve_fn, cfg

    def test_interpolation_matches_nearest_read_at_zero_fraction(self):
        """When delays land exactly on integer step boundaries (frac=0
        everywhere), the interpolated read must exactly reproduce the plain
        nearest-integer read, for every buffer strategy.

        Both sides use a shift-free solver: under Heun's stage-time shift the
        interpolating read deliberately lands half a step off the integer grid
        (frac = 0.5), so the reduction only holds at stage_time_centroid = 0.
        """
        key = jax.random.PRNGKey(11)
        dt = 0.1
        delays = jnp.round(jax.random.uniform(key, (4, 4)) * 30) * dt

        for strategy in self.BUFFER_STRATEGIES:
            with self.subTest(buffer_strategy=strategy):
                solve_fn_i, cfg_i = self._build(
                    delays,
                    buffer_strategy=strategy,
                    interpolate=True,
                    dt=dt,
                    solver=_NoShiftHeun(),
                )
                solve_fn_p, cfg_p = self._build(
                    delays,
                    buffer_strategy=strategy,
                    interpolate=False,
                    dt=dt,
                    solver=_NoShiftHeun(),
                )
                out_i = solve_fn_i(cfg_i).ys
                out_p = solve_fn_p(cfg_p).ys
                np.testing.assert_allclose(out_i, out_p, atol=1e-10)

    def test_strategies_agree_under_interpolation(self):
        """roll, circular, and preallocated must produce the same trajectory
        for the same (non-integer-step) delays: they are different buffer
        implementations of the same read. Disagreement would indicate one of
        them has the idx_hi direction backwards."""
        key = jax.random.PRNGKey(13)
        delays = jax.random.uniform(key, (4, 4)) * 3.0  # non-integer-step

        results = {}
        for strategy in self.BUFFER_STRATEGIES:
            solve_fn, cfg = self._build(delays, buffer_strategy=strategy)
            results[strategy] = solve_fn(cfg).ys

        # Exact equality, not a tolerance: the three strategies gather the same
        # rows in the same order, so they agree bit-for-bit. This once used
        # atol=1e-6, blamed on rounding order -- the residual was really a
        # one-step buffer offset (see TestDelayBufferSizing), and a tolerance
        # that hides that would also hide a backwards idx_hi.
        for strategy in self.BUFFER_STRATEGIES[1:]:
            np.testing.assert_array_equal(results["roll"], results[strategy])

    def test_gradient_matches_finite_differences(self):
        """jax.grad through the interpolated read must be nonzero and match
        central finite differences -- the actual Phase 2 deliverable
        (Phase 1's integer gather has zero gradient almost everywhere)."""
        import equinox as eqx

        key = jax.random.PRNGKey(17)
        delays = jax.random.uniform(key, (3, 3)) * 2.0

        for strategy in self.BUFFER_STRATEGIES:
            with self.subTest(buffer_strategy=strategy):
                solve_fn, cfg = self._build(delays, buffer_strategy=strategy)

                def loss(d, solve_fn=solve_fn, cfg=cfg):
                    cfg2 = eqx.tree_at(lambda c: c.graph.delays, cfg, d)
                    return jnp.sum(solve_fn(cfg2).ys ** 2)

                grad = jax.grad(loss)(delays)
                self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
                self.assertTrue(bool(jnp.any(jnp.abs(grad) > 0)))

                check_grads(
                    loss, (delays,), order=1, modes=["rev"], atol=1e-2, rtol=1e-2
                )

    def test_buffer_grows_for_interpolation_without_out_of_range_read(self):
        """A delay pinned exactly at max_delay_bound (k = max_delay_steps,
        frac = 0) must not read past the buffer prepare() sized -- exactly
        the boundary case the one-slot buffer growth exists for."""
        for strategy in self.BUFFER_STRATEGIES:
            with self.subTest(buffer_strategy=strategy):
                n_nodes = 3
                delays = jnp.full((n_nodes, n_nodes), 6.0)  # == max_delay_bound
                solve_fn, cfg = self._build(
                    delays, max_delay_bound=6.0, buffer_strategy=strategy
                )
                result = solve_fn(cfg)
                self.assertTrue(bool(jnp.all(jnp.isfinite(result.ys))))

    def test_warn_on_delay_clamp_emits_warning_under_interpolation(self):
        """warn_on_delay_clamp must still fire when history_interpolation="linear"
        and a delay walks past max_delay_bound: precompute() reuses the same
        flag for both read modes rather than adding a second one."""
        import equinox as eqx

        key = jax.random.PRNGKey(19)
        delays = jax.random.uniform(key, (3, 3)) * 2.0
        solve_fn, cfg = self._build(
            delays, max_delay_bound=6.0, warn_on_delay_clamp=True
        )
        cfg2 = eqx.tree_at(lambda c: c.graph.delays, cfg, delays + 20.0)

        with self.assertWarns(UserWarning):
            result = solve_fn(cfg2)
            jax.block_until_ready(result.ys)


class TestDenseLengthGraph(unittest.TestCase):
    """DenseLengthGraph owns lengths + speed and derives delays = lengths / speed.

    speed is a differentiable pytree leaf (the delay-domain twin of coupling G),
    so it is directly sweepable via cfg.graph.speed and reachable by jax.grad
    through the computed delays -- with the core still only ever seeing delays.
    """

    WEIGHTS = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    LENGTHS = jnp.array([[0.0, 10.0], [10.0, 0.0]])

    def _network(self, graph, strategy="circular"):
        coupling = DelayedLinearCoupling(
            incoming_states="S",
            G=0.2,
            buffer_strategy=strategy,
            history_interpolation="linear",
        )
        return Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=graph,
            noise=None,
        )

    def test_delays_property_is_lengths_over_speed(self):
        g = DenseLengthGraph(self.WEIGHTS, self.LENGTHS, 5.0, max_delay_bound=4.0)
        np.testing.assert_allclose(np.asarray(g.delays), np.asarray(self.LENGTHS) / 5.0)
        self.assertAlmostEqual(g.max_delay, 2.0)
        # delays is read-only: mutate speed (or lengths) instead
        with self.assertRaises(AttributeError):
            g.delays = self.LENGTHS

    def test_max_delay_bound_required_when_speed_is_a_tracer(self):
        """speed may be a tracer, so delays is a tracer and the static buffer
        length cannot be read off max(delays): the bound must be supplied."""
        with self.assertRaises(ValueError):
            jax.grad(
                lambda s: DenseLengthGraph(self.WEIGHTS, self.LENGTHS, s).delays.sum()
            )(3.0)
        # With the bound, construction under trace succeeds.
        jax.grad(
            lambda s: DenseLengthGraph(
                self.WEIGHTS, self.LENGTHS, s, max_delay_bound=4.0
            ).delays.sum()
        )(3.0)

    def test_undersized_bound_and_bad_inputs_raise(self):
        with self.assertRaises(ValueError):  # bound < max(delays)
            DenseLengthGraph(self.WEIGHTS, self.LENGTHS, 5.0, max_delay_bound=1.0)
        with self.assertRaises(ValueError):  # negative length
            DenseLengthGraph(
                self.WEIGHTS, self.LENGTHS.at[0, 1].set(-1.0), 5.0, max_delay_bound=4.0
            )
        with self.assertRaises(ValueError):  # non-positive speed
            DenseLengthGraph(self.WEIGHTS, self.LENGTHS, 0.0, max_delay_bound=4.0)

    def test_sweep_matches_equivalent_delay_graph(self):
        """A speed sweep must reproduce the trajectories of an explicit delay
        graph at the delays that speed induces, for every buffer strategy."""
        for strategy in ("roll", "circular", "preallocated"):
            for speed in (4.0, 5.0, 8.0):
                with self.subTest(buffer_strategy=strategy, speed=speed):
                    lg = DenseLengthGraph(
                        self.WEIGHTS, self.LENGTHS, speed, max_delay_bound=4.0
                    )
                    dg = DenseDelayGraph(
                        self.WEIGHTS, self.LENGTHS / speed, max_delay_bound=4.0
                    )
                    sf_l, cfg_l = prepare(
                        self._network(lg, strategy), Heun(), t0=0.0, t1=3.0, dt=0.1
                    )
                    sf_d, cfg_d = prepare(
                        self._network(dg, strategy), Heun(), t0=0.0, t1=3.0, dt=0.1
                    )
                    np.testing.assert_array_equal(
                        np.asarray(sf_l(cfg_l).ys), np.asarray(sf_d(cfg_d).ys)
                    )

    def test_speed_setter_is_isolated_by_copy(self):
        """cfg.copy(); cfg2.graph.speed = x must not mutate the original."""
        g = DenseLengthGraph(self.WEIGHTS, self.LENGTHS, 5.0, max_delay_bound=4.0)
        _, cfg = prepare(self._network(g), Heun(), t0=0.0, t1=1.0, dt=0.1)
        cfg2 = cfg.copy()
        cfg2.graph.speed = 2.0
        self.assertEqual(float(cfg.graph.delays[0, 1]), 2.0)  # 10 / 5
        self.assertEqual(float(cfg2.graph.delays[0, 1]), 5.0)  # 10 / 2

    def test_gradient_wrt_speed_is_nonzero_and_finite(self):
        import equinox as eqx

        g = DenseLengthGraph(self.WEIGHTS, self.LENGTHS, 5.0, max_delay_bound=4.0)
        solve_fn, cfg = prepare(self._network(g), Heun(), t0=0.0, t1=3.0, dt=0.1)

        def loss(speed):
            cfg2 = eqx.tree_at(lambda c: c.graph.speed, cfg, speed)
            return jnp.sum(solve_fn(cfg2).ys ** 2)

        grad = jax.grad(loss)(5.0)
        self.assertTrue(bool(jnp.isfinite(grad)))
        self.assertNotEqual(float(grad), 0.0)
        check_grads(loss, (5.0,), order=1, modes=["rev"], atol=1e-3, rtol=1e-3)


class TestDelayBufferSizing(unittest.TestCase):
    """The history buffer must be able to *represent* the delays it is sized for.

    Buffer length and delay read indices are derived independently -- the rows
    come from get_history(), the indices from max_delay_steps -- so a
    disagreement between them does not raise. It silently shifts every delayed
    read in time, which perturbs a smooth trajectory only slightly and slips
    past a loose-tolerance comparison. These tests pin the two together against
    ground truth instead: the delay a read *realizes* must equal the delay that
    was asked for.

    Everything is parametrized over the fractional part of max_delay / dt,
    because that fraction is what decides whether a rounding bug is visible at
    all. The natural values a user picks (a round max_delay_bound, a round dt)
    land on frac = 0, which is exactly where rounding-to-nearest breaks.
    """

    BUFFER_STRATEGIES = ("roll", "circular", "preallocated")
    DT = 0.1
    # max delays chosen so that (max_delay / dt) % 1 sweeps the whole range:
    # 0.0 exactly, below 0.5, exactly 0.5, and above 0.5.
    MAX_DELAYS = (1.0, 1.02, 1.05, 1.06)

    def _build(self, delay, strategy, interpolate, max_delay_bound=None, G=1.0):
        weights = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        delays = jnp.array([[0.0, delay], [delay, 0.0]])
        graph = DenseDelayGraph(
            weights=weights, delays=delays, max_delay_bound=max_delay_bound
        )
        coupling = DelayedLinearCoupling(
            incoming_states="S",
            G=G,
            buffer_strategy=strategy,
            history_interpolation="linear" if interpolate else None,
        )
        network = Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=graph,
            noise=None,
        )
        return network, coupling, graph

    def _realized_delay_steps(self, delay, strategy, interpolate):
        """Delay (in dt steps) that the coupling's history read actually lands on.

        Fills the history buffer so that each row holds its own physical index,
        then reads through the real compute() with G=1 and a one-edge weight
        matrix, so node 0's coupling input *is* the row (or interpolated blend
        of rows) that was read. The distance from the newest row back to that
        value is the realized delay -- an exact identity for the linear blend
        too, since idx_hi = idx_lo - 1 makes the blend (idx_lo - frac).

        Includes JAX's out-of-bounds index clamping, which is what makes an
        over-long read index silently wrong rather than an error.
        """
        network, coupling, graph = self._build(delay, strategy, interpolate)
        coupling_data, coupling_state = network.prepare(self.DT, 0.0, 5.0)
        data, state = coupling_data["delayed"], coupling_state["delayed"]

        n_rows = state.history.shape[0]
        tagged = jnp.broadcast_to(
            jnp.arange(n_rows, dtype=state.history.dtype)[:, None, None],
            state.history.shape,
        )
        state = Bunch({**state, "history": tagged})

        enriched = coupling.precompute(data, coupling.params, graph)
        read = coupling.compute(
            0.0, jnp.zeros((1, 2)), enriched, state, coupling.params, graph
        )
        # node 0 receives only the delayed state of node 1 (weights[0, 1] == 1)
        row_read = float(np.asarray(read).ravel()[0])

        if strategy == "preallocated":
            newest_row = int(state.write_idx) - 1
        else:
            newest_row = n_rows - 1
        return newest_row - row_read

    def test_realized_delay_matches_requested_delay(self):
        """The read must land exactly on the delay it was asked for.

        Under interpolation that is the continuous delay/dt; without it, the
        nearest whole step. A buffer one row short instead realizes delay - 1
        step, which no existing assertion catches.
        """
        for max_delay in self.MAX_DELAYS:
            for strategy in self.BUFFER_STRATEGIES:
                for interpolate in (False, True):
                    with self.subTest(
                        max_delay=max_delay,
                        buffer_strategy=strategy,
                        interpolate=interpolate,
                    ):
                        exact = max_delay / self.DT
                        expected = exact if interpolate else float(jnp.rint(exact))
                        realized = self._realized_delay_steps(
                            max_delay, strategy, interpolate
                        )
                        self.assertAlmostEqual(realized, expected, places=6)

    def test_longest_delay_keeps_its_gradient_without_a_declared_bound(self):
        """Every edge, including the longest, must carry a nonzero gradient.

        The longest delay sets the buffer bound when no max_delay_bound is
        declared. Rounding that bound to nearest leaves the longest delay
        outside the buffer, so precompute() clamps it: frac pins to 0, and its
        gradient is exactly zero while every other edge still looks healthy.
        """
        import equinox as eqx

        # (max delay / dt) % 1 == 0.4: rounding to nearest would round *down*
        delays = jnp.array([[0.31, 1.04], [0.77, 0.52]])

        for strategy in self.BUFFER_STRATEGIES:
            with self.subTest(buffer_strategy=strategy):
                network, _, _ = self._build(1.04, strategy, interpolate=True, G=0.2)
                solve_fn, cfg = prepare(network, Heun(), t0=0.0, t1=3.0, dt=self.DT)

                def loss(d, solve_fn=solve_fn, cfg=cfg):
                    cfg2 = eqx.tree_at(lambda c: c.graph.delays, cfg, d)
                    return jnp.sum(solve_fn(cfg2).ys ** 2)

                grad = jax.grad(loss)(delays)
                argmax = jnp.unravel_index(jnp.argmax(delays), delays.shape)
                self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
                self.assertNotEqual(float(grad[argmax]), 0.0)

    def test_strategies_agree_exactly(self):
        """The three strategies are buffer implementations of one read, so they
        must agree bit-for-bit, not merely closely. A one-step buffer offset
        hides inside a 1e-6 tolerance on a smooth trajectory."""
        for max_delay in self.MAX_DELAYS:
            for interpolate in (False, True):
                with self.subTest(max_delay=max_delay, interpolate=interpolate):
                    results = {}
                    for strategy in self.BUFFER_STRATEGIES:
                        network, _, _ = self._build(
                            max_delay, strategy, interpolate, G=0.2
                        )
                        solve_fn, cfg = prepare(
                            network, Heun(), t0=0.0, t1=3.0, dt=self.DT
                        )
                        results[strategy] = np.asarray(solve_fn(cfg).ys)
                    for strategy in self.BUFFER_STRATEGIES[1:]:
                        np.testing.assert_array_equal(
                            results["roll"], results[strategy]
                        )

    def test_no_clamp_warning_for_delays_within_the_bound(self):
        """warn_on_delay_clamp signals that a delay left the declared buffer. A
        delay that never exceeds the bound must not trip it -- least of all the
        network's own longest delay when no bound was declared at all."""
        for max_delay in self.MAX_DELAYS:
            for interpolate in (False, True):
                with self.subTest(max_delay=max_delay, interpolate=interpolate):
                    weights = jnp.array([[0.0, 1.0], [1.0, 0.0]])
                    delays = jnp.array([[0.0, max_delay], [max_delay, 0.0]])
                    graph = DenseDelayGraph(weights=weights, delays=delays)
                    coupling = DelayedLinearCoupling(
                        incoming_states="S",
                        G=0.2,
                        history_interpolation="linear" if interpolate else None,
                        warn_on_delay_clamp=True,
                    )
                    network = Network(
                        dynamics=ReducedWongWang(),
                        coupling={"delayed": coupling},
                        graph=graph,
                        noise=None,
                    )
                    solve_fn, cfg = prepare(network, Heun(), t0=0.0, t1=1.0, dt=self.DT)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        jax.block_until_ready(solve_fn(cfg).ys)
                    clamp_warnings = [w for w in caught if "clamped" in str(w.message)]
                    self.assertEqual(clamp_warnings, [])

    def test_history_rows_match_the_read_indices(self):
        """prepare() derives read indices from max_delay_steps but takes its rows
        from get_history(). Pin the two together, since a mismatch is the shared
        root cause of every failure above."""
        for max_delay in self.MAX_DELAYS:
            for interpolate in (False, True):
                with self.subTest(max_delay=max_delay, interpolate=interpolate):
                    network, _, _ = self._build(max_delay, "roll", interpolate)
                    coupling_data, coupling_state = network.prepare(self.DT, 0.0, 5.0)
                    data = coupling_data["delayed"]
                    n_rows = coupling_state["delayed"].history.shape[0]
                    expected = data.max_delay_steps + 1 + (1 if interpolate else 0)
                    self.assertEqual(n_rows, expected)

    def test_delay_steps_bound_rounds_up_but_tolerates_float_error(self):
        """The bound must cover the delay (round up), yet not add a spurious step
        for ratios that are integral in exact arithmetic: 1.0 / 0.1 evaluates to
        10.000000000000002 in binary floating point."""
        cases = [
            ((1.0, 0.1), 10),  # exact in decimal, not in binary
            ((1.02, 0.1), 11),  # rounds up rather than to nearest
            ((1.05, 0.1), 11),  # exactly halfway
            ((1.06, 0.1), 11),
            ((0.0, 0.1), 0),  # no delays
            ((3.0, 0.5), 6),
        ]
        for (max_delay, dt), expected in cases:
            with self.subTest(max_delay=max_delay, dt=dt):
                self.assertEqual(delay_steps_bound(max_delay, dt), expected)

    def test_effective_max_delay_is_concrete_float(self):
        """Callers use the result to size static buffers (int()) and to test for
        'no delays' (== 0.0). SparseDelayGraph stores max_delay as a 0-d array,
        so the helper must coerce rather than leak one through."""
        weights = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        delays = jnp.array([[0.0, 1.5], [1.5, 0.0]])
        graphs = [
            DenseGraph(weights),
            DenseDelayGraph(weights, delays),
            DenseDelayGraph(weights, delays, max_delay_bound=3.0),
            SparseDelayGraph(weights, delays),
            SparseDelayGraph(weights, delays, max_delay_bound=3.0),
        ]
        for graph in graphs:
            with self.subTest(graph=type(graph).__name__):
                self.assertIsInstance(effective_max_delay(graph), float)


class TestPrecomputeReachesEveryComputePath(unittest.TestCase):
    """DelayedCoupling resolves its delay read indices in precompute(), not
    prepare(), so compute() must never be handed raw prepare() output. solve.py
    honours that, but it is not the only caller: any path that reaches compute()
    on its own has to run precompute() first."""

    def _delayed_network(self):
        weights = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        delays = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        return Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": DelayedLinearCoupling(incoming_states="S", G=0.5)},
            graph=DenseDelayGraph(weights=weights, delays=delays),
            noise=None,
        )

    def test_network_compute_coupling_inputs(self):
        """Network.compute_coupling_inputs() is public and calls compute() directly."""
        network = self._delayed_network()
        coupling_data, coupling_state = network.prepare(0.1, 0.0, 5.0)
        inputs = network.compute_coupling_inputs(
            0.0, jnp.zeros((1, 2)), coupling_data, coupling_state
        )
        self.assertEqual(inputs["delayed"].shape, (1, 2))
        self.assertTrue(bool(jnp.all(jnp.isfinite(inputs["delayed"]))))

    def test_subspace_coupling_with_delayed_inner_coupling(self):
        """SubspaceCoupling.compute() forwards to inner_coupling.compute() with
        coupling_data.inner_data, which only precompute() can finish building."""
        n_nodes = 4
        node_graph = DenseGraph(jnp.zeros((n_nodes, n_nodes)))
        regional_graph = DenseDelayGraph(
            weights=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
            delays=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        )
        coupling = SubspaceCoupling(
            inner_coupling=DelayedLinearCoupling(incoming_states="S", G=0.5),
            region_mapping=jnp.array([0, 0, 1, 1]),
            regional_graph=regional_graph,
        )
        network = Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=node_graph,
            noise=None,
        )
        solve_fn, cfg = prepare(network, Heun(), t0=0.0, t1=1.0, dt=0.1)
        result = solve_fn(cfg)
        self.assertEqual(result.ys.shape[-1], n_nodes)
        self.assertTrue(bool(jnp.all(jnp.isfinite(result.ys))))


class TestStageTimeShift(unittest.TestCase):
    """The stage-time shift: freezing the coupling across
    solver stages is identical to lengthening every delay by
    ``stage_time_centroid * dt``. precompute() subtracts that back, which
    restores second-order accuracy in the delayed term at no extra gather.

    The bias is deterministic, so unlike a pathwise interpolation error it
    survives averaging over noise realizations -- which is why it is worth
    removing even though the solvers are only strong order 1 under noise.
    """

    BUFFER_STRATEGIES = ("roll", "circular", "preallocated")

    @classmethod
    def setUpClass(cls):
        # Distinguishing global order 1 from 2 needs more headroom than
        # float32 leaves between the discretization error and rounding noise.
        cls._x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", cls._x64)

    def _network(self, delays, dt, interpolate=True, warn=False, G=0.3):
        weights = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        graph = DenseDelayGraph(
            weights=weights, delays=jnp.asarray(delays), max_delay_bound=8.0
        )
        coupling = DelayedLinearCoupling(
            incoming_states="S",
            G=G,
            buffer_strategy="roll",
            history_interpolation="linear" if interpolate else None,
            warn_on_delay_clamp=warn,
        )
        return Network(
            dynamics=ReducedWongWang(),
            coupling={"delayed": coupling},
            graph=graph,
            noise=None,
        )

    def _final_state(self, delays, dt, solver, interpolate=True, t1=20.0):
        network = self._network(delays, dt, interpolate=interpolate)
        solve_fn, cfg = prepare(network, solver, t0=0.0, t1=t1, dt=dt)
        return np.asarray(jax.jit(solve_fn)(cfg).ys)[-1]

    def test_solver_stage_time_centroids(self):
        """sum_i b_i * c_i: 0 for a step-start method, 1/2 for order >= 2.

        BoundedSolver must forward it, or wrapping a solver would silently
        disable the shift.
        """
        cases = [
            (Euler(), 0.0),
            (Heun(), 0.5),
            (RungeKutta4(), 0.5),
            (BoundedSolver(Euler(), low=0.0, high=1.0), 0.0),
            (BoundedSolver(Heun(), low=0.0, high=1.0), 0.5),
            (BoundedSolver(RungeKutta4(), low=0.0, high=1.0), 0.5),
        ]
        for solver, expected in cases:
            with self.subTest(solver=type(solver).__name__, expected=expected):
                self.assertEqual(solver.stage_time_centroid, expected)

    def test_shift_recovers_second_order_in_the_delayed_term(self):
        """Frozen delayed coupling is first order; shifted is second order.

        Reference is the shifted scheme at a much finer dt, so it is the
        *convergence rate* being measured, not agreement with either variant.
        """
        delays = [[0.0, 4.0], [4.0, 0.0]]
        dts = [0.2, 0.1, 0.05]
        ref = self._final_state(delays, 0.05 / 64, Heun())

        errs = {}
        for label, solver in (("frozen", _NoShiftHeun()), ("shifted", Heun())):
            errs[label] = [
                np.max(np.abs(self._final_state(delays, dt, solver) - ref))
                for dt in dts
            ]

        for label, expected in (("frozen", 1.0), ("shifted", 2.0)):
            orders = [
                np.log2(errs[label][i - 1] / errs[label][i]) for i in range(1, len(dts))
            ]
            for dt, order in zip(dts[1:], orders):
                with self.subTest(read=label, dt=dt):
                    self.assertAlmostEqual(order, expected, delta=0.15)

        # The shift is not a wash: at the finest dt it must be far more
        # accurate, not merely differently rounded.
        self.assertLess(errs["shifted"][-1] * 100, errs["frozen"][-1])

    def test_shift_is_a_noop_without_interpolation(self):
        """The nearest-integer read is a piecewise-constant interpolant (q = 1),
        which caps the global order at 1 whatever the solver does, so there is
        no order for the shift to recover. Applying it there would only jitter
        the read between whole steps, so precompute() does not."""
        delays = [[0.0, 4.0], [4.0, 0.0]]
        for dt in (0.2, 0.1):
            with self.subTest(dt=dt):
                shifted = self._final_state(delays, dt, Heun(), interpolate=False)
                frozen = self._final_state(
                    delays, dt, _NoShiftHeun(), interpolate=False
                )
                np.testing.assert_array_equal(shifted, frozen)

    def test_read_lands_on_the_shifted_delay(self):
        """The realized read must be exactly ``delays/dt - centroid`` steps back.

        Tags every buffer row with its own index and reads through the real
        compute(), so this measures the delay a read *realizes* rather than
        comparing two implementations that could be wrong together. Covers all
        three buffer strategies, since each resolves the read index differently.
        """
        dt = 0.1
        delay = 1.03
        for strategy in self.BUFFER_STRATEGIES:
            for centroid in (0.0, 0.5):
                with self.subTest(buffer_strategy=strategy, centroid=centroid):
                    network = self._network([[0.0, delay], [delay, 0.0]], dt)
                    coupling = network.coupling["delayed"]
                    coupling.buffer_strategy = strategy
                    data, state = network.prepare(
                        dt, 0.0, 5.0, stage_time_centroid=centroid
                    )
                    data, state = data["delayed"], state["delayed"]

                    n_rows = state.history.shape[0]
                    tagged = jnp.broadcast_to(
                        jnp.arange(n_rows, dtype=state.history.dtype)[:, None, None],
                        state.history.shape,
                    )
                    state = Bunch({**state, "history": tagged})

                    enriched = coupling.precompute(data, coupling.params, network.graph)
                    read = coupling.compute(
                        0.0,
                        jnp.zeros((1, 2)),
                        enriched,
                        state,
                        coupling.params,
                        network.graph,
                    )
                    row_read = float(np.asarray(read).ravel()[0]) / 0.3  # undo G
                    newest = (
                        int(state.write_idx) - 1
                        if strategy == "preallocated"
                        else n_rows - 1
                    )
                    self.assertAlmostEqual(
                        newest - row_read, delay / dt - centroid, places=4
                    )

    def _kuramoto_final(self, dt, solver_cls, per_stage, tau=2.0, t1=20.0):
        """A delayed coupling that also reads the *local* state."""
        graph = DenseDelayGraph(
            weights=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
            delays=jnp.array([[0.0, tau], [tau, 0.0]]),
            max_delay_bound=4.0,
        )
        coupling = DelayedKuramotoCoupling(
            incoming_states="theta",
            local_states="theta",
            G=0.5,
            buffer_strategy="roll",
            history_interpolation="linear",
        )
        network = Network(
            dynamics=Kuramoto(omega=1.0),
            coupling={"delayed": coupling},
            graph=graph,
            noise=None,
        )
        solver = solver_cls(recompute_coupling_per_stage=per_stage)
        solve_fn, cfg = prepare(network, solver, t0=0.0, t1=t1, dt=dt)
        return np.asarray(jax.jit(solve_fn)(cfg).ys)[-1]

    def test_local_state_coupling_is_not_shifted_while_frozen(self):
        """Freezing a coupling that reads a local state commits two first-order
        errors: the delay bias, and the frozen local state. The shift removes
        only the first. Correcting one while the other stands is measurably
        *worse* (the two partially cancel), so precompute() withholds the shift
        until the solver also evaluates the local state per stage."""
        out = self._kuramoto_final(0.1, Heun, per_stage=False)
        unshifted = self._kuramoto_final(0.1, _NoShiftHeun, per_stage=False)
        np.testing.assert_array_equal(out, unshifted)

    def test_local_state_coupling_reaches_second_order_with_per_stage(self):
        """Frozen (shifted) gather + per-stage local evaluation is order 2;
        per-stage alone, without the shift, is not."""
        dts = [0.2, 0.1, 0.05]
        ref = self._kuramoto_final(0.05 / 64, Heun, per_stage=True)

        for solver_cls, expected in ((_NoShiftHeun, 1.0), (Heun, 2.0)):
            errs = [
                np.max(np.abs(self._kuramoto_final(dt, solver_cls, True) - ref))
                for dt in dts
            ]
            orders = [np.log2(errs[i - 1] / errs[i]) for i in range(1, len(dts))]
            for dt, order in zip(dts[1:], orders):
                with self.subTest(solver=solver_cls.__name__, dt=dt):
                    self.assertAlmostEqual(order, expected, delta=0.15)

    def test_clamp_warning_ignores_zero_weight_edges(self):
        """Every connectome has a zero diagonal in ``delays``. After the shift
        those entries go negative and clamp, but they carry no weight, so they
        are not a mis-simulation and must not trip the warning -- otherwise
        warn_on_delay_clamp fires on every run and stops meaning anything.

        An edge that *does* carry weight and sits below ``centroid * dt`` must
        still warn: it silently keeps the first-order bias the shift removes.
        """
        dt = 0.5  # centroid * dt = 0.25

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            network = self._network([[0.0, 4.0], [4.0, 0.0]], dt, warn=True)
            solve_fn, cfg = prepare(network, Heun(), t0=0.0, t1=2.0, dt=dt)
            jax.block_until_ready(solve_fn(cfg))
        self.assertEqual(
            [w for w in caught if "clamped" in str(w.message)],
            [],
            "zero-weight diagonal must not trip the clamp warning",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            network = self._network([[0.0, 4.0], [0.1, 0.0]], dt, warn=True)
            solve_fn, cfg = prepare(network, Heun(), t0=0.0, t1=2.0, dt=dt)
            jax.block_until_ready(solve_fn(cfg))
        self.assertTrue(
            any("clamped" in str(w.message) for w in caught),
            "a weighted edge shorter than centroid * dt must warn",
        )


if __name__ == "__main__":
    unittest.main()
