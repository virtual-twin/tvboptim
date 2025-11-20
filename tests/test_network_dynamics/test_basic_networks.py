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
from tvboptim.experimental.network_dynamics.solve import prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang, JansenRit
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling, DelayedLinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseGraph, DenseDelayGraph
from tvboptim.experimental.network_dynamics.solvers import Heun
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise


class TestBasicNetworks(unittest.TestCase):
    """Test all combinations of model × coupling × noise."""

    def setUp(self):
        """Set up test parameters."""
        self.n_nodes = 5
        self.t0 = 0.0
        self.t1 = 100.0
        self.dt = 0.1
        self.n_timesteps = int((self.t1 - self.t0) / self.dt)  # arange doesn't include endpoint
        self.base_key = jax.random.PRNGKey(42)

        # Use low noise and weak coupling for numerical stability
        self.sigma = 1e-5
        self.coupling_G = 0.1

    def test_network_configurations(self):
        """Test all combinations of model + coupling + noise."""

        # Define model configurations
        models = [
            ('rww', ReducedWongWang, 'S', 1),   # (name, class, coupling_var, n_states)
            ('jr', JansenRit, 'y1', 6),          # JR has 6 states
        ]

        # Define coupling types
        coupling_types = [
            ('linear', False),       # (name, uses_delay)
            ('delayed', True),
        ]

        # Define noise levels
        noise_configs = [
            ('no_noise', None),
            ('with_noise', self.sigma),
        ]

        # Test all combinations
        key = self.base_key
        for (model_name, model_class, coupling_var, n_states), \
            (coupling_name, with_delay), \
            (noise_name, noise_sigma) in product(models, coupling_types, noise_configs):

            with self.subTest(model=model_name, coupling=coupling_name, noise=noise_name):
                # Split key for reproducibility
                key, graph_key, delay_key = jax.random.split(key, 3)

                # 1. CREATE NETWORK
                # Create random graph
                graph = DenseGraph.random(n_nodes=self.n_nodes, key=graph_key)

                # Add delays if needed
                if with_delay:
                    delays = jax.random.uniform(delay_key, (self.n_nodes, self.n_nodes)) * 50.0
                    graph = DenseDelayGraph(weights=graph.weights, delays=delays)
                    coupling = DelayedLinearCoupling(incoming_states=coupling_var, G=self.coupling_G)
                else:
                    coupling = LinearCoupling(incoming_states=coupling_var, G=self.coupling_G)

                # Create dynamics
                dynamics = model_class()

                # Add noise if needed
                noise = AdditiveNoise(sigma=noise_sigma) if noise_sigma is not None else None

                # Assemble network
                network = Network(
                    dynamics=dynamics,
                    coupling={'instant': coupling},
                    graph=graph,
                    noise=noise
                )

                # 2. SOLVE WITH HEUN
                result = solve(network, Heun(), t0=self.t0, t1=self.t1, dt=self.dt)

                # 3. PREPARE AND JIT
                solve_fn, state = prepare(network, Heun(), t0=self.t0, t1=self.t1, dt=self.dt)
                solve_fn_jit = jax.jit(solve_fn)

                # Test non-jit version
                result_no_jit = solve_fn(state)

                # Test jit version
                result_jit = solve_fn_jit(state)

                # 4. CHECK JIT GIVES SAME OUTPUT (within numerical precision)
                np.testing.assert_allclose(
                    result_no_jit.ys, result_jit.ys,
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"JIT and non-JIT outputs differ significantly for {model_name}/{coupling_name}/{noise_name}"
                )

                # 5. CHECK OUTPUT SHAPES
                # result.ys should be [n_timesteps, n_states, n_nodes]
                self.assertEqual(
                    result.ys.shape,
                    (self.n_timesteps, n_states, self.n_nodes),
                    msg=f"Incorrect output shape for {model_name}/{coupling_name}/{noise_name}"
                )

                # result.ts should be [n_timesteps]
                self.assertEqual(
                    result.ts.shape,
                    (self.n_timesteps,),
                    msg=f"Incorrect time shape for {model_name}/{coupling_name}/{noise_name}"
                )

                # 6. CHECK NO NANS/INFS
                self.assertFalse(
                    jnp.isnan(result.ys).any(),
                    msg=f"NaN values found in output for {model_name}/{coupling_name}/{noise_name}"
                )
                self.assertFalse(
                    jnp.isinf(result.ys).any(),
                    msg=f"Inf values found in output for {model_name}/{coupling_name}/{noise_name}"
                )

                # 7. CHECK TIME ARRAY - just basic sanity checks
                self.assertAlmostEqual(
                    result.ts[0], self.t0, places=10,
                    msg=f"Start time incorrect for {model_name}/{coupling_name}/{noise_name}"
                )
                self.assertGreaterEqual(
                    result.ts[-1], self.t1 - self.dt,
                    msg=f"End time too early for {model_name}/{coupling_name}/{noise_name}"
                )

    def test_gradient_computation(self):
        """Test that gradients can be computed through the model."""

        # Test both models
        models = [
            ('rww', ReducedWongWang, 'S'),
            ('jr', JansenRit, 'y1'),
        ]

        for model_name, model_class, coupling_var in models:
            with self.subTest(model=model_name):
                # Create a simple network (no noise, no delay for gradient test)
                key = jax.random.PRNGKey(123)
                graph = DenseGraph.random(n_nodes=self.n_nodes, key=key)
                coupling = LinearCoupling(incoming_states=coupling_var, G=self.coupling_G)
                dynamics = model_class()

                network = Network(
                    dynamics=dynamics,
                    coupling={'instant': coupling},
                    graph=graph,
                    noise=None  # No noise for gradient test
                )

                # Prepare model
                solve_fn, state = prepare(network, Heun(), t0=self.t0, t1=self.t1, dt=self.dt)

                # Create wrapper function for gradient test
                def model_grad(G):
                    """Wrapper that takes coupling strength G and returns mean activity."""
                    # Create new state with updated G (avoid mutation)
                    from tvboptim.experimental.network_dynamics.core.bunch import Bunch

                    updated_coupling_params = Bunch(state.coupling.instant)
                    updated_coupling_params['G'] = G

                    updated_coupling = Bunch(state.coupling)
                    updated_coupling['instant'] = updated_coupling_params

                    updated_state = Bunch(state)
                    updated_state['coupling'] = updated_coupling

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
                    check_grads(model_grad, (self.coupling_G,), order=1, modes=['rev'], atol=1e-2, rtol=1e-2)
                except Exception as e:
                    self.fail(f"Gradient check failed for {model_name}: {str(e)}")


if __name__ == '__main__':
    unittest.main()
