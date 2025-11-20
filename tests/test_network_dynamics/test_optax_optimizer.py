"""Test OptaxOptimizer with NetworkDynamics models."""

import unittest

import jax
import jax.numpy as jnp
import optax

# Enable float64 for better numerical precision
jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.solve import prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun

from tvboptim.types import Parameter
from tvboptim.optim import OptaxOptimizer


class TestOptaxOptimizerWithNetworkDynamics(unittest.TestCase):
    """Test OptaxOptimizer with NetworkDynamics models."""

    def setUp(self):
        """Set up test parameters."""
        self.n_nodes = 3
        self.t0 = 0.0
        self.t1 = 10.0
        self.dt = 0.1
        self.target_firing_rate = 0.5
        self.base_key = jax.random.PRNGKey(42)

    def test_optimizer_basic_optimization(self):
        """Test basic optimization loop with OptaxOptimizer."""

        # Create RWW network
        key = self.base_key
        graph = DenseGraph.random(n_nodes=self.n_nodes, key=key)
        coupling = LinearCoupling(incoming_states='S', G=0.1)  # Start with G=0.1
        dynamics = ReducedWongWang()

        network = Network(
            dynamics=dynamics,
            coupling={'instant': coupling},
            graph=graph,
            noise=None
        )

        # Prepare model
        solve_fn, state = prepare(network, Heun(), t0=self.t0, t1=self.t1, dt=self.dt)

        # Mark G as optimizable parameter
        state.coupling.instant.G = Parameter(0.1)

        # Define loss function: we want mean firing rate close to target
        def loss_fn(state):
            """Loss function: squared error of mean firing rate from target."""
            # Run simulation
            result = solve_fn(state)

            # Compute mean firing rate (mean activity of S state)
            mean_firing_rate = jnp.mean(result.ys[:, 0, :])

            # Squared error from target
            loss = (mean_firing_rate - self.target_firing_rate) ** 2

            return loss

        # Get initial loss
        initial_loss = loss_fn(state)

        # Create optimizer
        optimizer = OptaxOptimizer(
            loss=loss_fn,
            optimizer=optax.adam(learning_rate=0.01)
        )

        # Run optimization
        final_state, fitting_data = optimizer.run(state, max_steps=10)

        # Get final loss
        final_loss = loss_fn(final_state)

        # Verify optimization ran
        self.assertIsNotNone(final_state, "Optimization should return final state")
        self.assertIsNotNone(fitting_data, "Optimization should return fitting data")

        # Verify loss decreased (or stayed roughly the same if already optimal)
        self.assertLessEqual(
            final_loss, initial_loss * 1.1,  # Allow 10% tolerance
            f"Final loss ({final_loss:.4f}) should be <= initial loss ({initial_loss:.4f})"
        )

        # Verify final loss is closer to target than initial
        initial_error = jnp.abs(jnp.sqrt(initial_loss))
        final_error = jnp.abs(jnp.sqrt(final_loss))

        # At least some improvement or already good
        self.assertTrue(
            final_error <= initial_error or final_error < 0.1,
            f"Final error ({final_error:.4f}) should improve or be small"
        )

        # Verify G parameter changed
        initial_G = 0.1
        final_G = final_state.coupling.instant.G

        self.assertNotEqual(
            final_G, initial_G,
            "G parameter should have changed during optimization"
        )

        # Print results for inspection
        print(f"\nOptimization Results:")
        print(f"  Initial G: {initial_G:.4f}")
        print(f"  Final G: {final_G:.4f}")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Target firing rate: {self.target_firing_rate:.4f}")
        print(f"  Initial error: {initial_error:.4f}")
        print(f"  Final error: {final_error:.4f}")

    def test_optimizer_with_multiple_parameters(self):
        """Test optimization with multiple parameters."""

        # Create RWW network
        key = self.base_key
        graph = DenseGraph.random(n_nodes=self.n_nodes, key=key)
        coupling = LinearCoupling(incoming_states='S', G=0.1)
        dynamics = ReducedWongWang()

        network = Network(
            dynamics=dynamics,
            coupling={'instant': coupling},
            graph=graph,
            noise=None
        )

        # Prepare model
        solve_fn, state = prepare(network, Heun(), t0=self.t0, t1=self.t1, dt=self.dt)

        # Mark both G and w as optimizable parameters
        state.coupling.instant.G = Parameter(0.1)
        state.dynamics.w = Parameter(0.9)

        # Define loss function
        def loss_fn(state):
            result = solve_fn(state)
            mean_firing_rate = jnp.mean(result.ys[:, 0, :])
            loss = (mean_firing_rate - self.target_firing_rate) ** 2
            return loss

        initial_G = 0.1
        initial_w = 0.9

        # Create and run optimizer
        optimizer = OptaxOptimizer(
            loss=loss_fn,
            optimizer=optax.adam(learning_rate=0.01)
        )

        final_state, fitting_data = optimizer.run(state, max_steps=5)  # Fewer steps for this test

        # Verify both parameters changed
        final_G = final_state.coupling.instant.G
        final_w = final_state.dynamics.w

        # At least one parameter should have changed significantly
        G_changed = jnp.abs(final_G - initial_G) > 1e-4
        w_changed = jnp.abs(final_w - initial_w) > 1e-4

        self.assertTrue(
            G_changed or w_changed,
            "At least one parameter should have changed during optimization"
        )

        print(f"\nMultiple Parameter Optimization:")
        print(f"  Initial G: {initial_G:.4f}, Final G: {final_G:.4f}")
        print(f"  Initial w: {initial_w:.4f}, Final w: {final_w:.4f}")


if __name__ == '__main__':
    unittest.main()
