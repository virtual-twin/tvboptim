"""Test execution module with NetworkDynamics models."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 for better numerical precision
jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.solve import prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun

from tvboptim.types import Space, GridAxis, UniformAxis, DataAxis
from tvboptim.execution import SequentialExecution, ParallelExecution


class TestExecutionWithNetworkDynamics(unittest.TestCase):
    """Test Sequential and Parallel execution with NetworkDynamics models."""

    def setUp(self):
        """Set up test parameters."""
        self.n_nodes = 3  # Smaller for faster execution
        self.t0 = 0.0
        self.t1 = 10.0  # Short simulation for fast tests
        self.dt = 0.1
        self.base_key = jax.random.PRNGKey(42)

    def test_execution_with_parameter_space(self):
        """Test Sequential and Parallel execution over parameter space."""

        # Create base network (RWW with linear coupling, no delay)
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

        # Embed axes directly into state structure
        # - GridAxis for coupling strength G
        # - UniformAxis for RWW parameter w
        # - DataAxis for input current I_o
        state.coupling.instant.G = GridAxis(0.05, 0.15, 3)  # 3 values
        state.dynamics.w = UniformAxis(0.8, 1.0, 2)  # 2 random values
        state.dynamics.I_o = DataAxis([0.3, 0.32])  # 2 values
        # Total: 3 × 2 × 2 = 12 parameter combinations

        # Create Space with the state containing axes (product mode for Cartesian product)
        space = Space(state, mode='product')

        # Define model function that takes state and returns mean activity
        def model_fn(state, key=None):
            """Run model with given state and return mean activity."""
            # Run model with the state (axes replaced with values)
            result = solve_fn(state)

            # Return mean activity of first state variable (S)
            first_state = result.ys[:, 0, :]  # [n_timesteps, n_nodes]
            return jnp.mean(first_state)

        # Test Sequential execution
        sequential_result = SequentialExecution(model_fn, space).run()

        # Verify sequential result
        self.assertEqual(len(sequential_result), 12, "Should have 12 results")
        self.assertFalse(any(jnp.isnan(r) for r in sequential_result), "No NaNs in sequential results")

        # Test Parallel execution
        parallel_result = ParallelExecution(model_fn, space).run()

        # Verify parallel result
        self.assertEqual(len(parallel_result), 12, "Should have 12 results")
        self.assertFalse(any(jnp.isnan(r) for r in parallel_result), "No NaNs in parallel results")

        # Compare Sequential and Parallel results
        for i, (seq_val, par_val) in enumerate(zip(sequential_result, parallel_result)):
            np.testing.assert_allclose(
                seq_val, par_val,
                rtol=1e-10, atol=1e-10,
                err_msg=f"Sequential and Parallel results differ at index {i}"
            )

        # Test iteration over results
        for i, result in enumerate(sequential_result):
            self.assertIsInstance(result, jnp.ndarray, f"Result {i} should be array")
            self.assertEqual(result.shape, (), f"Result {i} should be scalar")

        # Test indexing
        first_result = sequential_result[0]
        self.assertIsInstance(first_result, jnp.ndarray)

        # Test slicing
        slice_result = sequential_result[0:3]
        self.assertEqual(len(slice_result), 3)

    def test_execution_result_properties(self):
        """Test properties of execution results."""

        # Create base network
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

        # Prepare and embed axis
        solve_fn, state = prepare(network, Heun(), t0=self.t0, t1=self.t1, dt=self.dt)
        state.coupling.instant.G = GridAxis(0.1, 0.2, 2)  # 2 values

        # Create space
        space = Space(state)

        def model_fn(state, key=None):
            """Simple model function."""
            result = solve_fn(state)
            return jnp.mean(result.ys[:, 0, :])

        # Test Sequential
        seq_result = SequentialExecution(model_fn, space).run()

        # Test result container properties
        self.assertEqual(len(seq_result), 2)
        self.assertTrue(hasattr(seq_result, '__iter__'))
        self.assertTrue(hasattr(seq_result, '__getitem__'))

        # Test that results are arrays
        for r in seq_result:
            self.assertIsInstance(r, (jnp.ndarray, float))


if __name__ == '__main__':
    unittest.main()
