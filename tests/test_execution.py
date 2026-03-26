import os
import unittest

# Set up CPU environment for testing
cpu = True
if cpu:
    N = 8
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N}"
import jax.numpy as jnp

from tvboptim.execution import (
    ParallelExecution,
    ParallelResult,
    SequentialExecution,
    SequentialResult,
)
from tvboptim.types import DataAxis, Space


class TestExecution(unittest.TestCase):
    """Test cases for Sequential and Parallel execution."""

    def setUp(self):
        """Set up test fixtures before each test method."""

        # Create test state and data space
        # Use DataAxis to create a space with 10 parameter combinations
        self.state = {
            "a": DataAxis(jnp.linspace(1.0, 10.0, 10)),
            "b": DataAxis(jnp.linspace(10.0, 20.0, 10)),
        }
        self.ds = Space(self.state)

        # Define test function
        def f(state, key=None):
            # State is now a dict with 'a' and 'b' keys
            return [state["a"], state["b"]]

        self.f = f

    def test_sequential_execution(self):
        """Test SequentialExecution functionality."""
        result_s = SequentialExecution(self.f, self.ds).run()

        # Test result type
        self.assertIsInstance(result_s, SequentialResult)

        # Test result length
        self.assertEqual(len(result_s), 10)

        # Test iteration
        count = 0
        for r in result_s:
            count += 1
            self.assertIsInstance(r, list)
            self.assertEqual(len(r), 2)
        self.assertEqual(count, 10)

        # Test indexing
        first_result = result_s[0]
        self.assertIsInstance(first_result, list)
        self.assertEqual(len(first_result), 2)

    def test_parallel_execution_pmap(self):
        """Test ParallelExecution with n_pmap."""
        result_p = ParallelExecution(self.f, self.ds, n_pmap=5).run()

        # Test result type
        self.assertIsInstance(result_p, ParallelResult)

        # Test result length
        self.assertEqual(len(result_p), 10)

        # Test iteration
        count = 0
        for r in result_p:
            count += 1
            self.assertIsInstance(r, list)
            self.assertEqual(len(r), 2)
        self.assertEqual(count, 10)

        # Test indexing
        first_result = result_p[0]
        self.assertIsInstance(first_result, list)
        self.assertEqual(len(first_result), 2)

    def test_parallel_execution_vmap(self):
        """Test ParallelExecution with n_vmap."""
        result_p2 = ParallelExecution(self.f, self.ds, n_vmap=5).run()

        # Test result type
        self.assertIsInstance(result_p2, ParallelResult)

        # Test result length
        self.assertEqual(len(result_p2), 10)

        # Test iteration
        count = 0
        for r in result_p2:
            count += 1
            self.assertIsInstance(r, list)
            self.assertEqual(len(r), 2)
        self.assertEqual(count, 10)

        # Test indexing
        first_result = result_p2[0]
        self.assertIsInstance(first_result, list)
        self.assertEqual(len(first_result), 2)

    def test_execution_results_consistency(self):
        """Test that Sequential and Parallel executions produce consistent results."""
        result_s = SequentialExecution(self.f, self.ds).run()
        result_p = ParallelExecution(self.f, self.ds, n_pmap=5).run()
        result_p2 = ParallelExecution(self.f, self.ds, n_vmap=5).run()

        # Compare first 5 results
        for i in range(5):
            s_result = result_s[i]
            p_result = result_p[i]
            p2_result = result_p2[i]

            # Results should be approximately equal
            self.assertAlmostEqual(float(s_result[0]), float(p_result[0]), places=5)
            self.assertAlmostEqual(float(s_result[1]), float(p_result[1]), places=5)
            self.assertAlmostEqual(float(s_result[0]), float(p2_result[0]), places=5)
            self.assertAlmostEqual(float(s_result[1]), float(p2_result[1]), places=5)

    def test_result_slicing(self):
        """Test slicing functionality of results."""
        result_s = SequentialExecution(self.f, self.ds).run()
        result_p = ParallelExecution(self.f, self.ds, n_pmap=5).run()
        result_p2 = ParallelExecution(self.f, self.ds, n_vmap=5).run()

        # Test slicing
        s_slice = result_s[0:5]
        p_slice = result_p[0:5]
        p2_slice = result_p2[0:5]

        # All slices should have length 5
        self.assertEqual(len(s_slice), 5)
        self.assertEqual(len(p_slice), 5)
        self.assertEqual(len(p2_slice), 5)

        # Compare sliced results
        for i in range(5):
            self.assertAlmostEqual(float(s_slice[i][0]), float(p_slice[i][0]), places=5)
            self.assertAlmostEqual(float(s_slice[i][1]), float(p_slice[i][1]), places=5)
            self.assertAlmostEqual(
                float(s_slice[i][0]), float(p2_slice[i][0]), places=5
            )
            self.assertAlmostEqual(
                float(s_slice[i][1]), float(p2_slice[i][1]), places=5
            )

    def test_execution_with_different_functions(self):
        """Test execution with different model functions."""

        def sum_function(state, key=None):
            return [state["a"] + state["b"]]

        def product_function(state, key=None):
            return [state["a"] * state["b"]]

        # Test with sum function
        result_sum_s = SequentialExecution(sum_function, self.ds).run()
        result_sum_p = ParallelExecution(sum_function, self.ds, n_pmap=2).run()

        self.assertEqual(len(result_sum_s), 10)
        self.assertEqual(len(result_sum_p), 10)

        # Test with product function
        result_prod_s = SequentialExecution(product_function, self.ds).run()
        result_prod_p = ParallelExecution(product_function, self.ds, n_vmap=2).run()

        self.assertEqual(len(result_prod_s), 10)
        self.assertEqual(len(result_prod_p), 10)

        # Verify consistency between sequential and parallel for same function
        for i in range(5):
            self.assertAlmostEqual(
                float(result_sum_s[i][0]), float(result_sum_p[i][0]), places=5
            )
            self.assertAlmostEqual(
                float(result_prod_s[i][0]), float(result_prod_p[i][0]), places=5
            )


class TestResultToDataFrame(unittest.TestCase):
    """Test Result.to_dataframe() functionality."""

    def setUp(self):
        self.state = {
            "a": DataAxis(jnp.linspace(1.0, 5.0, 5)),
            "b": DataAxis(jnp.linspace(10.0, 50.0, 5)),
        }
        self.space = Space(self.state)

    def test_sequential_result_has_space(self):
        """SequentialResult stores space reference."""

        def f(state):
            return {"sum": state["a"] + state["b"]}

        result = SequentialExecution(f, self.space).run()
        self.assertIs(result.space, self.space)

    def test_sequential_dict_result_columns(self):
        """Dict results produce named columns."""

        def f(state):
            return {"sum": state["a"] + state["b"], "product": state["a"] * state["b"]}

        result = SequentialExecution(f, self.space).run()
        df = result.to_dataframe()

        self.assertEqual(len(df), 5)
        # Param columns
        self.assertIn("a", df.columns)
        self.assertIn("b", df.columns)
        # Result columns
        self.assertIn("sum", df.columns)
        self.assertIn("product", df.columns)

    def test_sequential_list_result_columns(self):
        """List results produce bracketed column names."""

        def f(state):
            return [state["a"], state["b"]]

        result = SequentialExecution(f, self.space).run()
        df = result.to_dataframe()

        self.assertIn("[0]", df.columns)
        self.assertIn("[1]", df.columns)

    def test_sequential_single_array_result(self):
        """Single array result produces 'value' column."""

        def f(state):
            return state["a"] + state["b"]

        result = SequentialExecution(f, self.space).run()
        df = result.to_dataframe()

        self.assertIn("value", df.columns)

    def test_name_collision_prefixed(self):
        """Result column colliding with param name gets 'result.' prefix."""

        def f(state):
            return {"a": state["a"] * 2}  # 'a' collides with param 'a'

        result = SequentialExecution(f, self.space).run()
        df = result.to_dataframe()

        self.assertIn("a", df.columns)  # param
        self.assertIn("result.a", df.columns)  # result, prefixed

    def test_parallel_result_has_space(self):
        """ParallelResult stores space reference."""

        def f(state):
            return {"sum": state["a"] + state["b"]}

        result = ParallelExecution(f, self.space, n_pmap=1).run()
        self.assertIs(result.space, self.space)

    def test_parallel_result_to_dataframe(self):
        """ParallelResult.to_dataframe works like sequential."""

        def f(state):
            return {"sum": state["a"] + state["b"]}

        result = ParallelExecution(f, self.space, n_pmap=1).run()
        df = result.to_dataframe()

        self.assertEqual(len(df), 5)
        self.assertIn("a", df.columns)
        self.assertIn("sum", df.columns)

    def test_array_result_in_cell(self):
        """Non-scalar results stored as array objects in cells."""
        import numpy as np

        def f(state):
            return {"timeseries": jnp.ones(10) * state["a"]}

        result = SequentialExecution(f, self.space).run()
        df = result.to_dataframe()

        self.assertIn("timeseries", df.columns)
        self.assertEqual(np.asarray(df["timeseries"].iloc[0]).shape, (10,))

    def test_nested_dict_result(self):
        """Nested dict results produce dot-separated names."""

        def f(state):
            return {"metrics": {"mean": state["a"], "std": state["b"]}}

        result = SequentialExecution(f, self.space).run()
        df = result.to_dataframe()

        self.assertIn("metrics.mean", df.columns)
        self.assertIn("metrics.std", df.columns)


if __name__ == "__main__":
    unittest.main()
