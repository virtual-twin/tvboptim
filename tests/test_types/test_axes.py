"""
Comprehensive tests for all axis implementations.

Tests cover:
- GridAxis: deterministic grid sampling with shape broadcasting
- UniformAxis: random sampling with shape broadcasting
- DataAxis: predefined data sampling
- NumPyroAxis: distribution sampling with broadcast vs independent modes
- AbstractAxis: base class behavior
"""

import unittest

import jax
import jax.numpy as jnp
import numpy.testing as np_testing

from tvboptim.types.spaces import AbstractAxis, DataAxis, GridAxis, UniformAxis

# NumPyro tests only if available
try:
    import numpyro.distributions as dist

    from tvboptim.types.spaces import NumPyroAxis

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False


class TestGridAxis(unittest.TestCase):
    """Test GridAxis deterministic grid sampling."""

    def test_basic_initialization(self):
        """Test basic GridAxis initialization."""
        axis = GridAxis(0.0, 1.0, 5)
        self.assertEqual(axis.low, 0.0)
        self.assertEqual(axis.high, 1.0)
        self.assertEqual(axis.n, 5)
        self.assertEqual(axis.size, 5)
        self.assertIsNone(axis.shape)

    def test_initialization_with_shape(self):
        """Test GridAxis initialization with shape parameter."""
        axis = GridAxis(0.0, 1.0, 3, shape=(2, 4))
        self.assertEqual(axis.shape, (2, 4))
        self.assertEqual(axis.size, 3)

    def test_initialization_errors(self):
        """Test GridAxis initialization error conditions."""
        # Invalid n
        with self.assertRaises(ValueError):
            GridAxis(0.0, 1.0, 0)
        with self.assertRaises(ValueError):
            GridAxis(0.0, 1.0, -1)

        # Invalid bounds
        with self.assertRaises(ValueError):
            GridAxis(1.0, 1.0, 5)  # low == high
        with self.assertRaises(ValueError):
            GridAxis(1.0, 0.0, 5)  # low > high

    def test_generate_values_basic(self):
        """Test basic value generation without shape."""
        axis = GridAxis(0.0, 1.0, 5)
        values = axis.generate_values()

        self.assertEqual(values.shape, (5,))
        expected = jnp.linspace(0.0, 1.0, 5)
        np_testing.assert_array_almost_equal(values, expected)

    def test_generate_values_with_shape(self):
        """Test value generation with shape broadcasting."""
        axis = GridAxis(0.0, 1.0, 3, shape=(2, 4))
        values = axis.generate_values()

        self.assertEqual(values.shape, (3, 2, 4))

        # Check that each (2, 4) slice has identical values
        for i in range(3):
            slice_vals = values[i]
            expected_val = jnp.linspace(0.0, 1.0, 3)[i]
            expected_slice = jnp.full((2, 4), expected_val)
            np_testing.assert_array_almost_equal(slice_vals, expected_slice)

    def test_generate_values_ignores_key(self):
        """Test that GridAxis ignores random key (deterministic)."""
        axis = GridAxis(0.0, 1.0, 5)
        values1 = axis.generate_values(jax.random.key(42))
        values2 = axis.generate_values(jax.random.key(123))

        np_testing.assert_array_equal(values1, values2)

    def test_repr(self):
        """Test string representation."""
        axis = GridAxis(0.0, 1.0, 5)
        repr_str = repr(axis)
        self.assertIn("GridAxis", repr_str)
        self.assertIn("0.0", repr_str)
        self.assertIn("1.0", repr_str)
        self.assertIn("5", repr_str)


class TestUniformAxis(unittest.TestCase):
    """Test UniformAxis random sampling."""

    def test_basic_initialization(self):
        """Test basic UniformAxis initialization."""
        axis = UniformAxis(0.0, 1.0, 5)
        self.assertEqual(axis.low, 0.0)
        self.assertEqual(axis.high, 1.0)
        self.assertEqual(axis.n, 5)
        self.assertEqual(axis.size, 5)
        self.assertIsNone(axis.shape)

    def test_initialization_with_shape(self):
        """Test UniformAxis initialization with shape parameter."""
        axis = UniformAxis(0.0, 1.0, 3, shape=(2, 4))
        self.assertEqual(axis.shape, (2, 4))
        self.assertEqual(axis.size, 3)

    def test_initialization_errors(self):
        """Test UniformAxis initialization error conditions."""
        # Invalid n
        with self.assertRaises(ValueError):
            UniformAxis(0.0, 1.0, 0)
        with self.assertRaises(ValueError):
            UniformAxis(0.0, 1.0, -1)

        # Invalid bounds
        with self.assertRaises(ValueError):
            UniformAxis(1.0, 1.0, 5)  # low == high
        with self.assertRaises(ValueError):
            UniformAxis(1.0, 0.0, 5)  # low > high

    def test_generate_values_basic(self):
        """Test basic value generation without shape."""
        axis = UniformAxis(0.0, 1.0, 10)
        values = axis.generate_values(jax.random.key(42))

        self.assertEqual(values.shape, (10,))
        # All values should be in bounds
        self.assertTrue(jnp.all(values >= 0.0))
        self.assertTrue(jnp.all(values <= 1.0))

    def test_generate_values_with_shape(self):
        """Test value generation with shape broadcasting."""
        axis = UniformAxis(0.0, 1.0, 3, shape=(2, 4))
        values = axis.generate_values(jax.random.key(42))

        self.assertEqual(values.shape, (3, 2, 4))

        # Check that each (2, 4) slice has identical values
        for i in range(3):
            slice_vals = values[i]
            first_val = slice_vals[0, 0]
            expected_slice = jnp.full((2, 4), first_val)
            np_testing.assert_array_equal(slice_vals, expected_slice)

    def test_generate_values_reproducible(self):
        """Test that same key produces same results."""
        axis = UniformAxis(0.0, 1.0, 5)
        values1 = axis.generate_values(jax.random.key(42))
        values2 = axis.generate_values(jax.random.key(42))

        np_testing.assert_array_equal(values1, values2)

    def test_generate_values_different_keys(self):
        """Test that different keys produce different results."""
        axis = UniformAxis(0.0, 1.0, 100)  # Large n for statistical difference
        values1 = axis.generate_values(jax.random.key(42))
        values2 = axis.generate_values(jax.random.key(123))

        # Should be statistically different
        self.assertFalse(jnp.allclose(values1, values2))

    def test_generate_values_default_key(self):
        """Test generation with default key (None)."""
        axis = UniformAxis(0.0, 1.0, 5)
        values = axis.generate_values()  # No key provided

        self.assertEqual(values.shape, (5,))
        self.assertTrue(jnp.all(values >= 0.0))
        self.assertTrue(jnp.all(values <= 1.0))

    def test_repr(self):
        """Test string representation."""
        axis = UniformAxis(0.0, 1.0, 5)
        repr_str = repr(axis)
        self.assertIn("UniformAxis", repr_str)
        self.assertIn("0.0", repr_str)
        self.assertIn("1.0", repr_str)
        self.assertIn("5", repr_str)


class TestDataAxis(unittest.TestCase):
    """Test DataAxis predefined data sampling."""

    def test_basic_initialization_list(self):
        """Test DataAxis initialization with Python list."""
        data = [1.0, 2.5, 3.7, 4.2]
        axis = DataAxis(data)
        self.assertEqual(axis.size, 4)
        np_testing.assert_array_equal(axis.values, jnp.array(data))

    def test_basic_initialization_array(self):
        """Test DataAxis initialization with JAX array."""
        data = jnp.linspace(0, 1, 5)
        axis = DataAxis(data)
        self.assertEqual(axis.size, 5)
        np_testing.assert_array_equal(axis.values, data)

    def test_multidimensional_data(self):
        """Test DataAxis with multidimensional data."""
        data = jnp.array([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
        axis = DataAxis(data)
        self.assertEqual(axis.size, 3)  # First dimension is axis size
        self.assertEqual(axis.values.shape, (3, 2))

    def test_initialization_error_empty(self):
        """Test DataAxis initialization with empty data."""
        with self.assertRaises(ValueError):
            DataAxis([])
        with self.assertRaises(ValueError):
            DataAxis(jnp.array([]))

    def test_generate_values(self):
        """Test value generation returns original data."""
        data = [1.0, 2.5, 3.7]
        axis = DataAxis(data)
        values = axis.generate_values()

        np_testing.assert_array_equal(values, jnp.array(data))

    def test_generate_values_ignores_key(self):
        """Test that DataAxis ignores random key (deterministic)."""
        data = [1.0, 2.5, 3.7]
        axis = DataAxis(data)
        values1 = axis.generate_values(jax.random.key(42))
        values2 = axis.generate_values(jax.random.key(123))

        np_testing.assert_array_equal(values1, values2)

    def test_repr(self):
        """Test string representation."""
        data = [1.0, 2.0, 3.0]
        axis = DataAxis(data)
        repr_str = repr(axis)
        self.assertIn("DataAxis", repr_str)


@unittest.skipUnless(NUMPYRO_AVAILABLE, "NumPyro not available")
class TestNumPyroAxis(unittest.TestCase):
    """Test NumPyroAxis distribution sampling."""

    def test_basic_initialization(self):
        """Test basic NumPyroAxis initialization."""
        distribution = dist.Normal(0.0, 1.0)
        axis = NumPyroAxis(distribution, n=5)

        self.assertEqual(axis.n, 5)
        self.assertEqual(axis.size, 5)
        self.assertEqual(axis.sample_shape, ())
        self.assertFalse(axis.broadcast_mode)

    def test_initialization_with_shape(self):
        """Test NumPyroAxis initialization with sample shape."""
        distribution = dist.Beta(2.0, 5.0)
        axis = NumPyroAxis(distribution, n=3, sample_shape=(2, 4), broadcast_mode=True)

        self.assertEqual(axis.sample_shape, (2, 4))
        self.assertTrue(axis.broadcast_mode)

    def test_initialization_errors(self):
        """Test NumPyroAxis initialization error conditions."""
        distribution = dist.Normal(0.0, 1.0)

        # Invalid n
        with self.assertRaises(ValueError):
            NumPyroAxis(distribution, n=0)
        with self.assertRaises(ValueError):
            NumPyroAxis(distribution, n=-1)

        # Invalid distribution
        with self.assertRaises(TypeError):
            NumPyroAxis("not_a_distribution", n=5)

    def test_generate_values_basic(self):
        """Test basic value generation without sample shape."""
        distribution = dist.Normal(0.0, 1.0)
        axis = NumPyroAxis(distribution, n=10)
        values = axis.generate_values(jax.random.key(42))

        self.assertEqual(values.shape, (10,))

    def test_generate_values_independent_mode(self):
        """Test independent sampling mode."""
        distribution = dist.Uniform(0.0, 1.0)
        axis = NumPyroAxis(distribution, n=3, sample_shape=(2, 4), broadcast_mode=False)
        values = axis.generate_values(jax.random.key(42))

        self.assertEqual(values.shape, (3, 2, 4))

        # Each element should be different (statistically)
        flat_values = values.flatten()
        unique_values = jnp.unique(flat_values)
        # Should have many unique values (not exactly 24 due to floating point, but close)
        self.assertGreater(len(unique_values), 20)

    def test_generate_values_broadcast_mode(self):
        """Test broadcast sampling mode."""
        distribution = dist.Uniform(0.0, 1.0)
        axis = NumPyroAxis(distribution, n=3, sample_shape=(2, 4), broadcast_mode=True)
        values = axis.generate_values(jax.random.key(42))

        self.assertEqual(values.shape, (3, 2, 4))

        # Each (2, 4) slice should have identical values
        for i in range(3):
            slice_vals = values[i]
            first_val = slice_vals[0, 0]
            expected_slice = jnp.full((2, 4), first_val)
            np_testing.assert_array_almost_equal(slice_vals, expected_slice)

    def test_generate_values_reproducible(self):
        """Test that same key produces same results."""
        distribution = dist.Normal(0.0, 1.0)
        axis = NumPyroAxis(distribution, n=5)
        values1 = axis.generate_values(jax.random.key(42))
        values2 = axis.generate_values(jax.random.key(42))

        np_testing.assert_array_equal(values1, values2)

    def test_different_distributions(self):
        """Test various NumPyro distributions."""
        key = jax.random.key(42)

        # Normal distribution
        normal_axis = NumPyroAxis(dist.Normal(0.0, 1.0), n=5)
        normal_values = normal_axis.generate_values(key)
        self.assertEqual(normal_values.shape, (5,))

        # Beta distribution
        beta_axis = NumPyroAxis(dist.Beta(2.0, 5.0), n=3)
        beta_values = beta_axis.generate_values(key)
        self.assertEqual(beta_values.shape, (3,))
        self.assertTrue(jnp.all(beta_values >= 0.0))
        self.assertTrue(jnp.all(beta_values <= 1.0))

        # Exponential distribution
        exp_axis = NumPyroAxis(dist.Exponential(1.0), n=4)
        exp_values = exp_axis.generate_values(key)
        self.assertEqual(exp_values.shape, (4,))
        self.assertTrue(jnp.all(exp_values >= 0.0))

    def test_multivariate_distribution(self):
        """Test with multivariate distribution."""
        loc = jnp.array([0.0, 1.0])
        cov = jnp.eye(2)
        distribution = dist.MultivariateNormal(loc, cov)

        axis = NumPyroAxis(distribution, n=3)
        values = axis.generate_values(jax.random.key(42))

        # Should have shape (n, event_shape)
        self.assertEqual(values.shape, (3, 2))

    def test_repr(self):
        """Test string representation."""
        distribution = dist.Normal(0.0, 1.0)
        axis = NumPyroAxis(distribution, n=5)
        repr_str = repr(axis)
        self.assertIn("NumPyroAxis", repr_str)
        self.assertIn("5", repr_str)


class TestAbstractAxisBehavior(unittest.TestCase):
    """Test AbstractAxis base class behavior and interface."""

    def test_cannot_instantiate_abstract_axis(self):
        """Test that AbstractAxis cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            AbstractAxis()

    def test_axis_interface_compliance(self):
        """Test that all concrete axes implement the required interface."""
        axes_to_test = [
            GridAxis(0.0, 1.0, 5),
            UniformAxis(0.0, 1.0, 5),
            DataAxis([1, 2, 3, 4, 5]),
        ]

        if NUMPYRO_AVAILABLE:
            axes_to_test.append(NumPyroAxis(dist.Normal(0.0, 1.0), 5))

        for axis in axes_to_test:
            # Should have generate_values method
            self.assertTrue(hasattr(axis, "generate_values"))
            self.assertTrue(callable(axis.generate_values))

            # Should have size property
            self.assertTrue(hasattr(axis, "size"))
            self.assertIsInstance(axis.size, int)
            self.assertGreater(axis.size, 0)

            # generate_values should return JAX array
            values = axis.generate_values(jax.random.key(42))
            self.assertIsInstance(values, jnp.ndarray)

            # First dimension should match size
            self.assertEqual(values.shape[0], axis.size)


if __name__ == "__main__":
    unittest.main()
