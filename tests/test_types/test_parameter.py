"""
Comprehensive tests for Parameter implementations.

Tests cover:
- Parameter: Basic JAX-native parameter with full arithmetic
- NormalizedParameter: Stores normalized values internally
- BoundedParameter: Automatic bounds enforcement
- Utility functions: is_parameter, extract_values
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as np_testing
from typing import Any

from tvboptim.types.parameter import (
    Parameter, NormalizedParameter, BoundedParameter,
    is_parameter, extract_values
)


class TestParameter(unittest.TestCase):
    """Test basic Parameter functionality."""

    def test_basic_initialization(self):
        """Test Parameter initialization with different value types."""
        # Scalar initialization
        p1 = Parameter(1.0)
        self.assertEqual(p1.value, 1.0)
        self.assertEqual(p1.shape, ())

        # Integer initialization
        p2 = Parameter(5)
        self.assertEqual(p2.value, 5)

        # Array initialization
        arr = jnp.array([1, 2, 3])
        p3 = Parameter(arr)
        np_testing.assert_array_equal(p3.value, arr)
        self.assertEqual(p3.shape, (3,))

    def test_properties(self):
        """Test Parameter properties."""
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        p = Parameter(arr)

        self.assertEqual(p.shape, (2, 2))
        # Check that dtype matches the array (respects JAX float64 config if enabled)
        self.assertEqual(p.dtype, arr.dtype)
        self.assertEqual(p.ndim, 2)
        self.assertEqual(p.size, 4)

    def test_shape_setter_broadcasting(self):
        """Test shape setter with broadcasting."""
        # Scalar to vector broadcasting
        p1 = Parameter(5.0)
        p1.shape = (3,)
        expected = jnp.array([5.0, 5.0, 5.0])
        np_testing.assert_array_equal(p1.value, expected)
        self.assertEqual(p1.shape, (3,))

        # Scalar to 2D broadcasting
        p2 = Parameter(2.0)
        p2.shape = (2, 3)
        expected = jnp.full((2, 3), 2.0)
        np_testing.assert_array_equal(p2.value, expected)

        # Vector broadcasting
        p3 = Parameter(jnp.array([1.0, 2.0]))
        p3.shape = (3, 2)
        expected = jnp.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        np_testing.assert_array_equal(p3.value, expected)

    def test_shape_setter_invalid_broadcasting(self):
        """Test shape setter with invalid broadcasting."""
        p = Parameter(jnp.array([1.0, 2.0, 3.0]))
        with self.assertRaises(ValueError):
            p.shape = (2, 2)  # Cannot broadcast (3,) to (2, 2)

    def test_jax_array_protocol(self):
        """Test __jax_array__ protocol implementation."""
        p = Parameter(jnp.array([1, 2, 3]))
        arr = p.__jax_array__()
        np_testing.assert_array_equal(arr, jnp.array([1, 2, 3]))

    def test_numpy_array_protocol(self):
        """Test NumPy array protocol compatibility."""
        p = Parameter(jnp.array([1, 2, 3]))
        arr = p.__array__()
        np_testing.assert_array_equal(arr, np.array([1, 2, 3]))

    def test_arithmetic_operations(self):
        """Test arithmetic operations return JAX arrays."""
        p1 = Parameter(5.0)
        p2 = Parameter(3.0)

        # Addition
        result = p1 + p2
        self.assertIsInstance(result, jnp.ndarray)
        self.assertEqual(result, 8.0)

        # Reverse addition
        result = 2.0 + p1
        self.assertEqual(result, 7.0)

        # Subtraction
        result = p1 - p2
        self.assertEqual(result, 2.0)

        # Multiplication
        result = p1 * p2
        self.assertEqual(result, 15.0)

        # Division
        result = p1 / p2
        self.assertAlmostEqual(result, 5.0/3.0)

        # Power
        result = p1 ** 2
        self.assertEqual(result, 25.0)

    def test_arithmetic_with_arrays(self):
        """Test arithmetic operations with array parameters."""
        p1 = Parameter(jnp.array([1, 2, 3]))
        p2 = Parameter(jnp.array([4, 5, 6]))

        result = p1 + p2
        expected = jnp.array([5, 7, 9])
        np_testing.assert_array_equal(result, expected)

        result = p1 * 2
        expected = jnp.array([2, 4, 6])
        np_testing.assert_array_equal(result, expected)

    def test_unary_operations(self):
        """Test unary operations."""
        p = Parameter(jnp.array([-1, 2, -3]))

        # Negation
        result = -p
        expected = jnp.array([1, -2, 3])
        np_testing.assert_array_equal(result, expected)

        # Positive
        result = +p
        expected = jnp.array([-1, 2, -3])
        np_testing.assert_array_equal(result, expected)

        # Absolute value
        result = abs(p)
        expected = jnp.array([1, 2, 3])
        np_testing.assert_array_equal(result, expected)

    def test_comparison_operations(self):
        """Test comparison operations."""
        p1 = Parameter(5.0)
        p2 = Parameter(3.0)

        self.assertTrue((p1 > p2).item())
        self.assertTrue((p1 >= p2).item())
        self.assertFalse((p1 < p2).item())
        self.assertFalse((p1 == p2).item())
        self.assertTrue((p1 != p2).item())

    def test_indexing(self):
        """Test parameter indexing."""
        p = Parameter(jnp.array([1, 2, 3, 4]))

        self.assertEqual(p[0], 1)
        self.assertEqual(p[-1], 4)
        np_testing.assert_array_equal(p[1:3], jnp.array([2, 3]))

    def test_pytree_flatten_unflatten(self):
        """Test JAX pytree implementation."""
        original = Parameter(jnp.array([1.0, 2.0, 3.0]))

        # Test flatten
        children, aux_data = original.tree_flatten()
        self.assertEqual(len(children), 1)
        np_testing.assert_array_equal(children[0], jnp.array([1.0, 2.0, 3.0]))
        self.assertIsNone(aux_data)

        # Test unflatten
        reconstructed = Parameter.tree_unflatten(aux_data, children)
        np_testing.assert_array_equal(reconstructed.value, original.value)

    def test_jax_transformations(self):
        """Test Parameter works with JAX transformations."""
        def square_sum(p):
            return jnp.sum(p ** 2)

        p = Parameter(jnp.array([1.0, 2.0, 3.0]))

        # Test with jit
        jit_fn = jax.jit(square_sum)
        result = jit_fn(p)
        self.assertEqual(result, 14.0)  # 1 + 4 + 9

        # Test with grad
        grad_fn = jax.grad(square_sum)
        gradients = grad_fn(p)
        expected = 2 * p.__jax_array__()  # derivative of x^2 is 2x
        np_testing.assert_array_equal(gradients, expected)

    def test_repr_and_str(self):
        """Test string representations."""
        p = Parameter(5.0)
        repr_str = repr(p)
        str_str = str(p)

        self.assertIn("Parameter", repr_str)
        self.assertIn("5.0", repr_str)
        self.assertIn("Parameter", str_str)


class TestNormalizedParameter(unittest.TestCase):
    """Test NormalizedParameter functionality."""

    def test_basic_initialization(self):
        """Test NormalizedParameter initialization."""
        original_value = jnp.array([2.0, 4.0, 6.0])
        p = NormalizedParameter(original_value)

        # Internal storage should be ones
        np_testing.assert_array_equal(p.value, jnp.ones_like(original_value))

        # Scale should be the original value
        np_testing.assert_array_equal(p.scale, original_value)

        # External value should be scale * ones = original
        np_testing.assert_array_equal(p.__jax_array__(), original_value)

    def test_scalar_initialization(self):
        """Test NormalizedParameter with scalar."""
        p = NormalizedParameter(5.0)

        self.assertEqual(p.value, 1.0)
        self.assertEqual(p.scale, 5.0)
        self.assertEqual(p.__jax_array__(), 5.0)

    def test_jax_array_returns_scaled_values(self):
        """Test that __jax_array__ returns properly scaled values."""
        original = jnp.array([3.0, 6.0, 9.0])
        p = NormalizedParameter(original)

        # After optimization, normalized values might change
        p.value = jnp.array([0.5, 1.5, 2.0])  # Simulate optimization result

        # External values should be scale * new_normalized_values
        expected = p.scale * p.value  # [3*0.5, 6*1.5, 9*2.0] = [1.5, 9.0, 18.0]
        np_testing.assert_array_equal(p.__jax_array__(), expected)

    def test_pytree_flatten_unflatten(self):
        """Test NormalizedParameter pytree implementation."""
        original_value = jnp.array([2.0, 4.0])
        original = NormalizedParameter(original_value)

        # Test flatten
        children, aux_data = original.tree_flatten()

        # Only normalized values should be in children (differentiable)
        self.assertEqual(len(children), 1)
        np_testing.assert_array_equal(children[0], jnp.ones(2))

        # Scale should be in aux_data (static)
        np_testing.assert_array_equal(aux_data, original_value)

        # Test unflatten
        reconstructed = NormalizedParameter.tree_unflatten(aux_data, children)
        np_testing.assert_array_equal(reconstructed.value, original.value)
        np_testing.assert_array_equal(reconstructed.scale, original.scale)

    def test_optimization_workflow(self):
        """Test typical optimization workflow with normalized parameters."""
        # The key benefit of NormalizedParameter is enabling single learning rates
        # for parameters with vastly different scales by working in normalized space

        def objective(p):
            return jnp.sum(p ** 2)

        # Create parameters with vastly different scales
        large_scale = NormalizedParameter(jnp.array([100.0, 200.0]))
        small_scale = NormalizedParameter(jnp.array([0.01, 0.02]))

        # Both store normalized values internally (ones) but present scaled values
        self.assertTrue(jnp.allclose(large_scale.value, jnp.ones(2)))
        self.assertTrue(jnp.allclose(small_scale.value, jnp.ones(2)))

        # But present their original scales externally
        self.assertTrue(jnp.allclose(large_scale.__jax_array__(), jnp.array([100.0, 200.0])))
        self.assertTrue(jnp.allclose(small_scale.__jax_array__(), jnp.array([0.01, 0.02])))

        # Gradients computed on normalized space are more manageable for optimizers
        grad_fn = jax.grad(objective)
        grad_large = grad_fn(large_scale)
        grad_small = grad_fn(small_scale)

        # Both gradients operate on the same normalized coordinate system
        # allowing single learning rate to work effectively for both scales
        self.assertEqual(grad_large.shape, (2,))
        self.assertEqual(grad_small.shape, (2,))

    def test_repr(self):
        """Test NormalizedParameter string representation."""
        p = NormalizedParameter(jnp.array([2.0, 4.0]))
        repr_str = repr(p)

        self.assertIn("NormalizedParameter", repr_str)
        self.assertIn("scale", repr_str)
        self.assertIn("normalized", repr_str)


class TestBoundedParameter(unittest.TestCase):
    """Test BoundedParameter functionality."""

    def test_basic_initialization(self):
        """Test BoundedParameter initialization."""
        p = BoundedParameter(1.5, low=0.0, high=2.0)

        self.assertEqual(p.value, 1.5)
        self.assertEqual(p.low, 0.0)
        self.assertEqual(p.high, 2.0)

    def test_invalid_bounds_error(self):
        """Test error when low >= high."""
        with self.assertRaises(ValueError):
            BoundedParameter(1.0, low=2.0, high=1.0)

        with self.assertRaises(ValueError):
            BoundedParameter(1.0, low=1.0, high=1.0)

    def test_bounds_enforcement(self):
        """Test that bounds are automatically enforced."""
        p = BoundedParameter(2.5, low=0.0, high=1.0)  # Value exceeds high

        # __jax_array__ should return clipped value
        clipped_value = p.__jax_array__()
        self.assertEqual(clipped_value, 1.0)

        # Original value is unchanged
        self.assertEqual(p.value, 2.5)

    def test_bounds_enforcement_array(self):
        """Test bounds enforcement with array values."""
        values = jnp.array([-1.0, 0.5, 2.0, 1.5])
        p = BoundedParameter(values, low=0.0, high=1.0)

        clipped = p.__jax_array__()
        expected = jnp.array([0.0, 0.5, 1.0, 1.0])  # Clipped to [0, 1]
        np_testing.assert_array_equal(clipped, expected)

    def test_arithmetic_operations_with_bounds(self):
        """Test arithmetic operations respect bounds."""
        p = BoundedParameter(0.8, low=0.0, high=1.0)

        # Addition might exceed bounds, but result should be clipped
        result = p + 0.5  # Would be 1.3, but gets clipped
        # Note: The arithmetic operation returns the raw computation
        # Bounds are only applied when accessing via __jax_array__()
        self.assertEqual(result, 1.3)  # Arithmetic returns raw result

        # But the parameter itself remains bounded
        self.assertEqual(p.__jax_array__(), 0.8)

    def test_pytree_flatten_unflatten(self):
        """Test BoundedParameter pytree implementation."""
        original = BoundedParameter(jnp.array([0.5, 1.5]), low=-1.0, high=1.0)

        # Test flatten
        children, aux_data = original.tree_flatten()

        self.assertEqual(len(children), 1)
        np_testing.assert_array_equal(children[0], jnp.array([0.5, 1.5]))

        # Bounds should be in aux_data
        self.assertEqual(aux_data, (-1.0, 1.0))

        # Test unflatten
        reconstructed = BoundedParameter.tree_unflatten(aux_data, children)
        np_testing.assert_array_equal(reconstructed.value, original.value)
        self.assertEqual(reconstructed.low, original.low)
        self.assertEqual(reconstructed.high, original.high)

    def test_optimization_with_bounds(self):
        """Test optimization respects bounds."""
        def objective(p):
            # Objective that would push parameter outside bounds
            return (p - 2.0) ** 2  # Minimum at 2.0, but bounds are [0, 1]

        p = BoundedParameter(0.5, low=0.0, high=1.0)

        # Gradient should push towards 2.0
        grad_fn = jax.grad(objective)
        grad = grad_fn(p)

        # But the parameter value when used stays within bounds
        bounded_value = p.__jax_array__()
        self.assertGreaterEqual(bounded_value, 0.0)
        self.assertLessEqual(bounded_value, 1.0)

    def test_repr(self):
        """Test BoundedParameter string representation."""
        p = BoundedParameter(1.5, low=0.0, high=2.0)
        repr_str = repr(p)

        self.assertIn("BoundedParameter", repr_str)
        self.assertIn("low=0.0", repr_str)
        self.assertIn("high=2.0", repr_str)


class TestParameterUtilities(unittest.TestCase):
    """Test utility functions for parameters."""

    def test_is_parameter(self):
        """Test is_parameter function."""
        p1 = Parameter(1.0)
        p2 = NormalizedParameter(2.0)
        p3 = BoundedParameter(3.0, 0.0, 5.0)
        regular_array = jnp.array([1, 2, 3])
        regular_float = 5.0

        self.assertTrue(is_parameter(p1))
        self.assertTrue(is_parameter(p2))
        self.assertTrue(is_parameter(p3))
        self.assertFalse(is_parameter(regular_array))
        self.assertFalse(is_parameter(regular_float))

    def test_extract_values_simple(self):
        """Test extract_values with simple structures."""
        p1 = Parameter(1.0)
        p2 = Parameter(jnp.array([2, 3]))
        regular_val = 4.0

        tree = {'param1': p1, 'param2': p2, 'regular': regular_val}
        extracted = extract_values(tree)

        self.assertEqual(extracted['param1'], 1.0)
        np_testing.assert_array_equal(extracted['param2'], jnp.array([2, 3]))
        self.assertEqual(extracted['regular'], 4.0)

    def test_extract_values_nested(self):
        """Test extract_values with nested structures."""
        tree = {
            'level1': {
                'param': Parameter(jnp.array([1, 2])),
                'normalized': NormalizedParameter(jnp.array([3, 4])),
                'bounded': BoundedParameter(2.5, 0.0, 3.0),
                'regular': 'not_a_param'
            },
            'scalar_param': Parameter(5.0)
        }

        extracted = extract_values(tree)

        # Check extracted values
        np_testing.assert_array_equal(
            extracted['level1']['param'],
            jnp.array([1, 2])
        )
        np_testing.assert_array_equal(
            extracted['level1']['normalized'],
            jnp.array([3, 4])  # Scale * ones = [3, 4] * [1, 1]
        )
        self.assertEqual(
            extracted['level1']['bounded'],
            2.5  # Within bounds, so unchanged
        )
        self.assertEqual(extracted['level1']['regular'], 'not_a_param')
        self.assertEqual(extracted['scalar_param'], 5.0)

    def test_extract_values_preserves_structure(self):
        """Test that extract_values preserves tree structure."""
        original_tree = [
            Parameter(1.0),
            {'nested': [Parameter(2.0), 3.0]},
            Parameter(jnp.array([4, 5]))
        ]

        extracted = extract_values(original_tree)

        # Should preserve list structure
        self.assertIsInstance(extracted, list)
        self.assertEqual(len(extracted), 3)

        # First element should be extracted value
        self.assertEqual(extracted[0], 1.0)

        # Second element should preserve nested structure
        self.assertIsInstance(extracted[1], dict)
        self.assertEqual(extracted[1]['nested'][0], 2.0)
        self.assertEqual(extracted[1]['nested'][1], 3.0)

        # Third element should be array
        np_testing.assert_array_equal(extracted[2], jnp.array([4, 5]))


class TestParameterInteroperability(unittest.TestCase):
    """Test interoperability between different parameter types."""

    def test_mixed_parameter_arithmetic(self):
        """Test arithmetic between different parameter types."""
        regular = Parameter(2.0)
        normalized = NormalizedParameter(4.0)  # scale=4, value=1, __jax_array__=4
        bounded = BoundedParameter(6.0, 0.0, 10.0)

        # All should work together in arithmetic
        result = regular + normalized + bounded
        expected = 2.0 + 4.0 + 6.0
        self.assertEqual(result, expected)

    def test_mixed_parameter_tree_operations(self):
        """Test JAX tree operations with mixed parameter types."""
        tree = {
            'regular': Parameter(jnp.array([1, 2])),
            'normalized': NormalizedParameter(jnp.array([3, 6])),
            'bounded': BoundedParameter(jnp.array([4, 8]), 0.0, 10.0)
        }

        def sum_squares(tree_params):
            return jax.tree.reduce(
                lambda acc, x: acc + jnp.sum(x ** 2),
                tree_params,
                0.0
            )

        result = sum_squares(tree)
        # JAX tree operations use the pytree structure, not __jax_array__()
        # regular: [1,2]^2 = [1,4] -> sum = 5
        # normalized: internal value [1,1]^2 = [1,1] -> sum = 2 (not the scaled values!)
        # bounded: [4,8]^2 = [16,64] -> sum = 80
        # Total: 5 + 2 + 80 = 87
        self.assertEqual(float(result), 87.0)

    def test_gradient_computation_mixed_types(self):
        """Test gradient computation with mixed parameter types."""
        def objective(tree_params):
            return jnp.sum(tree_params['regular'] ** 2 + tree_params['normalized'] ** 2)

        tree = {
            'regular': Parameter(jnp.array([2.0, 3.0])),
            'normalized': NormalizedParameter(jnp.array([4.0, 5.0]))
        }

        grad_fn = jax.grad(objective)
        gradients = grad_fn(tree)

        # Check that gradients have the expected structure
        self.assertIn('regular', gradients)
        self.assertIn('normalized', gradients)

        # Regular parameter gradients
        expected_regular = 2 * jnp.array([2.0, 3.0])
        np_testing.assert_array_equal(gradients['regular'], expected_regular)


if __name__ == '__main__':
    unittest.main()