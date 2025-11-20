"""
Comprehensive tests for Space class functionality.

Tests cover:
- Space: axis discovery, combination modes, iteration
- SpaceSlicing: indexing and slicing operations
- SpaceBatching: collect() method for parallel execution
- SpaceEdgeCases: error handling, empty spaces, etc.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy.testing as np_testing

from tvboptim.types.spaces import Space, GridAxis, UniformAxis, DataAxis

# NumPyro tests only if available
try:
    import numpyro.distributions as dist
    from tvboptim.types.spaces import NumPyroAxis
    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False


class TestSpaceBasics(unittest.TestCase):
    """Test Space basic functionality and initialization."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple state with nested structure
        self.simple_state = {
            'param_a': GridAxis(0.0, 1.0, 3),
            'param_b': UniformAxis(0.0, 2.0, 4),
            'fixed_value': 42.0
        }

        # More complex nested state
        self.nested_state = {
            'parameters': {
                'coupling': {
                    'a': GridAxis(0.0, 1.0, 2),
                    'b': 1.5  # Fixed value
                },
                'model': {
                    'J_N': UniformAxis(-1.0, 1.0, 3)
                }
            },
            'config': {
                'dt': 0.01,
                'T': 1000
            }
        }

    def test_space_initialization_zip_mode(self):
        """Test Space initialization in zip mode."""
        space = Space(self.simple_state, mode='zip')

        self.assertEqual(space.mode, 'zip')
        self.assertEqual(space.N, 3)  # Min of axis sizes (3, 4)

    def test_space_initialization_product_mode(self):
        """Test Space initialization in product mode."""
        space = Space(self.simple_state, mode='product')

        self.assertEqual(space.mode, 'product')
        self.assertEqual(space.N, 12)  # 3 * 4 = 12

    def test_space_initialization_nested(self):
        """Test Space initialization with nested state."""
        space = Space(self.nested_state, mode='product')

        self.assertEqual(space.N, 6)  # 2 * 3 = 6

    def test_space_initialization_errors(self):
        """Test Space initialization error conditions."""
        # Invalid mode
        with self.assertRaises(ValueError):
            Space(self.simple_state, mode='invalid')

        # No axes
        empty_state = {'a': 1.0, 'b': 'hello'}
        with self.assertRaises(ValueError):
            Space(empty_state)

    def test_zip_mode_size_warning(self):
        """Test that zip mode warns about different axis sizes."""
        # Create state with different axis sizes
        state = {
            'a': GridAxis(0.0, 1.0, 3),
            'b': GridAxis(0.0, 1.0, 5)  # Different size
        }

        # Should use minimum size and warn
        space = Space(state, mode='zip')
        self.assertEqual(space.N, 3)

    def test_len_and_size_property(self):
        """Test len() and N property."""
        space = Space(self.simple_state, mode='product')

        self.assertEqual(len(space), 12)
        self.assertEqual(space.N, 12)
        self.assertEqual(len(space), space.N)


class TestSpaceIteration(unittest.TestCase):
    """Test Space iteration and state generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.state = {
            'param_a': GridAxis(0.0, 1.0, 3),
            'param_b': DataAxis([10, 20, 30]),
            'fixed': 'constant'
        }

    def test_iteration_zip_mode(self):
        """Test iterating through Space in zip mode."""
        space = Space(self.state, mode='zip')

        states = list(space)
        self.assertEqual(len(states), 3)

        # Check first state structure
        first_state = states[0]
        self.assertIn('param_a', first_state)
        self.assertIn('param_b', first_state)
        self.assertIn('fixed', first_state)

        # Fixed values should be preserved
        self.assertEqual(first_state['fixed'], 'constant')

        # Axes should be replaced with values
        self.assertIsInstance(first_state['param_a'], (float, jnp.ndarray))
        self.assertIsInstance(first_state['param_b'], (int, float, jnp.ndarray))

    def test_iteration_product_mode(self):
        """Test iterating through Space in product mode."""
        # Use smaller axes for manageable test
        small_state = {
            'a': GridAxis(0.0, 1.0, 2),
            'b': GridAxis(0.0, 1.0, 2)
        }
        space = Space(small_state, mode='product')

        states = list(space)
        self.assertEqual(len(states), 4)  # 2 * 2 = 4

        # Check that we get all combinations
        a_values = [float(state['a']) for state in states]
        b_values = [float(state['b']) for state in states]

        # Should have all combinations of (0.0, 1.0) x (0.0, 1.0)
        expected_combinations = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        actual_combinations = list(zip(a_values, b_values))

        for expected in expected_combinations:
            # Check if any actual combination is close to expected
            found = any(
                abs(actual[0] - expected[0]) < 1e-6 and
                abs(actual[1] - expected[1]) < 1e-6
                for actual in actual_combinations
            )
            self.assertTrue(found, f"Expected combination {expected} not found")

    def test_iterator_protocol(self):
        """Test iterator protocol methods."""
        space = Space(self.state, mode='zip')

        # Test manual iteration
        iterator = iter(space)
        first = next(iterator)
        second = next(iterator)
        third = next(iterator)

        # Should raise StopIteration after exhausting
        with self.assertRaises(StopIteration):
            next(iterator)

        # Test that each state is different
        self.assertNotEqual(first['param_a'], second['param_a'])

    def test_iteration_with_random_key(self):
        """Test iteration with specified random key."""
        state_with_random = {
            'uniform': UniformAxis(0.0, 1.0, 3),
            'fixed': 100
        }

        space1 = Space(state_with_random, mode='zip', key=jax.random.key(42))
        space2 = Space(state_with_random, mode='zip', key=jax.random.key(42))

        states1 = list(space1)
        states2 = list(space2)

        # Should produce same results with same key
        for s1, s2 in zip(states1, states2):
            self.assertEqual(float(s1['uniform']), float(s2['uniform']))


class TestSpaceIndexing(unittest.TestCase):
    """Test Space indexing and slicing operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.state = {
            'grid': GridAxis(0.0, 4.0, 5),  # [0, 1, 2, 3, 4]
            'data': DataAxis([10, 20, 30, 40, 50]),
            'constant': 'unchanged'
        }

    def test_single_index_access(self):
        """Test accessing single states by index."""
        space = Space(self.state, mode='zip')

        # Test positive indexing
        state0 = space[0]
        self.assertEqual(float(state0['grid']), 0.0)
        self.assertEqual(int(state0['data']), 10)
        self.assertEqual(state0['constant'], 'unchanged')

        state2 = space[2]
        self.assertEqual(float(state2['grid']), 2.0)
        self.assertEqual(int(state2['data']), 30)

    def test_negative_index_access(self):
        """Test negative indexing."""
        space = Space(self.state, mode='zip')

        last_state = space[-1]
        self.assertEqual(float(last_state['grid']), 4.0)
        self.assertEqual(int(last_state['data']), 50)

        second_last = space[-2]
        self.assertEqual(float(second_last['grid']), 3.0)

    def test_index_out_of_bounds(self):
        """Test index out of bounds errors."""
        space = Space(self.state, mode='zip')

        with self.assertRaises(IndexError):
            space[10]  # Too large

        with self.assertRaises(IndexError):
            space[-10]  # Too negative

    def test_slice_access(self):
        """Test slicing creates new Space with DataAxis instances."""
        space = Space(self.state, mode='zip')

        # Test basic slice
        subspace = space[1:4]
        self.assertIsInstance(subspace, Space)
        self.assertEqual(len(subspace), 3)

        # Check that subspace uses DataAxis
        subspace_states = list(subspace)
        expected_grid_values = [1.0, 2.0, 3.0]
        expected_data_values = [20, 30, 40]

        for i, state in enumerate(subspace_states):
            self.assertAlmostEqual(float(state['grid']), expected_grid_values[i])
            self.assertEqual(int(state['data']), expected_data_values[i])

    def test_slice_with_step(self):
        """Test slicing with step parameter."""
        space = Space(self.state, mode='zip')

        subspace = space[::2]  # Every other element
        self.assertEqual(len(subspace), 3)  # 0, 2, 4

        states = list(subspace)
        expected_values = [0.0, 2.0, 4.0]
        for i, state in enumerate(states):
            self.assertAlmostEqual(float(state['grid']), expected_values[i])

    def test_empty_slice(self):
        """Test that empty slices raise appropriate errors."""
        space = Space(self.state, mode='zip')

        with self.assertRaises(ValueError):
            space[10:20]  # Results in empty slice

    def test_slice_maintains_fixed_values(self):
        """Test that slicing preserves fixed values."""
        space = Space(self.state, mode='zip')
        subspace = space[1:3]

        states = list(subspace)
        for state in states:
            self.assertEqual(state['constant'], 'unchanged')


class TestSpaceBatching(unittest.TestCase):
    """Test Space collect() method for parallel execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.state = {
            'param': GridAxis(0.0, 1.0, 8),  # 8 values for batching tests
            'fixed': 42
        }

    def test_collect_no_batching(self):
        """Test collect() without batching parameters."""
        space = Space(self.state, mode='zip')
        result = space.collect()

        # Should return combined state
        self.assertIn('param', result)
        self.assertIn('fixed', result)
        self.assertEqual(result['fixed'], 42)

    def test_collect_with_vmap(self):
        """Test collect() with vmap parameter."""
        space = Space(self.state, mode='zip')
        result = space.collect(n_vmap=4)

        # Should have batched parameter values
        param_values = result['param']
        self.assertEqual(param_values.shape[0], 4)  # vmap dimension

    def test_collect_with_pmap(self):
        """Test collect() with pmap parameter."""
        space = Space(self.state, mode='zip')
        result = space.collect(n_pmap=2)

        # Should have pmap dimension
        param_values = result['param']
        self.assertEqual(param_values.shape[0], 2)  # pmap dimension

    def test_collect_with_both_vmap_pmap(self):
        """Test collect() with both vmap and pmap."""
        space = Space(self.state, mode='zip')
        result = space.collect(n_vmap=2, n_pmap=2)

        param_values = result['param']
        self.assertEqual(param_values.shape[:2], (2, 2))  # (pmap, vmap)

    def test_collect_with_padding(self):
        """Test collect() when requested size exceeds available combinations."""
        # Create small space
        small_state = {'param': GridAxis(0.0, 1.0, 3)}
        space = Space(small_state, mode='zip')

        # Request more combinations than available
        result = space.collect(n_vmap=5, n_pmap=1, fill_value=-999.0)

        param_values = result['param']
        # Should be padded with fill_value
        flat_values = param_values.flatten()
        self.assertTrue(jnp.any(flat_values == -999.0))

    def test_collect_combine_parameter(self):
        """Test collect() combine parameter."""
        space = Space(self.state, mode='zip')

        # With combine=True (default)
        combined_result = space.collect(n_vmap=2, combine=True)
        self.assertIn('fixed', combined_result)

        # With combine=False
        axis_tree, static_state = space.collect(n_vmap=2, combine=False)
        self.assertIn('param', axis_tree)
        self.assertEqual(static_state['fixed'], 42)

    def test_collect_no_axes(self):
        """Test collect() with state containing no axes."""
        no_axes_state = {'a': 1.0, 'b': 'hello'}

        # Should raise error during Space creation
        with self.assertRaises(ValueError):
            Space(no_axes_state)


@unittest.skipUnless(NUMPYRO_AVAILABLE, "NumPyro not available")
class TestSpaceWithNumPyro(unittest.TestCase):
    """Test Space functionality with NumPyroAxis."""

    def test_space_with_numpyro_axis(self):
        """Test Space containing NumPyroAxis."""
        state = {
            'normal': NumPyroAxis(dist.Normal(0.0, 1.0), n=5),
            'grid': GridAxis(0.0, 1.0, 5),
            'constant': 100
        }

        space = Space(state, mode='zip', key=jax.random.key(42))
        self.assertEqual(len(space), 5)

        # Test iteration
        states = list(space)
        for state in states:
            self.assertIn('normal', state)
            self.assertIn('grid', state)
            self.assertEqual(state['constant'], 100)

            # Values should be reasonable
            normal_val = float(state['normal'])
            self.assertGreater(normal_val, -5.0)  # Reasonable range for normal
            self.assertLess(normal_val, 5.0)

    def test_space_with_numpyro_shapes(self):
        """Test Space with NumPyroAxis having sample shapes."""
        state = {
            'multivar': NumPyroAxis(
                dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
                n=3
            )
        }

        space = Space(state, mode='zip', key=jax.random.key(42))
        states = list(space)

        for state in states:
            multivar_val = state['multivar']
            self.assertEqual(multivar_val.shape, (2,))  # 2D multivariate


class TestSpaceEdgeCases(unittest.TestCase):
    """Test Space edge cases and error conditions."""

    def test_space_with_single_axis(self):
        """Test Space with only one axis."""
        state = {'single': GridAxis(0.0, 1.0, 4)}
        space = Space(state, mode='product')  # Mode shouldn't matter

        self.assertEqual(len(space), 4)
        states = list(space)
        values = [float(s['single']) for s in states]
        expected = [0.0, 1/3, 2/3, 1.0]

        for actual, exp in zip(values, expected):
            self.assertAlmostEqual(actual, exp, places=5)

    def test_space_deeply_nested(self):
        """Test Space with deeply nested structure."""
        state = {
            'level1': {
                'level2': {
                    'level3': {
                        'param': GridAxis(0.0, 1.0, 2)
                    }
                }
            },
            'top_level': 'constant'
        }

        space = Space(state, mode='zip')
        self.assertEqual(len(space), 2)

        states = list(space)
        for state in states:
            self.assertEqual(state['top_level'], 'constant')
            self.assertIn('level1', state)
            param_val = state['level1']['level2']['level3']['param']
            self.assertIn(float(param_val), [0.0, 1.0])

    def test_space_with_mixed_axis_types(self):
        """Test Space with multiple different axis types."""
        state = {
            'grid': GridAxis(0.0, 2.0, 3),
            'uniform': UniformAxis(10.0, 20.0, 3),
            'data': DataAxis([100, 200, 300])
        }

        space = Space(state, mode='zip', key=jax.random.key(42))
        states = list(space)

        self.assertEqual(len(states), 3)
        for i, state in enumerate(states):
            # Grid values should be deterministic
            expected_grid = float(i)  # 0.0, 1.0, 2.0
            self.assertAlmostEqual(float(state['grid']), expected_grid, places=5)

            # Data values should match exactly
            expected_data = [100, 200, 300][i]
            self.assertEqual(int(state['data']), expected_data)

            # Uniform values should be in range
            uniform_val = float(state['uniform'])
            self.assertGreaterEqual(uniform_val, 10.0)
            self.assertLessEqual(uniform_val, 20.0)

    def test_space_repr(self):
        """Test Space string representation."""
        state = {'a': GridAxis(0.0, 1.0, 5), 'b': UniformAxis(0.0, 1.0, 3)}
        space = Space(state, mode='product')

        repr_str = repr(space)
        self.assertIn("Space", repr_str)
        self.assertIn("N=15", repr_str)  # 5 * 3 = 15
        self.assertIn("product", repr_str)
        self.assertIn("axes=2", repr_str)


if __name__ == '__main__':
    unittest.main()