"""Clean axis and space system for TVBOptim.

This module implements a simple, composable parameter space system for systematic
exploration of brain simulation parameters.

The system consists of:

- **AbstractAxis**: Base class for sampling strategies (grid, uniform, data)
- **Space**: Composes axes into parameter combinations with product/zip modes
- **Centralized key management**: Single random key for all stochastic operations
- **JAX-native integration**: Full support for vmap/pmap parallel execution

Axes generate raw JAX arrays only. Parameter creation is handled separately by user code.

Examples
--------
>>> from tvboptim.types.spaces import Space, GridAxis, UniformAxis, NumPyroAxis
>>> import numpyro.distributions as dist
>>> import copy
>>>
>>> # Create exploration state with axes
>>> exploration_state = copy.deepcopy(state)
>>> exploration_state.parameters.coupling.a = GridAxis(0.0, 1.0, 10)
>>> exploration_state.parameters.model.J_N = UniformAxis(0.0, 1.0, 5)
>>> exploration_state.parameters.noise.sigma = NumPyroAxis(dist.LogNormal(0.0, 1.0), 8)
>>>
>>> # Create space for parameter exploration
>>> space = Space(exploration_state, mode='product')
>>> print(f"Total combinations: {space.N}")
>>>
>>> # Access individual combinations
>>> first_combination = space[0]
>>> subset = space[10:20]
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.utils import safe_reshape

try:
    import numpyro.distributions  # noqa: F401

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False

# =============================================================================
# Abstract Base Classes
# =============================================================================


class AbstractAxis(ABC):
    """Abstract base class for all parameter sampling axes.

    An axis defines how to sample values along one dimension of parameter space.
    Axes generate raw JAX arrays only - no parameter creation is performed.

    Notes
    -----
    All concrete axis implementations must provide:

    - `generate_values()` method to produce sample values
    - `size` property returning the number of values generated

    Examples
    --------
    >>> from tvboptim.types.spaces import GridAxis, UniformAxis, NumPyroAxis
    >>> import numpyro.distributions as dist
    >>>
    >>> # Grid sampling from 0 to 1 with 5 points
    >>> grid_axis = GridAxis(0.0, 1.0, 5)
    >>> values = grid_axis.generate_values()
    >>> print(f"Grid values: {values}")
    >>>
    >>> # Random uniform sampling
    >>> uniform_axis = UniformAxis(0.0, 1.0, 3)
    >>> values = uniform_axis.generate_values(jax.random.key(42))
    >>> print(f"Random values: {values}")
    >>>
    >>> # NumPyro distribution sampling
    >>> normal_axis = NumPyroAxis(dist.Normal(0.0, 1.0), 4)
    >>> values = normal_axis.generate_values(jax.random.key(42))
    >>> print(f"Normal samples: {values}")
    """

    def __init__(self):
        """Initialize axis."""
        pass

    @abstractmethod
    def generate_values(self, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Generate all values for this axis.

        Parameters
        ----------
        key : jax.random.PRNGKey, optional
            Random key for stochastic sampling. Ignored by deterministic axes.

        Returns
        -------
        jnp.ndarray
            Generated values as raw JAX arrays.
        """
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of values this axis generates.

        Returns
        -------
        int
            Total number of sample points along this axis.
        """
        pass


# =============================================================================
# Concrete Axis Implementations
# =============================================================================


class GridAxis(AbstractAxis):
    """Axis for systematic grid sampling over parameter bounds.

    Generates linearly spaced values between low and high bounds using
    deterministic grid sampling.

    Parameters
    ----------
    low : float
        Lower bound for sampling.
    high : float
        Upper bound for sampling.
    n : int
        Number of grid points to generate.

    Raises
    ------
    ValueError
        If n <= 0 or low >= high.

    Examples
    --------
    >>> grid = GridAxis(0.0, 1.0, 5)
    >>> values = grid.generate_values()
    >>> print(values)  # [0.0, 0.25, 0.5, 0.75, 1.0]
    """

    def __init__(
        self, low: float, high: float, n: int, shape: Optional[Tuple[int, ...]] = None
    ):
        """Initialize grid axis.

        Parameters
        ----------
        low : float
            Lower bound for sampling.
        high : float
            Upper bound for sampling.
        n : int
            Number of grid points to generate.
        shape : Tuple[int, ...], optional
            Shape of generated values. If specified, values are broadcast to this shape.
            Default is None.
        """
        super().__init__()
        self.low = low
        self.high = high
        self.n = n
        self.shape = shape

        if n <= 0:
            raise ValueError(f"Grid size must be positive, got {n}")
        if low >= high:
            raise ValueError(f"Low bound ({low}) must be less than high bound ({high})")

    def generate_values(self, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Generate linearly spaced grid values.

        Parameters
        ----------
        key : jax.random.PRNGKey, optional
            Random key. Ignored for deterministic grid sampling.

        Returns
        -------
        jnp.ndarray
            Array of linearly spaced values from low to high. If shape is specified,
            values are broadcast to shape (n,) + shape with identical values across
            additional dimensions.
        """
        values = jnp.linspace(self.low, self.high, self.n)
        if self.shape:
            # Broadcast to (n,) + shape - all values in additional dims are identical
            values = values.reshape(self.n, *([1] * len(self.shape)))
            values = jnp.broadcast_to(values, (self.n,) + self.shape)
        return values

    @property
    def size(self) -> int:
        """Number of grid points.

        Returns
        -------
        int
            Number of grid points (n).
        """
        return self.n

    def __repr__(self):
        return f"GridAxis(low={self.low}, high={self.high}, n={self.n})"


class UniformAxis(AbstractAxis):
    """Axis for uniform random sampling over parameter bounds.

    Generates random values uniformly distributed between low and high bounds
    using stochastic sampling.

    Parameters
    ----------
    low : float
        Lower bound for sampling.
    high : float
        Upper bound for sampling.
    n : int
        Number of random samples to generate.

    Raises
    ------
    ValueError
        If n <= 0 or low >= high.

    Examples
    --------
    >>> import jax
    >>> uniform = UniformAxis(0.0, 1.0, 3)
    >>> values = uniform.generate_values(jax.random.key(42))
    >>> print(values)  # Random values between 0 and 1
    """

    def __init__(
        self, low: float, high: float, n: int, shape: Optional[Tuple[int, ...]] = None
    ):
        """Initialize uniform axis.

        Parameters
        ----------
        low : float
            Lower bound for sampling.
        high : float
            Upper bound for sampling.
        n : int
            Number of random samples to generate.
        shape : Tuple[int, ...], optional
            Shape of generated values. If specified, values are broadcast to this shape.
            Default is None.
        """
        super().__init__()
        self.low = low
        self.high = high
        self.n = n
        self.shape = shape

        if n <= 0:
            raise ValueError(f"Sample size must be positive, got {n}")
        if low >= high:
            raise ValueError(f"Low bound ({low}) must be less than high bound ({high})")

    def generate_values(self, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Generate uniformly distributed random values.

        Parameters
        ----------
        key : jax.random.PRNGKey, optional
            Random key for sampling. If None, uses default key(0).

        Returns
        -------
        jnp.ndarray
            Array of uniformly distributed random values. If shape is specified,
            values are broadcast to shape (n,) + shape with identical values across
            additional dimensions.
        """
        if key is None:
            key = jax.random.key(0)

        values = jax.random.uniform(key, (self.n,), minval=self.low, maxval=self.high)
        if self.shape:
            # Broadcast to (n,) + shape - all values in additional dims are identical
            values = values.reshape(self.n, *([1] * len(self.shape)))
            values = jnp.broadcast_to(values, (self.n,) + self.shape)
        return values

    @property
    def size(self) -> int:
        """Number of random samples.

        Returns
        -------
        int
            Number of samples (n).
        """
        return self.n

    def __repr__(self):
        return f"UniformAxis(low={self.low}, high={self.high}, n={self.n})"


class DataAxis(AbstractAxis):
    """Axis for sampling from predefined data values.

    Uses a fixed set of values provided by the user. This axis type is
    deterministic and always returns the same predefined values. The first dimension of the data will always be used as axis dimension.

    Parameters
    ----------
    values : array-like
        Predefined values to sample from.

    Raises
    ------
    ValueError
        If values array is empty.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> data = DataAxis([1.0, 2.5, 3.7, 4.2])
    >>> values = data.generate_values()
    >>> print(values)  # [1.0, 2.5, 3.7, 4.2]
    >>>
    >>> # Can also use JAX arrays
    >>> data_jax = DataAxis(jnp.linspace(0, 1, 5))
    >>> print(data_jax.size)  # 5
    """

    def __init__(self, values: Union[List, jnp.ndarray]):
        """Initialize data axis.

        Parameters
        ----------
        values : array-like
            Predefined values to sample from.
        """
        super().__init__()
        self.values = jnp.asarray(values)

        if self.values.size == 0:
            raise ValueError("Data axis requires at least one value")

    def generate_values(self, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Return the predefined data values.

        Parameters
        ----------
        key : jax.random.PRNGKey, optional
            Random key. Ignored for deterministic data sampling.

        Returns
        -------
        jnp.ndarray
            The predefined data values.
        """
        return self.values

    @property
    def size(self) -> int:
        """Number of predefined values.

        Returns
        -------
        int
            Length of the values array.
        """
        return len(self.values)

    def __repr__(self):
        return f"DataAxis(values={self.values})"


class NumPyroAxis(AbstractAxis):
    """Axis for sampling from NumPyro probability distributions.

    Samples values from any NumPyro distribution with support for both
    independent sampling (each element different) and broadcast sampling
    (same sample broadcast to shape).

    Parameters
    ----------
    distribution : numpyro.distributions.Distribution
        NumPyro distribution to sample from.
    n : int
        Number of samples along the axis dimension.
    sample_shape : Tuple[int, ...], optional
        Additional shape dimensions for each sample. Default is None.
    broadcast_mode : bool, optional
        Sampling strategy:
        - False (default): Sample independently for each element in sample_shape
        - True: Sample once per axis point, then broadcast to sample_shape

    Raises
    ------
    ImportError
        If NumPyro is not installed.
    ValueError
        If n <= 0 or distribution is not a valid NumPyro distribution.
    TypeError
        If distribution is not a NumPyro distribution instance.

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>>
    >>> # Independent sampling: each element different
    >>> normal_axis = NumPyroAxis(dist.Normal(0.0, 1.0), n=5,
    ...                          sample_shape=(3, 4), broadcast_mode=False)
    >>> values = normal_axis.generate_values(jax.random.key(42))
    >>> print(values.shape)  # (5, 3, 4) with 60 different samples
    >>>
    >>> # Broadcast sampling: same sample per axis point
    >>> beta_axis = NumPyroAxis(dist.Beta(2.0, 5.0), n=3,
    ...                        sample_shape=(2, 2), broadcast_mode=True)
    >>> values = beta_axis.generate_values(jax.random.key(42))
    >>> print(values.shape)  # (3, 2, 2) with 3 unique samples broadcast
    >>>
    >>> # Simple 1D sampling
    >>> uniform_axis = NumPyroAxis(dist.Uniform(0.0, 1.0), n=10)
    >>> values = uniform_axis.generate_values(jax.random.key(42))
    >>> print(values.shape)  # (10,)
    """

    def __init__(
        self,
        distribution,
        n: int,
        sample_shape: Optional[Tuple[int, ...]] = None,
        broadcast_mode: bool = False,
    ):
        """Initialize NumPyro distribution axis.

        Parameters
        ----------
        distribution : numpyro.distributions.Distribution
            NumPyro distribution to sample from.
        n : int
            Number of samples along the axis dimension.
        sample_shape : Tuple[int, ...], optional
            Additional shape dimensions for each sample. Default is None.
        broadcast_mode : bool, optional
            If True, sample once per axis point then broadcast to sample_shape.
            If False, sample independently for each element. Default is False.
        """
        super().__init__()

        if not NUMPYRO_AVAILABLE:
            raise ImportError(
                "NumPyro is required for NumPyroAxis. Install with: pip install numpyro"
            )

        # Check if distribution is a NumPyro distribution
        if not hasattr(distribution, "sample") or not hasattr(
            distribution, "batch_shape"
        ):
            raise TypeError("distribution must be a NumPyro distribution instance")

        if n <= 0:
            raise ValueError(f"Sample size must be positive, got {n}")

        self.distribution = distribution
        self.n = n
        self.sample_shape = sample_shape if sample_shape is not None else ()
        self.broadcast_mode = broadcast_mode

    def generate_values(self, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Generate values by sampling from the NumPyro distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey, optional
            Random key for sampling. If None, uses default key(0).

        Returns
        -------
        jnp.ndarray
            Sampled values with shape (n,) + sample_shape. Sampling strategy
            depends on broadcast_mode:
            - broadcast_mode=False: Each element independently sampled
            - broadcast_mode=True: Each axis sample broadcast to sample_shape
        """
        if key is None:
            key = jax.random.key(0)

        if self.broadcast_mode and self.sample_shape:
            # Sample once per axis point, then broadcast
            samples = self.distribution.sample(key, sample_shape=(self.n,))

            # Add singleton dimensions and broadcast
            for _ in range(len(self.sample_shape)):
                samples = samples[..., None]
            samples = jnp.broadcast_to(samples, (self.n,) + self.sample_shape)

        else:
            # Sample independently for all elements
            full_sample_shape = (self.n,) + self.sample_shape
            samples = self.distribution.sample(key, sample_shape=full_sample_shape)

        return samples

    @property
    def size(self) -> int:
        """Number of samples along the axis dimension.

        Returns
        -------
        int
            Number of axis samples (n).
        """
        return self.n

    def __repr__(self):
        return (
            f"NumPyroAxis(distribution={self.distribution}, n={self.n}, "
            f"sample_shape={self.sample_shape}, broadcast_mode={self.broadcast_mode})"
        )


# =============================================================================
# Space Class for Composing Axes
# =============================================================================


class Space:
    """Composable parameter space built from multiple axes.

    Space discovers AbstractAxis instances in a parameter state tree and
    composes them to create parameter combinations. Supports both product
    (Cartesian product) and zip (parallel) combination modes.

    The Space class provides efficient parameter exploration by:

    - **Automatic axis discovery**: Finds AbstractAxis instances in state trees
    - **Flexible combination modes**: Product mode for full grid search, zip mode for parallel sampling
    - **Efficient iteration**: Pre-generates combinations for fast access
    - **Slicing support**: Create subspaces with `space[start:end]` syntax

    Parameters
    ----------
    state : dict
        State tree containing AbstractAxis instances and fixed values.
    mode : str, optional
        Combination mode: 'product' for Cartesian product, 'zip' for parallel.
        Default is 'zip'.
    key : jax.random.PRNGKey, optional
        Random key for stochastic axes. Default creates new key.

    Raises
    ------
    ValueError
        If mode is not 'product' or 'zip', or if no AbstractAxis instances found.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import copy
    >>> from tvboptim.types.spaces import Space, GridAxis, UniformAxis
    >>>
    >>> # Create exploration state
    >>> exploration_state = copy.deepcopy(base_state)
    >>> exploration_state.parameters.coupling.a = GridAxis(0.0, 1.0, 5)
    >>> exploration_state.parameters.model.J_N = UniformAxis(0.0, 1.0, 3)
    >>>
    >>> # Product mode: 5 × 3 = 15 combinations
    >>> space = Space(exploration_state, mode='product')
    >>> print(f"Total combinations: {space.N}")
    >>>
    >>> # Access individual combinations
    >>> first_state = space[0]
    >>> print(first_state.parameters.coupling.a)
    >>>
    >>> # Create subspace
    >>> subset = space[2:8]
    >>> print(f"Subset size: {len(subset)}")
    >>>
    >>> # Parallel execution
    >>> batched_states = space.collect(n_vmap=4, n_pmap=2)
    """

    def __init__(
        self,
        state: Dict[str, Any],
        mode: str = "zip",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        """Initialize space from state tree.

        Parameters
        ----------
        state : dict
            State tree containing AbstractAxis instances and fixed values.
        mode : str, optional
            Combination mode: 'product' for Cartesian product, 'zip' for parallel.
            Default is 'zip'.
        key : jax.random.PRNGKey, optional
            Random key for stochastic axes. Default creates new key.
        """
        self.state = state
        self.mode = mode
        self.key = key if key is not None else jax.random.key(0)

        if mode not in ["product", "zip"]:
            raise ValueError(f"Mode must be 'product' or 'zip', got {mode}")

        # Use equinox.partition to separate axes from static values
        self.axis_state, self.static_state = eqx.partition(
            state, lambda x: isinstance(x, AbstractAxis)
        )
        axes = jax.tree.leaves(
            self.axis_state, is_leaf=lambda x: isinstance(x, AbstractAxis)
        )
        if not axes:
            # No axes found, raise error
            raise ValueError("Space must contain at least one AbstractAxis instance")

        # Calculate total combinations
        self._N = self._calculate_total_size()

    def _calculate_total_size(self) -> int:
        """Calculate total number of parameter combinations."""
        axes = jax.tree.leaves(
            self.axis_state, is_leaf=lambda x: isinstance(x, AbstractAxis)
        )

        if not axes:
            return 1

        axis_sizes = [axis.size for axis in axes]

        if self.mode == "product":
            return int(np.prod(axis_sizes))
        elif self.mode == "zip":
            min_size = min(axis_sizes)
            max_size = max(axis_sizes)

            # Warn if axes have different sizes in zip mode
            if min_size != max_size:
                lost_combinations = sum(size - min_size for size in axis_sizes)
                print(
                    f"WARNING: In zip mode, axes have different sizes {axis_sizes}. "
                    f"Using minimum size {min_size}, losing {lost_combinations} combinations."
                )

            return min_size

    def _generate_axis_values(self) -> Dict[str, Any]:
        """Generate values for all axes using jax.tree.map."""
        # Pre-generate sub-keys for all axes to avoid correlations
        axes = jax.tree.leaves(
            self.axis_state, is_leaf=lambda x: isinstance(x, AbstractAxis)
        )
        keys = jax.random.split(self.key, len(axes) + 1)[1:]  # Skip first key

        # Create iterator for keys
        key_iter = iter(keys)

        def generate_values(axis):
            if isinstance(axis, AbstractAxis):
                # Always pass a key, even if not needed (to avoid correlations)
                axis_key = next(key_iter)
                return axis.generate_values(axis_key)
            return axis  # Not an axis, return as-is

        return jax.tree.map(
            generate_values,
            self.axis_state,
            is_leaf=lambda x: isinstance(x, AbstractAxis),
        )

    @property
    def N(self) -> int:
        """Total number of parameter combinations."""
        return self._N

    def _generate_all_combinations(self) -> Tuple[List[jnp.ndarray], Any]:
        """
        Generate all parameter combinations as flattened arrays.

        Returns
        -------
        tuple
            (flattened_arrays, tree_def) where flattened_arrays contains
            all parameter combinations and tree_def allows reconstruction.
        """
        # Generate values for all axes
        axis_values_tree = self._generate_axis_values()

        # Flatten to get list of arrays
        axis_values_list, axis_tree_def = jax.tree.flatten(
            axis_values_tree, is_leaf=lambda x: isinstance(x, jnp.ndarray)
        )

        if not axis_values_list:
            return [], axis_tree_def

        if self.mode == "product":
            # Create meshgrid for Cartesian product and flatten each
            meshgrid_arrays = jnp.meshgrid(*axis_values_list, indexing="ij")
            flattened_arrays = [arr.flatten() for arr in meshgrid_arrays]
        elif self.mode == "zip":
            # In zip mode, arrays are already aligned
            flattened_arrays = axis_values_list

        return flattened_arrays, axis_tree_def

    def _get_combination_at_index(self, index: int) -> Dict[str, Any]:
        """Get a single combination by indexing into flattened arrays."""
        flattened_arrays, axis_tree_def = self._generate_all_combinations()

        if not flattened_arrays:
            return self.static_state

        # Index into each flattened array to get values for this combination
        indexed_values = [arr[index] for arr in flattened_arrays]

        # Reconstruct axis tree with indexed values
        axis_values_tree = jax.tree.unflatten(axis_tree_def, indexed_values)

        # Replace axes with their values using tree_map
        def replace_axis_with_value(axis, value):
            if isinstance(axis, AbstractAxis):
                return value
            return axis

        processed_axis_tree = jax.tree.map(
            replace_axis_with_value,
            self.axis_state,
            axis_values_tree,
            is_leaf=lambda x: isinstance(x, AbstractAxis),
        )

        return eqx.combine(processed_axis_tree, self.static_state)

    # Iterator protocol for sequential execution
    def __iter__(self):
        """Initialize iterator with pre-generated combinations."""
        # Generate all combinations once and store
        self.flattened_arrays, self.axis_tree_def = self._generate_all_combinations()
        self.i = 0
        return self

    def __next__(self):
        """Get next parameter state using pre-generated arrays."""
        if self.i < self.N:
            if not self.flattened_arrays:
                # No axes case
                state = self.static_state
            else:
                # Index directly into pre-generated arrays
                indexed_values = [arr[self.i] for arr in self.flattened_arrays]
                axis_values_tree = jax.tree.unflatten(
                    self.axis_tree_def, indexed_values
                )

                # Replace axes with values
                def replace_axis_with_value(axis, value):
                    if isinstance(axis, AbstractAxis):
                        return value
                    return axis

                processed_axis_tree = jax.tree.map(
                    replace_axis_with_value,
                    self.axis_state,
                    axis_values_tree,
                    is_leaf=lambda x: isinstance(x, AbstractAxis),
                )

                state = eqx.combine(processed_axis_tree, self.static_state)

            self.i += 1
            return state
        raise StopIteration

    def __getitem__(self, index: Union[int, slice]) -> Union[Dict[str, Any], "Space"]:
        """
        Get specific parameter combination(s).

        Parameters
        ----------
        index : int or slice
            Index or slice for combinations.

        Returns
        -------
        dict or Space
            Single state (int index) or new Space with DataAxis instances (slice).
        """
        if isinstance(index, int):
            # Single index: return state dict
            if index < 0:
                index += self.N
            if index < 0 or index >= self.N:
                raise IndexError(
                    f"Index {index} out of range for {self.N} combinations"
                )
            return self._get_combination_at_index(index)

        elif isinstance(index, slice):
            # Slice: return new Space with DataAxis instances
            flattened_arrays, axis_tree_def = self._generate_all_combinations()

            if not flattened_arrays:
                # No axes, return space with only static state
                return Space(self.static_state, mode="zip", key=self.key)

            # Slice each array according to the index
            sliced_arrays = [arr[index] for arr in flattened_arrays]

            # Check if slice results in empty arrays
            if any(arr.size == 0 for arr in sliced_arrays):
                raise ValueError(
                    f"Slice {index} results in empty parameter space with no combinations"
                )

            # Create new axis state with DataAxis instances
            data_axes = [DataAxis(values) for values in sliced_arrays]
            new_axis_state = jax.tree.unflatten(axis_tree_def, data_axes)

            # Create new Space in zip mode
            return Space(
                eqx.combine(new_axis_state, self.static_state), mode="zip", key=self.key
            )

        else:
            raise TypeError(f"Index must be int or slice, got {type(index)}")

    def __len__(self) -> int:
        """Number of parameter combinations."""
        return self.N

    def collect(
        self,
        n_vmap: int = None,
        n_pmap: int = None,
        fill_value: float = jnp.nan,
        combine: bool = True,
    ) -> Dict[str, Any]:
        """Generate batched states for parallel execution.

        Creates parameter combinations organized for efficient JAX parallel execution
        using vmap and pmap. This method is essential for high-performance parameter
        exploration on modern accelerators.

        Parameters
        ----------
        n_vmap : int, optional
            Number of states to vectorize over using vmap. If None, defaults to 1.
        n_pmap : int, optional
            Number of devices for parallel mapping with pmap. If None, defaults to 1.
        fill_value : float, optional
            Value used for padding when reshaping to target dimensions.
            Default is jnp.nan.
        combine : bool, optional
            Whether to combine the static state with the parameter combinations
            into a complete state. If False, returns (axis_tree, static_state) tuple.
            Default is True.

        Returns
        -------
        dict or tuple
            If combine=True: State tree with batched parameter combinations shaped
            for parallel execution with dimensions (n_pmap, n_vmap, n_map).
            If combine=False: Tuple of (batched_axis_tree, static_state).

        Warnings
        --------
        If the total requested size (n_pmap * n_vmap * n_map) exceeds the number
        of combinations N, padding with fill_value will be used.

        Examples
        --------
        >>> # Create space with 100 combinations
        >>> space = Space(exploration_state, mode='product')
        >>>
        >>> # Batch for parallel execution: 2 devices, 8 vectors each
        >>> batched_states = space.collect(n_vmap=8, n_pmap=2)
        >>> print(batched_states.shape)  # (2, 8, n_map)
        >>>
        >>> # Use with JAX transformations
        >>> results = jax.pmap(jax.vmap(simulation_fn))(batched_states)
        """
        # Calculate batch dimensions
        shape = tuple()
        if n_vmap is None:
            n_vmap = 1
        else:
            shape = (n_vmap,) + shape
        if n_pmap is None:
            n_pmap = 1
        else:
            shape = (n_pmap,) + shape
        n_map = int(jnp.ceil(self.N / (n_vmap * n_pmap)))
        shape = shape + (n_map,)

        # If n_pmap * n_vmap * n_map > N, warn that fill value is used
        if n_pmap * n_vmap * n_map > self.N:
            print(
                f"WARNING: Total requested size {n_pmap * n_vmap * n_map} exceeds N = {self.N}. Using fill value: {fill_value}"
            )

        # Generate all combinations as flattened arrays
        array_list, axis_tree_def = self._generate_all_combinations()

        if not array_list:
            # No axes, just return static state with proper batching
            return self.static_state

        # Reshape first dimension of each flattened array to batch dimensions
        batched_arrays = [
            safe_reshape(arr, shape + arr.shape[1:], fill_value) for arr in array_list
        ]

        # Reconstruct the axis tree with batched arrays
        batched_axis_tree = jax.tree.unflatten(axis_tree_def, batched_arrays)

        if combine:
            # Combine with static state
            return eqx.combine(batched_axis_tree, self.static_state)
        else:
            return batched_axis_tree, self.static_state

    def __repr__(self):
        axes = jax.tree.leaves(
            self.axis_state, is_leaf=lambda x: isinstance(x, AbstractAxis)
        )
        return f"Space(N={self.N}, mode='{self.mode}', axes={len(axes)})"
