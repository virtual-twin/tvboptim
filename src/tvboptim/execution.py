from abc import ABC

import jax
import jax.numpy as jnp
from tqdm import tqdm

from tvboptim.types.stateutils import combine_state


class Execution(ABC):
    pass


class Result(ABC):
    """
    Result type to provide unified indexing
    """

    pass


class SequentialExecution(Execution):
    """
    Sequential execution of models across parameter spaces with progress tracking.

    SequentialExecution provides a straightforward approach to executing a model
    function across all parameter combinations in a given state space. Unlike
    ParallelExecution, it processes parameter combinations one at a time, making
    it ideal for debugging, memory-constrained environments, or when parallel
    execution is not feasible or desired.

    Parameters
    ----------
    model : callable
        Model function to execute. Should accept a state parameter and return
        simulation results. Signature: model(state, *args, **kwargs).
    statespace : AbstractSpace
        Parameter space (DataSpace, UniformSpace, or GridSpace) defining the
        parameter combinations to execute across.
    *args : tuple
        Positional arguments passed directly to the model function.
    **kwargs : dict
        Keyword arguments passed directly to the model function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tvboptim.types.spaces import GridSpace
    >>> from tvboptim.types.parameter import Parameter
    >>>
    >>> # Define a simple model
    >>> def simulate(state, noise_level=0.0):
    ...     result = state['param1'] * state['param2'] + noise_level
    ...     return {'output': result, 'inputs': state}
    >>>
    >>> # Create parameter space
    >>> state = {
    ...     'param1': Parameter("param1", 0.0, low=0.0, high=1.0, free=True),
    ...     'param2': Parameter("param2", 0.0, low=-1.0, high=1.0, free=True)
    ... }
    >>> space = GridSpace(state, n=5)  # 25 parameter combinations
    >>>
    >>> # Set up sequential execution
    >>> executor = SequentialExecution(
    ...     model=simulate,
    ...     statespace=space,
    ...     noise_level=0.1  # Keyword argument passed to model
    ... )
    >>>
    >>> # Execute across all parameter combinations
    >>> results = executor.run()  # Shows progress bar
    >>>
    >>> # Access results
    >>> first_result = results[0]
    >>> all_results = list(results)  # Convert to list
    >>> total_count = len(results)

    """

    def __init__(self, model, statespace, collect=True, *args, **kwargs):
        self.model = model
        self.statespace = statespace
        self.args = args
        self.kwargs = kwargs
        # if collect:
        # self.model = lambda state, *args, **kwargs: model(collect_parameters(state), *args, **kwargs)

    def run(self):
        """
        Execute the model across all parameter combinations in parallel.
        """
        results = []
        for state in tqdm(self.statespace):
            results.append(
                jax.block_until_ready(self.model(state, *self.args, **self.kwargs))
            )
            # results.append(jax.block_until_ready(self.model(state, *self.args, **self.kwargs)))
        return SequentialResult(results)


class SequentialResult(Result):
    def __init__(self, results):
        self.results = results

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index):
        return self.results[index]

    def __len__(self):
        return len(self.results)


class ParallelExecution(Execution):
    """
    Efficient parallel execution of models across parameter spaces using JAX.

    ParallelExecution orchestrates the parallel computation of a model function
    across all parameter combinations in a given state space. It leverages JAX's
    pmap (for multi-device parallelism) and vmap (for vectorization).

    Parameters
    ----------
    model : callable
        Model function to execute. Should accept a state parameter and return
        simulation results. Signature: model(state, *args, **kwargs).
    space : AbstractSpace
        Parameter space (DataSpace, UniformSpace, or GridSpace) defining the
        parameter combinations to execute across.
    *args : tuple
        Positional arguments passed directly to the model function.
    n_vmap : int, optional
        Number of states to vectorize over using jax.vmap. Controls batch size
        for vectorized execution within each device. Default is 1.
    n_pmap : int, optional
        Number of devices to parallelize over using jax.pmap. Should typically
        match the number of available devices. Default is 1.
    **kwargs : dict
        Keyword arguments passed directly to the model function.


    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from tvboptim.types.spaces import GridSpace
    >>> from tvboptim.types.parameter import Parameter
    >>>
    >>> # Define a simple model
    >>> def simulate(state):
    ...     return state['param1'] * state['param2']
    >>>
    >>> # Create parameter space
    >>> state = {
    ...     'param1': Parameter("param1", 0.0, low=0.0, high=1.0, free=True),
    ...     'param2': Parameter("param2", 0.0, low=-1.0, high=1.0, free=True)
    ... }
    >>> space = GridSpace(state, n=10)  # 100 parameter combinations
    >>>
    >>> # Set up parallel execution
    >>> n_devices = jax.device_count()
    >>> executor = ParallelExecution(
    ...     model=simulate,
    ...     statespace=space,
    ...     n_vmap=5,           # Vectorize over 5 states per device
    ...     n_pmap=n_devices    # Use all available devices
    ... )
    >>>
    >>> # Execute across all parameter combinations
    >>> results = executor.run()
    >>>
    >>> # Access individual results
    >>> first_result = results[0]
    >>> all_results = list(results)  # Convert to list
    >>> subset_results = results[10:20]  # Slice notation

    Notes
    -----
    For optimal performance:

    - Set n_pmap to match the number of available devices - on CPU use the pmap trick to force XLA to use N devices:
      os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={N}'
    - Tune n_vmap based on memory constraints and model complexity

    The execution uses jax.block_until_ready() to ensure all computation completes
    before returning results, providing accurate timing measurements.
    """

    def __init__(self, model, space, *args, n_vmap=1, n_pmap=1, **kwargs):
        self.model = model
        self.space = space
        self.n_vmap = n_vmap
        self.n_pmap = n_pmap
        self.args = args
        self.kwargs = kwargs
        diff_state, static_state = space.collect(n_pmap=n_pmap, combine=False)

        self.diff_state = diff_state

        # kwargs args probably go directly into model
        def _model(d):
            state = combine_state(d, static_state)
            return model(state, *self.args, **self.kwargs)

        def map_model(d):
            return jax.lax.map(_model, d, batch_size=n_vmap)

        self.map_model = map_model

    def run(self):
        """
        Execute the model across all parameter combinations in parallel.
        """
        res = jax.block_until_ready(
            jax.pmap(self.map_model, in_axes=0)(self.diff_state)
        )
        return ParallelResult(res, self.space.N, self.n_vmap, self.n_pmap)


class ParallelResult(Result):
    def __init__(self, results, N, n_vmap, n_pmap):
        self.results = results
        self.N = N
        self.n_vmap = n_vmap
        self.n_pmap = n_pmap

        # Calculate second dimension size
        if n_pmap is None:
            self.second_dim_size = max(1, int(jnp.ceil(N / n_vmap)))
        else:
            self.second_dim_size = max(1, int(jnp.ceil(N / n_pmap)))

    def __len__(self):
        return self.N

    def __iter__(self):
        for i in range(self.N):
            yield self[i]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.N))]

        if index < 0:
            index += self.N
        if index < 0 or index >= self.N:
            raise IndexError(f"Index {index} out of range for length {self.N}")

        if self.n_pmap is None:
            # No first dimension: results shape is (second_dim_size, ...)
            # Second dimension contains n_vmap * n_map elements
            vmap_idx = index // (self.second_dim_size // self.n_vmap)
            map_idx = index % (self.second_dim_size // self.n_vmap)
            second_dim_idx = vmap_idx * (self.second_dim_size // self.n_vmap) + map_idx

            def extract_at_indices(leaf):
                return leaf[second_dim_idx]

        else:
            # First dimension is n_pmap: results shape is (n_pmap, second_dim_size, ...)
            pmap_idx = index // self.second_dim_size
            second_dim_idx = index % self.second_dim_size

            def extract_at_indices(leaf):
                return leaf[pmap_idx, second_dim_idx]

        return jax.tree.map(extract_at_indices, self.results)
