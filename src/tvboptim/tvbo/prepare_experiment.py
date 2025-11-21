"""TVBO experiment preparation with unified dispatch interface.

This module extends the prepare() multimethod with support for TVBO SimulationExperiment
objects. It uses conditional dispatch registration to make TVBO an optional dependency.
"""

from typing import Any, Callable, Tuple

from plum import dispatch

# Import the prepare multimethod from network_dynamics to extend it
from tvboptim.experimental.network_dynamics.solve import prepare

__all__ = ["prepare", "HAS_TVBO"]

# ============================================================================
# OPTIONAL TVBO DISPATCH
# ============================================================================

try:
    import jax
    import jax.numpy as jnp
    from tvbo.export.experiment import SimulationExperiment

    @dispatch
    def prepare(
        experiment: SimulationExperiment,
        t0: float = 0.0,
        t1: float = 100.0,
        dt: float = 0.1,
        enable_x64: bool = True,
        replace_temporal_averaging: bool = False,
        return_new_ics: bool = False,
        scalar_pre: bool = False,
        bold_fft_convolve: bool = True,
        small_dt: bool = False,
        **kwargs,
    ) -> Tuple[Callable, Any]:
        """Convert TVBO SimulationExperiment to JAX-compatible model function and state.

        This function transforms a TVBO simulation experiment into a JAX-compiled
        model function and corresponding state object for efficient brain simulation.
        The resulting model supports automatic differentiation and parallel execution.

        Parameters
        ----------
        experiment : tvbo.export.experiment.SimulationExperiment
            TVBO SimulationExperiment containing model, connectivity, coupling,
            integration, and monitor specifications.
        t0 : float, optional
            Start time for simulation. Default is 0.0.
            Note: Currently not used, reserved for future integration.
        t1 : float, optional
            End time for simulation. Default is 100.0.
            Note: Currently not used, reserved for future integration.
        dt : float, optional
            Time step for simulation. Default is 0.1.
            Note: Currently not used, reserved for future integration.
        enable_x64 : bool, optional
            If True, use float64 precision; otherwise float32. Transforms all arrays
            in state to correct precision and sets JAX config 'jax_enable_x64'.
            Default is True.
        replace_temporal_averaging : bool, optional
            If False, BOLD uses TemporalAverage monitor as TVB does. If True,
            uses faster SubSample monitor with similar results. Default is False.
        return_new_ics : bool, optional
            If True, model returns updated initial conditions TimeSeries along
            with simulation output for continuing simulations. Changes output
            from result to [result, initial_conditions]. Default is False.
        scalar_pre : bool, optional
            If True, applies performance optimization replacing dot product with
            matmul in coupling term. Only works with scalar-only pre expressions,
            no delays, and when pre expression has single x_j occurrence.
            Default is False.
        bold_fft_convolve : bool, optional
            If True, BOLD monitor uses FFT convolution instead of dot product.
            Faster for most cases, time doesn't scale with BOLD period. Dot
            product can be faster for large period values. Default is True.
        small_dt : bool, optional
            Uses full history storage for faster simulations at small dt. Can
            cause memory explosion under jax.grad transformation. Default is False.
        **kwargs : dict
            Additional keyword arguments passed to experiment.execute().

        Returns
        -------
        tuple[Callable, Any]
            A tuple containing (model_function, state) where:

            - model_function : Callable that takes state and returns simulation results
            - state : JAX PyTree containing all simulation parameters and initial conditions

        Examples
        --------
        >>> from tvbo.export.experiment import SimulationExperiment
        >>> from tvboptim import prepare
        >>>
        >>> # Create TVBO experiment
        >>> experiment = SimulationExperiment(...)
        >>>
        >>> # Convert to JAX
        >>> model, state = prepare(experiment, enable_x64=True, scalar_pre=True)
        >>>
        >>> # Run simulation
        >>> result = model(state)
        >>> raw_data, bold_data = result
        >>>
        >>> # Use with JAX transformations
        >>> grad_fn = jax.grad(lambda s: model(s)[0].data.sum())
        >>> gradients = grad_fn(state)

        Notes
        -----
        The returned model function is JAX-compiled and supports:

        - Automatic differentiation with jax.grad, jax.jacobian
        - Parallel execution with jax.vmap, jax.pmap
        - Just-in-time compilation for optimal performance
        - Integration with JAX ecosystem (optax, equinox, etc.)

        The state object is a JAX PyTree that can be used with all JAX transformations
        and contains Parameter objects for optimization workflows.
        """
        state = experiment.collect_state()
        jax.config.update("jax_enable_x64", enable_x64)

        if enable_x64:
            state = state.convert_dtype(target_dtype=jnp.float64)
        else:
            state = state.convert_dtype(target_dtype=jnp.float32)

        _module = experiment.execute(
            format="jax",
            replace_temporal_averaging=replace_temporal_averaging,
            return_new_ics=return_new_ics,
            scalar_pre=scalar_pre,
            bold_fft_convolve=bold_fft_convolve,
            small_dt=small_dt,
            **kwargs,
        )
        simulator = _module["kernel"]

        return simulator, state

    HAS_TVBO = True

except ImportError:
    HAS_TVBO = False
