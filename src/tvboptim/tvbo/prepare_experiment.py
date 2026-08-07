"""TVBO experiment preparation with unified dispatch interface.

This module extends the prepare() multimethod with support for TVBO
SimulationExperiment objects, keeping TVBO an optional dependency.

TVBO is imported lazily. Importing it costs seconds, mostly in its ontology and
atlas modules, which is too much to pay on every ``import tvboptim`` for a
dispatch most workflows never reach. Instead a fallback ``prepare`` method
catches arguments no other method claims, imports TVBO only when the argument
actually came from it, registers the real method, and re-dispatches. A
SimulationExperiment can only exist if the caller imported TVBO already, so the
lazy path costs them nothing they had not already spent.
"""

import importlib.util
import sys
from typing import Any, Callable, Tuple

from plum import NotFoundLookupError, dispatch

# Import the prepare multimethod from network_dynamics to extend it
from tvboptim.experimental.network_dynamics.solve import prepare

__all__ = ["prepare", "HAS_TVBO"]

#: Whether TVBO is available. Annotated but deliberately left unassigned so that
#: module ``__getattr__`` resolves it on access instead of at import time.
HAS_TVBO: bool

#: Set to False to suppress the notice printed before the slow TVBO import.
ANNOUNCE_TVBO_IMPORT = True

# None until an import has been attempted, then the definitive result.
_tvbo_loaded: bool | None = None


def _tvbo_is_installed() -> bool:
    """Report whether TVBO can be found, without importing it."""
    if _tvbo_loaded is not None:
        return _tvbo_loaded
    try:
        return importlib.util.find_spec("tvbo") is not None
    except (ImportError, ValueError):
        return False


def _is_tvbo_object(value: Any) -> bool:
    """Cheap test for an object defined by TVBO, by module name only."""
    return type(value).__module__.partition(".")[0] == "tvbo"


def _load_tvbo() -> bool:
    """Import TVBO and register its ``prepare`` method. Cached after the first call."""
    global _tvbo_loaded
    if _tvbo_loaded is not None:
        return _tvbo_loaded
    if not _tvbo_is_installed():
        _tvbo_loaded = False
        return False

    # Only announce when this call is the one paying the cost.
    if ANNOUNCE_TVBO_IMPORT and "tvbo" not in sys.modules:
        print(
            "Importing TVB-O (first use, this takes a few seconds) ...",
            file=sys.stderr,
            flush=True,
        )
    try:
        _register_tvbo_prepare()
    except ImportError:
        _tvbo_loaded = False
        return False

    _tvbo_loaded = True
    return True


@dispatch
def prepare(experiment: Any, *args, **kwargs) -> Any:
    """Resolve a TVBO experiment, importing TVBO on first use.

    Registered for arguments no other ``prepare`` method claims. Only TVBO
    objects trigger the import, so an ordinary type error does not pay for it.
    """
    if _is_tvbo_object(experiment) and _tvbo_loaded is None and _load_tvbo():
        # The real method now exists; plum picks it up on re-dispatch.
        return prepare(experiment, *args, **kwargs)
    # Report as plum would have, had this fallback not claimed the call.
    target = tuple(type(value) for value in (experiment, *args))
    raise NotFoundLookupError("prepare", target, prepare.methods)


def _register_tvbo_prepare() -> None:
    """Import TVBO and add its ``prepare`` method to the multimethod."""
    import jax
    import jax.numpy as jnp
    from tvbo import SimulationExperiment

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
        experiment : tvbo.classes.SimulationExperiment
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
        >>> from tvbo import SimulationExperiment
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


def __getattr__(name: str) -> Any:
    """Resolve ``HAS_TVBO`` on access rather than at import time.

    Answers from :func:`importlib.util.find_spec` while TVBO is still unloaded,
    so merely asking whether TVBO is available stays cheap. Once an import has
    been attempted the cached, definitive result is returned instead.
    """
    if name == "HAS_TVBO":
        return _tvbo_is_installed()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
