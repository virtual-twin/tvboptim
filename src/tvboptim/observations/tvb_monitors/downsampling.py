"""Downsampling strategies for neural activity timeseries.

This module provides different methods for reducing the temporal resolution
of simulation outputs, commonly used before BOLD signal computation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.core.observation import (
    PreparedObservation,
    SimulationGrid,
    _stateless,
    apply_observation,
    sampling_stride,
)


def _selection_indices(voi, n_variables):
    """Resolve a NumPy/JAX-style variable selection to concrete indices."""
    try:
        selected = np.arange(n_variables)[voi]
    except (IndexError, TypeError) as exc:
        raise ValueError(
            f"Invalid variable selection {voi!r} for {n_variables} input channels"
        ) from exc
    indices = np.asarray(selected)
    if indices.ndim == 0:
        indices = indices.reshape(1)
    indices = tuple(int(index) for index in indices.tolist())
    if not indices:
        raise ValueError("Temporal observation variable selection must not be empty")
    return indices


def _resolve_selection(voi, n_variables, variable_names):
    """Resolve selected indices and their corresponding channel names."""
    indices = _selection_indices(voi, n_variables)
    names = (
        None
        if variable_names is None
        else tuple(variable_names[index] for index in indices)
    )
    return indices, names


_integer_stride = sampling_stride


def _validate_monitor_input(sample, variable_names):
    """Validate the common per-step monitor input description."""
    if len(sample.shape) != 2:
        raise ValueError(
            "Temporal observations require per-step input shaped "
            f"[channels, nodes]; got {sample.shape}"
        )
    if variable_names is not None and len(variable_names) != sample.shape[0]:
        raise ValueError(
            "variable_names must describe the monitor input channel axis; "
            f"got {len(variable_names)} names for {sample.shape[0]} channels"
        )


def _window_average(values, samples_per_window):
    """Average complete, disjoint windows along the leading axis."""
    n_windows = values.shape[0] // samples_per_window
    trimmed = values[: n_windows * samples_per_window]
    windows = trimmed.reshape((n_windows, samples_per_window) + values.shape[1:])
    return jnp.mean(windows, axis=1)


def _slice_variable_names(sol, voi):
    """Apply voi slicing to a solution's variable_names, if available."""
    names = getattr(sol, "variable_names", None)
    if names is None:
        return None
    try:
        indices = _selection_indices(voi, len(names))
        return tuple(names[index] for index in indices)
    except (TypeError, ValueError):
        return None


class AbstractMonitor(eqx.Module):
    """Base class for monitoring and downsampling strategies.

    Provides common functionality for handling variable of interest (voi)
    parameter normalization to ensure dimensions are preserved.

    Attributes:
        voi: Variable of interest (state variable index) to extract.
             Normalized to preserve dimensions.
        period: Sampling/averaging period in milliseconds
    """

    voi: object = eqx.field(static=True)
    period: float = eqx.field(static=True)

    @staticmethod
    def _resolve_dt(sol) -> float:
        """Return dt from solution, raising a clear error if not set.

        Args:
            sol: Solution object with .dt attribute

        Returns:
            Concrete Python float for use in int/round patterns

        Raises:
            ValueError: If sol.dt is None (e.g. diffrax solution with non-ts saveat)
        """
        if sol.dt is not None:
            return sol.dt
        raise ValueError(
            "Solution has dt=None. Monitors require a concrete dt to compute sampling "
            "intervals. When using a DiffraxSolver, pass a ts-based saveat: "
            "SaveAt(ts=jnp.arange(t0, t1, dt)) so the effective save step can be "
            "inferred at prepare time."
        )

    @staticmethod
    def _normalize_voi(voi):
        """Normalize voi to preserve dimensions.

        Converts integers and single-index slices to dimension-preserving slices.

        Args:
            voi: Variable of interest specification. Can be:
                - None: uses all variables (jnp.s_[:])
                - int: single index (converted to slice)
                - slice: slice object (normalized if single-index)
                - other: passed through as-is

        Returns:
            Normalized voi that preserves dimensions
        """
        if voi is None:
            return jnp.s_[:]
        elif isinstance(voi, int):
            return jnp.s_[voi : voi + 1]  # Preserves dimension
        elif isinstance(voi, slice):
            # Check if it's a single-index slice like jnp.s_[0]
            # These have start, stop, step where stop is None and step is None
            if voi.stop is None and voi.step is None and isinstance(voi.start, int):
                # Convert to dimension-preserving slice
                return jnp.s_[voi.start : voi.start + 1]
            else:
                return voi
        else:
            # Assume it's already in the correct format
            return voi


class SubSampling(AbstractMonitor):
    """Downsample timeseries by selecting every nth sample.

    This is a simple point-wise downsampling strategy that picks samples
    at regular intervals without any averaging.

    Attributes:
        voi: Variable of interest (state variable index) to extract.
             If None, uses all variables.
        period: Sampling period in milliseconds (default: 4.0)
    """

    period: float = eqx.field(static=True)

    def __init__(self, voi=None, period=4.0):
        """Initialize SubSampling.

        Args:
            voi: Variable of interest index. If None, extracts all state variables.
                 If integer, the dimension is preserved using slice notation.
            period: Sampling period in milliseconds (default: 4.0)
        """
        self.voi = self._normalize_voi(voi)
        self.period = period

    def __call__(self, sol, t_offset=0.0):
        """Downsample the solution by selecting every nth sample.

        Args:
            sol: Simulation solution with .ys, .ts, and .dt attributes
                 Works with NativeSolution (requires dt as auxiliary data)
            t_offset: Time offset to add to output timestamps (default: 0.0)

        Returns:
            NativeSolution with downsampled timeseries
        """
        return apply_observation(self, sol, t_offset=t_offset)

    def prepare(self, grid, sample, variable_names):
        """Prepare point sampling for block-wise execution."""
        return _prepare_subsampling(self, grid, sample, variable_names)


def _subsampling_update(indices, stride):
    """Build endpoint selection for one aligned raw block."""

    def update(state, block, params):
        del params
        return state, block[:, indices, :][stride - 1 :: stride]

    return update


def _prepare_subsampling(
    monitor,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    variable_names: tuple | None,
) -> PreparedObservation:
    """Prepare point sampling for an aligned native-solver grid."""
    _validate_monitor_input(sample, variable_names)
    stride = _integer_stride(monitor.period, grid.dt, label="SubSampling period")
    indices, names = _resolve_selection(monitor.voi, sample.shape[0], variable_names)
    return PreparedObservation(
        params=Bunch(),
        init=_stateless,
        update=_subsampling_update(jnp.asarray(indices, dtype=int), stride),
        output=grid.output(monitor.period, variable_names=names),
    )


class TemporalAverage(AbstractMonitor):
    """Downsample timeseries by averaging over temporal windows.

    This downsampling strategy computes the mean over non-overlapping
    temporal windows, providing smoother output than simple subsampling.
    The output timestamps are centered within each averaging window.

    Attributes:
        voi: Variable of interest (state variable index) to extract.
             If None, uses all variables.
        period: Averaging window size in milliseconds (default: 4.0)
    """

    # period: float = 4.0
    period: float = eqx.field(static=True)

    def __init__(self, voi=None, period=4.0):
        """Initialize TemporalAverage.

        Args:
            voi: Variable of interest index. If None, extracts all state variables.
                 If integer, the dimension is preserved using slice notation.
            period: Averaging window size in milliseconds (default: 4.0)
        """
        self.voi = self._normalize_voi(voi)
        self.period = period

    def __call__(self, sol, t_offset=0.0):
        """Downsample by averaging over temporal windows.

        Args:
            sol: Simulation solution with .ys, .ts, and .dt attributes
                 Works with NativeSolution (requires dt as auxiliary data)

        Returns:
            NativeSolution with temporally averaged timeseries
        """
        return apply_observation(self, sol, t_offset=t_offset)

    def prepare(self, grid, sample, variable_names):
        """Prepare complete-window averaging for block-wise execution."""
        return _prepare_temporal_average(self, grid, sample, variable_names)


def _temporal_average_update(indices, stride):
    """Build complete-window averaging for one aligned raw block."""

    def update(state, block, params):
        del params
        return state, _window_average(block[:, indices, :], stride)

    return update


def _prepare_temporal_average(
    monitor,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    variable_names: tuple | None,
) -> PreparedObservation:
    """Prepare complete-window averaging for an aligned native grid."""
    _validate_monitor_input(sample, variable_names)
    stride = _integer_stride(monitor.period, grid.dt, label="TemporalAverage period")
    indices, names = _resolve_selection(monitor.voi, sample.shape[0], variable_names)
    return PreparedObservation(
        params=Bunch(),
        init=_stateless,
        update=_temporal_average_update(jnp.asarray(indices, dtype=int), stride),
        output=grid.output(monitor.period, label="window_center", variable_names=names),
    )
