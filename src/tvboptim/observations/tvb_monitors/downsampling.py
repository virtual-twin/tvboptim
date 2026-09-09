"""Downsampling strategies for neural activity timeseries.

This module provides different methods for reducing the temporal resolution
of simulation outputs, commonly used before BOLD signal computation.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.core.observation import (
    PreparedObservation,
    SimulationGrid,
    prepare_observation,
)
from tvboptim.experimental.network_dynamics.result import NativeSolution


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
    names = tuple(variable_names[index] for index in indices)
    return indices, names


def _integer_stride(period, dt, *, label="period"):
    """Validate a positive period and return its integer number of steps."""
    period = float(period)
    dt = float(dt)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError(f"{label} must be finite and positive; got {period!r}")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Simulation dt must be finite and positive; got {dt!r}")
    ratio = period / dt
    stride = round(ratio)
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"{label} {period:g} must be an integer multiple of simulation "
            f"dt={dt:g}; got {ratio:g} steps"
        )
    return int(stride)


def _validate_monitor_input(sample, variable_names):
    """Validate the common per-step monitor input description."""
    if len(sample.shape) != 2:
        raise ValueError(
            "Temporal observations require per-step input shaped "
            f"[channels, nodes]; got {sample.shape}"
        )
    if len(variable_names) != sample.shape[0]:
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
        ts, ys = sol.ts, sol.ys
        # Use sol.dt from auxiliary data and convert with Python int()
        # This keeps sample_step concrete during JIT compilation
        sample_step = _integer_stride(
            self.period, self._resolve_dt(sol), label="SubSampling period"
        )
        indices = _selection_indices(self.voi, ys.shape[1])
        # Select indices at regular intervals
        sample_indices = jnp.arange(sample_step - 1, ts.shape[0], sample_step)
        return NativeSolution(
            ts=ts[sample_indices] + t_offset,
            ys=ys[sample_indices][:, jnp.asarray(indices), ...],
            dt=self.period,
            variable_names=_slice_variable_names(sol, self.voi),
        )


def _subsampling_update(data, state, block, params):
    """Select completed cadence endpoints from one aligned raw block."""
    del params
    selected = block[:, data.indices, :]
    return state, selected[data.stride - 1 :: data.stride]


@prepare_observation.dispatch
def _prepare_subsampling(
    monitor: SubSampling,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    variable_names: tuple,
) -> PreparedObservation:
    """Prepare point sampling for an aligned native-solver grid."""
    _validate_monitor_input(sample, variable_names)
    stride = _integer_stride(monitor.period, grid.dt, label="SubSampling period")
    indices, names = _resolve_selection(monitor.voi, sample.shape[0], variable_names)
    return PreparedObservation(
        data=Bunch(indices=jnp.asarray(indices, dtype=int), stride=stride),
        state0=None,
        params=Bunch(),
        update=_subsampling_update,
        period=float(monitor.period),
        first_sample_offset=float(monitor.period),
        variable_names=names,
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

    def __call__(self, sol):
        """Downsample by averaging over temporal windows.

        Args:
            sol: Simulation solution with .ys, .ts, and .dt attributes
                 Works with NativeSolution (requires dt as auxiliary data)

        Returns:
            NativeSolution with temporally averaged timeseries
        """
        indices = _selection_indices(self.voi, sol.ys.shape[1])
        dt = self._resolve_dt(sol)
        samples_per_window = _integer_stride(
            self.period, dt, label="TemporalAverage period"
        )
        averaged_trace = _window_average(
            sol.ys[:, jnp.asarray(indices), :], samples_per_window
        )
        centered_times = _window_average(sol.ts, samples_per_window)

        return NativeSolution(
            ts=centered_times,
            ys=averaged_trace,
            dt=self.period,
            variable_names=_slice_variable_names(sol, self.voi),
        )


def _temporal_average_update(data, state, block, params):
    """Average complete windows from one aligned raw block."""
    del params
    selected = block[:, data.indices, :]
    return state, _window_average(selected, data.stride)


@prepare_observation.dispatch
def _prepare_temporal_average(
    monitor: TemporalAverage,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    variable_names: tuple,
) -> PreparedObservation:
    """Prepare complete-window averaging for an aligned native grid."""
    _validate_monitor_input(sample, variable_names)
    stride = _integer_stride(monitor.period, grid.dt, label="TemporalAverage period")
    indices, names = _resolve_selection(monitor.voi, sample.shape[0], variable_names)
    return PreparedObservation(
        data=Bunch(indices=jnp.asarray(indices, dtype=int), stride=stride),
        state0=None,
        params=Bunch(),
        update=_temporal_average_update,
        period=float(monitor.period),
        first_sample_offset=float((monitor.period + grid.dt) / 2.0),
        variable_names=names,
    )
