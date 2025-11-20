"""Downsampling strategies for neural activity timeseries.

This module provides different methods for reducing the temporal resolution
of simulation outputs, commonly used before BOLD signal computation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics.result import NativeSolution


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
        sample_step = int(round(self.period / sol.dt))
        # Select indices at regular intervals
        sample_indices = jnp.arange(sample_step - 1, ts.shape[0], sample_step)
        return NativeSolution(
            ts=ts[sample_indices] + t_offset,
            ys=ys[sample_indices, self.voi, ...],
            dt=self.period,
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
        ts, ys = sol.ts, sol.ys

        # Apply voi slicing first
        ys_sliced = ys[:, self.voi, ...]

        # Number of samples per averaging window
        # Use sol.dt from auxiliary data and convert with Python int()
        samples_per_window = int(round(self.period / sol.dt))

        # Map time points to sample indices
        time_indices = (ts[::samples_per_window] / sol.dt).astype(int)

        def average_window(start_idx):
            """Compute average over a temporal window."""
            # Define slice starting point for all dimensions
            start_indices = (start_idx,) + (0,) * (ys_sliced.ndim - 1)
            # Define slice size based on the sliced array shape
            slice_sizes = (samples_per_window,) + ys_sliced.shape[1:]

            # Extract window and compute mean over time axis
            return jnp.mean(
                jax.lax.dynamic_slice(ys_sliced, start_indices, slice_sizes),
                axis=0,
            )

        # Vectorized averaging over all windows
        averaged_trace = jax.vmap(average_window)(time_indices)

        # Create time indices centered in each window
        # Offset by half window to center timestamps
        center_offset = (samples_per_window - 2) // 2
        centered_indices = jnp.arange(center_offset, ts.shape[0], samples_per_window)

        return NativeSolution(
            ts=ts[centered_indices],
            ys=averaged_trace[: centered_indices.shape[0], ...],
            dt=self.period,
        )
