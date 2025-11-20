"""Data-based external inputs using interpolation.

This module provides external inputs from time-series data using Diffrax
interpolation for smooth, resolution-independent stimulus delivery.
"""

from typing import Literal

import diffrax
import jax.numpy as jnp

from ..core.bunch import Bunch
from .base import AbstractExternalInput


class DataInput(AbstractExternalInput):
    """External input from interpolated time-series data.

    Provides stimulus from discrete time-series data using interpolation,
    allowing simulation dt to differ from data sampling rate.

    Data shapes:
        - Global scalar: data=[n_times] → broadcasts to all nodes
        - Per-node: data=[n_times, n_nodes] → node-specific signals
        - Multi-dimensional: data=[n_times, n_dims, n_nodes] → multiple signals per node

    Interpolation types:
        - 'linear': Fast piecewise linear interpolation
        - 'cubic': Smooth cubic Hermite spline (backward differences)

    Args:
        times: Time points [n_times] (must be increasing)
        data: Data values at each time point (see shapes above)
        interpolation: Interpolation method ('linear' or 'cubic')

    Example:
        >>> # Global stimulus from data
        >>> times = jnp.linspace(0, 10, 100)
        >>> data = jnp.sin(times)
        >>> stimulus = DataInput(times, data, interpolation='cubic')
        >>>
        >>> # Per-node stimulus
        >>> n_nodes = 5
        >>> data = jnp.sin(times[:, None] + jnp.arange(n_nodes))  # [100, 5]
        >>> stimulus = DataInput(times, data)

    Attributes:
        N_OUTPUT_DIMS: Inferred from data shape (1 for scalar/per-node, n_dims for multi-dim)
        DEFAULT_PARAMS: Contains times, data, interpolation_type for reference
    """

    def __init__(
        self,
        times: jnp.ndarray,
        data: jnp.ndarray,
        interpolation: Literal["linear", "cubic"] = "linear",
        **kwargs,
    ):
        """Initialize DataInput with time-series data.

        Args:
            times: Time points [n_times]
            data: Data values (flexible shape)
            interpolation: 'linear' or 'cubic'
            **kwargs: Additional parameters (unused, for consistency)
        """
        # Validate inputs
        times = jnp.asarray(times)
        data = jnp.asarray(data)

        if times.ndim != 1:
            raise ValueError(f"times must be 1D array, got shape {times.shape}")

        if data.shape[0] != times.shape[0]:
            raise ValueError(
                f"First dimension of data {data.shape[0]} must match "
                f"times length {times.shape[0]}"
            )

        if interpolation not in ("linear", "cubic"):
            raise ValueError(
                f"interpolation must be 'linear' or 'cubic', got {interpolation}"
            )

        # Determine output dimensionality from data shape
        if data.ndim == 1:
            # [n_times] - global scalar
            n_output_dims = 1
        elif data.ndim == 2:
            # [n_times, n_nodes] - per-node scalar
            n_output_dims = 1
        elif data.ndim == 3:
            # [n_times, n_dims, n_nodes] - multi-dimensional
            n_output_dims = data.shape[1]
        else:
            raise ValueError(
                f"data must have 1, 2, or 3 dimensions, got shape {data.shape}"
            )

        self.N_OUTPUT_DIMS = n_output_dims

        # Store data as parameters (for reference, not optimization)
        self.DEFAULT_PARAMS = Bunch(
            times=times,
            data=data,
            interpolation_type=interpolation,
        )

        # Initialize base class
        super().__init__(**kwargs)

    def prepare(self, network, dt: float):
        """Prepare interpolation object for simulation.

        Creates Diffrax interpolation object based on interpolation type.

        Args:
            network: Network instance (used to get n_nodes)
            dt: Integration time step

        Returns:
            Tuple of (input_data, input_state)
            - input_data: Bunch with interpolator and metadata
            - input_state: Empty Bunch (stateless)
        """
        times = self.params.times
        data = self.params.data
        interp_type = self.params.interpolation_type

        # Create interpolation object
        if interp_type == "linear":
            interpolator = diffrax.LinearInterpolation(ts=times, ys=data)
        else:  # cubic
            # Compute cubic spline coefficients
            coeffs = diffrax.backward_hermite_coefficients(ts=times, ys=data)
            interpolator = diffrax.CubicInterpolation(ts=times, coeffs=coeffs)

        # Store interpolator and metadata
        input_data = Bunch(
            interpolator=interpolator,
            n_nodes=network.graph.n_nodes,
            data_shape=tuple(
                data.shape
            ),  # Original data shape for broadcasting logic (as tuple)
        )

        # Stateless - no state to track
        input_state = Bunch()

        return input_data, input_state

    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        input_data: Bunch,
        input_state: Bunch,
        params: Bunch,
    ) -> jnp.ndarray:
        """Compute interpolated input at time t.

        Args:
            t: Current time
            state: Network state [n_state_vars, n_nodes] (unused for stateless input)
            input_data: Bunch with interpolator and metadata
            input_state: Empty Bunch (stateless)
            params: Parameters with times, data, interpolation_type

        Returns:
            Input array [n_dims, n_nodes] with interpolated values
        """
        interpolator = input_data.interpolator
        n_nodes = input_data.n_nodes
        data_shape = input_data.data_shape

        # Evaluate interpolation at time t
        interpolated = interpolator.evaluate(t)

        # Handle broadcasting based on original data shape
        if len(data_shape) == 1:
            # [n_times] → scalar → broadcast to [1, n_nodes]
            return jnp.full((1, n_nodes), interpolated)
        elif len(data_shape) == 2:
            # [n_times, n_nodes] → [n_nodes] → [1, n_nodes]
            return interpolated[None, :]
        else:
            # [n_times, n_dims, n_nodes] → [n_dims, n_nodes]
            return interpolated

    def update_state(
        self, input_data: Bunch, input_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """Update input state (no-op for stateless input).

        Args:
            input_data: Static input data
            input_state: Current state (empty)
            new_state: New network state

        Returns:
            Unchanged input_state (empty Bunch)
        """
        return input_state
