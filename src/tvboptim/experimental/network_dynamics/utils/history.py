"""Utilities for history buffer extraction and manipulation."""

import jax.numpy as jnp
from jax import vmap
from typing import Tuple, Callable, Optional


def interpolate_history(
    old_ts: jnp.ndarray, old_ys: jnp.ndarray, n_steps: int
) -> jnp.ndarray:
    """Interpolate history to match target dt.

    Args:
        old_ts: Original time points [n_time_old]
        old_ys: Original trajectory [n_time_old, ...] (any trailing dimensions)
        n_steps: Number of steps needed in output

    Returns:
        Interpolated history [n_steps, ...] (preserving trailing dimensions)
    """
    # Create new time grid
    new_ts = jnp.linspace(old_ts[0], old_ts[-1], n_steps)

    # Flatten all trailing dimensions for vectorized interpolation
    n_time_old = old_ys.shape[0]
    trailing_shape = old_ys.shape[1:]
    old_ys_flat = old_ys.reshape(n_time_old, -1)  # [n_time_old, prod(trailing_dims)]

    # Interpolate each flattened component
    def interp_1d(y_values):
        return jnp.interp(new_ts, old_ts, y_values)

    # Vectorize over all trailing dimensions
    new_ys_flat = vmap(interp_1d, in_axes=1, out_axes=1)(old_ys_flat)

    # Reshape back to original structure
    return new_ys_flat.reshape(n_steps, *trailing_shape)


def extract_history_window(
    hist_ts: jnp.ndarray,
    hist_ys: jnp.ndarray,
    max_delay: float,
    dt: float,
    transform_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> jnp.ndarray:
    """Extract history window with optional transformation.

    This is the core logic for extracting a delay buffer from stored simulation
    history. Handles interpolation when dt doesn't match, padding when history
    is too short, and optional per-timestep transformation.

    Args:
        hist_ts: History time points [n_time_hist]
        hist_ys: History trajectory [n_time_hist, ...] (e.g., [n_time, n_states, n_nodes])
        max_delay: Maximum delay time (determines window length)
        dt: Target integration timestep
        transform_fn: Optional function to apply to each timestep of extracted history.
            Signature: transform_fn(ys_at_t) -> transformed_ys_at_t
            Applied BEFORE interpolation to maintain accuracy.

    Returns:
        History buffer [n_steps, ...] where n_steps = ceil(max_delay / dt) + 1
        If transform_fn provided, trailing dimensions may differ from input.

    Example:
        # Without transformation (direct extraction)
        buffer = extract_history_window(ts, ys, max_delay=20.0, dt=1.0)
        # Returns [21, n_states, n_nodes]

        # With aggregation (e.g., for regional coupling)
        def aggregate(node_ys):
            return region_one_hot_normalized.T @ node_ys.T
        buffer = extract_history_window(ts, ys, max_delay=20.0, dt=1.0,
                                       transform_fn=aggregate)
        # Returns [21, n_states, n_regions]
    """
    # Calculate required number of steps
    n_steps_needed = int(jnp.rint(max_delay / dt)) + 1

    # Calculate history dt (assume uniform spacing)
    hist_dt = hist_ts[1] - hist_ts[0] if len(hist_ts) > 1 else dt

    # Check if we need interpolation (allow small numerical tolerance)
    needs_interpolation = jnp.abs(hist_dt - dt) > 1e-9

    # Calculate time coverage and check if we need padding
    time_coverage = hist_ts[-1] - hist_ts[0]
    needs_padding = time_coverage < max_delay

    # Apply transformation if provided (before interpolation for accuracy)
    if transform_fn is not None:
        # Transform each timestep: [n_time, ...] -> [n_time, ...transformed...]
        hist_ys_transformed = jnp.array([transform_fn(hist_ys[t]) for t in range(len(hist_ts))])
    else:
        hist_ys_transformed = hist_ys

    if needs_padding:
        # Pad at the beginning with the first timestep
        n_steps_available = len(hist_ts)
        n_steps_to_pad = n_steps_needed - n_steps_available

        if n_steps_to_pad > 0:
            # Repeat first timestep
            first_state = hist_ys_transformed[0:1]  # [1, ...]
            padding = jnp.tile(first_state, (n_steps_to_pad,) + (1,) * (hist_ys_transformed.ndim - 1))

            if needs_interpolation:
                # Interpolate available data, then pad
                interpolated = interpolate_history(
                    hist_ts, hist_ys_transformed, n_steps_needed - n_steps_to_pad
                )
                return jnp.concatenate([padding, interpolated], axis=0)
            else:
                # Just pad and concatenate
                return jnp.concatenate([padding, hist_ys_transformed], axis=0)
        else:
            # We have enough data but may need interpolation
            if needs_interpolation:
                return interpolate_history(hist_ts, hist_ys_transformed, n_steps_needed)
            else:
                return hist_ys_transformed[-n_steps_needed:]
    else:
        # Sufficient time coverage - extract last max_delay seconds
        if needs_interpolation:
            # Find the time window we need
            t_start = hist_ts[-1] - max_delay
            start_idx = jnp.searchsorted(hist_ts, t_start)

            # Extract and interpolate
            window_ts = hist_ts[start_idx:]
            window_ys = hist_ys_transformed[start_idx:]

            return interpolate_history(window_ts, window_ys, n_steps_needed)
        else:
            # No interpolation needed, just extract
            return hist_ys_transformed[-n_steps_needed:]
