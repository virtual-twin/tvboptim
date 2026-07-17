"""Array-only message transport primitives.

The helpers in this module deliberately know nothing about dynamics state names,
history buffers, or graph classes.  Keeping that boundary small makes the
orientation and reduction rules reusable for future rectangular projections.
"""

import jax
import jax.numpy as jnp


def _aggregate_nodes(messages, weights):
    """Reduce node messages through ``weights[target, source]``.

    Parameters
    ----------
    messages : array, shape [Q, N_source]
        One value per output channel and source node.
    weights : array, shape [N_target, N_source]
        Directed connection weights.
    """
    return messages @ weights.T


def _reduce_edges(messages, weights_e, target_e, n_target):
    """Reduce prepared-edge messages without constructing a sparse matrix.

    ``pre``-style channel-major messages cross the boundary as ``[Q, E]``;
    ``segment_sum`` consumes the transposed E-major representation and groups
    each weighted message by its target node.
    """
    weighted_e = messages.T * weights_e[:, None]
    return jax.ops.segment_sum(
        weighted_e,
        target_e,
        num_segments=n_target,
    ).T


def _gather_history(history, read_t, source):
    """Read transmitted signals without knowing graph or state semantics.

    ``history`` is channel-major within time, ``[T, Q, N_source]``. The
    matching ``read_t`` and ``source`` index arrays may describe either a
    dense target/source grid or an edge vector. Moving the channel axis first
    therefore returns ``[Q, N_target, N_source]`` or ``[Q, E]`` respectively.
    """
    return jnp.moveaxis(history[read_t, :, source], -1, 0)


def _interpolate_history(history, read_lo, read_hi, source, fraction):
    """Linearly blend two array-only history reads in the same layout."""
    lo = _gather_history(history, read_lo, source)
    hi = _gather_history(history, read_hi, source)
    return (1.0 - fraction[None, ...]) * lo + fraction[None, ...] * hi


def _roll_history(history, transmitted):
    """Advance a rolling history and append one transmitted signal sample."""
    return jnp.roll(history, -1, axis=0).at[-1].set(transmitted)


def _write_history(history, write_idx, transmitted):
    """Write one transmitted signal sample into an indexed history buffer."""
    return history.at[write_idx].set(transmitted)
