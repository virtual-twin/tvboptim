"""Array-only message transport primitives.

The helpers in this module deliberately know nothing about dynamics state names,
history buffers, or graph classes.  Keeping that boundary small makes the
orientation and reduction rules reusable for future rectangular projections.
"""

import jax


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
