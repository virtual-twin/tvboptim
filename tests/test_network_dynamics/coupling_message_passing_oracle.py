"""Small NumPy/f64 oracle for coupling message-passing tests.

The implementation is deliberately scalar over graph edges. It encodes the
repository convention ``weights[target, source]`` without sharing vectorized
JAX expressions with the production implementation.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Pre = Callable[[FloatArray, FloatArray | None], ArrayLike]
Post = Callable[[FloatArray, FloatArray | None], ArrayLike]


def coupling_oracle(
    source_signal: ArrayLike,
    weights: ArrayLike,
    *,
    pre: Pre,
    post: Post,
    target_local: ArrayLike | None = None,
    history: ArrayLike | None = None,
    delay_steps: ArrayLike | None = None,
    edge_indices: ArrayLike | None = None,
) -> FloatArray:
    """Evaluate coupling in declared edge order using NumPy float64.

    Parameters follow the future transport seam rather than today's flat state:

    - ``source_signal`` is ``[Q_source, N_source]``;
    - ``target_local`` is optional ``[Q_local, N_target]``;
    - ``history`` is optional oldest-to-newest ``[T, Q_source, N_source]``;
    - ``delay_steps[target, source]`` is an integer history offset;
    - ``edge_indices`` is optional sparse ``[E, 2]`` in ``(target, source)``
      order. If omitted, every dense target/source pair is evaluated.

    ``pre`` receives one source and optional target vector at a time and returns
    one vector of output-channel messages. ``post`` receives the fully reduced
    ``[Q_output, N_target]`` array.
    """
    source = np.asarray(source_signal, dtype=np.float64)
    weight_matrix = np.asarray(weights, dtype=np.float64)
    local = None if target_local is None else np.asarray(target_local, dtype=np.float64)

    if source.ndim != 2:
        raise ValueError(f"source_signal must be rank 2, got {source.shape}")
    if weight_matrix.ndim != 2:
        raise ValueError(f"weights must be rank 2, got {weight_matrix.shape}")

    n_target, n_source = weight_matrix.shape
    if source.shape[1] != n_source:
        raise ValueError(
            f"source_signal has {source.shape[1]} nodes, expected {n_source}"
        )
    if local is not None and (local.ndim != 2 or local.shape[1] != n_target):
        raise ValueError(
            f"target_local must end in N_target={n_target}, got {local.shape}"
        )

    hist = None if history is None else np.asarray(history, dtype=np.float64)
    delays = None if delay_steps is None else np.asarray(delay_steps, dtype=np.int64)
    if (hist is None) != (delays is None):
        raise ValueError("history and delay_steps must be supplied together")
    if hist is not None:
        if hist.ndim != 3 or hist.shape[1:] != source.shape:
            raise ValueError(
                "history must have shape [T, Q_source, N_source], got "
                f"{hist.shape} for source {source.shape}"
            )
        if delays.shape != weight_matrix.shape:
            raise ValueError(
                f"delay_steps must match weights {weight_matrix.shape}, got "
                f"{delays.shape}"
            )
        if np.any(delays < 0) or np.any(delays >= hist.shape[0]):
            raise ValueError("delay_steps address samples outside history")

    if edge_indices is None:
        edges = np.array(
            [(target, src) for target in range(n_target) for src in range(n_source)],
            dtype=np.int64,
        )
    else:
        edges = np.asarray(edge_indices, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edge_indices must have shape [E, 2], got {edges.shape}")

    summed = None
    newest = None if hist is None else hist.shape[0] - 1
    for target, src in edges:
        if target < 0 or target >= n_target or src < 0 or src >= n_source:
            raise ValueError(f"edge ({target}, {src}) is outside weights shape")

        source_value = (
            source[:, src]
            if hist is None
            else hist[newest - delays[target, src], :, src]
        )
        target_value = None if local is None else local[:, target]
        message = np.atleast_1d(
            np.asarray(pre(source_value, target_value), dtype=np.float64)
        )

        if summed is None:
            summed = np.zeros((message.shape[0], n_target), dtype=np.float64)
        elif message.shape != (summed.shape[0],):
            raise ValueError(
                f"pre returned inconsistent shape {message.shape}; expected "
                f"{(summed.shape[0],)}"
            )

        summed[:, target] += message * weight_matrix[target, src]

    if summed is None:
        probe_source = np.zeros(source.shape[0], dtype=np.float64)
        probe_target = None if local is None else np.zeros(local.shape[0])
        n_output = np.atleast_1d(pre(probe_source, probe_target)).shape[0]
        summed = np.zeros((n_output, n_target), dtype=np.float64)

    result = np.asarray(post(summed, local), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != n_target:
        raise ValueError(f"post must return [Q_output, N_target], got {result.shape}")
    return result
