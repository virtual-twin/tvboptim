"""Typed readouts and shared heterogeneous readout preparation helpers."""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Readout:
    """Wrap a readout callable with an explicit input-space declaration.

    Args:
        fn: ``fn(values, params) -> [channels, nodes]`` callable.
        name: Optional display name used in validation errors.
        space: ``"state"`` for route readouts or ``"recorded"`` for
            observation readouts.

    Parameters remain in the position-specific route or observation mappings.
    Shared parameter ownership is an additive follow-up to this wrapper.
    """

    fn: Callable
    name: str | None = None
    space: str = "state"

    def __post_init__(self):
        if not callable(self.fn):
            raise TypeError("Readout.fn must be callable")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError("Readout.name must be a non-empty string or None")
        if self.space not in {"state", "recorded"}:
            raise ValueError("Readout.space must be 'state' or 'recorded'")

    def __call__(self, values, params):
        return self.fn(values, params)

    @property
    def display_name(self):
        return self.name or getattr(self.fn, "__name__", self.fn.__class__.__name__)


def prepare_readouts(
    readouts,
    params,
    *,
    probe_values,
    names,
    group_nodes,
    role,
    params_name,
    space,
):
    """Probe and resolve group readouts against one explicit value namespace."""
    prepared = []
    widths = set()
    dtypes = []
    for group_name in sorted(readouts):
        readout = readouts[group_name]
        values = probe_values[group_name]
        nodes = jnp.asarray(group_nodes[group_name], dtype=int)
        if callable(readout):
            if isinstance(readout, Readout) and readout.space != space:
                raise ValueError(
                    f"{role} readout {readout.display_name!r} for group "
                    f"{group_name!r} declares space={readout.space!r}, but this "
                    f"position reads {space!r} values; use space={space!r}."
                )
            try:
                shaped = jax.eval_shape(readout, values, params[group_name])
            except Exception as exc:
                hint = (
                    f" Its parameters {params_name}[{group_name!r}] are empty; "
                    f"if the readout reads parameters, pass them there."
                    if not params[group_name]
                    else ""
                )
                raise ValueError(
                    f"{role} readout for group {group_name!r} could not be "
                    f"evaluated as readout({space}, params) "
                    f"({type(exc).__name__}: {exc})." + hint
                ) from exc
            if not hasattr(shaped, "shape"):
                raise ValueError(
                    f"{role} readout for group {group_name!r} must return one array"
                )
            expected_nodes = len(group_nodes[group_name])
            if len(shaped.shape) != 2 or shaped.shape[1] != expected_nodes:
                raise ValueError(
                    f"{role} readout for group {group_name!r} returned shape "
                    f"{shaped.shape}; expected [Q, {expected_nodes}]"
                )
            width = shaped.shape[0]
            dtype = shaped.dtype
            prepared.append((group_name, nodes, readout, None))
        else:
            namespace = tuple(names[group_name])
            missing = [name for name in readout if name not in namespace]
            if missing:
                raise ValueError(
                    f"{role} readout for group {group_name!r} names unknown "
                    f"{space} variables {missing}; available {list(namespace)}"
                )
            indices = jnp.asarray(
                [namespace.index(name) for name in readout], dtype=jnp.int32
            )
            width = len(readout)
            dtype = values.dtype
            prepared.append((group_name, nodes, None, indices))
        widths.add(width)
        dtypes.append(dtype)
    if len(widths) != 1:
        raise ValueError(
            f"{role} readouts must share one channel width, got {sorted(widths)}"
        )
    return tuple(prepared), widths.pop(), tuple(dtypes)


def pack_readouts(
    grouped_values, specs, width, dtype, params, node_count, fill_value=0.0
):
    """Pack group-local readouts into one graph-order signal array."""
    signal = jnp.full((width, node_count), fill_value, dtype=dtype)
    for group_name, nodes, readout, indices in specs:
        values = (
            grouped_values[group_name][indices]
            if readout is None
            else readout(grouped_values[group_name], params[group_name])
        )
        signal = signal.at[:, nodes].set(values)
    return signal


def pack_history_readouts(history, specs, width, dtype, params, node_count):
    """Pack time-stacked group state into graph-order readout history."""
    n_times = history.ts.shape[0]
    signal = jnp.zeros((n_times, width, node_count), dtype=dtype)
    for group_name, nodes, readout, indices in specs:
        states = history.groups[group_name]
        values = (
            states[:, indices, :]
            if readout is None
            else jax.vmap(readout, in_axes=(0, None))(states, params[group_name])
        )
        signal = signal.at[:, :, nodes].set(values)
    return signal
