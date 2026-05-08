"""Result types for  Network Dynamics native solvers.

This module provides Diffrax-like solution objects for native  Network Dynamics solvers,
ensuring consistent API across solver types.
"""

import warnings
from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from jax import tree_util


@tree_util.register_pytree_with_keys_class
class NativeSolution:
    """Solution object for Network Dynamics native solvers.

    Provides the same interface as Diffrax solutions (.ys, .ts) while being
    a proper JAX PyTree for compatibility with JAX transformations.

    Attributes:
        ts: Time points, shape [n_time]
        ys: Trajectory data, shape [n_time, n_variables, n_nodes]
        dt: Time step (optional), stored as static auxiliary data
        variable_names: Names of the variables along axis 1 of ys, in order.
            Contains state variables and/or auxiliary variables, depending on
            the dynamics' VARIABLES_OF_INTEREST. None if unknown.
    """

    def __init__(
        self,
        ts: jnp.ndarray,
        ys: jnp.ndarray,
        dt: float = None,
        variable_names: Optional[Tuple[str, ...]] = None,
    ):
        """Initialize native solution.

        Args:
            ts: Time points, shape [n_time]
            ys: Trajectory array, shape [n_time, n_variables, n_nodes]
            dt: Time step (optional), stored as static auxiliary data for JIT compatibility
            variable_names: Names of the variables along axis 1 of ys, in order.
        """
        self.ts = ts
        self.ys = ys
        self.dt = dt
        self.variable_names = (
            tuple(variable_names) if variable_names is not None else None
        )

    @property
    def time(self):
        return self.ts

    @property
    def data(self):
        return self.ys

    def tree_flatten(self):
        """JAX PyTree flatten for transformations."""
        children = (self.ts, self.ys)
        aux_data = {"dt": self.dt, "variable_names": self.variable_names}
        return children, aux_data

    def tree_flatten_with_keys(self):
        """JAX PyTree flatten with named keys (so paths render as ``.ts``/``.ys``)."""
        children = (
            (tree_util.GetAttrKey("ts"), self.ts),
            (tree_util.GetAttrKey("ys"), self.ys),
        )
        aux_data = {"dt": self.dt, "variable_names": self.variable_names}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """JAX PyTree unflatten for transformations."""
        ts, ys = children
        dt = aux_data.get("dt") if aux_data else None
        variable_names = aux_data.get("variable_names") if aux_data else None
        return cls(ts, ys, dt=dt, variable_names=variable_names)

    def __repr__(self):
        """String representation."""
        return (
            f"NativeSolution(shape={self.ys.shape}, "
            f"t=[{self.ts[0]:.2f}, {self.ts[-1]:.2f}], "
            f"variable_names={self.variable_names})"
        )

    def _resolve_variables(
        self,
        variables: Optional[Sequence[Union[str, int]]],
    ) -> Tuple[list, list]:
        """Normalize variable selection to (indices, labels)."""
        n_vars = self.ys.shape[1]
        if variables is None:
            idx = list(range(n_vars))
        else:
            idx = []
            for v in variables:
                if isinstance(v, str):
                    if self.variable_names is None:
                        raise ValueError(
                            "Cannot select variables by name: variable_names "
                            "is not set on this solution. Pass integer indices."
                        )
                    if v not in self.variable_names:
                        raise ValueError(
                            f"Variable {v!r} not in variable_names="
                            f"{self.variable_names}."
                        )
                    idx.append(self.variable_names.index(v))
                else:
                    vi = int(v)
                    if not -n_vars <= vi < n_vars:
                        raise IndexError(
                            f"Variable index {vi} out of bounds for n_variables={n_vars}."
                        )
                    idx.append(vi % n_vars)

        labels = [
            self.variable_names[i] if self.variable_names is not None else f"var[{i}]"
            for i in idx
        ]
        return idx, labels

    def _resolve_nodes(
        self,
        nodes: Optional[Union[int, Sequence[int]]],
        n_nodes: int,
        max_nodes: int,
    ) -> list:
        """Normalize node selection to a list of indices."""
        if nodes is None:
            if n_nodes > max_nodes:
                warnings.warn(
                    f"Solution has {n_nodes} nodes; plotting first {max_nodes}. "
                    f"Pass nodes=[...] or max_nodes=N to override.",
                    stacklevel=3,
                )
                return list(range(max_nodes))
            return list(range(n_nodes))
        if isinstance(nodes, int):
            return list(range(min(nodes, n_nodes)))
        idx = [int(n) for n in nodes]
        for ni in idx:
            if not -n_nodes <= ni < n_nodes:
                raise IndexError(
                    f"Node index {ni} out of bounds for n_nodes={n_nodes}."
                )
        return [ni % n_nodes for ni in idx]

    def _resolve_t_mask(
        self,
        t_range: Optional[Tuple[Optional[float], Optional[float]]],
        default_window: Optional[float],
    ) -> jnp.ndarray:
        """Build a boolean time mask with soft clipping."""
        ts = self.ts
        t_min, t_max = float(ts[0]), float(ts[-1])
        user_supplied = t_range is not None

        if t_range is None:
            t0 = t_min
            t1 = t_max if default_window is None else min(t_min + default_window, t_max)
        else:
            req_t0, req_t1 = t_range
            t0 = t_min if req_t0 is None else float(req_t0)
            t1 = t_max if req_t1 is None else float(req_t1)

        clipped_t0 = max(t0, t_min)
        clipped_t1 = min(t1, t_max)

        if clipped_t0 >= clipped_t1:
            raise ValueError(
                f"Requested t_range=({t0}, {t1}) has no overlap with "
                f"available range [{t_min}, {t_max}]."
            )

        if user_supplied and (clipped_t0 != t0 or clipped_t1 != t1):
            requested_span = t1 - t0
            actual_span = clipped_t1 - clipped_t0
            if actual_span < 0.9 * requested_span:
                warnings.warn(
                    f"Requested t_range=({t0}, {t1}) only partially overlaps "
                    f"available [{t_min}, {t_max}]; plotting "
                    f"[{clipped_t0}, {clipped_t1}].",
                    stacklevel=3,
                )

        return (ts >= clipped_t0) & (ts <= clipped_t1)

    @staticmethod
    def _prepare_axes(ax, n_vars, figsize, dpi):
        """Set up or validate axes for n_vars subplots."""
        import matplotlib.pyplot as plt

        if ax is None:
            if figsize is None:
                figsize = (10, max(2.0, 2.0 * n_vars))
            fig, axes = plt.subplots(
                n_vars, 1, figsize=figsize, dpi=dpi, sharex=True, squeeze=False
            )
            return fig, list(axes[:, 0])

        # Single Axes
        if hasattr(ax, "plot"):
            if n_vars != 1:
                raise ValueError(
                    f"Got a single Axes but {n_vars} variables to plot; "
                    f"pass a sequence of {n_vars} Axes."
                )
            return ax.figure, [ax]

        # Sequence of Axes
        axes_list = list(ax)
        if len(axes_list) != n_vars:
            raise ValueError(
                f"Got {len(axes_list)} Axes but {n_vars} variables to plot."
            )
        return axes_list[0].figure, axes_list

    def plot(
        self,
        variables: Optional[Sequence[Union[str, int]]] = None,
        nodes: Optional[Union[int, Sequence[int]]] = None,
        t_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        max_nodes: int = 10,
        default_window: Optional[float] = 10_000.0,
        ax=None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: float = 150,
        **plot_kwargs,
    ):
        """Quick-look plot of the solution trajectory.

        One subplot per selected variable, with selected nodes overlaid as lines.
        Intended for first-pass inspection, not publication.

        Args:
            variables: Variables to plot. Names (requires ``variable_names``)
                or integer indices, mixable. ``None`` plots all variables.
            nodes: Node selection. ``None`` plots all nodes up to ``max_nodes``;
                ``int N`` plots the first ``N``; a sequence selects explicitly.
            t_range: ``(t_start, t_end)`` time window. Either bound may be
                ``None`` for open-ended. Soft-clipped to available data with a
                warning if the overlap is partial.
            max_nodes: Cap when ``nodes=None`` to avoid plotting hundreds of
                lines. Emits a warning when the cap kicks in.
            default_window: When ``t_range`` is None, plot the first
                ``default_window`` time units (default 10_000, matching ms).
                Set to ``None`` to plot the full range.
            ax: Existing Axes (single, when one variable) or sequence of Axes
                (one per variable). ``None`` creates a new figure.
            figsize: Figure size when creating a new figure.
            dpi: Figure DPI when creating a new figure (ignored if ``ax`` is
                supplied). Default 150.
            **plot_kwargs: Forwarded to ``ax.plot`` (e.g. ``color``, ``lw``).

        Returns:
            ``(fig, axes)`` where ``axes`` is a list with one Axes per variable.
        """
        var_idx, var_labels = self._resolve_variables(variables)
        node_idx = self._resolve_nodes(nodes, self.ys.shape[2], max_nodes)
        t_mask = self._resolve_t_mask(t_range, default_window)

        ts_sel = self.ts[t_mask]
        ys_sel = self.ys[t_mask][:, var_idx, :][:, :, node_idx]

        fig, axes = self._prepare_axes(ax, len(var_idx), figsize, dpi)

        plot_kwargs.setdefault("alpha", 0.7 if len(node_idx) > 3 else 1.0)
        show_legend = len(node_idx) <= 8

        for k, label in enumerate(var_labels):
            a = axes[k]
            for j, ni in enumerate(node_idx):
                a.plot(
                    ts_sel,
                    ys_sel[:, k, j],
                    label=f"node {ni}" if show_legend else None,
                    **plot_kwargs,
                )
            a.set_ylabel(label)
            a.set_title(label)
            a.grid(True, alpha=0.3)
            if show_legend:
                a.legend(fontsize="small", loc="best")

        axes[-1].set_xlabel("time")
        fig.tight_layout()
        return fig, axes


@tree_util.register_pytree_with_keys_class
class DiffraxSolution(NativeSolution):
    """Solution wrapper for Diffrax solvers, compatible with NativeSolution interface.

    Wraps a diffrax Solution object and exposes the NativeSolution interface
    (.ts, .ys, .dt), while preserving access to the full diffrax solution via
    ``_solution`` (use this to reach ``.stats``, ``.result``, etc.).

    Note: the Diffrax prepare path does not apply ``VARIABLES_OF_INTEREST``
    filtering and discards auxiliaries, so ``ys`` always has shape
    ``[n_time, N_STATES, n_nodes]`` and ``variable_names`` always equals
    ``STATE_NAMES``. Use a native solver if you need VOI filtering or
    auxiliary outputs.

    Attributes:
        _solution: The underlying diffrax Solution object
        dt: Effective save time step inferred from saveat.ts at prepare time,
            or None if saveat was not ts-based. Only meaningful for uniformly
            spaced ``SaveAt(ts=...)``.
        variable_names: Names of the variables along axis 1 of ys, in order.
    """

    def __init__(
        self,
        solution,
        dt=None,
        variable_names: Optional[Tuple[str, ...]] = None,
    ):
        self._solution = solution
        self.dt = dt
        self.variable_names = (
            tuple(variable_names) if variable_names is not None else None
        )

    @property
    def ts(self):
        return self._solution.ts

    @property
    def ys(self):
        return self._solution.ys

    @property
    def stats(self):
        return self._solution.stats

    @property
    def result(self):
        return self._solution.result

    def tree_flatten(self):
        children = (self._solution,)
        aux_data = {"dt": self.dt, "variable_names": self.variable_names}
        return children, aux_data

    def tree_flatten_with_keys(self):
        children = ((tree_util.GetAttrKey("solution"), self._solution),)
        aux_data = {"dt": self.dt, "variable_names": self.variable_names}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            children[0],
            dt=aux_data.get("dt"),
            variable_names=aux_data.get("variable_names"),
        )

    def __repr__(self):
        return (
            f"DiffraxSolution(shape={self.ys.shape}, "
            f"t=[{self.ts[0]:.2f}, {self.ts[-1]:.2f}], dt={self.dt}, "
            f"variable_names={self.variable_names})"
        )


def wrap_native_result(
    trajectory: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    variable_names: Optional[Tuple[str, ...]] = None,
) -> NativeSolution:
    """Wrap native solver trajectory in solution object.

    Args:
        trajectory: Trajectory array from native solver, shape [n_time, n_variables, n_nodes]
        t0: Start time
        t1: End time
        dt: Time step
        variable_names: Names of the variables along axis 1 of trajectory, in order.

    Returns:
        NativeSolution with .ys, .ts, and .variable_names attributes like Diffrax
    """
    n_steps = trajectory.shape[0]
    # Native solvers scan over time_steps = arange(t0, t1, dt) and emit the
    # post-step state on each iteration. Trajectory[i] is therefore the state
    # at time t0 + (i + 1) * dt, so the save grid runs (t0, t1] with the
    # endpoint t1 included and the initial state t0 excluded.
    ts = t0 + (jnp.arange(n_steps) + 1) * dt
    return NativeSolution(ts=ts, ys=trajectory, dt=dt, variable_names=variable_names)
