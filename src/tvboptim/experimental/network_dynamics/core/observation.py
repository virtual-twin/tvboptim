"""Preparation contracts and graph-order heterogeneous observations."""

import abc
import copy
import dataclasses
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .bunch import Bunch
from .heterogeneous import _normalize_readout, _require_name


@dataclass(frozen=True)
class SimulationGrid:
    """Describe the regular input grid presented to a temporal monitor.

    ``t0`` is the original simulation origin and may be a scalar JAX value.
    The other fields are structural. ``first_sample_offset`` describes the
    first input label relative to ``t0`` and defaults to ``dt`` for raw native
    solver output.

    Parameters
    ----------
    t0
        Original simulation origin. Callback code may use it in JAX arithmetic
        but must not require its Python value.
    dt
        Input interval in milliseconds.
    n_steps
        Number of input samples in the complete trajectory.
    first_sample_offset
        Label of the first input sample relative to ``t0``. ``None`` means
        ``dt``, the native solver's post-step convention.
    """

    t0: Any
    dt: float
    n_steps: int
    first_sample_offset: float | None = None

    def output(self, period, *, label="last_input", variable_names=None):
        """Describe an output sampled every ``period`` on this grid.

        Parameters
        ----------
        period
            Output interval. It must be a positive integer multiple of ``dt``.
        label
            Timestamp rule for each emitted sample. ``"last_input"`` labels it
            with the last consumed input sample, ``"window_center"`` with the
            center of the completed window, and ``"period_end"`` with the end
            of the period measured from the original origin ``t0``. Labels do
            not change when a sample is emitted: every rule emits only after
            its complete window was consumed.
        variable_names
            Output channel names, or ``None``.

        Returns
        -------
        ObservationOutput
        """
        stride = sampling_stride(period, self.dt, label="Output period")
        first = float(_input_first_sample_offset(self))
        if label == "last_input":
            offset = first + (stride - 1) * float(self.dt)
        elif label == "window_center":
            offset = first + (stride - 1) * float(self.dt) / 2.0
        elif label == "period_end":
            offset = float(period)
        else:
            raise ValueError(
                "Output label must be 'last_input', 'window_center', or "
                f"'period_end'; got {label!r}"
            )
        return ObservationOutput(
            period=float(period),
            first_sample_offset=offset,
            variable_names=variable_names,
        )


@dataclass(frozen=True)
class ObservationOutput:
    """Describe one uniformly sampled observation output.

    Parameters
    ----------
    period
        Output interval in milliseconds. It must be a positive integer multiple
        of the receiving ``SimulationGrid`` interval.
    first_sample_offset
        Label of the first emitted sample relative to the original simulation
        origin. It must satisfy ``0 < first_sample_offset <= period``.
    variable_names
        Output channel names, or ``None`` when the input has no names. When
        supplied, the tuple length must equal the emitted channel count.
    """

    period: float
    first_sample_offset: float
    variable_names: tuple[str, ...] | None


@dataclass(frozen=True)
class PreparedObservation:
    """Define a temporal monitor's block execution recipe.

    ``prepare_observation`` returns this immutable recipe. The native solver
    and ``apply_observation`` use the same callbacks, so scientific update code
    is implemented once.

    Fixed preparation values such as indices, strides, kernels, and prepared
    child recipes are captured by the callbacks. Current live values arrive
    through ``params``.

    Parameters
    ----------
    params
        Initial live parameter PyTree. Native preparation publishes it as
        ``config.monitor`` so compatible values can be swept or differentiated.
    init
        JAX-compatible callback ``init(params) -> state``. It runs once at each
        model invocation, before block scanning, and constructs the scientific
        state from the current live parameters. Stateless monitors return
        ``None``.
    update
        JAX-compatible callback
        ``update(state, block, params) -> (next_state, samples)``. ``block`` is
        shaped ``[input_samples, channels, nodes]`` and ``samples`` must be rank
        three with the declared completed-period sample count. The callback
        must preserve the state signature and node axis.
    output
        Metadata for the emitted series. A mapping is reserved for prepared
        composites such as ``JointObservation``; an ordinary monitor emits one
        ``ObservationOutput``.
    alignment_steps
        Optional positive alignment in input samples. It must be a multiple of
        every output cadence. Regular solver blocks and truncation windows must
        honor this alignment; only the final incomplete tail may be shorter.

    Notes
    -----
    Preparation and abstract validation may trace both callbacks more than
    once. Callbacks execute inside JAX transforms and must avoid host transfers,
    mutation, and Python control flow that depends on traced values. State and
    output structures and shapes are static. ``init`` may choose state dtypes
    from live parameters, after which every update must preserve that
    invocation's state dtypes.
    """

    params: Any
    init: Callable
    update: Callable
    output: ObservationOutput | dict[str, ObservationOutput]
    alignment_steps: int | None = None


def _stateless(params):
    """Initializer for monitors without scientific state."""
    del params
    return None


def sampling_stride(period, dt, *, label="period") -> int:
    """Validate a positive sampling period and return its integer stride.

    Parameters
    ----------
    period
        Requested output period.
    dt
        Receiving input interval, in the same units as ``period``.
    label
        Name used in validation errors.

    Returns
    -------
    int
        ``period / dt`` after validating that the ratio is a positive integer.
    """
    try:
        period = float(period)
        dt = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} and dt must be concrete real scalars") from exc
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


def _input_first_sample_offset(grid: SimulationGrid) -> float:
    return grid.dt if grid.first_sample_offset is None else grid.first_sample_offset


def _validate_scalar(value, name):
    """Validate scalar shape without requiring a concrete scalar value."""
    try:
        shape = jnp.shape(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a scalar") from exc
    if shape != ():
        raise ValueError(f"{name} must be a scalar; got shape {shape}")


def _validate_grid(grid: SimulationGrid):
    sampling_stride(grid.dt, grid.dt, label="Simulation dt")
    try:
        n_steps = operator.index(grid.n_steps)
    except TypeError as exc:
        raise ValueError(
            f"Simulation n_steps must be a non-negative integer; got {grid.n_steps!r}"
        ) from exc
    if n_steps < 0:
        raise ValueError(
            f"Simulation n_steps must be a non-negative integer; got {n_steps!r}"
        )
    _validate_scalar(grid.t0, "Simulation t0")
    first = _input_first_sample_offset(grid)
    try:
        first = float(first)
    except (TypeError, ValueError) as exc:
        raise ValueError("Input first_sample_offset must be a concrete scalar") from exc
    if (
        not math.isfinite(first)
        or first <= 0.0
        or (
            first > float(grid.dt)
            and not math.isclose(first, float(grid.dt), rel_tol=1e-9, abs_tol=1e-9)
        )
    ):
        raise ValueError(
            "Input first_sample_offset must satisfy 0 < offset <= dt; "
            f"got offset={first!r}, dt={grid.dt!r}"
        )


def _output_items(prepared: PreparedObservation):
    output = prepared.output
    if isinstance(output, Mapping):
        if not output:
            raise ValueError("PreparedObservation output mapping must not be empty")
        for name, descriptor in output.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "PreparedObservation output keys must be non-empty strings"
                )
            if not isinstance(descriptor, ObservationOutput):
                raise TypeError(
                    f"PreparedObservation output {name!r} must be an ObservationOutput"
                )
        return tuple(output.items())
    if not isinstance(output, ObservationOutput):
        raise TypeError(
            "PreparedObservation.output must be an ObservationOutput or a "
            "non-empty mapping of them"
        )
    return ((None, output),)


def _validate_output_descriptor(name, output, grid):
    path = "output" if name is None else f"output {name!r}"
    stride = sampling_stride(output.period, grid.dt, label=f"{path} period")
    try:
        offset = float(output.first_sample_offset)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} first_sample_offset must be concrete") from exc
    if (
        not math.isfinite(offset)
        or offset <= 0.0
        or (
            offset > float(output.period)
            and not math.isclose(
                offset, float(output.period), rel_tol=1e-9, abs_tol=1e-9
            )
        )
    ):
        raise ValueError(
            f"{path} first_sample_offset must satisfy 0 < offset <= period; "
            f"got offset={offset!r}, period={output.period!r}"
        )
    names = output.variable_names
    if names is not None:
        if not isinstance(names, tuple):
            raise TypeError(f"{path} variable_names must be a tuple or None")
        if any(not isinstance(item, str) or not item for item in names):
            raise ValueError(f"{path} variable_names must contain non-empty strings")
    return stride


def observation_alignment_steps(prepared: PreparedObservation, grid: SimulationGrid):
    """Return the validated common alignment for a prepared recipe."""
    strides = [
        _validate_output_descriptor(name, output, grid)
        for name, output in _output_items(prepared)
    ]
    cadence_alignment = math.lcm(*strides)
    if prepared.alignment_steps is None:
        return cadence_alignment
    try:
        alignment = operator.index(prepared.alignment_steps)
    except TypeError as exc:
        raise ValueError(
            "PreparedObservation alignment_steps must be an integer"
        ) from exc
    if alignment <= 0 or any(alignment % stride for stride in strides):
        raise ValueError(
            "PreparedObservation alignment_steps must be a positive multiple "
            f"of every output stride {tuple(strides)}; got {alignment!r}"
        )
    return alignment


def _abstract_signature(tree):
    return jax.tree.map(
        lambda leaf: (tuple(leaf.shape), leaf.dtype),
        tree,
    )


def _same_abstract_signature(left, right):
    left_leaves, left_tree = jax.tree.flatten(_abstract_signature(left))
    right_leaves, right_tree = jax.tree.flatten(_abstract_signature(right))
    return left_tree == right_tree and left_leaves == right_leaves


def _validate_emission(
    monitor_name,
    output_name,
    values,
    descriptor,
    stride,
    block_length,
    n_nodes,
):
    path = monitor_name if output_name is None else f"{monitor_name}.{output_name}"
    if not hasattr(values, "shape") or len(values.shape) != 3:
        raise ValueError(
            f"{path}.update must emit a rank-three array shaped "
            "[samples, channels, nodes]"
        )
    expected_count = block_length // stride
    if values.shape[0] != expected_count:
        raise ValueError(
            f"{path}.update returned {values.shape[0]} samples for "
            f"{block_length} input steps at cadence {stride}. Expected "
            f"{expected_count} completed-period samples, shaped "
            f"[{expected_count}, channels, nodes]."
        )
    if values.shape[2] != n_nodes:
        raise ValueError(
            f"{path}.update changed the node axis: expected {n_nodes}, "
            f"got {values.shape[2]}"
        )
    names = descriptor.variable_names
    if names is not None and len(names) != values.shape[1]:
        raise ValueError(
            f"{path} declares {len(names)} variable names for "
            f"{values.shape[1]} output channels"
        )


def _validation_lengths(block_lengths, default_length):
    lengths = tuple(dict.fromkeys(int(length) for length in block_lengths))
    if not lengths:
        lengths = (int(default_length),)
    if any(length < 0 for length in lengths):
        raise ValueError("Observation block lengths must be non-negative")
    return lengths


def validate_prepared_observation(
    monitor,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    prepared: PreparedObservation,
    *,
    block_lengths=(),
):
    """Validate a prepared recipe and its abstract invocation signatures.

    Validation runs during preparation/tracing. It never executes the monitor
    numerically or transfers trajectory data to the host.
    """
    _validate_grid(grid)
    if not isinstance(prepared, PreparedObservation):
        raise TypeError(
            f"{type(monitor).__name__} preparation must return PreparedObservation"
        )
    if not callable(prepared.update):
        raise TypeError("PreparedObservation.update must be callable")
    if not callable(prepared.init):
        raise TypeError("PreparedObservation.init must be callable")
    if len(sample.shape) != 2:
        raise ValueError(
            "Temporal observations require per-step input shaped "
            f"[channels, nodes]; got {sample.shape}"
        )

    items = _output_items(prepared)
    strides = {
        name: _validate_output_descriptor(name, output, grid) for name, output in items
    }
    alignment = observation_alignment_steps(prepared, grid)
    lengths = _validation_lengths(block_lengths, grid.n_steps)

    monitor_name = type(monitor).__name__
    try:
        state = jax.eval_shape(prepared.init, prepared.params)
    except Exception as exc:
        raise ValueError(f"{monitor_name}.init failed abstract validation") from exc

    output_signatures = {}
    for length in lengths:
        block = jax.ShapeDtypeStruct((length,) + sample.shape, sample.dtype)
        try:
            result = jax.eval_shape(
                prepared.update,
                state,
                block,
                prepared.params,
            )
        except Exception as exc:
            raise ValueError(
                f"{monitor_name}.update failed abstract validation for block "
                f"length {length}"
            ) from exc
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(f"{monitor_name}.update must return (state, samples)")
        next_state, values = result
        if not _same_abstract_signature(state, next_state):
            raise ValueError(
                f"{monitor_name}.update must preserve state structure, shapes, and dtypes"
            )

        if isinstance(prepared.output, Mapping):
            if not isinstance(values, Mapping) or set(values) != set(prepared.output):
                raise ValueError(
                    f"{monitor_name}.update output keys must exactly match "
                    f"{tuple(prepared.output)}"
                )
            emitted = values.items()
        else:
            if isinstance(values, Mapping):
                raise ValueError(f"{monitor_name}.update must emit one array")
            emitted = ((None, values),)

        for name, value in emitted:
            descriptor = dict(items)[name]
            _validate_emission(
                monitor_name,
                name,
                value,
                descriptor,
                strides[name],
                length,
                sample.shape[1],
            )
            signature = (value.shape[1:], value.dtype)
            if name in output_signatures and output_signatures[name] != signature:
                raise ValueError(
                    f"{monitor_name} output {name!r} changes channel shape or dtype "
                    "between regular blocks and the final tail"
                )
            output_signatures[name] = signature
    return alignment


def validate_observation_input(monitor_name, expected, block):
    """Require the trace-time block signature used during preparation."""
    actual_shape = tuple(block.shape[1:])
    expected_shape = tuple(expected.shape)
    actual_dtype = jnp.dtype(block.dtype)
    expected_dtype = jnp.dtype(expected.dtype)
    if actual_shape != expected_shape or actual_dtype != expected_dtype:
        raise ValueError(
            f"{monitor_name} was prepared for input shape/dtype "
            f"{expected_shape}/{expected_dtype}, but received "
            f"{actual_shape}/{actual_dtype}. Prepare the model again for the "
            "changed observation input specification."
        )


def _shape_structure(tree):
    leaves, structure = jax.tree.flatten(tree)
    return structure, tuple(tuple(leaf.shape) for leaf in leaves)


def validate_observation_invocation(
    monitor_name, prepared, params, sample, block_lengths
):
    """Validate every executed chunk for the current live parameter signature."""
    lengths = _validation_lengths(block_lengths, 0)
    try:
        default_state = jax.eval_shape(prepared.init, prepared.params)
        state = jax.eval_shape(prepared.init, params)
    except Exception as exc:
        raise ValueError(
            f"{monitor_name}.init does not support the current monitor "
            "parameter shape/dtype signature"
        ) from exc
    if _shape_structure(default_state) != _shape_structure(state):
        raise ValueError(
            f"{monitor_name}.init changed state structure or shape for "
            "the current monitor parameters; prepare the model again"
        )

    output_signatures = None
    for length in lengths:
        block = jax.ShapeDtypeStruct((length,) + tuple(sample.shape), sample.dtype)
        try:
            default_result = jax.eval_shape(
                prepared.update, default_state, block, prepared.params
            )
            result = jax.eval_shape(prepared.update, state, block, params)
        except Exception as exc:
            raise ValueError(
                f"{monitor_name}.update does not support the current monitor "
                f"parameter shape/dtype signature for block length {length}"
            ) from exc
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                f"{monitor_name}.update must return (state, samples) for "
                f"block length {length}"
            )
        next_state, values = result
        if not _same_abstract_signature(state, next_state):
            raise ValueError(
                f"{monitor_name}.update must preserve state structure, shapes, "
                f"and dtypes for block length {length}"
            )

        if not isinstance(default_result, tuple) or len(default_result) != 2:
            raise ValueError(f"{monitor_name}.update must return (state, samples)")
        default_values = default_result[1]
        default_leaves, default_tree = jax.tree.flatten(default_values)
        leaves, tree = jax.tree.flatten(values)
        default_shapes = tuple(leaf.shape for leaf in default_leaves)
        shapes = tuple(leaf.shape for leaf in leaves)
        if tree != default_tree or shapes != default_shapes:
            raise ValueError(
                f"{monitor_name}.update changed its output structure or shape for "
                "the current monitor parameters at block length "
                f"{length}; prepare the model again"
            )

        signatures = tuple((leaf.shape[1:], leaf.dtype) for leaf in leaves)
        if output_signatures is None:
            output_signatures = signatures
        elif signatures != output_signatures:
            raise ValueError(
                f"{monitor_name}.update changed output channel shape or dtype "
                "between executed block lengths for the current monitor parameters; "
                "prepare the model again"
            )


def prepared_single_output_sample(prepared, sample, block_length):
    """Infer one prepared stage's emitted per-sample shape and dtype."""
    if not isinstance(prepared.output, ObservationOutput):
        raise ValueError(
            "A shared or child observation stage must emit one time series"
        )
    block = jax.ShapeDtypeStruct(
        (int(block_length),) + tuple(sample.shape), sample.dtype
    )

    def run(block_values, params):
        return prepared.update(prepared.init(params), block_values, params)[1]

    values = jax.eval_shape(run, block, prepared.params)
    if not hasattr(values, "shape") or len(values.shape) != 3:
        raise ValueError(
            "A shared or child observation stage must emit [samples, channels, nodes]"
        )
    return jax.ShapeDtypeStruct(values.shape[1:], values.dtype)


def _make_observation_result(values, t0, output, t_offset=0.0):
    from ..result import NativeSolution

    ts = (
        t0
        + output.first_sample_offset
        + jnp.arange(values.shape[0]) * output.period
        + t_offset
    )
    return NativeSolution(
        ts=ts,
        ys=values,
        dt=output.period,
        variable_names=output.variable_names,
    )


def observation_result(values, t0, prepared, *, t_offset=0.0):
    """Construct one or several solutions from a prepared recipe's metadata."""
    if isinstance(prepared.output, Mapping):
        return {
            name: _make_observation_result(values[name], t0, output, t_offset=t_offset)
            for name, output in prepared.output.items()
        }
    return _make_observation_result(values, t0, prepared.output, t_offset=t_offset)


def apply_observation(monitor, solution, *, t_offset=0.0):
    """Apply a streaming monitor's recipe to a stored trajectory.

    This is the shared post-hoc executor for custom monitors. A monitor normally
    delegates its ``__call__`` method here so post-hoc and native ``observe=``
    execution use the same prepared callbacks.

    Parameters
    ----------
    monitor
        A streaming monitor: any object with a callable ``prepare`` method
        returning a ``PreparedObservation``.
    solution
        An object exposing ``ys``, ``ts``, and a concrete ``dt``. Values must be
        shaped ``[samples, channels, nodes]`` with one timestamp per sample.
    t_offset
        Scalar added to output timestamps. It may be a scalar JAX value under
        ``jax.jit`` and does not change the observed values.

    Returns
    -------
    NativeSolution or dict[str, NativeSolution]
        One regular observed series, or the flat named result mapping produced
        by ``JointObservation``.
    """
    if not hasattr(solution, "ys") or not hasattr(solution, "ts"):
        raise TypeError("solution must expose ys and ts arrays")
    dt = getattr(solution, "dt", None)
    if dt is None:
        raise ValueError("Observations require a solution with a concrete dt")
    dt = float(dt)
    _validate_scalar(t_offset, "t_offset")
    n_steps = solution.ys.shape[0]
    if solution.ts.shape != (n_steps,):
        raise ValueError(
            "solution.ts must have one entry per trajectory sample; "
            f"got ts shape {solution.ts.shape} and ys shape {solution.ys.shape}"
        )
    if solution.ys.ndim != 3:
        raise ValueError(
            "Temporal observations require solution.ys shaped "
            f"[samples, channels, nodes]; got {solution.ys.shape}"
        )
    t0 = solution.ts[0] - dt if n_steps else jnp.asarray(0.0)
    grid = SimulationGrid(t0=t0, dt=dt, n_steps=n_steps)
    sample = jax.ShapeDtypeStruct(solution.ys.shape[1:], solution.ys.dtype)
    variable_names = getattr(solution, "variable_names", None)
    prepared = prepare_observation(monitor, grid, sample, variable_names)
    validate_prepared_observation(
        monitor, grid, sample, prepared, block_lengths=(n_steps,)
    )
    state = prepared.init(prepared.params)
    _state, values = prepared.update(state, solution.ys, prepared.params)
    return observation_result(values, t0, prepared, t_offset=t_offset)


class StreamingMonitor(abc.ABC):
    """Optional base for monitors that run inside solver blocks.

    A streaming monitor is any object with a callable
    ``prepare(grid, sample, variable_names)`` method returning a
    ``PreparedObservation``. Inheriting this class is optional: it declares the
    method and supplies ``__call__``, which applies the same recipe to a stored
    solution. ``isinstance(m, StreamingMonitor)`` is structural and holds for
    every object whose class defines a callable ``prepare``.
    """

    @abc.abstractmethod
    def prepare(
        self,
        grid: SimulationGrid,
        sample: jax.ShapeDtypeStruct,
        variable_names: tuple | None,
    ) -> PreparedObservation:
        """Return the block execution recipe for this monitor on ``grid``."""

    def __call__(self, solution, t_offset=0.0):
        return apply_observation(self, solution, t_offset=t_offset)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is StreamingMonitor and callable(getattr(C, "prepare", None)):
            return True
        return NotImplemented


class Identity(StreamingMonitor):
    """Preserve a stage's values, sampling grid, dtype, and channel names.

    Use ``Identity()`` as a ``JointObservation`` branch to retain the shared
    stage directly. Passing it as a BOLD monitor's explicit ``downsample`` also
    selects the actual incoming grid instead of that monitor's default temporal
    averaging.
    """

    def prepare(self, grid, sample, variable_names):
        return _prepare_identity(self, grid, sample, variable_names)


class JointObservation(StreamingMonitor):
    """Apply one optional shared stage followed by flat named branches.

    Parameters
    ----------
    outputs
        Non-empty mapping from output names to ordinary temporal monitors. Every
        branch receives the shared stage's output and returns its own regular
        ``NativeSolution`` with an independent cadence and timestamp grid.
    preprocess
        Optional ordinary monitor evaluated once before all branches. ``None``
        passes the raw observation input to each branch.

    Notes
    -----
    Results are an ordinary ``dict`` keyed by ``outputs``. Prepared parameters
    and state use ``preprocess`` and ``outputs.<name>`` namespaces. Composition
    is intentionally limited to one shared stage and flat branches: nested
    joints and mapping-valued stages are rejected.
    """

    def __init__(self, *, outputs, preprocess=None):
        if not isinstance(outputs, Mapping):
            raise TypeError("JointObservation outputs must be a mapping")
        if not outputs:
            raise ValueError("JointObservation requires at least one output")
        normalized = {}
        for name, monitor in outputs.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "JointObservation output names must be non-empty strings"
                )
            if isinstance(monitor, JointObservation):
                raise ValueError("JointObservation branches cannot be nested joints")
            normalized[name] = monitor
        if isinstance(preprocess, JointObservation):
            raise ValueError("JointObservation preprocess cannot be another joint")
        self.outputs = normalized
        self.preprocess = preprocess

    def prepare(self, grid, sample, variable_names):
        return _prepare_joint_observation(self, grid, sample, variable_names)


def _owned_copy(tree):
    """Deep-copy ``tree`` so later source mutation has no effect.

    The copy includes static metadata held in the tree structure, such as
    static fields of nested Equinox modules. JAX array leaves, including
    tracers, are immutable and shared, which keeps dependencies on traced
    preparation inputs.
    """
    memo = {
        id(leaf): leaf for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)
    }
    return copy.deepcopy(tree, memo)


_JAX_SCALARS = (bool, int, float, complex, np.generic, np.ndarray, jax.Array)


class SampledMonitor(StreamingMonitor, eqx.Module):
    """Per-sample recurrence sampled at completed period endpoints.

    Subclasses declare a ``period`` and implement ``init`` and ``step``. Each
    input sample produces one candidate output shaped like the input sample;
    the candidate at the last input of every completed period is emitted and
    labelled with that input's timestamp.

    Non-static fields are live parameters and appear under their field names in
    ``config.monitor``. Fields declared with ``eqx.field(static=True)``, such
    as ``period``, are structural and require preparing again when changed.
    Methods run on an instance rebuilt from the current live values, so
    ``self.<field>`` always reads the invocation's value. The rebuilt instance
    is created without calling the constructor.

    Candidate outputs keep the input channels, so channel names are inherited.
    Preserving channel order and meaning is the author's responsibility. Use
    ``StreamingMonitor`` for outputs that transform channels.
    """

    @abc.abstractmethod
    def init(self, sample_spec: jax.ShapeDtypeStruct):
        """Return the initial state for inputs described by ``sample_spec``."""

    @abc.abstractmethod
    def step(self, state, sample, dt):
        """Advance ``state`` by one input sample; return ``(state, value)``."""

    def _live_fields(self):
        live, fixed = {}, {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.metadata.get("static", False):
                fixed[field.name] = _owned_copy(value)
                continue
            for leaf in jax.tree.leaves(value):
                if not isinstance(leaf, _JAX_SCALARS):
                    raise TypeError(
                        f"{type(self).__name__}.{field.name} is a live field but "
                        f"holds {type(leaf).__name__}, which is not a JAX value. "
                        "Declare structural settings with eqx.field(static=True)."
                    )
            live[field.name] = _owned_copy(value)
        return live, fixed

    def prepare(self, grid, sample, variable_names):
        monitor_type = type(self)
        name = monitor_type.__name__
        live, fixed = self._live_fields()
        stride = sampling_stride(self.period, grid.dt, label=f"{name} period")
        dt = float(grid.dt)

        def bind(params):
            monitor = object.__new__(monitor_type)
            for key, value in fixed.items():
                object.__setattr__(monitor, key, value)
            for key in live:
                object.__setattr__(monitor, key, params[key])
            return monitor

        def init(params):
            return bind(params).init(sample)

        def update(state, block, params):
            monitor = bind(params)
            state, values = jax.lax.scan(
                lambda current, value: monitor.step(current, value, dt), state, block
            )
            if values.shape[1:] != tuple(sample.shape):
                raise ValueError(
                    f"{name}.step must return one value shaped like the input "
                    f"sample {tuple(sample.shape)}; got {values.shape[1:]}"
                )
            return state, values[stride - 1 :: stride]

        return PreparedObservation(
            params=Bunch(live),
            init=init,
            update=update,
            output=grid.output(self.period, variable_names=variable_names),
        )


def prepare_observation(
    monitor,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    variable_names: tuple | None,
) -> PreparedObservation:
    """Prepare a streaming monitor for block-wise execution.

    Calls ``monitor.prepare(grid, sample, variable_names)``, which receives a
    ``SimulationGrid``, a ``jax.ShapeDtypeStruct`` describing one
    ``[channels, nodes]`` input value, and optional channel names, and returns
    a ``PreparedObservation``. Subclasses select their preparation through
    ordinary method resolution. To stream a type you cannot modify, wrap it in
    a ``StreamingMonitor`` subclass that implements ``prepare``.

    The returned recipe is validated abstractly for cadence, callback return
    structure, state invariance, rank, node count, dtype, and channel metadata
    before execution.
    """
    if isinstance(monitor, type):
        raise TypeError(
            f"Pass a monitor instance, e.g. {monitor.__name__}(...), not the class"
        )
    prepare = getattr(monitor, "prepare", None)
    name = type(monitor).__name__
    if not callable(prepare):
        raise TypeError(
            f"{name} has no streaming preparation method. Implement "
            "prepare(grid, sample, variable_names) returning a "
            "PreparedObservation, or wrap the object in a StreamingMonitor "
            "subclass. Post-hoc use through its own call remains available."
        )
    prepared = prepare(grid, sample, variable_names)
    if not isinstance(prepared, PreparedObservation):
        raise TypeError(
            f"{name}.prepare(...) must return PreparedObservation; "
            f"returned {type(prepared).__name__}"
        )
    return prepared


def _identity_update(state, block, params):
    del params
    return state, block


def _prepare_identity(
    monitor,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    variable_names: tuple | None,
) -> PreparedObservation:
    """Prepare a pass-through stage on its actual incoming grid."""
    del monitor, sample
    return PreparedObservation(
        params=Bunch(),
        init=_stateless,
        update=_identity_update,
        output=grid.output(grid.dt, variable_names=variable_names),
    )


def _prepare_joint_observation(
    monitor,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    variable_names: tuple | None,
) -> PreparedObservation:
    """Prepare one shared stage and a flat mapping of named child stages."""
    shared_monitor = Identity() if monitor.preprocess is None else monitor.preprocess
    try:
        shared = prepare_observation(shared_monitor, grid, sample, variable_names)
    except Exception as exc:
        raise ValueError("JointObservation preprocess could not be prepared") from exc
    if not isinstance(shared.output, ObservationOutput):
        raise ValueError(
            "JointObservation preprocess must produce one time series, not a mapping"
        )

    shared_stride = sampling_stride(
        shared.output.period, grid.dt, label="JointObservation preprocess period"
    )
    try:
        shared_alignment = observation_alignment_steps(shared, grid)
        validate_prepared_observation(
            shared_monitor,
            grid,
            sample,
            shared,
            block_lengths=(shared_alignment,),
        )
        shared_sample = prepared_single_output_sample(shared, sample, shared_alignment)
    except Exception as exc:
        raise ValueError("JointObservation preprocess is invalid") from exc

    child_grid = SimulationGrid(
        t0=grid.t0,
        dt=float(shared.output.period),
        n_steps=grid.n_steps // shared_stride,
        first_sample_offset=float(shared.output.first_sample_offset),
    )
    children = Bunch()
    child_params = Bunch()
    outputs = {}
    alignments = [shared_alignment]
    names = tuple(monitor.outputs)
    for name in names:
        child_monitor = monitor.outputs[name]
        try:
            child = prepare_observation(
                child_monitor,
                child_grid,
                shared_sample,
                shared.output.variable_names,
            )
        except Exception as exc:
            raise ValueError(
                f"JointObservation output {name!r} could not be prepared"
            ) from exc
        if not isinstance(child.output, ObservationOutput):
            raise ValueError(
                f"JointObservation output {name!r} must produce one time series, "
                "not a mapping"
            )
        try:
            child_alignment = observation_alignment_steps(child, child_grid)
            validate_prepared_observation(
                child_monitor,
                child_grid,
                shared_sample,
                child,
                block_lengths=(child_alignment,),
            )
        except Exception as exc:
            raise ValueError(f"JointObservation output {name!r} is invalid") from exc
        children[name] = child
        child_params[name] = child.params
        outputs[name] = child.output
        alignments.append(shared_stride * child_alignment)

    def init(params):
        child_states = Bunch()
        for name in names:
            child_states[name] = children[name].init(params.outputs[name])
        return Bunch(preprocess=shared.init(params.preprocess), outputs=child_states)

    def update(state, block, params):
        shared_state, shared_values = shared.update(
            state.preprocess, block, params.preprocess
        )
        _validate_emission(
            "JointObservation preprocess",
            None,
            shared_values,
            shared.output,
            shared_stride,
            block.shape[0],
            sample.shape[1],
        )
        validate_observation_input(
            "JointObservation preprocess", shared_sample, shared_values
        )
        child_states = Bunch()
        chunks = {}
        for name in names:
            child_states[name], chunks[name] = children[name].update(
                state.outputs[name], shared_values, params.outputs[name]
            )
        return Bunch(preprocess=shared_state, outputs=child_states), chunks

    return PreparedObservation(
        params=Bunch(preprocess=shared.params, outputs=child_params),
        init=init,
        update=update,
        output=outputs,
        alignment_steps=math.lcm(*alignments),
    )


class GroupObservation:
    """Project group variables of interest into common graph-order channels.

    Each mapping value is a variable-of-interest name, a tuple of names, or a
    ``readout(voi, params) -> [Q, n_group_nodes]`` callable. Equal channel
    width only establishes shape compatibility; users remain responsible for
    making group-specific transformations scientifically commensurable.

    ``monitor`` optionally applies a temporal monitor or ``JointObservation``
    to the common channels within solver blocks. Readout parameters stay in
    ``config.observation`` and monitor parameters in ``config.monitor``.

    Partial coverage is allowed for ordinary projection. Before a temporal
    monitor it is rejected unless ``allow_partial_coverage=True``. The opt-in
    makes ``fill_value`` a real monitor input; it does not make BOLD or later
    statistics fill-aware. The deprecated ``reduce=`` path applies the same
    coverage rule during its compatibility period.
    """

    def __init__(
        self,
        readouts: Mapping[str, Any],
        *,
        params: Mapping[str, Any] | None = None,
        channels,
        fill_value=0.0,
        allow_partial_coverage: bool = False,
        monitor=None,
    ):
        if not isinstance(readouts, Mapping):
            raise TypeError("GroupObservation readouts must be a mapping")
        if not readouts:
            raise ValueError("GroupObservation requires at least one group")
        if params is not None and not isinstance(params, Mapping):
            raise TypeError("GroupObservation.params must be a mapping or None")
        if not isinstance(channels, (tuple, list)):
            raise TypeError("GroupObservation.channels must be a tuple or list")
        channels = tuple(channels)
        if not channels:
            raise ValueError("GroupObservation.channels must not be empty")
        if any(not isinstance(name, str) or not name for name in channels):
            raise ValueError("GroupObservation channel names must be non-empty strings")
        if len(set(channels)) != len(channels):
            raise ValueError("GroupObservation channel names must be unique")
        if not isinstance(allow_partial_coverage, bool):
            raise TypeError("allow_partial_coverage must be bool")
        if isinstance(monitor, GroupObservation):
            raise TypeError("GroupObservation.monitor must be a temporal monitor")

        self.readouts = {
            _require_name(name, "observation group"): _normalize_readout(
                value, "observation"
            )
            for name, value in readouts.items()
        }
        unknown_params = set(params or {}) - set(self.readouts)
        if unknown_params:
            raise ValueError(
                "GroupObservation.params must refer to observed groups; "
                f"unknown {sorted(unknown_params)}"
            )
        self.params = {
            name: (params or {}).get(name, Bunch()) for name in self.readouts
        }
        self.channels = channels
        self.fill_value = fill_value
        self.allow_partial_coverage = allow_partial_coverage
        self.monitor = monitor
