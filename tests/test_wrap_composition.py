"""Tests for optimising a swept slot via ``AbstractAxis(wrap=...)``.

Covers the composition that previously required a manual re-wrap inside the
mapped model: an axis declares how its substituted value becomes a Parameter,
so `Space`/`ParallelExecution` and `OptaxOptimizer` compose directly.

Also covers the tracing policy in `OptaxOptimizer.run` that keeps graph size,
and therefore memory, independent of `max_steps` under `ParallelExecution`.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
import pytest

jax.config.update("jax_enable_x64", True)

from tvboptim.execution import ParallelExecution, SequentialExecution
from tvboptim.optim import OptaxOptimizer
from tvboptim.types import (
    DataAxis,
    GridAxis,
    Parameter,
    SigmoidBoundedParameter,
    Space,
)
from tvboptim.types.parameter import LogPositiveParameter

LOW, HIGH = 0.0, 5.0
STARTS = jnp.array([0.5, 1.5, 2.5, 3.5])
TARGET = 2.0


def _sigmoid_wrap():
    return partial(SigmoidBoundedParameter, low=LOW, high=HIGH)


def _loss(state):
    return jnp.sum((state["x"].constrained_value - state["target"]) ** 2)


def _optimize(state, max_steps=5, **kwargs):
    opt = OptaxOptimizer(loss=_loss, optimizer=optax.sgd(0.5))
    final, _ = opt.run(state, max_steps=max_steps, **kwargs)
    return final["x"].constrained_value


# ---------------------------------------------------------------- wrap basics


def test_axis_without_wrap_substitutes_raw_array():
    """The pre-existing behaviour must be byte-for-byte unchanged."""
    space = Space({"x": DataAxis(STARTS), "target": TARGET}, mode="zip")
    for i, start in enumerate(STARTS):
        value = space[i]["x"]
        assert not isinstance(value, Parameter)
        assert float(value) == float(start)


def test_wrap_produces_parameter_at_substitution():
    space = Space(
        {"x": DataAxis(STARTS, wrap=_sigmoid_wrap()), "target": TARGET}, mode="zip"
    )
    for i, start in enumerate(STARTS):
        value = space[i]["x"]
        assert isinstance(value, SigmoidBoundedParameter)
        assert float(value.constrained_value) == pytest.approx(float(start), abs=1e-9)


def test_parameter_bounds_are_explicit_and_independent_of_axis_bounds():
    axis = GridAxis(
        LOW + 1.0,
        HIGH - 1.0,
        4,
        wrap=partial(SigmoidBoundedParameter, low=LOW, high=HIGH),
    )
    p = axis.wrap(jnp.asarray(2.0))
    assert isinstance(p, SigmoidBoundedParameter)
    assert (p.low, p.high) == (LOW, HIGH)


def test_axis_does_not_override_explicit_parameter_bounds():
    axis = GridAxis(
        0.0, 10.0, 3, wrap=partial(SigmoidBoundedParameter, low=1.0, high=2.0)
    )
    p = axis.wrap(jnp.asarray(1.5))
    assert (p.low, p.high) == (1.0, 2.0)


@pytest.mark.parametrize(
    "make_axis",
    [
        lambda: DataAxis(STARTS, wrap=SigmoidBoundedParameter),
        lambda: GridAxis(LOW, HIGH, 4, wrap=SigmoidBoundedParameter),
    ],
)
def test_bounded_wrap_requires_explicit_bounds_at_materialization(make_axis):
    space = Space({"x": make_axis()})
    with pytest.raises(TypeError, match="required positional arguments"):
        _ = space[0]


def test_non_callable_wrap_raises_at_materialization():
    space = Space({"x": DataAxis(STARTS, wrap=42)})
    with pytest.raises(TypeError, match="not callable"):
        _ = space[0]


def test_wrap_does_not_affect_space_size_or_dataframe():
    """Wrapping happens after combination, so sweep bookkeeping is untouched."""
    plain = Space({"x": DataAxis(STARTS), "target": TARGET}, mode="zip")
    wrapped = Space(
        {"x": DataAxis(STARTS, wrap=_sigmoid_wrap()), "target": TARGET}, mode="zip"
    )
    assert wrapped.N == plain.N
    assert wrapped.to_dataframe().equals(plain.to_dataframe())


def test_collect_complete_state_rejects_wrapped_axes():
    """A batched Parameter is not generally a batch of lane Parameters."""
    space = Space({"x": DataAxis(STARTS, wrap=Parameter)}, mode="zip")
    with pytest.raises(ValueError, match=r"collect\(combine=True\).+wrapped axes"):
        space.collect(combine=True)


def test_collect_raw_inputs_remains_available_for_wrapped_axes():
    space = Space({"x": DataAxis(STARTS, wrap=Parameter)}, mode="zip")
    axis_state, static_state = space.collect(combine=False)
    assert not isinstance(axis_state["x"], Parameter)
    assert static_state["x"] is None
    assert jnp.array_equal(jnp.ravel(axis_state["x"]), STARTS)


def test_space_keeps_a_static_parameter_intact():
    """A Parameter on no axis must survive partitioning whole.

    Without Parameter as a partition leaf, eqx.partition descends into it and
    leaves a hollow copy (value=None) in axis_state, which then wins the
    recombine. Reading it raised rather than returning the value.
    """
    static = SigmoidBoundedParameter(1.0, low=LOW, high=HIGH)
    space = Space({"x": DataAxis(STARTS), "w": static}, mode="zip")
    for i in range(space.N):
        got = space[i]["w"]
        assert isinstance(got, SigmoidBoundedParameter)
        assert got.value is not None
        assert float(got.constrained_value) == pytest.approx(1.0, abs=1e-9)


def test_static_parameter_is_optimized_in_every_lane():
    """The Hopf shape: some slots swept, one optimisable slot on no axis."""

    def fit(state):
        opt = OptaxOptimizer(
            loss=lambda s: (s["w"].constrained_value - s["x"]) ** 2,
            optimizer=optax.sgd(0.5),
        )
        final, _ = opt.run(state, max_steps=30)
        return final["w"].constrained_value

    space = Space(
        {"x": DataAxis(STARTS), "w": SigmoidBoundedParameter(1.0, low=LOW, high=HIGH)},
        mode="zip",
    )
    got = jnp.ravel(
        jnp.asarray(ParallelExecution(fit, space, n_vmap=2, n_pmap=1).run().results)
    )
    # Each lane pulls the shared static Parameter toward its own swept target.
    assert jnp.all(jnp.abs(got - STARTS) < jnp.abs(1.0 - STARTS))


def test_wrap_survives_slicing():
    """space[a:b] rebuilds DataAxis objects; the wrap must be forwarded."""
    space = Space(
        {"x": DataAxis(STARTS, wrap=_sigmoid_wrap()), "target": TARGET}, mode="zip"
    )
    subset = space[1:3]
    assert isinstance(subset[0]["x"], SigmoidBoundedParameter)
    assert float(subset[0]["x"].constrained_value) == pytest.approx(
        float(STARTS[1]), abs=1e-9
    )


def test_invalid_strict_wrap_is_rejected_before_sequential_materialization():
    space = Space({"x": DataAxis(jnp.asarray([-1.0]), wrap=LogPositiveParameter)})
    with pytest.raises(ValueError, match="must be > 0.0"):
        _ = space[0]


def test_invalid_strict_wrap_is_rejected_before_parallel_trace():
    space = Space({"x": DataAxis(jnp.asarray([-1.0]), wrap=LogPositiveParameter)})
    with pytest.raises(ValueError, match="must be > 0.0"):
        ParallelExecution(lambda state: state["x"].constrained_value, space, n_pmap=1)


def test_strict_wrap_validation_honors_partial_configuration():
    space = Space(
        {
            "x": DataAxis(
                jnp.asarray([0.5]),
                wrap=partial(LogPositiveParameter, lower=1.0),
            )
        }
    )
    with pytest.raises(ValueError, match="must be > 1.0"):
        _ = space[0]


# ------------------------------------------------------- composition end-to-end


def _serial_results(max_steps=5):
    return [
        float(
            _optimize(
                {"x": SigmoidBoundedParameter(s, low=LOW, high=HIGH), "target": TARGET},
                max_steps=max_steps,
            )
        )
        for s in STARTS
    ]


def test_sequential_execution_optimizes_without_manual_rewrap():
    space = Space(
        {"x": DataAxis(STARTS, wrap=_sigmoid_wrap()), "target": TARGET}, mode="zip"
    )
    results = SequentialExecution(model=_optimize, statespace=space).run().results
    for got, want in zip(results, _serial_results()):
        assert float(got) == pytest.approx(want, abs=1e-10)


def test_parallel_execution_matches_serial_optimization():
    """The batched lanes reproduce serial runs from the same starts.

    `_optimize` never mentions a Parameter type: the axis carries the wrap.
    """
    space = Space(
        {"x": DataAxis(STARTS, wrap=_sigmoid_wrap()), "target": TARGET}, mode="zip"
    )
    batched = ParallelExecution(model=_optimize, space=space, n_vmap=2, n_pmap=1).run()
    got = jnp.ravel(jnp.asarray(batched.results))
    for lane, want in zip(got, _serial_results()):
        assert float(lane) == pytest.approx(want, abs=1e-10)


def test_optimization_moves_toward_target():
    """Guards against the trajectory being flat, which a no-op would also give."""
    space = Space(
        {"x": DataAxis(STARTS, wrap=_sigmoid_wrap()), "target": TARGET}, mode="zip"
    )
    final = jnp.ravel(
        jnp.asarray(
            ParallelExecution(
                model=partial(_optimize, max_steps=40), space=space, n_vmap=2, n_pmap=1
            )
            .run()
            .results
        )
    )
    assert jnp.all(jnp.abs(final - TARGET) < jnp.abs(STARTS - TARGET))


# --------------------------------------------------------------- tracing policy


def _traced_jaxpr(max_steps, chunk_size=None):
    def model(start):
        state = {
            "x": SigmoidBoundedParameter(start, low=LOW, high=HIGH),
            "target": TARGET,
        }
        return _optimize(state, max_steps=max_steps, chunk_size=chunk_size)

    return jax.make_jaxpr(lambda d: jax.lax.map(model, d, batch_size=2))(STARTS)


def test_traced_graph_size_is_independent_of_max_steps():
    """The whole point: memory under ParallelExecution must not track max_steps.

    Before the tracing policy the Python step loop unrolled, and this ratio was
    roughly 5x across this range.
    """
    small = len(str(_traced_jaxpr(10)))
    large = len(str(_traced_jaxpr(50)))
    assert small == large


def test_explicit_chunk_size_is_respected_under_trace():
    """An explicit chunk_size is not silently overridden to max_steps."""
    assert len(str(_traced_jaxpr(50, chunk_size=10))) > len(
        str(_traced_jaxpr(50, chunk_size=50))
    )


def test_traced_run_matches_untraced_run():
    """Taking the scan path under trace must not change the result."""
    space = Space(
        {"x": DataAxis(STARTS, wrap=_sigmoid_wrap()), "target": TARGET}, mode="zip"
    )
    traced = jnp.ravel(
        jnp.asarray(
            ParallelExecution(model=_optimize, space=space, n_vmap=2, n_pmap=1)
            .run()
            .results
        )
    )
    for lane, want in zip(traced, _serial_results()):
        assert float(lane) == pytest.approx(want, abs=1e-10)


def test_callback_under_trace_raises():
    def cb(i, diff_state, static_state, fitting_data, aux, loss, grads):
        return False, diff_state, static_state

    def model(start):
        state = {
            "x": SigmoidBoundedParameter(start, low=LOW, high=HIGH),
            "target": TARGET,
        }
        opt = OptaxOptimizer(loss=_loss, optimizer=optax.sgd(0.5), callback=cb)
        final, _ = opt.run(state, max_steps=3)
        return final["x"].constrained_value

    with pytest.raises(ValueError, match="callback cannot run under a trace"):
        jax.make_jaxpr(lambda d: jax.lax.map(model, d, batch_size=2))(STARTS)


# ------------------------------------------------------------ empty diff_state


def test_run_without_parameters_raises():
    """A missing wrap= leaves nothing to optimize; that must not be silent."""
    opt = OptaxOptimizer(loss=_loss, optimizer=optax.sgd(0.5))
    with pytest.raises(ValueError, match="No Parameter leaves to optimize"):
        opt.run({"x": jnp.asarray(1.0), "target": TARGET}, max_steps=3)


def test_run_without_parameters_names_wrap_as_a_cause():
    opt = OptaxOptimizer(loss=_loss, optimizer=optax.sgd(0.5))
    with pytest.raises(ValueError, match="wrap="):
        opt.run({"x": jnp.asarray(1.0), "target": TARGET}, max_steps=3)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_chunk_size_raises(bad):
    opt = OptaxOptimizer(loss=_loss, optimizer=optax.sgd(0.5))
    state = {"x": SigmoidBoundedParameter(1.0, low=LOW, high=HIGH), "target": TARGET}
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        opt.run(state, max_steps=3, chunk_size=bad)
