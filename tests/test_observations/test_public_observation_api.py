"""Public temporal-observation authoring contract."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from docs.advanced.low_pass_monitor import LowPass
from tvboptim.execution import ParallelExecution
from tvboptim.experimental.network_dynamics import prepare, solve
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.solvers import Euler
from tvboptim.observations import (
    Identity,
    JointObservation,
    ObservationOutput,
    PreparedObservation,
    SampledMonitor,
    SimulationGrid,
    StreamingMonitor,
    apply_observation,
    prepare_observation,
    sampling_stride,
)
from tvboptim.types import DataAxis, Space

PUBLIC_AUTHORING_NAMES = (
    "SimulationGrid",
    "ObservationOutput",
    "PreparedObservation",
    "StreamingMonitor",
    "SampledMonitor",
    "prepare_observation",
    "apply_observation",
    "sampling_stride",
    "Identity",
    "JointObservation",
)
BUILTIN_TEMPORAL_MONITOR_NAMES = (
    "SubSampling",
    "TemporalAverage",
    "HRFBold",
    "BalloonWindkesselBold",
)


def test_public_authoring_contract_is_exported_from_observations_facade():
    import tvboptim.observations as observations

    expected = {
        "SimulationGrid": SimulationGrid,
        "ObservationOutput": ObservationOutput,
        "PreparedObservation": PreparedObservation,
        "StreamingMonitor": StreamingMonitor,
        "SampledMonitor": SampledMonitor,
        "prepare_observation": prepare_observation,
        "apply_observation": apply_observation,
        "sampling_stride": sampling_stride,
        "Identity": Identity,
        "JointObservation": JointObservation,
    }
    assert set(PUBLIC_AUTHORING_NAMES) <= set(observations.__all__)
    assert {name: getattr(observations, name) for name in expected} == expected


def test_builtin_temporal_monitors_are_exported_from_their_public_module():
    from tvboptim.observations import tvb_monitors

    assert set(BUILTIN_TEMPORAL_MONITOR_NAMES) <= set(tvb_monitors.__all__)
    assert all(
        getattr(tvb_monitors, name) is not None
        for name in BUILTIN_TEMPORAL_MONITOR_NAMES
    )


@pytest.mark.parametrize(
    "modules",
    [
        (
            "tvboptim.experimental.network_dynamics.solve",
            "tvboptim.observations",
        ),
        (
            "tvboptim.observations",
            "tvboptim.experimental.network_dynamics.solve",
        ),
    ],
)
def test_public_authoring_contract_imports_in_either_fresh_process_order(modules):
    root = Path(__file__).resolve().parents[2]
    pythonpath = str(root / "src")
    if existing := os.environ.get("PYTHONPATH"):
        pythonpath = os.pathsep.join((pythonpath, existing))
    body = (
        "import importlib\n"
        f"importlib.import_module({modules[0]!r})\n"
        f"importlib.import_module({modules[1]!r})\n"
        "import tvboptim.observations as observations\n"
        f"names = {PUBLIC_AUTHORING_NAMES!r}\n"
        "assert set(names) <= set(observations.__all__)\n"
        "assert all(getattr(observations, name) is not None for name in names)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", body],
        cwd=root,
        env=os.environ | {"PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@dataclass(frozen=True)
class MalformedMonitor:
    kind: str

    def prepare(self, grid, sample, variable_names):
        return _prepare_malformed(self, grid, sample, variable_names)


@dataclass(frozen=True)
class TailSensitiveMonitor:
    kind: str = "valid"

    def prepare(self, grid, sample, variable_names):
        return _prepare_tail_sensitive(self, grid, sample, variable_names)


def _malformed_init(kind):
    def init(params):
        del params
        if kind == "init":
            raise ValueError("broken initializer")
        return None

    return init


def _malformed_update(kind):
    return lambda state, block, params: _malformed_emission(kind, state, block)


def _malformed_emission(kind, state, block):
    if kind == "bad_return":
        return block
    if kind == "rank":
        return state, block[:, 0, :]
    if kind == "nodes":
        return state, block[:, :, :1]
    if kind == "state":
        return (state,), block
    if kind == "keys":
        return state, {"left": block}
    return state, block


def _tail_sensitive_init(kind):
    def init(params):
        if kind == "init_shape":
            return jnp.zeros(jnp.shape(params["gain"]), dtype=jnp.float32)
        return jnp.asarray(0.0, dtype=jnp.float32)

    return init


def _tail_sensitive_update(kind):
    return lambda state, block, params: _tail_sensitive_emission(
        kind, state, block, params
    )


def _tail_sensitive_emission(kind, state, block, params):
    values = block.astype(params["gain"].dtype) * params["gain"]
    if block.shape[0] < 4:
        if kind == "count" and params["gain"].ndim:
            values = jnp.repeat(values, 1 + params["gain"].ndim, axis=0)
        elif kind == "channels" and params["gain"].ndim:
            values = jnp.repeat(values, 1 + params["gain"].ndim, axis=1)
        elif kind == "state_shape" and params["gain"].ndim:
            state = jnp.reshape(params["gain"], (-1,))
        elif kind == "state_dtype":
            state = state.astype(jnp.result_type(state.dtype, params["gain"].dtype))
    return state, values


def _prepare_malformed(monitor, grid, sample, variable_names):
    del sample
    if monitor.kind == "keys":
        output = {
            "left": ObservationOutput(grid.dt, grid.dt, variable_names),
            "right": ObservationOutput(grid.dt, grid.dt, variable_names),
        }
    else:
        output = ObservationOutput(
            1.5 * grid.dt if monitor.kind == "period" else grid.dt,
            grid.dt,
            ("one", "two") if monitor.kind == "names" else variable_names,
        )
    return PreparedObservation(
        params={},
        init=_malformed_init(monitor.kind),
        update=_malformed_update(monitor.kind),
        output=output,
    )


def _prepare_tail_sensitive(monitor, grid, sample, variable_names):
    del sample
    input_offset = (
        grid.dt if grid.first_sample_offset is None else grid.first_sample_offset
    )
    return PreparedObservation(
        params={"gain": jnp.asarray(1.0, dtype=jnp.float16)},
        init=_tail_sensitive_init(monitor.kind),
        update=_tail_sensitive_update(monitor.kind),
        output=ObservationOutput(grid.dt, input_offset, variable_names),
    )


class DrivenDynamics(AbstractDynamics):
    STATE_NAMES = ("signal",)
    INITIAL_STATE = (0.0,)
    DEFAULT_PARAMS = Bunch(rate=1.0)

    def dynamics(self, t, state, params, coupling, external):
        del t, state, coupling, external
        return jnp.asarray([[params.rate]])


def _reference_low_pass(values, dt, tau, stride):
    decay = jnp.exp(-dt / tau)

    def step(filtered, sample):
        filtered = decay * filtered + (1.0 - decay) * sample
        return filtered, filtered

    _, filtered = jax.lax.scan(step, jnp.zeros_like(values[0]), values)
    return filtered[stride - 1 :: stride]


def test_external_monitor_agrees_online_posthoc_and_with_independent_reference():
    monitor = LowPass(period=2.0, tau=3.0)
    raw = solve(DrivenDynamics(), Euler(block_size=4), t1=9.0, dt=1.0)
    posthoc = monitor(raw)
    online = solve(
        DrivenDynamics(),
        Euler(block_size=4),
        t1=9.0,
        dt=1.0,
        observe=monitor,
    )
    reference = _reference_low_pass(raw.ys, 1.0, 3.0, 2)

    assert jnp.allclose(posthoc.ys, reference)
    assert jnp.allclose(online.ys, reference)
    assert jnp.array_equal(online.ts, jnp.asarray([2.0, 4.0, 6.0, 8.0]))
    assert online.variable_names == ("signal",)


def test_external_monitor_live_parameter_is_owned_by_config_and_differentiates():
    model, config = prepare(
        DrivenDynamics(),
        Euler(block_size=4),
        t1=8.0,
        dt=1.0,
        observe=LowPass(period=2.0, tau=3.0),
    )
    assert config.monitor == {"tau": 3.0}

    def loss(tau):
        current = config.copy()
        current.monitor["tau"] = tau
        return jnp.sum(model(current).ys)

    gradient = jax.grad(loss)(jnp.asarray(3.0))
    assert jnp.isfinite(gradient)
    assert gradient != 0.0

    invalid = config.copy()
    invalid.monitor["tau"] = jnp.ones((3,))
    with pytest.raises(ValueError, match="current monitor parameter shape/dtype"):
        model(invalid)


def test_external_monitor_live_parameter_supports_space_sweeps():
    model, config = prepare(
        DrivenDynamics(),
        Euler(block_size=4),
        t1=8.0,
        dt=1.0,
        observe=LowPass(period=2.0, tau=3.0),
    )
    swept = config.copy()
    swept.monitor["tau"] = DataAxis(jnp.asarray([2.0, 4.0]))
    values = jnp.asarray(
        ParallelExecution(
            lambda current: model(current).ys,
            Space(swept),
            n_vmap=1,
            n_pmap=1,
        ).run()
    )

    expected = []
    for tau in (2.0, 4.0):
        current = config.copy()
        current.monitor["tau"] = tau
        expected.append(model(current).ys)
    assert jnp.allclose(values, jnp.stack(expected))


def test_online_monitor_rejects_changed_input_signature_at_trace_time():
    model, config = prepare(
        DrivenDynamics(),
        Euler(block_size=2),
        t1=4.0,
        dt=1.0,
        observe=LowPass(period=2.0),
    )
    changed = config.copy()
    changed.initial_state = jnp.zeros((1, 2), dtype=config.initial_state.dtype)
    with pytest.raises(ValueError, match="Prepare the model again"):
        model(changed)


def test_external_monitor_posthoc_supports_jit_dynamic_origin_offset_and_tau():
    solution = NativeSolution(
        ts=jnp.arange(11.0, 19.0),
        ys=jnp.arange(8.0).reshape(8, 1, 1),
        dt=1.0,
        variable_names=None,
    )

    @jax.jit
    def run(current, tau, t_offset):
        return LowPass(period=2.0, tau=tau)(current, t_offset=t_offset)

    result = run(solution, jnp.asarray(3.0), jnp.asarray(7.0))
    expected = _reference_low_pass(solution.ys, 1.0, 3.0, 2)
    assert jnp.allclose(result.ys, expected)
    assert jnp.array_equal(result.ts, jnp.asarray([19.0, 21.0, 23.0, 25.0]))
    assert result.variable_names is None


def test_common_validation_reports_wrong_external_emission_count():
    def update(state, block, params):
        del params
        return state, block

    @dataclass(frozen=True)
    class BrokenMonitor:
        period: float = 2.0

        def prepare(self, grid, sample, variable_names):
            del grid, sample
            return PreparedObservation(
                params={},
                init=lambda params: None,
                update=update,
                output=ObservationOutput(2.0, 2.0, variable_names),
            )

    solution = NativeSolution(
        ts=jnp.arange(1.0, 5.0),
        ys=jnp.zeros((4, 1, 1)),
        dt=1.0,
        variable_names=("signal",),
    )
    try:
        apply_observation(BrokenMonitor(), solution)
    except ValueError as exc:
        assert "returned 4 samples for 4 input steps" in str(exc)
        assert "Expected 2 completed-period samples" in str(exc)
    else:
        raise AssertionError("Broken monitor unexpectedly passed validation")


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("init", "init failed abstract validation"),
        ("bad_return", r"update must return \(state, samples\)"),
        ("rank", "rank-three"),
        ("nodes", "changed the node axis"),
        ("state", "preserve state structure"),
        ("keys", "output keys must exactly match"),
        ("period", "integer multiple"),
        ("names", "declares 2 variable names"),
    ],
)
def test_common_validation_rejects_malformed_external_recipes(kind, message):
    solution = NativeSolution(
        ts=jnp.arange(1.0, 5.0),
        ys=jnp.zeros((4, 1, 2)),
        dt=1.0,
        variable_names=("signal",),
    )
    with pytest.raises(ValueError, match=message):
        apply_observation(MalformedMonitor(kind), solution)


def _tail_sensitive_model(monitor, solver=None, n_steps=9):
    return prepare(
        DrivenDynamics(),
        Euler(block_size=4) if solver is None else solver,
        t1=float(n_steps),
        dt=1.0,
        observe=monitor,
    )


def _with_monitor_gain(config, gain, *, output=None):
    current = config.copy()
    params = current.monitor if output is None else current.monitor.outputs[output]
    params["gain"] = gain
    return current


def test_live_parameter_tail_output_count_is_validated_before_scan():
    model, config = _tail_sensitive_model(TailSensitiveMonitor("count"))
    current = _with_monitor_gain(config, jnp.ones((1,), dtype=jnp.float32))
    with pytest.raises(ValueError, match=r"output structure or shape.*block length 1"):
        model(current)


@pytest.mark.parametrize("kind", ["channels", "state_shape"])
def test_live_parameter_tail_shape_violations_are_validated_before_scan(kind):
    model, config = _tail_sensitive_model(TailSensitiveMonitor(kind))
    current = _with_monitor_gain(config, jnp.ones((1,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="block length 1"):
        model(current)


def test_live_parameter_init_state_shape_is_validated_before_scan():
    model, config = _tail_sensitive_model(TailSensitiveMonitor("init_shape"))
    current = _with_monitor_gain(config, jnp.ones((1,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="init changed state structure or shape"):
        model(current)


def test_live_parameter_tail_state_dtype_is_validated_before_scan():
    model, config = _tail_sensitive_model(TailSensitiveMonitor("state_dtype"))
    with jax.enable_x64():
        current = _with_monitor_gain(config, jnp.asarray(1.0, dtype=jnp.float64))
        with pytest.raises(ValueError, match=r"preserve state.*block length 1"):
            model(current)


def test_live_parameter_tail_validation_covers_named_joint_branch():
    monitor = JointObservation(outputs={"bad": TailSensitiveMonitor("count")})
    model, config = _tail_sensitive_model(monitor)
    current = _with_monitor_gain(
        config, jnp.ones((1,), dtype=jnp.float32), output="bad"
    )
    with pytest.raises(
        ValueError, match=r"JointObservation.update changed.*block length 1"
    ):
        model(current)


def test_live_parameter_tail_validation_uses_gradient_horizon_schedule():
    with pytest.warns(UserWarning, match="without block_size"):
        model, config = _tail_sensitive_model(
            TailSensitiveMonitor("count"),
            solver=Euler(grad_horizon=4),
        )
    current = _with_monitor_gain(config, jnp.ones((1,), dtype=jnp.float32))
    with pytest.raises(ValueError, match=r"output structure or shape.*block length 1"):
        model(current)


@pytest.mark.parametrize("n_steps", [0, 1])
def test_valid_live_parameter_promotion_supports_empty_and_short_runs(n_steps):
    model, config = _tail_sensitive_model(TailSensitiveMonitor(), n_steps=n_steps)
    current = _with_monitor_gain(config, jnp.asarray([1.5], dtype=jnp.float32))
    result = model(current)
    assert result.ys.shape == (n_steps, 1, 1)
    assert result.ys.dtype == jnp.float32
