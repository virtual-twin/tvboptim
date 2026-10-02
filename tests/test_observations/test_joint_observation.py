"""Bounded shared-stage observation composition."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from docs.advanced.low_pass_monitor import LowPass
from tvboptim.experimental.network_dynamics import prepare, solve
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.solvers import Euler
from tvboptim.observations import (
    Identity,
    JointObservation,
    ObservationOutput,
    PreparedObservation,
    SimulationGrid,
    apply_observation,
    prepare_observation,
)
from tvboptim.observations.tvb_monitors import (
    BalloonWindkesselBold,
    GammaHRFKernel,
    HRFBold,
    SubSampling,
    TemporalAverage,
)


class DrivenDynamics(AbstractDynamics):
    STATE_NAMES = ("signal",)
    INITIAL_STATE = (0.0,)
    DEFAULT_PARAMS = Bunch(rate=1.0)

    def dynamics(self, t, state, params, coupling, external):
        del t, state, coupling, external
        return jnp.asarray([[params.rate]])


@dataclass(frozen=True)
class AlignedMonitor:
    period: float
    alignment_steps: int

    def prepare(self, grid, sample, variable_names):
        return _prepare_aligned(self, grid, sample, variable_names)


def _aligned_update(stride):
    def update(state, block, params):
        del params
        return state, block[stride - 1 :: stride]

    return update


def _prepare_aligned(monitor, grid, sample, variable_names):
    del sample
    stride = round(monitor.period / grid.dt)
    return PreparedObservation(
        params=Bunch(),
        init=lambda params: None,
        update=_aligned_update(stride),
        output=ObservationOutput(
            period=monitor.period,
            first_sample_offset=(
                (
                    grid.dt
                    if grid.first_sample_offset is None
                    else grid.first_sample_offset
                )
                + (stride - 1) * grid.dt
            ),
            variable_names=variable_names,
        ),
        alignment_steps=monitor.alignment_steps,
    )


def _raw_solution(n_steps=13, *, t0=0.0, dt=1.0, n_nodes=1):
    values = jnp.arange(1, n_steps + 1, dtype=jnp.float32)
    ys = jnp.broadcast_to(values[:, None, None], (n_steps, 1, n_nodes))
    return NativeSolution(
        ts=t0 + jnp.arange(1, n_steps + 1) * dt,
        ys=ys,
        dt=dt,
        variable_names=("signal",),
    )


def _reference_valid_hrf(values, dt, monitor, history=None):
    kernel_samples = int(jnp.ceil(monitor.kernel.duration / dt))
    kernel_t = jnp.linspace(0.0, monitor.kernel.duration, kernel_samples)
    kernel = monitor.kernel(kernel_t, dt)
    if history is None:
        history = jnp.zeros((kernel_samples,) + values.shape[1:], values.dtype)
    else:
        history = jnp.asarray(history, dtype=values.dtype)
        if history.shape[0] < kernel_samples:
            history = jnp.concatenate(
                [
                    jnp.zeros(
                        (kernel_samples - history.shape[0],) + values.shape[1:],
                        values.dtype,
                    ),
                    history,
                ]
            )
        history = history[-kernel_samples:]
    signal = jnp.concatenate([history, values])
    stride = round(monitor.period / dt)
    starts = range(stride, signal.shape[0] - kernel_samples + 1, stride)
    convolved = jnp.stack(
        [
            jnp.tensordot(kernel[::-1], signal[i : i + kernel_samples], axes=(0, 0))
            for i in starts
        ]
    )
    return monitor.k_1 * monitor.V_0 * (convolved - 1.0)


def test_identity_preserves_grid_values_names_and_dynamic_offsets():
    solution = _raw_solution(5, t0=10.0, dt=2.0, n_nodes=2)

    @jax.jit
    def run(current, offset):
        return Identity()(current, t_offset=offset)

    result = run(solution, jnp.asarray(3.0))
    assert jnp.array_equal(result.ys, solution.ys)
    assert jnp.array_equal(result.ts, solution.ts + 3.0)
    assert result.dt == 2.0
    assert result.variable_names == ("signal",)


def test_joint_independent_branches_use_lcm_alignment_and_independent_tails():
    monitor = JointObservation(
        outputs={
            "six": SubSampling(period=6.0),
            "ten": SubSampling(period=10.0),
        }
    )
    result = monitor(_raw_solution(37))
    assert result["six"].ys.shape == (6, 1, 1)
    assert result["ten"].ys.shape == (3, 1, 1)
    assert jnp.array_equal(result["six"].ts, jnp.arange(6.0, 37.0, 6.0))
    assert jnp.array_equal(result["ten"].ts, jnp.arange(10.0, 31.0, 10.0))

    with pytest.raises(ValueError, match="multiple of 30 steps"):
        prepare(
            DrivenDynamics(),
            Euler(block_size=10),
            t1=37.0,
            dt=1.0,
            observe=monitor,
        )
    model, config = prepare(
        DrivenDynamics(),
        Euler(block_size=30),
        t1=37.0,
        dt=1.0,
        observe=monitor,
    )
    online = model(config)
    assert online["six"].ys.shape[0] == 6
    assert online["ten"].ys.shape[0] == 3


def test_joint_includes_stronger_child_alignment_and_names_invalid_branch():
    monitor = JointObservation(
        outputs={
            "aligned": AlignedMonitor(period=1.0, alignment_steps=4),
            "six": SubSampling(period=6.0),
        }
    )
    with pytest.raises(ValueError, match="multiple of 12 steps"):
        prepare(
            DrivenDynamics(),
            Euler(block_size=6),
            t1=13.0,
            dt=1.0,
            observe=monitor,
        )
    model, config = prepare(
        DrivenDynamics(),
        Euler(block_size=12),
        t1=13.0,
        dt=1.0,
        observe=monitor,
    )
    assert model(config)["aligned"].ys.shape[0] == 13

    invalid = JointObservation(
        outputs={"bad": AlignedMonitor(period=2.0, alignment_steps=3)}
    )
    with pytest.raises(ValueError, match="output 'bad' is invalid"):
        apply_observation(invalid, _raw_solution(4))


def test_joint_returns_independently_shaped_empty_named_outputs():
    monitor = JointObservation(
        outputs={
            "four": SubSampling(period=4.0),
            "eight": SubSampling(period=8.0),
        }
    )
    posthoc = monitor(_raw_solution(3))
    online = solve(
        DrivenDynamics(),
        Euler(block_size=8),
        t1=3.0,
        dt=1.0,
        observe=monitor,
    )
    for result in (posthoc, online):
        assert result["four"].ys.shape == (0, 1, 1)
        assert result["eight"].ys.shape == (0, 1, 1)
        assert result["four"].ts.shape == (0,)
        assert result["eight"].ts.shape == (0,)


def test_named_joint_results_match_by_key_through_jit_vmap_and_insertion_order():
    solution = _raw_solution(8)
    left = JointObservation(
        outputs={"coarse": SubSampling(period=4.0), "raw": Identity()}
    )
    right = JointObservation(
        outputs={"raw": Identity(), "coarse": SubSampling(period=4.0)}
    )

    @jax.jit
    def run(monitor_input, offset):
        return left(monitor_input, t_offset=offset)

    actual = run(solution, jnp.asarray(3.0))
    expected = right(solution, t_offset=3.0)
    assert set(actual) == set(expected) == {"coarse", "raw"}
    for name in actual:
        assert jnp.array_equal(actual[name].ys, expected[name].ys)
        assert jnp.array_equal(actual[name].ts, expected[name].ts)

    batched_values = jnp.stack([solution.ys, 2.0 * solution.ys])

    def observe_values(values):
        current = NativeSolution(
            ts=solution.ts,
            ys=values,
            dt=solution.dt,
            variable_names=solution.variable_names,
        )
        result = left(current)
        return {name: item.ys for name, item in result.items()}

    batched = jax.vmap(observe_values)(batched_values)
    assert jnp.array_equal(batched["raw"][1], 2.0 * batched["raw"][0])
    assert jnp.array_equal(batched["coarse"][1], 2.0 * batched["coarse"][0])


def test_shared_average_identity_and_hrf_match_online_posthoc_and_reference():
    history = NativeSolution(
        ts=jnp.asarray([-4.5, -0.5, 3.5]),
        ys=jnp.asarray([2.0, 4.0, 6.0]).reshape(3, 1, 1),
        dt=4.0,
        variable_names=("signal",),
    )
    hrf = HRFBold(
        period=8.0,
        downsample=Identity(),
        kernel=GammaHRFKernel(duration=8.0),
        history=history,
    )
    monitor = JointObservation(
        preprocess=TemporalAverage(period=4.0),
        outputs={"neural": Identity(), "bold": hrf},
    )
    raw = solve(DrivenDynamics(), Euler(block_size=8), t0=5.0, t1=18.0, dt=1.0)
    posthoc = monitor(raw)
    online = solve(
        DrivenDynamics(),
        Euler(block_size=8),
        t0=5.0,
        t1=18.0,
        dt=1.0,
        observe=monitor,
    )

    averaged = raw.ys[:12].reshape(3, 4, 1, 1).mean(axis=1)
    expected_bold = _reference_valid_hrf(averaged, 4.0, hrf, history.ys)
    for actual in (posthoc, online):
        assert jnp.allclose(actual["neural"].ys, averaged)
        assert jnp.allclose(actual["bold"].ys, expected_bold)
        assert jnp.array_equal(actual["neural"].ts, jnp.asarray([7.5, 11.5, 15.5]))
        assert jnp.array_equal(actual["bold"].ts, jnp.asarray([13.0]))
        assert actual["neural"].variable_names == ("signal",)
        assert actual["bold"].variable_names == ("BOLD(signal)",)


def test_trainable_shared_low_pass_owns_namespace_and_adds_branch_gradients():
    monitor = JointObservation(
        preprocess=LowPass(period=2.0, tau=3.0),
        outputs={"filtered": Identity(), "slow": SubSampling(period=4.0)},
    )
    model, config = prepare(
        DrivenDynamics(),
        Euler(block_size=4),
        t1=8.0,
        dt=1.0,
        observe=monitor,
    )
    assert config.monitor.preprocess == {"tau": 3.0}
    assert set(config.monitor.outputs) == {"filtered", "slow"}

    def loss(tau):
        current = config.copy()
        current.monitor.preprocess["tau"] = tau
        result = model(current)
        return result["filtered"].ys.sum() + result["slow"].ys.sum()

    raw = solve(DrivenDynamics(), Euler(block_size=4), t1=8.0, dt=1.0).ys

    def reference_loss(tau):
        decay = jnp.exp(-1.0 / tau)

        def step(state, value):
            state = decay * state + (1.0 - decay) * value
            return state, state

        _, filtered = jax.lax.scan(step, jnp.zeros_like(raw[0]), raw)
        shared = filtered[1::2]
        return shared.sum() + shared[1::2].sum()

    tau = jnp.asarray(3.0)
    assert jnp.allclose(loss(tau), reference_loss(tau))
    assert jnp.allclose(jax.grad(loss)(tau), jax.grad(reference_loss)(tau))
    first = model(config)
    second = model(config)
    assert jnp.array_equal(first["filtered"].ys, second["filtered"].ys)


def test_joint_truncation_matches_independently_detached_combined_carry():
    from tvboptim.experimental.network_dynamics.solve import _run_observed_scan

    monitor = JointObservation(
        preprocess=LowPass(period=1.0, tau=2.0),
        outputs={
            "filtered": Identity(),
            "bold": HRFBold(
                period=4.0,
                downsample=Identity(),
                kernel=GammaHRFKernel(duration=4.0),
            ),
        },
    )
    n_steps = 8
    window_size = 4
    solver = Euler(block_size=4, grad_horizon=window_size)
    scan_inputs = jnp.arange(n_steps, dtype=jnp.float32)
    simulation_state0 = jnp.full((1, 1), 0.2, dtype=jnp.float32)
    prepared = prepare_observation(
        monitor,
        SimulationGrid(t0=0.0, dt=1.0, n_steps=n_steps),
        jax.ShapeDtypeStruct((1, 1), jnp.float32),
        ("drive",),
    )

    def run(rate, tau, volume, explicit_reference):
        params = prepared.params.copy()
        params.preprocess["tau"] = tau
        params.outputs.bold.V_0 = volume

        def op(state, _time):
            next_state = state + rate
            return next_state, next_state

        if not explicit_reference:
            _, chunks = _run_observed_scan(
                op,
                simulation_state0,
                scan_inputs,
                n_steps,
                solver,
                prepared,
                params,
                window_size=window_size,
            )
        else:
            observation_state = prepared.init(params)
            carry = (simulation_state0, observation_state)
            output_chunks = {name: [] for name in prepared.output}
            for start in range(0, n_steps, window_size):
                simulation_state, observation_state = jax.lax.stop_gradient(carry)
                simulation_state, raw = jax.lax.scan(
                    op,
                    simulation_state,
                    scan_inputs[start : start + window_size],
                )
                observation_state, chunk = prepared.update(
                    observation_state, raw, params
                )
                carry = (simulation_state, observation_state)
                for name in output_chunks:
                    output_chunks[name].append(chunk[name])
            chunks = {
                name: jnp.concatenate(parts) for name, parts in output_chunks.items()
            }
        loss = sum(jnp.sum(chunk**2) for chunk in chunks.values())
        return loss, chunks

    arguments = (jnp.asarray(0.03), jnp.asarray(2.0), jnp.asarray(0.02))
    actual = jax.value_and_grad(
        lambda rate, tau, volume: run(rate, tau, volume, False),
        argnums=(0, 1, 2),
        has_aux=True,
    )(*arguments)
    expected = jax.value_and_grad(
        lambda rate, tau, volume: run(rate, tau, volume, True),
        argnums=(0, 1, 2),
        has_aux=True,
    )(*arguments)

    (actual_loss, actual_chunks), actual_gradients = actual
    (expected_loss, expected_chunks), expected_gradients = expected
    assert jnp.allclose(actual_loss, expected_loss)
    for name in actual_chunks:
        assert jnp.allclose(actual_chunks[name], expected_chunks[name])
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert jnp.allclose(actual_gradient, expected_gradient)


@pytest.mark.parametrize("injected", [False, True])
def test_stochastic_joint_matches_same_raw_block_grid_and_gradient(injected):
    monitor = JointObservation(
        preprocess=LowPass(period=1.0, tau=3.0),
        outputs={"filtered": Identity(), "coarse": SubSampling(period=4.0)},
    )
    noise = AdditiveNoise(sigma=0.2, key=jax.random.key(17))
    solver = Euler(block_size=4)
    kwargs = dict(t1=9.0, dt=1.0, noise=noise)
    raw_fn, raw_config = prepare(DrivenDynamics(), solver, **kwargs)
    observed_fn, observed_config = prepare(
        DrivenDynamics(), solver, observe=monitor, **kwargs
    )
    if injected:
        samples = jnp.arange(9, dtype=jnp.float32).reshape(9, 1, 1) / 100.0
        raw_config._internal.noise_samples = samples
        observed_config._internal.noise_samples = samples

    expected = monitor(raw_fn(raw_config))
    actual = observed_fn(observed_config)
    for name in actual:
        assert jnp.array_equal(actual[name].ys, expected[name].ys)
        assert jnp.array_equal(actual[name].ts, expected[name].ts)

    def loss(model, config, rate, posthoc):
        current = config.copy()
        current.dynamics.rate = rate
        result = model(current)
        result = monitor(result) if posthoc else result
        return sum(jnp.mean(item.ys**2) for item in result.values())

    rate = jnp.asarray(1.1)
    expected_gradient = jax.grad(lambda value: loss(raw_fn, raw_config, value, True))(
        rate
    )
    actual_gradient = jax.grad(
        lambda value: loss(observed_fn, observed_config, value, False)
    )(rate)
    assert jnp.allclose(actual_gradient, expected_gradient)


def test_joint_hrf_warm_history_gradient_matches_independent_reference():
    raw = solve(DrivenDynamics(), Euler(block_size=4), t1=8.0, dt=1.0)
    averaged = raw.ys.reshape(4, 2, 1, 1).mean(axis=1)

    def hrf(history):
        return HRFBold(
            period=4.0,
            downsample=Identity(),
            kernel=GammaHRFKernel(duration=4.0),
            history=history,
        )

    def posthoc_loss(history):
        result = JointObservation(
            preprocess=TemporalAverage(period=2.0),
            outputs={"bold": hrf(history)},
        )(raw)
        return jnp.sum(result["bold"].ys)

    def online_loss(history):
        result = solve(
            DrivenDynamics(),
            Euler(block_size=4),
            t1=8.0,
            dt=1.0,
            observe=JointObservation(
                preprocess=TemporalAverage(period=2.0),
                outputs={"bold": hrf(history)},
            ),
        )
        return jnp.sum(result["bold"].ys)

    def reference_loss(history):
        return jnp.sum(_reference_valid_hrf(averaged, 2.0, hrf(history), history))

    def solution_history_loss(history):
        history_solution = NativeSolution(
            ts=jnp.arange(-3.0, 7.0, 2.0),
            ys=history,
            dt=2.0,
            variable_names=("signal",),
        )
        result = JointObservation(
            preprocess=TemporalAverage(period=2.0),
            outputs={"bold": hrf(history_solution)},
        )(raw)
        return jnp.sum(result["bold"].ys)

    history = jnp.arange(1, 6, dtype=jnp.float32).reshape(5, 1, 1)
    expected = jax.value_and_grad(reference_loss)(history)
    posthoc = jax.value_and_grad(posthoc_loss)(history)
    online = jax.value_and_grad(online_loss)(history)
    solution_history = jax.value_and_grad(solution_history_loss)(history)
    assert jnp.allclose(posthoc[0], expected[0])
    assert jnp.allclose(online[0], expected[0])
    assert jnp.allclose(posthoc[1], expected[1])
    assert jnp.allclose(online[1], expected[1])
    assert jnp.allclose(solution_history[0], expected[0])
    assert jnp.allclose(solution_history[1], expected[1])


@pytest.mark.parametrize("dt", [2.0, 8.0])
def test_hrf_identity_resolves_effective_grid_and_warm_history(dt):
    solution = _raw_solution(8, dt=dt)
    period = 2 * dt
    history = jnp.arange(1, 7, dtype=jnp.float32).reshape(6, 1, 1)
    monitor = HRFBold(
        period=period,
        downsample=Identity(),
        kernel=GammaHRFKernel(duration=16.0),
        history=history,
    )
    result = monitor(solution)
    assert jnp.allclose(
        result.ys, _reference_valid_hrf(solution.ys, dt, monitor, history)
    )

    raw = solve(DrivenDynamics(), Euler(block_size=2), t1=8 * dt, dt=dt)
    expected_online = monitor(raw)
    online = solve(
        DrivenDynamics(),
        Euler(block_size=2),
        t1=8 * dt,
        dt=dt,
        observe=monitor,
    )
    joint = solve(
        DrivenDynamics(),
        Euler(block_size=2),
        t1=8 * dt,
        dt=dt,
        observe=JointObservation(outputs={"bold": monitor}),
    )["bold"]
    assert jnp.allclose(online.ys, expected_online.ys)
    assert jnp.allclose(joint.ys, expected_online.ys)

    history_solution = NativeSolution(
        ts=jnp.arange(1, 7) * dt,
        ys=history,
        dt=dt,
        variable_names=("signal",),
    )
    solution_history_result = HRFBold(
        period=period,
        downsample=Identity(),
        kernel=GammaHRFKernel(duration=16.0),
        history=history_solution,
    )(solution)
    assert jnp.allclose(solution_history_result.ys, result.ys)

    mismatched = NativeSolution(
        ts=history_solution.ts,
        ys=history,
        dt=dt / 2,
        variable_names=("signal",),
    )
    with pytest.raises(ValueError, match="must resolve.*interval"):
        HRFBold(
            period=period,
            downsample=Identity(),
            kernel=GammaHRFKernel(duration=16.0),
            history=mismatched,
        )(solution)


def test_balloon_windkessel_accepts_explicit_identity():
    solution = _raw_solution(8, dt=2.0)
    result = BalloonWindkesselBold(period=4.0, dt_bw=1.0, downsample=Identity())(
        solution
    )
    assert result.ys.shape == (4, 1, 1)
    assert result.dt == 4.0


class DropsTailSample:
    """A shared stage that loses one sample from a three-step tail."""

    def prepare(self, grid, sample, variable_names):
        del sample

        def update(state, block, params):
            del params
            return state, block[1:] if block.shape[0] == 3 else block

        return PreparedObservation(
            params=Bunch(),
            init=lambda params: None,
            update=update,
            output=grid.output(grid.dt, variable_names=variable_names),
        )


@pytest.mark.parametrize(
    "observe",
    [
        DropsTailSample(),
        JointObservation(
            preprocess=DropsTailSample(), outputs={"sampled": SubSampling(period=2.0)}
        ),
    ],
    ids=["standalone", "joint_shared_stage"],
)
def test_shared_stage_emission_count_is_validated_for_every_block_length(observe):
    with pytest.raises(ValueError) as info:
        solve(DrivenDynamics(), Euler(block_size=4), t1=7.0, dt=1.0, observe=observe)
    messages = f"{info.value} {info.value.__cause__}"
    assert "returned 2 samples for 3 input steps" in messages


def test_joint_rejects_nesting_and_mapping_valued_stages():
    leaf = JointObservation(outputs={"value": Identity()})
    with pytest.raises(ValueError, match="branches cannot be nested"):
        JointObservation(outputs={"nested": leaf})
    with pytest.raises(ValueError, match="preprocess cannot be another joint"):
        JointObservation(preprocess=leaf, outputs={"value": Identity()})

    def update(state, block, params):
        del params
        return state, {"value": block}

    @dataclass(frozen=True)
    class MappingMonitor:
        def prepare(self, grid, sample, variable_names):
            del sample
            return PreparedObservation(
                params={},
                init=lambda params: None,
                update=update,
                output={
                    "value": ObservationOutput(
                        grid.dt,
                        grid.dt
                        if grid.first_sample_offset is None
                        else grid.first_sample_offset,
                        variable_names,
                    )
                },
            )

    with pytest.raises(ValueError, match="preprocess must produce one time series"):
        apply_observation(
            JointObservation(
                preprocess=MappingMonitor(), outputs={"value": Identity()}
            ),
            _raw_solution(4),
        )
    with pytest.raises(ValueError, match="output 'mapped'.*one time series"):
        apply_observation(
            JointObservation(outputs={"mapped": MappingMonitor()}),
            _raw_solution(4),
        )
