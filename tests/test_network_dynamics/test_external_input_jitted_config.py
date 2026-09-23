"""A prepared solve function stays jittable when an external input carries a non-array parameter.

`prepare` returns a pure function of its config, so the documented way to run it is under `jax.jit`. Every parameter of an external input went into that config, and a string is not a valid JAX type there, so an input carrying construction metadata raised on its first traced call. `DataInput` carries `interpolation_type`, which made it unusable under `jit` on every ungrouped path; the grouped (heterogeneous) path already partitioned its parameters. The partition must not cost an input its liveness: what stays in the config is still read every step, still differentiable, and still overrides what the input object holds.
"""

import jax
import jax.numpy as jnp
import pytest

import diffrax

from tvboptim.experimental.network_dynamics import DenseGraph, Network, prepare
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import Generic2dOscillator
from tvboptim.experimental.network_dynamics.external_input import ConstantInput, DataInput
from tvboptim.experimental.network_dynamics.solvers import DiffraxSolver, Heun

N_NODES = 2
TIMES = jnp.linspace(0.0, 1.0, 11)
DATA = jnp.sin(2 * jnp.pi * TIMES)

SAVE_AT = jnp.linspace(0.0, 1.0, 21)

# An explicit `saveat` keeps diffrax from padding `ys` to `max_steps` with inf, which no assertion about the trajectory could survive.
SOLVERS = {
    "native": lambda: Heun(),
    "diffrax": lambda: DiffraxSolver(diffrax.Euler(), saveat=diffrax.SaveAt(ts=SAVE_AT)),
}


def _network(external):
    return Network(
        dynamics=Generic2dOscillator(),
        coupling={"instant": LinearCoupling(source="V", G=0.0)},
        graph=DenseGraph(jnp.zeros((N_NODES, N_NODES))),
        external_input={"stimulus": external},
    )


def _prepared(external, solver, on_network):
    """The four ungrouped `prepare` overloads: a Network or bare dynamics, times a native or diffrax solver."""
    if on_network:
        return prepare(_network(external), solver, t0=0.0, t1=1.0, dt=0.05)
    return prepare(
        Generic2dOscillator(),
        solver,
        t0=0.0,
        t1=1.0,
        dt=0.05,
        n_nodes=N_NODES,
        externals={"stimulus": external},
    )


PATHS = [
    pytest.param(solver, on_network, id=f"{'network' if on_network else 'dynamics'}-{kind}")
    for kind, solver in SOLVERS.items()
    for on_network in (True, False)
]


@pytest.mark.parametrize("solver,on_network", PATHS)
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_a_data_input_runs_under_jit(solver, on_network, interpolation):
    """The regression: `TypeError: ... type <class 'str'> ... config['external']['stimulus']['interpolation_type']`."""
    simulate, config = _prepared(
        DataInput(TIMES, DATA, interpolation=interpolation), solver(), on_network
    )
    assert jnp.all(jnp.isfinite(jax.jit(simulate)(config).ys))


@pytest.mark.parametrize("solver,on_network", PATHS)
def test_construction_metadata_is_kept_out_of_the_jitted_config(solver, on_network):
    _, config = _prepared(DataInput(TIMES, DATA), solver(), on_network)
    assert "interpolation_type" not in config.external.stimulus
    assert {"times", "data"} <= set(config.external.stimulus)


@pytest.mark.parametrize("solver,on_network", PATHS)
def test_the_config_still_wins_over_the_input_it_was_prepared_from(solver, on_network):
    """The half that stays in the config is merged over the input's own parameters, not under them: an edited sample array must reach the solver."""
    simulate, config = _prepared(DataInput(TIMES, DATA), solver(), on_network)
    edited = config.copy()
    edited.external.stimulus.data = 5.0 * jnp.ones_like(DATA)
    assert not jnp.allclose(jax.jit(simulate)(config).ys, jax.jit(simulate)(edited).ys)


@pytest.mark.parametrize("solver,on_network", PATHS)
def test_a_live_parameter_stays_differentiable(solver, on_network):
    """The fix must cost an input nothing. `DataInput` is not the witness: `compute` reads `times`/`data` through the interpolation, while `ConstantInput.amplitude` is a plain per-step factor."""
    simulate, config = _prepared(ConstantInput(amplitude=0.3), solver(), on_network)

    def final_state(amplitude):
        driven = config.copy()
        driven.external.stimulus.amplitude = amplitude
        return jnp.sum(jax.jit(simulate)(driven).ys[-1])

    gradient = jax.grad(final_state)(0.3)
    assert jnp.isfinite(gradient)
    assert gradient != 0.0
