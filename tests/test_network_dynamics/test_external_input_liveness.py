"""External input parameters stay live on the config after prepare().

A prepared input must read its parameters at every step rather than freezing
them at prepare() time. Freezing is silent: the trajectory still looks
plausible, but config edits are ignored, gradients are zero and vmap returns
one trajectory for every batch element.
"""

import jax
import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics import DenseGraph, Network, prepare
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import Generic2dOscillator
from tvboptim.experimental.network_dynamics.external_input import (
    ConstantInput,
    DataInput,
    PulseInput,
    PulseTrainInput,
    RampInput,
    SineInput,
)
from tvboptim.experimental.network_dynamics.solvers import Heun

N_NODES = 2
TIMES = jnp.linspace(0.0, 1.0, 11)
DATA = jnp.sin(2 * jnp.pi * TIMES)

# (label, factory, leaf edited on the config, two distinguishable values)
INPUTS = [
    ("constant", lambda: ConstantInput(amplitude=0.3), "amplitude", 0.3, 0.9),
    ("sine", lambda: SineInput(frequency=3.0), "amplitude", 1.0, 2.5),
    ("pulse", lambda: PulseInput(onset=0.1, duration=0.5), "amplitude", 1.0, 3.0),
    ("pulse_train", lambda: PulseTrainInput(frequency=3.0), "amplitude", 1.0, 3.0),
    ("ramp", lambda: RampInput(t_end=1.0), "amplitude", 1.0, 4.0),
    ("data", lambda: DataInput(TIMES, DATA), "data", DATA, 5.0 * jnp.ones_like(TIMES)),
    (
        "data_cubic",
        lambda: DataInput(TIMES, DATA, interpolation="cubic"),
        "data",
        DATA,
        5.0 * jnp.ones_like(TIMES),
    ),
]
IDS = [case[0] for case in INPUTS]


def _prepared(external):
    network = Network(
        dynamics=Generic2dOscillator(),
        coupling={"instant": LinearCoupling(source="V", G=0.0)},
        graph=DenseGraph(jnp.zeros((N_NODES, N_NODES))),
        external_input={"stimulus": external},
    )
    return prepare(network, Heun(), t0=0.0, t1=1.0, dt=0.05)


def _with(config, leaf, value):
    varied = config.copy()
    setattr(varied.external.stimulus, leaf, value)
    return varied


@pytest.mark.parametrize("_label,factory,leaf,base,other", INPUTS, ids=IDS)
def test_external_parameter_is_live_on_the_config(_label, factory, leaf, base, other):
    simulate, config = _prepared(factory())

    baseline = simulate(_with(config, leaf, base)).ys
    edited = simulate(_with(config, leaf, other)).ys
    assert not jnp.allclose(baseline, edited), "config edit did not reach the solver"

    def loss(value):
        return jnp.square(simulate(_with(config, leaf, value)).ys).sum()

    grad = jax.grad(loss)(base)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0), "parameter is not on the differentiable path"

    batch = jnp.stack([base, other])
    batched = jax.vmap(lambda v: simulate(_with(config, leaf, v)).ys)(batch)
    assert not jnp.allclose(batched[0], batched[1]), "vmap collapsed the batch"


def test_data_input_runs_without_a_network():
    """DataInput must not need a graph: prepare() has no network to read."""
    simulate, config = prepare(
        Generic2dOscillator(),
        Heun(),
        t0=0.0,
        t1=1.0,
        dt=0.05,
        n_nodes=1,
        externals={"stimulus": DataInput(TIMES, DATA)},
    )
    baseline = simulate(config).ys
    edited = simulate(_with(config, "data", 5.0 * jnp.ones_like(TIMES))).ys
    assert jnp.all(jnp.isfinite(baseline))
    assert not jnp.allclose(baseline, edited)
