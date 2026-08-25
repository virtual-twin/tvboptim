"""Every declared coupling and external input must reach the vector field.

A declared input that silently does nothing is the one class of
docstring-versus-code mismatch that can be caught mechanically. It hides well:
the model still integrates and the trajectory still looks plausible.

The assertion is that each declared channel is *referenced* by ``dynamics()``,
not that it changes the derivative at default parameters. Several models scale
their coupling by a gain that defaults to zero (``Epileptor.Kvf``,
``MontbrioPazoRoxin.cv``), and others multiply it into a term that vanishes at
the default initial state. Those are wired correctly; only a channel that never
appears in the traced computation is a defect.
"""

import importlib
import pkgutil

import jax
import jax.numpy as jnp
import pytest

import tvboptim.experimental.network_dynamics.dynamics as dynamics_pkg
from tvboptim.experimental.network_dynamics.core import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics

N_NODES = 3


def _concrete_dynamics():
    """Every concrete AbstractDynamics subclass the package exposes."""
    found = {}
    for module in pkgutil.walk_packages(
        dynamics_pkg.__path__, dynamics_pkg.__name__ + "."
    ):
        try:
            imported = importlib.import_module(module.name)
        except Exception:  # an optional backend must not break collection
            continue
        for obj in vars(imported).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, AbstractDynamics)
                and obj is not AbstractDynamics
                and not getattr(obj, "__abstractmethods__", None)
            ):
                found[obj.__name__] = obj
    return [found[name] for name in sorted(found)]


ALL_DYNAMICS = _concrete_dynamics()
IDS = [cls.__name__ for cls in ALL_DYNAMICS]


def _channels(model):
    """Declared (kind, name, n_dims) triples, coupling first."""
    return [("coupling", n, d) for n, d in model.COUPLING_INPUTS.items()] + [
        ("external", n, d) for n, d in getattr(model, "EXTERNAL_INPUTS", {}).items()
    ]


def test_dynamics_classes_were_discovered():
    assert len(ALL_DYNAMICS) >= 10, f"only found {IDS}"


@pytest.mark.parametrize("dynamics_class", ALL_DYNAMICS, ids=IDS)
def test_declared_inputs_reach_the_vector_field(dynamics_class):
    model = dynamics_class()
    channels = _channels(model)
    assert channels, f"{dynamics_class.__name__} declares no inputs"

    state = jnp.tile(jnp.asarray(model.INITIAL_STATE, dtype=float)[:, None], N_NODES)

    def vector_field(*blocks):
        coupling, external = Bunch(), Bunch()
        for (kind, name, _), block in zip(channels, blocks):
            (coupling if kind == "coupling" else external)[name] = block
        out = model.dynamics(0.0, state, model.params, coupling, external)
        return out[0] if isinstance(out, tuple) else out

    blocks = [jnp.zeros((n_dims, N_NODES)) for _, _, n_dims in channels]
    closed = jax.make_jaxpr(vector_field)(*blocks).jaxpr

    referenced = {
        var
        for eqn in closed.eqns
        for var in eqn.invars
        if isinstance(var, jax.extend.core.Var)
    }
    for (kind, name, _), var in zip(channels, closed.invars):
        assert var in referenced, (
            f"{dynamics_class.__name__}: {kind} input {name!r} is declared but "
            "never referenced by dynamics()"
        )
