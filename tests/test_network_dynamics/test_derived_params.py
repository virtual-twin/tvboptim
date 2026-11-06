"""Derived parameters: values computed from primary parameters once per solve.

A model that declares ``DERIVED_PARAMS`` must behave exactly like the same
model computing those values inline in ``dynamics()``, on every dispatch path,
for trajectories and for gradients with respect to the primary parameters.
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tvboptim.execution import ParallelExecution
from tvboptim.experimental.network_dynamics import (
    Bunch,
    HeterogeneousNetwork,
    Network,
    NodeGroup,
    prepare,
)
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import DiffraxSolver, Heun
from tvboptim.observations.tvb_monitors import SubSampling
from tvboptim.types import DataAxis, Space, TransformedParameter

T1 = 1.0
DT = 0.1
N_NODES = 3
WEIGHTS = jnp.array(
    [
        [0.0, 0.4, 0.0, 0.2, 0.0],
        [0.3, 0.0, 0.1, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.5],
        [0.1, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.4, 0.2, 0.0],
    ]
)


def _positive(value):
    return TransformedParameter(value, jnp.exp, jnp.log)


class InlineLeak(AbstractDynamics):
    """Reference: computes everything inside ``dynamics()``."""

    STATE_NAMES = ("x",)
    INITIAL_STATE = (0.5,)
    DEFAULT_PARAMS = Bunch(tau=2.0, k=0.7)
    COUPLING_INPUTS = {"drive": 1}

    def dynamics(self, t, state, params, coupling, external):
        rate = 1.0 / params.tau
        gain = jnp.tanh(params.k) * rate
        return jnp.stack((-rate * state[0] + gain * coupling.drive[0] + 0.1,))


class DerivedLeak(AbstractDynamics):
    """Same model with ``rate`` and ``gain`` declared as derived values."""

    STATE_NAMES = ("x",)
    INITIAL_STATE = (0.5,)
    DEFAULT_PARAMS = Bunch(tau=2.0, k=0.7)
    DERIVED_PARAMS = ("rate", "gain")
    COUPLING_INPUTS = {"drive": 1}

    def derived_params(self, params):
        # gain depends on the other derived value through a local variable.
        rate = 1.0 / params.tau
        return Bunch(rate=rate, gain=jnp.tanh(params.k) * rate)

    def dynamics(self, t, state, params, coupling, external):
        return jnp.stack(
            (-params.rate * state[0] + params.gain * coupling.drive[0] + 0.1,)
        )


def _diffrax():
    return DiffraxSolver(
        diffrax.Tsit5(), saveat=diffrax.SaveAt(ts=jnp.arange(0.0, T1, DT))
    )


def _network(model_cls, n_nodes=N_NODES):
    return Network(
        dynamics=model_cls(tau=jnp.linspace(1.5, 3.0, n_nodes)),
        coupling={"drive": LinearCoupling(source="x", G=0.5)},
        graph=DenseGraph(WEIGHTS[:n_nodes, :n_nodes]),
    )


def _heterogeneous(model_cls):
    # Groups of different sizes with group-local (scalar) primary values.
    return HeterogeneousNetwork(
        graph=DenseGraph(WEIGHTS),
        groups={
            "a": NodeGroup(model_cls(tau=1.5), [0, 2]),
            "b": NodeGroup(model_cls(tau=3.0, k=0.2), [1, 3, 4]),
        },
    )


def _dynamics_locations(config):
    """Primary-parameter Bunches of a config, flat or grouped."""
    if "groups" in config:
        return [config.groups[name].dynamics for name in config.groups]
    return [config.dynamics]


# (label, prepare(model_cls) -> (solve_fn, config))
PATHS = [
    (
        "bare_native",
        lambda cls: prepare(cls(), Heun(), t1=T1, dt=DT, n_nodes=N_NODES),
    ),
    (
        "bare_native_observe",
        lambda cls: prepare(
            cls(),
            Heun(block_size=2),
            t1=T1,
            dt=DT,
            n_nodes=N_NODES,
            observe=SubSampling(period=0.2),
        ),
    ),
    (
        "bare_diffrax",
        lambda cls: prepare(cls(), _diffrax(), t1=T1, dt=DT, n_nodes=N_NODES),
    ),
    ("network_native", lambda cls: prepare(_network(cls), Heun(), t1=T1, dt=DT)),
    (
        "network_native_observe",
        lambda cls: prepare(
            _network(cls),
            Heun(block_size=2),
            t1=T1,
            dt=DT,
            observe=SubSampling(period=0.2),
        ),
    ),
    ("network_diffrax", lambda cls: prepare(_network(cls), _diffrax(), t1=T1, dt=DT)),
    (
        "heterogeneous_native",
        lambda cls: prepare(_heterogeneous(cls), Heun(), t1=T1, dt=DT),
    ),
]
PATH_IDS = [label for label, _ in PATHS]


def _loss(solve_fn, config, tau_scale):
    varied = config.copy()
    for params in _dynamics_locations(varied):
        params.tau = params.tau * tau_scale
    leaves = jax.tree.leaves(solve_fn(varied).ys)
    return sum(jnp.square(leaf).sum() for leaf in leaves)


@pytest.mark.parametrize("_label,build", PATHS, ids=PATH_IDS)
def test_derived_matches_inline_trajectories_and_gradients(_label, build):
    inline_fn, inline_cfg = build(InlineLeak)
    derived_fn, derived_cfg = build(DerivedLeak)

    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6),
        inline_fn(inline_cfg).ys,
        derived_fn(derived_cfg).ys,
    )

    inline_grad = jax.grad(lambda s: _loss(inline_fn, inline_cfg, s))(1.0)
    derived_grad = jax.grad(lambda s: _loss(derived_fn, derived_cfg, s))(1.0)
    assert inline_grad != 0.0
    np.testing.assert_allclose(derived_grad, inline_grad, rtol=1e-5)


@pytest.mark.parametrize("_label,build", PATHS, ids=PATH_IDS)
def test_config_holds_only_primary_parameters(_label, build):
    _, config = build(DerivedLeak)
    for params in _dynamics_locations(config):
        assert set(params) == set(DerivedLeak.DEFAULT_PARAMS)


@pytest.mark.parametrize("_label,build", PATHS, ids=PATH_IDS)
def test_derived_name_in_config_is_rejected_with_location(_label, build):
    solve_fn, config = build(DerivedLeak)
    _dynamics_locations(config)[0].rate = 0.5
    expected = "config.groups.a.dynamics" if "groups" in config else "config.dynamics"
    with pytest.raises(ValueError, match=rf"\['rate'\] in {expected}"):
        solve_fn(config)


def test_space_sweep_over_primary_updates_derived_values():
    solve_fn, config = prepare(_network(DerivedLeak), Heun(), t1=T1, dt=DT)
    inline_fn, inline_cfg = prepare(_network(InlineLeak), Heun(), t1=T1, dt=DT)
    taus = jnp.array([1.0, 2.0, 4.0])

    def final_mean(fn):
        return lambda cfg: fn(cfg).ys[-1].mean()

    results = []
    for fn, cfg in ((solve_fn, config), (inline_fn, inline_cfg)):
        swept = cfg.copy()
        swept.dynamics.tau = DataAxis(taus)
        execution = ParallelExecution(final_mean(fn), Space(swept), n_vmap=3, n_pmap=1)
        results.append(np.asarray([np.asarray(v) for v in execution.run()]))

    derived, inline = results
    np.testing.assert_allclose(derived, inline, rtol=1e-6)
    assert len(np.unique(derived)) == len(taus), "derived value went stale"


# Model-level API --------------------------------------------------------------


def test_resolve_params_merges_without_mutating_input():
    model = DerivedLeak(tau=4.0)
    primary = model.params.copy()
    resolved = model.resolve_params(model.params)

    assert dict(model.params) == dict(primary)
    assert set(resolved) == {"tau", "k", "rate", "gain"}
    np.testing.assert_allclose(resolved.rate, 0.25)

    model.params.tau = 2.0
    np.testing.assert_allclose(model.resolve_params(model.params).rate, 0.5)


def test_resolve_params_is_identity_without_derived_values():
    model = InlineLeak()
    assert model.resolve_params(model.params) is model.params


def test_verify_and_simulate_resolve_derived_values():
    inline, derived = InlineLeak(), DerivedLeak()
    assert derived.verify(n_nodes=2, verbose=False)
    _, inline_ys = inline.simulate(t1=T1, dt=DT, n_nodes=2)
    _, derived_ys = derived.simulate(t1=T1, dt=DT, n_nodes=2)
    np.testing.assert_allclose(derived_ys, inline_ys, rtol=1e-6)


def test_resolved_values_can_drive_direct_dynamics_calls():
    inline, derived = InlineLeak(), DerivedLeak()
    state = jnp.ones((1, 2))
    coupling = Bunch(drive=jnp.full((1, 2), 0.3))
    np.testing.assert_allclose(
        derived.dynamics(
            0.0, state, derived.resolve_params(derived.params), coupling, Bunch()
        ),
        inline.dynamics(0.0, state, inline.params, coupling, Bunch()),
    )


def test_identity_derivation_keeps_transformed_parameter_gradients():
    class InlineTau(InlineLeak):
        pass

    class IdentityTau(InlineLeak):
        DERIVED_PARAMS = ("tau_used",)

        def derived_params(self, params):
            return Bunch(tau_used=params.tau)

        def dynamics(self, t, state, params, coupling, external):
            return InlineLeak.dynamics(
                self, t, state, Bunch(params, tau=params.tau_used), coupling, external
            )

    def grad_for(cls):
        solve_fn, config = prepare(cls(), Heun(), t1=T1, dt=DT, n_nodes=2)

        def loss(tau):
            varied = config.copy()
            varied.dynamics.tau = tau
            return jnp.square(solve_fn(varied).ys).sum()

        return jax.grad(loss)(_positive(2.0)).value

    expected = grad_for(InlineTau)
    assert expected != 0.0
    np.testing.assert_allclose(grad_for(IdentityTau), expected, rtol=1e-6)


# Inheritance ------------------------------------------------------------------


class ScaledGain(DerivedLeak):
    """Overrides the formula, keeps the inherited names."""

    def derived_params(self, params):
        rate = 1.0 / params.tau
        return Bunch(rate=rate, gain=2.0 * rate)


class WithOffset(DerivedLeak):
    """Extends the parent declaration via ``super()``."""

    DERIVED_PARAMS = DerivedLeak.DERIVED_PARAMS + ("double_rate",)

    def derived_params(self, params):
        derived = super().derived_params(params)
        derived.double_rate = 2.0 * derived.rate
        return derived


class FreeGain(DerivedLeak):
    """Promotes every derived value to a primary parameter."""

    DEFAULT_PARAMS = Bunch(rate=0.5, gain=0.3)
    DERIVED_PARAMS = ()

    def derived_params(self, params):
        return Bunch()


class FreeRate(DerivedLeak):
    """Partial promotion: ``rate`` becomes primary, ``gain`` stays derived."""

    DEFAULT_PARAMS = Bunch(rate=0.5)
    DERIVED_PARAMS = ("gain",)

    def derived_params(self, params):
        return Bunch(gain=jnp.tanh(params.k) * params.rate)


class Unchanged(DerivedLeak):
    pass


INHERITANCE = [
    (Unchanged, ("rate", "gain"), {"tau", "k"}, {"rate": 0.5}),
    (ScaledGain, ("rate", "gain"), {"tau", "k"}, {"gain": 1.0}),
    (WithOffset, ("rate", "gain", "double_rate"), {"tau", "k"}, {"double_rate": 1.0}),
    (FreeGain, (), {"tau", "k", "rate", "gain"}, {"rate": 0.5, "gain": 0.3}),
    (FreeRate, ("gain",), {"tau", "k", "rate"}, {"rate": 0.5}),
]


@pytest.mark.parametrize(
    "cls,derived_names,primary_names,expected",
    INHERITANCE,
    ids=[case[0].__name__ for case in INHERITANCE],
)
def test_inheritance_of_derived_declarations(
    cls, derived_names, primary_names, expected
):
    model = cls()
    assert cls.DERIVED_PARAMS == derived_names
    assert set(model.params) == primary_names
    resolved = model.resolve_params(model.params)
    assert set(resolved) == primary_names | set(derived_names)
    for name, value in expected.items():
        np.testing.assert_allclose(resolved[name], value)
    assert model.verify(verbose=False)


# Errors -----------------------------------------------------------------------


def _subclass(body):
    namespace = {
        "STATE_NAMES": ("x",),
        "INITIAL_STATE": (0.0,),
        "DEFAULT_PARAMS": Bunch(tau=1.0),
        "dynamics": lambda self, t, state, params, coupling, external: state,
        **body,
    }
    return type("Broken", (AbstractDynamics,), namespace)


def _hook(result):
    return {"derived_params": lambda self, params: result}


CLASS_ERRORS = [
    (
        "not_a_tuple",
        {"DERIVED_PARAMS": ["rate"], **_hook(Bunch(rate=1.0))},
        TypeError,
        "tuple of strings",
    ),
    (
        "not_strings",
        {"DERIVED_PARAMS": (1,), **_hook(Bunch())},
        TypeError,
        "tuple of strings",
    ),
    (
        "duplicates",
        {"DERIVED_PARAMS": ("r", "r"), **_hook(Bunch(r=1.0))},
        ValueError,
        "duplicate",
    ),
    (
        "overlaps_primary",
        {"DERIVED_PARAMS": ("tau",), **_hook(Bunch(tau=1.0))},
        ValueError,
        "either primary or derived",
    ),
    (
        "missing_hook",
        {"DERIVED_PARAMS": ("rate",)},
        TypeError,
        "does not implement or inherit",
    ),
]


@pytest.mark.parametrize(
    "_label,body,error,match", CLASS_ERRORS, ids=[c[0] for c in CLASS_ERRORS]
)
def test_invalid_declarations_fail_at_class_creation(_label, body, error, match):
    with pytest.raises(error, match=match):
        _subclass(body)


def test_overlap_after_inheritance_points_to_promotion():
    with pytest.raises(ValueError, match="remove it from DERIVED_PARAMS"):

        class ForgotToRemove(DerivedLeak):
            DEFAULT_PARAMS = Bunch(rate=0.5)


RESOLUTION_ERRORS = [
    ("plain_dict", {"rate": 1.0}, TypeError, "must return a Bunch"),
    ("missing", Bunch(), ValueError, r"missing \['rate'\]"),
    ("extra", Bunch(rate=1.0, other=2.0), ValueError, r"unexpected \['other'\]"),
]


@pytest.mark.parametrize(
    "_label,result,error,match",
    RESOLUTION_ERRORS,
    ids=[c[0] for c in RESOLUTION_ERRORS],
)
@pytest.mark.parametrize("entry", ["resolve", "simulate"])
def test_malformed_hook_output_fails_at_resolution(_label, result, error, match, entry):
    model = _subclass({"DERIVED_PARAMS": ("rate",), **_hook(result)})()
    with pytest.raises(error, match=match):
        if entry == "resolve":
            model.resolve_params(model.params)
        else:
            solve_fn, config = prepare(model, Heun(), t1=T1, dt=DT)
            solve_fn(config)


DERIVED_INPUTS = [
    ("plain_value", 0.5),
    ("parameter", _positive(0.5)),
]


@pytest.mark.parametrize(
    "_label,value", DERIVED_INPUTS, ids=[c[0] for c in DERIVED_INPUTS]
)
def test_derived_name_in_input_is_rejected(_label, value):
    model = DerivedLeak()
    with pytest.raises(ValueError, match="Fitting a derived quantity"):
        model.resolve_params(Bunch(model.params, rate=value))


def test_resolving_twice_is_rejected():
    model = DerivedLeak()
    with pytest.raises(ValueError, match="cannot be set, swept or optimized"):
        model.resolve_params(model.resolve_params(model.params))


def test_setting_derived_value_in_constructor_is_rejected():
    with pytest.raises(ValueError, match="computed by DerivedLeak; set its primary"):
        DerivedLeak(rate=0.5)


@pytest.mark.parametrize("build", [_network, _heterogeneous], ids=["flat", "grouped"])
def test_printer_lists_derived_names(build):
    from tvboptim.experimental.network_dynamics.utils.printer import format_network

    text = format_network(build(DerivedLeak))
    assert "Derived: rate, gain" in text
    if build is _network:
        assert "def derived_params" in text
