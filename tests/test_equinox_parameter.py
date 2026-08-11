"""Focused integration tests for trainable Equinox modules."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from tvboptim.analysis import loss_hessian
from tvboptim.execution import ParallelExecution
from tvboptim.optim import OptaxOptimizer
from tvboptim.types import (
    DataAxis,
    EquinoxParameter,
    Parameter,
    Space,
    collect_parameters,
    combine_state,
    partition_state,
    show_parameters,
)

INPUT = jnp.asarray([0.75])
TARGET = jnp.asarray([0.2])


def _mlp(key=0):
    return eqx.nn.MLP(
        in_size=1,
        out_size=1,
        width_size=3,
        depth=1,
        activation=jax.nn.tanh,
        key=jax.random.key(key),
    )


def _array_leaves(tree):
    return [leaf for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf)]


def _loss(state):
    prediction = state["correction"](INPUT)
    if "offset" in state:
        prediction = prediction + state["offset"]
    return jnp.sum((prediction - TARGET) ** 2)


def test_wrapper_delegates_to_enclosed_module():
    module = _mlp()
    wrapped = EquinoxParameter(module)

    assert jnp.allclose(wrapped(INPUT), module(INPUT))


def test_partition_trains_only_inexact_module_arrays_and_preserves_static_fields():
    wrapped = EquinoxParameter(_mlp())
    state = {"correction": wrapped, "fixed": jnp.asarray(3, dtype=jnp.int32)}

    diff_state, static_state = partition_state(state)

    diff_leaves = jax.tree.leaves(diff_state)
    assert diff_leaves
    assert all(eqx.is_inexact_array(leaf) for leaf in diff_leaves)
    assert static_state["fixed"].dtype == jnp.int32

    restored = combine_state(diff_state, static_state)
    restored_module = restored["correction"].module
    assert isinstance(restored["correction"], EquinoxParameter)
    assert isinstance(restored_module, eqx.nn.MLP)
    assert restored_module.activation is wrapped.module.activation
    assert restored_module.final_activation is wrapped.module.final_activation
    assert jnp.allclose(restored["correction"](INPUT), wrapped(INPUT))


class _MixedArrayModule(eqx.Module):
    weight: jax.Array
    count: jax.Array

    def __call__(self, value):
        return self.weight * value


def test_exact_array_inside_module_remains_static():
    wrapped = EquinoxParameter(
        _MixedArrayModule(
            weight=jnp.asarray(2.0), count=jnp.asarray(3, dtype=jnp.int32)
        )
    )

    diff_state, static_state = partition_state({"correction": wrapped})

    assert diff_state["correction"].module.weight is not None
    assert diff_state["correction"].module.count is None
    assert static_state["correction"].module.weight is None
    assert static_state["correction"].module.count == 3


def test_collection_unwraps_module_and_display_includes_owner(capsys):
    wrapped = EquinoxParameter(_mlp())

    collected = collect_parameters({"correction": wrapped})
    show_parameters({"correction": wrapped})

    assert collected["correction"] is wrapped.module
    assert "correction" in capsys.readouterr().out


def test_one_optax_update_changes_module_arrays_and_prediction():
    state = {"correction": EquinoxParameter(_mlp())}
    before_arrays = _array_leaves(state["correction"])
    before_prediction = state["correction"](INPUT)

    fitted, _ = OptaxOptimizer(_loss, optax.sgd(0.1)).run(state, max_steps=1)

    after_arrays = _array_leaves(fitted["correction"])
    assert any(not jnp.allclose(a, b) for a, b in zip(before_arrays, after_arrays))
    assert not jnp.allclose(before_prediction, fitted["correction"](INPUT))


@pytest.mark.parametrize(
    ("mode", "chunk_size"),
    [("rev", None), ("rev", 2), ("fwd", None), ("fwd", 2)],
)
def test_mixed_parameters_optimize_in_eager_and_chunked_modes(mode, chunk_size):
    state = {
        "correction": EquinoxParameter(_mlp()),
        "offset": Parameter(0.4),
    }
    initial_loss = _loss(state)
    initial_offset = state["offset"].value

    fitted, _ = OptaxOptimizer(_loss, optax.sgd(0.05)).run(
        state, max_steps=4, mode=mode, chunk_size=chunk_size
    )

    assert _loss(fitted) < initial_loss
    assert not jnp.allclose(fitted["offset"].value, initial_offset)
    assert isinstance(fitted["correction"], EquinoxParameter)
    assert (
        fitted["correction"].module.activation is state["correction"].module.activation
    )


def test_chunked_optimizer_works_under_parallel_execution_trace():
    starts = jnp.asarray([-0.3, 0.3])
    correction = EquinoxParameter(_mlp())
    space = Space({"start": DataAxis(starts), "correction": correction}, mode="zip")

    def fit(state):
        optim_state = {
            "offset": Parameter(state["start"]),
            "correction": state["correction"],
        }
        fitted, _ = OptaxOptimizer(_loss, optax.sgd(0.05)).run(
            optim_state, max_steps=3, chunk_size=2
        )
        return _loss(fitted)

    losses = jnp.ravel(
        jnp.asarray(ParallelExecution(fit, space, n_vmap=2, n_pmap=1).run().results)
    )
    assert losses.shape == starts.shape
    assert jnp.all(jnp.isfinite(losses))


def test_space_preserves_fixed_module_as_one_object():
    correction = EquinoxParameter(_mlp())
    space = Space(
        {"sample": DataAxis(jnp.asarray([1.0, 2.0])), "correction": correction}
    )

    for state in space:
        assert state["correction"] is correction
        assert jnp.allclose(state["correction"](INPUT), correction(INPUT))

    assert space.collect(combine=True)["correction"] is correction
    assert space[:1][0]["correction"] is correction


class _AxisModule(eqx.Module):
    weight: Any

    def __call__(self, value):
        return self.weight * value


def test_space_rejects_axis_nested_inside_equinox_parameter():
    state = {
        "correction": EquinoxParameter(_AxisModule(DataAxis(jnp.asarray([1.0, 2.0]))))
    }

    with pytest.raises(ValueError, match="EquinoxParameter.+axis"):
        Space(state)


def test_identifiability_flattens_module_with_one_label_per_scalar():
    correction = EquinoxParameter(
        eqx.nn.Linear(in_features=1, out_features=1, key=jax.random.key(1))
    )
    state = {"correction": correction}

    hessian, theta, labels, _ = loss_hessian(_loss, state, check_gradient=False)

    assert theta.shape == (2,)
    assert hessian.shape == (2, 2)
    assert len(labels) == theta.size
    assert len(set(labels)) == theta.size
