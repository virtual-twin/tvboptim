"""Heterogeneous result shape and projection contracts."""

import jax
import jax.numpy as jnp
import pytest

from tvboptim.experimental.network_dynamics import HeterogeneousSolution


def _solution():
    ts = jnp.array([0.1, 0.2, 0.3])
    return HeterogeneousSolution(
        ts,
        {
            "a": jnp.arange(18.0).reshape(3, 2, 3),
            "b": (100.0 + jnp.arange(9.0)).reshape(3, 1, 3),
        },
        dt=0.1,
        variable_names={"a": ("shared", "a_only"), "b": ("shared",)},
        group_nodes={"a": (0, 2, 5), "b": (1, 3, 4)},
        n_nodes=6,
    )


def test_group_views_keep_natural_shapes_and_named_selection():
    result = _solution()
    assert result.groups.a.ys.shape == (3, 2, 3)
    assert result.groups.b.ys.shape == (3, 1, 3)
    assert result.groups.a.sel("shared").shape == (3, 3)
    assert result.groups.a.sel(("shared", "a_only")).shape == (3, 2, 3)
    assert jnp.array_equal(result.group_nodes.a, jnp.array([0, 2, 5]))


def test_explicit_projection_uses_graph_node_order():
    result = _solution()
    projected = result.to_graph("shared")
    assert projected.shape == (3, 6)
    assert jnp.array_equal(projected[:, [0, 2, 5]], result.ys.a[:, 0, :])
    assert jnp.array_equal(projected[:, [1, 3, 4]], result.ys.b[:, 0, :])


def test_partial_projection_fills_unrepresented_nodes():
    result = _solution()
    projected = result.to_graph("a_only", groups=["a"], fill_value=-1.0)
    assert jnp.array_equal(projected[:, [0, 2, 5]], result.ys.a[:, 1, :])
    assert jnp.all(projected[:, [1, 3, 4]] == -1.0)


def test_projection_rejects_unknown_group_or_variable():
    result = _solution()
    with pytest.raises(ValueError, match="Unknown solution groups"):
        result.to_graph("shared", groups=["missing"])
    with pytest.raises(ValueError, match="not available"):
        result.to_graph("a_only", groups=["b"])


def test_heterogeneous_solution_is_a_jittable_pytree():
    result = _solution()

    @jax.jit
    def scale(value):
        return jax.tree.map(lambda leaf: 2.0 * leaf, value)

    scaled = scale(result)
    assert isinstance(scaled, HeterogeneousSolution)
    assert jnp.array_equal(scaled.ys.a, 2.0 * result.ys.a)
    assert scaled.variable_names == result.variable_names
    assert scaled._group_nodes == result._group_nodes
