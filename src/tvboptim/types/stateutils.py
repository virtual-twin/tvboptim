# %%
import equinox as eqx
import jax

from tvboptim.types.equinox_parameter import EquinoxParameter
from tvboptim.types.parameter import Parameter
from tvboptim.utils import format_pytree_as_string


def is_parameter_owner(value):
    """Whether ``value`` explicitly owns trainable parameter leaves."""
    return isinstance(value, (Parameter, EquinoxParameter))


def is_partition_leaf(value):
    """Whether state partition/combine must treat ``value`` atomically."""
    return isinstance(value, Parameter)


def parameter_filter(value):
    """Build the nested differentiability filter for one parameter owner."""
    if isinstance(value, Parameter):
        return True
    if isinstance(value, EquinoxParameter):
        return jax.tree.map(eqx.is_inexact_array, value)
    return False


def collect_parameters(state):
    """Extract values and modules from parameter owners in a state tree.

    This function traverses a JAX PyTree state and extracts the underlying
    values from ``Parameter`` objects and modules from ``EquinoxParameter``
    objects while leaving other values unchanged.

    Parameters
    ----------
    state : Any
        JAX PyTree containing Parameter objects and other values.

    Returns
    -------
    Any
        JAX PyTree with same structure as input, but Parameter objects
        replaced by their underlying values.

    Examples
    --------
    >>> from tvboptim.types import Parameter
    >>> import jax.numpy as jnp
    >>>
    >>> # Create state with Parameter objects
    >>> state = {
    ...     'param1': Parameter(jnp.array(1.5)),
    ...     'param2': jnp.array(2.0),
    ...     'nested': {'param3': Parameter(jnp.array([1, 2, 3]))}
    ... }
    >>>
    >>> # Extract values
    >>> values = collect_parameters(state)
    >>> print(values['param1'])  # jnp.array(1.5)
    >>> print(values['param2'])  # jnp.array(2.0)

    Notes
    -----
    This function is useful when you need to extract raw JAX arrays from
    a state tree for operations that don't require Parameter metadata.
    With the new Parameter system, this function may become less necessary
    as Parameters support the JAX array protocol directly.
    """

    def _collect_parameters(leaf):
        if isinstance(leaf, Parameter):
            return leaf.__jax_array__()
        elif isinstance(leaf, EquinoxParameter):
            return leaf.module
        else:
            return leaf

    return jax.tree.map(
        lambda x: _collect_parameters(x),
        state,
        is_leaf=is_parameter_owner,
    )


# %%
def mark_parameters(state):
    """Mark ordinary parameters and nested Equinox array leaves."""
    return jax.tree.map(parameter_filter, state, is_leaf=is_parameter_owner)


# %%
def partition_state(state):
    """Separate explicitly trainable leaves from static state."""
    param_mask = mark_parameters(state)
    diff_state, static_state = eqx.partition(
        state, param_mask, is_leaf=is_partition_leaf
    )
    return diff_state, static_state


def combine_state(diff_state, static_state):
    """Recombine optimized parameters with static values."""
    return eqx.combine(diff_state, static_state, is_leaf=is_partition_leaf)


# def show_free(tree):
#     def _show_free(leaf):
#         if isinstance(leaf, Value) and leaf.free:
#             print(leaf)
#     jax.tree.map(_show_free, tree, is_leaf=lambda x: isinstance(x, Value))
def show_parameters(tree):
    """Show explicitly trainable parameter leaves in the tree."""
    param_mask = mark_parameters(tree)
    diff_model, _ = eqx.partition(tree, param_mask, is_leaf=is_partition_leaf)
    print(
        format_pytree_as_string(
            diff_model, hide_none=True, name="Parameters", show_array_values=True
        )
    )
    return None
