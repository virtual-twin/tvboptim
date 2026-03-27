"""Parameter container with attribute access and JAX PyTree support."""

from typing import Any, Tuple

import jax


class Bunch(dict):
    """Dictionary with attribute access for parameters.

    A JAX PyTree-compatible parameter container that allows both dict['key']
    and dict.key access patterns. Designed for neural dynamics parameters
    with support for JAX transformations.

    Examples:
        >>> params = Bunch(a=1.0, b=2.0)
        >>> params.a  # attribute access
        1.0
        >>> params['b']  # dict access
        2.0
        >>> jax.tree.map(lambda x: x * 2, params)  # JAX transformations
        Bunch(a=2.0, b=4.0)
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"{self.__class__.__name__}({items})"

    def copy(self) -> "Bunch":
        """Create a shallow copy of the Bunch."""
        return Bunch(super().copy())


# Register Bunch as a JAX PyTree with named keys so that
# jax.tree_util.tree_flatten_with_path produces DictKey paths instead of
# FlattenedIndexKey positional indices.
def _bunch_tree_flatten_with_keys(bunch: Bunch) -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
    """Flatten Bunch into (key, value) pairs and aux keys for JAX PyTree."""
    keys = tuple(sorted(bunch.keys()))  # Sort for deterministic order
    children_with_keys = [(jax.tree_util.DictKey(k), bunch[k]) for k in keys]
    return children_with_keys, keys


def _bunch_tree_unflatten(keys: Tuple[str, ...], values: Tuple[Any, ...]) -> Bunch:
    """Reconstruct Bunch from keys and values for JAX PyTree."""
    return Bunch(zip(keys, values))


# register_pytree_with_keys provides both regular and key-aware flattening
jax.tree_util.register_pytree_with_keys(
    Bunch, _bunch_tree_flatten_with_keys, _bunch_tree_unflatten
)
