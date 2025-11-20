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


# Register Bunch as a JAX PyTree
def _bunch_tree_flatten(bunch: Bunch) -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
    """Flatten Bunch into values and keys for JAX PyTree."""
    keys = tuple(sorted(bunch.keys()))  # Sort for deterministic order
    values = tuple(bunch[key] for key in keys)
    return values, keys


def _bunch_tree_unflatten(keys: Tuple[str, ...], values: Tuple[Any, ...]) -> Bunch:
    """Reconstruct Bunch from keys and values for JAX PyTree."""
    return Bunch(zip(keys, values))


# Register the PyTree
jax.tree_util.register_pytree_node(Bunch, _bunch_tree_flatten, _bunch_tree_unflatten)
