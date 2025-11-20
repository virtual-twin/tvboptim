import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def format_pytree_as_string(
    pytree,
    name: str = "root",
    prefix: str = "",
    is_last: bool = False,
    show_numerical_only: bool = False,
    is_root: bool = True,
    hide_none: bool = False,
    show_array_values: bool = False,
):
    """
    Recursively formats a JAX pytree structure as a string with Unicode box-drawing characters.

    Args:
        pytree: The pytree to format
        name: The name of the current node
        prefix: Current line prefix
        is_last: Whether the current node is the last child of its parent
        show_numerical_only: If True, only show arrays and numerical types (float, int, etc.)
        is_root: Whether this node is the root of the tree
        hide_none: If True, fields with None values will be hidden

    Returns:
        str: The formatted string representation of the pytree
    """
    # Unicode box-drawing characters for the tree structure
    space = "    "
    branch = "│   "
    tee = "├── "
    last = "└── "

    # Initialize the result string
    result = []

    # Special handling for root element
    if is_root:
        current_prefix = ""  # Root has no prefix
        next_prefix = ""  # Children of root start without vertical bars
    else:
        # Determine the current line prefix
        current_prefix = prefix + (last if is_last else tee)
        # Determine the prefix for children
        next_prefix = prefix + (space if is_last else branch)

    # Check if the object is a string
    if isinstance(pytree, str):
        if not show_numerical_only:
            result.append(f'{current_prefix}{name}: "{pytree}"')
        return "\n".join(result)

    # Check if the object is None
    if pytree is None:
        if not hide_none and not show_numerical_only:
            result.append(f"{current_prefix}{name}: NoneType")
        return "\n".join(result)

    # Check if the object is a JAX array
    if isinstance(pytree, (jnp.ndarray, np.ndarray)):
        # result.append(f"{current_prefix}{name}: Array(shape={pytree.shape}, dtype={pytree.dtype})")
        if show_array_values:
            # result.append(f"{current_prefix}{name}: Array({shape_str}, {dtype_str})")
            # result.append(f"{current_prefix}{name}: No({shape_str}, {dtype_str})")
            result.append(f"{current_prefix}{name}: {pytree}")
        else:
            result.append(f"{current_prefix}{name}: {eqx.tree_pformat(pytree)}")
        return "\n".join(result)

    # Try to flatten the pytree
    try:
        leaves, _ = jax.tree_util.tree_flatten(pytree)
        # If it's a leaf (i.e., it has no children), format its type
        if not leaves or (len(leaves) == 1 and pytree is leaves[0]):
            # For numerical types, always display the value
            if isinstance(pytree, (int, float, bool, complex)):
                result.append(f"{current_prefix}{name}: {pytree}")
            # For other types, check filter setting
            elif not show_numerical_only:
                result.append(f"{current_prefix}{name}: {type(pytree).__name__}")
            return "\n".join(result)

        # Otherwise, format it as a container and process its children
        result.append(f"{current_prefix}{name}")

        # If it's a dictionary, iterate through its key-value pairs
        if isinstance(pytree, dict):
            items = list(pytree.items())
            for i, (key, value) in enumerate(items):
                child_result = format_pytree_as_string(
                    value,
                    str(key),
                    next_prefix,
                    i == len(items) - 1,
                    show_numerical_only,
                    False,
                    hide_none,
                    show_array_values,
                )
                if child_result:  # Only append if there's content (might be empty with show_numerical_only)
                    result.append(child_result)

        # If it's a dataclass or a custom class with __dict__ attribute
        elif hasattr(pytree, "__dict__"):
            items = list(pytree.__dict__.items())
            for i, (key, value) in enumerate(items):
                child_result = format_pytree_as_string(
                    value,
                    key,
                    next_prefix,
                    i == len(items) - 1,
                    show_numerical_only,
                    False,
                    hide_none,
                    show_array_values,
                )
                if child_result:
                    result.append(child_result)

        # If it's a sequence (like list or tuple)
        elif hasattr(pytree, "__len__") and not isinstance(
            pytree, (str, bytes, bytearray)
        ):
            for i, item in enumerate(pytree):
                child_result = format_pytree_as_string(
                    item,
                    f"[{i}]",
                    next_prefix,
                    i == len(pytree) - 1,
                    show_numerical_only,
                    False,
                    hide_none,
                    show_array_values,
                )
                if child_result:
                    result.append(child_result)

        # For other types of containers
        else:
            result.append(
                f"{current_prefix}{name}: {type(pytree).__name__} (unknown structure)"
            )

    except Exception:
        # If we can't flatten it as a pytree, treat it as a leaf
        # For strings, display the string value if not filtering
        if isinstance(pytree, str):
            if not show_numerical_only:
                result.append(f'{current_prefix}{name}: "{pytree}"')
        # For numerical types, always display the value
        elif isinstance(pytree, (int, float, bool, complex)):
            result.append(f"{current_prefix}{name}: {pytree}")
        # For other types, check filter setting
        elif not show_numerical_only:
            result.append(f"{current_prefix}{name}: {type(pytree).__name__}")

    return "\n".join(result)


def pretty_print_pytree(
    pytree,
    name: str = "root",
    prefix: str = "",
    show_numerical_only: bool = False,
    hide_none: bool = False,
):
    """
    Prints a pretty formatted representation of a JAX pytree structure.

    Args:
        pytree: The pytree to print
        name: The name of the current node
        prefix: Current line prefix
        show_numerical_only: If True, only show arrays and numerical types (float, int, etc.)
        hide_none: If True, fields with None values will be hidden
    """
    formatted_string = format_pytree_as_string(
        pytree, name, prefix, False, show_numerical_only, True, hide_none
    )
    print(formatted_string)


def safe_reshape(arr, new_shape, fill_value=jnp.nan):
    """
    A safe reshaping function with the following properties:
    - If new_shape has fewer elements than arr, raises an error
    - If new_shape has equal elements to arr, performs standard reshape
    - If new_shape requires more elements than arr, fills extra space with fill_value

    Args:
        arr: JAX array to reshape
        new_shape: Tuple of integers specifying the new shape
        fill_value: Value to use for filling extra elements (default: jnp.nan)

    Returns:
        Reshaped JAX array
    """
    # Calculate current and new sizes
    old_size = arr.size
    new_size = jnp.prod(jnp.abs(jnp.array(new_shape)))

    # Case 1: New shape has fewer elements than the original array
    if new_size < old_size:
        raise ValueError(
            f"New shape {new_shape} has fewer elements ({new_size}) "
            f"than original array ({old_size} elements)"
        )

    # Case 2: New shape has equal elements to the original array
    elif new_size == old_size:
        return arr.reshape(new_shape)

    # Case 3: New shape requires more elements than the original array
    else:
        # Create a new array filled with the fill_value
        result = jnp.full(new_shape, fill_value, dtype=arr.dtype)

        # Create a flattened view of the result array
        result_flat = result.reshape(-1)

        # For newer JAX versions:
        result_flat = result_flat.at[:old_size].set(arr.reshape(-1))

        return result_flat.reshape(new_shape)


def broadcast_1d_array(arr_1d, additional_dims=()):
    """
    Broadcast a 1D array of shape (N,) to shape (N, *additional_dims)
    with a single reshape operation.

    Parameters:
    -----------
    arr_1d : numpy.ndarray or jax.numpy.ndarray
        1D input array of shape (N,)
    additional_dims : tuple
        Additional dimensions to broadcast to. Can be empty tuple () for no additional dimensions.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Broadcasted array of shape (N, *additional_dims)
    """
    # Handle the case of no additional dimensions
    if not additional_dims:
        return arr_1d

    N_shape = jnp.shape(arr_1d)
    # N = arr_1d.shape[0]

    # Create a target shape with ones for the new dimensions
    reshape_dims = N_shape + (1,) * len(additional_dims)

    # Reshape in a single operation
    reshaped = arr_1d.reshape(reshape_dims)

    # Create the final target shape
    target_shape = N_shape + additional_dims

    # Use broadcast_to for memory efficiency
    return jnp.broadcast_to(reshaped, target_shape)
