"""
New minimal Parameter implementation for TVBOptim refactoring.

This module implements a simplified Parameter class that:
- Removes redundant fields (name, doc, free/grad)
- Provides native JAX integration via __jax_array__
- Implements full arithmetic operations
- Supports specialized subclasses (BoundedParameter)
"""

from typing import Union, Tuple, Any
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np


@register_pytree_node_class
class Parameter:
    """
    A minimal JAX-native parameter with full arithmetic support.
    
    If placed at a position in the state tree, that position will take part in optimizations and will be differentiated. A state can be split into parameters and static parts by partition_state and combined by combine_state.
    
    Parameters
    ----------
    value : Union[float, int, jnp.ndarray]
        The parameter value. Will be converted to JAX array.
    
    Examples
    --------
    >>> p1 = Parameter(1.0)
    >>> p2 = Parameter(jnp.array([1, 2, 3]))
    >>> result = p1 + p2  # Works with JAX arithmetic
    >>> grad_fn = jax.grad(lambda p: jnp.sum(p**2))
    >>> gradients = grad_fn(p1)  # Seamless gradients
    """
    
    def __init__(self, value: Union[float, int, jnp.ndarray]) -> None:
        self.value = jnp.asarray(value)
    
    def __repr__(self) -> str:
        # return f"Parameter({self.__jax_array__()})"
        return f"Parameter({self.value})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def __format__(self, format_spec: str) -> str:
        """Support f-string formatting by delegating to the underlying value."""
        return format(float(self.__jax_array__()), format_spec)

    # JAX Integration
    def __jax_array__(self) -> jnp.ndarray:
        """JAX array protocol - makes Parameter behave like native JAX array."""
        return self.value
    
    def __array__(self) -> jnp.ndarray:
        """NumPy array protocol compatibility."""
        # return self.value  # Return JAX array directly instead of converting to numpy
        return np.array(self.__jax_array__())  # Return JAX array directly instead of converting to numpy
    
    # Properties
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the parameter value."""
        return self.value.shape
    
    @shape.setter
    def shape(self, shape: Tuple[int, ...]) -> None:
        """
        Broadcasts the value to the desired shape if possible.
        
        Useful to turn a global parameter local.
        
        Args:
            shape: Desired shape for the parameter
            
        Raises:
            ValueError: If the parameter shape cannot be broadcasted
        """
        if self.shape == shape:
            pass
        else:
            try:
                self.value = jnp.broadcast_to(self.value, shape)
            except ValueError:
                raise ValueError(f"Parameter shape {shape} does not match value shape {self.shape} and can not be broadcasted automatically.")
    
    @property
    def dtype(self) -> jnp.dtype:
        """Data type of the parameter value."""
        return self.value.dtype
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.value.ndim
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.value.size
    
    # Arithmetic Operations - Return JAX arrays for seamless integration
    def __add__(self, other) -> jnp.ndarray:
        """Addition: param + other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() + other_val
    
    def __radd__(self, other) -> jnp.ndarray:
        """Reverse addition: other + param"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return other_val + self.__jax_array__()
    
    def __sub__(self, other) -> jnp.ndarray:
        """Subtraction: param - other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() - other_val
    
    def __rsub__(self, other) -> jnp.ndarray:
        """Reverse subtraction: other - param"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return other_val - self.__jax_array__()
    
    def __mul__(self, other) -> jnp.ndarray:
        """Multiplication: param * other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() * other_val
    
    def __rmul__(self, other) -> jnp.ndarray:
        """Reverse multiplication: other * param"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return other_val * self.__jax_array__()
    
    def __truediv__(self, other) -> jnp.ndarray:
        """Division: param / other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() / other_val
    
    def __rtruediv__(self, other) -> jnp.ndarray:
        """Reverse division: other / param"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return other_val / self.__jax_array__()
    
    def __floordiv__(self, other) -> jnp.ndarray:
        """Floor division: param // other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() // other_val
    
    def __rfloordiv__(self, other) -> jnp.ndarray:
        """Reverse floor division: other // param"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return other_val // self.__jax_array__()
    
    def __mod__(self, other) -> jnp.ndarray:
        """Modulo: param % other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() % other_val
    
    def __rmod__(self, other) -> jnp.ndarray:
        """Reverse modulo: other % param"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return other_val % self.__jax_array__()
    
    def __pow__(self, other) -> jnp.ndarray:
        """Power: param ** other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() ** other_val
    
    def __rpow__(self, other) -> jnp.ndarray:
        """Reverse power: other ** param"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return other_val ** self.__jax_array__()
    
    # Unary Operations
    def __neg__(self) -> jnp.ndarray:
        """Negation: -param"""
        return -self.__jax_array__()
    
    def __pos__(self) -> jnp.ndarray:
        """Positive: +param"""
        return +self.__jax_array__()
    
    def __abs__(self) -> jnp.ndarray:
        """Absolute value: abs(param)"""
        return jnp.abs(self.__jax_array__())
    
    # Comparison Operations  
    def __eq__(self, other) -> jnp.ndarray:
        """Equality: param == other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() == other_val
    
    def __ne__(self, other) -> jnp.ndarray:
        """Inequality: param != other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() != other_val
    
    def __lt__(self, other) -> jnp.ndarray:
        """Less than: param < other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() < other_val
    
    def __le__(self, other) -> jnp.ndarray:
        """Less than or equal: param <= other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() <= other_val
    
    def __gt__(self, other) -> jnp.ndarray:
        """Greater than: param > other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() > other_val
    
    def __ge__(self, other) -> jnp.ndarray:
        """Greater than or equal: param >= other"""
        other_val = other.__jax_array__() if hasattr(other, '__jax_array__') else other
        return self.__jax_array__() >= other_val
    
    # Indexing
    def __getitem__(self, key) -> jnp.ndarray:
        """Indexing: param[key]"""
        return self.__jax_array__()[key]
    
    # PyTree Implementation
    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray], None]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)
        aux_data = None
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data: None, children: Tuple[jnp.ndarray]) -> 'Parameter':
        """Unflatten for JAX pytree registration."""
        instance = cls.__new__(cls)
        instance.value = children[0]  # Normalized values (ones)
        return instance


@register_pytree_node_class
class NormalizedParameter(Parameter):
    """
    Parameter that stores normalized values (ones) internally but presents
    scaled values (scale * ones) to the outside world.
    
    This enables optimization with normalized coordinates where gradients
    have consistent magnitudes across different parameter scales, while
    still returning properly scaled values for computation.
    
    Parameters
    ----------
    value : Union[float, int, jnp.ndarray]
        The original parameter value used to compute the static scale.
        
    Examples
    --------
    >>> param = NormalizedParameter(jnp.array([2.0, 4.0, 6.0]))
    >>> param.value  # Internal normalized storage (ones)
    Array([1., 1., 1.], dtype=float32)
    >>> param.__jax_array__()  # External scaled values (scale * ones)
    Array([2., 4., 6.], dtype=float32)
    >>> param.scale  # Static scale factor
    Array([2., 4., 6.], dtype=float32)
    """
    
    def __init__(self, value: Union[float, int, jnp.ndarray]) -> None:
        # Store original value as static scale  
        original_value = jnp.asarray(value)
        self.scale = original_value
        
        # Store normalized values (ones) internally
        normalized_value = jnp.ones_like(original_value)
        
        # Initialize parent with normalized value
        super().__init__(normalized_value)
    
    def __repr__(self) -> str:
        return f"NormalizedParameter(scale={self.scale}, normalized={self.value}, value ={self.__jax_array__()})"
    
    def __jax_array__(self) -> jnp.ndarray:
        """Return scaled values: scale * normalized_ones."""
        return self.scale * self.value  # scale * ones = original values
    
    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray], jnp.ndarray]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)  # Only normalized ones are differentiable
        aux_data = self.scale     # Scale is static (not differentiable)
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data: jnp.ndarray, 
                      children: Tuple[jnp.ndarray]) -> 'NormalizedParameter':
        """Unflatten for JAX pytree registration."""
        # Reconstruct from normalized values and static scale
        instance = cls.__new__(cls)
        instance.value = children[0]  # Normalized values (ones)
        instance.scale = aux_data     # Static scale factor
        return instance


@register_pytree_node_class
class TransformedParameter(Parameter):
    """
    Parameter with custom forward and reverse transformations.

    This enables arbitrary parameter transformations by applying:
    - inverse transform: constrained → unconstrained (at initialization)
    - forward transform: unconstrained → constrained (when used as array)

    The parameter is stored internally in unconstrained space for smooth
    optimization, but presents constrained values when used.

    Parameters
    ----------
    value : Union[float, int, jnp.ndarray]
        The initial constrained parameter value.
    forward_transform : callable
        Function to transform from unconstrained to constrained space.
        Applied every time the parameter is used as a JAX array.
    inverse_transform : callable
        Function to transform from constrained to unconstrained space.
        Applied once at initialization.

    Examples
    --------
    >>> # Log-normal parameter
    >>> forward = lambda x: jnp.exp(x)  # unconstrained → constrained
    >>> inverse = lambda x: jnp.log(x)  # constrained → unconstrained
    >>> param = TransformedParameter(2.0, forward, inverse)
    >>> param.__jax_array__()  # Returns exp(log(2.0)) = 2.0

    >>> # Sigmoid bounded parameter
    >>> forward = lambda x: jax.nn.sigmoid(x)  # unconstrained → constrained
    >>> inverse = lambda x: jnp.log(x / (1 - x))  # constrained → unconstrained
    >>> param = TransformedParameter(0.7, forward, inverse)
    """

    def __init__(self, value: Union[float, int, jnp.ndarray],
                 forward_transform: callable, inverse_transform: callable) -> None:
        # Apply inverse transform to store in unconstrained space
        unconstrained_value = inverse_transform(jnp.asarray(value))
        super().__init__(unconstrained_value)
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

    def __repr__(self) -> str:
        return f"TransformedParameter(original={self.__jax_array__()}, transformed={self.value})"


    def __jax_array__(self) -> jnp.ndarray:
        """Return transformed value in constrained space."""
        return self.forward_transform(self.value)

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray], Tuple[callable, callable]]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)  # Only unconstrained values are differentiable
        aux_data = (self.forward_transform, self.inverse_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[callable, callable],
                      children: Tuple[jnp.ndarray]) -> 'TransformedParameter':
        """Unflatten for JAX pytree registration."""
        forward_transform, inverse_transform = aux_data
        instance = cls.__new__(cls)
        instance.value = children[0]  # Unconstrained values
        instance.forward_transform = forward_transform
        instance.inverse_transform = inverse_transform
        return instance


@register_pytree_node_class
class SigmoidBoundedParameter(TransformedParameter):
    """
    Parameter with sigmoid-based bounds enforcement.

    Uses sigmoid transformation to map from unconstrained real space to
    bounded interval [low, high]. This provides smooth gradients and
    avoids the gradient issues of hard clipping.

    Parameters
    ----------
    value : Union[float, int, jnp.ndarray]
        The initial parameter value (will be clipped to bounds).
    low : float
        Lower bound for the parameter value.
    high : float
        Upper bound for the parameter value.

    Examples
    --------
    >>> param = SigmoidBoundedParameter(0.7, low=0.0, high=1.0)
    >>> param.__jax_array__()  # Returns value in [0, 1]
    >>> # Gradients flow smoothly through sigmoid transformation
    """

    def __init__(self, value: Union[float, int, jnp.ndarray],
                 low: float, high: float) -> None:
        if low >= high:
            raise ValueError(f"Invalid bounds: low ({low}) must be < high ({high})")

        self.low = low
        self.high = high

        # Clip initial value to bounds
        clipped_value = jnp.clip(jnp.asarray(value), low, high)

        def forward_transform(x):
            """Map from unconstrained space to [low, high] using sigmoid."""
            return low + (high - low) * jax.nn.sigmoid(x)

        def inverse_transform(x):
            """Map from [low, high] to unconstrained space using logit."""
            # Normalize to [0, 1]
            normalized = (x - low) / (high - low)
            # Clamp to avoid numerical issues with log(0) or log(inf)
            normalized = jnp.clip(normalized, 1e-7, 1 - 1e-7)
            # Apply logit
            return jnp.log(normalized / (1 - normalized))

        super().__init__(clipped_value, forward_transform, inverse_transform)

    def __repr__(self) -> str:
        return f"SigmoidBoundedParameter({self.__jax_array__()}, low={self.low}, high={self.high})"

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray], Tuple[float, float, callable, callable]]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)
        aux_data = (self.low, self.high, self.forward_transform, self.inverse_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[float, float, callable, callable],
                      children: Tuple[jnp.ndarray]) -> 'SigmoidBoundedParameter':
        """Unflatten for JAX pytree registration."""
        low, high, forward_transform, inverse_transform = aux_data
        instance = cls.__new__(cls)
        instance.value = children[0]
        instance.low = low
        instance.high = high
        instance.forward_transform = forward_transform
        instance.inverse_transform = inverse_transform
        return instance


@register_pytree_node_class
class LogPositiveParameter(TransformedParameter):
    """
    Parameter constrained to be positive using log transformation.

    Maps from unconstrained real space to (lower, ∞) using log transformation.
    This provides smooth gradients and is ideal for scale parameters like
    conductances, time constants, etc.

    Parameters
    ----------
    value : Union[float, int, jnp.ndarray]
        The initial parameter value (must be > lower).
    lower : float, optional
        Lower bound for the parameter value. Default is 0.0.

    Examples
    --------
    >>> param = LogPositiveParameter(2.0)  # Always positive
    >>> param = LogPositiveParameter(5.0, lower=1.0)  # Always > 1.0
    """

    def __init__(self, value: Union[float, int, jnp.ndarray], lower: float = 0.0) -> None:
        value_array = jnp.asarray(value)
        if jnp.any(value_array <= lower):
            raise ValueError(f"Initial value must be > {lower}, got {value}")

        self.lower = lower

        def forward_transform(x):
            """Map from unconstrained space to (lower, ∞) using exp."""
            return lower + jnp.exp(x)

        def inverse_transform(x):
            """Map from (lower, ∞) to unconstrained space using log."""
            return jnp.log(x - lower)

        super().__init__(value, forward_transform, inverse_transform)

    def __repr__(self) -> str:
        return f"LogPositiveParameter({self.__jax_array__()}, lower={self.lower})"

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray], Tuple[float, callable, callable]]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)
        aux_data = (self.lower, self.forward_transform, self.inverse_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[float, callable, callable],
                      children: Tuple[jnp.ndarray]) -> 'LogPositiveParameter':
        """Unflatten for JAX pytree registration."""
        lower, forward_transform, inverse_transform = aux_data
        instance = cls.__new__(cls)
        instance.value = children[0]
        instance.lower = lower
        instance.forward_transform = forward_transform
        instance.inverse_transform = inverse_transform
        return instance


@register_pytree_node_class
class LogNegativeParameter(TransformedParameter):
    """
    Parameter constrained to be negative using log transformation.

    Maps from unconstrained real space to (-∞, upper) using negative log transformation.
    Useful for parameters that must remain below a certain threshold.

    Parameters
    ----------
    value : Union[float, int, jnp.ndarray]
        The initial parameter value (must be < upper).
    upper : float, optional
        Upper bound for the parameter value. Default is 0.0.

    Examples
    --------
    >>> param = LogNegativeParameter(-2.0)  # Always negative
    >>> param = LogNegativeParameter(-1.0, upper=5.0)  # Always < 5.0
    """

    def __init__(self, value: Union[float, int, jnp.ndarray], upper: float = 0.0) -> None:
        value_array = jnp.asarray(value)
        if jnp.any(value_array >= upper):
            raise ValueError(f"Initial value must be < {upper}, got {value}")

        self.upper = upper

        def forward_transform(x):
            """Map from unconstrained space to (-∞, upper) using -exp."""
            return upper - jnp.exp(x)

        def inverse_transform(x):
            """Map from (-∞, upper) to unconstrained space using log."""
            return jnp.log(upper - x)

        super().__init__(value, forward_transform, inverse_transform)

    def __repr__(self) -> str:
        return f"LogNegativeParameter({self.__jax_array__()}, upper={self.upper})"

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray], Tuple[float, callable, callable]]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)
        aux_data = (self.upper, self.forward_transform, self.inverse_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[float, callable, callable],
                      children: Tuple[jnp.ndarray]) -> 'LogNegativeParameter':
        """Unflatten for JAX pytree registration."""
        upper, forward_transform, inverse_transform = aux_data
        instance = cls.__new__(cls)
        instance.value = children[0]
        instance.upper = upper
        instance.forward_transform = forward_transform
        instance.inverse_transform = inverse_transform
        return instance


@register_pytree_node_class
class MaskedParameter(TransformedParameter):
    """
    Parameter that keeps masked entries fixed at their initial values.

    This allows selective optimization where only certain entries in an array
    are subject to optimization while others remain frozen. Useful for maintaining
    structural constraints like sparsity patterns, symmetries, or fixed values.

    Parameters
    ----------
    value : Union[float, int, jnp.ndarray]
        The initial parameter values.
    mask : jnp.ndarray
        Boolean mask where True indicates optimizable entries and False indicates
        frozen entries that maintain their initial values.

    Examples
    --------
    >>> # Keep zero entries frozen, optimize non-zero entries
    >>> value = jnp.array([1.0, 0.0, 3.0, 0.0, 5.0])
    >>> mask = value != 0.0  # True for non-zero, False for zero
    >>> param = MaskedParameter(value, mask)
    >>> # Only positions [0, 2, 4] will be optimized

    >>> # Upper triangular matrix optimization
    >>> matrix = jnp.array([[1, 2], [0, 3]])
    >>> mask = jnp.triu(jnp.ones_like(matrix)).astype(bool)
    >>> param = MaskedParameter(matrix, mask)
    """

    def __init__(self, value: Union[float, int, jnp.ndarray],
                 mask: jnp.ndarray) -> None:
        value_array = jnp.asarray(value)
        self.mask = jnp.asarray(mask, dtype=bool)

        # Ensure mask and value have compatible shapes
        if self.mask.shape != value_array.shape:
            raise ValueError(f"Mask shape {self.mask.shape} must match value shape {value_array.shape}")

        # Store frozen values (entries where mask is False)
        self.frozen_values = jnp.where(self.mask, 0.0, value_array)

        def forward_transform(x):
            """Apply mask: use optimized values where mask=True, frozen values where mask=False."""
            return jnp.where(self.mask, x, self.frozen_values)

        def inverse_transform(x):
            """Extract optimizable values: set frozen entries to 0 in unconstrained space."""
            return jnp.where(self.mask, x, 0.0)

        super().__init__(value, forward_transform, inverse_transform)

    def __repr__(self) -> str:
        n_optimizable = jnp.sum(self.mask)
        n_frozen = jnp.sum(~self.mask)
        return f"MaskedParameter(shape={self.mask.shape}, optimizable={n_optimizable}, frozen={n_frozen})"

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, callable, callable]]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)
        aux_data = (self.mask, self.frozen_values, self.forward_transform, self.inverse_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[jnp.ndarray, jnp.ndarray, callable, callable],
                      children: Tuple[jnp.ndarray]) -> 'MaskedParameter':
        """Unflatten for JAX pytree registration."""
        mask, frozen_values, forward_transform, inverse_transform = aux_data
        instance = cls.__new__(cls)
        instance.value = children[0]
        instance.mask = mask
        instance.frozen_values = frozen_values
        instance.forward_transform = forward_transform
        instance.inverse_transform = inverse_transform
        return instance


@register_pytree_node_class
class BoundedParameter(Parameter):
    """
    Parameter with automatic bounds enforcement.
    
    The bounds are applied transparently whenever the parameter is used
    as a JAX array, ensuring constraints are always satisfied.
    
    Parameters
    ----------
    value : Union[float, int, jnp.ndarray]
        The parameter value. Will be converted to JAX array.
    low : float
        Lower bound for the parameter value.
    high : float  
        Upper bound for the parameter value.
    
    Examples
    --------
    >>> param = BoundedParameter(1.5, low=0.0, high=1.0)
    >>> result = param + 0.1  # Automatically clips to [0, 1] 
    >>> print(result)  # 1.1
    """
    
    def __init__(self, value: Union[float, int, jnp.ndarray], 
                 low: float, high: float) -> None:
        super().__init__(value)
        self.low = low
        self.high = high
        
        if low >= high:
            raise ValueError(f"Invalid bounds: low ({low}) must be < high ({high})")
    
    def __repr__(self) -> str:
        return f"BoundedParameter({self.value}, low={self.low}, high={self.high})"
        # return f"BoundedParameter({self.__jax_array__()}, low={self.low}, high={self.high})"
    
    def __jax_array__(self) -> jnp.ndarray:
        """Return clipped value respecting bounds."""
        return jnp.clip(self.value, self.low, self.high)
    
    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray], Tuple[float, float]]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)
        aux_data = (self.low, self.high)
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[float, float], 
                      children: Tuple[jnp.ndarray]) -> 'BoundedParameter':
        """Unflatten for JAX pytree registration."""
        low, high = aux_data
        return cls(children[0], low, high)


# Utility functions for working with parameters
def is_parameter(obj: Any) -> bool:
    """Check if object is a Parameter instance."""
    return isinstance(obj, Parameter)


def extract_values(tree: Any) -> Any:
    """
    Extract parameter values from a pytree, leaving other types unchanged.
    
    This replaces the old collect_parameters() function.
    
    Parameters
    ----------
    tree : Any
        PyTree potentially containing Parameter objects
        
    Returns
    -------
    Any
        PyTree with Parameter objects replaced by their values
    """
    return jax.tree.map(
        lambda x: x.__jax_array__() if is_parameter(x) else x,
        tree,
        is_leaf=is_parameter
    )