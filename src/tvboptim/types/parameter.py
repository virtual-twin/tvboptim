"""
New minimal Parameter implementation for TVB-Optim refactoring.

This module implements a simplified Parameter class that:
- Removes redundant fields (name, doc, free/grad)
- Provides native JAX integration via __jax_array__
- Implements full arithmetic operations
- Supports specialized subclasses (BoundedParameter)
"""

from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class


def _coerce_operand(value: Any) -> Any:
    """Convert array-like operands only when they expose a callable protocol."""
    jax_array = getattr(value, "__jax_array__", None)
    return jax_array() if callable(jax_array) else value


def _is_concrete(value: jnp.ndarray) -> bool:
    """Whether a value can be inspected on the host.

    Constructors that reject out-of-range input must not branch on a traced
    array, so the check is skipped under a trace. The value is still clamped
    into the valid range in that case, which keeps the leaf finite.
    """
    return not isinstance(value, jax.core.Tracer)


@register_pytree_node_class
class Parameter:
    """
    A minimal JAX-native parameter with full arithmetic support.

    If placed at a position in the state tree, that position will take part in optimizations and will be differentiated. A state can be split into parameters and static parts by partition_state and combined by combine_state.

    ``.value`` is the raw differentiable leaf updated by the optimizer. The value
    used in computation is ``.constrained_value`` (or ``collect_parameters``);
    for plain ``Parameter`` the two coincide, but subclasses differ.

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
        return np.array(
            self.__jax_array__()
        )  # Return JAX array directly instead of converting to numpy

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
                raise ValueError(
                    f"Parameter shape {shape} does not match value shape {self.shape} and can not be broadcasted automatically."
                )

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

    @property
    def constrained_value(self) -> jnp.ndarray:
        """Value as used in computation (post-transform / post-clip).

        Unlike ``.value`` (the raw differentiable leaf), this is consistent
        across all parameter types and always satisfies the constraints.
        """
        return self.__jax_array__()

    # Arithmetic Operations - Return JAX arrays for seamless integration
    def __add__(self, other) -> jnp.ndarray:
        """Addition: param + other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() + other_val

    def __radd__(self, other) -> jnp.ndarray:
        """Reverse addition: other + param"""
        other_val = _coerce_operand(other)
        return other_val + self.__jax_array__()

    def __sub__(self, other) -> jnp.ndarray:
        """Subtraction: param - other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() - other_val

    def __rsub__(self, other) -> jnp.ndarray:
        """Reverse subtraction: other - param"""
        other_val = _coerce_operand(other)
        return other_val - self.__jax_array__()

    def __mul__(self, other) -> jnp.ndarray:
        """Multiplication: param * other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() * other_val

    def __rmul__(self, other) -> jnp.ndarray:
        """Reverse multiplication: other * param"""
        other_val = _coerce_operand(other)
        return other_val * self.__jax_array__()

    def __truediv__(self, other) -> jnp.ndarray:
        """Division: param / other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() / other_val

    def __rtruediv__(self, other) -> jnp.ndarray:
        """Reverse division: other / param"""
        other_val = _coerce_operand(other)
        return other_val / self.__jax_array__()

    def __floordiv__(self, other) -> jnp.ndarray:
        """Floor division: param // other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() // other_val

    def __rfloordiv__(self, other) -> jnp.ndarray:
        """Reverse floor division: other // param"""
        other_val = _coerce_operand(other)
        return other_val // self.__jax_array__()

    def __mod__(self, other) -> jnp.ndarray:
        """Modulo: param % other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() % other_val

    def __rmod__(self, other) -> jnp.ndarray:
        """Reverse modulo: other % param"""
        other_val = _coerce_operand(other)
        return other_val % self.__jax_array__()

    def __pow__(self, other) -> jnp.ndarray:
        """Power: param ** other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() ** other_val

    def __rpow__(self, other) -> jnp.ndarray:
        """Reverse power: other ** param"""
        other_val = _coerce_operand(other)
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
        other_val = _coerce_operand(other)
        return self.__jax_array__() == other_val

    def __ne__(self, other) -> jnp.ndarray:
        """Inequality: param != other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() != other_val

    def __lt__(self, other) -> jnp.ndarray:
        """Less than: param < other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() < other_val

    def __le__(self, other) -> jnp.ndarray:
        """Less than or equal: param <= other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() <= other_val

    def __gt__(self, other) -> jnp.ndarray:
        """Greater than: param > other"""
        other_val = _coerce_operand(other)
        return self.__jax_array__() > other_val

    def __ge__(self, other) -> jnp.ndarray:
        """Greater than or equal: param >= other"""
        other_val = _coerce_operand(other)
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
    def tree_unflatten(
        cls, aux_data: None, children: Tuple[jnp.ndarray]
    ) -> "Parameter":
        """Unflatten for JAX pytree registration."""
        instance = cls.__new__(cls)
        instance.value = children[0]  # Normalized values (ones)
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

    Note: ``.value`` is the unconstrained pre-image; the constrained value is
    ``forward_transform(.value)``, exposed as ``.constrained_value``.

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

    def __init__(
        self,
        value: Union[float, int, jnp.ndarray],
        forward_transform: callable,
        inverse_transform: callable,
    ) -> None:
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
    def tree_unflatten(
        cls, aux_data: Tuple[callable, callable], children: Tuple[jnp.ndarray]
    ) -> "TransformedParameter":
        """Unflatten for JAX pytree registration."""
        forward_transform, inverse_transform = aux_data
        instance = cls.__new__(cls)
        instance.value = children[0]  # Unconstrained values
        instance.forward_transform = forward_transform
        instance.inverse_transform = inverse_transform
        return instance


@register_pytree_node_class
class RescaledParameter(TransformedParameter):
    """
    Parameter optimized in units of a chosen scale.

    Gradient optimizers step the differentiable leaf, not the physical value,
    and Adam in particular moves each leaf by roughly the learning rate per
    step whatever the gradient magnitude. Parameters that live on different
    numeric scales therefore explore by very different *relative* amounts under
    a single learning rate. Dividing each leaf by a scale of your choosing puts
    them on a common footing.

    Pick `scale` as the range the parameter should explore. That is often, but
    not always, its starting value.

    Parameters
    ----------
    value : Union[float, int, jnp.ndarray]
        Initial value, in physical units.
    scale : Union[float, int, jnp.ndarray]
        Divisor applied to obtain the differentiable leaf. Must be non-zero,
        and broadcastable against `value`.

    Notes
    -----
    When the meaningful range of a parameter straddles zero, scaling by its own
    starting value is the wrong choice: a parameter starting at 0.01 that has
    to reach -0.05 would need its leaf to travel from 1 to -5, which a learning
    rate tuned for the other parameters will never cover. Scale it by the width
    of the range it must cross instead.

    Use `NormalizedParameter` for the common case where the starting value is
    itself the right scale.

    Examples
    --------
    >>> # A coupling gain that should explore roughly its own magnitude
    >>> g = RescaledParameter(0.15, scale=0.15)
    >>> g.value            # leaf, order one
    Array(1., dtype=float32)
    >>> g.constrained_value  # physical value
    Array(0.15, dtype=float32)
    >>>
    >>> # A bifurcation parameter starting near zero that must cross it
    >>> a = RescaledParameter(0.01, scale=0.1)
    >>> a.value            # leaf is 0.1, so a step of 1.0 reaches -0.09
    Array(0.1, dtype=float32)
    """

    def __init__(
        self,
        value: Union[float, int, jnp.ndarray],
        scale: Union[float, int, jnp.ndarray],
    ) -> None:
        self._initialize_rescaling(value, scale, validate_scale=True)

    def _initialize_rescaling(self, value, scale, *, validate_scale: bool) -> None:
        scale = jnp.asarray(scale)
        if validate_scale and _is_concrete(scale) and bool(jnp.any(scale == 0)):
            raise ValueError("scale must be non-zero")
        # The scale is captured in the transforms rather than stored as pytree
        # metadata. Metadata must be hashable with simple equality, which
        # arrays are not, so holding it there breaks any structural comparison
        # (lax.scan carries, value_and_grad output checks) once the values are
        # traced or non-scalar.
        super().__init__(value, lambda x: x * scale, lambda x: x / scale)

    @property
    def scale(self) -> jnp.ndarray:
        """The divisor relating the leaf to the physical value."""
        return self.forward_transform(jnp.ones(()))

    def __repr__(self) -> str:
        return (
            f"RescaledParameter(value={self.__jax_array__()}, "
            f"scale={self.scale}, leaf={self.value})"
        )


@register_pytree_node_class
class NormalizedParameter(RescaledParameter):
    """
    Parameter rescaled by its own initial value, so its leaf starts at one.

    The common case of `RescaledParameter`: the starting value is also the
    scale, which suits parameters that should explore a range comparable to
    where they begin. `.value` holds ones and `.constrained_value` returns
    `scale * .value`.

    Prefer `RescaledParameter` with an explicit scale when the parameter starts
    near zero or has to change sign, since its own start is then far smaller
    than the range it needs to cover.

    Parameters
    ----------
    value : Union[float, int, jnp.ndarray]
        The original parameter value, used as the static scale.

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
        value = jnp.asarray(value)
        # Zero starting values intentionally have zero scale: their normalized
        # leaf is set to one below and their physical value remains fixed at
        # zero. Explicit RescaledParameter scales remain strictly non-zero.
        self._initialize_rescaling(value, scale=value, validate_scale=False)
        # scale == value, so the leaf is ones. Set it directly rather than
        # relying on value / scale, which is nan where an entry is zero.
        self.value = jnp.ones_like(value)

    def __repr__(self) -> str:
        return (
            f"NormalizedParameter(scale={self.scale}, normalized={self.value}, "
            f"value ={self.__jax_array__()})"
        )


@register_pytree_node_class
class SigmoidBoundedParameter(TransformedParameter):
    """
    Parameter with sigmoid-based bounds enforcement.

    Uses sigmoid transformation to map from unconstrained real space to
    bounded interval [low, high]. This provides smooth gradients and
    avoids the gradient issues of hard clipping.

    Note: ``.value`` is in logit space and may be negative; this is expected.
    Use ``.constrained_value`` for the value in [low, high].

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

    def __init__(
        self, value: Union[float, int, jnp.ndarray], low: float, high: float
    ) -> None:
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

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[jnp.ndarray], Tuple[float, float, callable, callable]]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)
        aux_data = (self.low, self.high, self.forward_transform, self.inverse_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[float, float, callable, callable],
        children: Tuple[jnp.ndarray],
    ) -> "SigmoidBoundedParameter":
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

    def __init__(
        self, value: Union[float, int, jnp.ndarray], lower: float = 0.0
    ) -> None:
        value_array = jnp.asarray(value)
        self.validate_initial_value(value_array, lower=lower)

        if not jnp.issubdtype(value_array.dtype, jnp.inexact):
            value_array = value_array.astype(jnp.result_type(value_array, 1.0))

        self.lower = lower

        # log(x - lower) is -inf at the bound and nan below it, so the offset is
        # held just inside the open interval. Concrete input has already been
        # rejected above; under a trace this is what keeps the leaf finite.
        # The clamp is applied to the offset rather than to x, because
        # `lower + tiny` rounds back to `lower` for any nonzero bound.
        tiny = jnp.finfo(value_array.dtype).tiny

        def forward_transform(x):
            """Map from unconstrained space to (lower, ∞) using exp."""
            return lower + jnp.exp(x)

        def inverse_transform(x):
            """Map from (lower, ∞) to unconstrained space using log."""
            return jnp.log(jnp.maximum(x - lower, tiny))

        super().__init__(value_array, forward_transform, inverse_transform)

    @staticmethod
    def validate_initial_value(value, *, lower: float = 0.0) -> None:
        """Reject concrete values outside the open lower bound."""
        value_array = jnp.asarray(value)
        if _is_concrete(value_array) and bool(jnp.any(value_array <= lower)):
            raise ValueError(f"Initial value must be > {lower}, got {value}")

    def __repr__(self) -> str:
        return f"LogPositiveParameter({self.__jax_array__()}, lower={self.lower})"

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[jnp.ndarray], Tuple[float, callable, callable]]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)
        aux_data = (self.lower, self.forward_transform, self.inverse_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Tuple[float, callable, callable], children: Tuple[jnp.ndarray]
    ) -> "LogPositiveParameter":
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

    def __init__(
        self, value: Union[float, int, jnp.ndarray], upper: float = 0.0
    ) -> None:
        value_array = jnp.asarray(value)
        self.validate_initial_value(value_array, upper=upper)

        if not jnp.issubdtype(value_array.dtype, jnp.inexact):
            value_array = value_array.astype(jnp.result_type(value_array, 1.0))

        self.upper = upper

        # log(upper - x) is -inf at the bound and nan above it, so the offset is
        # held just inside the open interval. See LogPositiveParameter.
        tiny = jnp.finfo(value_array.dtype).tiny

        def forward_transform(x):
            """Map from unconstrained space to (-∞, upper) using -exp."""
            return upper - jnp.exp(x)

        def inverse_transform(x):
            """Map from (-∞, upper) to unconstrained space using log."""
            return jnp.log(jnp.maximum(upper - x, tiny))

        super().__init__(value_array, forward_transform, inverse_transform)

    @staticmethod
    def validate_initial_value(value, *, upper: float = 0.0) -> None:
        """Reject concrete values outside the open upper bound."""
        value_array = jnp.asarray(value)
        if _is_concrete(value_array) and bool(jnp.any(value_array >= upper)):
            raise ValueError(f"Initial value must be < {upper}, got {value}")

    def __repr__(self) -> str:
        return f"LogNegativeParameter({self.__jax_array__()}, upper={self.upper})"

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[jnp.ndarray], Tuple[float, callable, callable]]:
        """Flatten for JAX pytree registration."""
        children = (self.value,)
        aux_data = (self.upper, self.forward_transform, self.inverse_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Tuple[float, callable, callable], children: Tuple[jnp.ndarray]
    ) -> "LogNegativeParameter":
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

    def __init__(
        self, value: Union[float, int, jnp.ndarray], mask: jnp.ndarray
    ) -> None:
        value_array = jnp.asarray(value)
        mask = jnp.asarray(mask, dtype=bool)

        # Ensure mask and value have compatible shapes
        if mask.shape != value_array.shape:
            raise ValueError(
                f"Mask shape {mask.shape} must match value shape {value_array.shape}"
            )

        # Frozen values (entries where mask is False). Both this and the mask
        # are captured by the transforms below rather than stored as pytree
        # metadata; see the `mask` property.
        frozen_values = jnp.where(mask, 0.0, value_array)

        def forward_transform(x):
            """Apply mask: use optimized values where mask=True, frozen values where mask=False."""
            return jnp.where(mask, x, frozen_values)

        def inverse_transform(x):
            """Extract optimizable values: set frozen entries to 0 in unconstrained space."""
            return jnp.where(mask, x, 0.0)

        super().__init__(value, forward_transform, inverse_transform)

    @property
    def mask(self) -> jnp.ndarray:
        """Which entries are optimized.

        Recovered from the transform rather than stored as pytree metadata,
        which must be hashable with simple equality and so cannot hold arrays.
        `inverse_transform` maps ones to one where the entry is free and zero
        where it is frozen.
        """
        return self.inverse_transform(jnp.ones_like(self.value)).astype(bool)

    @property
    def frozen_values(self) -> jnp.ndarray:
        """The values held fixed at masked-out entries.

        `forward_transform` maps zeros to zero at free entries and to the
        frozen value elsewhere, which is exactly how they were stored.
        """
        return self.forward_transform(jnp.zeros_like(self.value))

    def __repr__(self) -> str:
        mask = self.mask
        n_optimizable = jnp.sum(mask)
        n_frozen = jnp.sum(~mask)
        return f"MaskedParameter(shape={mask.shape}, optimizable={n_optimizable}, frozen={n_frozen})"


@register_pytree_node_class
class BoundedParameter(Parameter):
    """
    Parameter with automatic bounds enforcement.

    The bounds are applied transparently whenever the parameter is used
    as a JAX array, ensuring constraints are always satisfied.

    Note: ``.value`` is the unclipped value and may sit outside the bounds;
    clipping happens on read. Use ``.constrained_value`` for the clipped value.

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

    def __init__(
        self, value: Union[float, int, jnp.ndarray], low: float, high: float
    ) -> None:
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
    def tree_unflatten(
        cls, aux_data: Tuple[float, float], children: Tuple[jnp.ndarray]
    ) -> "BoundedParameter":
        """Unflatten for JAX pytree registration."""
        low, high = aux_data
        instance = cls.__new__(cls)
        instance.value = children[0]
        instance.low = low
        instance.high = high
        return instance


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
        is_leaf=is_parameter,
    )
