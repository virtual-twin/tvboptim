"""Concrete coupling implementations.

This module provides common coupling functions:
- LinearCoupling: Simple linear coupling
- DifferenceCoupling: Coupling based on state differences
- DelayedLinearCoupling: Linear coupling with delays
"""

import jax.numpy as jnp

from ..core.bunch import Bunch
from .base import DelayedCoupling, InstantaneousCoupling
from .transport import _aggregate_nodes


class LinearCoupling(InstantaneousCoupling):
    """Simple linear coupling function.

    Standard linear coupling with gain and offset parameters, implementing the
    transformation:

    $$c = G \\cdot \\sum_{j} w_{ij} x_j + b$$

    where $w_{ij}$ are the graph weights, $x_j$ are the incoming states, $G$ is the
    global coupling strength, and $b$ is an offset term.

    Parameters
    ----------
    incoming_states : str or list of str
        State name(s) to collect from connected nodes (required)
    local_states : str or list of str, optional
        State name(s) from current node (default: ``[]``)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (global coupling strength), ``b=0.0`` (offset/bias)

    Examples
    --------
    >>> # Couple via 'x' state from connected nodes
    >>> coupling = LinearCoupling(incoming_states='x', G=1.0)
    >>>
    >>> # With offset
    >>> coupling = LinearCoupling(incoming_states='x', G=0.5, b=0.1)
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0, b=0.0)

    # No pre() override - uses default identity which returns per-edge format
    # This matches v1 LinearCoupling behavior

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply linear transformation to summed inputs.

        Parameters
        ----------
        summed_inputs : jnp.ndarray
            Summed coupling inputs with shape ``[n_inputs, n_nodes]``
        local_states : jnp.ndarray
            Local states (not used)
        params : Bunch
            Parameters with G (coupling strength) and b (offset)

        Returns
        -------
        jnp.ndarray
            Transformed coupling with shape ``[n_inputs, n_nodes]``
        """
        return params.G * summed_inputs + params.b


class FastLinearCoupling(InstantaneousCoupling):
    """Fast linear coupling using vectorized mode.

    Uses local states (instead of incoming states) to trigger vectorized matrix
    multiplication instead of per-edge element-wise operations. This is significantly
    faster for dense all-to-all connectivity patterns.

    Implements the same transformation as LinearCoupling:

    $$c = G \\cdot \\sum_{j} w_{ij} x_j + b$$

    but uses optimized matrix operations.

    Parameters
    ----------
    local_states : str or list of str
        State name(s) from current node (required for vectorized mode)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (coupling strength), ``b=0.0`` (offset/bias)

    Notes
    -----
    Use this variant when you have dense connectivity and want better performance.
    For sparse graphs, the standard LinearCoupling may be more efficient.

    Examples
    --------
    >>> # Fast vectorized coupling for dense networks
    >>> coupling = FastLinearCoupling(local_states='S', G=1.0)
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0, b=0.0)

    def __init__(self, incoming_states=None, local_states=None, **kwargs):
        """Accept the historical ``local_states`` spelling as a source alias."""
        if incoming_states is not None and local_states is not None:
            raise ValueError(
                "FastLinearCoupling accepts either incoming_states or the "
                "historical local_states alias, not both"
            )
        source_states = incoming_states if incoming_states is not None else local_states
        super().__init__(incoming_states=source_states, local_states=[], **kwargs)

    def pre(self, incoming_states, local_states, params):
        """Identity transform over transmitted source states."""
        return incoming_states

    def compute(self, t, state, coupling_data, coupling_state, params, graph):
        """Preserve the historical hand-rolled node-vector fast path."""
        del t, coupling_state
        incoming_states = state[coupling_data.incoming_indices]
        local_states = state[coupling_data.local_indices]
        summed = _aggregate_nodes(incoming_states, graph.weights)
        return self.post(summed, local_states, params)

    def post(self, summed_inputs, local_states, params):
        """Apply linear transformation to summed inputs."""
        return params.G * summed_inputs + params.b


class DifferenceCoupling(InstantaneousCoupling):
    """Diffusive coupling based on state differences.

    Computes coupling based on the difference between incoming and local states,
    implementing:

    $$c_i = G \\cdot \\sum_{j} w_{ij} (x_j - x_i)$$

    This type of coupling is useful for synchronization and consensus dynamics,
    as it drives nodes toward common states. Automatically uses sparse-optimized
    computation for sparse graphs.

    Parameters
    ----------
    incoming_states : str or list of str
        State name(s) to collect from connected nodes (required)
    local_states : str or list of str
        State name(s) from current node (required for computing differences)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (global coupling strength)

    Examples
    --------
    >>> # Diffusive coupling via 'x' state
    >>> coupling = DifferenceCoupling(incoming_states='x', local_states='x', G=1.0)
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0)
    PRE_USES_LOCAL = True

    def pre(
        self, incoming_states: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Compute difference between incoming and local states (per-edge).

        Args:
            incoming_states: States from connected nodes in per-edge format
                           [n_incoming, n_nodes_target, n_nodes_source]
            local_states: States from current node [n_local, n_nodes]
            params: Coupling parameters (not used in pre)

        Returns:
            State differences [n_incoming, n_nodes, n_nodes] (per-edge)
        """
        return incoming_states - local_states

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply coupling strength to summed differences.

        Args:
            summed_inputs: Summed differences [n_inputs, n_nodes]
            local_states: Local states (not used)
            params: Bunch with G

        Returns:
            Scaled coupling [n_inputs, n_nodes]
        """
        return params.G * summed_inputs


class SigmoidCoupling(InstantaneousCoupling):
    """Linear coupling with sigmoid post-processing.

    Applies a sigmoid nonlinearity after linear transformation, implementing:

    $$c = G \\cdot \\sigma\\left(s \\cdot (a \\cdot \\sum_{j} w_{ij} x_j + b - m)\\right)$$

    where $\\sigma(x) = 1/(1+e^{-x})$ is the sigmoid function, $s$ is the slope,
    and $m$ is the midpoint. Useful for saturating coupling effects.

    Parameters
    ----------
    incoming_states : str or list of str
        State name(s) to collect from connected nodes (required)
    local_states : str or list of str, optional
        State name(s) from current node (default: ``[]``)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (global coupling strength), ``a=1.0`` (input scaling),
        ``b=0.0`` (offset), ``slope=1.0`` (sigmoid steepness), ``midpoint=0.0`` (sigmoid center)

    Examples
    --------
    >>> coupling = SigmoidCoupling(incoming_states='x', G=1.0, slope=2.0, midpoint=0.0)
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0, a=1.0, b=0.0, slope=1.0, midpoint=0.0)

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply linear transformation followed by sigmoid.

        Args:
            summed_inputs: Summed coupling inputs [n_inputs, n_nodes]
            local_states: Local states (not used)
            params: Bunch with G, a, b, slope, midpoint

        Returns:
            Sigmoid-transformed coupling [n_inputs, n_nodes]
        """
        import jax.nn

        linear = params.a * summed_inputs + params.b
        sigmoid = jax.nn.sigmoid(params.slope * (linear - params.midpoint))
        return params.G * sigmoid


class TanhCoupling(InstantaneousCoupling):
    """Coupling with hyperbolic tangent saturation.

    Applies tanh nonlinearity for symmetric saturation, implementing:

    $$c = G \\cdot \\tanh\\left(s \\cdot \\sum_{j} w_{ij} x_j\\right)$$

    where $s$ is the scaling factor. The tanh function provides symmetric saturation
    in the range $(-1, 1)$.

    Parameters
    ----------
    incoming_states : str or list of str
        State name(s) to collect from connected nodes (required)
    local_states : str or list of str, optional
        State name(s) from current node (default: ``[]``)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=0.5`` (global coupling strength), ``scale=2.0`` (scaling before tanh)

    Examples
    --------
    >>> coupling = TanhCoupling(incoming_states='x', G=0.5, scale=2.0)
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=0.5, scale=2.0)

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply tanh saturation to summed inputs.

        Args:
            summed_inputs: Summed coupling inputs [n_inputs, n_nodes]
            local_states: Local states (not used)
            params: Bunch with G, scale

        Returns:
            Tanh-transformed coupling [n_inputs, n_nodes]
        """
        return params.G * jnp.tanh(params.scale * summed_inputs)


# Delayed coupling variants


class DelayedLinearCoupling(DelayedCoupling):
    """Linear coupling with transmission delays.

    Standard linear coupling with delays between nodes, implementing:

    $$c_i(t) = G \\cdot \\sum_{j} w_{ij} x_j(t - \\tau_{ij}) + b$$

    where $\\tau_{ij}$ are the transmission delays between nodes.

    Parameters
    ----------
    incoming_states : str or list of str
        State name(s) to collect from connected nodes (required)
    local_states : str or list of str, optional
        State name(s) from current node (default: ``[]``)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (coupling strength), ``b=0.0`` (offset/bias)

    Examples
    --------
    >>> coupling = DelayedLinearCoupling(incoming_states='S', G=1.0)
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0, b=0.0)

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply linear transformation to summed delayed inputs.

        Args:
            summed_inputs: Summed delayed coupling inputs [n_inputs, n_nodes]
            local_states: Local states (not used)
            params: Bunch with G, b

        Returns:
            Transformed coupling [n_inputs, n_nodes]
        """
        return params.G * summed_inputs + params.b


class DelayedDifferenceCoupling(DelayedCoupling):
    """Diffusive coupling with transmission delays.

    Computes coupling based on the difference between delayed incoming states
    and current local states, implementing:

    $$c_i(t) = G \\cdot \\sum_{j} w_{ij} (x_j(t - \\tau_{ij}) - x_i(t))$$

    This combines diffusive coupling with delayed transmission, useful for modeling
    synchronization dynamics with finite propagation speeds.

    Parameters
    ----------
    incoming_states : str or list of str
        State name(s) to collect from connected nodes (required)
    local_states : str or list of str
        State name(s) from current node (required for computing differences)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (global coupling strength)

    Examples
    --------
    >>> coupling = DelayedDifferenceCoupling(incoming_states='x', local_states='x', G=1.0)
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0)
    PRE_USES_LOCAL = True

    def pre(
        self, delayed_states: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Compute difference between delayed and local states (per-edge).

        Args:
            delayed_states: Delayed states from history in per-edge format
                          [n_incoming, n_nodes_target, n_nodes_source]
            local_states: Current local states [n_local, n_nodes]
            params: Coupling parameters (not used in pre)

        Returns:
            State differences [n_incoming, n_nodes, n_nodes] (per-edge)
        """
        return delayed_states - local_states

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply coupling strength to summed delayed differences.

        Args:
            summed_inputs: Summed differences [n_inputs, n_nodes]
            local_states: Local states (not used)
            params: Bunch with G

        Returns:
            Scaled coupling [n_inputs, n_nodes]
        """
        return params.G * summed_inputs


class DelayedSigmoidCoupling(DelayedCoupling):
    """Sigmoid coupling with transmission delays.

    Applies sigmoid nonlinearity to delayed coupling inputs, implementing:

    $$c_i(t) = G \\cdot \\sigma\\left(s \\cdot \\left(\\sum_{j} w_{ij} x_j(t - \\tau_{ij}) - m\\right)\\right)$$

    where $\\sigma(x) = 1/(1+e^{-x})$ is the sigmoid function, $s$ is the slope,
    and $m$ is the midpoint.

    Parameters
    ----------
    incoming_states : str or list of str
        State name(s) to collect from connected nodes (required)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (global coupling strength), ``slope=1.0`` (sigmoid steepness),
        ``midpoint=0.0`` (sigmoid center)

    Examples
    --------
    >>> coupling = DelayedSigmoidCoupling(incoming_states='x', G=1.0, slope=1.0, midpoint=0.0)
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0, slope=1.0, midpoint=0.0)

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply sigmoid to summed delayed inputs.

        Args:
            summed_inputs: Summed delayed coupling inputs [n_inputs, n_nodes]
            local_states: Local states (not used)
            params: Bunch with G, slope, midpoint

        Returns:
            Sigmoid-transformed coupling [n_inputs, n_nodes]
        """
        import jax.nn

        sigmoid = jax.nn.sigmoid(params.slope * (summed_inputs - params.midpoint))
        return params.G * sigmoid
