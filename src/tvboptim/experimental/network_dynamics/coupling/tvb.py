"""TVB coupling implementations."""

import jax.numpy as jnp

from ..core.bunch import Bunch
from .base import InstantaneousCoupling, DelayedCoupling


class SigmoidalJansenRit(InstantaneousCoupling):
    """Sigmoidal Jansen-Rit coupling function.

    Implements the coupling function used in Jansen and Rit neural mass models.
    Applies a sigmoidal transformation to the difference between two state variables
    (typically y1 and y2) before network summation, then scales the result.

    Notes
    -----
    The coupling implements:

    $$c_i = G \\cdot \\sum_{j} w_{ij} \\sigma(y1_j - y2_j)$$

    where the sigmoid function is defined as:

    $$\\sigma(x) = c_{\\text{min}} + \\frac{c_{\\text{max}} - c_{\\text{min}}}{1 + e^{r(m - x)}}$$

    with $m$ being the midpoint and $r$ the steepness.

    Parameters
    ----------
    incoming_states : tuple of str
        Tuple of two state names, e.g., ``('y1', 'y2')`` (required)
    local_states : str or list of str, optional
        State name(s) from current node (default: ``[]``)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (global coupling strength), ``cmin=0.0`` (sigmoid minimum),
        ``cmax=0.005`` (sigmoid maximum), ``midpoint=6.0`` (sigmoid center), ``r=0.56`` (sigmoid steepness)

    Examples
    --------
    >>> # Typical Jansen-Rit coupling
    >>> coupling = SigmoidalJansenRit(
    ...     incoming_states=('y1', 'y2'),
    ...     G=1.0,
    ...     cmax=0.005
    ... )
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(
        G=1.0,
        cmin=0.0,
        cmax=0.005,  # 2 * 0.0025 from TVB default
        midpoint=6.0,
        r=0.56,
    )

    def pre(
        self, incoming_states: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply sigmoidal transformation to state difference.

        Args:
            incoming_states: States from connected nodes in per-edge format
                           [n_incoming, n_nodes_target, n_nodes_source]
                           Expected: n_incoming=2 (y1 and y2)
            local_states: Local node states (unused in this implementation)
            params: Coupling parameters (cmin, cmax, midpoint, r)

        Returns:
            Transformed states after sigmoidal coupling
            [1, n_nodes_target, n_nodes_source]
        """
        # Extract first two state variables: y1 and y2
        # incoming_states[0] = y1, incoming_states[1] = y2
        # Each has shape [n_nodes_target, n_nodes_source]
        state_diff = incoming_states[0] - incoming_states[1]

        # Apply sigmoidal transformation
        exp_term = jnp.exp(params.r * (params.midpoint - state_diff))
        coupling_term = params.cmin + (params.cmax - params.cmin) / (1.0 + exp_term)

        # Return as [1, n_nodes_target, n_nodes_source] for matrix multiplication
        return coupling_term[jnp.newaxis, :, :]

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Scale summed coupling inputs.

        Args:
            summed_inputs: Sum of coupling inputs after network summation
                         [1, n_nodes]
            local_states: Local node states (unused)
            params: Coupling parameters (G)

        Returns:
            Scaled coupling output [1, n_nodes]
        """
        return params.G * summed_inputs

class DelayedSigmoidalJansenRit(DelayedCoupling):
    """Sigmoidal Jansen-Rit coupling with transmission delays.

    Implements the delayed coupling function used in Jansen and Rit neural mass models.
    Applies a sigmoidal transformation to the difference between two delayed state variables
    (typically y1 and y2) before network summation, then scales the result.

    Notes
    -----
    The coupling implements:

    $$c_i(t) = G \\cdot \\sum_{j} w_{ij} \\sigma(y1_j(t - \\tau_{ij}) - y2_j(t - \\tau_{ij}))$$

    where $\\tau_{ij}$ are the transmission delays and the sigmoid function is defined as:

    $$\\sigma(x) = c_{\\text{min}} + \\frac{c_{\\text{max}} - c_{\\text{min}}}{1 + e^{r(m - x)}}$$

    with $m$ being the midpoint and $r$ the steepness.

    Parameters
    ----------
    incoming_states : tuple of str
        Tuple of two state names, e.g., ``('y1', 'y2')`` (required)
    local_states : str or list of str, optional
        State name(s) from current node (default: ``[]``)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (global coupling strength), ``cmin=0.0`` (sigmoid minimum),
        ``cmax=0.005`` (sigmoid maximum), ``midpoint=6.0`` (sigmoid center), ``r=0.56`` (sigmoid steepness)

    Examples
    --------
    >>> # Typical delayed Jansen-Rit coupling
    >>> coupling = DelayedSigmoidalJansenRit(
    ...     incoming_states=('y1', 'y2'),
    ...     G=1.0,
    ...     cmax=0.005
    ... )
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(
        G=1.0,
        cmin=0.0,
        cmax=0.005,  # 2 * 0.0025 from TVB default
        midpoint=6.0,
        r=0.56,
    )

    def pre(
        self, incoming_states: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply sigmoidal transformation to state difference.

        Args:
            incoming_states: States from connected nodes in per-edge format
                           [n_incoming, n_nodes_target, n_nodes_source]
                           Expected: n_incoming=2 (y1 and y2)
            local_states: Local node states (unused in this implementation)
            params: Coupling parameters (cmin, cmax, midpoint, r)

        Returns:
            Transformed states after sigmoidal coupling
            [1, n_nodes_target, n_nodes_source]
        """
        # Extract first two state variables: y1 and y2
        # incoming_states[0] = y1, incoming_states[1] = y2
        # Each has shape [n_nodes_target, n_nodes_source]
        state_diff = incoming_states[0] - incoming_states[1]

        # Apply sigmoidal transformation
        exp_term = jnp.exp(params.r * (params.midpoint - state_diff))
        coupling_term = params.cmin + (params.cmax - params.cmin) / (1.0 + exp_term)

        # Return as [1, n_nodes_target, n_nodes_source] for matrix multiplication
        return coupling_term[jnp.newaxis, :, :]

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Scale summed coupling inputs.

        Args:
            summed_inputs: Sum of coupling inputs after network summation
                         [1, n_nodes]
            local_states: Local node states (unused)
            params: Coupling parameters (G)

        Returns:
            Scaled coupling output [1, n_nodes]
        """
        return params.G * summed_inputs