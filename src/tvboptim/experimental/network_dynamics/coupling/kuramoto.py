"""Kuramoto-style phase-difference coupling.

Implements the sin(theta_j - theta_i) interaction term of the Kuramoto model
of coupled phase oscillators. dynamics.tvb.Kuramoto expects this transform to
be supplied by the coupling's pre()/post() (the TVB-style separation of
dynamics and coupling), so it has no coupling of its own -- these are it.
"""

import jax.numpy as jnp

from ..core.bunch import Bunch
from .base import DelayedCoupling, InstantaneousCoupling


class KuramotoCoupling(InstantaneousCoupling):
    """Phase-difference coupling for Kuramoto oscillators (no delay).

    Implements the classic Kuramoto interaction:

    $$c_i = G \\cdot \\sum_{j} w_{ij} \\sin(\\theta_j - \\theta_i)$$

    Parameters
    ----------
    incoming_states : str or list of str
        State name(s) to collect from connected nodes (typically ``'theta'``)
    local_states : str or list of str
        State name(s) from current node (required for the phase difference)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (global coupling strength)

    Notes
    -----
    G is not normalized by network size or degree; scale it (e.g. G/N) to
    match a particular Kuramoto convention.

    Examples
    --------
    >>> coupling = KuramotoCoupling(incoming_states='theta', local_states='theta', G=1.0)
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0)
    PRE_USES_LOCAL = True

    def pre(
        self, incoming_states: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Compute sin(theta_j - theta_i) per edge.

        Args:
            incoming_states: Source phases ``[n_incoming, *M]``.
            local_states: Target phases aligned as ``[n_local, *M]``.
            params: Coupling parameters (not used in pre)

        Returns:
            Phase-difference sine ``[n_output, *M]``.
        """
        return jnp.sin(incoming_states - local_states)

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply coupling strength to summed phase interactions.

        Args:
            summed_inputs: Summed sin(theta_j - theta_i) terms [n_inputs, n_nodes]
            local_states: Local states (not used)
            params: Bunch with G

        Returns:
            Scaled coupling [n_inputs, n_nodes]
        """
        return params.G * summed_inputs


class DelayedKuramotoCoupling(DelayedCoupling):
    """Phase-difference coupling for Kuramoto oscillators with transmission delays.

    Implements the delayed Kuramoto interaction:

    $$c_i(t) = G \\cdot \\sum_{j} w_{ij} \\sin(\\theta_j(t - \\tau_{ij}) - \\theta_i(t))$$

    where $\\tau_{ij}$ are the transmission delays between nodes. This is the
    standard model used to study delay-induced (de)synchronization, e.g. the
    two-oscillator multistability of Yeung & Strogatz (1999) and the
    conduction-speed-dependent synchronization resonances of Petkoski & Jirsa
    (2019) on brain networks.

    Parameters
    ----------
    incoming_states : str or list of str
        State name(s) to collect from connected nodes (typically ``'theta'``)
    local_states : str or list of str
        State name(s) from current node (required for the phase difference)

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ``1``
    DEFAULT_PARAMS : Bunch
        Default parameters: ``G=1.0`` (global coupling strength)

    Notes
    -----
    G is not normalized by network size or degree; scale it (e.g. G/N) to
    match a particular Kuramoto convention.

    Examples
    --------
    >>> coupling = DelayedKuramotoCoupling(incoming_states='theta', local_states='theta', G=1.0)

    References
    ----------
    Yeung, M. K. S., & Strogatz, S. H. (1999). Time delay in the Kuramoto
    model of coupled oscillators. Physical Review Letters, 82(3), 648.

    Petkoski, S., & Jirsa, V. K. (2019). Transmission time delays organize
    the brain network synchronization. Philosophical Transactions of the
    Royal Society A, 377(2153), 20180132.
    """

    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0)
    PRE_USES_LOCAL = True

    def pre(
        self, delayed_states: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Compute sin(theta_j(t - tau) - theta_i(t)) per edge.

        Args:
            delayed_states: Delayed source phases ``[n_incoming, *M]``.
            local_states: Current target phases aligned as ``[n_local, *M]``.
            params: Coupling parameters (not used in pre)

        Returns:
            Delayed phase-difference sine ``[n_output, *M]``.
        """
        return jnp.sin(delayed_states - local_states)

    def post(
        self, summed_inputs: jnp.ndarray, local_states: jnp.ndarray, params: Bunch
    ) -> jnp.ndarray:
        """Apply coupling strength to summed delayed phase interactions.

        Args:
            summed_inputs: Summed delayed sin(theta_j - theta_i) terms [n_inputs, n_nodes]
            local_states: Local states (not used)
            params: Bunch with G

        Returns:
            Scaled coupling [n_inputs, n_nodes]
        """
        return params.G * summed_inputs
