"""Kuramoto phase oscillator model.

Classic model of coupled phase oscillators, widely used to study
synchronization phenomena in networks of oscillatory units.

References:
    - Kuramoto (1984). Chemical Oscillations, Waves, and Turbulence.
      Springer-Verlag, Berlin.
"""

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class Kuramoto(AbstractDynamics):
    """Kuramoto phase oscillator model.

    Single-state model representing the phase angle of an oscillator, widely used to
    study synchronization phenomena in networks of oscillatory units.

    Notes
    -----
    The model describes synchronization dynamics in networks of coupled oscillators
    through phase interactions.

    **State equation:**

    $$\\frac{d\\theta}{dt} = \\omega + c_{\\text{delayed}} + c_{\\text{instant}}$$

    where:

    - $\\theta$: Phase angle $[0, 2\\pi]$
    - $\\omega$: Natural frequency of oscillation
    - $c_{\\text{instant}}$: Local coupling (additive)
    - $c_{\\text{delayed}}$: Long-range delayed coupling (additive)

    Both coupling channels enter additively. The characteristic
    $\\sin(\\theta_j - \\theta_i)$ phase interaction is supplied by
    `KuramotoCoupling` or `DelayedKuramotoCoupling`, not by the dynamics.

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variable: ``('theta',)``
    INITIAL_STATE : tuple of float
        Default initial condition: ``(0.1,)``
    AUXILIARY_NAMES : tuple of str
        No auxiliary variables: ``()``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 1, 'delayed': 1}``
    DEFAULT_PARAMS : Bunch
        Natural frequency ``omega=1.0`` (rad/s or Hz depending on units)

    References
    ----------
    Kuramoto (1984). Chemical Oscillations, Waves, and Turbulence. Springer-Verlag, Berlin.
    """

    STATE_NAMES = ("theta",)
    INITIAL_STATE = (0.1,)

    DEFAULT_PARAMS = Bunch(
        omega=1.0,  # Natural frequency (rad/s or Hz depending on units)
    )

    COUPLING_INPUTS = {
        "instant": 1,  # Local coupling (phase-dependent)
        "delayed": 1,  # Long-range delayed coupling
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> jnp.ndarray:
        """Compute Kuramoto dynamics.

        Parameters
        ----------
        t : float
            Current time
        state : jnp.ndarray
            Current state with shape ``[1, n_nodes]`` containing theta (phase angle)
        params : Bunch
            Model parameters (omega: natural frequency)
        coupling : Bunch
            Coupling inputs with attributes:

            - ``.instant[0]``: local coupling (used in sin transform)
            - ``.delayed[0]``: long-range coupling (additive)
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            Phase velocity with shape ``[1, n_nodes]``
        """
        # The phase velocity does not depend on the node's own phase: the
        # sin(theta_j - theta_i) difference is formed by the coupling, so
        # `state` is unused here.
        del state

        # Unpack coupling
        c_instant = coupling.instant[0]  # Local coupling
        c_delayed = coupling.delayed[0]  # Long-range coupling

        # Phase update: natural frequency + long-range + local coupling.
        # Both coupling channels enter additively. The sin(theta_j - theta_i)
        # phase interaction belongs to KuramotoCoupling / DelayedKuramotoCoupling,
        # so applying a second sine here would transform it twice.
        dtheta_dt = params.omega + c_delayed + c_instant

        # Package results
        derivatives = jnp.array([dtheta_dt])

        return derivatives
