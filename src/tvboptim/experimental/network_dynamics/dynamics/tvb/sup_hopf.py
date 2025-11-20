"""Supercritical Hopf bifurcation normal form oscillator.

A two-dimensional oscillator in Cartesian coordinates representing the
normal form of a supercritical Hopf bifurcation, widely used for modeling
oscillatory dynamics near bifurcation points.

References:
    - Deco et al. (2017). The dynamics of resting fluctuations in the brain:
      metastability and its dynamical cortical core. Scientific Reports, 7, 3095.
    - Kuznetsov (2004). Elements of Applied Bifurcation Theory (3rd ed.).
      Springer-Verlag, New York.
"""

from typing import Tuple

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class SupHopf(AbstractDynamics):
    """Supercritical Hopf bifurcation oscillator.

    Two-state model in Cartesian coordinates representing the normal form of a
    supercritical Hopf bifurcation, widely used for modeling oscillatory dynamics
    near bifurcation points.

    Notes
    -----
    **State equations:**

    $$
    \\begin{aligned}
    \\frac{dx}{dt} &= (a - x^2 - y^2) x - \\omega y + c_{\\text{delayed},x} + c_{\\text{instant}} \\\\
    \\frac{dy}{dt} &= (a - x^2 - y^2) y + \\omega x + c_{\\text{delayed},y}
    \\end{aligned}
    $$

    where:

    - $a$: Bifurcation parameter ($a < 0$: stable fixed point, $a > 0$: limit cycle)
    - $\\omega$: Angular frequency of oscillation
    - Limit cycle amplitude: $\\sqrt{a}$ for $a > 0$

    The delayed coupling is 2-dimensional, allowing separate coupling for x and y
    components.

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variables: ``('x', 'y')``
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(0.1, 0.0)``
    AUXILIARY_NAMES : tuple of str
        No auxiliary variables: ``()``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 1, 'delayed': 2}``
    DEFAULT_PARAMS : Bunch
        Parameters: ``a=-0.5`` (bifurcation), ``omega=1.0`` (frequency)

    References
    ----------
    Deco et al. (2017). The dynamics of resting fluctuations in the brain:
    metastability and its dynamical cortical core. Scientific Reports, 7, 3095.
    """

    STATE_NAMES = ('x', 'y')
    INITIAL_STATE = (0.1, 0.0)

    DEFAULT_PARAMS = Bunch(
        a=-0.5,            # Bifurcation parameter (a > 0: oscillations)
        omega=1.0,         # Angular frequency (rad/s or Hz)
    )

    COUPLING_INPUTS = {
        'instant': 1,   # Local coupling (x-component only)
        'delayed': 2,   # Long-range coupling [x-component, y-component]
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch
    ) -> jnp.ndarray:
        """Compute supercritical Hopf dynamics.

        Parameters
        ----------
        t : float
            Current time
        state : jnp.ndarray
            Current state with shape ``[2, n_nodes]`` containing ``(x, y)``
        params : Bunch
            Model parameters (a: bifurcation, omega: frequency)
        coupling : Bunch
            Coupling inputs with attributes:

            - ``.instant[0]``: local coupling (x only)
            - ``.delayed[0]``: long-range x-coupling
            - ``.delayed[1]``: long-range y-coupling
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[2, n_nodes]``
        """
        # Unpack state variables
        x = state[0]  # x-component
        y = state[1]  # y-component

        # Unpack coupling
        c_instant = coupling.instant[0]      # Local coupling (x only)
        c_delayed_x = coupling.delayed[0]    # Long-range x-coupling
        c_delayed_y = coupling.delayed[1]    # Long-range y-coupling

        # Amplitude term (Hopf bifurcation nonlinearity)
        r_squared = x**2 + y**2
        amplitude_term = params.a - r_squared

        # Supercritical Hopf dynamics in Cartesian coordinates
        dx_dt = amplitude_term * x - params.omega * y + c_delayed_x + c_instant
        dy_dt = amplitude_term * y + params.omega * x + c_delayed_y

        # Package results
        derivatives = jnp.array([dx_dt, dy_dt])

        return derivatives
