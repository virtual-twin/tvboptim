"""Lorenz chaotic dynamical system with multi-coupling support."""

import jax.numpy as jnp

from ..core.bunch import Bunch
from .base import AbstractDynamics


class Lorenz(AbstractDynamics):
    """Lorenz chaotic dynamical system with multi-coupling support.

    The classic three-dimensional chaotic system with support for structural coupling.

    Notes
    -----
    **State equations:**

    $$
    \\begin{aligned}
    \\frac{dx}{dt} &= \\sigma(y - x) + c_{\\text{structural}} \\\\
    \\frac{dy}{dt} &= x(\\rho - z) - y \\\\
    \\frac{dz}{dt} &= xy - \\beta z
    \\end{aligned}
    $$

    where $c_{\\text{structural}}$ is the structural coupling input.

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variable names: ``("x", "y", "z")``
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(1.0, 1.0, 1.0)``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'structural': 1}`` (single structural coupling to x)
    DEFAULT_PARAMS : Bunch
        Model parameters: ``sigma=10.0`` (Prandtl number), ``rho=28.0`` (Rayleigh number),
        ``beta=8/3`` (geometric parameter)

    Examples
    --------
    >>> dynamics = Lorenz()
    >>> dynamics = Lorenz(sigma=15.0, rho=30.0)
    """

    STATE_NAMES = ("x", "y", "z")
    INITIAL_STATE = (1.0, 1.0, 1.0)

    AUXILIARY_NAMES = ()

    DEFAULT_PARAMS = Bunch(
        sigma=10.0,  # Prandtl number
        rho=28.0,  # Rayleigh number
        beta=8.0 / 3.0,  # Geometric parameter
    )

    # NEW: Declare expected coupling inputs
    COUPLING_INPUTS = {
        "structural": 1,  # Structural connectivity coupling
    }

    def dynamics(
        self, t: float, state: jnp.ndarray, params: Bunch, coupling: Bunch, external: Bunch
    ) -> jnp.ndarray:
        """Compute Lorenz system derivatives with coupling.

        Parameters
        ----------
        t : float
            Current time (unused for autonomous system)
        state : jnp.ndarray
            Current state with shape ``[3, n_nodes]`` containing ``(x, y, z)``
        params : Bunch
            Model parameters: sigma, rho, beta
        coupling : Bunch
            Coupling inputs with attribute ``.structural[1, n_nodes]``
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[3, n_nodes]`` containing ``(dx/dt, dy/dt, dz/dt)``
        """
        x, y, z = state[0], state[1], state[2]

        # Access structural coupling via Bunch attribute
        structural = coupling.structural[0]

        # Lorenz equations with coupling
        dxdt = params.sigma * (y - x) + structural
        dydt = x * (params.rho - z) - y
        dzdt = x * y - params.beta * z

        return jnp.array([dxdt, dydt, dzdt])


class FlexibleLorenz(AbstractDynamics):
    """Lorenz system with multiple coupling types for demonstration.

    Extends the Lorenz system to support multiple coupling mechanisms:
    structural (diffusive) and modulatory (gain modulation).

    Notes
    -----
    **State equations:**

    $$
    \\begin{aligned}
    \\frac{dx}{dt} &= \\sigma_{\\text{eff}}(y - x) + c_{\\text{structural}} \\\\
    \\frac{dy}{dt} &= x(\\rho - z) - y \\\\
    \\frac{dz}{dt} &= xy - \\beta z
    \\end{aligned}
    $$

    where the effective sigma parameter is modulated by coupling:

    $$\\sigma_{\\text{eff}} = \\sigma (1 + \\alpha \\cdot c_{\\text{modulatory}})$$

    with $\\alpha$ being the modulation strength parameter.

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variable names: ``("x", "y", "z")``
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(1.0, 1.0, 1.0)``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'structural': 1, 'modulatory': 1}``
    DEFAULT_PARAMS : Bunch
        Model parameters including ``modulation_strength=0.3`` to control
        how much modulatory coupling affects sigma

    Examples
    --------
    >>> # Use both couplings
    >>> dynamics = FlexibleLorenz()
    >>> # If only structural is provided, modulatory gets zeros automatically
    """

    STATE_NAMES = ("x", "y", "z")
    INITIAL_STATE = (1.0, 1.0, 1.0)

    AUXILIARY_NAMES = ()

    DEFAULT_PARAMS = Bunch(
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        modulation_strength=0.3,  # How much modulatory coupling affects sigma
    )

    # Declare multiple coupling inputs
    COUPLING_INPUTS = {
        "structural": 1,  # Structural connectivity
        "modulatory": 1,  # Gain modulation
    }

    def dynamics(
        self, t: float, state: jnp.ndarray, params: Bunch, coupling: Bunch, external: Bunch
    ) -> jnp.ndarray:
        """Compute Lorenz derivatives with multi-coupling.

        Parameters
        ----------
        t : float
            Current time (unused for autonomous system)
        state : jnp.ndarray
            Current state with shape ``[3, n_nodes]`` containing ``(x, y, z)``
        params : Bunch
            Model parameters: sigma, rho, beta, modulation_strength
        coupling : Bunch
            Coupling inputs with attributes ``.structural`` and ``.modulatory``
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[3, n_nodes]`` containing ``(dx/dt, dy/dt, dz/dt)``
        """
        x, y, z = state[0], state[1], state[2]

        # Access both couplings via Bunch attributes
        structural = coupling.structural[0]
        modulatory = coupling.modulatory[0]

        # Model-specific combination logic:
        # Modulatory coupling affects the effective sigma parameter
        effective_sigma = params.sigma * (1.0 + params.modulation_strength * modulatory)

        # Lorenz equations with modulated sigma and structural coupling
        dxdt = effective_sigma * (y - x) + structural
        dydt = x * (params.rho - z) - y
        dzdt = x * y - params.beta * z

        return jnp.array([dxdt, dydt, dzdt])