"""Coombes-Byrne 2D infinite theta neuron model.

Mean field model of infinite populations of all-to-all coupled QIF neurons
with synaptic conductance dynamics, derived via the Ott-Antonsen reduction.

References:
    - Coombes & Byrne (2019). Next generation neural mass models. In Nonlinear
      Dynamics in Computational Neuroscience (pp. 1-16). Springer.
"""

from typing import Tuple

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class CoombesByrne2D(AbstractDynamics):
    """Coombes-Byrne 2D infinite theta neuron population model.

    Two-dimensional mean field model with conductance-based synaptic interactions,
    derived via the Ott-Antonsen reduction. Unlike the Montbrio-Pazo-Roxin model,
    this includes implicit synaptic conductance proportional to firing rate.

    Notes
    -----
    **State variables:**

    - $r$: Average firing rate of the population
    - $V$: Average membrane potential of the population

    **Synaptic conductance:**

    $$g = \\kappa \\pi r$$

    where $\\kappa$ is the synaptic conductance scaling factor.

    **State equations:**

    $$
    \\begin{aligned}
    \\frac{dr}{dt} &= \\frac{\\Delta}{\\pi} + 2Vr - gr^2 \\\\
    \\frac{dV}{dt} &= V^2 - (\\pi r)^2 + \\eta + (v_{\\text{syn}} - V)g + c_{\\text{coup}}
    \\end{aligned}
    $$

    where $c_{\\text{coup}}$ is the combined instant and delayed coupling.

    The conductance $g = \\kappa \\pi r$ creates a quadratic nonlinearity in the
    firing rate equation, leading to different dynamical regimes compared to the
    standard Montbrio-Pazo-Roxin model.

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variables: ``('r', 'V')``
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(0.1, 0.0)``
    AUXILIARY_NAMES : tuple of str
        Auxiliary variable: ``('g',)`` (synaptic conductance)
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 1, 'delayed': 1}``
    DEFAULT_PARAMS : Bunch
        Standard Coombes-Byrne parameters

    References
    ----------
    Coombes & Byrne (2019). Next generation neural mass models. In Nonlinear
    Dynamics in Computational Neuroscience (pp. 1-16). Springer.
    """

    STATE_NAMES = ("r", "V")
    INITIAL_STATE = (0.1, 0.0)

    AUXILIARY_NAMES = ("g",)

    DEFAULT_PARAMS = Bunch(
        Delta=1.0,  # Width of heterogeneous noise distribution
        eta=2.0,  # Constant external input
        k=1.0,  # Synaptic conductance scaling (kappa)
        v_syn=0.0,  # Synaptic reversal potential
    )

    COUPLING_INPUTS = {
        "instant": 1,  # Local coupling through r
        "delayed": 1,  # Long-range coupling through r
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Coombes-Byrne 2D dynamics.

        Parameters
        ----------
        t : float
            Current time
        state : jnp.ndarray
            Current state with shape ``[2, n_nodes]`` containing ``(r, V)``
        params : Bunch
            Model parameters
        coupling : Bunch
            Coupling inputs with attributes:

            - ``.instant[0]``: local r-coupling
            - ``.delayed[0]``: long-range r-coupling
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[2, n_nodes]``
        auxiliaries : jnp.ndarray
            Auxiliary variables with shape ``[1, n_nodes]`` containing synaptic conductance g
        """
        # Unpack state variables
        r = state[0]  # Average firing rate
        V = state[1]  # Average membrane potential

        # Unpack coupling
        c_instant = coupling.instant[0]  # Local coupling
        c_delayed = coupling.delayed[0]  # Long-range coupling

        # Total coupling (enters V equation)
        coupling_total = c_instant + c_delayed

        # Synaptic conductance (proportional to firing rate)
        g = params.k * jnp.pi * r

        # Mean field dynamics
        # Firing rate dynamics (with quadratic damping from conductance)
        dr_dt = params.Delta / jnp.pi + 2.0 * V * r - g * r**2

        # Membrane potential dynamics (with conductance-based interaction)
        dV_dt = (
            V**2
            - (jnp.pi * r) ** 2
            + params.eta
            + (params.v_syn - V) * g
            + coupling_total
        )

        # Package results
        derivatives = jnp.array([dr_dt, dV_dt])
        auxiliaries = jnp.array([g])

        return derivatives, auxiliaries
