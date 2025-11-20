"""Montbrio-Pazo-Roxin infinite theta neuron model.

Mean field model of infinite populations of all-to-all coupled quadratic
integrate-and-fire (QIF) neurons, derived via the Ott-Antonsen reduction.

References:
    - Montbrio, Pazo & Roxin (2015). Macroscopic description for networks of
      spiking neurons. Physical Review X, 5(2), 021028.
"""

from typing import Tuple

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class MontbrioPazoRoxin(AbstractDynamics):
    """Montbrio-Pazo-Roxin infinite theta neuron population model.

    Two-dimensional mean field model describing the Ott-Antonsen reduction of
    infinite all-to-all coupled quadratic integrate-and-fire (QIF) neurons.

    Notes
    -----
    **State variables:**

    - $r$: Average firing rate of the population
    - $V$: Average membrane potential of the population

    **State equations:**

    $$
    \\begin{aligned}
    \\frac{dr}{dt} &= \\frac{1}{\\tau} \\left(\\frac{\\Delta}{\\pi \\tau} + 2Vr\\right) \\\\
    \\frac{dV}{dt} &= \\frac{1}{\\tau} \\left(V^2 - (\\pi \\tau r)^2 + \\eta + J\\tau r + I + c_r c_{\\text{coup},r} + c_v c_{\\text{coup},V}\\right)
    \\end{aligned}
    $$

    where $c_{\\text{coup},r}$ and $c_{\\text{coup},V}$ are the combined instant and
    delayed coupling components, and $c_r$, $c_v$ are coupling weights.

    The model has 2-dimensional coupling allowing independent coupling through firing
    rate (r) and membrane potential (V).

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variables: ``('r', 'V')``
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(0.1, 0.0)``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 2, 'delayed': 2}``
    DEFAULT_PARAMS : Bunch
        Standard parameters for QIF neuron population

    References
    ----------
    Montbrio, Pazo & Roxin (2015). Macroscopic description for networks of
    spiking neurons. Physical Review X, 5(2), 021028.
    """

    STATE_NAMES = ('r', 'V')
    INITIAL_STATE = (0.1, 0.0)

    DEFAULT_PARAMS = Bunch(
        tau=1.0,           # Characteristic time scale
        I=0.0,             # External current
        Delta=1.0,         # Width of heterogeneous noise distribution
        J=15.0,            # Mean synaptic weight
        eta=-5.0,          # Constant external input scaling
        cr=1.0,            # Coupling weight through r (firing rate)
        cv=0.0,            # Coupling weight through V (membrane potential)
    )

    COUPLING_INPUTS = {
        'instant': 2,   # Local coupling [r-component, V-component]
        'delayed': 2,   # Long-range coupling [r-component, V-component]
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch
    ) -> jnp.ndarray:
        """Compute Montbrio-Pazo-Roxin dynamics.

        Args:
            t: Current time
            state: State [2, n_nodes] with (r, V)
            params: Model parameters
            coupling: Coupling inputs
                - .instant[0]: local r-coupling
                - .instant[1]: local V-coupling
                - .delayed[0]: long-range r-coupling
                - .delayed[1]: long-range V-coupling

        Returns:
            derivatives: [2, n_nodes] state derivatives
        """
        # Unpack state variables
        r = state[0]  # Average firing rate
        V = state[1]  # Average membrane potential

        # Unpack coupling (both have r and V components)
        c_instant_r = coupling.instant[0]    # Local r-coupling
        c_instant_V = coupling.instant[1]    # Local V-coupling
        c_delayed_r = coupling.delayed[0]    # Long-range r-coupling
        c_delayed_V = coupling.delayed[1]    # Long-range V-coupling

        # Total coupling for each variable
        coupling_r = params.cr * (c_instant_r + c_delayed_r)
        coupling_V = params.cv * (c_instant_V + c_delayed_V)

        # Mean field dynamics
        # Firing rate dynamics
        dr_dt = (1.0 / params.tau) * (
            params.Delta / (jnp.pi * params.tau) + 2.0 * V * r
        )

        # Membrane potential dynamics
        dV_dt = (1.0 / params.tau) * (
            V**2 -
            (jnp.pi * params.tau * r)**2 +
            params.eta +
            params.J * params.tau * r +
            params.I +
            coupling_r +
            coupling_V
        )

        # Package results
        derivatives = jnp.array([dr_dt, dV_dt])

        return derivatives
