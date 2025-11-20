"""Jansen-Rit neural mass model with multi-coupling support."""

from typing import Tuple

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class JansenRit(AbstractDynamics):
    """Jansen-Rit neural mass model with multi-coupling support.

    This implementation accepts two coupling inputs (instantaneous and delayed),
    both of which add into the excitatory interneuron population equation.

    Notes
    -----
    The model describes the dynamics of three neural populations: pyramidal cells,
    excitatory interneurons, and inhibitory interneurons.

    **State equations:**

    $$
    \\begin{aligned}
    \\frac{dy_0}{dt} &= y_3 \\\\
    \\frac{dy_3}{dt} &= A a S(y_1 - y_2) - 2a y_3 - a^2 y_0 \\\\
    \\frac{dy_1}{dt} &= y_4 \\\\
    \\frac{dy_4}{dt} &= A a (\\mu + a_2 J S(a_1 J y_0) + c_{\\text{instant}} + c_{\\text{delayed}}) - 2a y_4 - a^2 y_1 \\\\
    \\frac{dy_2}{dt} &= y_5 \\\\
    \\frac{dy_5}{dt} &= B b (a_4 J S(a_3 J y_0)) - 2b y_5 - b^2 y_2
    \\end{aligned}
    $$

    **Sigmoid function:**

    $$S(v) = \\frac{2 \\nu_{\\text{max}}}{1 + \\exp(r(v_0 - v))}$$

    **State variables:**

    - $y_0$: Average membrane potential of pyramidal cells
    - $y_1$: Average membrane potential of excitatory interneurons
    - $y_2$: Average membrane potential of inhibitory interneurons
    - $y_3, y_4, y_5$: Time derivatives of $y_0, y_1, y_2$

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variable names: ``("y0", "y1", "y2", "y3", "y4", "y5")``
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(0.0, 5.0, 5.0, 0.0, 0.0, 0.0)``
    COUPLING_INPUTS : dict
        Coupling input specification: ``{'instant': 1, 'delayed': 1}``
    DEFAULT_PARAMS : Bunch
        Standard Jansen-Rit parameters (A, B, a, b, v0, nu_max, r, J, a_1-a_4, mu)
    """

    STATE_NAMES = ("y0", "y1", "y2", "y3", "y4", "y5")
    INITIAL_STATE = (0.0, 5.0, 5.0, 0.0, 0.0, 0.0)

    AUXILIARY_NAMES = ("sigm_y1_y2", "sigm_y0_1", "sigm_y0_3")

    DEFAULT_PARAMS = Bunch(
        A=3.25,          # Maximum amplitude of EPSP [mV]
        B=22.0,          # Maximum amplitude of IPSP [mV]
        a=0.1,           # Reciprocal of membrane time constant [ms^-1]
        b=0.05,          # Reciprocal of membrane time constant [ms^-1]
        v0=5.52,         # Firing threshold [mV]
        nu_max=0.0025,   # Maximum firing rate [ms^-1]
        r=0.56,          # Steepness of sigmoid [mV^-1]
        J=135.0,         # Average number of synapses
        a_1=1.0,         # Excitatory feedback probability
        a_2=0.8,         # Slow excitatory feedback probability
        a_3=0.25,        # Inhibitory feedback probability
        a_4=0.25,        # Slow inhibitory feedback probability
        mu=0.22,         # Mean input firing rate
    )

    # Multi-coupling: instantaneous and delayed
    COUPLING_INPUTS = {
        'instant': 1,
        'delayed': 1,
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Jansen-Rit dynamics with two coupling inputs.

        Parameters
        ----------
        t : float
            Current time (unused for autonomous system)
        state : jnp.ndarray
            Current state with shape ``[6, n_nodes]`` containing ``(y0, y1, y2, y3, y4, y5)``
        params : Bunch
            Model parameters including: A, B, a, b, v0, nu_max, r, J, a_1, a_2, a_3, a_4, mu
        coupling : Bunch
            Coupling inputs with attributes ``.instant[1, n_nodes]`` and ``.delayed[1, n_nodes]``
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[6, n_nodes]``
        auxiliaries : jnp.ndarray
            Auxiliary variables with shape ``[3, n_nodes]`` containing sigmoid function values
        """
        # Unpack parameters
        A, B = params.A, params.B
        a, b = params.a, params.b
        v0, nu_max, r = params.v0, params.nu_max, params.r
        J = params.J
        a_1, a_2, a_3, a_4 = params.a_1, params.a_2, params.a_3, params.a_4
        mu = params.mu

        # Unpack state variables
        y0, y1, y2, y3, y4, y5 = state[0], state[1], state[2], state[3], state[4], state[5]

        # Unpack coupling inputs
        c_instant = coupling.instant[0]
        c_delayed = coupling.delayed[0]

        # Sigmoid functions
        sigm_y1_y2 = 2.0 * nu_max / (1.0 + jnp.exp(r * (v0 - (y1 - y2))))
        sigm_y0_1 = 2.0 * nu_max / (1.0 + jnp.exp(r * (v0 - (a_1 * J * y0))))
        sigm_y0_3 = 2.0 * nu_max / (1.0 + jnp.exp(r * (v0 - (a_3 * J * y0))))

        # State derivatives (both couplings add to excitatory interneuron)
        dy0_dt = y3
        dy1_dt = y4
        dy2_dt = y5
        dy3_dt = A * a * sigm_y1_y2 - 2.0 * a * y3 - a**2 * y0
        dy4_dt = A * a * (mu + a_2 * J * sigm_y0_1 + c_instant + c_delayed) - 2.0 * a * y4 - a**2 * y1
        dy5_dt = B * b * (a_4 * J * sigm_y0_3) - 2.0 * b * y5 - b**2 * y2

        # Package results
        derivatives = jnp.array([dy0_dt, dy1_dt, dy2_dt, dy3_dt, dy4_dt, dy5_dt])
        auxiliaries = jnp.array([sigm_y1_y2, sigm_y0_1, sigm_y0_3])

        return derivatives, auxiliaries
