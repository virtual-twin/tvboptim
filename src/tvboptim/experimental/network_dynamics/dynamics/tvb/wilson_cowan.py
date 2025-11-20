"""Wilson-Cowan neural mass model.

Classic two-population model with excitatory and inhibitory interactions,
based on Wilson & Cowan's seminal work on neural population dynamics.

References:
    - Wilson & Cowan (1972). Excitatory and inhibitory interactions in
      localized populations of model neurons. Biophysical Journal, 12, 1-24.
    - Wilson & Cowan (1973). A Mathematical Theory of the Functional Dynamics
      of Cortical and Thalamic Nervous Tissue.
    - Daffertshofer & van Wijk (2011). On the influence of amplitude on the
      connectivity between phases. Frontiers in Neuroinformatics.
"""

from typing import Tuple

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class WilsonCowan(AbstractDynamics):
    """Wilson-Cowan neural mass model with excitatory and inhibitory populations.

    Two-population model representing the mean firing rates of excitatory (E) and
    inhibitory (I) neural populations with sigmoid activation functions and
    mutual interactions.

    Notes
    -----
    **State equations:**

    $$
    \\begin{aligned}
    \\frac{dE}{dt} &= \\frac{-E + (k_e - r_e E) S_e}{\\tau_e} \\\\
    \\frac{dI}{dt} &= \\frac{-I + (k_i - r_i I) S_i}{\\tau_i}
    \\end{aligned}
    $$

    **Sigmoid activation functions:**

    $$
    \\begin{aligned}
    x_e &= \\alpha_e (c_{ee} E - c_{ei} I + P - \\theta_e + c_{\\text{delayed}} + \\text{lc}_e + \\text{lc}_i) \\\\
    x_i &= \\alpha_i (c_{ie} E - c_{ii} I + Q - \\theta_i + \\text{lc}_e + \\text{lc}_i)
    \\end{aligned}
    $$

    If ``shift_sigmoid=True`` (baseline-corrected):

    $$S_e = c_e \\left(\\frac{1}{1+\\exp(-a_e(x_e-b_e))} - \\frac{1}{1+\\exp(a_e b_e)}\\right)$$

    Otherwise (standard sigmoid):

    $$S_e = \\frac{c_e}{1 + \\exp(-a_e(x_e - b_e))}$$

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variables: ``('E', 'I')``
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(0.1, 0.05)``
    AUXILIARY_NAMES : tuple of str
        Auxiliary variables: ``('S_e', 'S_i')`` (sigmoid outputs)
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 2, 'delayed': 1}``
        Instant coupling has 2 components [E, I] feeding into both populations;
        delayed coupling feeds only into E population
    DEFAULT_PARAMS : Bunch
        Standard Wilson-Cowan parameters

    References
    ----------
    Wilson & Cowan (1972). Excitatory and inhibitory interactions in localized
    populations of model neurons. Biophysical Journal, 12, 1-24.
    """

    STATE_NAMES = ("E", "I")
    INITIAL_STATE = (0.1, 0.05)

    AUXILIARY_NAMES = ("S_e", "S_i")

    DEFAULT_PARAMS = Bunch(
        # Local connectivity weights
        c_ee=12.0,  # Excitatory to excitatory
        c_ei=4.0,  # Inhibitory to excitatory
        c_ie=13.0,  # Excitatory to inhibitory
        c_ii=11.0,  # Inhibitory to inhibitory
        # Time constants (ms)
        tau_e=10.0,  # Excitatory population
        tau_i=10.0,  # Inhibitory population
        # Sigmoid function parameters (excitatory)
        a_e=1.2,  # Gain/steepness
        b_e=2.8,  # Threshold
        c_e=1.0,  # Maximum response
        theta_e=0.0,  # Baseline threshold
        # Sigmoid function parameters (inhibitory)
        a_i=1.0,  # Gain/steepness
        b_i=4.0,  # Threshold
        c_i=1.0,  # Maximum response
        theta_i=0.0,  # Baseline threshold
        # Response modulation parameters
        r_e=1.0,  # Excitatory refractory parameter
        r_i=1.0,  # Inhibitory refractory parameter
        k_e=1.0,  # Excitatory gain parameter
        k_i=1.0,  # Inhibitory gain parameter
        # External inputs
        P=0.0,  # External input to excitatory population
        Q=0.0,  # External input to inhibitory population
        # Input gain parameters
        alpha_e=1.0,  # Excitatory input scaling
        alpha_i=1.0,  # Inhibitory input scaling
        # Sigmoid shift option
        shift_sigmoid=False,  # Whether to use baseline-corrected sigmoid
    )

    COUPLING_INPUTS = {
        "instant": 2,  # Local coupling [from E, from I]
        "delayed": 1,  # Long-range delayed coupling
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Wilson-Cowan dynamics.

        Parameters
        ----------
        t : float
            Current time (ms)
        state : jnp.ndarray
            Current state with shape ``[2, n_nodes]`` containing (E, I)
        params : Bunch
            Model parameters
        coupling : Bunch
            Coupling inputs with attributes:

            - ``.instant[0]``: local E coupling
            - ``.instant[1]``: local I coupling
            - ``.delayed[0]``: long-range coupling
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[2, n_nodes]``
        auxiliaries : jnp.ndarray
            Auxiliary variables with shape ``[2, n_nodes]`` containing ``(S_e, S_i)`` sigmoid outputs
        """
        # Unpack state variables
        E = state[0]  # Excitatory activity
        I = state[1]  # Inhibitory activity

        # Unpack coupling
        lc_e = coupling.instant[0]  # Local coupling from E
        lc_i = coupling.instant[1]  # Local coupling from I
        c_delayed = coupling.delayed[0]  # Long-range coupling

        # Compute inputs to populations
        # Both local couplings (E and I) feed into both populations
        # Long-range coupling only feeds into excitatory
        x_e = params.alpha_e * (
            params.c_ee * E
            - params.c_ei * I
            + params.P
            - params.theta_e
            + c_delayed
            + lc_e
            + lc_i
        )

        x_i = params.alpha_i * (
            params.c_ie * E - params.c_ii * I + params.Q - params.theta_i + lc_e + lc_i
        )

        # Compute sigmoid activation functions
        if params.shift_sigmoid:
            # Baseline-corrected sigmoid (passes through zero)
            sigmoid_offset_e = 1.0 / (1.0 + jnp.exp(params.a_e * params.b_e))
            S_e = params.c_e * (
                1.0 / (1.0 + jnp.exp(-params.a_e * (x_e - params.b_e)))
                - sigmoid_offset_e
            )

            sigmoid_offset_i = 1.0 / (1.0 + jnp.exp(params.a_i * params.b_i))
            S_i = params.c_i * (
                1.0 / (1.0 + jnp.exp(-params.a_i * (x_i - params.b_i)))
                - sigmoid_offset_i
            )
        else:
            # Standard sigmoid
            S_e = params.c_e / (1.0 + jnp.exp(-params.a_e * (x_e - params.b_e)))
            S_i = params.c_i / (1.0 + jnp.exp(-params.a_i * (x_i - params.b_i)))

        # Population dynamics
        dE_dt = (-E + (params.k_e - params.r_e * E) * S_e) / params.tau_e
        dI_dt = (-I + (params.k_i - params.r_i * I) * S_i) / params.tau_i

        # Package results
        derivatives = jnp.array([dE_dt, dI_dt])
        auxiliaries = jnp.array([S_e, S_i])

        return derivatives, auxiliaries
