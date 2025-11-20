"""Wong-Wang model with separate excitatory and inhibitory populations.

Reduced mean-field model with explicit excitatory and inhibitory populations,
mutually coupled through local connections and long-range network coupling.

References:
    - Wong & Wang (2006). A Recurrent Network Mechanism of Time Integration in
      Perceptual Decisions. Journal of Neuroscience, 26(4), 1314-1328.
    - Deco et al. (2014). How Local Excitation-Inhibition Ratio Impacts the
      Whole Brain Dynamics. Journal of Neuroscience, 34(23), 7886-7898.
"""

from typing import Tuple

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class WongWangExcInh(AbstractDynamics):
    """Wong-Wang neural mass model with excitatory and inhibitory populations.

    Two-population model representing local excitatory-inhibitory dynamics
    with NMDA-mediated recurrence and long-range coupling.

    Notes
    -----
    **State equations:**

    $$
    \\begin{aligned}
    x_e &= w_p J_N S_e - J_i S_i + W_e I_o + c_{\\text{total}} + I_{\\text{ext}} \\\\
    H_e &= \\frac{a_e x_e - b_e}{1 - \\exp(-d_e(a_e x_e - b_e))} \\\\
    \\frac{dS_e}{dt} &= -\\frac{S_e}{\\tau_e} + (1 - S_e) H_e \\gamma_e
    \\end{aligned}
    $$

    $$
    \\begin{aligned}
    x_i &= J_N S_e - S_i + W_i I_o + \\lambda c_{\\text{total}} \\\\
    H_i &= \\frac{a_i x_i - b_i}{1 - \\exp(-d_i(a_i x_i - b_i))} \\\\
    \\frac{dS_i}{dt} &= -\\frac{S_i}{\\tau_i} + H_i \\gamma_i
    \\end{aligned}
    $$

    where: $c_{\\text{total}} = G J_N (c_{\\text{delayed}} + c_{\\text{instant}})$

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variables: ``('S_e', 'S_i')`` (excitatory and inhibitory synaptic gating)
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(0.001, 0.001)``
    AUXILIARY_NAMES : tuple of str
        Auxiliary variables: ``('H_e', 'H_i')`` (transfer functions)
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 1, 'delayed': 1}``
    DEFAULT_PARAMS : Bunch
        Standard Wong-Wang parameters for excitatory/inhibitory populations

    References
    ----------
    Wong & Wang (2006). A Recurrent Network Mechanism of Time Integration in
    Perceptual Decisions. Journal of Neuroscience, 26(4), 1314-1328.
    """

    STATE_NAMES = ("S_e", "S_i")
    INITIAL_STATE = (0.001, 0.001)

    AUXILIARY_NAMES = ("H_e", "H_i")

    DEFAULT_PARAMS = Bunch(
        # Excitatory population parameters
        a_e=310.0,  # [n/C] Input gain parameter
        b_e=125.0,  # [Hz] Input shift parameter
        d_e=0.160,  # [s] Input scaling parameter
        gamma_e=0.641 / 1000,  # Kinetic parameter
        tau_e=100.0,  # [ms] NMDA decay time constant
        w_p=1.4,  # Recurrence weight
        W_e=1.0,  # External input scaling weight
        # Inhibitory population parameters
        a_i=615.0,  # [n/C] Input gain parameter
        b_i=177.0,  # [Hz] Input shift parameter
        d_i=0.087,  # [s] Input scaling parameter
        gamma_i=1.0 / 1000,  # Kinetic parameter
        tau_i=10.0,  # [ms] NMDA decay time constant
        W_i=0.7,  # External input scaling weight
        # Synaptic weights
        J_N=0.15,  # [nA] NMDA current
        J_i=1.0,  # Inhibitory synaptic weight
        # External inputs
        I_o=0.382,  # Background input current
        I_ext=0.0,  # External stimulation current
        # Coupling parameters
        G=2.0,  # Global coupling strength
        lamda=0.0,  # Lambda: inhibitory coupling weight
    )

    COUPLING_INPUTS = {
        "instant": 1,  # Local/instantaneous coupling
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
        """Compute Wong-Wang excitatory-inhibitory dynamics.

        Parameters
        ----------
        t : float
            Current time (ms)
        state : jnp.ndarray
            Current state with shape ``[2, n_nodes]`` containing ``(S_e, S_i)``
        params : Bunch
            Model parameters
        coupling : Bunch
            Coupling inputs with attributes ``.instant`` and ``.delayed``
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[2, n_nodes]``
        auxiliaries : jnp.ndarray
            Auxiliary variables with shape ``[2, n_nodes]`` containing ``(H_e, H_i)`` transfer functions
        """
        # Unpack state variables
        S_e = state[0]  # Excitatory synaptic gating
        S_i = state[1]  # Inhibitory synaptic gating

        # Unpack coupling
        c_instant = coupling.instant[0]  # Local coupling (proportional to S_e)
        c_delayed = coupling.delayed[0]  # Long-range coupling

        # Combined coupling (both instant and delayed)
        coupling_total = params.G * params.J_N * (c_delayed + c_instant)

        # Excitatory population input
        J_N_S_e = params.J_N * S_e
        x_e = (
            params.w_p * J_N_S_e
            - params.J_i * S_i
            + params.W_e * params.I_o
            + coupling_total
            + params.I_ext
        )

        # Excitatory transfer function
        x_e_scaled = params.a_e * x_e - params.b_e
        H_e = x_e_scaled / (1.0 - jnp.exp(-params.d_e * x_e_scaled))

        # Excitatory dynamics
        dS_e_dt = -(S_e / params.tau_e) + (1.0 - S_e) * H_e * params.gamma_e

        # Inhibitory population input
        x_i = J_N_S_e - S_i + params.W_i * params.I_o + params.lamda * coupling_total

        # Inhibitory transfer function
        x_i_scaled = params.a_i * x_i - params.b_i
        H_i = x_i_scaled / (1.0 - jnp.exp(-params.d_i * x_i_scaled))

        # Inhibitory dynamics
        dS_i_dt = -(S_i / params.tau_i) + H_i * params.gamma_i

        # Package results
        derivatives = jnp.array([dS_e_dt, dS_i_dt])
        auxiliaries = jnp.array([H_e, H_i])

        return derivatives, auxiliaries
