"""Reduced Wong-Wang neural mass model with multi-coupling support."""

from typing import Tuple

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class ReducedWongWang(AbstractDynamics):
    """Reduced Wong-Wang neural mass model with multi-coupling support.

    This implementation accepts two coupling inputs (instantaneous and delayed),
    both of which contribute to the total synaptic input.

    Notes
    -----
    The model describes the dynamics of a single synaptic gating variable representing
    NMDA receptor activity.

    **State equation:**

    $$\\frac{dS}{dt} = -\\frac{S}{\\tau_s} + (1-S) H(x) \\gamma$$

    **Transfer function:**

    $$H(x) = \\frac{ax - b}{1 - \\exp(-d(ax - b))}$$

    **Total input:**

    $$x = w J_N S + I_o + J_N (c_{\\text{instant}} + c_{\\text{delayed}})$$

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variable name: ``("S",)`` (synaptic gating variable)
    INITIAL_STATE : tuple of float
        Default initial condition: ``(0.1,)``
    AUXILIARY_NAMES : tuple of str
        Auxiliary variable: ``("H",)`` (transfer function output)
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 1, 'delayed': 1}``
    DEFAULT_PARAMS : Bunch
        Standard Wong-Wang parameters (a, b, d, gamma, tau_s, w, J_N, I_o)
    """

    STATE_NAMES = ("S",)
    INITIAL_STATE = (0.1,)  # Small initial synaptic activity

    AUXILIARY_NAMES = ("H",)
    
    DEFAULT_PARAMS = Bunch(
        a=0.270,         # Input gain parameter [n/C]
        b=0.108,         # Input shift parameter [kHz]
        d=154.0,         # Parameter for H function [ms]
        gamma=0.641,     # Kinetic parameter
        tau_s=100.0,     # NMDA decay time constant [ms]
        w=0.6,           # Excitatory recurrence
        J_N=0.2609,      # Excitatory recurrence
        I_o=0.33,        # Effective external input [nA]
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
        """Compute Wong-Wang dynamics with two coupling inputs.

        Parameters
        ----------
        t : float
            Current time (unused for autonomous system)
        state : jnp.ndarray
            Current state with shape ``[1, n_nodes]`` containing S (synaptic gating variable)
        params : Bunch
            Model parameters: a, b, d, gamma, tau_s, w, J_N, I_o
        coupling : Bunch
            Coupling inputs with attributes ``.instant[1, n_nodes]`` and ``.delayed[1, n_nodes]``
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[1, n_nodes]`` containing dS/dt
        auxiliaries : jnp.ndarray
            Auxiliary variables with shape ``[1, n_nodes]`` containing H (transfer function)
        """
        # Unpack parameters
        a, b, d = params.a, params.b, params.d
        gamma, tau_s = params.gamma, params.tau_s
        w, J_N, I_o = params.w, params.J_N, params.I_o

        # Unpack state and coupling
        S = state[0]  # Synaptic gating variable
        c_instant = coupling.instant[0]
        c_delayed = coupling.delayed[0]

        # Total input to population (both couplings add via J_N)
        x = w * J_N * S + I_o + J_N * c_instant + J_N * c_delayed

        # Transfer function H(x)
        ax_minus_b = a * x - b
        H = ax_minus_b / (1 - jnp.exp(-d * ax_minus_b))

        # Population dynamics
        dS_dt = -(S / tau_s) + (1 - S) * H * gamma

        # Package results
        derivatives = jnp.array([dS_dt])
        auxiliaries = jnp.array([H])

        return derivatives, auxiliaries