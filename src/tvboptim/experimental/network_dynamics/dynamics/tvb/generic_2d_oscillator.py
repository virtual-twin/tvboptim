"""Generic 2D oscillator model.

A generic two-state-variable dynamic system with configurable nullclines,
capable of generating a wide range of dynamical behaviors including
excitability, bistability, and oscillations.

References:
    - FitzHugh (1961). Impulses and physiological states in theoretical models
      of nerve membrane. Biophysical Journal, 1, 445.
    - Nagumo et al. (1962). An Active Pulse Transmission Line Simulating Nerve
      Axon. Proceedings of the IRE, 50, 2061.
    - Stefanescu & Jirsa (2008). A low dimensional description of globally
      coupled heterogeneous neural networks of excitatory and inhibitory neurons.
      PLoS Computational Biology, 4(11).
    - Jirsa & Stefanescu (2010). Neural population modes capture biologically
      realistic large-scale network dynamics. Bulletin of Mathematical Biology.
"""

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class Generic2dOscillator(AbstractDynamics):
    """Generic 2D oscillator with configurable nullclines.

    A flexible two-variable dynamical system where V typically represents a fast
    variable (e.g., membrane potential) and W a slow recovery variable. The model
    can exhibit FitzHugh-Nagumo dynamics, bistability, or other behaviors depending
    on parameters.

    Notes
    -----
    **State equations:**

    $$
    \\begin{aligned}
    \\frac{dV}{dt} &= d \\tau (-f V^3 + e V^2 + g V + \\alpha W + \\gamma I + \\gamma c_{\\text{delayed}} + c_{\\text{instant}}) \\\\
    \\frac{dW}{dt} &= \\frac{d}{\\tau} (a + b V + c V^2 - \\beta W)
    \\end{aligned}
    $$

    **Parameter regimes:**

    - **Excitable (FitzHugh-Nagumo-like)**: $a=-2.0, b=-10.0, c=0.0, d=0.02, I=0.0$
    - **Bistable**: $a=1.0, b=0.0, c=-5.0, d=0.02, I=0.0$
    - **Morris-Lecar-like**: $a=0.5, b=0.6, c=-4.0, d=0.02, I=0.0$

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variables: ``('V', 'W')`` (fast and slow variables)
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(0.0, 0.0)``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 1, 'delayed': 1}``
    DEFAULT_PARAMS : Bunch
        Configurable nullcline parameters

    References
    ----------
    FitzHugh (1961). Impulses and physiological states in theoretical models
    of nerve membrane. Biophysical Journal, 1, 445.
    """

    STATE_NAMES = ("V", "W")
    INITIAL_STATE = (0.0, 0.0)

    DEFAULT_PARAMS = Bunch(
        # Nullcline parameters
        a=-2.0,  # Linear coefficient in W equation
        b=-10.0,  # Linear V coefficient in W equation
        c=0.0,  # Quadratic V coefficient in W equation
        d=0.02,  # Global time scaling
        e=3.0,  # Quadratic coefficient in V equation
        f=1.0,  # Cubic coefficient in V equation
        g=0.0,  # Linear coefficient in V equation
        # Coupling parameters
        alpha=1.0,  # W to V coupling strength
        beta=1.0,  # W decay rate
        gamma=1.0,  # External input strength
        # Other parameters
        tau=1.0,  # Time scale separation (tau > 1: V faster than W)
        I=0.0,  # External input current
    )

    COUPLING_INPUTS = {
        "instant": 1,  # Local/instantaneous coupling
        "delayed": 1,  # Long-range delayed coupling
    }

    EXTERNAL_INPUTS = {"stimulus": 1}

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> jnp.ndarray:
        """Compute Generic2dOscillator dynamics.

        Args:
            t: Current time
            state: State [2, n_nodes] with (V, W)
            params: Model parameters
            coupling: Coupling inputs (.instant, .delayed)

        Returns:
            derivatives: [2, n_nodes] state derivatives
        """
        # Unpack state variables
        V = state[0]  # Fast variable (e.g., voltage)
        W = state[1]  # Slow variable (e.g., recovery)

        # Unpack coupling
        c_instant = coupling.instant[0]  # Local coupling
        c_delayed = coupling.delayed[0]  # Long-range coupling

        # Unpack external stimulus
        stim = external.stimulus[0]  # Extract [1, n_nodes] -> [n_nodes]

        # V dynamics (cubic nullcline with external input)
        dV_dt = (
            params.d
            * params.tau
            * (
                -params.f * V**3
                + params.e * V**2
                + params.g * V
                + params.alpha * W
                + params.gamma * params.I
                + params.gamma * c_delayed
                + params.gamma * c_instant
            )
            + stim
        )

        # W dynamics (polynomial nullcline)
        dW_dt = (params.d / params.tau) * (
            params.a + params.b * V + params.c * V**2 - params.beta * W
        )

        # Package results
        derivatives = jnp.array([dV_dt, dW_dt])

        return derivatives
