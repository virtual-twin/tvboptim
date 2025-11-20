"""Epileptor model for seizure dynamics.

The Hindmarsh-Rose-Jirsa Epileptor is a composite neural mass model with six
state variables, crafted to capture the phenomenology of epileptic seizures
including transitions between interictal and ictal states.

References:
    - Jirsa et al. (2014). On the nature of seizure dynamics. Brain, 137(8),
      2210-2230.
    - Proix et al. (2014). Permittivity coupling across brain regions determines
      seizure recruitment in partial epilepsy. Journal of Neuroscience, 34(45),
      15009-15021.
"""

from typing import Tuple

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class Epileptor(AbstractDynamics):
    """Epileptor model for epileptic seizure dynamics.

    Six-dimensional model with two coupled populations operating on different time
    scales, designed to reproduce transitions between interictal (between seizures)
    and ictal (during seizure) states.

    Notes
    -----
    The model consists of:

    - **Population 1** (x1, y1, z): Fast time scale, represents rapid ictal activity
    - **Population 2** (x2, y2): Ultra-slow time scale, controls interictal background
    - **Filter variable** g: Low-pass filtered version of x1

    **State equations:**

    $$
    \\begin{aligned}
    \\frac{dx_1}{dt} &= t_t (y_1 - z + I_{\\text{ext}} + K_{vf} c_{\\text{pop1}} + f_1(x_1,x_2) x_1) \\\\
    \\frac{dy_1}{dt} &= t_t (c - d x_1^2 - y_1) \\\\
    \\frac{dz}{dt} &= t_t r (h(x_1,z) - z + K_s c_{\\text{pop1}}) \\\\
    \\frac{dx_2}{dt} &= t_t (-y_2 + x_2 - x_2^3 + I_{\\text{ext2}} + bb \\cdot g - 0.3(z-3.5) + K_f c_{\\text{pop2}}) \\\\
    \\frac{dy_2}{dt} &= t_t \\frac{-y_2 + f_2(x_2)}{\\tau} \\\\
    \\frac{dg}{dt} &= t_t (-0.01(g - 0.1 x_1))
    \\end{aligned}
    $$

    where $f_1$, $f_2$, and $h$ are piecewise nonlinear functions modeling different
    dynamical regimes.

    **Key parameters:**

    - $x_0$: Epileptogenicity parameter (controls seizure threshold)
    - $K_{vf}, K_f, K_s$: Coupling strengths at different time scales
    - ``modification``: If True, uses nonlinear (sigmoidal) permittivity

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variables: ``('x1', 'y1', 'z', 'x2', 'y2', 'g')``
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(-1.5, -10.0, 3.5, -1.0, 0.0, 0.0)``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 2, 'delayed': 2}``
        Components [0] couple to population 1 (fast), [1] to population 2 (slow)
    DEFAULT_PARAMS : Bunch
        Standard Epileptor parameters from Jirsa et al. 2014

    References
    ----------
    Jirsa et al. (2014). On the nature of seizure dynamics. Brain, 137(8), 2210-2230.
    """

    STATE_NAMES = ('x1', 'y1', 'z', 'x2', 'y2', 'g')
    INITIAL_STATE = (-1.5, -10.0, 3.5, -1.0, 0.0, 0.0)

    DEFAULT_PARAMS = Bunch(
        # Population 1 parameters
        a=1.0,             # Cubic term coefficient in x1
        b=3.0,             # Squared term coefficient in x1
        c=1.0,             # Additive coefficient in y1
        d=5.0,             # Squared term coefficient in y1
        r=0.00035,         # Temporal scaling in z (1/tau_0)
        s=4.0,             # Linear coefficient in z
        x0=-1.6,           # Epileptogenicity parameter
        Iext=3.1,          # External input to population 1
        slope=0.0,         # Linear coefficient in x1

        # Population 2 parameters
        Iext2=0.45,        # External input to population 2
        tau=10.0,          # Temporal scaling in y2
        aa=6.0,            # Linear coefficient in y2
        bb=2.0,            # Coupling coefficient from g to x2

        # Coupling parameters
        Kvf=0.0,           # Very fast time scale coupling (to x1)
        Kf=0.0,            # Fast time scale coupling (to x2)
        Ks=0.0,            # Slow time scale coupling (to z, permittivity)

        # Global parameters
        tt=1.0,            # Global time scaling
        modification=False,  # Use nonlinear permittivity influence on z
    )

    COUPLING_INPUTS = {
        'instant': 2,   # Local coupling [pop1, pop2]
        'delayed': 2,   # Long-range coupling [pop1, pop2]
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch
    ) -> jnp.ndarray:
        """Compute Epileptor dynamics.

        Parameters
        ----------
        t : float
            Current time
        state : jnp.ndarray
            Current state with shape ``[6, n_nodes]`` containing ``(x1, y1, z, x2, y2, g)``
        params : Bunch
            Model parameters
        coupling : Bunch
            Coupling inputs with attributes:

            - ``.instant[0], .delayed[0]``: population 1 coupling
            - ``.instant[1], .delayed[1]``: population 2 coupling
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[6, n_nodes]``
        """
        # Unpack state variables
        x1 = state[0]  # Fast population membrane potential
        y1 = state[1]  # Fast population recovery
        z = state[2]   # Slow permittivity variable
        x2 = state[3]  # Slow population membrane potential
        y2 = state[4]  # Slow population recovery
        g = state[5]   # Low-pass filter of x1

        # Unpack coupling
        c_pop1 = coupling.instant[0] + coupling.delayed[0]  # Population 1 coupling
        c_pop2 = coupling.instant[1] + coupling.delayed[1]  # Population 2 coupling

        # Population 1 dynamics (fast time scale)
        # Piecewise function f1(x1, x2)
        f1_if_neg = -params.a * x1**2 + params.b * x1
        f1_if_pos = params.slope - x2 + 0.6 * (z - 4.0)**2
        f1 = jnp.where(x1 < 0.0, f1_if_neg, f1_if_pos)

        dx1_dt = params.tt * (
            y1 - z + params.Iext + params.Kvf * c_pop1 + f1 * x1
        )

        dy1_dt = params.tt * (params.c - params.d * x1**2 - y1)

        # Energy/permittivity variable (slow time scale)
        # Piecewise nonlinearity in z dynamics
        z_nonlin_if_neg = -0.1 * z**7
        z_nonlin_if_pos = 0.0
        z_nonlin = jnp.where(z < 0.0, z_nonlin_if_neg, z_nonlin_if_pos)

        if params.modification:
            # Nonlinear (sigmoidal) permittivity influence
            h = params.x0 + 3.0 / (1.0 + jnp.exp(-(x1 + 0.5) / 0.1))
        else:
            # Linear permittivity influence
            h = 4.0 * (x1 - params.x0) + z_nonlin

        dz_dt = params.tt * params.r * (h - z + params.Ks * c_pop1)

        # Population 2 dynamics (ultra-slow time scale)
        dx2_dt = params.tt * (
            -y2 + x2 - x2**3 + params.Iext2 + params.bb * g -
            0.3 * (z - 3.5) + params.Kf * c_pop2
        )

        # Piecewise function f2(x2)
        f2_if_neg = 0.0
        f2_if_pos = params.aa * (x2 + 0.25)
        f2 = jnp.where(x2 < -0.25, f2_if_neg, f2_if_pos)

        dy2_dt = params.tt * ((-y2 + f2) / params.tau)

        # Low-pass filter
        dg_dt = params.tt * (-0.01 * (g - 0.1 * x1))

        # Package results
        derivatives = jnp.array([dx1_dt, dy1_dt, dz_dt, dx2_dt, dy2_dt, dg_dt])

        return derivatives
