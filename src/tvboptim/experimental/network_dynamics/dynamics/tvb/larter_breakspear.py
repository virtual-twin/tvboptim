"""Larter-Breakspear neural mass model.

A modified Morris-Lecar model that includes a third equation simulating
the effect of inhibitory interneurons synapsing on pyramidal cells.

References:
    - Larter et al. (1999). A coupled ordinary differential equation lattice
      model for the simulation of epileptic seizures. Chaos, 9(3), 795.
    - Breakspear et al. (2003). Modulation of excitatory synaptic coupling
      facilitates synchronization and complex dynamics in a biophysical model
      of neuronal dynamics. Network: Computation in Neural Systems, 14, 703-732.
    - Honey et al. (2007). Network structure of cerebral cortex shapes
      functional connectivity on multiple time scales. PNAS, 104, 10240.
    - Alstott et al. (2009). Modeling the impact of lesions in the human brain.
      PLoS Comput Biol, 5, e1000408.
"""

from typing import Tuple

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class LarterBreakspear(AbstractDynamics):
    """Larter-Breakspear neural mass model.

    A modified Morris-Lecar model with three state variables representing pyramidal
    cells, potassium dynamics, and inhibitory interneurons. The model exhibits rich
    dynamical behaviors including fixed points, limit cycles, and chaos.

    Notes
    -----
    **State variables:**

    - $V$: Membrane potential of pyramidal cells
    - $W$: Potassium channel gating variable
    - $Z$: Inhibitory interneuron activity

    **Dynamical regimes:**

    - $d_V < 0.55$: Fixed point dynamics
    - $0.55 < d_V < 0.59$: Limit cycle attractors
    - $d_V > 0.59$: Chaotic attractors

    **Auxiliary variables:**

    $$
    \\begin{aligned}
    Q_V &= \\frac{1}{2} Q_{V,\\text{max}} \\left(1 + \\tanh\\left(\\frac{V - V_T}{d_V}\\right)\\right) \\quad \\text{(pyramidal firing rate)} \\\\
    Q_Z &= \\frac{1}{2} Q_{Z,\\text{max}} \\left(1 + \\tanh\\left(\\frac{Z - Z_T}{d_Z}\\right)\\right) \\quad \\text{(inhibitory firing rate)} \\\\
    m_{\\text{Ca}} &= \\frac{1}{2} \\left(1 + \\tanh\\left(\\frac{V - T_{\\text{Ca}}}{d_{\\text{Ca}}}\\right)\\right) \\quad \\text{(Ca channel gating)}
    \\end{aligned}
    $$

    with similar expressions for $m_{\\text{Na}}$ and $m_K$.

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variables: ``('V', 'W', 'Z')``
    INITIAL_STATE : tuple of float
        Default initial conditions: ``(0.0, 0.0, 0.0)``
    AUXILIARY_NAMES : tuple of str
        Auxiliary variables: ``('QV', 'QZ', 'm_Ca', 'm_Na', 'm_K')``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 1, 'delayed': 1}``
    DEFAULT_PARAMS : Bunch
        Conductances, thresholds, and synaptic weights

    References
    ----------
    Breakspear et al. (2003). Modulation of excitatory synaptic coupling facilitates
    synchronization and complex dynamics in a biophysical model of neuronal dynamics.
    Network: Computation in Neural Systems, 14, 703-732.
    """

    STATE_NAMES = ('V', 'W', 'Z')
    INITIAL_STATE = (0.0, 0.0, 0.0)

    AUXILIARY_NAMES = ('QV', 'QZ', 'm_Ca', 'm_Na', 'm_K')

    DEFAULT_PARAMS = Bunch(
        # Ion channel conductances
        gCa=1.1,      # Ca++ channel conductance
        gK=2.0,       # K+ channel conductance
        gL=0.5,       # Leak channel conductance
        gNa=6.7,      # Na+ channel conductance

        # Channel activation thresholds
        TCa=-0.01,    # Ca channel threshold
        TK=0.0,       # K channel threshold
        TNa=0.3,      # Na channel threshold

        # Channel activation slope parameters
        d_Ca=0.15,    # Ca channel threshold variance
        d_K=0.3,      # K channel threshold variance
        d_Na=0.15,    # Na channel threshold variance

        # Nernst potentials
        VCa=1.0,      # Ca Nernst potential
        VK=-0.7,      # K Nernst potential
        VL=-0.5,      # Leak Nernst potential
        VNa=0.53,     # Na Nernst potential

        # Kinetic parameters
        phi=0.7,      # Temperature scaling factor
        tau_K=1.0,    # K relaxation time constant (ms)

        # Synaptic coupling strengths
        aee=0.4,      # Excitatory-to-excitatory
        aei=2.0,      # Excitatory-to-inhibitory
        aie=2.0,      # Inhibitory-to-excitatory
        ane=1.0,      # External-to-excitatory
        ani=0.4,      # External-to-inhibitory

        # Other parameters
        b=0.1,        # Inhibitory feedback strength
        C=0.1,        # Long-range coupling weight (vs local)
        Iext=0.3,     # External input current
        rNMDA=0.25,   # NMDA receptor strength

        # Firing rate function parameters
        VT=0.0,       # Pyramidal cell firing threshold
        d_V=0.65,     # Pyramidal cell threshold variance
        ZT=0.0,       # Inhibitory cell firing threshold
        d_Z=0.7,      # Inhibitory cell threshold variance
        QV_max=1.0,   # Maximum pyramidal firing rate
        QZ_max=1.0,   # Maximum inhibitory firing rate

        # Time scaling
        t_scale=1.0,  # Global time scale factor
    )

    COUPLING_INPUTS = {
        'instant': 1,   # Local/instantaneous coupling
        'delayed': 1,   # Long-range delayed coupling
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Larter-Breakspear dynamics.

        Parameters
        ----------
        t : float
            Current time (ms)
        state : jnp.ndarray
            Current state with shape ``[3, n_nodes]`` containing ``(V, W, Z)``
        params : Bunch
            Model parameters
        coupling : Bunch
            Coupling inputs with attributes ``.instant`` and ``.delayed``
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[3, n_nodes]``
        auxiliaries : jnp.ndarray
            Auxiliary variables with shape ``[5, n_nodes]`` containing ``(QV, QZ, m_Ca, m_Na, m_K)``
        """
        # Unpack state variables
        V = state[0]  # Membrane potential
        W = state[1]  # K channel gating
        Z = state[2]  # Inhibitory activity

        # Unpack coupling
        c_instant = coupling.instant[0]  # Local coupling
        c_delayed = coupling.delayed[0]  # Long-range coupling

        # Channel activation functions (sigmoid gating)
        m_Ca = 0.5 * (1.0 + jnp.tanh((V - params.TCa) / params.d_Ca))
        m_Na = 0.5 * (1.0 + jnp.tanh((V - params.TNa) / params.d_Na))
        m_K = 0.5 * (1.0 + jnp.tanh((V - params.TK) / params.d_K))

        # Firing rate functions
        QV = 0.5 * params.QV_max * (1.0 + jnp.tanh((V - params.VT) / params.d_V))
        QZ = 0.5 * params.QZ_max * (1.0 + jnp.tanh((Z - params.ZT) / params.d_Z))

        # Voltage dynamics
        # Ion channel currents
        I_Ca = (params.gCa +
                (1.0 - params.C) * params.rNMDA * params.aee * (QV + c_instant) +
                params.C * params.rNMDA * params.aee * c_delayed) * m_Ca * (V - params.VCa)

        I_K = params.gK * W * (V - params.VK)

        I_L = params.gL * (V - params.VL)

        I_Na = (params.gNa * m_Na +
                (1.0 - params.C) * params.aee * (QV + c_instant) +
                params.C * params.aee * c_delayed) * (V - params.VNa)

        I_inh = params.aie * Z * QZ

        I_ext = params.ane * params.Iext

        dV_dt = params.t_scale * (-I_Ca - I_K - I_L - I_Na - I_inh + I_ext)

        # Potassium channel dynamics
        dW_dt = params.t_scale * params.phi * (m_K - W) / params.tau_K

        # Inhibitory population dynamics
        dZ_dt = params.t_scale * params.b * (params.ani * params.Iext + params.aei * V * QV)

        # Package results
        derivatives = jnp.array([dV_dt, dW_dt, dZ_dt])
        auxiliaries = jnp.array([QV, QZ, m_Ca, m_Na, m_K])

        return derivatives, auxiliaries
