"""Linear neural mass model.

A simple linear model with damping, useful for testing and as a baseline
for comparison with nonlinear models.

References:
    - Used as a canonical test model in TVB for validating simulation pipelines
      and network dynamics.
"""

import jax.numpy as jnp

from ...core.bunch import Bunch
from ..base import AbstractDynamics


class Linear(AbstractDynamics):
    """Linear neural mass model with damping.

    Single-state variable model representing simple damped linear dynamics, useful
    for testing, debugging, and understanding basic coupling mechanisms without
    nonlinear complications.

    Notes
    -----
    **State equation:**

    $$\\frac{dx}{dt} = \\gamma x + c_{\\text{delayed}} + c_{\\text{instant}}$$

    where:

    - $x$: State variable
    - $\\gamma$: Damping coefficient (must be negative for stability)
    - $c_{\\text{delayed}}$: Long-range coupling input
    - $c_{\\text{instant}}$: Local coupling input (proportional to x)

    The damping coefficient $\\gamma$ should be negative and its magnitude should
    exceed the node's in-degree to ensure stability. For $\\gamma > 0$, the system
    will exhibit exponential growth.

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variable: ``('x',)``
    INITIAL_STATE : tuple of float
        Default initial condition: ``(0.01,)``
    AUXILIARY_NAMES : tuple of str
        No auxiliary variables: ``()``
    COUPLING_INPUTS : dict
        Coupling specification: ``{'instant': 1, 'delayed': 1}``
    DEFAULT_PARAMS : Bunch
        Damping coefficient ``gamma=-10.0`` (must be negative for stability)
    """

    STATE_NAMES = ("x",)
    INITIAL_STATE = (0.01,)

    DEFAULT_PARAMS = Bunch(
        gamma=-10.0,  # Damping coefficient (must be negative for stability)
    )

    COUPLING_INPUTS = {
        "instant": 1,  # Local coupling (proportional to x)
        "delayed": 1,  # Long-range coupling
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> jnp.ndarray:
        """Compute linear dynamics.

        Parameters
        ----------
        t : float
            Current time
        state : jnp.ndarray
            Current state with shape ``[1, n_nodes]`` containing x
        params : Bunch
            Model parameters (gamma: damping coefficient)
        coupling : Bunch
            Coupling inputs with attributes:

            - ``.instant[0]``: local coupling
            - ``.delayed[0]``: long-range coupling
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivative with shape ``[1, n_nodes]``
        """
        # Unpack state
        x = state[0]

        # Unpack coupling
        c_instant = coupling.instant[0]  # Local coupling
        c_delayed = coupling.delayed[0]  # Long-range coupling

        # Linear dynamics with damping and coupling
        dx_dt = params.gamma * x + c_delayed + c_instant

        # Package results
        derivatives = jnp.array([dx_dt])

        return derivatives
