"""The MontbrioPazoRoxin r-coupling scales with tau, like the recurrent term.

In the reduced MPR equations the recurrent synaptic input enters the voltage
equation as ``J * tau * r``, where the ``tau`` converts a firing rate into the
voltage-like units of ``V**2``, ``eta`` and ``I``. Long-range coupling built
from neighbouring firing rates needs the same conversion.

``test_tvb_comparison`` runs every model at its default parameters, and this
scaling is invisible at the default ``tau = 1``, so it needs its own check.
"""

import jax.numpy as jnp

from tvboptim.experimental.network_dynamics.core import Bunch
from tvboptim.experimental.network_dynamics.dynamics.tvb import MontbrioPazoRoxin

N_NODES = 2
STATE = jnp.array([[0.4, 0.4], [-1.2, -1.2]])  # synchronised (r, V)
ROW_SUM = 0.75  # summed connectivity weight seen by each node
TAU = 2.0
CR = 0.6
J = 14.5


def _dv(model, coupling_r):
    coupling = Bunch(
        instant=jnp.stack([jnp.full((N_NODES,), coupling_r), jnp.zeros(N_NODES)]),
        delayed=jnp.zeros((2, N_NODES)),
    )
    return model.dynamics(0.0, STATE, model.params, coupling, Bunch())[1]


def test_r_coupling_is_equivalent_to_a_shift_in_J():
    """At tau != 1, r-coupling must still act exactly like extra recurrent gain.

    Without the tau factor the effective coupling strength would depend on the
    population time constant, which is not a property the model should have.
    """
    r = STATE[0, 0]
    coupled = MontbrioPazoRoxin(tau=TAU, J=J, cr=CR, cv=0.0)
    absorbed = MontbrioPazoRoxin(tau=TAU, J=J + CR * ROW_SUM, cr=0.0, cv=0.0)

    assert jnp.allclose(_dv(coupled, ROW_SUM * r), _dv(absorbed, 0.0), rtol=1e-6)
