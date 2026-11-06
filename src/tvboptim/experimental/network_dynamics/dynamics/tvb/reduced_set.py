"""Reduced set of FitzHugh-Nagumo oscillators (Stefanescu & Jirsa 2008)."""

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from ...core.bunch import Bunch
from ..base import AbstractDynamics

_N_MODES = 3
_MODE_STATES = ("xi", "eta", "alpha", "beta")


def _mode_indicators(n_samples: int) -> np.ndarray:
    """Partition ``n_samples`` sample points into equally sized modes."""
    per_mode = n_samples // _N_MODES
    indicators = np.zeros((_N_MODES, n_samples))
    for i in range(_N_MODES):
        indicators[i, i * per_mode : (i + 1) * per_mode] = 1.0
    return indicators


def _scalar(model: str, name: str, value) -> jnp.ndarray:
    value = jnp.asarray(value)
    if value.size != 1:
        raise ValueError(
            f"{model}: '{name}' must be a scalar (the population distribution "
            f"is shared by all nodes); got shape {value.shape}"
        )
    return value.reshape(())


class ReducedSetFitzHughNagumo(AbstractDynamics):
    """Mode decomposition of a heterogeneous population of FitzHugh-Nagumo neurons.

    Each node models a population of excitatory and inhibitory FitzHugh-Nagumo
    oscillators whose input currents are normally distributed with mean ``mu``
    and standard deviation ``sigma``. The population is reduced to three modes
    per subpopulation [Stefanescu & Jirsa 2008]. Port of TVB's
    ``ReducedSetFitzHughNagumo``.

    Notes
    -----
    **State equations** for mode $i$ (sums over modes $k$):

    $$
    \\begin{aligned}
    \\dot{\\xi}_i &= \\tau\\left(\\xi_i - e_i\\frac{\\xi_i^3}{3} - \\eta_i\\right)
        + K_{11}\\left(\\sum_k A_{ki}\\xi_k - \\xi_i\\right)
        - K_{12}\\left(\\sum_k B_{ki}\\alpha_k - \\xi_i\\right)
        + \\tau\\left(I^E_i + c\\right) \\\\
    \\dot{\\eta}_i &= \\left(\\xi_i - b\\eta_i + m_i\\right) / \\tau \\\\
    \\dot{\\alpha}_i &= \\tau\\left(\\alpha_i - f_i\\frac{\\alpha_i^3}{3} - \\beta_i\\right)
        + K_{21}\\left(\\sum_k C_{ki}\\xi_k - \\alpha_i\\right)
        + \\tau\\left(I^I_i + c\\right) \\\\
    \\dot{\\beta}_i &= \\left(\\alpha_i - b\\beta_i + n_i\\right) / \\tau
    \\end{aligned}
    $$

    where $c$ is the network input to the summed $\\xi$ modes.

    **Derived parameters.** The mode coefficients $A, B, C, e, f, I^E, I^I,
    m, n$ are integrals of the mode functions against the input distribution
    $\\mathcal{N}(\\mu, \\sigma)$. They are computed from ``mu``, ``sigma`` and
    ``a`` once per solve by ``derived_params``, so ``mu`` and ``sigma`` can be
    swept or fitted by gradient descent. Inspect them with
    ``model.resolve_params(model.params)``.

    **Layout.** TVB's ``(4, nodes, modes)`` state is flattened to 12 states
    ordered ``xi1..3, eta1..3, alpha1..3, beta1..3``. Per-mode derived values
    have shape ``(3, 1)``; ``Aik``, ``Bik``, ``Cik`` keep TVB's ``(3, 3)`` layout.

    **Coupling.** As in TVB, the network input is the coupled sum of the three
    $\\xi$ modes and drives every mode. ``SumLinearCoupling`` sums the modes
    before transport:

    ```python
    coupling = {"instant": SumLinearCoupling(source=("xi1", "xi2", "xi3"))}
    ```

    ``mu`` and ``sigma`` must be scalars. ``a``, ``b``, ``tau`` and the
    ``K`` gains may be per node.

    Parameters
    ----------
    nv, nu : int
        Number of sample points of the excitatory and inhibitory input
        distributions (default 1500). They set array shapes, so they are
        constructor arguments rather than parameters, and must be divisible
        by 3.
    **kwargs
        Parameter overrides, see ``DEFAULT_PARAMS``.
    """

    STATE_NAMES = tuple(
        f"{name}{i}" for name in _MODE_STATES for i in range(1, _N_MODES + 1)
    )
    INITIAL_STATE = (0.0,) * (len(_MODE_STATES) * _N_MODES)

    DEFAULT_PARAMS = Bunch(
        tau=3.0,  # Time-scale separation
        a=0.45,
        b=0.9,
        K11=0.5,  # Internal coupling, excitatory to excitatory
        K12=0.15,  # Internal coupling, inhibitory to excitatory
        K21=0.15,  # Internal coupling, excitatory to inhibitory
        sigma=0.35,  # Standard deviation of the input distribution
        mu=0.0,  # Mean of the input distribution
    )

    DERIVED_PARAMS = ("Aik", "Bik", "Cik", "e_i", "f_i", "IE_i", "II_i", "m_i", "n_i")

    COUPLING_INPUTS = {
        "instant": 1,  # Coupled sum of the xi modes
        "delayed": 1,
    }

    def __init__(self, nv: int = 1500, nu: int = 1500, **kwargs):
        for name, n in (("nv", nv), ("nu", nu)):
            if not isinstance(n, int) or n <= 0 or n % _N_MODES:
                raise ValueError(
                    f"{self.__class__.__name__}: {name} must be a positive int "
                    f"divisible by {_N_MODES}, got {n!r}"
                )
        super().__init__(**kwargs)
        self.nv, self.nu = nv, nu
        # Quantile levels and mode partitions depend only on the sample counts.
        self._pv = np.arange(1, nv + 1) / (nv + 1)
        self._pu = np.arange(1, nu + 1) / (nu + 1)
        self._V = _mode_indicators(nv)
        self._U = _mode_indicators(nu)

    def derived_params(self, params: Bunch) -> Bunch:
        name = self.__class__.__name__
        mu = _scalar(name, "mu", params.mu)
        sigma = _scalar(name, "sigma", params.sigma)
        trapz = jnp.trapezoid

        Zv = norm.ppf(self._pv, loc=mu, scale=sigma)
        Zu = norm.ppf(self._pu, loc=mu, scale=sigma)
        V = self._V / jnp.sqrt(trapz(self._V**2, Zv, axis=1))[:, None]
        U = self._U / jnp.sqrt(trapz(self._U**2, Zu, axis=1))[:, None]
        g1 = norm.pdf(Zv, loc=mu, scale=sigma)
        g2 = norm.pdf(Zu, loc=mu, scale=sigma)

        int_V = trapz(V, Zv, axis=1)[:, None]  # (modes, 1)
        int_U = trapz(U, Zu, axis=1)[:, None]
        int_g1V = trapz(g1 * V, Zv, axis=1)[None, :]  # (1, modes)
        int_g2U = trapz(g2 * U, Zu, axis=1)[None, :]

        return Bunch(
            Aik=(int_V @ int_g1V).T,
            Bik=int_V @ int_g2U,
            Cik=(int_U @ int_g1V).T,
            e_i=trapz(V**4, Zv, axis=1)[:, None],
            f_i=trapz(U**4, Zu, axis=1)[:, None],
            IE_i=trapz(Zv * V, Zv, axis=1)[:, None],
            II_i=trapz(Zu * U, Zu, axis=1)[:, None],
            m_i=params.a * int_V,
            n_i=params.a * int_U,
        )

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> jnp.ndarray:
        """Compute the reduced FitzHugh-Nagumo dynamics.

        Parameters
        ----------
        t : float
            Current time (unused for autonomous system)
        state : jnp.ndarray
            Current state with shape ``[12, n_nodes]``
        params : Bunch
            Primary and derived parameters (see ``resolve_params``)
        coupling : Bunch
            Coupling inputs ``.instant`` and ``.delayed``, each ``[1, n_nodes]``
            (coupled sum of the xi modes)
        external : Bunch
            External inputs (unused)

        Returns
        -------
        jnp.ndarray
            State derivatives with shape ``[12, n_nodes]``
        """
        xi, eta, alpha, beta = jnp.split(state, len(_MODE_STATES))
        c = coupling.instant[0] + coupling.delayed[0]
        tau = params.tau

        dxi = (
            tau * (xi - params.e_i * xi**3 / 3.0 - eta)
            + params.K11 * (params.Aik.T @ xi - xi)
            - params.K12 * (params.Bik.T @ alpha - xi)
            + tau * (params.IE_i + c)
        )
        deta = (xi - params.b * eta + params.m_i) / tau
        dalpha = (
            tau * (alpha - params.f_i * alpha**3 / 3.0 - beta)
            + params.K21 * (params.Cik.T @ xi - alpha)
            + tau * (params.II_i + c)
        )
        dbeta = (alpha - params.b * beta + params.n_i) / tau
        return jnp.concatenate((dxi, deta, dalpha, dbeta))
