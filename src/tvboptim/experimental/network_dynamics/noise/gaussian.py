"""Gaussian noise processes for neural networks."""

import jax.numpy as jnp

from ..core.bunch import Bunch
from .base import AbstractNoise


class AdditiveNoise(AbstractNoise):
    """Additive Gaussian noise: sigma * dW_t.

    Simple additive white noise with constant variance.
    The diffusion coefficient is constant across time and states.

    Parameters:
        sigma: Standard deviation of the noise (default: 0.1)
               Can be scalar (same for all states) or array (per-state)

    Note:
        When converting from TVB's HeunStochastic integrator:
        TVB uses 'nsig' parameter which is NOT the standard deviation.
        The conversion is: nsig_tvb = 0.5 * sigma^2
        where sigma is the standard deviation used here.
    """

    DEFAULT_PARAMS = Bunch(sigma=0.1)

    def diffusion(self, t: float, state: jnp.ndarray, params: Bunch) -> jnp.ndarray:
        """Compute constant diffusion coefficient.

        Args:
            t: Current time (unused for additive noise)
            state: Current state, shape [n_states, n_nodes]
            params: Noise parameters with 'sigma'

        Returns:
            Raw diffusion coefficient(s) - broadcasting handled by network
        """
        return params.sigma


class MultiplicativeNoise(AbstractNoise):
    """State-dependent multiplicative noise: sigma * (1 + alpha * |state|) * dW_t.

    The noise intensity depends on the current state values.
    Commonly used to model state-dependent fluctuations in neural systems.

    Parameters:
        sigma: Base noise level (default: 0.1)
        state_scaling: State dependence strength alpha (default: 0.1)
    """

    DEFAULT_PARAMS = Bunch(sigma=0.1, state_scaling=0.1)

    def diffusion(self, t: float, state: jnp.ndarray, params: Bunch) -> jnp.ndarray:
        """Compute state-dependent diffusion coefficient.

        Args:
            t: Current time (unused)
            state: Current state, shape [n_states, n_nodes]
            params: Noise parameters with 'sigma' and 'state_scaling'

        Returns:
            State-dependent diffusion coefficients, shape [n_noise_states, n_nodes]
        """
        # Extract states that receive noise
        relevant_states = state[self._state_indices]  # [n_noise_states, n_nodes]

        # Multiplicative scaling: sigma * (1 + alpha * |state|)
        scaling = 1.0 + params.state_scaling * jnp.abs(relevant_states)
        return params.sigma * scaling
