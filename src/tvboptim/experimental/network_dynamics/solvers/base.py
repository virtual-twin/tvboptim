"""Base solver classes for  Network Dynamics.

This module defines the abstract base classes for all solver types.
"""

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp

from ..core.bunch import Bunch


class AbstractSolver(ABC):
    """Base class for all solver types."""
    pass


class NativeSolver(AbstractSolver):
    """Base class for  Network Dynamics's native solvers (manual implementations)."""
    
    @abstractmethod
    def step(self, dynamics_fn: Callable, t: float, state: jnp.ndarray, 
            coupling_input: jnp.ndarray, dt: float, params: Bunch) -> jnp.ndarray:
        """Single integration step.
        
        Args:
            dynamics_fn: Network dynamics function
            t: Current time
            state: Current state, shape [n_states, n_nodes]
            coupling_input: Coupling input, shape [n_coupling_inputs, n_nodes]  
            dt: Timestep
            params: Network parameters
            
        Returns:
            next_state: Updated state, shape [n_states, n_nodes]
        """
        pass