"""Abstract base class for noise processes in neural networks."""

from abc import ABC, abstractmethod
from typing import Optional, Union, List

import jax
import jax.numpy as jnp

from ..core.bunch import Bunch
from ..dynamics.base import AbstractDynamics


class AbstractNoise(ABC):
    """Base class for stochastic processes in neural networks.
    
    Handles the diffusion coefficient g(t, state, params) part of SDEs.
    The Brownian motion dW is handled by Diffrax integration.
    """
    
    # Default parameters (to be overridden by subclasses)
    DEFAULT_PARAMS: Bunch = Bunch()
    
    def __init__(self, apply_to: Optional[Union[str, List[str], List[int]]] = None, 
                 key=None, **kwargs):
        """Initialize noise process.
        
        Args:
            apply_to: States to apply noise to. Options:
                - None: Apply to all states (default)
                - str: Single state name, e.g., "x"  
                - List[str]: Multiple state names, e.g., ["x", "v"]
                - List[int]: State indices, e.g., [0, 3]
            key: JAX random key for reproducible noise generation
                 If None, uses jax.random.key(0)
            **kwargs: Parameter overrides for DEFAULT_PARAMS
        """
        self.apply_to = apply_to
        self._state_indices = None  # Computed when combined with dynamics
        self.key = key if key is not None else jax.random.key(0)
        
        # Initialize parameters following dynamics pattern
        self.params = Bunch(self.DEFAULT_PARAMS)  # Copy default parameters
        
        # Update with user-provided parameters
        for key_name, value in kwargs.items():
            if key_name in self.DEFAULT_PARAMS:
                self.params[key_name] = value
            else:
                raise ValueError(f"Unknown parameter '{key_name}' for {self.__class__.__name__}. "
                               f"Valid parameters: {list(self.DEFAULT_PARAMS.keys())}")
    
    def _resolve_state_indices(self, dynamics: AbstractDynamics) -> jnp.ndarray:
        """Resolve which states receive noise based on dynamics.
        
        Args:
            dynamics: Dynamics model to resolve state names/indices against
            
        Returns:
            Array of state indices that should receive noise
            
        Raises:
            ValueError: If apply_to specification is invalid
        """
        if self.apply_to is None:
            # Default: all states get noise
            return jnp.arange(dynamics.N_STATES)
        
        elif isinstance(self.apply_to, str):
            # Single state name
            return dynamics.name_to_index([self.apply_to])
        
        elif isinstance(self.apply_to, list):
            if len(self.apply_to) == 0:
                raise ValueError("apply_to list cannot be empty")
                
            if isinstance(self.apply_to[0], str):
                # List of state names
                return dynamics.name_to_index(self.apply_to)
            else:
                # List of indices - validate they're within bounds
                indices = jnp.array(self.apply_to)
                if jnp.any(indices >= dynamics.N_STATES) or jnp.any(indices < 0):
                    raise ValueError(f"State indices {self.apply_to} out of bounds for dynamics with {dynamics.N_STATES} states")
                return indices
        else:
            raise ValueError(f"Invalid apply_to specification: {self.apply_to}. "
                           f"Expected None, str, List[str], or List[int], got {type(self.apply_to)}")
    
    @abstractmethod
    def diffusion(self, t: float, state: jnp.ndarray, 
                  params: Bunch) -> jnp.ndarray:
        """Compute diffusion coefficient g(t, state, params).
        
        Args:
            t: Current time
            state: Current state, shape [n_states, n_nodes]
            params: Noise parameters (Bunch object)
            
        Returns:
            Diffusion coefficient, shape [n_noise_states, n_nodes]
            where n_noise_states = len(self._state_indices)
        """
        pass
    
    def generate_noise_samples(self, shape: tuple) -> jnp.ndarray:
        """Generate standard Gaussian noise samples.
        
        Args:
            shape: Shape of noise samples to generate
                   Typically (n_steps, n_noise_states, n_nodes)
            
        Returns:
            Raw Gaussian noise samples ~ N(0,1) with the requested shape
        """
        return jax.random.normal(self.key, shape)
    
    def verify(self, dynamics: AbstractDynamics, verbose: bool = True) -> bool:
        """Verify noise configuration is valid for given dynamics.
        
        Args:
            dynamics: Dynamics model to verify against
            verbose: Whether to print verification results
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Test state index resolution
            indices = self._resolve_state_indices(dynamics)
            
            if verbose:
                print(f"{self.__class__.__name__} verification:")
                print(f"  - Dynamics: {dynamics.__class__.__name__} ({dynamics.N_STATES} states)")
                print(f"  - State names: {dynamics.STATE_NAMES}")
                print(f"  - apply_to: {self.apply_to}")
                print(f"  - Resolved indices: {indices}")
                print(f"  - Parameters: {self.params}")
                print(f"  -  Configuration valid")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"{self.__class__.__name__} verification:")
                print(f"  -  Configuration invalid: {e}")
            return False
    
    def __repr__(self):
        """String representation of noise."""
        apply_str = f", apply_to={self.apply_to}" if self.apply_to is not None else ""
        return f"{self.__class__.__name__}(params={self.params}{apply_str})"