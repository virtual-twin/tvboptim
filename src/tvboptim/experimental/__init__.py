"""Experimental modules for TVB-Optim.

This package contains experimental features and integrations that are under
active development. APIs may change without notice.

Modules:
    network_dynamics: JAX-based brain network modeling framework with support
                      for ODEs, DDEs, SDEs, and SDDEs with flexible coupling
                      and graph structures.
"""

# Import network_dynamics as a submodule
from . import network_dynamics

__all__ = ["network_dynamics"]
