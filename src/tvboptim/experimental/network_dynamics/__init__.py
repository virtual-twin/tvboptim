"""Network Dynamics: JAX-based brain network modeling interface.

Part of TVBOptim experimental modules.
"""

from .core import Bunch, Network
from .graph import Graph, DelayGraph, SparseGraph, SparseDelayGraph, DenseGraph, DenseDelayGraph
from .solve import solve, prepare

__all__ = ["Bunch", "Graph", "DelayGraph", "SparseGraph", "SparseDelayGraph", "DenseGraph", "DenseDelayGraph", "solve", "prepare", "Network"]