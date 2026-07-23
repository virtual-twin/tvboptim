"""Network Dynamics: JAX-based brain network modeling interface.

Part of TVB-Optim experimental modules.
"""

from .core import Bunch, DynamicsGroup, HeterogeneousNetwork, Network, SignalRoute
from .graph import (
    DelayGraph,
    DenseDelayGraph,
    DenseGraph,
    DenseLengthGraph,
    Graph,
    SparseDelayGraph,
    SparseGraph,
)
from .result import GroupedSolution
from .solve import prepare, solve

__all__ = [
    "Bunch",
    "DynamicsGroup",
    "Graph",
    "GroupedSolution",
    "DelayGraph",
    "SparseGraph",
    "SparseDelayGraph",
    "DenseGraph",
    "DenseDelayGraph",
    "DenseLengthGraph",
    "solve",
    "prepare",
    "Network",
    "HeterogeneousNetwork",
    "SignalRoute",
]
