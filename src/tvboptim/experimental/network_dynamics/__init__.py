"""Network Dynamics: JAX-based brain network modeling interface.

Part of TVB-Optim experimental modules.
"""

from .core import Bunch, Network
from .graph import (
    DelayGraph,
    DenseDelayGraph,
    DenseGraph,
    DenseLengthGraph,
    Graph,
    SparseDelayGraph,
    SparseGraph,
)
from .solve import prepare, solve

__all__ = [
    "Bunch",
    "Graph",
    "DelayGraph",
    "SparseGraph",
    "SparseDelayGraph",
    "DenseGraph",
    "DenseDelayGraph",
    "DenseLengthGraph",
    "solve",
    "prepare",
    "Network",
]
