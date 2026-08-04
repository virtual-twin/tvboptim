"""Network Dynamics: JAX-based brain network modeling interface.

Part of TVB-Optim experimental modules.
"""

from .core import (
    Bunch,
    DynamicsGroup,
    GroupObservation,
    HeterogeneousNetwork,
    Network,
    Readout,
    SignalRoute,
)
from .graph import (
    DelayGraph,
    DenseDelayGraph,
    DenseGraph,
    DenseLengthGraph,
    Graph,
    SparseDelayGraph,
    SparseGraph,
)
from .result import HeterogeneousSolution
from .solve import prepare, solve

__all__ = [
    "Bunch",
    "DynamicsGroup",
    "Graph",
    "GroupObservation",
    "HeterogeneousSolution",
    "DelayGraph",
    "SparseGraph",
    "SparseDelayGraph",
    "DenseGraph",
    "DenseDelayGraph",
    "DenseLengthGraph",
    "solve",
    "prepare",
    "Network",
    "Readout",
    "HeterogeneousNetwork",
    "SignalRoute",
]
