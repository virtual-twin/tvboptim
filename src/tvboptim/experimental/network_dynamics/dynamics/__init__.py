"""Neural dynamics models."""

from .base import AbstractDynamics
from .lorenz import Lorenz
from .tvb import ReducedWongWang, JansenRit

__all__ = ["AbstractDynamics", "Lorenz", "ReducedWongWang", "JansenRit"]