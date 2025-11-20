"""TVBOptim - Optimization tools for The Virtual Brain."""

__version__ = "0.1.0"

from .types import spaces, parameter
from .model import jaxify
from . import data

__all__ = ["spaces", "jaxify", "parameter", "data"]