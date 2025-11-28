"""TVBOptim - Optimization tools for The Virtual Brain."""

try:
    from importlib.metadata import version
    __version__ = version("tvboptim")
except Exception:
    __version__ = "unknown"

from . import data
from .tvbo.prepare_experiment import HAS_TVBO, prepare
from .types import parameter, spaces

__all__ = ["spaces", "prepare", "HAS_TVBO", "parameter", "data"]
