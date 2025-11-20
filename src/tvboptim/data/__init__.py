"""Data loading utilities for tvboptim.

This module provides convenient access to built-in connectivity and
functional connectivity datasets.
"""

from .loaders import (
    load_structural_connectivity,
    load_functional_connectivity,
)

__all__ = [
    "load_structural_connectivity",
    "load_functional_connectivity",
]
