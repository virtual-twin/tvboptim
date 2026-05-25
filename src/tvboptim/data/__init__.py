"""Data loading utilities for tvboptim.

This module provides convenient access to built-in connectivity and
functional connectivity datasets.
"""

from .loaders import (
    load_fcd_distribution,
    load_functional_connectivity,
    load_structural_connectivity,
)

__all__ = [
    "load_structural_connectivity",
    "load_functional_connectivity",
    "load_fcd_distribution",
]
