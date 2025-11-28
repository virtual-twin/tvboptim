"""Data loading functions for built-in datasets."""

from pathlib import Path
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

# Get the data directory - use importlib.resources for better compatibility
try:
    from importlib.resources import files
    _DATA_DIR = Path(str(files('tvboptim.data')))
except (TypeError, ModuleNotFoundError):
    # Fallback for development without package installation
    _DATA_DIR = Path(__file__).parent


def load_structural_connectivity(
    name: str = "dk_average",
) -> Tuple[jnp.ndarray, jnp.ndarray, List[str]]:
    """
    Load structural connectivity data.

    Returns raw connectivity weights, tract lengths, and region labels.
    Users can normalize or preprocess as needed.

    Available datasets:
        - "dk_average": Average structural connectivity from Desikan-Killiany
                        parcellation (84 regions)
        - "dTOR": Virtual DBS structural connectivity with Glasser cortical
                  parcellation plus distal and striatum regions (370 regions)

    Args:
        name: Dataset name (default: "dk_average")

    Returns:
        weights: Connectivity weights [n_nodes, n_nodes]
        lengths: Tract lengths in mm [n_nodes, n_nodes]
        region_labels: List of region acronyms (e.g., ['L.BSTS', 'R.MTG', ...])

    Examples:
        >>> from tvboptim.data import load_structural_connectivity
        >>> weights, lengths, region_labels = load_structural_connectivity()
        >>> print(weights.shape)
        (84, 84)
        >>> print(len(region_labels))
        84
        >>> print(region_labels[:3])
        ['L.BSTS', 'L.CACG', 'L.CMFG']

        >>> # Load dTOR dataset
        >>> weights, lengths, region_labels = load_structural_connectivity("dTOR")
        >>> print(weights.shape)
        (370, 370)

        >>> # Normalize if needed
        >>> weights_norm = weights / jnp.max(weights)
    """
    data_path = _DATA_DIR / "connectivity" / name / "data.npz"

    if not data_path.exists():
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: dk_average, dTOR"
        )

    data = np.load(data_path, allow_pickle=True)
    weights = jnp.array(data["weights"])
    lengths = jnp.array(data["lengths"])
    region_labels = data["region_labels"].tolist()

    return weights, lengths, region_labels


def load_functional_connectivity(name: str = "dk_average") -> jnp.ndarray:
    """
    Load functional connectivity data.

    Returns empirical functional connectivity matrix as JAX array.

    Available datasets:
        - "dk_average": Average functional connectivity from Desikan-Killiany
                        parcellation (84 regions)

    Args:
        name: Dataset name (default: "dk_average")

    Returns:
        fc: Functional connectivity matrix [n_nodes, n_nodes]

    Examples:
        >>> from tvboptim.data import load_functional_connectivity
        >>> fc_target = load_functional_connectivity()
        >>> print(fc_target.shape)
        (84, 84)
    """
    data_path = _DATA_DIR / "functional" / name / "data.npz"

    if not data_path.exists():
        raise ValueError(f"Dataset '{name}' not found. Available datasets: dk_average")

    data = np.load(data_path)
    fc = jnp.array(data["fc"])

    return fc
