"""Preparation contracts and graph-order heterogeneous observations."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
from plum import dispatch

from .bunch import Bunch
from .heterogeneous import _normalize_readout, _require_name


@dataclass(frozen=True)
class SimulationGrid:
    """Static description of the native solver's post-step output grid."""

    t0: float
    dt: float
    n_steps: int


@dataclass(frozen=True)
class PreparedObservation:
    """Private execution recipe for one monitor on one simulation grid."""

    data: Any
    state0: Any
    params: Any
    update: Any
    period: float
    first_sample_offset: float
    variable_names: tuple[str, ...]
    initialize: Any = None


@dispatch
def prepare_observation(
    monitor: object,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    variable_names: tuple,
) -> PreparedObservation:
    """Prepare a supported temporal monitor for block-wise execution.

    Concrete overloads are registered by monitor modules. This fallback keeps
    the network solver independent of those concrete monitor classes while
    producing a useful preparation-time error for unsupported values.
    """
    del grid, sample, variable_names
    raise TypeError(
        f"Unsupported temporal observation type {type(monitor).__name__}. "
        "Use SubSampling, TemporalAverage, BalloonWindkesselBold, or HRFBold."
    )


class GroupObservation:
    """Project group variables of interest into common graph-order channels.

    Each mapping value is a variable-of-interest name, a tuple of names, or a
    ``readout(voi, params) -> [Q, n_group_nodes]`` callable. Equal channel
    width only establishes shape compatibility; users remain responsible for
    making group-specific transformations scientifically commensurable.

    Partial coverage is allowed for ordinary projection. Before a temporal
    monitor it is rejected unless ``allow_partial_coverage=True``. The opt-in
    makes ``fill_value`` a real monitor input; it does not make BOLD or later
    statistics fill-aware. The deprecated ``reduce=`` path applies the same
    coverage rule during its compatibility period.
    """

    def __init__(
        self,
        readouts: Mapping[str, Any],
        *,
        params: Mapping[str, Any] | None = None,
        channels,
        fill_value=0.0,
        allow_partial_coverage: bool = False,
    ):
        if not isinstance(readouts, Mapping):
            raise TypeError("GroupObservation readouts must be a mapping")
        if not readouts:
            raise ValueError("GroupObservation requires at least one group")
        if params is not None and not isinstance(params, Mapping):
            raise TypeError("GroupObservation.params must be a mapping or None")
        if not isinstance(channels, (tuple, list)):
            raise TypeError("GroupObservation.channels must be a tuple or list")
        channels = tuple(channels)
        if not channels:
            raise ValueError("GroupObservation.channels must not be empty")
        if any(not isinstance(name, str) or not name for name in channels):
            raise ValueError("GroupObservation channel names must be non-empty strings")
        if len(set(channels)) != len(channels):
            raise ValueError("GroupObservation channel names must be unique")
        if not isinstance(allow_partial_coverage, bool):
            raise TypeError("allow_partial_coverage must be bool")

        self.readouts = {
            _require_name(name, "observation group"): _normalize_readout(
                value, "observation"
            )
            for name, value in readouts.items()
        }
        unknown_params = set(params or {}) - set(self.readouts)
        if unknown_params:
            raise ValueError(
                "GroupObservation.params must refer to observed groups; "
                f"unknown {sorted(unknown_params)}"
            )
        self.params = {
            name: (params or {}).get(name, Bunch()) for name in self.readouts
        }
        self.channels = channels
        self.fill_value = fill_value
        self.allow_partial_coverage = allow_partial_coverage
