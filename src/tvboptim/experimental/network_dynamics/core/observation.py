"""Graph-order observations for heterogeneous networks."""

from collections.abc import Mapping
from typing import Any

from .bunch import Bunch
from .heterogeneous import _normalize_readout, _require_name


class GroupObservation:
    """Project recorded group outputs into common graph-order channels.

    Each mapping value is a recorded variable name, a tuple of names, or a
    ``readout(recorded, params) -> [Q, n_group_nodes]`` callable. Equal channel
    width only establishes shape compatibility; users remain responsible for
    making group-specific transformations scientifically commensurable.

    Partial coverage is allowed for ordinary projection. With ``reduce=`` it is
    rejected unless ``allow_partial_coverage=True``. That opt-in is only for a
    reducer the user has verified is fill-aware: it does not prevent NaN rows
    in covariance reducers or stop BOLD reducers treating fill as real drive.
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
