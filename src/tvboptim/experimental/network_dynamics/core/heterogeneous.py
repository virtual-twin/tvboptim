"""Static construction contracts for shared-graph heterogeneous networks."""

import warnings
from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..coupling.base import PrePostCoupling
from ..dynamics.base import AbstractDynamics
from ..graph.base import AbstractGraph, effective_max_delay
from .bunch import Bunch


def _require_name(name: Any, role: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{role} names must be non-empty strings, got {name!r}")
    return name


def _normalize_readout(value: Any, role: str):
    if isinstance(value, str):
        return (_require_name(value, f"{role} state"),)
    if isinstance(value, tuple):
        if not value:
            raise ValueError(f"{role} state tuple must not be empty")
        return tuple(_require_name(name, f"{role} state") for name in value)
    if callable(value):
        return value
    raise TypeError(
        f"{role} must be a state name, tuple of state names, or callable; "
        f"got {type(value).__name__}"
    )


class DynamicsGroup:
    """One dynamics implementation assigned to a static graph-node subset."""

    def __init__(
        self,
        dynamics: AbstractDynamics,
        nodes,
        *,
        noise=None,
        external_input: Mapping[str, Any] | None = None,
        initial_state=None,
    ):
        if not isinstance(dynamics, AbstractDynamics):
            raise TypeError("dynamics must be an AbstractDynamics instance")
        self.dynamics = dynamics
        self.nodes = nodes
        self.noise = noise
        self.externals = dict(external_input or {})
        self.initial_state = initial_state

        unknown = set(self.externals) - set(dynamics.EXTERNAL_INPUTS)
        if unknown:
            raise ValueError(
                f"Unknown external inputs {sorted(unknown)} for "
                f"{dynamics.__class__.__name__}; expected "
                f"{list(dynamics.EXTERNAL_INPUTS)}"
            )

    def __repr__(self):
        return (
            f"DynamicsGroup(dynamics={self.dynamics.__class__.__name__}, "
            f"nodes={self.nodes!r})"
        )


class SignalRoute:
    """A canonical source schema, route-compatible coupling, and destinations."""

    def __init__(
        self,
        *,
        source: Mapping[str, Any],
        coupling: PrePostCoupling,
        target: Mapping[str, Any],
        local: Mapping[str, Any] | None = None,
        source_params: Mapping[str, Any] | None = None,
        target_params: Mapping[str, Any] | None = None,
    ):
        if not isinstance(coupling, PrePostCoupling):
            raise TypeError(
                "SignalRoute currently supports PrePostCoupling instances only"
            )
        if coupling.INCOMING_STATE_NAMES or coupling.LOCAL_STATE_NAMES:
            raise ValueError(
                "SignalRoute owns source/local readouts; construct its coupling "
                "without incoming_states or local_states"
            )
        if not source:
            raise ValueError("SignalRoute.source must contain at least one group")
        if not target:
            raise ValueError("SignalRoute.target must contain at least one group")

        self.source = {
            _require_name(name, "source group"): _normalize_readout(value, "source")
            for name, value in source.items()
        }
        self.local = {
            _require_name(name, "local group"): _normalize_readout(value, "local")
            for name, value in (local or {}).items()
        }
        self.coupling = coupling
        self.target = {
            _require_name(name, "target group"): self._normalize_target(value)
            for name, value in target.items()
        }
        self.source_params = dict(source_params or {})
        self.target_params = dict(target_params or {})

        if coupling.PRE_USES_LOCAL:
            missing = set(self.target) - set(self.local)
            if missing:
                raise ValueError(
                    f"{coupling.__class__.__name__} requires local readouts for "
                    f"target groups {sorted(missing)}"
                )
        elif self.local:
            raise ValueError(
                f"{coupling.__class__.__name__} does not use local signals"
            )

        unknown_source_params = set(self.source_params) - set(self.source)
        unknown_target_params = set(self.target_params) - set(self.target)
        if unknown_source_params or unknown_target_params:
            raise ValueError(
                "Route parameter mappings must refer to matching source/target "
                f"groups; unknown source={sorted(unknown_source_params)}, "
                f"target={sorted(unknown_target_params)}"
            )

    @staticmethod
    def _normalize_target(value):
        if isinstance(value, str):
            return (_require_name(value, "target input"), None)
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], Callable)
        ):
            return (_require_name(value[0], "target input"), value[1])
        raise TypeError(
            "target must be an input name or (input_name, conversion_callable)"
        )

    def __repr__(self):
        return (
            f"SignalRoute(sources={list(self.source)}, "
            f"targets={list(self.target)}, "
            f"coupling={self.coupling.__class__.__name__})"
        )


class HeterogeneousNetwork:
    """Different dynamics groups on one shared square graph."""

    def __init__(
        self,
        *,
        graph: AbstractGraph,
        groups: Mapping[str, DynamicsGroup],
        routes: Mapping[str, SignalRoute] | None = None,
        history=None,
    ):
        if not isinstance(graph, AbstractGraph):
            raise TypeError("graph must be an AbstractGraph instance")
        if tuple(graph.weights.shape)[0] != tuple(graph.weights.shape)[1]:
            raise ValueError("HeterogeneousNetwork requires one square shared graph")
        if not groups:
            raise ValueError("HeterogeneousNetwork requires at least one group")

        self.graph = graph
        self.groups = {
            _require_name(name, "group"): group for name, group in groups.items()
        }
        self.routes = {
            _require_name(name, "route"): route
            for name, route in (routes or {}).items()
        }
        if not all(isinstance(group, DynamicsGroup) for group in self.groups.values()):
            raise TypeError("groups must map names to DynamicsGroup instances")
        if not all(isinstance(route, SignalRoute) for route in self.routes.values()):
            raise TypeError("routes must map names to SignalRoute instances")

        self.group_names = tuple(sorted(self.groups))
        self.route_names = tuple(sorted(self.routes))
        self.n_nodes = int(graph.weights.shape[0])
        self.group_nodes = {
            name: self._normalize_nodes(name, self.groups[name].nodes)
            for name in self.group_names
        }
        self._validate_partition()
        self._validate_routes()
        self._validate_initial_states()
        self._history = None
        if history is not None:
            self.update_history(history)

    def _normalize_nodes(self, name, nodes):
        values = np.asarray(nodes)
        if values.dtype == np.bool_:
            if values.ndim != 1 or values.shape[0] != self.n_nodes:
                raise ValueError(
                    f"Boolean nodes for group {name!r} must have shape "
                    f"({self.n_nodes},), got {values.shape}"
                )
            values = np.flatnonzero(values)
        else:
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise TypeError(
                    f"nodes for group {name!r} must be a 1D integer array or mask"
                )
            values = values.astype(np.int64, copy=False)
        if values.size == 0:
            raise ValueError(f"group {name!r} must contain at least one node")
        if np.any(values < 0) or np.any(values >= self.n_nodes):
            raise ValueError(
                f"group {name!r} contains nodes outside [0, {self.n_nodes})"
            )
        if np.unique(values).size != values.size:
            raise ValueError(f"group {name!r} contains duplicate nodes")
        return tuple(int(value) for value in values)

    def _validate_partition(self):
        owner = np.full(self.n_nodes, -1, dtype=np.int64)
        for group_index, name in enumerate(self.group_names):
            nodes = np.asarray(self.group_nodes[name])
            overlap = nodes[owner[nodes] != -1]
            if overlap.size:
                raise ValueError(f"groups overlap at graph nodes {overlap.tolist()}")
            owner[nodes] = group_index
        missing = np.flatnonzero(owner == -1)
        if missing.size:
            raise ValueError(
                f"groups must cover every graph node; missing {missing.tolist()}"
            )

    def _validate_routes(self):
        known = set(self.groups)
        for route_name, route in self.routes.items():
            referenced = set(route.source) | set(route.local) | set(route.target)
            unknown = referenced - known
            if unknown:
                raise ValueError(
                    f"route {route_name!r} references unknown groups {sorted(unknown)}"
                )

            known_widths = set()
            for group_name, readout in route.source.items():
                if callable(readout):
                    continue
                dynamics = self.groups[group_name].dynamics
                missing = set(readout) - set(dynamics.STATE_NAMES)
                if missing:
                    raise ValueError(
                        f"route {route_name!r} source group {group_name!r} "
                        f"has unknown states {sorted(missing)}"
                    )
                known_widths.add(len(readout))
            if len(known_widths) > 1:
                raise ValueError(
                    f"route {route_name!r} source readouts have inconsistent "
                    f"known widths {sorted(known_widths)}"
                )

            for group_name, readout in route.local.items():
                if callable(readout):
                    continue
                dynamics = self.groups[group_name].dynamics
                missing = set(readout) - set(dynamics.STATE_NAMES)
                if missing:
                    raise ValueError(
                        f"route {route_name!r} local group {group_name!r} "
                        f"has unknown states {sorted(missing)}"
                    )

            for group_name, (input_name, conversion) in route.target.items():
                dynamics = self.groups[group_name].dynamics
                if input_name not in dynamics.COUPLING_INPUTS:
                    raise ValueError(
                        f"route {route_name!r} targets unknown input {input_name!r} "
                        f"on group {group_name!r}; expected "
                        f"{list(dynamics.COUPLING_INPUTS)}"
                    )
                if conversion is None:
                    expected = dynamics.COUPLING_INPUTS[input_name]
                    if route.coupling.N_OUTPUT_STATES != expected:
                        raise ValueError(
                            f"route {route_name!r} outputs "
                            f"{route.coupling.N_OUTPUT_STATES} channels but "
                            f"{group_name!r}.{input_name} expects {expected}; "
                            "provide an explicit conversion"
                        )

    def _validate_initial_states(self):
        for name in self.group_names:
            group = self.groups[name]
            n_group_nodes = len(self.group_nodes[name])
            if group.initial_state is None:
                continue
            shape = tuple(jnp.shape(group.initial_state))
            expected = (group.dynamics.N_STATES, n_group_nodes)
            if shape != expected:
                raise ValueError(
                    f"initial_state for group {name!r} has shape {shape}; "
                    f"expected {expected}"
                )

    def initial_state_for(self, name: str):
        if self._history is not None:
            return self._history.groups[name][-1]
        group = self.groups[name]
        if group.initial_state is not None:
            return jnp.asarray(group.initial_state)
        return group.dynamics.get_default_initial_state(len(self.group_nodes[name]))

    def update_history(self, solution) -> None:
        """Use a grouped result to warm-start state and delayed route history.

        All integrated state variables must be present. This mirrors
        ``Network.update_history``: variables-of-interest may be reordered, but
        auxiliary-only or reduced results cannot initialize a continuation.
        """
        if solution is None:
            self._history = None
            return

        from ..result import GroupedSolution

        if not isinstance(solution, GroupedSolution):
            raise TypeError("history must be a GroupedSolution or None")
        if set(solution.ys) != set(self.group_names):
            raise ValueError(
                "history groups must exactly match the heterogeneous network; "
                f"got {sorted(solution.ys)}, expected {list(self.group_names)}"
            )
        if solution.n_nodes != self.n_nodes:
            raise ValueError(
                f"history has n_nodes={solution.n_nodes}, expected {self.n_nodes}"
            )

        histories = Bunch()
        for name in self.group_names:
            actual_nodes = tuple(int(node) for node in solution.group_nodes[name])
            if actual_nodes != self.group_nodes[name]:
                raise ValueError(
                    f"history node ordering for group {name!r} is {actual_nodes}, "
                    f"expected {self.group_nodes[name]}"
                )
            names = solution.variable_names[name]
            if names is None:
                raise ValueError(
                    f"history group {name!r} has no variable_names metadata"
                )
            required = tuple(self.groups[name].dynamics.STATE_NAMES)
            missing = [state for state in required if state not in names]
            if missing:
                raise ValueError(
                    f"history group {name!r} is missing integrated states {missing}"
                )
            indices = [names.index(state) for state in required]
            histories[name] = solution.ys[name][:, indices, :]

        if len(solution.ts) == 0:
            raise ValueError("history must contain at least one time point")
        if len(solution.ts) > 1:
            coverage = float(solution.ts[-1] - solution.ts[0])
            required_coverage = effective_max_delay(self.graph)
            if coverage < required_coverage:
                warnings.warn(
                    f"History covers {coverage:.3f}s but network needs "
                    f"{required_coverage:.3f}s for delays. History will be padded.",
                    stacklevel=2,
                )
        self._history = Bunch(ts=solution.ts, groups=histories)

    def __repr__(self):
        groups = ", ".join(
            f"{name}:{self.groups[name].dynamics.__class__.__name__}"
            for name in self.group_names
        )
        return (
            f"HeterogeneousNetwork(n_nodes={self.n_nodes}, "
            f"groups={{ {groups} }}, routes={list(self.route_names)})"
        )
