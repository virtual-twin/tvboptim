"""Private prepared topology view and runtime integrity checks."""

from dataclasses import dataclass
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

TOPOLOGY_CHANGE_ERROR = (
    "Graph topology changed after prepare(); numerical edge data may be replaced "
    "or swept, but edge indices and representation must remain fixed. Reconstruct "
    "the graph and run prepare() again."
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PreparedTopology:
    """Static graph structure shared by every coupling in a prepared solve.

    Public graphs are currently square, but source and target sizes are kept
    separate so the transport layer does not bake that restriction in.
    Sparse edge indices preserve the graph's public ``(target, source)`` order.
    """

    representation: str
    n_source: int
    n_target: int
    has_delays: bool
    edge_indices: Optional[Any] = None

    @property
    def is_sparse(self) -> bool:
        return self.representation == "sparse"

    @property
    def n_edges(self) -> int:
        if self.edge_indices is None:
            return self.n_source * self.n_target
        return self.edge_indices.shape[0]

    @property
    def target_e(self):
        if self.edge_indices is None:
            raise AttributeError("Dense prepared topology has no COO target indices")
        return self.edge_indices[:, 0]

    @property
    def source_e(self):
        if self.edge_indices is None:
            raise AttributeError("Dense prepared topology has no COO source indices")
        return self.edge_indices[:, 1]

    def tree_flatten(self):
        # Keep arity fixed when Equinox partitions this PyTree and replaces a
        # child with None; changing custom-node arity makes the partial trees
        # impossible to combine again inside mapped execution.
        children = (self.edge_indices,)
        aux_data = (
            self.representation,
            self.n_source,
            self.n_target,
            self.has_delays,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        representation, n_source, n_target, has_delays = aux_data
        edge_indices = children[0]
        return cls(
            representation=representation,
            n_source=n_source,
            n_target=n_target,
            has_delays=has_delays,
            edge_indices=edge_indices,
        )


def prepare_graph_topology(graph) -> PreparedTopology:
    """Snapshot graph structure without copying numerical edge data."""
    from .sparse import SparseGraph

    weights = graph.weights
    n_target, n_source = weights.shape
    if isinstance(graph, SparseGraph):
        return PreparedTopology(
            representation="sparse",
            n_source=n_source,
            n_target=n_target,
            has_delays=hasattr(graph, "delays"),
            edge_indices=graph.edge_indices,
        )
    return PreparedTopology(
        representation="dense",
        n_source=n_source,
        n_target=n_target,
        has_delays=hasattr(graph, "delays"),
    )


def _require_shape(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{TOPOLOGY_CHANGE_ERROR} {label} shape is {actual}, expected {expected}."
        )


def validate_graph_topology(prepared: PreparedTopology, graph, anchor):
    """Attach a prepared-vs-live topology check to a used array.

    Shape and representation changes fail while tracing. Same-size sparse
    reorders require a value check; ``eqx.error_if`` ties that check to
    ``anchor``, which callers must use downstream so JIT cannot discard it.
    """
    from .sparse import SparseDelayGraph, SparseGraph

    is_sparse = isinstance(graph, SparseGraph)
    if is_sparse != prepared.is_sparse:
        raise ValueError(TOPOLOGY_CHANGE_ERROR)

    weights = graph.weights
    expected_shape = (prepared.n_target, prepared.n_source)
    _require_shape(weights.shape, expected_shape, "Weight matrix")

    has_delays = hasattr(graph, "delays")
    if has_delays != prepared.has_delays:
        raise ValueError(TOPOLOGY_CHANGE_ERROR)

    if not prepared.is_sparse:
        if has_delays:
            _require_shape(graph.delays.shape, expected_shape, "Delay matrix")
        return anchor

    if not hasattr(weights, "indices"):
        raise ValueError(TOPOLOGY_CHANGE_ERROR)
    current_weight_indices = weights.indices
    _require_shape(
        current_weight_indices.shape,
        prepared.edge_indices.shape,
        "Sparse weight indices",
    )
    changed = jnp.any(current_weight_indices != prepared.edge_indices)

    if prepared.has_delays:
        if not isinstance(graph, SparseDelayGraph) or not hasattr(
            graph.delays, "indices"
        ):
            raise ValueError(TOPOLOGY_CHANGE_ERROR)
        current_delay_indices = graph.delays.indices
        _require_shape(
            current_delay_indices.shape,
            prepared.edge_indices.shape,
            "Sparse delay indices",
        )
        changed = changed | jnp.any(current_delay_indices != prepared.edge_indices)

    return eqx.error_if(anchor, changed, TOPOLOGY_CHANGE_ERROR)
