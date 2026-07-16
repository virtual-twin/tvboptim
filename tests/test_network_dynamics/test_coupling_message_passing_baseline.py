"""Pre-refactor coupling matrix pinned before message-passing implementation.

Dense cells must agree with the independent NumPy/f64 oracle. Sparse cells do
the same except for the six failures that motivate the refactor; those are
strict xfails, so fixing one without updating this inventory fails as XPASS.

``SubspaceCoupling`` is intentionally outside this direct pre/post matrix. Its
nested sparse-node/dense-regional integration fixture belongs to P4.
"""

from dataclasses import dataclass
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedDifferenceCoupling,
    DelayedKuramotoCoupling,
    DelayedLinearCoupling,
    DelayedSigmoidalJansenRit,
    DifferenceCoupling,
    FastLinearCoupling,
    KuramotoCoupling,
    LinearCoupling,
    SigmoidalJansenRit,
)
from tvboptim.experimental.network_dynamics.coupling.linear import (
    DelayedSigmoidCoupling,
    SigmoidCoupling,
    TanhCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    JansenRit,
    Kuramoto,
    Linear,
)
from tvboptim.experimental.network_dynamics.graph import (
    DenseDelayGraph,
    DenseGraph,
    SparseDelayGraph,
    SparseGraph,
)

from .coupling_message_passing_oracle import coupling_oracle

WEIGHTS = np.array(
    [
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
DELAYS = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
EDGE_INDICES = np.argwhere(WEIGHTS != 0.0)


@dataclass(frozen=True)
class CouplingCase:
    name: str
    delayed: bool
    kind: str
    dynamics_factory: object
    coupling_factory: object
    incoming_names: tuple[str, ...]
    sparse_failure: bool = False


CASES = (
    CouplingCase(
        "linear",
        False,
        "linear",
        Linear,
        partial(LinearCoupling, incoming_states="x", G=0.7, b=0.1),
        ("x",),
    ),
    CouplingCase(
        "fast_linear",
        False,
        "linear",
        Linear,
        partial(FastLinearCoupling, local_states="x", G=0.7, b=0.1),
        ("x",),
    ),
    CouplingCase(
        "difference",
        False,
        "difference",
        Linear,
        partial(DifferenceCoupling, incoming_states="x", local_states="x", G=0.7),
        ("x",),
        sparse_failure=True,
    ),
    CouplingCase(
        "sigmoid",
        False,
        "sigmoid",
        Linear,
        partial(
            SigmoidCoupling,
            incoming_states="x",
            G=0.8,
            a=1.2,
            b=-0.1,
            slope=0.9,
            midpoint=0.3,
        ),
        ("x",),
    ),
    CouplingCase(
        "tanh",
        False,
        "tanh",
        Linear,
        partial(TanhCoupling, incoming_states="x", G=0.6, scale=1.1),
        ("x",),
    ),
    CouplingCase(
        "jansen_rit",
        False,
        "jansen_rit",
        JansenRit,
        partial(
            SigmoidalJansenRit,
            incoming_states=("y1", "y2"),
            G=1.1,
            cmin=0.02,
            cmax=0.8,
            midpoint=1.2,
            r=0.6,
        ),
        ("y1", "y2"),
        sparse_failure=True,
    ),
    CouplingCase(
        "kuramoto",
        False,
        "kuramoto",
        Kuramoto,
        partial(
            KuramotoCoupling,
            incoming_states="theta",
            local_states="theta",
            G=0.9,
        ),
        ("theta",),
        sparse_failure=True,
    ),
    CouplingCase(
        "delayed_linear",
        True,
        "linear",
        Linear,
        partial(DelayedLinearCoupling, incoming_states="x", G=0.7, b=0.1),
        ("x",),
    ),
    CouplingCase(
        "delayed_difference",
        True,
        "difference",
        Linear,
        partial(
            DelayedDifferenceCoupling,
            incoming_states="x",
            local_states="x",
            G=0.7,
        ),
        ("x",),
        sparse_failure=True,
    ),
    CouplingCase(
        "delayed_sigmoid",
        True,
        "delayed_sigmoid",
        Linear,
        partial(
            DelayedSigmoidCoupling,
            incoming_states="x",
            G=0.8,
            slope=0.9,
            midpoint=0.3,
        ),
        ("x",),
    ),
    CouplingCase(
        "delayed_jansen_rit",
        True,
        "jansen_rit",
        JansenRit,
        partial(
            DelayedSigmoidalJansenRit,
            incoming_states=("y1", "y2"),
            G=1.1,
            cmin=0.02,
            cmax=0.8,
            midpoint=1.2,
            r=0.6,
        ),
        ("y1", "y2"),
        sparse_failure=True,
    ),
    CouplingCase(
        "delayed_kuramoto",
        True,
        "kuramoto",
        Kuramoto,
        partial(
            DelayedKuramotoCoupling,
            incoming_states="theta",
            local_states="theta",
            G=0.9,
        ),
        ("theta",),
        sparse_failure=True,
    ),
)

EXPECTED_SPARSE_FAILURES = {
    "difference",
    "jansen_rit",
    "kuramoto",
    "delayed_difference",
    "delayed_jansen_rit",
    "delayed_kuramoto",
}


def _state(case):
    if case.kind == "jansen_rit":
        return np.array(
            [
                [0.2, 0.1, -0.3],
                [5.2, 6.1, 7.4],
                [1.1, 2.3, 3.2],
                [0.0, 0.2, 0.4],
                [-0.1, 0.3, 0.1],
                [0.5, -0.2, 0.6],
            ],
            dtype=np.float64,
        )
    if case.kind == "kuramoto":
        return np.array([[0.2, 1.1, -0.7]], dtype=np.float64)
    return np.array([[5.0, 7.0, 11.0]], dtype=np.float64)


def _history(current):
    delta = np.arange(1, current.size + 1, dtype=np.float64).reshape(current.shape)
    delta *= 0.025
    return np.stack([current - 2.0 * delta, current - delta, current])


def _graph(case, sparse):
    if case.delayed:
        cls = SparseDelayGraph if sparse else DenseDelayGraph
        return cls(jnp.asarray(WEIGHTS), jnp.asarray(DELAYS))
    cls = SparseGraph if sparse else DenseGraph
    return cls(jnp.asarray(WEIGHTS))


def _name_indices(dynamics, names):
    return np.array([dynamics.STATE_NAMES.index(name) for name in names], dtype=int)


def _pre_post(case, params):
    if case.kind == "linear":

        def pre(source, _target):
            return source

        def post(summed, _local):
            return float(params.G) * summed + float(params.b)

    elif case.kind == "difference":

        def pre(source, target):
            return source - target

        def post(summed, _local):
            return float(params.G) * summed

    elif case.kind == "sigmoid":

        def pre(source, _target):
            return source

        def post(summed, _local):
            linear = float(params.a) * summed + float(params.b)
            z = float(params.slope) * (linear - float(params.midpoint))
            return float(params.G) / (1.0 + np.exp(-z))

    elif case.kind == "delayed_sigmoid":

        def pre(source, _target):
            return source

        def post(summed, _local):
            z = float(params.slope) * (summed - float(params.midpoint))
            return float(params.G) / (1.0 + np.exp(-z))

    elif case.kind == "tanh":

        def pre(source, _target):
            return source

        def post(summed, _local):
            return float(params.G) * np.tanh(float(params.scale) * summed)

    elif case.kind == "jansen_rit":

        def pre(source, _target):
            state_difference = source[0] - source[1]
            denominator = 1.0 + np.exp(
                float(params.r) * (float(params.midpoint) - state_difference)
            )
            return np.array(
                [
                    float(params.cmin)
                    + (float(params.cmax) - float(params.cmin)) / denominator
                ]
            )

        def post(summed, _local):
            return float(params.G) * summed

    elif case.kind == "kuramoto":

        def pre(source, target):
            return np.sin(source - target)

        def post(summed, _local):
            return float(params.G) * summed

    else:  # pragma: no cover - CASES is the closed inventory
        raise AssertionError(f"Unknown oracle kind: {case.kind}")
    return pre, post


def _oracle(case, sparse):
    dynamics = case.dynamics_factory()
    coupling = case.coupling_factory()
    current = _state(case)
    history = _history(current)
    incoming_idx = _name_indices(dynamics, case.incoming_names)
    local_names = coupling.LOCAL_STATE_NAMES
    if isinstance(local_names, str):
        local_names = (local_names,)
    local_idx = _name_indices(dynamics, tuple(local_names))
    local = current[local_idx] if local_idx.size else None
    pre, post = _pre_post(case, coupling.params)

    return coupling_oracle(
        current[incoming_idx],
        WEIGHTS,
        pre=pre,
        post=post,
        target_local=local,
        history=history[:, incoming_idx] if case.delayed else None,
        delay_steps=DELAYS.astype(int) if case.delayed else None,
        edge_indices=EDGE_INDICES if sparse else None,
    )


def _compute(case, sparse):
    dynamics = case.dynamics_factory()
    coupling = case.coupling_factory()
    slot = "delayed" if case.delayed else "instant"
    graph = _graph(case, sparse)
    network = Network(dynamics, {slot: coupling}, graph)
    data, coupling_state = network.prepare(dt=1.0, t0=0.0, t1=1.0)
    current = _state(case)

    if case.delayed:
        incoming_idx = np.asarray(data[slot].incoming_indices)
        coupling_state[slot].history = jnp.asarray(_history(current)[:, incoming_idx])

    return np.asarray(
        network.compute_coupling_inputs(
            0.0, jnp.asarray(current), data, coupling_state
        )[slot]
    )


def _assert_normwise(actual, expected, *, rtol=1e-12, atol=1e-14):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    difference = actual - expected
    assert np.linalg.norm(difference) <= atol + rtol * np.linalg.norm(expected)
    assert np.max(np.abs(difference), initial=0.0) <= 1e-11


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_dense_builtin_matches_declared_order_oracle(case):
    _assert_normwise(_compute(case, sparse=False), _oracle(case, sparse=False))


def _sparse_case(case):
    if case.sparse_failure:
        return pytest.param(
            case,
            marks=pytest.mark.xfail(
                strict=True,
                reason="pre-refactor sparsify(pre) cannot handle this sparse body",
            ),
            id=case.name,
        )
    return pytest.param(case, id=case.name)


@pytest.mark.parametrize("case", [_sparse_case(case) for case in CASES])
def test_sparse_builtin_matches_declared_order_oracle(case):
    sparse = _compute(case, sparse=True)
    _assert_normwise(sparse, _oracle(case, sparse=True))
    _assert_normwise(sparse, _compute(case, sparse=False))


def test_inventory_pins_exactly_the_six_known_sparse_failures():
    actual = {case.name for case in CASES if case.sparse_failure}
    assert actual == EXPECTED_SPARSE_FAILURES
