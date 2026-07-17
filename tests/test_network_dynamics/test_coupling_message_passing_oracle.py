"""Contract tests for the independent coupling oracle."""

import numpy as np
import pytest

from .coupling_message_passing_oracle import coupling_oracle

SOURCE = np.array(
    [
        [1.0, 2.0, 4.0],
        [-1.0, 0.5, 3.0],
        [2.0, -2.0, 1.0],
    ]
)
TARGET = np.array(
    [
        [0.25, -0.5],
        [1.5, 0.75],
        [-1.0, 2.0],
    ]
)
WEIGHTS = np.array(
    [
        [0.0, 2.0, -1.0],
        [3.0, 0.0, 4.0],
    ]
)
EDGE_INDICES = np.argwhere(WEIGHTS != 0.0)
DELAY_STEPS = np.array(
    [
        [0, 1, 2],
        [3, 0, 1],
    ]
)
HISTORY_DELTA = np.array(
    [
        [0.1, 0.2, 0.3],
        [-0.2, 0.1, 0.4],
        [0.5, -0.25, 0.1],
    ]
)
HISTORY = np.stack([SOURCE + (step - 3) * HISTORY_DELTA for step in range(4)])

EXPECTED = {
    (False, False): np.array(
        [
            [-0.075, 18.825],
            [-2.175, 8.325],
            [-5.325, 9.375],
        ]
    ),
    (False, True): np.array(
        [
            [0.0125, 14.625],
            [-1.825, 4.5625],
            [-3.05, 2.2],
        ]
    ),
    (True, False): np.array(
        [
            [0.135, 16.62],
            [-1.545, 8.535],
            [-4.59, 4.23],
        ]
    ),
    (True, True): np.array(
        [
            [0.1525, 13.155],
            [-1.405, 4.7025],
            [-2.56, -1.23],
        ]
    ),
}


@pytest.mark.parametrize("representation", ["dense", "sparse"])
@pytest.mark.parametrize("uses_local", [False, True])
@pytest.mark.parametrize("delayed", [False, True])
def test_declared_order_oracle_covers_all_transport_quadrants(
    representation, uses_local, delayed
):
    """Dense/sparse, source/local, and instant/delayed share one orientation."""

    def pre(source, target):
        if uses_local:
            return source - 0.5 * target
        return 1.5 * source - 0.25

    result = coupling_oracle(
        SOURCE,
        WEIGHTS,
        pre=pre,
        post=lambda summed, _local: 0.7 * summed + 0.1,
        target_local=TARGET if uses_local else None,
        history=HISTORY if delayed else None,
        delay_steps=DELAY_STEPS if delayed else None,
        edge_indices=EDGE_INDICES if representation == "sparse" else None,
    )

    assert result.dtype == np.float64
    assert result.shape == (3, 2)  # Q=3, rectangular N_target=2/N_source=3
    np.testing.assert_allclose(
        result, EXPECTED[(delayed, uses_local)], rtol=0, atol=1e-14
    )


def test_oracle_rejects_history_without_delay_steps():
    with pytest.raises(ValueError, match="history and delay_steps"):
        coupling_oracle(
            SOURCE,
            WEIGHTS,
            pre=lambda source, _target: source,
            post=lambda summed, _local: summed,
            history=HISTORY,
        )
