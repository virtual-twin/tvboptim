import warnings

from tvboptim.experimental.network_dynamics.coupling import (
    DelayedDifferenceCoupling,
    DelayedKuramotoCoupling,
    DelayedLinearCoupling,
    DelayedSigmoidalJansenRit,
    DifferenceCoupling,
    KuramotoCoupling,
    LinearCoupling,
    SigmoidalJansenRit,
)


def test_repo_is_free_of_deprecated_selector_warnings():
    """Representative public constructions use only source=/local=."""
    constructors = (
        lambda: LinearCoupling(source="x"),
        lambda: DifferenceCoupling(source="x", local="x"),
        lambda: SigmoidalJansenRit(source=("y1", "y2")),
        lambda: KuramotoCoupling(source="theta", local="theta"),
        lambda: DelayedLinearCoupling(source="x"),
        lambda: DelayedDifferenceCoupling(source="x", local="x"),
        lambda: DelayedSigmoidalJansenRit(source=("y1", "y2")),
        lambda: DelayedKuramotoCoupling(source="theta", local="theta"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for construct in constructors:
            construct()
