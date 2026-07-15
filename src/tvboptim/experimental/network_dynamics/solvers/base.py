"""Base solver classes for  Network Dynamics.

This module defines the abstract base classes for all solver types.
"""

from abc import ABC


class AbstractSolver(ABC):
    """Base class for all solver types.

    Attributes
    ----------
    stage_time_centroid : float
        The method's stage-time centroid ``sum_i b_i * c_i``: where in the
        step the method's drift contribution is effectively evaluated, as a
        fraction of ``dt``. 0 for a single-stage method evaluated at the step
        start (Euler), 1/2 for any method satisfying the order-2 condition
        (Heun, RK4).

        Delayed couplings read it to undo the bias that freezing the coupling
        across stages introduces: holding the coupling at its step-start value
        is the same as evaluating it on time with every delay lengthened by
        ``stage_time_centroid * dt``, so the read subtracts that back.

        A custom solver that leaves this at the default 0.0 gets the old
        (unshifted) read, which is correct for a step-start evaluation and
        merely first-order accurate otherwise.
    """

    stage_time_centroid: float = 0.0
