"""Tests for HRF kernels (HRFKernel subclasses).

All tests are parametrized over KERNELS, so a new kernel type only needs to be
appended to that list to inherit the whole suite.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from tvboptim.observations.tvb_monitors import FirstOrderVolterraHRFKernel, HRFKernel

# Representative instance of every HRF kernel. New kernels only need to be
# appended here to be covered by every test below.
KERNELS = [
    FirstOrderVolterraHRFKernel(),
]

# Downsample step (ms) used as the convolution grid when evaluating kernels.
DT = 4.0


@pytest.fixture(params=KERNELS, ids=lambda k: type(k).__name__)
def kernel(request):
    return request.param


def _eval_over_support(kernel):
    """Evaluate `kernel` across its full support on a DT-spaced grid."""
    t = jnp.arange(0.0, kernel.duration, DT)
    return t, kernel(t, DT)


def test_is_hrf_kernel(kernel):
    assert isinstance(kernel, HRFKernel)


def test_duration_positive(kernel):
    assert kernel.duration > 0


def test_output_shape_matches_input(kernel):
    t, values = _eval_over_support(kernel)
    assert values.shape == t.shape


def test_output_finite(kernel):
    _, values = _eval_over_support(kernel)
    assert jnp.all(jnp.isfinite(values))


def test_starts_at_rest(kernel):
    # The hemodynamic response builds up from baseline: the first sample of the
    # kernel evaluated over its full support is 0. Evaluating over the whole
    # support (rather than at the single point t=0) matches how kernels are used
    # and keeps support-normalized kernels well-defined.
    _, values = _eval_over_support(kernel)
    assert jnp.allclose(values[0], 0.0, atol=1e-6)


def test_decayed_at_end_of_support(kernel):
    # `duration` is defined as the kernel's support, so the kernel should be
    # negligible compared to its peak by the time it ends.
    _, values = _eval_over_support(kernel)
    peak = jnp.max(jnp.abs(values))
    assert jnp.abs(values[-1]) < 0.05 * peak


def test_jittable(kernel):
    # Kernels are evaluated inside jitted optimization loops in tvboptim.
    # float32 transcendentals (exp/sin) differ by a few ULP between eager and
    # compiled execution, which blows up the relative error near a kernel's
    # zero crossings. Compare with a tolerance scaled to the kernel's peak.
    t = jnp.arange(0.0, kernel.duration, DT)
    eager = kernel(t, DT)
    jitted = jax.jit(lambda tt: kernel(tt, DT))(t)
    peak = jnp.max(jnp.abs(eager))
    assert jnp.allclose(eager, jitted, atol=1e-5 * peak, rtol=1e-4)