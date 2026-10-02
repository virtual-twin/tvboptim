"""External stateful temporal monitor using the public observation contract.

The same ``LowPass`` instance can run inside a native solve or on a stored
trajectory::

    monitor = LowPass(period=10.0, tau=50.0)
    online = solve(network, solver, t1=1000.0, dt=1.0, observe=monitor)
    posthoc = monitor(raw_solution)

After ``prepare(..., observe=monitor)``, ``tau`` is available as the live value
``config.monitor["tau"]``. No solver module or private monitor helper is used.
"""

import equinox as eqx
import jax.numpy as jnp

from tvboptim.observations import SampledMonitor


class LowPass(SampledMonitor):
    """Exponential low-pass filter sampled at completed period endpoints."""

    period: float = eqx.field(static=True, default=10.0)
    tau: object = 50.0

    def init(self, sample_spec):
        if jnp.shape(self.tau) not in ((), (sample_spec.shape[1],)):
            raise ValueError("LowPass tau must be scalar or have one value per node")
        dtype = jnp.result_type(sample_spec.dtype, self.tau)
        return jnp.zeros(sample_spec.shape, dtype=dtype)

    def step(self, state, sample, dt):
        decay = jnp.exp(-dt / self.tau)
        filtered = decay * state + (1.0 - decay) * sample
        return filtered, filtered
