# Exploration and optimization

## Contents

- [Prepare a pure objective](#prepare-a-pure-objective)
- [Explore parameter spaces](#explore-parameter-spaces)
- [Run sequentially or in parallel](#run-sequentially-or-in-parallel)
- [Mark and constrain parameters](#mark-and-constrain-parameters)
- [Optimize with OptaxOptimizer](#optimize-with-optaxoptimizer)
- [Verify exploration and fitting](#verify-exploration-and-fitting)

## Prepare a pure objective

Build one pure function from prepared config to the smallest result needed by the task:

```python
solve_fn, config = prepare(network, solver, t0=0.0, t1=1000.0, dt=1.0)

def observation(config):
    result = solve_fn(config)
    return compute_fc(result, s_var=0, skip_t=100)

def loss(config):
    simulated_fc = observation(config)
    return rmse(simulated_fc, empirical_fc)
```

Use the same pure loss for a grid, direct `jax.grad`, and `OptaxOptimizer`. Keep plotting, pandas, file I/O, caching, and NumPy conversion outside the transformed function.

For BOLD-derived FC, apply the selected BOLD monitor before `compute_fc`; do not compare a `[time, variable, node]` trajectory directly with a `[node, node]` target.

## Explore parameter spaces

Copy the prepared config, replace only intended leaves with axes, and choose combination semantics explicitly:

```python
import copy
import jax

from tvboptim.execution import SequentialExecution
from tvboptim.types import GridAxis, Space

grid_config = copy.deepcopy(config)
grid_config.dynamics.w = GridAxis(0.3, 0.8, 4)
grid_config.coupling.instant.G = GridAxis(0.05, 0.4, 5)

space = Space(grid_config, mode="product", key=jax.random.key(0))
results = SequentialExecution(loss, space).run()
frame = results.to_dataframe()
assert len(results) == 20
```

Available axes include `GridAxis`, `LogGridAxis`, `UniformAxis`, `DataAxis`, and `NumPyroAxis` from `tvboptim.types.spaces`. Stochastic axes use keys; record them.

Use `mode="product"` for a Cartesian product. Use `mode="zip"` for corresponding samples; different axis lengths are truncated to the minimum with a warning. Assign the same non-`None` `group` to axes that must be zipped together before the outer product/zip is formed.

Use `space.to_dataframe()` or an execution result's `to_dataframe()` for postprocessing. Keep array-valued cells as arrays rather than flattening away regional structure without explanation.

To optimize a swept slot, give the axis a `wrap=`. It is applied to that axis' value for each combination as it is substituted, so the model receives a `Parameter` and needs no wrapping logic of its own:

```python
from functools import partial
from tvboptim.types import DataAxis, GridAxis, SigmoidBoundedParameter

# Bounded axes supply their own low/high to wrap
config.coupling.instant.G = GridAxis(0.05, 0.4, 5, wrap=SigmoidBoundedParameter)

# DataAxis and NumPyroAxis carry no bounds, so bind them
config.dynamics.w = DataAxis(
    starts, wrap=partial(SigmoidBoundedParameter, low=0.3, high=0.8)
)
```

Arguments bound with `functools.partial` take precedence over the axis. Do not re-wrap substituted values by hand inside the model. Axes without `wrap` substitute raw arrays; `wrap` runs per combination, so `Space.N`, grouping, and `to_dataframe` are unaffected.

## Run sequentially or in parallel

Start with `SequentialExecution` for debuggability and memory visibility. Move to:

```python
from tvboptim.execution import ParallelExecution

execution = ParallelExecution(
    loss,
    space,
    n_pmap=jax.device_count(),
    n_vmap=4,
)
results = execution.run()
```

`n_pmap` is the number of devices. `n_vmap` is currently passed as the `batch_size` of `jax.lax.map` within each device despite its historical name. The space collector pads and trims as necessary; `n_pmap * n_vmap` does not need to equal `len(space)`.

Set `XLA_FLAGS=--xla_force_host_platform_device_count=N` before importing JAX only when deliberately testing multi-device CPU execution. Do not change global device configuration inside reusable library code or after JAX initialization.

When varying noise realizations, vary `config.noise.key` with a stochastic/data axis or a JAX tree update. Do not invent an integrator noise path.

## Mark and constrain parameters

Replace only leaves to optimize with parameter wrappers:

```python
from tvboptim.types import Parameter, SigmoidBoundedParameter, show_parameters

config.dynamics.w = SigmoidBoundedParameter(
    config.dynamics.w, low=0.0, high=1.0
)
config.coupling.instant.G = Parameter(config.coupling.instant.G)
show_parameters(config)
```

Choose wrappers by meaning:

| Wrapper | Use |
|---|---|
| `Parameter` | Unconstrained differentiable leaf |
| `BoundedParameter` | Hard clipping is intentional; gradients can vanish outside bounds |
| `SigmoidBoundedParameter` | Smooth bounded transform; can still numerically saturate |
| `NormalizedParameter` | Rescale parameters with very different magnitudes |
| `MaskedParameter` | Optimize selected array elements |
| `TransformedParameter` | Custom invertible constraint |

Use `.constrained_value` or `collect_parameters` for explicit postprocessing values. `.value` is the optimizer's raw leaf and may be in normalized, logit, or another transformed space.

## Optimize with OptaxOptimizer

```python
import optax

from tvboptim.optim import OptaxOptimizer

optimizer = OptaxOptimizer(loss, optax.adam(1e-3))
fitted_config, fitting_data = optimizer.run(config, max_steps=100, mode="rev")
```

If the loss returns auxiliary data, configure it explicitly:

```python
def loss_with_aux(config):
    simulated_fc = observation(config)
    return rmse(simulated_fc, empirical_fc), simulated_fc

optimizer = OptaxOptimizer(
    loss_with_aux,
    optax.adam(1e-3),
    has_aux=True,
)
```

Reverse mode is the default and is normally appropriate for scalar losses with many parameters. Forward mode can help when few differentiable parameters make reverse-mode simulation expensive or unstable; measure rather than assume.

Callbacks control printing, saving, stopping, and fitting data. `chunk_size` fuses several optimizer steps and invokes callbacks only at chunk boundaries using the last step in the chunk. Do not expect per-step history when chunking.

Optimizing a state with no `Parameter` leaves raises: there would be nothing to differentiate, and the run would return its starting state looking like convergence. If the state came from a `Space`, the axis is missing a `wrap=`.

Running the optimizer inside a traced call, such as a `ParallelExecution` model, compiles the step loop as one `lax.scan` so memory does not scale with `max_steps`. Do not reach for `n_vmap` to fix step-count memory; it controls batch width. Callbacks cannot run under a trace and raise, since they would fire once at trace time on tracers.

For long or chaotic simulations, `block_size` addresses activation memory and `grad_horizon` truncates temporal credit assignment. A finite, faster gradient is not automatically a scientifically acceptable gradient; compare with finite differences or a trusted short-horizon reference where feasible.

## Verify exploration and fitting

- Print marked parameter paths before optimization.
- Evaluate and record the initial loss.
- Assert exploration row count and inspect collected parameter columns.
- Compare a few sequential and parallel results using identical configs and keys.
- Confirm the fitted loss is finite and lower than the initial loss; avoid promising monotonic descent for every optimizer step.
- Extract constrained parameter values before reporting them.
- Re-run the fitted config outside the optimizer and recompute the reported observation.
- Check sensitivity to seeds, transient length, observation window, and bounds before scientific interpretation.
