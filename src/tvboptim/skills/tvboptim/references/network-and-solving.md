# Network construction and solving

## Contents

- [Choose the simulation level](#choose-the-simulation-level)
- [Build a network](#build-a-network)
- [Use solve and prepare](#use-solve-and-prepare)
- [Choose graphs, coupling, and solvers](#choose-graphs-coupling-and-solvers)
- [Sweep and differentiate delays](#sweep-and-differentiate-delays)
- [Warm-start simulations](#warm-start-simulations)
- [Control long-simulation memory and gradients](#control-long-simulation-memory-and-gradients)
- [Diagnose failures](#diagnose-failures)

## Choose the simulation level

Use a bare `AbstractDynamics` instance for a single node or several uncoupled nodes. Pass `n_nodes` only on the bare-dynamics path:

```python
from tvboptim.experimental.network_dynamics import solve
from tvboptim.experimental.network_dynamics.dynamics import Lorenz
from tvboptim.experimental.network_dynamics.solvers import Heun

result = solve(Lorenz(), Heun(), t0=0.0, t1=2.0, dt=0.01, n_nodes=3)
```

Use `Network` when structural connectivity, coupling, delays, or node-to-node interactions matter. One `Network` applies one dynamics model to every node. When different node subsets need different dynamics on one shared graph, read [heterogeneous-networks.md](heterogeneous-networks.md) instead.

## Build a network

This is the minimal current pattern for a delayed Reduced Wong-Wang network:

```python
import jax
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics import Network, prepare, solve
from tvboptim.experimental.network_dynamics.coupling import DelayedLinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import Heun

weights = jnp.array([[0.0, 1.0], [1.0, 0.0]])
delays = jnp.array([[0.0, 2.0], [2.0, 0.0]])

network = Network(
    dynamics=ReducedWongWang(w=0.7, INITIAL_STATE=(0.1,)),
    coupling={
        "delayed": DelayedLinearCoupling(source="S", G=0.1)
    },
    graph=DenseDelayGraph(weights=weights, delays=delays),
    noise=AdditiveNoise(sigma=1e-5, key=jax.random.key(0)),
)

result = solve(network, Heun(), t0=0.0, t1=4.0, dt=0.1)
assert result.ys.shape == (40, 1, 2)
assert result.variable_names == ("S",)
assert jnp.isfinite(result.ys).all()
```

For bundled anatomy, load raw arrays and make preprocessing explicit:

```python
from tvboptim.data import load_structural_connectivity

weights, lengths, labels = load_structural_connectivity("dk_average")
weights = weights / jnp.max(weights)
delays = lengths / 3.0  # mm divided by mm/ms gives ms
graph = DenseDelayGraph(weights, delays, region_labels=labels)
```

Do not silently normalize weights or choose a conduction speed for the user. Record the transformation and units.

## Use solve and prepare

`solve` prepares and immediately runs a simulation. `prepare` returns a pure function plus a mutable `Bunch` config:

```python
solve_fn, config = prepare(network, Heun(), t0=0.0, t1=4.0, dt=0.1)
result_eager = solve_fn(config)
result_jit = jax.jit(solve_fn)(config)

assert jnp.allclose(result_eager.ys, result_jit.ys)
print(config)
```

For a prepared network, expect these top-level paths:

- `config.dynamics`: dynamics parameters such as Reduced Wong-Wang `w`;
- `config.coupling.<name>`: parameters for each supplied coupling, such as `G`;
- `config.graph`: graph PyTree;
- `config.noise`: noise parameters and key when noise is present;
- `config.external`: external-input parameters when present;
- `config.initial_state.dynamics`, `.coupling`, and `.external`: runtime carry state;
- `config._internal`: prepared implementation data; do not edit casually.

Print or traverse the actual config before assigning a parameter. Reduced Wong-Wang has no dynamics parameter named `G`; global coupling belongs under `config.coupling.<name>.G`.

Native solvers save post-step states. For `t0=0`, `t1=4`, and `dt=0.1`, timestamps run from `0.1` through `4.0` and contain 40 samples.

## Choose graphs, coupling, and solvers

Use compatible component pairs:

| Need | Graph | Coupling |
|---|---|---|
| Instantaneous dense network | `DenseGraph` | `LinearCoupling` or another instantaneous subclass |
| Delayed dense network | `DenseDelayGraph` | `DelayedLinearCoupling` or another delayed subclass |
| Instantaneous sparse network | `SparseGraph` | a compatible instantaneous coupling |
| Delayed sparse network | `SparseDelayGraph` | a compatible delayed coupling |

Use measured memory/runtime behavior to choose dense versus sparse representations; do not rely on a universal node-count threshold.

## Sweep and differentiate delays

Delay read indices are rebuilt from the graph on every forward pass, so delays are live values. Assigning to them from a `GridAxis`, a `Space` sweep, or a gradient step changes the simulation without another `prepare()` call.

Use `DenseLengthGraph` when conduction speed is the quantity of interest. It owns `lengths` and a scalar `speed` and derives `delays = lengths / speed`, which makes speed a single sweepable leaf at `config.graph.speed`:

```python
from tvboptim.experimental.network_dynamics.graph import DenseLengthGraph

graph = DenseLengthGraph(weights=weights, lengths=lengths, speed=3.0)
```

Delay gradients need an interpolating history read. The default nearest-slot read leaves delays sweepable but not gradient-accessible, because the read index is piecewise constant:

```python
coupling = DelayedLinearCoupling(source="S", G=0.1, history_interpolation="linear")
```

`history_interpolation="linear"` blends the two neighboring history slots so `d/d(delays)` and `d/d(speed)` carry signal, and it enables the solver's stage-time shift, which removes the delay bias introduced by freezing coupling across stages. Choose it deliberately: it changes the numerical result, not only the gradient.

`KuramotoCoupling` and `DelayedKuramotoCoupling` are available for phase models.

Use native `Euler`, `Heun`, or `RungeKutta4` for delays, auxiliaries, variables-of-interest filtering, block scans, and streaming reductions. Use `BoundedSolver` only when state clipping is an intentional numerical/modeling choice.

Before choosing `DiffraxSolver`, inspect the current dispatch. The current experimental path rejects delayed coupling, streaming `reduce`, auxiliary recording, and variables-of-interest filtering in cases the native path supports. When exact output times matter, configure a `SaveAt(ts=...)` grid to avoid padded or irregular output.

Native `Heun` and `RungeKutta4` freeze coupling across stages by default. Set `recompute_coupling_per_stage=True` when instantaneous coupling must retain the base method's full stage order; delayed coupling does not need repeated history gathers.

## Warm-start simulations

Use a transient run when the scientific workflow requires settled dynamics:

```python
transient = solve(network, Heun(), t0=0.0, t1=1000.0, dt=1.0)
network.update_history(transient)
solve_fn, config = prepare(network, Heun(), t0=0.0, t1=5000.0, dt=1.0)
result = solve_fn(config)
```

`update_history` requires a three-dimensional solution whose variable names contain every entry in the dynamics model's `STATE_NAMES`. It slices away recorded auxiliaries and warns when the supplied history is shorter than the maximum delay.

## Control long-simulation memory and gradients

Treat these as separate controls:

- `Heun(block_size=K)` checkpoints a nested scan for backward memory and enables block-wise noise generation. It does not remove the stacked trajectory by itself.
- `reduce=(init, update, finalize)` folds outputs instead of returning the trajectory. It is supported by native solvers.
- `Heun(grad_horizon=W)` applies truncated backpropagation through time. The forward trajectory is unchanged, but gradients no longer cross window boundaries.
- `block_size` controls memory granularity; `grad_horizon` controls scientific credit-assignment horizon. Do not equate them.

For delayed networks, prefer `roll` or `circular` history buffers when bounding memory. The `preallocated` strategy grows with the simulation length and can dominate any checkpointing benefit.

Read the observation reference before using `welford_cov` or streaming BOLD reducers.

## Diagnose failures

- Unknown coupling key: compare `network.coupling` keys with `dynamics.COUPLING_INPUTS`.
- Missing coupling key: confirm zero-filling is scientifically intended.
- Delay error: pair a delayed coupling subclass with a delay graph and ensure history covers the maximum delay.
- Shape mismatch: distinguish integrated states from recorded variables of interest.
- Different stochastic results: compare keys and block/noise strategy before blaming the solver.
- Recompilation or tracer errors: keep shapes/static choices fixed and avoid Python/NumPy conversions inside transformed functions.
- Unexpected parameter invariance: confirm the edited config path is read by the selected component and assert an output changes.
