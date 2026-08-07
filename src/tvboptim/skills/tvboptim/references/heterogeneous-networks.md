# Heterogeneous networks

## Contents

- [Decide whether the task needs one](#decide-whether-the-task-needs-one)
- [Partition the graph into groups](#partition-the-graph-into-groups)
- [Route signals between groups](#route-signals-between-groups)
- [Observe groups in graph order](#observe-groups-in-graph-order)
- [Read the prepared config](#read-the-prepared-config)
- [Read the solution](#read-the-solution)
- [Warm-start a continuation](#warm-start-a-continuation)
- [Respect current limitations](#respect-current-limitations)
- [Diagnose failures](#diagnose-failures)

## Decide whether the task needs one

Use `HeterogeneousNetwork` only when distinct node subsets of one shared graph need distinct dynamics, state dimensions, initial states, noise, or external inputs. A cortical population coupled to a differently parameterized subcortical population is the typical case.

Use an ordinary `Network` when one dynamics model applies to every node, including when its parameters vary per node. Per-node parameter arrays are simpler, cheaper, and fully supported on the standard path; reach for groups only when the state vectors themselves differ.

Group state stays an unpadded segmented PyTree, so groups with different state counts do not pay for a common padded width.

## Partition the graph into groups

`groups` must be an exhaustive, non-overlapping partition of every graph node. Nodes may be interleaved in graph order:

```python
import jax

from tvboptim.experimental.network_dynamics import HeterogeneousNetwork, NodeGroup
from tvboptim.experimental.network_dynamics.dynamics.tvb import JansenRit, Linear
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise

network = HeterogeneousNetwork(
    graph=DenseGraph(weights),
    groups={
        "cortex": NodeGroup(JansenRit(), nodes=[0, 1, 2]),
        "sub": NodeGroup(
            Linear(gamma=-0.2),
            nodes=[3, 4, 5],
            noise=AdditiveNoise(sigma=1e-3, key=jax.random.key(0)),
        ),
    },
)
```

Node selectors are either one-dimensional integer indices or a graph-sized boolean mask. Integer sequences preserve the supplied order; boolean masks normalize to ascending graph-node order. A group's `initial_state` has shape `[n_states, n_group_nodes]` and its columns follow that normalized order, which `network.group_nodes[name]` reports.

Noise and external inputs are group-local. `network.group_names` is sorted, and that order is canonical everywhere else.

## Route signals between groups

A `SignalRoute` names one signal transport across the shared graph. It packs a `[Q, n_nodes]` signal, performs one graph traversal, and delivers the result to named coupling inputs:

```python
from tvboptim.experimental.network_dynamics import SignalRoute
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling

route = SignalRoute(
    source={
        "cortex": lambda state, params: state[1:2] - state[2:3],
        "sub": "x",
    },
    coupling=LinearCoupling(G=0.1),
    target={"cortex": "instant", "sub": "instant"},
)
```

The route, not the coupling, owns signal selection. Construct the route's coupling **without** `source=` or `local=`; supplying either is an error.

- `source` maps a group name to a state name, a tuple of state names, or a `readout(state, params) -> [Q, n_group_nodes]` callable. Every source readout on one route must emit the same channel count `Q`.
- `target` maps a group name to a coupling-input name, or to `(input_name, conversion)` where `conversion(signal, params)` transforms the target-local transported slice. Without a conversion, the coupling's `N_OUTPUT_STATES` must equal the target's declared `COUPLING_INPUTS[input_name]`.
- `local` supplies target-local readouts and is required exactly when the coupling sets `PRE_USES_LOCAL`, as `DifferenceCoupling` does. Supplying it for a coupling that does not use local signals is an error.
- `source_params`, `local_params`, and `target_params` hold per-group readout parameters. They stay separate so a target-only group can carry local-readout parameters and so source and local readouts on one group do not share a namespace.

Routes that target the same input on the same group are summed after their conversions. Use distinct input names when the dynamics must receive the contributions separately.

Wrap a callable in `Readout(fn, name=..., reads="state")` when a clearer validation error is worth the extra line. Route readouts consume integrated `state`; observation readouts consume selected variables of interest and declare `reads="voi"`. A mismatch is rejected during `prepare`.

## Observe groups in graph order

Groups have different variables, so a common signal requires an explicit projection. `GroupObservation` maps each group's variables of interest into shared named channels:

```python
from tvboptim.experimental.network_dynamics import GroupObservation, prepare
from tvboptim.experimental.network_dynamics.solvers import Heun

observe = GroupObservation(
    {"cortex": lambda voi, params: voi[1:2] - voi[2:3], "sub": "x"},
    channels=("activity",),
)
solve_fn, config = prepare(network, Heun(), t0=0.0, t1=20.0, dt=0.1, observe=observe)
```

`len(channels)` must equal the readout channel width `Q`. Equal width establishes shape compatibility only; making group-specific transforms scientifically commensurable remains the user's responsibility, and the skill should say so rather than implying the projection validates the science.

Passing `observe=` changes the returned type. The result becomes a graph-order `NativeSolution` of shape `[n_time, n_channels, n_nodes]` whose `variable_names` are the channel names, not a `HeterogeneousSolution`.

Partial coverage is allowed for plain projection and fills uncovered nodes with `fill_value`. With `reduce=` it is rejected unless `allow_partial_coverage=True`, which is only appropriate for a reducer verified to be fill-aware; it does not stop a covariance reducer producing NaN rows or a BOLD reducer treating fill as real drive.

`reduce=` requires `observe=`, and it bounds forward trajectory memory only when the solver sets `block_size`. Without `block_size` the full trajectory is materialized before reduction and a warning says so.

## Read the prepared config

Inspect the config before assigning to it. Heterogeneous paths are group- and route-scoped:

- `config.groups.<group>.dynamics.<param>`;
- `config.groups.<group>.noise.sigma` and `.key` for a group with noise;
- `config.groups.<group>.external.<input_name>`;
- `config.routes.<route>.coupling.<param>`, such as the gain `G`;
- `config.routes.<route>.source_params.<group>`, `.local_params.<group>`, `.target_params.<group>`;
- `config.graph`, `config.initial_state.<group>`, and `config.observation.<group>` when observing.

Route and observation readout parameters stay live after `prepare()`, so they are available to sweeps and gradients alongside dynamics and coupling parameters. Route structure, readout callables, group membership, and sparse topology are static.

## Read the solution

Without `observe=`, `solve_fn` returns a `HeterogeneousSolution` that preserves natural group shapes:

```python
result = solve_fn(config)
assert result.ys["cortex"].shape == (200, 6, 3)   # [time, variables, group nodes]
assert result.ys["sub"].shape == (200, 1, 3)
assert result.variable_names["cortex"] == ("y0", "y1", "y2", "y3", "y4", "y5")
```

- `result.ys[group]` gives raw arrays; `result.groups[group]` gives a `NativeSolution` view.
- `result.sel(group, variables, nodes=None)` selects by name.
- `result.to_graph(variable, groups=None, fill_value=jnp.nan)` projects one variable back to `[n_time, n_nodes]` graph order, filling nodes whose group lacks that variable.
- `result.plot(...)` draws a single group or a multi-group overview.

## Warm-start a continuation

Pass a previous `HeterogeneousSolution` to warm-start integrated state and delayed route history:

```python
transient = solve_fn(config)
network.update_history(transient)
solve_fn, config = prepare(network, Heun(), t0=0.0, t1=5000.0, dt=1.0)
```

The history must come from the same partition: identical group names, identical node ordering per group, the same `n_nodes`, and every integrated state present in `variable_names`. Variables of interest may be reordered, but an auxiliary-only or reduced result cannot initialize a continuation. Short history relative to the maximum delay warns and is padded.

## Respect current limitations

State these when they constrain the requested design rather than working around them silently:

- one fixed square shared graph, one time step, and an exhaustive static node partition;
- native fixed-step solvers only; heterogeneous Diffrax execution is unsupported;
- routes accept `PrePostCoupling` implementations only, instantaneous or delayed;
- group membership cannot change after `prepare()`;
- reduction requires an explicit `GroupObservation`, and reducer observations must cover every graph node unless a fill-aware reducer opts into partial coverage;
- multiple clocks or node spaces, shared readout parameters across positions, and route-signal recording are not supported.

Supported alongside these: dense and sparse graphs, delayed history and continuation, noise, external inputs, `jit`, autodiff, `vmap`, checkpointing, `grad_horizon`, and `Space` sweeps.

Report the assembled structure instead of inferring it from the constructor call:

```python
from tvboptim.experimental.network_dynamics.utils import format_network, print_network

print_network(network)
```

Both cover groups, routes, readouts, targets, coupling inputs, and graph structure.

## Diagnose failures

- Partition error: groups overlap or leave nodes uncovered; compare `network.group_nodes` against `n_nodes`.
- Route rejects its coupling: the coupling was constructed with `source=` or `local=`, or is not a `PrePostCoupling`.
- Channel-width error: source readouts on one route disagree on `Q`, or `GroupObservation.channels` does not match the readout width.
- Unknown target input: the named input is absent from that group's `COUPLING_INPUTS`.
- Channel-count mismatch at a target: supply an explicit conversion instead of changing the coupling.
- Readout evaluation error: the callable must accept `(values, params)` and return `[Q, n_group_nodes]`; an empty parameter mapping for a parameterized readout is a common cause.
- Missing local readouts: the coupling sets `PRE_USES_LOCAL` and every target group needs a `local` entry.
- Continuation rejected: group names, node ordering, or integrated states do not match the current network.
