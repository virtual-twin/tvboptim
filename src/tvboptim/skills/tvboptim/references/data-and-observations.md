# Data and observations

## Contents

- [Load bundled data](#load-bundled-data)
- [Choose and apply an observation](#choose-and-apply-an-observation)
- [Compute FC and FCD](#compute-fc-and-fcd)
- [Build an FC fitting objective](#build-an-fc-fitting-objective)
- [Stream long simulations](#stream-long-simulations)
- [Verify observation semantics](#verify-observation-semantics)

## Load bundled data

Load structural arrays without hidden preprocessing:

```python
import jax.numpy as jnp

from tvboptim.data import (
    load_fcd_distribution,
    load_functional_connectivity,
    load_structural_connectivity,
)

weights, lengths, labels = load_structural_connectivity("dk_average")
fc_target = load_functional_connectivity("dk_average")
fcd_midpoints, fcd_density = load_fcd_distribution("dk_average")

assert weights.shape == lengths.shape == fc_target.shape == (84, 84)
assert len(labels) == 84
```

The structural datasets are `dk_average` with 84 regions and `dTOR` with 370 regions. Only `dk_average` currently provides bundled FC and FCD references. The loaders return raw weights and tract lengths in millimeters. Make normalization, thresholding, symmetrization, and the conduction-speed conversion explicit.

The bundled FCD distribution is a qualitative reference because its original sliding-window parameters are not recorded. Do not present a match to it as calibrated validation.

Graph constructors and `.random(...)` take a `density` parameter: the fraction of connections present (1.0 = fully connected). It defaults to 1.0. Graphs also expose a `.density` property with the same meaning.

## Choose and apply an observation

Native network results use `[time, variables_of_interest, nodes]`. Observation functions also accept raw arrays, and some accept four-dimensional TVB-style data. Preserve the node axis and select the intended variable.

Use current monitor names:

```python
from tvboptim.observations.tvb_monitors import HRFBold

monitor = HRFBold(
    period=1000.0,
    downsample_period=4.0,
    voi=0,
    history=transient_result,
)
bold_result = monitor(simulation_result)
```

`HRFBold` downsamples and convolves with an HRF kernel. The default kernel is `FirstOrderVolterraHRFKernel`; alternative kernels include gamma-based forms. Use `BalloonWindkesselBold` when the nonlinear hemodynamic ODE is the intended model.

Set `history` to a compatible prior `NativeSolution` or array when the HRF convolution needs a warm start. Avoid applying `t_offset` twice when chaining chunks; inspect timestamps at chunk boundaries.

For a `DiffraxSolver`, monitors require a concrete solution `dt`. Supply a `SaveAt(ts=...)` grid when the save interval cannot otherwise be inferred.

## Compute FC and FCD

```python
from tvboptim.observations import (
    compute_fc,
    compute_fcd,
    fc_corr,
    fcd_distribution,
    ks_distance,
    rmse,
    wasserstein_1d,
)

fc = compute_fc(bold_result, s_var=0, skip_t=20)
fcd, window_fcs = compute_fcd(
    bold_result,
    t_window=30,
    step_size=2,
    s_var=0,
    skip_t=20,
)
density = fcd_distribution(fcd, midpoints=fcd_midpoints)

fc_loss = rmse(fc, fc_target)
fcd_loss = wasserstein_1d(density, fcd_density, fcd_midpoints)
fcd_ks = ks_distance(density, fcd_density)
```

`skip_t`, `t_window`, and `step_size` count samples, not milliseconds. Convert scientific durations using the observation sampling period. `compute_fc` zeroes its diagonal. `compute_fcd` returns both the window-to-window FCD matrix and every window FC matrix, which can be memory-heavy.

`fc_corr` currently correlates flattened matrices, including their diagonals. If the analysis requires an upper-triangle-only statistic, implement that selection explicitly and state it.

## Build an FC fitting objective

Use the canonical sequence:

```text
prepared simulation -> neural trajectory -> BOLD observation -> FC -> scalar loss
```

For example:

```python
def observation(config):
    neural = solve_fn(config)
    bold = monitor(neural)
    return compute_fc(bold, s_var=0, skip_t=20)


def loss(config):
    return rmse(observation(config), fc_target)
```

Warm up the dynamics and HRF when required by the experiment. Keep empirical arrays static and outside the transformed loss. Before a long optimization, run one forward call, one gradient call, and a short optimizer trial.

## Stream long simulations

Use an online FC reducer when only long-run neural-state FC is required:

```python
from tvboptim.experimental.network_dynamics import solve
from tvboptim.experimental.network_dynamics.solvers import Heun
from tvboptim.observations import welford_cov

fc = solve(
    network,
    Heun(block_size=2000),
    t0=0.0,
    t1=120_000.0,
    dt=1.0,
    reduce=welford_cov(s_var=0),
)
```

This returns FC without stacking the trajectory. It keeps an `O(nodes²)` accumulator. Blocking alone does not remove trajectory memory.

For streamed HRF BOLD:

```python
from tvboptim.observations.tvb_monitors import (
    HRFBold,
    SubSampling,
    streaming_hrf_bold,
)

monitor = HRFBold(
    period=1000.0,
    downsample_period=10.0,
    downsample=SubSampling(period=10.0),
    voi=0,
)
bold = solve(
    network,
    Heun(block_size=2000),
    t0=0.0,
    t1=120_000.0,
    dt=1.0,
    reduce=streaming_hrf_bold(monitor, dt=1.0),
)
```

The streaming HRF reducer requires `SubSampling`; temporal averaging is not streamable through this path. Every block and the total step count must be multiples of `period / dt`. It returns a BOLD array, not a `NativeSolution`.

Blocked stochastic simulations generate noise per block and therefore do not reproduce the monolithic noise draw. Compare blocked and streaming variants only when they use the same key and block strategy.

## Verify observation semantics

- Assert the neural and observed time axes and sampling intervals.
- Assert `fc.shape == (n_nodes, n_nodes)` and finite off-diagonal values.
- Verify the transient skip is in samples and leaves enough data.
- Verify FCD has enough windows for a meaningful correlation matrix.
- Compare post-hoc and streamed results under compatible keys/blocking before trusting a reducer.
- Separate FC matrix RMSE, FC correlation, FCD distribution distance, and scientific goodness-of-fit; they answer different questions.
- Record preprocessing, parcellation, conduction speed, random key, observation model, TR, window, step, and skipped transient with reported results.
