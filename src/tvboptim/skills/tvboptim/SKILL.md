---
name: tvboptim
description: Build, run, inspect, extend, explore, and optimize differentiable brain-network simulations with the Python TVB-Optim package. Use when a task involves tvboptim or its experimental Network Dynamics API, custom dynamics or coupling, heterogeneous multi-group networks, ReducedWongWang or JansenRit networks, Parameter and OptaxOptimizer, Space and ParallelExecution, bundled connectivity data, BOLD/FC/FCD observations, delays, or memory-conscious long JAX simulations. Do not use for generic TVB, JAX, Optax, or neuroscience questions that do not involve TVB-Optim.
---

# TVB-Optim

Build workflows against the TVB-Optim version that is actually present, then verify the smallest representative computation before scaling it up. Treat `tvboptim.experimental.network_dynamics` as experimental even when it is the appropriate interface.

## Establish the source of truth

1. In a source checkout, read `pyproject.toml`, the relevant implementation under `src/tvboptim`, and nearby tests before trusting prose examples.
2. Outside a checkout, inspect `importlib.metadata.version("tvboptim")`, import paths, and callable signatures from the installed distribution.
3. Prefer current source and tests over an example written for another release. State explicitly when code targets the experimental Network Dynamics API.
4. Do not mix APIs from a repository checkout, a different installed wheel, published documentation, or upstream TVB without checking compatibility.

## Route the task

- Read [network-and-solving.md](references/network-and-solving.md) for network assembly, graphs, delays, noise, solvers, `solve`, `prepare`, warm starts, or long simulations.
- Read [heterogeneous-networks.md](references/heterogeneous-networks.md) only when distinct node subsets of one shared graph need distinct dynamics, for `HeterogeneousNetwork`, `NodeGroup`, `SignalRoute`, `Readout`, `GroupObservation`, or `HeterogeneousSolution`. Per-node parameter variation within one dynamics model does not need it.
- Read [custom-dynamics-and-coupling.md](references/custom-dynamics-and-coupling.md) before implementing or reviewing a dynamics model or coupling class.
- Read [exploration-and-optimization.md](references/exploration-and-optimization.md) for axes, parameter spaces, sequential/parallel execution, parameter constraints, gradients, callbacks, or fitting.
- Read [data-and-observations.md](references/data-and-observations.md) for bundled SC/FC/FCD data, monitors, BOLD, FC/FCD metrics, or streaming observations.

Load only the references needed for the requested task. For an end-to-end fitting workflow, read network/solving first, then data/observations and exploration/optimization.

## Follow the common workflow

1. Classify the task as simulation, extension, exploration, optimization, observation, or a combination.
2. Preserve the requested dynamics, solver, delays, noise, data, precision, and hardware assumptions. Ask only when a missing scientific choice would materially change the result.
3. Inspect component declarations and the prepared config instead of guessing coupling names, state names, or parameter paths.
4. Use `solve(...)` for one immediate run. Use `prepare(...)` when a pure `solve_fn(config)` is needed for JIT, differentiation, repeated execution, exploration, optimization, or runtime config changes.
5. Start with a deterministic, short, small-node smoke case. Add a fixed JAX random key when noise or stochastic axes are present.
6. Verify imports, time grid, output names and ranks, finite values, intended parameter effects, and eager/JIT agreement where relevant.
7. Scale duration, node count, parameter-space size, parallelism, and gradient memory only after the smoke case passes.

## Respect model and array semantics

- Treat solver output as `[time, variables_of_interest, nodes]`; axis 1 is not necessarily every integrated state because variables of interest may include or omit auxiliaries.
- Treat native-solver timestamps as post-step samples on `(t0, t1]`. Require `(t1 - t0) / dt` to be integral when exact endpoint behavior matters.
- Match `Network.coupling` dictionary keys to `dynamics.COUPLING_INPUTS`. Missing declared inputs are zero-filled; unknown keys are errors.
- Match delayed coupling classes with delay graphs. A dictionary key named `delayed` does not turn an instantaneous coupling class into a delayed one.
- Select coupling states with `source=` and `local=`. The older `incoming_states=` and `local_states=` keywords still work, warn, and are removed in 1.0; do not write them into new code.
- For a prepared network, treat `config.initial_state` as structured dynamics, coupling, and external state. Inspect it before editing.
- Change a real prepared path such as `config.coupling.instant.G`, `config.coupling.delayed.G`, `config.dynamics.w`, or `config.noise.sigma`; do not invent paths from model terminology.
- Keep JAX-traced work in JAX arrays and pure functions. Avoid converting traced values to Python or NumPy inside losses and mapped functions.

## Verify before handing off

Run the narrowest relevant checks and report what was actually exercised:

- execute the minimal example or targeted test;
- assert expected output shape and variable names;
- assert finite results;
- compare eager and `jax.jit` output when compilation is part of the task;
- demonstrate that the selected parameter changes the intended result;
- for optimization, compare initial and final loss and inspect marked parameters;
- for stochastic work, record keys and avoid comparing different noise realizations as if they were numerical disagreement;
- for long simulations, distinguish trajectory memory, noise memory, backward activation memory, and gradient-horizon stability.

Do not claim scientific validity from a smoke test. Separate software correctness, numerical agreement, and scientific interpretation.

## Capture reusable improvements

Keep the user's task primary. When TVB-Optim itself is broken, unsupported, or unusually difficult, or when the work produces something broadly reusable, capture the evidence while it is fresh and offer an upstream path:

- For a reproducible defect or concrete missing capability, offer a GitHub issue draft.
- For an open usage question, API-design tradeoff, or uncertain proposal, suggest a GitHub Discussion when the repository enables it.
- For a useful workflow or workaround, offer a compact documentation example or regression test instead of leaving it only in application code.
- For a generic new dynamics model, coupling, solver, monitor, metric, observation, or data adapter, offer to extract a focused upstream patch with tests and documentation. Include scientific references, equations or conventions, parameter units and defaults, state/coupling/output names, and comparison against a trusted implementation when applicable.

Before proposing upstream feedback:

1. Confirm the behavior against the current source, documentation, and tests; distinguish TVB-Optim from user-code, dependency, JAX, hardware, or agent-harness failures.
2. Reduce failures to a minimal reproduction. Record the TVB-Optim version or commit, expected and actual behavior, traceback, platform, JAX backend, shapes/dtypes, random key, and eager/JIT distinction when relevant.
3. Check for an existing issue, discussion, test, or implementation when network access is permitted.
4. Prefer a copy-ready title/body or local patch for review. Never post an issue, discussion, comment, branch, or pull request without the user's explicit approval.
5. Remove credentials, private data, proprietary code, identifying local paths, and large logs. Confirm that the user can share contributed code and scientific material.

Do not silently collect or upload telemetry. Do not interrupt a viable local workaround merely to pursue upstream feedback; finish or unblock the requested task first, then offer the contribution.
