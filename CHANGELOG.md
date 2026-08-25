# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - Unreleased

### Added

- Added `EquinoxParameter`, allowing the inexact array leaves of an Equinox
  module embedded in a state tree to be differentiated and updated alongside
  ordinary `Parameter` values.
- Added an advanced UDE tutorial demonstrating a shared neural correction in a
  stochastic coupled FitzHugh--Nagumo network.
- **Heterogeneous neural-mass networks on one shared connectome.**
  `HeterogeneousNetwork` partitions graph nodes into named `NodeGroup`s with
  independent dynamics, state dimensions, initial states, noise, and external
  inputs while keeping state as an unpadded segmented PyTree.
  - `SignalRoute` maps group-specific source readouts through one instantaneous
    or delayed coupling operation to named target inputs. Each route traverses
    dense or sparse connectivity once, and routes targeting the same input sum.
  - `Readout` distinguishes callables that consume integrated `state` from
    selected variables of interest (`voi`). Role-specific parameter mappings
    remain live after `prepare()` for sweeps and differentiation.
  - `GroupObservation` transforms group outputs into a common graph-order
    signal for returned trajectories and existing reducers. With blockwise
    execution, streaming reduction bounds forward trajectory memory.
  - `HeterogeneousSolution` preserves natural group shapes and provides named
    selection, graph projection, single-group plots, and a multi-group overview.
  - Dense and sparse graphs, delayed history and continuation, native fixed-step
    solvers, noise, external inputs, `jit`, autodiff, `vmap`, checkpointing,
    `grad_horizon`, and `Space` sweeps are supported.
  - `format_network()` and `print_network()` describe groups, routes, readouts,
    targets, coupling inputs, and graph structure.
- Added an executable heterogeneous-network tutorial covering mixed routing,
  observations and streaming reduction, sweeps, optimization, continuation,
  plotting, and fixed-work group-count benchmarks.
- **Bundled agent skill.** `src/tvboptim/skills/tvboptim` ships an
  agent-neutral skill in the open Agent Skills format, covering network
  assembly and solving, heterogeneous networks, custom dynamics and coupling,
  data and observations, and exploration and optimization. It is included in
  the wheel and its examples are executed by the test suite.
  - A `tvboptim` console script installs it: `tvboptim skills install --agent
    claude-code --scope project`. `--agent` selects only the destination
    directory, since clients share the bundle format but not their discovery
    paths; `--destination` bypasses that mapping entirely.
  - `tvboptim skills export`, `status`, and `uninstall` complete the workflow,
    and `tvboptim.skills.install()` exposes the same behavior from Python.
  - Installation is opt-in and never happens as a side effect of installing the
    package. Installs record their version and a content hash, and refuse to
    overwrite or remove a locally modified copy without `--force`.
  - See `docs/basics/agent_skill.qmd` for the supported clients, the scopes,
    the Python API, and how to place the files by hand.

### Changed

- Coupling state selectors are now named `source=` and `local=`. The previous
  `incoming_states=` and `local_states=` spellings became misleading once a
  route, rather than the coupling, owns signal selection.
- TVB-O is now imported lazily, roughly halving `import tvboptim` when the
  optional package is installed. A fallback `prepare()` method imports TVB-O
  and registers its dispatch the first time a TVB-O object is passed, printing
  a short notice because that import takes seconds. Passing an unsupported type
  no longer triggers the import and still raises `NotFoundLookupError`.
  `HAS_TVBO` is resolved on access and now reports whether TVB-O is installed
  rather than whether it has been imported.
- **`MontbrioPazoRoxin` scales its r-coupling by `tau`.** The voltage equation
  now reads `... + I + tau * cr * c_coup_r + cv * c_coup_V`. The `tau` converts
  a firing rate into the voltage-like units of `V**2`, `eta` and `I`, exactly
  as the recurrent term `J * tau * r` does; coupling through `V` is already a
  voltage and correctly takes none. Without it the effective coupling strength
  depended on the population time constant. At the default `tau = 1.0` this is
  a no-op, so existing results and TVB parity are unaffected; only runs with
  `tau != 1` change.

### Fixed

- **`DataInput` parameters are now live on the prepared config.** `prepare()`
  built a diffrax interpolator from `times` and `data` and closed it over the
  step function, so `config.external.<name>.data` was published but never read.
  Editing it had no effect, its gradient was exactly zero, and `vmap` over a
  batch of signals returned the same trajectory for every element, all without
  an error. `compute()` now interpolates from `params` like every other
  external input, so the samples are an ordinary differentiable, sweepable
  leaf. Rebuilding the interpolation per step costs nothing: the cubic
  coefficients do not depend on `t`, so XLA hoists them out of the integration
  loop.
- `DataInput` now runs on the bare-dynamics path. `prepare()` read
  `network.graph.n_nodes` for broadcasting and raised `AttributeError` when
  `solve`/`prepare` passed no network. The node count now comes from the state,
  as it already did for the parametric inputs.
- **Corrected model equations in the API reference.** An audit of all 13
  dynamics classes against their implementations found three documented
  equations that did not match the code, in every case because the docstring
  was wrong rather than the model. `CoombesByrne2D` documented a rate damping
  term cubic in `r` where the conductance `g = k pi r` makes it quadratic
  (`- g r`, matching TVB). `Generic2dOscillator` omitted the `gamma` scaling on
  the instantaneous coupling and the external stimulus term entirely, and now
  states that the stimulus is added outside the `d tau` factor, as TVB's
  integrator does. `Kuramoto` documented a sinusoidal transform of the
  instantaneous coupling that the code never applied.
- **Removed dead code in `Kuramoto.dynamics()`.** A local-coupling term was
  computed as `sin(0 * theta)`, identically zero for every input, ported from
  TVB where the factor is the surface-local connectivity kernel that tvboptim
  does not have. The phase interaction is supplied by `KuramotoCoupling` and
  `DelayedKuramotoCoupling`; applying a second sine here would transform it
  twice. Behaviour is unchanged.
- Documented the `LarterBreakspear` state equations and ionic currents, which
  were previously absent from its docstring, and noted that `CoombesByrne2D`
  and `MontbrioPazoRoxin` inherit different time conventions from TVB, so
  comparing them requires `tau = 1`.


### Deprecated

- Deprecated `incoming_states=` and `local_states=` on couplings, removed in
  1.0. Both remain accepted as aliases and emit a `DeprecationWarning`; passing
  a name and its alias together is an error. Use `source=` and `local=`.

### Known limitations

- Networks use one fixed square graph, an exhaustive static node partition, one
  time step, native fixed-step solvers, and `PrePostCoupling` routes.
- Heterogeneous reduction requires an explicit `GroupObservation`; reducer
  observations must cover every graph node unless a fill-aware reducer opts
  into partial coverage.
- Heterogeneous Diffrax execution, changing group membership after `prepare()`,
  multiple clocks or node spaces, shared readout parameters, and route-signal
  recording are not yet supported.

## [0.4.0] - 2026-07-17

### Added

- **Transmission delays are now live values that can be swept and
  differentiated.** Delay read indices are rebuilt from the graph on every
  forward pass, so assigning to `delays` (from a `GridAxis`, a `Space` sweep, or
  a `jax.grad` step) changes the simulation without another `prepare()` call.
  - Added `DenseLengthGraph`, which owns `lengths` and a scalar `speed` and
    derives `delays = lengths / speed`. Conduction speed becomes a single
    sweepable, differentiable leaf, the delay-domain counterpart of the coupling
    gain `G`.
  - Added `KuramotoCoupling` and `DelayedKuramotoCoupling`.
  - Added `history_interpolation="linear"` on delayed couplings, which blends
    two history slots so `d/d(delays)` is informative. The default nearest-slot
    read leaves delays sweepable but not gradient-accessible, and is unchanged.
  - `history_interpolation="linear"` also enables the solver's stage-time shift,
    which removes the delay bias introduced by freezing the coupling across
    solver stages and restores second-order accuracy in the delayed term. This
    is opt-in: default delayed simulations are numerically unchanged.
  - Added `max_delay_bound` on the delay graphs, declaring buffer headroom so a
    sweep or gradient step can grow a delay beyond its initial value, plus
    `warn_on_delay_clamp` on delayed couplings to surface delays that exceed it.
  - Added the `effective_max_delay` and `delay_steps_bound` helpers.
  - See `docs/workflows/Delay_Speed_Synchronization.qmd` for a worked
    speed-sweep and gradient workflow.
- **Swept values can now initialize optimisable parameters.** An axis accepts an
  optional `wrap=` callable, applied when each selected value is materialised
  into the state. This supports multi-start optimisation without parameter
  construction inside the mapped model.
  - Wrapper arguments are explicit because axis sampling bounds and parameter
    constraints are separate concepts. Bind bounds with `functools.partial`,
    for example `wrap=partial(SigmoidBoundedParameter, low=0.05, high=3.0)`.
  - Wrapper configuration is explicit; invalid callables fail naturally when
    a selected value is materialised.
  - The wrap runs on the per-combination value rather than the stacked array,
    so `size`, `Space.N`, axis grouping, and `to_dataframe` are unaffected.
    Axes without `wrap` substitute raw arrays.
  - `Space.collect(combine=True)` rejects wrapped spaces because a vector
    parameter is not generally equivalent to independently materialised lane
    parameters. `collect(combine=False)` returns raw batched axis arrays.
  - See `docs/basics/axes_and_spaces.qmd` for a worked multi-start optimisation.
- **Added `RescaledParameter` for optimisation in user-defined units.** Its leaf
  is `value / scale`, so the scale relates optimiser updates to changes in the
  physical parameter.
  - Choose a scale from the characteristic expected change. Initial-value
    scaling can be unsuitable near zero or when the parameter must change sign.
  - `NormalizedParameter` is now the `scale = value` special case and is
    implemented on top of it. Its public behaviour is unchanged: `.value` holds
    ones, `.scale` returns the original value, and `.constrained_value` returns
    `scale * .value`.
  - See `docs/workflows/Hopf_Pareto_ParallelOpt.qmd`, where the coupling gain
    and the bifurcation parameter need different scales.
- Added `Parameter.constrained_value` and clearer parameter display helpers.
- Added Python 3.14 support.
- Added stable sparse `edge_indices` and `gather_edges()` APIs for constructing
  edge-aligned parameters without reaching into prepared coupling internals.

### Changed

- **`OptaxOptimizer.run` uses `lax.scan` when traced.** This prevents the
  compiled graph from growing linearly with `max_steps`. If `chunk_size` is
  unset, traced execution uses one scan. An explicit smaller value produces
  multiple sequential scans and may increase compilation overhead.
  - Traced execution rejects Python callbacks because they cannot observe
    individual runtime steps.
  - `chunk_size` values below 1 are rejected; `chunk_size=0` previously hung.
- **Array-valued parameter configuration is no longer stored as pytree
  metadata.** JAX metadata requires scalar equality and hashability. The scale
  for `NormalizedParameter`, and the mask and fixed values for
  `MaskedParameter`, are now captured by their transforms. This permits stable
  tree comparisons in `lax.scan` and batched optimisation.
- **`LogPositiveParameter` and `LogNegativeParameter` support traced
  construction.** Concrete inputs retain strict range validation. Wrapped axis
  values are validated before mapped execution, and traced inverse transforms
  clamp their bound offsets to a positive finite value.
- **`Space` preserves parameters that are not on an axis.** States can combine
  swept values with existing optimisable parameters.
- **`OptaxOptimizer.run` rejects states without `Parameter` leaves.** This
  prevents an ineffective optimisation when a swept value is missing `wrap=`.
- **Custom coupling `pre()` now receives pre-aligned operands and must be
  elementwise.** The framework fetches source and target values, applies edge
  weights, and reduces; `pre()` performs only elementwise math on operands that
  already share a layout. Built-in couplings are migrated, so this affects
  custom couplings only.
  - Remove explicit message-axis reshapes such as `[:, :, None]` or
    `[None, :, :]`.
  - Set `PRE_USES_LOCAL = True` when `pre()` folds in the target state, and
    list edge-shaped parameters in `EDGE_PARAMS`.
  - Incoming-only `pre()` now receives `local_states=None`. Post-aggregation
    local effects belong in `post()`, which still receives node-shaped local
    states.
  - `prepare()` validates the contract before compilation and raises with a
    migration message, so an unmigrated coupling fails loudly rather than
    silently computing a wrong result.
  - See `docs/network_dynamics/coupling.qmd` for the migration guide.
- Reworked instantaneous and delayed coupling around aligned node/edge message
  passing. Sparse local, nonlinear, edge-parameter, and delayed transforms now
  execute in O(E) rather than materializing node-by-node operands.
- Coupling sums now reduce in edge order, so results shift within dtype
  tolerance relative to 0.3.x. Comparisons against older versions should use a
  normwise tolerance rather than bit-exact equality.
- Graph topology is now fixed for a prepared solve. Numerical weight, delay, and
  edge-parameter data may still be replaced, swept, or differentiated, but
  replacing or reordering edge indices after `prepare()` is rejected with a
  reconstruction message, and `Space` rejects an axis placed on an index leaf.
- Switched the documentation build to committed Quarto freeze artifacts.
- Refined dense and sparse random graph generation, including explicit density
  semantics, low-density edge handling, and unique edge sampling. Sparse
  generators no longer emit duplicate edges, so a given seed produces a
  different graph than in 0.3.x.

### Deprecated

- Deprecated `FastLinearCoupling`. `LinearCoupling` now uses the same optimized
  incoming-only path, so the separate class is unnecessary. It remains as a
  compatibility wrapper that maps its historical `local_states=...` spelling
  onto `incoming_states` and emits a `DeprecationWarning`; use
  `LinearCoupling(incoming_states=...)` instead.

### Fixed

- Fixed the crash that made difference, phase, and Jansen-Rit style couplings
  unusable on sparse graphs (`NotImplementedError: Subtraction between sparse
  and dense array`). `DifferenceCoupling`, `KuramotoCoupling`,
  `SigmoidalJansenRit`, and their delayed variants now run on sparse graphs and
  agree with their dense results within tolerance.
- Corrected `FastLinearCoupling` to reduce dense directed connectivity using
  the documented `weights[target, source]` orientation. Results on asymmetric
  graphs now agree with `LinearCoupling`; symmetric graphs are unchanged.
- Fixed parameter arithmetic when values are JAX tracers.

## [0.3.1] - 2026-06-26

### Added

- Added solver controls for truncated gradients, block-wise checkpointing and
  streaming noise, plus per-call streaming reductions for long simulations.
- Added adiabatic scans and refined Lyapunov analysis.

### Changed

- Moved noise state into the scanned solver carry and consolidated solver
  execution around the block-size controls.

### Fixed

- Added a NumPy/Numba compatibility floor so supported Python versions resolve
  installable documentation and test environments.

## [0.3.0] - 2026-06-02

### Changed

- **Network coupling is now frozen across solver stages by default.** On the
  `Network` + `NativeSolver` path, multi-stage solvers (Heun, RK4) evaluate
  the coupling input once per step at `(t_n, y_n)` and reuse it for every
  stage, instead of recomputing it at each stage's own `(time, state)`. This
  matches TVB's integration scheme and avoids the redundant per-stage coupling
  cost (2x for Heun, 4x for RK4).
  - For delayed coupling the change is bit-identical: the delay history buffer
    is a step-level carry that does not depend on the stage state.
  - For instantaneous (state-dependent) coupling results will differ. Freezing
    pins the coupling component to first order regardless of the base method.
    Pass `recompute_coupling_per_stage=True` to the solver (e.g.
    `Heun(recompute_coupling_per_stage=True)`) to restore the previous
    per-stage behavior and recover full method order.
  - Euler (single stage) and the Diffrax path are unaffected. External inputs
    are always evaluated per stage regardless of the flag.
  - See `docs/advanced/coupling_freezing.qmd` for the accuracy/performance
    trade-off in detail.
- Made `prepare()` snapshot its input state as documented, preventing later
  mutation of authoring objects from changing an already prepared solve.

### Added

- `recompute_coupling_per_stage` flag on `NativeSolver` (and subclasses
  `Euler`, `Heun`, `RungeKutta4`, `BoundedSolver`).

### Removed

- Removed the unused abstract `NativeSolver` stub in
  `solvers/base.py`. The public `network_dynamics.solvers.NativeSolver` now
  refers to the concrete base class that `Euler`/`Heun`/`RungeKutta4` actually
  inherit from, so `isinstance(Heun(), NativeSolver)` is now `True`.

## [0.2.11] - 2026-05-26

### Added

- Added qualitative FCD reference data and related observation functions.
- Added local parameter-identifiability analysis and tests.

## [0.2.10] - 2026-05-22

### Added

- Added gamma, double-exponential, and mixture-of-gammas HRF kernels.
- Added plotting for external inputs and Bayesian and multi-objective Hopf
  workflows.

### Changed

- Cleaned up and expanded tests for the existing first-order Volterra HRF
  kernel.

## [0.2.9] - 2026-05-18

### Added

- Added native-solver gradient checkpointing for long delayed simulations.
- Made noise realizations sweepable by moving sampling out of deep
  preparation.

### Changed

- Flattened result trees with key-aware paths and rewrote the axes/space
  documentation around the current API.

## [0.2.8] - 2026-04-22

### Added

- Added the Balloon-Windkessel BOLD monitor.
- Added variable-name metadata to solution objects.
- Added chunked Optax execution for small optimizations.

### Fixed

- Fixed BOLD downsampling-period propagation and network-history slicing.
- Made monitor and solver time/variable naming consistent.
- Reduced optimizer scan memory and fixed the captured `value_fn` closure.

## [0.2.7] - 2026-03-27

### Changed

- Registered `Bunch` as a key-aware JAX PyTree.
- Vectorized `Space`/result DataFrame conversion to remove its Python loop.

## [0.2.6] - 2026-03-26

### Added

- Added grouped parameter axes and `to_dataframe()` on spaces and execution
  results.
- Added package attestations to the publishing workflow.

### Fixed

- Corrected the TVB dependency declaration.

## [0.2.5] - 2026-03-17

### Added

- Added maximum and full-spectrum Lyapunov analysis.
- Added direct solving of dynamics without constructing a network.
- Made graph weights part of the differentiable prepared configuration.
- Exposed `dt` on wrapped Diffrax solutions where available.

### Changed

- Standardized the network attribute name as `network.coupling`.

### Fixed

- Fixed broadcasting in `BoundedSolver`.

## [0.2.4] - 2025-12-17

### Added

- Added `SubspaceCoupling` for surface and other two-scale network simulations.
- Included example datasets in built distributions.

### Fixed

- Corrected package `__version__` reporting.

## [0.2.3] - 2025-11-28

### Added

- Added `LogGridAxis` and selectable delay-buffer strategies.
- Added Colab-ready committed notebooks and tutorial links.

### Changed

- Refactored the base coupling hierarchy.
- Switched packaged data loading to `importlib.resources`.

## [0.2.2] - 2025-11-26

### Fixed

- Fixed delay coupling at infinite conduction speed and corrected history
  updates.
- Corrected several dynamics-model differences found by new TVB comparison
  tests.

## [0.2.1] - 2025-11-21

### Changed

- Renamed the compilation entry point from `jaxify` to `prepare`.
- Added automated tests, publishing, documentation builds, and Ruff checks.

## [0.2.0] - 2025-11-20

- Initial public release.

---

Historical additions were reconstructed from tagged commit ranges and the
explicit `0.2.11` version-bump commit. They summarize notable user-facing
changes and are intentionally not an exhaustive commit log.
