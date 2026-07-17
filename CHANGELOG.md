# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `Parameter.constrained_value` and clearer parameter display helpers.
- Added Python 3.14 support.
- Exposed transmission delays as live values for parameter sweeps and gradients,
  with interpolated delay reads and delayed Kuramoto coverage.
- Added stable sparse `edge_indices` and `gather_edges()` APIs for constructing
  edge-aligned parameters without reaching into prepared coupling internals.

### Changed

- Switched the documentation build to committed Quarto freeze artifacts.
- Refined dense and sparse random graph generation, including explicit density
  semantics and low-density edge handling.
- Reworked instantaneous and delayed coupling around aligned node/edge message
  passing. Sparse local, nonlinear, edge-parameter, and delayed transforms now
  execute in O(E), and custom `pre()` implementations use one elementwise
  contract across dense and sparse graphs.
- Deprecated `FastLinearCoupling`; `LinearCoupling` now uses the same optimized
  incoming-only path, while the old class remains as a compatibility wrapper.

### Fixed

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
