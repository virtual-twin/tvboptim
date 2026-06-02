# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

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

### Added

- `recompute_coupling_per_stage` flag on `NativeSolver` (and subclasses
  `Euler`, `Heun`, `RungeKutta4`, `BoundedSolver`).

### Removed

- Removed the unused abstract `NativeSolver` stub in
  `solvers/base.py`. The public `network_dynamics.solvers.NativeSolver` now
  refers to the concrete base class that `Euler`/`Heun`/`RungeKutta4` actually
  inherit from, so `isinstance(Heun(), NativeSolver)` is now `True`.
