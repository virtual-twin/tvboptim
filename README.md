<p align="center">
  <img src="https://raw.githubusercontent.com/virtual-twin/tvboptim/main/docs/images/tvboptim.png" width="60%">
</p>

# TVB-Optim

<!-- [![Ruff](https://github.com/virtual-twin/tvboptim/actions/workflows/ruff.yml/badge.svg)](https://github.com/virtual-twin/tvboptim/actions/workflows/ruff.yml) -->
[![Tests](https://github.com/virtual-twin/tvboptim/actions/workflows/python-package.yml/badge.svg)](https://github.com/virtual-twin/tvboptim/actions/workflows/python-package.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/tvboptim.svg)](https://pypi.org/project/tvboptim/)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://virtual-twin.github.io/tvboptim)
<!-- [![Downloads](https://img.shields.io/pypi/dm/tvboptim.svg)](https://pypi.org/project/tvboptim/) -->
<!-- [![codecov](https://codecov.io/gh/virtual-twin/tvboptim/branch/main/graph/badge.svg)](https://codecov.io/gh/virtual-twin/tvboptim) -->

**Fast, differentiable, and parallel whole-brain simulation and inference in
[JAX](https://jax.readthedocs.io/en/latest/).**

Use the same brain network model code on CPUs and GPUs. Evaluate parameter
ensembles in parallel and choose the inference strategy that fits your problem,
from end-to-end automatic differentiation to simulation-based and evolutionary
methods.

## Why TVB-Optim?

- **Accelerated simulation:** JIT-compile brain network models and run them on
  CPUs and GPUs.
- **Parallel inference:** Batch parameter sets, stochastic realizations, and
  candidate populations with `vmap`, then distribute them across devices with
  `pmap`.
- **End-to-end automatic differentiation:** Differentiate through dynamics,
  coupling, transmission delays, observation models, and summary statistics.
- **Inference-method agnostic:** Use the same simulator for gradient descent,
  Bayesian and simulation-based inference, genetic algorithms, parameter
  sweeps, and ensemble analysis.
- **Composable whole-brain models:** Combine neural dynamics, connectivity,
  delays, coupling, noise, external inputs, solvers, and observation models
  with [Network Dynamics](https://virtual-twin.github.io/tvboptim/network_dynamics/network_dynamics.html).
- **Structured parameter exploration:** Mark trainable values with
  [`Parameter`](https://virtual-twin.github.io/tvboptim/basics/parameters_and_optimization.html)
  and express Cartesian, zipped, grouped, or sampled parameter spaces with
  [Axes and Spaces](https://virtual-twin.github.io/tvboptim/basics/axes_and_spaces.html).
  Existing TVB workflows are supported through
  [TVB-O](https://github.com/virtual-twin/tvbo).

## Installation

**Requires Python 3.11 or above**

```bash
# Using uv (recommended)
uv pip install tvboptim

# Using pip
pip install tvboptim
```

## Quick Example

Fit the coupling strength of an 84-region whole-brain model to empirical fMRI
functional connectivity:

<details>
<summary>Imports (expand to run)</summary>

```python
import jax
import jax.numpy as jnp
import optax

from tvboptim.data import load_functional_connectivity, load_structural_connectivity
from tvboptim.experimental.network_dynamics import Network, prepare, solve
from tvboptim.experimental.network_dynamics.coupling import DelayedLinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations import compute_fc, rmse
from tvboptim.observations.tvb_monitors import Bold
from tvboptim.optim import OptaxOptimizer
from tvboptim.types import Parameter
```

</details>

```python
# Bundled structural and functional connectivity for 84 brain regions.
weights, lengths, labels = load_structural_connectivity("dk_average")
weights = weights / jnp.max(weights)
target_fc = load_functional_connectivity("dk_average")

# Build a delayed whole-brain network from the structural connectome.
network = Network(
    dynamics=ReducedWongWang(),
    coupling={"delayed": DelayedLinearCoupling(incoming_states="S", G=0.5)},
    graph=DenseDelayGraph(
        weights=weights,
        delays=lengths / 3.0,
        region_labels=labels,
    ),
    noise=AdditiveNoise(sigma=0.01, key=jax.random.key(42)),
)

# Simulate once to initialize the delay history.
solver = BoundedSolver(Heun(), low=0.0, high=1.0)
result = solve(network, solver, t0=0.0, t1=60_000.0, dt=1.0)
network.update_history(result)

# Prepare a differentiable simulator and convert neural activity to BOLD.
simulator, params = prepare(network, solver, t0=0.0, t1=60_000.0, dt=1.0)
params.coupling.delayed.G = Parameter(0.5)
bold = Bold(history=result, period=720.0)

# Compare simulated and empirical functional connectivity.
def loss(params):
    predicted_fc = compute_fc(bold(simulator(params)))
    return rmse(predicted_fc, target_fc)

# Fit the global coupling strength with Adam.
optimizer = OptaxOptimizer(loss, optax.adam(learning_rate=0.03))
fitted_params, history = optimizer.run(params, max_steps=5)
```

For a complete 84-region model fitting empirical fMRI functional connectivity,
see the [whole-brain optimization workflow](https://virtual-twin.github.io/tvboptim/workflows/RWW.html)
or [run it in Google Colab](https://colab.research.google.com/github/virtual-twin/tvboptim/blob/main/docs/workflows/RWW.ipynb).

## [Documentation](https://virtual-twin.github.io/tvboptim)

- **[Get Started](https://virtual-twin.github.io/tvboptim/basics/get_started.html):** Build and simulate your first model
- **[Network Dynamics](https://virtual-twin.github.io/tvboptim/network_dynamics/network_dynamics.html):** Compose differentiable whole-brain models
- **[Parameters & Optimization](https://virtual-twin.github.io/tvboptim/basics/parameters_and_optimization.html):** Define trainable parameters and fit models with gradients
- **[Axes & Spaces](https://virtual-twin.github.io/tvboptim/basics/axes_and_spaces.html):** Explore and evaluate parameter spaces in parallel
- **[Bayesian Inference](https://virtual-twin.github.io/tvboptim/workflows/Stimulation_with_Bayesian_Inference.html):** Connect forward simulation to NumPyro
- **[Genetic + Gradient Optimization](https://virtual-twin.github.io/tvboptim/workflows/Hopf_Pareto_ParallelOpt.html):** Combine NSGA-II pre-search with parallel gradient refinement
- **[Differentiable Delays](https://virtual-twin.github.io/tvboptim/workflows/Delay_Speed_Synchronization.html):** Sweep and fit delays and conduction speed
- **[API Reference](https://virtual-twin.github.io/tvboptim/reference/index.html):** Complete API documentation

## Contributing

We welcome contributions and questions from the community!

- **Report Issues**: [Open an issue](https://github.com/virtual-twin/tvboptim/issues)
- **Ask Questions**: [Start a discussion](https://github.com/virtual-twin/tvboptim/discussions)
- **Contribute Code**: [Open a pull request](https://github.com/virtual-twin/tvboptim/pulls)

## Citation

If you use TVB-Optim in your research, please cite:

```bibtex
@article{2025tvboptim,
  title={Fast and Easy Whole-Brain Network Model Parameter Estimation with Automatic Differentiation},
  author={Pille, Marius and Martin, Leon and Richter, Emilius and Perdikis, Dionysios and Schirner, Michael and Ritter, Petra},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.11.18.689003}
}
```

Copyright © 2026 Charité Universitätsmedizin Berlin
