<p align="center">
  <img src="https://raw.githubusercontent.com/virtual-twin/tvboptim/main/docs/images/tvboptim.png" width="60%">
</p>

# TVB-Optim

[![Tests](https://github.com/virtual-twin/tvboptim/actions/workflows/python-package.yml/badge.svg)](https://github.com/virtual-twin/tvboptim/actions/workflows/python-package.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
<!-- [![codecov](https://codecov.io/gh/virtual-twin/tvboptim/branch/main/graph/badge.svg)](https://codecov.io/gh/virtual-twin/tvboptim) -->

[JAX](https://jax.readthedocs.io/en/latest/)-based framework for brain network simulation and gradient-based optimization.

**[Documentation](https://virtual-twin.github.io/tvboptim)** | **[Get Started](https://virtual-twin.github.io/tvboptim/get_started.html)** | **[Examples](https://virtual-twin.github.io/tvboptim/network_dynamics/network_dynamics.html)**

## Key Features

- **Gradient-based optimization** - Fit thousands of parameters using automatic differentiation through the entire simulation pipeline
- **Performance** - JAX-powered with seamless GPU/TPU scaling
- **Flexible & extensible** - Build models with [Network Dynamics](https://virtual-twin.github.io/tvboptim/network_dynamics/network_dynamics.html), a composable framework for whole-brain modeling. Existing TVB workflows supported via [TVB-O](https://github.com/virtual-twin/tvbo).
- **Intuitive parameter control** - Mark values for optimization with [Parameter()](https://virtual-twin.github.io/tvboptim/parameters_and_optimization.html). Define exploration spaces with [Axes](https://virtual-twin.github.io/tvboptim/axes_and_spaces.html) for automatic parallel evaluation via JAX vmap/pmap.

## Installation

**Requires Python 3.11 or above**

```bash
# Using uv (recommended)
uv pip install tvboptim

# Using pip
pip install tvboptim
```

## Quick Example

```python
import jax.numpy as jnp
from tvboptim.experimental.network_dynamics import Network, solve, prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
from tvboptim.observations.tvb_monitors import Bold
from tvboptim.observations import compute_fc, rmse
from tvboptim.optim import OptaxOptimizer
import optax

# Build brain network model
network = Network(
    dynamics=ReducedWongWang(),
    coupling={'delayed': LinearCoupling(incoming_states="S", G=0.5)},
    graph=DenseDelayGraph(weights, delays)
)

# Run simulation
result = solve(network, Heun(), t0=0.0, t1=60_000.0, dt=1.0)

# Optimize coupling strength to match empirical functional connectivity
simulator, params = prepare(network, Heun(), t0=0.0, t1=60_000.0, dt=1.0)
bold_monitor = Bold(history=result, period=720.0)

def loss(params):
    predicted_fc = compute_fc(bold_monitor(simulator(params)))
    return rmse(predicted_fc, target_fc)

opt = OptaxOptimizer(loss, optax.adam(learning_rate=0.03))
final_params, history = opt.run(params, max_steps=50)
```

See the [full example with visualization](https://virtual-twin.github.io/tvboptim/) in the documentation.

## Documentation

- **[Get Started](https://virtual-twin.github.io/tvboptim/get_started.html)** - Introduction and basic workflows
- **[Network Dynamics](https://virtual-twin.github.io/tvboptim/network_dynamics/network_dynamics.html)** - Build differentiable brain network models
- **[Parameters & Optimization](https://virtual-twin.github.io/tvboptim/parameters_and_optimization.html)** - Gradient-based parameter inference
- **[API Reference](https://virtual-twin.github.io/tvboptim/reference/index.html)** - Complete API documentation

## How It Works

**Functional Composition** - Models are composable functions that can be inspected, modified, and extended. Build complexity incrementally by wrapping functions.

**End-to-End Differentiability** - Compute gradients through the full pipeline: neural dynamics → hemodynamic models → empirical observables.

**Interoperability** - Convert existing TVB simulations to JAX-compatible functions via the TVB-O framework.

## Contributing

We welcome contributions and questions from the community!

- **Report Issues**: [Open an issue](https://github.com/virtual-twin/tvboptim/issues)
- **Ask Questions**: [Start a discussion](https://github.com/virtual-twin/tvboptim/discussions)
- **Contribute Code**: [Open a pull request](https://github.com/virtual-twin/tvboptim/pulls)

## Contact

Questions or want to collaborate? Reach out:

* marius.pille[at]bih-charite.de

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

Copyright © 2025 Charité Universitätsmedizin Berlin