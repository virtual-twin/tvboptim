# %% Setup
import jax
import jax.numpy as jnp
import optax

from tvboptim.experimental.network_dynamics import Bunch  # dict with attribute access
from tvboptim.experimental.network_dynamics.dynamics import ReducedWongWang
from tvboptim.optim import (
    DefaultPrintCallback,
    MultiCallback,
    OptaxOptimizer,
    PrintParameterCallback,
    SavingLossCallback,
    SavingParametersCallback,
    StopLossCallback,
)
from tvboptim.types import (
    BoundedParameter,
    SigmoidBoundedParameter,
    collect_parameters,
)

# %% Load the model and inspect its default parameters
m = ReducedWongWang()
dfun = m.dynamics
m.DEFAULT_PARAMS

# %% Define a fixed-point loss and evaluate it at the defaults
# Fresh copy so we never mutate the shared DEFAULT_PARAMS in place.
p_init = m.DEFAULT_PARAMS.copy()


def loss_fixed_point(p, S_0=[0.8], coupling=Bunch(instant=[0.0], delayed=[0.0])):
    return jnp.abs(jnp.sum(dfun(0.0, S_0, p, coupling, coupling)[0]))


print(f"Initial loss: {loss_fixed_point(p_init):.6f}")

# %% Gradients of the loss w.r.t. all (unconstrained) parameters
gradients = jax.grad(loss_fixed_point)(p_init)
print("Gradients for all parameters:", gradients)


# %% The two views of a constrained parameter
# .value is the raw differentiable leaf the optimizer updates. For a sigmoid
# bound it lives in logit space and can fall outside [low, high]; for a hard
# bound it is the unclipped value. .constrained_value is what the model sees,
# always inside the bounds. This is the same value any jnp/np op gets.
def show_views(name, p):
    print(
        f"{name:8s} .value={float(p.value):+.4f}  .constrained_value={float(p.constrained_value):.4f}"
    )


gamma_default = m.DEFAULT_PARAMS.gamma  # 0.641
show_views("sigmoid", SigmoidBoundedParameter(gamma_default, low=0, high=1))
show_views("bounded", BoundedParameter(gamma_default, low=0, high=1))

# %% Mark gamma and tau_s for optimization with SigmoidBoundedParameter
# Wrap the raw default values directly (not an already-wrapped Parameter).
p_init.gamma = SigmoidBoundedParameter(m.DEFAULT_PARAMS.gamma, low=0, high=1)
p_init.tau_s = SigmoidBoundedParameter(m.DEFAULT_PARAMS.tau_s, low=0, high=300)

# constrained_value reproduces the originals, the leaf does not:
show_views("gamma", p_init.gamma)
show_views("tau_s", p_init.tau_s)

# %% Configure the optimizer and callbacks
cbs = MultiCallback(
    [
        DefaultPrintCallback(every=10),  # loss every 10 steps
        PrintParameterCallback(every=50),  # constrained values every 50 steps
        SavingLossCallback(),  # loss history
        SavingParametersCallback(),  # parameter history
        StopLossCallback(stop_loss=1e-6),  # stop once loss is small enough
    ]
)

opt = OptaxOptimizer(loss_fixed_point, optax.adam(0.01), callback=cbs)

# %% Run the optimization
print("Starting optimization with SigmoidBoundedParameters...")
p_opt, fitting_data = opt.run(p_init, max_steps=100)
print(f"Final loss: {fitting_data['loss'].save.values[-1]:.8f}")

# %% Inspect the result through its constrained values
# collect_parameters returns the constrained values (what the model uses),
# so gamma stays in [0, 1] and tau_s in [0, 300] regardless of the leaf.
p_opt_c = collect_parameters(p_opt)
print(f"Optimized gamma (constrained): {float(p_opt_c.gamma):.4f}")
print(f"Optimized tau_s (constrained): {float(p_opt_c.tau_s):.4f}")
