# Custom dynamics and coupling

## Contents

- [Choose the extension point](#choose-the-extension-point)
- [Implement dynamics](#implement-dynamics)
- [Implement coupling](#implement-coupling)
- [Integrate and verify the extension](#integrate-and-verify-the-extension)
- [Avoid common extension errors](#avoid-common-extension-errors)

## Choose the extension point

Subclass `AbstractDynamics` for new local differential equations. Subclass `InstantaneousCoupling` or `DelayedCoupling` when the standard state-selection, aggregation, and history machinery applies and only `pre` or `post` changes. Subclass `AbstractCoupling` only for a computation that does not fit the pre-aggregate-post pattern.

Start by searching current built-ins and tests for the closest state count, auxiliary behavior, coupling mode, or delay behavior. Reuse the public contract rather than copying internal scan code.

## Implement dynamics

Declare the interface at class level:

- `STATE_NAMES` and equal-length `INITIAL_STATE` for integrated variables;
- `DEFAULT_PARAMS` as a `Bunch`;
- optional `AUXILIARY_NAMES` for derived, non-integrated outputs;
- `COUPLING_INPUTS = {name: dimension}`;
- optional `EXTERNAL_INPUTS = {name: dimension}`;
- optional `VARIABLES_OF_INTEREST`, empty by default to record integrated states only.

Implement `dynamics(t, state, params, coupling, external)`. Inputs have shapes `[n_states, n_nodes]` or `[declared_dimension, n_nodes]`. Return derivatives with `[n_states, n_nodes]`; if auxiliaries are declared, return `(derivatives, auxiliaries)`.

This compact FitzHugh-Nagumo implementation exercises the full contract:

```python
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics import Bunch
from tvboptim.experimental.network_dynamics.dynamics import AbstractDynamics


class FitzHughNagumo(AbstractDynamics):
    STATE_NAMES = ("V", "W")
    INITIAL_STATE = (-1.2, -0.62)
    AUXILIARY_NAMES = ("I_mem",)
    DEFAULT_PARAMS = Bunch(a=0.7, b=0.8, tau=12.5, I=0.3)
    COUPLING_INPUTS = {"structural": 1}

    def dynamics(self, t, state, params, coupling, external):
        V, W = state
        intrinsic = V - V**3 / 3.0 - W
        dV = intrinsic + params.I + coupling.structural[0]
        dW = (V + params.a - params.b * W) / params.tau
        return jnp.stack((dV, dW)), jnp.stack((intrinsic,))


model = FitzHughNagumo(VARIABLES_OF_INTEREST=("V", "W", "I_mem"))
assert model.verify(n_nodes=2)
```

Declared but unsupplied coupling and external inputs are zero-filled by the solve/network layer. Passing an unknown parameter to the constructor, or an unknown coupling key to `Network`, is an error. Do not hide misspellings with permissive dictionaries.

Use `model.verify()` as an interface check, then simulate bare dynamics to test numerical behavior without graph complexity:

```python
from tvboptim.experimental.network_dynamics import solve
from tvboptim.experimental.network_dynamics.solvers import Euler

result = solve(model, Euler(), t0=0.0, t1=10.0, dt=0.1, n_nodes=2)
assert result.ys.shape == (100, 3, 2)
assert result.variable_names == ("V", "W", "I_mem")
```

## Implement coupling

Standard couplings perform:

```text
selected incoming/local states -> pre -> weighted aggregation -> post
```

Returning per-edge values from `pre` gives `[inputs, targets, sources]` and allows edge-dependent transforms. Returning local node values gives `[inputs, nodes]` and selects the vectorized dense path. Confirm graph orientation from current code/tests before writing asymmetric connectivity logic.

For many custom instantaneous couplings, implement only `post`:

```python
from tvboptim.experimental.network_dynamics import Bunch
from tvboptim.experimental.network_dynamics.coupling import InstantaneousCoupling


class AdaptiveGainCoupling(InstantaneousCoupling):
    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0, alpha=0.5)

    def post(self, summed_inputs, local_states, params):
        activity = jnp.abs(local_states[0])
        gain = params.G * (1.0 - params.alpha * activity)
        return gain * summed_inputs
```

Instantiate it with both state sources required by the implementation:

```python
coupling = AdaptiveGainCoupling(
    source="V",
    local="V",
    G=0.2,
    alpha=0.5,
)
```

Selector keywords are `source=` for the states carried across edges and `local=` for the target-local states `post` receives. The resolved values are exposed as `SOURCE_STATE_NAMES` and `LOCAL_STATE_NAMES`. The older `incoming_states=` and `local_states=` keywords still work, emit a `DeprecationWarning`, and are removed in 1.0; supplying a name together with its alias is an error. Do not rename the `local_states` parameter in the `post` signature, which is unrelated to the constructor keyword.

Use `precompute(coupling_data, params, graph)` for parameter-dependent quantities that should be computed once per forward call while preserving gradient flow. Do not precompute traced parameter combinations during Python construction.

Delayed subclasses receive delayed incoming state through the base machinery and manage their history via `prepare` and `update_state`. Use a delayed base class rather than recreating delay indexing inside `post`.

## Integrate and verify the extension

Test progressively:

1. Call `dynamics.verify(n_nodes=2)`.
2. Simulate the dynamics without a network and assert ranks, names, finite values, and qualitative behavior.
3. Build a two-node network with an explicit graph and weak coupling.
4. Compare zero coupling with nonzero coupling and assert the trajectory changes.
5. Compare eager execution with `jax.jit` on the prepared solve function.
6. Differentiate a scalar summary with respect to a real dynamics or coupling parameter and assert finite gradients.
7. Add delays, noise, large graphs, or auxiliary recording one feature at a time.

When the dynamics declares several coupling channels, provide a dictionary keyed by those declarations. It is valid to omit a declared channel only when a zero input is intentional.

## Avoid common extension errors

- Returning Python lists instead of stacked JAX arrays.
- Dropping the node dimension for single-node tests.
- Declaring auxiliaries but returning the wrong auxiliary count.
- Saving an auxiliary without adding it to `VARIABLES_OF_INTEREST`.
- Reading a coupling/external name not declared by the class.
- Selecting `source` when `post` needs `local` state, or vice versa.
- Writing the deprecated `incoming_states=`/`local_states=` keywords into new code.
- Treating an instantaneous class as delayed because of a dictionary label.
- Converting traced values to NumPy or Python scalars.
- Testing only one node, which can conceal broadcasting errors.
- Claiming model validity because `verify` passes; it checks the software contract, not scientific correctness.
