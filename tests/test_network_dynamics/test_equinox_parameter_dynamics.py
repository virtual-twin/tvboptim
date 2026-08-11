"""End-to-end dynamics proof for an EquinoxParameter learned term."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from tvboptim.experimental.network_dynamics import (
    Bunch,
    DenseGraph,
    Network,
    prepare,
)
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.solvers import Euler
from tvboptim.optim import OptaxOptimizer
from tvboptim.types import EquinoxParameter, combine_state, partition_state


class _LearnedDrift(AbstractDynamics):
    """One mechanistic state plus a node-local learned correction."""

    STATE_NAMES = ("x",)
    INITIAL_STATE = (0.2,)
    DEFAULT_PARAMS = Bunch(decay=0.4, correction=None)
    COUPLING_INPUTS = {"drive": 1}

    def dynamics(self, t, state, params, coupling, external):
        del t, external
        x = state[0]
        inputs = jnp.stack((x, coupling.drive[0]), axis=-1)
        learned_term = jax.vmap(params.correction)(inputs)[:, 0]
        base_drift = -params.decay * x + coupling.drive[0]
        return jnp.stack((base_drift + learned_term,))


def _correction(weight, bias, *, key):
    module = eqx.nn.MLP(
        in_size=2,
        out_size=1,
        width_size=2,
        depth=0,
        activation=jax.nn.tanh,
        key=jax.random.key(key),
    )
    module = eqx.tree_at(
        lambda model: (model.layers[0].weight, model.layers[0].bias),
        module,
        (jnp.asarray(weight), jnp.asarray([bias])),
    )
    return EquinoxParameter(module)


def _setup(correction):
    dynamics = _LearnedDrift(correction=correction)
    network = Network(
        dynamics=dynamics,
        coupling={"drive": LinearCoupling(source="x", G=0.15)},
        graph=DenseGraph(jnp.asarray([[0.0, 0.5], [0.25, 0.0]])),
    )
    solve_fn, config = prepare(network, Euler(), t0=0.0, t1=0.5, dt=0.05)
    config.initial_state.dynamics = jnp.asarray([[0.2, -0.1]])
    return dynamics, solve_fn, config


def test_network_rollout_works_eagerly_and_through_filter_jit():
    dynamics, solve_fn, config = _setup(_correction([[-0.2, 0.1]], 0.0, key=0))

    assert dynamics.verify(n_nodes=2)
    eager = solve_fn(config)
    compiled = eqx.filter_jit(solve_fn)(config)

    assert eager.variable_names == ("x",)
    assert eager.ys.shape == (10, 1, 2)
    assert jnp.all(jnp.isfinite(eager.ys))
    assert jnp.allclose(compiled.ys, eager.ys)


def test_rollout_gradients_reach_equinox_arrays():
    _, solve_fn, config = _setup(_correction([[-0.2, 0.1]], 0.0, key=0))
    diff_config, static_config = partition_state(config)

    def loss(diff):
        solution = solve_fn(combine_state(diff, static_config))
        return jnp.mean(solution.ys**2)

    grads = jax.grad(loss)(diff_config)
    grad_leaves = jax.tree.leaves(grads)

    assert len(grad_leaves) == 2
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)
    assert all(jnp.any(leaf != 0.0) for leaf in grad_leaves)


def test_short_optimization_reduces_deterministic_rollout_loss():
    _, solve_fn, config = _setup(_correction([[-0.2, 0.1]], 0.0, key=0))
    target_config = config.copy()
    target_config.dynamics.correction = _correction([[0.35, -0.25]], 0.08, key=1)
    target = solve_fn(target_config).ys

    def loss(state):
        return jnp.mean((solve_fn(state).ys - target) ** 2)

    initial_loss = loss(config)
    fitted, _ = OptaxOptimizer(loss, optax.adam(0.03)).run(
        config, max_steps=30, chunk_size=5
    )
    final_loss = loss(fitted)

    assert jnp.isfinite(final_loss)
    assert final_loss < initial_loss * 0.1
