import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.execution import ParallelExecution, SequentialExecution
from tvboptim.experimental.network_dynamics import (
    Bunch,
    GroupObservation,
    HeterogeneousNetwork,
    Network,
    NodeGroup,
    SignalRoute,
    prepare,
    solve,
)
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedLinearCoupling,
    InstantaneousCoupling,
    LinearCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics import AbstractDynamics
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    JansenRit,
    Linear,
    ReducedWongWang,
)
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import Euler, Heun
from tvboptim.types import GridAxis, Space

ROOT = Path(__file__).parents[1]
SKILL_DIR = ROOT / "src" / "tvboptim" / "skills" / "tvboptim"
SKILL_MD = SKILL_DIR / "SKILL.md"


def _frontmatter(text):
    assert text.startswith("---\n")
    raw, _body = text[4:].split("\n---\n", 1)
    entries = {}
    for line in raw.splitlines():
        key, value = line.split(":", 1)
        entries[key.strip()] = value.strip()
    return entries


def test_skill_structure_and_portability():
    text = SKILL_MD.read_text()
    metadata = _frontmatter(text)

    assert metadata.keys() == {"name", "description"}
    assert metadata["name"] == SKILL_DIR.name == "tvboptim"
    assert "Use when" in metadata["description"]
    assert "Do not use" in metadata["description"]
    assert "TODO" not in text
    assert "Do not silently collect or upload telemetry" in text
    assert "without the user's explicit approval" in text
    assert "minimal reproduction" in text
    assert "generic new dynamics model" in text

    reference_links = set(re.findall(r"\]\((references/[^)#]+\.md)\)", text))
    expected = {
        f"references/{path.name}" for path in (SKILL_DIR / "references").glob("*.md")
    }
    assert reference_links == expected
    assert "references/heterogeneous-networks.md" in reference_links
    assert all((SKILL_DIR / link).is_file() for link in reference_links)

    all_text = "\n".join(path.read_text() for path in SKILL_DIR.rglob("*.md"))
    assert not re.search(r"\bBold\(", all_text)
    assert "LotkaVolterraHRFKernel" not in all_text
    assert "FastLinearCoupling" not in all_text
    assert "config.dynamics.G" not in all_text
    assert "config.integrator.noise.nsig" not in all_text
    assert not re.search(r"/tvboptim-[a-z]", all_text)
    assert not re.search(r"\b(?:Claude|Codex|Cursor)\b", all_text)

    # Examples must teach the 0.5 coupling selectors. The deprecated keywords may
    # still be named in prose that documents their removal, but never used.
    examples = "\n".join(re.findall(r"```python\n(.*?)```", all_text, re.DOTALL))
    assert not re.search(r"\b(?:incoming_states|local_states)\s*=", examples)
    assert "source=" in examples
    assert "local=" in examples


def test_canonical_delayed_network_pattern_runs():
    weights = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    delays = jnp.array([[0.0, 2.0], [2.0, 0.0]])
    network = Network(
        dynamics=ReducedWongWang(w=0.7, INITIAL_STATE=(0.1,)),
        coupling={"delayed": DelayedLinearCoupling(source="S", G=0.1)},
        graph=DenseDelayGraph(weights=weights, delays=delays),
        noise=AdditiveNoise(sigma=1e-5, key=jax.random.key(0)),
    )

    direct = solve(network, Heun(), t0=0.0, t1=4.0, dt=0.1)
    solve_fn, config = prepare(network, Heun(), t0=0.0, t1=4.0, dt=0.1)
    eager = solve_fn(config)
    compiled = jax.jit(solve_fn)(config)

    assert direct.ys.shape == (40, 1, 2)
    assert direct.variable_names == ("S",)
    assert jnp.isfinite(direct.ys).all()
    assert config.coupling.delayed.G == 0.1
    np.testing.assert_allclose(eager.ys, compiled.ys, rtol=1e-5, atol=1e-6)


class _FitzHughNagumo(AbstractDynamics):
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


class _AdaptiveGainCoupling(InstantaneousCoupling):
    N_OUTPUT_STATES = 1
    DEFAULT_PARAMS = Bunch(G=1.0, alpha=0.5)

    def post(self, summed_inputs, local_states, params):
        activity = jnp.abs(local_states[0])
        return params.G * (1.0 - params.alpha * activity) * summed_inputs


def test_custom_extension_pattern_runs():
    dynamics = _FitzHughNagumo(VARIABLES_OF_INTEREST=("V", "W", "I_mem"))
    assert dynamics.verify(n_nodes=2, verbose=False)

    bare = solve(dynamics, Euler(), t0=0.0, t1=2.0, dt=0.1, n_nodes=2)
    assert bare.ys.shape == (20, 3, 2)
    assert bare.variable_names == ("V", "W", "I_mem")

    network = Network(
        dynamics=dynamics,
        coupling={
            "structural": _AdaptiveGainCoupling(
                source="V",
                local="V",
                G=0.05,
                alpha=0.1,
            )
        },
        graph=DenseGraph(jnp.array([[0.0, 1.0], [1.0, 0.0]])),
    )
    solve_fn, config = prepare(network, Heun(), t0=0.0, t1=2.0, dt=0.1)
    coupled = solve_fn(config)
    compiled = jax.jit(solve_fn)(config)

    assert coupled.ys.shape == (20, 3, 2)
    assert jnp.isfinite(coupled.ys).all()
    np.testing.assert_allclose(coupled.ys, compiled.ys, rtol=1e-5, atol=1e-6)


def _heterogeneous_network():
    weights = jnp.array(
        [
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    return HeterogeneousNetwork(
        graph=DenseGraph(weights),
        groups={
            "cortex": NodeGroup(JansenRit(), nodes=[0, 1, 2]),
            "sub": NodeGroup(
                Linear(gamma=-0.2),
                nodes=[3, 4, 5],
                noise=AdditiveNoise(sigma=1e-3, key=jax.random.key(0)),
            ),
        },
        routes={
            "excitatory": SignalRoute(
                source={
                    "cortex": lambda state, params: state[1:2] - state[2:3],
                    "sub": "x",
                },
                coupling=LinearCoupling(G=0.1),
                target={"cortex": "instant", "sub": "instant"},
            )
        },
    )


def test_canonical_heterogeneous_pattern_runs():
    network = _heterogeneous_network()
    solve_fn, config = prepare(network, Heun(), t0=0.0, t1=20.0, dt=0.1)
    result = solve_fn(config)

    assert network.group_names == ("cortex", "sub")
    assert network.group_nodes == {"cortex": (0, 1, 2), "sub": (3, 4, 5)}
    assert result.ys["cortex"].shape == (200, 6, 3)
    assert result.ys["sub"].shape == (200, 1, 3)
    assert result.variable_names["sub"] == ("x",)
    assert config.routes.excitatory.coupling.G == 0.1
    assert config.groups.sub.noise.sigma == 1e-3
    assert jnp.isfinite(result.ys["cortex"]).all()
    assert result.to_graph("x", groups=["sub"]).shape == (200, 6)

    compiled = jax.jit(solve_fn)(config)
    np.testing.assert_allclose(
        result.ys["cortex"], compiled.ys["cortex"], rtol=1e-5, atol=1e-6
    )


def test_canonical_group_observation_pattern_runs():
    observe = GroupObservation(
        {"cortex": lambda voi, params: voi[1:2] - voi[2:3], "sub": "x"},
        channels=("activity",),
    )
    solve_fn, config = prepare(
        _heterogeneous_network(), Heun(), t0=0.0, t1=20.0, dt=0.1, observe=observe
    )
    observed = solve_fn(config)

    # observe= returns one graph-order solution, not per-group trajectories.
    assert observed.ys.shape == (200, 1, 6)
    assert observed.variable_names == ("activity",)
    assert jnp.isfinite(observed.ys).all()


def test_non_divisible_parallel_space_pattern_runs_on_one_device():
    state = Bunch(
        a=GridAxis(0.0, 1.0, 3),
        b=GridAxis(0.0, 1.0, 5),
    )
    space = Space(state, mode="product", key=jax.random.key(0))

    def objective(config):
        return config.a + 2.0 * config.b

    sequential = SequentialExecution(objective, space).run()
    parallel = ParallelExecution(
        objective,
        space,
        n_pmap=1,
        n_vmap=4,
    ).run()

    assert len(space) == len(sequential) == len(parallel) == 15
    np.testing.assert_allclose(
        np.asarray(list(sequential)),
        np.asarray(list(parallel)),
    )
