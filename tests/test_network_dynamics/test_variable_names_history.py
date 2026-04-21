"""Tests for variable_names metadata on Solutions and update_history slicing."""

import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tvboptim.experimental.network_dynamics import Network, solve
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.dynamics.tvb import JansenRit
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.solvers import Heun


def _make_network(voi=None):
    n_nodes = 3
    key = jax.random.PRNGKey(0)
    graph = DenseGraph.random(n_nodes=n_nodes, key=key)
    dynamics = JansenRit()
    if voi is not None:
        dynamics.set_variables_of_interest(voi)
    coupling = LinearCoupling(incoming_states="y1", G=0.05)
    return Network(
        dynamics=dynamics,
        coupling={"instant": coupling},
        graph=graph,
    )


class TestVariableNamesOnSolution(unittest.TestCase):
    def test_solution_carries_variable_names(self):
        network = _make_network()
        result = solve(network, Heun(), t0=0.0, t1=5.0, dt=0.1)
        # Default VOI records only state variables
        self.assertEqual(result.variable_names, tuple(network.dynamics.STATE_NAMES))
        self.assertEqual(result.ys.shape[1], len(network.dynamics.STATE_NAMES))

    def test_repr_mentions_variable_names(self):
        network = _make_network()
        result = solve(network, Heun(), t0=0.0, t1=2.0, dt=0.1)
        self.assertIn("variable_names", repr(result))

    def test_voi_including_auxiliary(self):
        aux = JansenRit.AUXILIARY_NAMES[0]
        network = _make_network(voi=JansenRit.STATE_NAMES + (aux,))
        result = solve(network, Heun(), t0=0.0, t1=5.0, dt=0.1)
        expected = tuple(network.dynamics.STATE_NAMES) + (aux,)
        self.assertEqual(result.variable_names, expected)
        self.assertEqual(result.ys.shape[1], len(expected))


class TestUpdateHistoryWithVariableNames(unittest.TestCase):
    def test_roundtrip_with_auxiliary_in_voi(self):
        aux = JansenRit.AUXILIARY_NAMES[0]
        network = _make_network(voi=JansenRit.STATE_NAMES + (aux,))
        result = solve(network, Heun(), t0=0.0, t1=5.0, dt=0.1)

        # Previously raised due to 7 vs 6 state mismatch.
        network.update_history(result)

        n_states = len(network.dynamics.STATE_NAMES)
        self.assertEqual(network._history.ys.shape[1], n_states)
        self.assertEqual(
            network._history.variable_names,
            tuple(network.dynamics.STATE_NAMES),
        )

        # initial_state now reflects the sliced history
        self.assertEqual(
            network.initial_state.shape,
            (n_states, network.graph.n_nodes),
        )

    def test_missing_state_raises(self):
        # Record only a subset of states → missing "y0"
        voi = ("y1", "y2", "y3", "y4", "y5")
        network = _make_network(voi=voi)
        result = solve(network, Heun(), t0=0.0, t1=2.0, dt=0.1)

        with self.assertRaises(ValueError) as cm:
            network.update_history(result)
        msg = str(cm.exception)
        self.assertIn("y0", msg)
        self.assertIn("VARIABLES_OF_INTEREST", msg)

    def test_solution_without_variable_names_raises(self):
        network = _make_network()
        n_states = len(network.dynamics.STATE_NAMES)
        n_nodes = network.graph.n_nodes

        ts = jnp.linspace(0.0, 1.0, 10)
        ys = jnp.zeros((10, n_states, n_nodes))
        unlabeled = NativeSolution(ts=ts, ys=ys, dt=0.1, variable_names=None)

        with self.assertRaises(ValueError) as cm:
            network.update_history(unlabeled)
        self.assertIn("variable_names", str(cm.exception))


class TestSetVariablesOfInterest(unittest.TestCase):
    def test_valid_update(self):
        dynamics = JansenRit()
        dynamics.set_variables_of_interest(("y0", "y1"))
        self.assertEqual(dynamics.VARIABLES_OF_INTEREST, ("y0", "y1"))

    def test_invalid_name_rolls_back(self):
        dynamics = JansenRit()
        original = dynamics.VARIABLES_OF_INTEREST
        with self.assertRaises(ValueError):
            dynamics.set_variables_of_interest(("not_a_var",))
        self.assertEqual(dynamics.VARIABLES_OF_INTEREST, original)


if __name__ == "__main__":
    unittest.main()
