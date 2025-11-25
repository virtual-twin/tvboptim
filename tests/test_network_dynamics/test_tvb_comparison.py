"""Test TVB vs TVBOptim implementation comparison across all models.

This test suite validates that TVBOptim implementations match TVB reference
implementations for all neural mass models with both instant and delayed coupling.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 for better numerical precision
jax.config.update("jax_enable_x64", True)

# TVB imports
from tvb.simulator.lab import (
    connectivity,
    coupling,
    integrators,
    models,
    monitors,
    simulator,
)

# TVBOptim imports
from tvboptim.data import load_structural_connectivity
from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.coupling import (
    DelayedLinearCoupling,
    LinearCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.tvb import (
    CoombesByrne2D,
    Epileptor,
    Generic2dOscillator,
    JansenRit,
    Kuramoto,
    LarterBreakspear,
    Linear,
    MontbrioPazoRoxin,
    ReducedWongWang,
    SupHopf,
    WilsonCowan,
    WongWangExcInh,
)
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.solvers import Heun

# ==============================================================================
# MODEL REGISTRY
# ==============================================================================
MODEL_REGISTRY = {
    "Linear": {
        "tvb_class": models.Linear,
        "tvboptim_class": Linear,
        "coupling_var": "x",
        "state_indices": [0],
        "tvb_cvar": [0],
        "params": {},
    },
    "Kuramoto": {
        "tvb_class": models.Kuramoto,
        "tvboptim_class": Kuramoto,
        "coupling_var": "theta",
        "state_indices": [0],
        "tvb_cvar": [0],
        "params": {},
    },
    "ReducedWongWang": {
        "tvb_class": models.ReducedWongWang,
        "tvboptim_class": ReducedWongWang,
        "coupling_var": "S",
        "state_indices": [0],
        "tvb_cvar": [0],
        "params": {},
    },
    "Generic2dOscillator": {
        "tvb_class": models.Generic2dOscillator,
        "tvboptim_class": Generic2dOscillator,
        "coupling_var": "V",
        "state_indices": [0, 1],
        "tvb_cvar": [0],
        "params": {
            "a": np.array([-1.5]),
            "tau": np.array([4.0]),
            "b": np.array([-15.0]),
            "c": np.array([0.0]),
            "d": np.array([0.015]),
            "e": np.array([3.0]),
            "f": np.array([1.0]),
            "g": np.array([0.0]),
            "I": np.array([1.9]),
        },
    },
    "MontbrioPazoRoxin": {
        "tvb_class": models.MontbrioPazoRoxin,
        "tvboptim_class": MontbrioPazoRoxin,
        "coupling_var": "r",
        "state_indices": [0, 1],
        "tvb_cvar": [0, 1],
        "params": {},
    },
    "SupHopf": {
        "tvb_class": models.SupHopf,
        "tvboptim_class": SupHopf,
        "coupling_var": "x",
        "state_indices": [0, 1],
        "tvb_cvar": [0, 1],
        "params": {},
    },
    "CoombesByrne2D": {
        "tvb_class": models.CoombesByrne2D,
        "tvboptim_class": CoombesByrne2D,
        "coupling_var": "r",
        "state_indices": [0, 1],
        "tvb_cvar": [0, 1],
        "params": {},
    },
    "WilsonCowan": {
        "tvb_class": models.WilsonCowan,
        "tvboptim_class": WilsonCowan,
        "coupling_var": "E",
        "state_indices": [0, 1],
        "tvb_cvar": [0, 1],
        "params": {},
    },
    "WongWangExcInh": {
        "tvb_class": models.ReducedWongWangExcInh,
        "tvboptim_class": WongWangExcInh,
        "coupling_var": "S_e",
        "state_indices": [0, 1],
        "tvb_cvar": [0],
        "params": {},
    },
    "JansenRit": {
        "tvb_class": models.JansenRit,
        "tvboptim_class": JansenRit,
        "coupling_var": "y1",
        "state_indices": [0, 1, 2, 3, 4, 5],
        "tvb_cvar": [1, 2],
        "params": {},
    },
    "LarterBreakspear": {
        "tvb_class": models.LarterBreakspear,
        "tvboptim_class": LarterBreakspear,
        "coupling_var": "V",
        "state_indices": [0, 1, 2],
        "tvb_cvar": [0],
        "params": {},
    },
    "Epileptor": {
        "tvb_class": models.Epileptor,
        "tvboptim_class": Epileptor,
        "coupling_var": "x1",
        "state_indices": [0, 1, 2, 3, 4, 5],
        "tvb_cvar": [0, 3],
        "params": {},
    },
}


# ==============================================================================
# TEST CLASS
# ==============================================================================
class TestTVBComparison(unittest.TestCase):
    """Compare TVB and TVBOptim implementations across all models."""

    # ==================== EASY CONFIGURATION SECTION ====================

    # Simulation parameters
    SIMULATION_LENGTH = 100.0  # ms - short to avoid chaos
    DT = 0.1  # ms
    CONNECTIVITY_NAME = "dk_average"  # 84 regions - fast

    # Tolerance thresholds (EASY TO ADJUST)
    TOLERANCE = {
        "correlation_min": 0.99,  # Min correlation to pass
        "rel_rmse_max": 1.0,  # Max relative RMSE % to pass
    }

    # Coupling configurations to test
    COUPLING_CONFIGS = [
        {"name": "instant", "speed": np.inf, "G": 0.0005},
        {"name": "delayed", "speed": 3.0, "G": 0.0005},
    ]

    # Models to test (EASY TO ENABLE/DISABLE - just comment out)
    MODELS_TO_TEST = [
        "Linear",
        "Kuramoto",
        "ReducedWongWang",
        "Generic2dOscillator",
        "MontbrioPazoRoxin",
        # 'SupHopf', # Hard to find good ICs for SupHopf that are stable
        "CoombesByrne2D",
        "WilsonCowan",
        "WongWangExcInh",
        "JansenRit",
        "LarterBreakspear",
        "Epileptor",
    ]

    def setUp(self):
        """Load connectivity once for all tests."""
        # Load connectivity
        weights, lengths, labels = load_structural_connectivity(self.CONNECTIVITY_NAME)

        # Normalize weights (TVB pattern)
        self.weights = weights / jnp.max(weights)
        self.lengths = lengths
        self.region_labels = labels
        self.n_nodes = weights.shape[0]

        # Create TVB connectivity object
        self.conn = connectivity.Connectivity(
            weights=np.array(self.weights),
            tract_lengths=np.array(self.lengths),
            region_labels=np.array(self.region_labels),
            centres=np.zeros((self.n_nodes, 3)),
        )

    def test_model_comparison(self):
        """Test all model-coupling combinations."""
        for model_name in self.MODELS_TO_TEST:
            for coupling_config in self.COUPLING_CONFIGS:
                with self.subTest(model=model_name, coupling=coupling_config["name"]):
                    self._test_single_configuration(
                        model_name=model_name,
                        coupling_config=coupling_config,
                    )

    def _test_single_configuration(self, model_name, coupling_config):
        """Test a single model-coupling combination."""
        # 1. Get model configuration
        config = MODEL_REGISTRY[model_name]

        # 2. Run TVB simulation (returns data and ICs)
        tvb_result = self._run_tvb_simulation(config, coupling_config)

        # 3. Run TVBOptim simulation with matching ICs
        tvboptim_result = self._run_tvboptim_simulation(
            config,
            coupling_config,
            tvb_initial_conditions=tvb_result["ics"],
            tvb_conn=tvb_result["conn"],
        )

        # 4. Compare results
        comparison_metrics = self._compare_results(
            tvb_data=tvb_result["data"],
            tvboptim_data=tvboptim_result["data"],
            state_indices=config["state_indices"],
        )

        # 5. Assert tolerances
        self._assert_tolerances(
            metrics=comparison_metrics,
            model_name=model_name,
            coupling_name=coupling_config["name"],
        )

    def _run_tvb_simulation(self, config, coupling_config):
        """Run TVB simulation."""
        # Configure connectivity with speed
        conn = connectivity.Connectivity(
            weights=np.array(self.weights),
            tract_lengths=np.array(self.lengths),
            region_labels=np.array(self.region_labels),
            centres=np.zeros((self.n_nodes, 3)),
            speed=np.array([coupling_config["speed"]]),
        )
        conn.configure()
        conn.set_idelays(self.DT)

        # Create model with all state variables as VOI for complete comparison
        model_params = config["params"].copy()
        tvb_model = config["tvb_class"](
            **model_params, variables_of_interest=config["tvb_class"].state_variables
        )

        # Set coupling variable
        tvb_model.cvar = np.array(config["tvb_cvar"])

        # Create and configure simulator (without ICs first)
        sim = simulator.Simulator(
            model=tvb_model,
            connectivity=conn,
            coupling=coupling.Linear(a=np.array([coupling_config["G"]])),
            integrator=integrators.HeunDeterministic(dt=self.DT),
            monitors=[monitors.Raw()],
            simulation_length=self.SIMULATION_LENGTH,
        ).configure()

        # Generate initial conditions properly
        n_time, n_svar, n_node, n_mode = sim.good_history_shape
        initial_conditions = sim.model.initial_for_simulator(
            sim.integrator, (n_time, n_svar, n_node, n_mode)
        )

        # Extract ICs for TVBOptim (already in correct order)
        initial_conditions_tvboptim = jnp.array(
            initial_conditions[:, :, :, 0], dtype=jnp.float64
        )

        # Set ICs and reconfigure
        sim.initial_conditions = initial_conditions
        sim.configure()

        # Run simulation
        ((time_tvb, data_tvb),) = sim.run()

        return {
            "time": time_tvb,
            "data": data_tvb,
            "ics": initial_conditions_tvboptim,
            "conn": conn,
        }

    def _run_tvboptim_simulation(
        self, config, coupling_config, tvb_initial_conditions, tvb_conn
    ):
        """Run TVBOptim simulation with matching initial conditions from TVB."""
        # Compute delays
        speed = coupling_config["speed"]
        if np.isinf(speed):
            # Instant coupling - use DenseGraph
            graph = DenseGraph(self.weights, region_labels=self.region_labels)
            coupling_obj = LinearCoupling(
                incoming_states=config["coupling_var"], G=coupling_config["G"]
            )
            coupling_dict = {"instant": coupling_obj}
            t_offset = 0.0
        else:
            # Delayed coupling - use DenseDelayGraph
            delays = self.lengths / speed
            graph = DenseDelayGraph(
                self.weights, delays, region_labels=self.region_labels
            )
            coupling_obj = DelayedLinearCoupling(
                incoming_states=config["coupling_var"], G=coupling_config["G"]
            )
            coupling_dict = {"delayed": coupling_obj}
            t_offset = float(tvb_conn.horizon * self.DT)

        # Create dynamics
        model_params = config["params"].copy()
        dynamics = config["tvboptim_class"](
            **model_params,
        )

        # Create network
        network = Network(
            dynamics=dynamics,
            coupling=coupling_dict,
            graph=graph,
            noise=AdditiveNoise(sigma=0.0, key=jax.random.key(42)),
        )

        # Set initial conditions from TVB (already in correct order)
        t_max = tvb_conn.horizon
        ts_tvb = jnp.arange(0.0, t_max * self.DT, self.DT)
        network.update_history(NativeSolution(ts_tvb, tvb_initial_conditions))

        # Prepare and solve
        model_fn, state = prepare(
            network,
            Heun(),
            t0=0.0 + t_offset,
            t1=self.SIMULATION_LENGTH + t_offset,
            dt=self.DT,
        )
        model_fn = jax.jit(model_fn)
        result = model_fn(state)

        return {"time": np.array(result.ts), "data": np.array(result.ys)}

    def _compare_results(self, tvb_data, tvboptim_data, state_indices):
        """Compare TVB and TVBOptim results."""
        # TVB data shape: [time, state_var, regions, mode]
        # TVBOptim data shape: [time, state_var, regions]

        # Extract relevant states from TVB (remove mode dimension)
        # Use advanced indexing to select specific states
        V_tvb = tvb_data[:, :, :, 0]  # Shape: [time, all_states, regions]
        V_tvb = np.stack(
            [V_tvb[:, i, :] for i in state_indices], axis=1
        )  # [time, n_states, regions]

        # Extract relevant states from TVBOptim
        V_tvboptim = np.stack(
            [tvboptim_data[:, i, :] for i in state_indices], axis=1
        )  # [time, n_states, regions]

        # Handle length mismatch (truncate to minimum)
        min_len = min(V_tvb.shape[0], V_tvboptim.shape[0])
        V_tvb = V_tvb[:min_len]
        V_tvboptim = V_tvboptim[:min_len]

        # Compute overall metrics (flatten across all states and regions)
        overall_correlation = np.corrcoef(V_tvb.flatten(), V_tvboptim.flatten())[0, 1]

        # RMSE
        overall_rmse = np.sqrt(np.mean((V_tvb - V_tvboptim) ** 2))

        # Relative RMSE
        data_range = V_tvb.max() - V_tvb.min()
        if data_range > 0:
            overall_rel_rmse = (overall_rmse / data_range) * 100
        else:
            overall_rel_rmse = 0.0

        return {
            "overall_correlation": overall_correlation,
            "overall_rmse": overall_rmse,
            "overall_rel_rmse": overall_rel_rmse,
        }

    def _assert_tolerances(self, metrics, model_name, coupling_name):
        """Assert that metrics meet tolerance thresholds."""
        corr_min = self.TOLERANCE["correlation_min"]
        rmse_max = self.TOLERANCE["rel_rmse_max"]

        # Check for NaN values (indicates numerical issues)
        if np.isnan(metrics["overall_correlation"]):
            self.fail(
                f"{model_name} ({coupling_name}): correlation is NaN - numerical instability detected"
            )

        # Check correlation
        self.assertGreaterEqual(
            metrics["overall_correlation"],
            corr_min,
            msg=f"{model_name} ({coupling_name}): correlation {metrics['overall_correlation']:.4f} < {corr_min}",
        )

        # Check relative RMSE
        self.assertLessEqual(
            metrics["overall_rel_rmse"],
            rmse_max,
            msg=f"{model_name} ({coupling_name}): rel_rmse {metrics['overall_rel_rmse']:.2f}% > {rmse_max}%",
        )

        # Optional: Print success message
        print(
            f"✓ {model_name} ({coupling_name}): corr={metrics['overall_correlation']:.4f}, "
            f"rel_rmse={metrics['overall_rel_rmse']:.2f}%"
        )


if __name__ == "__main__":
    unittest.main()
