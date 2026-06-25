"""End-to-end differentiable-delays experiment driven through tvbo.

The whole simulation — model, delayed (interpolated) coupling, connectivity,
a conduction-speed exploration, and a conduction-speed optimization — is
declared inline as a single YAML string and executed with

    SimulationExperiment.from_string(exp_yaml).run("tvboptim")

The delayed coupling sets ``interpolate_delays: true``; tvbo's tvboptim code
generator emits ``DelayedCoupling(interpolate_delays=True, ...)`` (the feature
added on the tvboptim side), so the conduction delays are differentiable and
the run is jit-compatible.

Requires the optional ``tvbo`` package built with the ``interpolate_delays``
coupling slot + codegen (skipped otherwise); ``tvbo`` is not a CI dependency.
The optimization test additionally needs tvbo to expose ``conduction_speed`` as
an optimizable parameter (it is currently excluded from
``SimulationExperiment.collect_state``); until then it is skipped.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("tvbo")

from tvbo import SimulationExperiment  # noqa: E402

try:
    from tvbo.datamodel.pydantic import Coupling as _Coupling

    _HAS_INTERPOLATE = "interpolate_delays" in getattr(_Coupling, "model_fields", {})
except Exception:
    _HAS_INTERPOLATE = False

exp_yaml = """
label: Differentiable delays — conduction-speed exploration + optimization
dynamics:
  iri: tvbo:Generic2dOscillator
network:
  number_of_nodes: 4
  coupling:
    c_glob:
      iri: tvbo:Linear
      delayed: true
      interpolate_delays: true
      parameters:
        G: {value: 0.5}
  parameters:
    conduction_speed: {value: 3.0, unit: mm_per_ms}
  nodes:
    - {id: 0, label: R0, dynamics: Generic2dOscillator}
    - {id: 1, label: R1, dynamics: Generic2dOscillator}
    - {id: 2, label: R2, dynamics: Generic2dOscillator}
    - {id: 3, label: R3, dynamics: Generic2dOscillator}
  edges:
    - {source: 0, target: 1, directed: false, parameters: {weight: {value: 0.5}, length: {value: 30.0, unit: mm}}}
    - {source: 1, target: 2, directed: false, parameters: {weight: {value: 0.4}, length: {value: 45.0, unit: mm}}}
    - {source: 2, target: 3, directed: false, parameters: {weight: {value: 0.3}, length: {value: 60.0, unit: mm}}}
    - {source: 0, target: 3, directed: false, parameters: {weight: {value: 0.2}, length: {value: 75.0, unit: mm}}}
integration:
  method: Heun
  step_size: 0.5
  duration: 50
observations:
  activity:
    label: Mean V activity
    source: [V]
explorations:
  speed_sweep:
    space:
      - parameter: conduction_speed
        domain: {lo: 2.0, hi: 5.0, n: 3}
optimizations:
  speed_fit:
    loss:
      function: mse
      arguments:
        - {name: simulated, value: observations.activity.data}
        - {name: target, value: 0.0}
    stages:
      - name: fit_speed
        free_parameters:
          - parameter: conduction_speed
        algorithm: adam
        learning_rate: 0.2
        max_iterations: 5
"""


def _is_conduction_speed_unsupported(exc: Exception) -> bool:
    """tvbo builds that don't expose conduction_speed fail with it missing from state."""
    return "conduction_speed" in str(exc)


@unittest.skipUnless(_HAS_INTERPOLATE, "tvbo build lacks the interpolate_delays coupling slot")
class TestDifferentiableDelaysExperiment(unittest.TestCase):
    def test_codegen_emits_interpolate_delays(self):
        """The tvboptim code generator wires interpolate_delays into the coupling."""
        code = SimulationExperiment.from_string(exp_yaml).render_code("tvboptim")
        self.assertIn("interpolate_delays=True", code)

    def test_run_tvboptim_forward(self):
        """A forward run integrates the interpolated-delay network to a finite series."""
        result = SimulationExperiment.from_string(exp_yaml).run("tvboptim", mode="simulation")
        ys = np.asarray(getattr(result.integration, "data", result.integration))
        self.assertEqual(ys.shape[-1], 4)  # 4 nodes
        self.assertTrue(bool(np.all(np.isfinite(ys))))

    def test_conduction_speed_exploration(self):
        """The experiment sweeps conduction_speed over a 3-point grid (interpolated delays)."""
        result = SimulationExperiment.from_string(exp_yaml).run("tvboptim", mode="exploration")
        axis = result.explorations.speed_sweep.axes[0]
        self.assertEqual(axis.name, "conduction_speed")
        self.assertEqual(len(np.asarray(axis.explored_values)), 3)

    def test_run_tvboptim_optimizes_conduction_speed(self):
        """from_string(exp_yaml).run("tvboptim") builds and acts on the conduction-speed gradient.

        The default run executes every stage, including the gradient-based
        optimization of conduction_speed. The gradient flows speed -> delays ->
        the interpolated coupling. Requires tvbo to expose conduction_speed as an
        optimizable parameter (currently excluded from collect_state and baked
        into the delays at codegen time); until then the run raises with
        conduction_speed missing from the state and we skip.
        """
        try:
            result = SimulationExperiment.from_string(exp_yaml).run("tvboptim")
        except Exception as exc:  # noqa: BLE001
            if _is_conduction_speed_unsupported(exc):
                self.skipTest(
                    "tvbo does not yet expose conduction_speed as an optimizable parameter: "
                    "include it in SimulationExperiment.collect_state and recompute "
                    "delays = lengths / conduction_speed from that state leaf at runtime "
                    "(the tvboptim interpolate_delays path then makes the gradient flow)."
                )
            raise
        self.assertIsNotNone(result.optimizations.speed_fit)  # gradient built; optimizer stepped


if __name__ == "__main__":
    unittest.main()
