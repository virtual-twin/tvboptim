"""TVBO must not be imported until a TVBO object actually reaches prepare().

Import-time behaviour cannot be observed from a process that has already
imported the package, so these run in a subprocess and assert on ``sys.modules``
rather than on wall-clock time.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from plum import NotFoundLookupError

from tvboptim.tvbo import prepare_experiment as module

SRC = str(Path(__file__).parents[1] / "src")


def run(body: str) -> subprocess.CompletedProcess:
    """Execute ``body`` in a fresh interpreter importing this checkout.

    Inherit the ambient environment and override only PYTHONPATH. Replacing it
    wholesale breaks Windows: without SystemRoot, Winsock cannot initialize, and
    jaxtyping imports unittest.mock, which imports asyncio, which imports
    _overlapped -- so every subprocess dies with WinError 10106 before reaching
    the assertion. Isolation here comes from the subprocess, not the env.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_tvbo_stays_unimported_until_a_tvbo_object_arrives():
    # Importing, asking whether TVBO exists, and making an ordinary type error
    # must all avoid the multi-second TVBO import.
    result = run("""
        import sys
        import tvboptim
        from plum import NotFoundLookupError

        loaded = {"after import": "tvbo" in sys.modules}
        assert isinstance(tvboptim.HAS_TVBO, bool)
        loaded["after HAS_TVBO"] = "tvbo" in sys.modules
        try:
            tvboptim.prepare(3)
        except NotFoundLookupError:
            pass
        loaded["after type error"] = "tvbo" in sys.modules
        print(loaded)
    """)
    loaded = eval(result.stdout)
    assert loaded == dict.fromkeys(loaded, False), loaded


def test_fallback_does_not_shadow_network_dispatch():
    # The catch-all must never win over a more specific prepare() method.
    result = run("""
        import jax.numpy as jnp, sys
        from tvboptim.experimental.network_dynamics import Network, prepare
        from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
        from tvboptim.experimental.network_dynamics.dynamics.tvb import Linear
        from tvboptim.experimental.network_dynamics.graph import DenseGraph
        from tvboptim.experimental.network_dynamics.solvers import Heun

        network = Network(
            dynamics=Linear(gamma=-0.2),
            coupling={"instant": LinearCoupling(source="x", G=0.1)},
            graph=DenseGraph(jnp.array([[0.0, 1.0], [1.0, 0.0]])),
        )
        solve_fn, config = prepare(network, Heun(), t0=0.0, t1=2.0, dt=0.1)
        print(solve_fn(config).ys.shape, "tvbo" in sys.modules)
    """)
    assert result.stdout.strip() == "(20, 1, 2) False"


def test_unsupported_type_raises_the_error_plum_would_have():
    # Constructing NotFoundLookupError takes (f_name, target, methods); passing
    # a bare message makes the constructor itself raise TypeError.
    import tvboptim

    with pytest.raises(NotFoundLookupError, match="could not be resolved"):
        tvboptim.prepare(object())


@pytest.mark.skipif(
    not module._tvbo_is_installed(), reason="requires the optional tvbo package"
)
def test_tvbo_object_triggers_one_announced_import_and_registration():
    result = run("""
        import sys
        import tvboptim
        from plum import NotFoundLookupError

        before = len(tvboptim.prepare.methods)

        # Shaped like a TVBO object without being a real experiment: enough to
        # trigger the lazy load, not enough to dispatch.
        probe = type("Probe", (), {})
        probe.__module__ = "tvbo.export.experiment"
        for _ in range(2):
            try:
                tvboptim.prepare(probe())
            except NotFoundLookupError:
                pass

        print(
            "tvbo" in sys.modules,
            len(tvboptim.prepare.methods) > before,
            any("SimulationExperiment" in str(m.signature)
                for m in tvboptim.prepare.methods),
        )
    """)
    assert result.stdout.strip() == "True True True"
    assert result.stderr.count("Importing TVB-O") == 1
