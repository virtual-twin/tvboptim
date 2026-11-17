"""Every coupling class in the coupling package is part of its public API."""

import importlib
import inspect
import pkgutil

import pytest

import tvboptim.experimental.network_dynamics.coupling as coupling_pkg
from tvboptim.experimental.network_dynamics.coupling.base import AbstractCoupling


def _defined_couplings():
    found = []
    for info in pkgutil.iter_modules(coupling_pkg.__path__):
        module = importlib.import_module(f"{coupling_pkg.__name__}.{info.name}")
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, AbstractCoupling)
                and obj.__module__ == module.__name__
                and not name.startswith("_")
            ):
                found.append(name)
    return sorted(found)


@pytest.mark.parametrize("name", _defined_couplings())
def test_coupling_is_exported(name):
    assert name in coupling_pkg.__all__
    assert getattr(coupling_pkg, name).__name__ == name
