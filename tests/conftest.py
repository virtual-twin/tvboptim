"""Test-suite configuration that must run before JAX is imported."""

import os

# ParallelExecution tests exercise pmap with up to five CPU devices. Configure
# the host device count during pytest startup; setting this in an individual test
# module is too late when an earlier-collected module has already initialized JAX.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
