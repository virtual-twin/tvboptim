"""External input system for network dynamics.

This module provides external inputs that can drive network dynamics
independently of coupling. External inputs are time-dependent (and optionally
state-dependent) signals.
"""

from .base import AbstractExternalInput
from .parametric import (
    SineInput,
    PulseInput,
    PulseTrainInput,
    RampInput,
    ConstantInput,
)
from .data import DataInput

__all__ = [
    'AbstractExternalInput',
    'SineInput',
    'PulseInput',
    'PulseTrainInput',
    'RampInput',
    'ConstantInput',
    'DataInput',
]
