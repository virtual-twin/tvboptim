"""Parametric external inputs - stateless time-dependent signals.

This module provides common parametric input patterns like sine waves,
pulses, and ramps. These are stateless inputs that compute output as
pure functions of time and parameters.
"""

from typing import Tuple

import jax.numpy as jnp

from ..core.bunch import Bunch
from .base import AbstractExternalInput


class SineInput(AbstractExternalInput):
    """Sinusoidal input: amplitude * sin(2π * frequency * t + phase).

    Parameters support broadcasting for spatial patterns:
    - amplitude: Scalar or [n_nodes] array
    - phase: Scalar or [n_nodes] array for spatial phase patterns
    - frequency: Typically scalar (same frequency all nodes)

    Examples:
        # Global sine wave
        stim = SineInput(frequency=10.0, amplitude=1.0, phase=0.0)

        # Spatial amplitude gradient
        amps = jnp.linspace(0, 1, n_nodes)
        stim = SineInput(frequency=10.0, amplitude=amps)

        # Traveling wave (phase gradient)
        phases = jnp.linspace(0, 2*jnp.pi, n_nodes)
        stim = SineInput(frequency=10.0, amplitude=1.0, phase=phases)
    """

    N_OUTPUT_DIMS = 1
    DEFAULT_PARAMS = Bunch(frequency=10.0, amplitude=1.0, phase=0.0)

    def prepare(self, network, dt: float) -> Tuple[Bunch, Bunch]:
        """No preparation needed for sine input."""
        return Bunch(), Bunch()

    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        input_data: Bunch,
        input_state: Bunch,
        params: Bunch,
    ) -> jnp.ndarray:
        """Compute sine wave at time t."""
        signal = params.amplitude * jnp.sin(
            2 * jnp.pi * params.frequency * t + params.phase
        )

        # Handle broadcasting: scalar or [n_nodes] -> [1, n_nodes]
        if jnp.ndim(signal) == 0:
            # Scalar: broadcast to all nodes
            return jnp.full((1, state.shape[1]), signal)
        else:
            # Already [n_nodes]: add output dimension
            return signal[None, :]

    def update_state(
        self, input_data: Bunch, input_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """No state to update (stateless)."""
        return input_state


class PulseInput(AbstractExternalInput):
    """Single rectangular pulse.

    Outputs amplitude during [onset, onset+duration], zero otherwise.

    Parameters:
        onset: Time when pulse starts
        duration: How long pulse lasts
        amplitude: Pulse height (scalar or [n_nodes])

    Example:
        # 5-unit pulse starting at t=10
        stim = PulseInput(onset=10.0, duration=5.0, amplitude=1.0)

        # Pulse with node-specific amplitudes
        amps = jnp.array([1.0, 0.5, 0.0, 0.5, 1.0])
        stim = PulseInput(onset=10.0, duration=5.0, amplitude=amps)
    """

    N_OUTPUT_DIMS = 1
    DEFAULT_PARAMS = Bunch(onset=0.0, duration=1.0, amplitude=1.0)

    def prepare(self, network, dt: float) -> Tuple[Bunch, Bunch]:
        """No preparation needed."""
        return Bunch(), Bunch()

    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        input_data: Bunch,
        input_state: Bunch,
        params: Bunch,
    ) -> jnp.ndarray:
        """Compute pulse at time t."""
        # Check if we're within the pulse window
        active = (t >= params.onset) & (t < params.onset + params.duration)
        signal = jnp.where(active, params.amplitude, 0.0)

        # Handle broadcasting
        if jnp.ndim(signal) == 0:
            return jnp.full((1, state.shape[1]), signal)
        else:
            return signal[None, :]

    def update_state(
        self, input_data: Bunch, input_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """No state to update."""
        return input_state


class PulseTrainInput(AbstractExternalInput):
    """Periodic rectangular pulse train.

    Parameters:
        frequency: Pulse repetition rate (Hz)
        duty_cycle: Fraction of period when pulse is active (0-1)
        amplitude: Pulse height (scalar or [n_nodes])

    Example:
        # 1 Hz pulse train, 50% duty cycle
        stim = PulseTrainInput(frequency=1.0, duty_cycle=0.5, amplitude=1.0)

        # 10 Hz pulses, 20% duty (short pulses)
        stim = PulseTrainInput(frequency=10.0, duty_cycle=0.2, amplitude=2.0)
    """

    N_OUTPUT_DIMS = 1
    DEFAULT_PARAMS = Bunch(frequency=1.0, duty_cycle=0.5, amplitude=1.0)

    def prepare(self, network, dt: float) -> Tuple[Bunch, Bunch]:
        """No preparation needed."""
        return Bunch(), Bunch()

    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        input_data: Bunch,
        input_state: Bunch,
        params: Bunch,
    ) -> jnp.ndarray:
        """Compute pulse train at time t."""
        period = 1.0 / params.frequency
        phase = (t % period) / period  # Normalized position in cycle [0, 1)
        active = phase < params.duty_cycle
        signal = jnp.where(active, params.amplitude, 0.0)

        # Handle broadcasting
        if jnp.ndim(signal) == 0:
            return jnp.full((1, state.shape[1]), signal)
        else:
            return signal[None, :]

    def update_state(
        self, input_data: Bunch, input_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """No state to update."""
        return input_state


class RampInput(AbstractExternalInput):
    """Linear ramp from t_start to t_end.

    The signal linearly increases from 0 to amplitude over the ramp duration,
    then stays at amplitude.

    Parameters:
        t_start: When ramp begins
        t_end: When ramp finishes
        amplitude: Final value (scalar or [n_nodes])

    Example:
        # Ramp up from t=0 to t=10
        stim = RampInput(t_start=0.0, t_end=10.0, amplitude=1.0)
    """

    N_OUTPUT_DIMS = 1
    DEFAULT_PARAMS = Bunch(t_start=0.0, t_end=10.0, amplitude=1.0)

    def prepare(self, network, dt: float) -> Tuple[Bunch, Bunch]:
        """No preparation needed."""
        return Bunch(), Bunch()

    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        input_data: Bunch,
        input_state: Bunch,
        params: Bunch,
    ) -> jnp.ndarray:
        """Compute ramp at time t."""
        duration = params.t_end - params.t_start

        # Fraction of ramp completed (clipped to [0, 1])
        fraction = jnp.clip((t - params.t_start) / duration, 0.0, 1.0)
        signal = fraction * params.amplitude

        # Handle broadcasting
        if jnp.ndim(signal) == 0:
            return jnp.full((1, state.shape[1]), signal)
        else:
            return signal[None, :]

    def update_state(
        self, input_data: Bunch, input_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """No state to update."""
        return input_state


class ConstantInput(AbstractExternalInput):
    """Constant input signal.

    Simplest possible input - just returns a constant value.

    Parameters:
        amplitude: Constant value (scalar or [n_nodes])

    Example:
        # Global constant
        stim = ConstantInput(amplitude=0.5)

        # Node-specific constants (spatial pattern)
        values = jnp.array([0.0, 0.5, 1.0, 0.5, 0.0])
        stim = ConstantInput(amplitude=values)
    """

    N_OUTPUT_DIMS = 1
    DEFAULT_PARAMS = Bunch(amplitude=1.0)

    def prepare(self, network, dt: float) -> Tuple[Bunch, Bunch]:
        """No preparation needed."""
        return Bunch(), Bunch()

    def compute(
        self,
        t: float,
        state: jnp.ndarray,
        input_data: Bunch,
        input_state: Bunch,
        params: Bunch,
    ) -> jnp.ndarray:
        """Return constant value."""
        signal = params.amplitude

        # Handle broadcasting
        if jnp.ndim(signal) == 0:
            return jnp.full((1, state.shape[1]), signal)
        else:
            return signal[None, :]

    def update_state(
        self, input_data: Bunch, input_state: Bunch, new_state: jnp.ndarray
    ) -> Bunch:
        """No state to update."""
        return input_state
