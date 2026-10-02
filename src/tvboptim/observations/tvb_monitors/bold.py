import math
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.core.observation import (
    ObservationOutput,
    PreparedObservation,
    SimulationGrid,
    apply_observation,
    observation_alignment_steps,
    observation_result,
    prepare_observation,
    prepared_single_output_sample,
    validate_prepared_observation,
)
from tvboptim.experimental.network_dynamics.result import NativeSolution

from .downsampling import (
    AbstractMonitor,
    SubSampling,
    TemporalAverage,
    _integer_stride,
    _resolve_selection,
    _slice_variable_names,
    _validate_monitor_input,
)


def _bold_variable_names(sol, voi=None):
    """Build BOLD output variable names by wrapping source names as 'BOLD(<name>)'.

    If `voi` is provided, source names are sliced by it first (used when the
    monitor does its own voi selection instead of delegating to a downsampler).
    Returns None if the source has no variable_names.
    """
    names = (
        _slice_variable_names(sol, voi)
        if voi is not None
        else getattr(sol, "variable_names", None)
    )
    if names is None:
        return None
    return tuple(f"BOLD({n})" for n in names)


def _bold_names(names):
    """Wrap already-resolved source channel names as BOLD names."""
    if names is None:
        return None
    return tuple(f"BOLD({name})" for name in names)


class HRFKernel(eqx.Module):
    """Base class for hemodynamic response function kernels.

    A kernel is a function of time that defines the hemodynamic response.
    Subclasses must implement the kernel computation and specify its duration.

    Attributes:
        duration: Duration of kernel support in milliseconds
    """

    duration: eqx.AbstractVar[float]

    def __call__(self, t: jax.Array, downsample_dt: float) -> jax.Array:
        """Compute kernel values at time points.

        Args:
            t: Time points at which to evaluate the kernel
            downsample_dt: Step size of the convolution grid `t` is sampled on.
                Closed-form kernels (e.g. FirstOrderVolterraHRFKernel) are exact
                functions of `t` and ignore this. It is part of the interface
                for grid-dependent kernels not yet implemented — e.g. an
                empirical/tabulated HRF that must resample itself onto the
                convolution grid, or a kernel that scales by the sample spacing
                to approximate the convolution integral.

        Returns:
            Kernel values at the specified time points
        """
        raise NotImplementedError

    def plot(self, dt=1.0, ax=None):
        """Plot the kernel function over its duration.

        Args:
            dt: Time step in milliseconds (default: 1.0 ms)
            ax: Matplotlib axis to plot on (default: None, creates new figure)

        Returns:
            Matplotlib axis object
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        # Compute number of samples from duration and dt
        n_samples = int(jnp.ceil(self.duration / dt))

        # Evaluate kernel over its duration
        t = jnp.linspace(0.0, self.duration, n_samples)
        kernel_values = self(t, dt)

        # Plot
        ax.plot(t, kernel_values)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Kernel value")
        ax.set_title(f"{self.__class__.__name__}")
        ax.grid(True, alpha=0.3)

        return ax


class FirstOrderVolterraHRFKernel(HRFKernel):
    """First-order Volterra kernel of the hemodynamic system.

    This is the canonical damped-oscillator HRF used in standard BOLD signal
    modeling — the first-order Volterra kernel of the Balloon/Windkessel
    hemodynamics (Friston et al. 2000). Ported from TVB's ``FirstOrderVolterra``
    equation.

    Despite the shared name of the mathematician Vito Volterra, this is
    unrelated to Lotka-Volterra (predator-prey) dynamics.

    Attributes:
        tau_s: Signal decay time constant in seconds (default: 0.8 s)
        tau_f: Feedback time constant in seconds (default: 0.4 s)
        scaling: Kernel amplitude scaling factor (default: 1/3)
        duration: Kernel support duration in ms (default: 20,000 ms = 20 s)

    Note:
        The tau parameters are in seconds (not ms) to match the standard HRF
        formulation. Time input to __call__ is expected in milliseconds and
        converted internally.

        This is the *underdamped* solution: the oscillation frequency
        ``omega = sqrt(1/tau_f - 1/(4*tau_s**2))`` is real only when
        ``4*tau_s**2 > tau_f``. The defaults satisfy this. Parameter values that
        violate it make ``omega`` NaN and the whole kernel NaN — the overdamped
        regime (which would need ``sinh`` instead of ``sin``) is not supported.
    """

    tau_s: float = 0.8  # seconds
    tau_f: float = 0.4  # seconds
    scaling: float = 1.0 / 3.0
    duration: float = 20_000.0  # ms (20 seconds)

    def __call__(self, t: jax.Array, downsample_dt: float) -> jax.Array:
        """Compute the first-order Volterra HRF kernel.

        Args:
            t: Time points in milliseconds at which to evaluate the kernel
            downsample_dt: Not used for this kernel

        Returns:
            HRF kernel values
        """
        # Convert time from ms to seconds for the HRF formula
        t_seconds = t / 1000.0

        omega = jnp.sqrt(1.0 / self.tau_f - 1.0 / (4.0 * self.tau_s**2))
        return (
            self.scaling
            * jnp.exp(-0.5 * (t_seconds / self.tau_s))
            * jnp.sin(omega * t_seconds)
            / omega
        )


class GammaHRFKernel(HRFKernel):
    """
    Gamma HRF kernel, ported from TVBSim's Gamma class.

    h(t) = ((t/tau)^(n-1) * exp(-(t/tau))) / (tau * (n-1)!)
    normalized and scaled by amplitude factor `a`.

    Parameters
    ----------
    tau : float
        Exponential time constant in seconds (default: 1.08 s)
    n : float
        Phase delay / shape parameter (default: 3.0)
    a : float
        Amplitude scaling factor after normalization (default: 0.1)
    duration : float
        Kernel support duration in ms (default: 20_000 ms)

    Reference
    ---------
    Boynton et al. (1996). Linear Systems Analysis of fMRI in Human V1.
    J Neurosci 16: 4207-4221.
    """

    tau: float = 1.08  # seconds
    n: float = 3.0
    a: float = 0.1
    duration: float = 20_000.0  # ms

    def __call__(self, t: jax.Array, downsample_dt: float) -> jax.Array:
        # Convert time from ms to seconds for the HRF formula
        t_s = t / 1000.0

        factorial = math.factorial(int(self.n) - 1)

        kernel = ((t_s / self.tau) ** (self.n - 1) * jnp.exp(-(t_s / self.tau))) / (
            self.tau * factorial
        )

        # Replicate TVBSim's normalization and amplitude scaling from evaluate()
        peak = jnp.max(kernel)
        peak = jnp.where(peak > 0, peak, 1.0)  # Avoid division by zero
        kernel = kernel / peak
        kernel = kernel * self.a

        return kernel


class DoubleExponentialHRFKernel(HRFKernel):
    """
    A difference of two exponential functions to define a kernel for the bold monitor, ported from TVBSim's DoubleExponential class.

    h(t) = amp_1*exp(-t/tau_1)*sin(2*pi*f_1*t) - amp_2*exp(-t/tau_2)*sin(2*pi*f_2*t)
    normalized and scaled by amplitude factor `a`.

    Parameters
    ----------
    tau_1 : float
        Time constant of the first exponential function [s] (default: 7.22)
    tau_2 : float
        Time constant of the second exponential function [s] (default: 7.4)
    f_1 : float
        Frequency of the first sine function [Hz] (default: 0.03)
    f_2 : float
        Frequency of the second sine function [Hz] (default: 0.12)
    amp_1 : float
        Amplitude of the first exponential function (default: 0.1)
    amp_2 : float
        Amplitude of the second exponential function. (default: 0.1)
    a : float
        Amplitude factor after normalization (default: 0.1)

    Reference
    ---------
        Alex Polonsky, Randolph Blake, Jochen Braun and David J. Heeger
        (2000). Neuronal activity in human primary visual cortex correlates with
        perception during binocular rivalry. Nature Neuroscience 3: 1153-1159

    """

    tau_1: float = 7.22
    tau_2: float = 7.4
    f_1: float = 0.03
    f_2: float = 0.12
    amp_1: float = 0.1
    amp_2: float = 0.1
    a: float = 0.1
    duration: float = 40_000.0  # ms

    def __call__(self, t: jax.Array, downsample_dt: float) -> jax.Array:
        # Convert ms to seconds
        t_s = t / 1000.0

        kernel = (
            self.amp_1
            * jnp.exp(-t_s / self.tau_1)
            * jnp.sin(2 * math.pi * self.f_1 * t_s)
        ) - (
            self.amp_2
            * jnp.exp(-t_s / self.tau_2)
            * jnp.sin(2 * math.pi * self.f_2 * t_s)
        )

        # Replicate TVBSim's normalization + amplitude scaling from evaluate()
        peak = jnp.max(kernel)
        peak = jnp.where(peak > 0, peak, 1.0)  # Avoid division by zero
        kernel = kernel / peak
        kernel = kernel * self.a

        return kernel


class MixtureOfGammasHRFKernel(HRFKernel):
    """
    Mixture of two gamma distributions HRF kernel, ported from TVBSim's MixtureOfGammas.

    hrf(t) = (l*t)^(a_1-1) * exp(-l*t) / Γ(a_1)
           - c * (l*t)^(a_2-1) * exp(-l*t) / Γ(a_2)

    Parameters
    ----------
    a_1 : float
        Shape parameter of the first gamma pdf (default: 6.0)
    a_2 : float
        Shape parameter of the second gamma pdf (default: 13.0)
    l : float
        Scale parameter / lambda (default: 1.0)
    c : float
        Scaling factor of the second gamma pdf (default: 0.4)
    duration : float
        Kernel support duration [ms] (default: 20_000)

    Reference
    ---------
    Glover (1999). Deconvolution of Impulse Response in Event-Related BOLD fMRI.
    NeuroImage 9, 416-429.
    """

    a_1: float = 6.0
    a_2: float = 13.0
    l: float = 1.0
    c: float = 0.4
    duration: float = 20_000.0

    def __call__(self, t: jax.Array, downsample_dt: float) -> jax.Array:
        t_s = t / 1000.0

        gamma_a_1 = jsp.special.gamma(self.a_1)
        gamma_a_2 = jsp.special.gamma(self.a_2)

        return (self.l * t_s) ** (self.a_1 - 1) * jnp.exp(
            -self.l * t_s
        ) / gamma_a_1 - self.c * (self.l * t_s) ** (self.a_2 - 1) * jnp.exp(
            -self.l * t_s
        ) / gamma_a_2


def LotkaVolterraHRFKernel(*args, **kwargs):
    """Deprecated: use FirstOrderVolterraHRFKernel.

    The kernel was misnamed — it is the first-order Volterra kernel of the
    hemodynamic system (Friston 2000), unrelated to Lotka-Volterra dynamics.
    """
    warnings.warn(
        "LotkaVolterraHRFKernel is deprecated and will be removed in a future "
        "version. Use FirstOrderVolterraHRFKernel instead — the kernel is the "
        "first-order Volterra kernel of the hemodynamic system (Friston 2000), "
        "unrelated to Lotka-Volterra dynamics.",
        DeprecationWarning,
        stacklevel=2,
    )
    return FirstOrderVolterraHRFKernel(*args, **kwargs)


def _build_hrf_kernel(kernel, downsample_dt):
    """Evaluate an HRF kernel on its fixed intermediate sampling grid."""
    duration = float(kernel.duration)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(
            f"HRF kernel duration must be finite and positive; got {duration!r}"
        )
    downsample_dt = float(downsample_dt)
    if not math.isfinite(downsample_dt) or downsample_dt <= 0.0:
        raise ValueError(
            "HRF downsampling period must be finite and positive; "
            f"got {downsample_dt!r}"
        )
    kernel_samples = int(math.ceil(duration / downsample_dt))
    kernel_time = jnp.linspace(0.0, duration, kernel_samples)
    values = jnp.asarray(kernel(kernel_time, downsample_dt))
    if values.shape != (kernel_samples,):
        raise ValueError(
            "HRF kernel must return one value per kernel sample; "
            f"expected {(kernel_samples,)}, got {values.shape}"
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(jnp.result_type(values.dtype, 1.0))
    return values


def _normalize_hrf_history(history, kernel_samples, n_channels, n_nodes, dtype):
    """Normalize HRF history to ``[kernel_samples, channels, nodes]``."""
    if history is None:
        return jnp.zeros((kernel_samples, n_channels, n_nodes), dtype=dtype)

    normalized = jnp.asarray(history)
    if normalized.ndim == 2 and n_channels == 1:
        normalized = normalized[:, None, :]
    if normalized.ndim != 3:
        raise ValueError(
            "HRFBold history must have shape [time, channels, nodes]; "
            f"got {normalized.shape}"
        )
    if normalized.shape[1:] != (n_channels, n_nodes):
        raise ValueError(
            "HRFBold history channel/node shape does not match the selected "
            f"monitor input: got {normalized.shape[1:]}, expected "
            f"{(n_channels, n_nodes)}"
        )
    normalized = normalized.astype(dtype)
    if normalized.shape[0] < kernel_samples:
        padding = jnp.zeros(
            (kernel_samples - normalized.shape[0], n_channels, n_nodes),
            dtype=dtype,
        )
        normalized = jnp.concatenate([padding, normalized], axis=0)
    return normalized[-kernel_samples:]


def _hrf_dtype(input_dtype, hrf, history, params):
    """Choose one computation dtype for signal, kernel, history, and scaling."""
    leaves = [input_dtype, hrf.dtype, *jax.tree.leaves(params)]
    if history is not None:
        leaves.append(jnp.asarray(history).dtype)
    return jnp.result_type(*leaves)


def _apply_downsample(downsample, solution):
    if callable(downsample):
        return downsample(solution)
    return apply_observation(downsample, solution)


def _hrf_history_values(history, downsample, input_dt):
    """Resolve solution-valued history on the effective convolution grid."""
    if history is None or not (hasattr(history, "ys") and hasattr(history, "ts")):
        return history
    resolved = _apply_downsample(downsample, history)
    resolved_dt = getattr(resolved, "dt", None)
    if resolved_dt is None or not math.isclose(
        float(resolved_dt), float(input_dt), rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(
            "HRFBold solution history must resolve to the convolution-input "
            f"interval {input_dt:g}; got {resolved_dt!r}"
        )
    return resolved.ys


def _convolve_hrf(signal, hrf, mode):
    """Convolve time for every channel/node pair."""

    def convolve_single(values):
        return jsp.signal.fftconvolve(values, hrf, mode=mode)

    return jax.vmap(
        lambda channel: jax.vmap(convolve_single, in_axes=1, out_axes=1)(channel),
        in_axes=1,
        out_axes=1,
    )(signal)


def _sample_valid_hrf(signal, hrf, final_stride):
    """Evaluate valid convolution only at completed TR endpoints."""
    n_valid = signal.shape[0] - hrf.shape[0] + 1
    sample_indices = jnp.arange(final_stride, n_valid, final_stride)
    reversed_hrf = hrf[::-1]

    def convolve_at(start):
        window = jax.lax.dynamic_slice_in_dim(signal, start, hrf.shape[0], axis=0)
        return jnp.tensordot(reversed_hrf, window, axes=(0, 0))

    return jax.lax.map(convolve_at, sample_indices)


def _prefer_direct_hrf(signal_samples, kernel_samples, final_stride):
    """Choose endpoint dots when their static work is below one padded FFT."""
    n_valid = max(0, signal_samples - kernel_samples + 1)
    n_emit = max(0, (n_valid - 1) // final_stride)
    direct_work = n_emit * kernel_samples
    full_length = max(1, signal_samples + kernel_samples - 1)
    fft_length = 1 << (full_length - 1).bit_length()
    fft_work = fft_length * max(1, fft_length.bit_length() - 1)
    return direct_work <= fft_work


def _valid_hrf_samples(signal, hrf, final_stride):
    """Select valid-convolution TR endpoints with a static direct/FFT choice."""
    n_valid = signal.shape[0] - hrf.shape[0] + 1
    if n_valid <= final_stride:
        return signal[:0]
    if _prefer_direct_hrf(signal.shape[0], hrf.shape[0], final_stride):
        return _sample_valid_hrf(signal, hrf, final_stride)
    return _convolve_hrf(signal, hrf, "valid")[final_stride::final_stride]


def _hrf_valid_block(history, block, hrf, final_stride, params):
    """Convolve one downsampled block and emit completed TR endpoints."""
    block = block.astype(history.dtype)
    signal = jnp.concatenate([history, block], axis=0)
    samples = _valid_hrf_samples(signal, hrf, final_stride)
    bold = params.k_1 * params.V_0 * (samples - 1.0)
    return signal[-history.shape[0] :], bold


def _hrf_params(monitor):
    """Collect the live HRF signal scaling parameters."""
    return Bunch(k_1=monitor.k_1, V_0=monitor.V_0)


def _hrf_update(hrf, final_stride):
    """Build causal HRF convolution for one solver block."""

    def update(history, block, params):
        return _hrf_valid_block(history, block, hrf, final_stride, params)

    return update


def _with_downsample(prepared, downsample):
    """Compose a prepared input monitor, carrying its state and live parameters."""
    if downsample is None:
        return prepared
    bold_keys = tuple(prepared.params)

    def bold_params(params):
        return Bunch({key: params[key] for key in bold_keys})

    def init(params):
        return (
            downsample.init(params.downsample),
            prepared.init(bold_params(params)),
        )

    def update(state, block, params):
        downsample_state, bold_state = state
        downsample_state, block = downsample.update(
            downsample_state, block, params.downsample
        )
        bold_state, values = prepared.update(bold_state, block, bold_params(params))
        return (downsample_state, bold_state), values

    return PreparedObservation(
        params=Bunch({**prepared.params, "downsample": downsample.params}),
        init=init,
        update=update,
        output=prepared.output,
        alignment_steps=prepared.alignment_steps,
    )


class HRFBold(AbstractMonitor):
    """BOLD signal monitor using hemodynamic response function convolution.

    This monitor simulates the Blood Oxygen Level Dependent (BOLD) signal by:
    1. Downsampling the neural activity
    2. Convolving with a hemodynamic response function kernel
    3. Downsampling to the final BOLD sampling period

    Calling the monitor applies it to a stored solution. Passing it through a
    native solver's ``observe=`` keyword computes the same valid causal
    convolution block-wise; online execution requires
    ``convolution_mode="valid"`` and a downsampler implementing the streaming
    preparation contract. Any callable downsampler is supported post-hoc.
    Downsampler state is carried between blocks and its live parameters are
    available under ``config.monitor.downsample``. By default the convolution
    input is a ``TemporalAverage`` at ``downsample_period``. Pass ``downsample=Identity()``
    to use the actual incoming grid without another averaging stage.

    Warm history may be an already-downsampled array or a solution on the
    monitor's pre-downsampling input grid. In a joint branch, that input is the
    shared stage's output; raw history is not replayed through the shared stage
    automatically.
    """

    # BOLD model parameters
    k_1: float  # Signal scaling factor
    V_0: float  # Resting blood volume fraction

    # Sampling parameters
    period: float = eqx.field(static=True)  # ms, final BOLD sampling period
    downsample_period: float = eqx.field(static=True)  # ms, intermediate grid

    # Processing configuration
    kernel: HRFKernel = eqx.field(static=True)
    downsample: eqx.Module = eqx.field(static=True)
    convolution_mode: str = eqx.field(static=True)

    # History buffer for continuous monitoring
    history: object = None

    def __init__(
        self,
        k_1=5.6,
        V_0=0.02,
        period=1000.0,
        downsample_period=4.0,
        kernel=None,
        downsample=None,
        voi=None,
        convolution_mode="valid",
        history=None,
    ):
        """Initialize BOLD monitor.

        Args:
            k_1: Signal scaling factor (default: 5.6)
            V_0: Resting blood volume fraction (default: 0.02)
            period: Final BOLD sampling period in ms (default: 1000.0)
            downsample_period: Intermediate downsampling period in ms (default: 4.0)
            kernel: HRF kernel to use (default: FirstOrderVolterraHRFKernel())
            downsample: Downsampling strategy (default: TemporalAverage with voi)
            voi: Variable of interest index for downsampling
            convolution_mode: Convolution mode - 'valid', 'same', or 'full' (default: 'valid')
            history: Prior data for warm start. Can be None (zeros), jax.Array, or NativeSolution
        """
        # Normalize voi using base class method
        self.voi = self._normalize_voi(voi)

        self.k_1 = k_1
        self.V_0 = V_0
        self.period = period
        self.downsample_period = downsample_period
        self.convolution_mode = convolution_mode

        # Set up kernel
        if kernel is None:
            self.kernel = FirstOrderVolterraHRFKernel()
        else:
            self.kernel = kernel

        # Set up downsampling
        if downsample is None:
            # Pass the already normalized voi to the downsampler
            self.downsample = TemporalAverage(voi=self.voi, period=downsample_period)
        else:
            self.downsample = downsample
            # Sync downsample_period with the actual monitor's period
            # so kernel sampling and final BOLD subsampling use the correct grid
            if hasattr(downsample, "period"):
                self.downsample_period = downsample.period

        # Preserve the original history until the effective convolution grid is
        # known. Preparation and post-hoc execution normalize it on that grid.
        self.history = history

    def prepare(self, grid, sample, variable_names):
        """Prepare causal valid-mode HRF convolution for block-wise execution."""
        return _prepare_hrf_bold(self, grid, sample, variable_names)

    def __call__(self, sol, t_offset=0.0):
        """Apply BOLD monitor to simulation results.

        Args:
            sol: Simulation solution with .ys, .ts, and .dt attributes
                 Works with NativeSolution (requires dt as auxiliary data)
            t_offset: Time offset to add to output timestamps (default: 0.0)

        Returns:
            NativeSolution with BOLD signal timeseries
        """
        dt = self._resolve_dt(sol)
        # sol.ts follows the native-solver convention where ts[0] = t0 + dt
        # (post-step state); recover the simulation start t0 to anchor BOLD
        # samples at clean period-multiples.
        t0 = sol.ts[0] - dt if sol.ts.shape[0] else 0.0

        # --- Downsample neural activity ---
        downsampled = _apply_downsample(self.downsample, sol)
        ys_downsampled = downsampled.ys

        input_dt = self._resolve_dt(downsampled)
        hrf = _build_hrf_kernel(self.kernel, input_dt)
        params = _hrf_params(self)
        history_values = _hrf_history_values(self.history, self.downsample, input_dt)
        dtype = _hrf_dtype(ys_downsampled.dtype, hrf, history_values, params)
        hrf = hrf.astype(dtype)
        history = _normalize_hrf_history(
            history_values,
            hrf.shape[0],
            ys_downsampled.shape[1],
            ys_downsampled.shape[2],
            dtype,
        )
        ys_with_history = jnp.concatenate(
            [history, ys_downsampled.astype(dtype)], axis=0
        )
        _integer_stride(self.period, dt, label="BOLD period")
        final_idx_step = _integer_stride(
            self.period,
            input_dt,
            label="BOLD period",
        )

        # Index 0 of valid convolution contains only the history. Sampling begins
        # at one completed TR. Non-valid post-hoc modes retain their established
        # full convolution behavior; prepared execution rejects those modes.
        if self.convolution_mode == "valid":
            samples = _valid_hrf_samples(ys_with_history, hrf, final_idx_step)
            bold_signal = params.k_1 * params.V_0 * (samples - 1.0)
        else:
            bold = (
                params.k_1
                * params.V_0
                * (_convolve_hrf(ys_with_history, hrf, self.convolution_mode) - 1.0)
            )
            bold_indices = jnp.arange(final_idx_step, bold.shape[0], final_idx_step)
            bold_signal = bold[bold_indices, ...]

        n_bold = bold_signal.shape[0]
        bold_time = t0 + (jnp.arange(n_bold) + 1) * self.period + t_offset

        return NativeSolution(
            ts=bold_time,
            ys=bold_signal,
            dt=self.period,
            variable_names=_bold_variable_names(downsampled),
        )


def _prepare_hrf_bold(
    monitor,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    variable_names: tuple | None,
) -> PreparedObservation:
    """Prepare causal valid-mode HRF convolution on aligned solver blocks."""
    _validate_monitor_input(sample, variable_names)
    if monitor.convolution_mode != "valid":
        raise ValueError(
            "Prepared HRFBold supports convolution_mode='valid'; "
            f"got {monitor.convolution_mode!r}"
        )
    _integer_stride(monitor.period, grid.dt, label="BOLD period")
    prepared_downsample = prepare_observation(
        monitor.downsample, grid, sample, variable_names
    )
    if not isinstance(prepared_downsample.output, ObservationOutput):
        raise ValueError("HRFBold downsample must produce one time series")
    downsample_output = prepared_downsample.output
    input_dt = downsample_output.period
    _integer_stride(monitor.period, input_dt, label="BOLD period")
    hrf = _build_hrf_kernel(monitor.kernel, input_dt)
    params = _hrf_params(monitor)
    history_values = _hrf_history_values(monitor.history, monitor.downsample, input_dt)
    downsample_alignment = observation_alignment_steps(prepared_downsample, grid)
    downsample_sample = prepared_single_output_sample(
        prepared_downsample, sample, downsample_alignment
    )
    dtype = _hrf_dtype(downsample_sample.dtype, hrf, history_values, params)
    hrf = hrf.astype(dtype)
    n_channels, n_nodes = downsample_sample.shape
    history = _normalize_hrf_history(
        history_values,
        hrf.shape[0],
        n_channels,
        n_nodes,
        dtype,
    )
    final_stride = _integer_stride(monitor.period, input_dt, label="BOLD period")

    prepared = PreparedObservation(
        params=params,
        # Warm history is prepared state, not a live parameter.
        init=lambda params: history,
        update=_hrf_update(hrf, final_stride),
        output=grid.output(
            monitor.period,
            label="period_end",
            variable_names=_bold_names(downsample_output.variable_names),
        ),
        alignment_steps=math.lcm(
            downsample_alignment,
            _integer_stride(monitor.period, grid.dt, label="BOLD period"),
        ),
    )
    return _with_downsample(prepared, prepared_downsample)


def streaming_hrf_bold(monitor, dt):
    """Block-level streaming reducer form of :class:`HRFBold`.

    Returns an ``(init, update, finalize)`` triple for the ``reduce=`` kwarg of
    ``prepare`` / ``solve``, computing the same BOLD signal as
    ``monitor(full_solution)`` without ever stacking the full neural trajectory.
    It reuses the monitor's kernel, periods and BOLD scaling, and carries a
    downsampled-history ring buffer plus a preallocated BOLD output buffer; each
    block subsamples and evaluates the shared causal valid convolution at BOLD
    TR boundaries before writing those samples into the buffer.

    Requirements (so blocks align with the decimation and BOLD grids):

    - this legacy reducer requires a ``SubSampling`` downsampler. Use
      ``observe=monitor`` for prepared ``TemporalAverage`` support.
    - ``block_size`` and ``n_steps`` must be multiples of the BOLD period in raw
      steps (``period / dt``); the per-block update asserts this.

    Warm start / chaining: ``init`` seeds the ring from ``monitor.history`` when
    set (the same warm-start the monitor accepts); ``finalize`` returns the BOLD
    buffer ``[n_bold, n_voi, n_nodes]``.
    """
    warnings.warn(
        "streaming_hrf_bold is deprecated with the reduce= API in 0.5.0 and "
        "will be removed in 0.6.0. Pass the HRFBold monitor through "
        "observe= instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ds_period = monitor.downsample_period
    period = monitor.period
    voi = monitor.downsample.voi
    conv_mode = monitor.convolution_mode
    kernel = monitor.kernel
    warm_history = monitor.history
    params = _hrf_params(monitor)

    if not isinstance(monitor.downsample, SubSampling):
        raise ValueError("streaming_hrf_bold requires a SubSampling downsampler")
    if conv_mode != "valid":
        raise ValueError("streaming_hrf_bold requires convolution_mode='valid'")

    ds_steps = _integer_stride(ds_period, dt, label="HRF downsampling period")
    final_idx_step = _integer_stride(period, ds_period, label="BOLD period")
    period_in_steps = ds_steps * final_idx_step
    hrf = _build_hrf_kernel(kernel, ds_period)
    kernel_samples = hrf.shape[0]
    warm_history_values = _hrf_history_values(
        warm_history, monitor.downsample, ds_period
    )

    def init(template, n_steps):
        t_sel = template[voi, :]  # [n_voi, n_nodes]
        n_voi, n_nodes = t_sel.shape
        # SubSampling emits these indices; n_bold matches HRFBold's bold_indices.
        n_ds = len(range(ds_steps - 1, n_steps, ds_steps))
        n_bold = len(range(final_idx_step, n_ds + 1, final_idx_step))
        dtype = _hrf_dtype(template.dtype, hrf, warm_history_values, params)
        ring0 = _normalize_hrf_history(
            warm_history_values,
            kernel_samples,
            n_voi,
            n_nodes,
            dtype,
        )
        bold0 = jnp.zeros((n_bold, n_voi, n_nodes), dtype=dtype)
        return (ring0, bold0, jnp.array(0))

    def update(acc, block):
        ring, bold_buffer, ds_count = acc
        y = block[:, voi, :]  # [block_len, n_voi, n_nodes]
        block_len = y.shape[0]
        assert block_len % period_in_steps == 0, (
            "streaming_hrf_bold requires each block length to be a multiple of "
            f"the BOLD period in steps ({period_in_steps} = period/dt); got "
            f"{block_len}. Set block_size and n_steps to multiples of period/dt."
        )
        block_ds = y[ds_steps - 1 :: ds_steps]  # SubSampling, [m_b, n_voi, n_nodes]
        m_b = block_ds.shape[0]
        ring, bold_samples = _hrf_valid_block(
            ring,
            block_ds,
            hrf.astype(ring.dtype),
            final_idx_step,
            params,
        )
        start = ds_count // final_idx_step
        bold_buffer = jax.lax.dynamic_update_slice(
            bold_buffer, bold_samples, (start,) + (0,) * (bold_buffer.ndim - 1)
        )
        return (ring, bold_buffer, ds_count + m_b)

    def finalize(acc):
        _ring, bold_buffer, _ds_count = acc
        return bold_buffer

    return (init, update, finalize)


def _bw_resampling_factors(input_dt, dt_bw):
    """Return static repeat/decimation factors for the BW input grid."""
    input_dt = float(input_dt)
    dt_bw = float(dt_bw)
    if input_dt >= dt_bw:
        return (
            _integer_stride(input_dt, dt_bw, label="BW input interval"),
            1,
        )
    return (
        1,
        _integer_stride(dt_bw, input_dt, label="BW integration interval"),
    )


def _resample_bw_input(values, repeat, decimate):
    """Move regularly sampled neural drive onto the BW integration grid."""
    if repeat > 1:
        return jnp.repeat(values, repeat, axis=0)
    if decimate > 1:
        # The drive at each coarser integration endpoint represents the completed
        # input interval, matching the endpoint convention of SubSampling.
        return values[decimate - 1 :: decimate]
    return values


def _bw_signal(state, params):
    """Evaluate the BOLD signal from the current hemodynamic state."""
    _s, _f, v, q = state
    k1 = 4.3 * 40.3 * params.Eo * params.TE
    k2 = 25.0 * params.Eo * params.TE
    return params.vo * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + params.k3 * (1.0 - v))


def _integrate_bw(state, drive, params, dt_bw):
    """Integrate a block of neural drive and return all endpoint signals."""
    dt_s = dt_bw / 1000.0

    def step(current, firing_rate):
        s, f, v, q = current
        ds = firing_rate - s / params.taus - (f - 1.0) / params.tauf
        df = s
        dv = (f - v ** (1.0 / params.alpha)) / params.tauo
        dq = (
            f * (1.0 - (1.0 - params.Eo) ** (1.0 / f)) / params.Eo
            - v ** (1.0 / params.alpha - 1.0) * q
        ) / params.tauo

        next_state = (
            s + dt_s * ds,
            f + dt_s * df,
            v + dt_s * dv,
            q + dt_s * dq,
        )
        return next_state, _bw_signal(next_state, params)

    return jax.lax.scan(step, state, drive)


def _bw_params(monitor):
    """Collect differentiable BW parameters for post-hoc and prepared paths."""
    return Bunch(
        taus=monitor.taus,
        tauf=monitor.tauf,
        tauo=monitor.tauo,
        alpha=monitor.alpha,
        Eo=monitor.Eo,
        vo=monitor.vo,
        TE=monitor.TE,
        k3=monitor.k3,
    )


def _bw_init(input_shape):
    """Build the resting four-vector carry for the live parameter dtype."""
    n_nodes = input_shape.shape[1]

    def init(params):
        dtype = jnp.result_type(input_shape.dtype, *jax.tree.leaves(params))
        return (
            jnp.zeros(n_nodes, dtype=dtype),
            jnp.ones(n_nodes, dtype=dtype),
            jnp.ones(n_nodes, dtype=dtype),
            jnp.ones(n_nodes, dtype=dtype),
        )

    return init


def _bw_update(indices, repeat, decimate, save_every, dt_bw):
    """Build BW observation of one aligned block with a four-vector carry."""

    def update(state, block, params):
        drive = block[:, indices, :].squeeze(axis=1)
        drive = _resample_bw_input(drive, repeat, decimate)
        state, bold = _integrate_bw(state, drive, params, dt_bw)
        return state, bold[save_every - 1 :: save_every, None, :]

    return update


class BalloonWindkesselBold(AbstractMonitor):
    """BOLD signal monitor using Balloon-Windkessel hemodynamic ODE.

    Integrates a four-variable ODE system (vasodilatory signal, blood flow,
    blood volume, deoxyhemoglobin) driven by neural firing rates, then
    computes BOLD signal from the hemodynamic state.

    The user-facing interface uses milliseconds for time parameters (period,
    dt_bw). Internally the BW ODE is integrated in seconds, matching the
    standard reference implementation (Friston 2000, Deco 2014).

    The monitor supports both stored solutions and native ``observe=``
    execution. Pass ``downsample=Identity()`` to drive it directly on the actual
    incoming grid, including the output grid of a shared joint stage.
    Post-hoc execution accepts any callable downsampler. Prepared execution
    accepts downsamplers implementing the streaming preparation contract,
    carries their state between blocks, and exposes their live parameters
    under ``config.monitor.downsample``.

    Parameters (user-facing, in ms)
    --------------------------------
    period : float
        BOLD sampling period (TR) in ms (default: 2000.0)
    dt_bw : float
        Integration time step for BW equations in ms (default: 1.0)

    Parameters (hemodynamic, in seconds)
    -------------------------------------
    taus : float
        Vasodilatory signal decay time constant in s (default: 0.65)
    tauf : float
        Autoregulatory feedback time constant in s (default: 0.41)
    tauo : float
        Transit time in s (default: 0.98)
    alpha : float
        Grubb's vessel stiffness exponent (default: 0.32)
    Eo : float
        Resting oxygen extraction fraction (default: 0.4)
    vo : float
        Resting blood volume fraction (default: 0.04)
    TE : float
        Echo time in s (default: 0.04)
    k3 : float
        Extravascular signal coefficient (default: 1.0). Keyword-only.

    Notes
    -----
    ``k1`` and ``k2`` remain readable attributes, but are derived from the live
    ``Eo`` and ``TE`` values instead of independent parameter leaves. Code that
    previously replaced those derived leaves should update ``Eo`` or ``TE``;
    ``k3`` remains an independently configurable signal coefficient.

    References
    ----------
    - Friston et al. (2000). Nonlinear Responses in fMRI: The Balloon Model,
      Volterra Kernels, and Other Hemodynamics. NeuroImage, 12(4), 466-477.
    - Deco et al. (2014). How Local Excitation-Inhibition Ratio Impacts the
      Whole Brain Dynamics. Journal of Neuroscience, 34(23), 7886-7898.
    """

    period: float = eqx.field(static=True)
    dt_bw: float = eqx.field(static=True)

    # Hemodynamic parameters (in seconds)
    taus: float
    tauf: float
    tauo: float
    alpha: float
    Eo: float
    vo: float

    # BOLD signal parameters. k1 and k2 are derived properties so changing Eo
    # or TE cannot leave the signal coefficients stale.
    TE: float
    k3: float

    # Optional downsampling before BW integration
    downsample: eqx.Module = eqx.field(static=True)

    def __init__(
        self,
        period=2000.0,
        dt_bw=1.0,
        taus=0.65,
        tauf=0.41,
        tauo=0.98,
        alpha=0.32,
        Eo=0.4,
        vo=0.04,
        TE=0.04,
        voi=None,
        downsample=None,
        *,
        k3=1.0,
    ):
        self.voi = self._normalize_voi(voi)
        self.period = period
        self.dt_bw = dt_bw

        self.taus = taus
        self.tauf = tauf
        self.tauo = tauo
        self.alpha = alpha
        self.Eo = Eo
        self.vo = vo
        self.TE = TE
        self.k3 = k3

        self.downsample = downsample

    @property
    def k1(self):
        """Oxygenation coefficient derived from the live Eo and TE values."""
        return 4.3 * 40.3 * self.Eo * self.TE

    @property
    def k2(self):
        """Intravascular coefficient derived from the live Eo and TE values."""
        return 25.0 * self.Eo * self.TE

    def prepare(self, grid, sample, variable_names):
        """Prepare stateful BW integration for block-wise execution."""
        return _prepare_balloon_windkessel(self, grid, sample, variable_names)

    def __call__(self, sol, t_offset=0.0):
        """Apply Balloon-Windkessel BOLD model to simulation results.

        Parameters
        ----------
        sol : NativeSolution
            Simulation result with .ys [T, n_voi, N], .ts (ms), .dt (ms).
            The selected variable of interest should contain firing rates
            in Hz.
        t_offset : float
            Time offset added to output timestamps in ms (default: 0.0)

        Returns
        -------
        NativeSolution
            BOLD signal with shape [T_bold, 1, N], timestamps in ms.
        """
        if self.downsample is not None:
            # Post-hoc execution accepts any callable input monitor.
            raw_dt = self._resolve_dt(sol)
            _integer_stride(self.period, raw_dt, label="BOLD period")
            t0 = sol.ts[0] - raw_dt if sol.ts.shape[0] else 0.0
            downsampled = _apply_downsample(self.downsample, sol)
            input_dt = self._resolve_dt(downsampled)
            sample = jax.ShapeDtypeStruct(
                downsampled.ys.shape[1:], downsampled.ys.dtype
            )
            prepared = _prepare_bw_drive(
                self, input_dt, sample, getattr(downsampled, "variable_names", None)
            )
            grid = SimulationGrid(t0, input_dt, downsampled.ys.shape[0])
            validate_prepared_observation(self, grid, sample, prepared)
            state = prepared.init(prepared.params)
            _, values = prepared.update(state, downsampled.ys, prepared.params)
            return observation_result(values, t0, prepared, t_offset=t_offset)
        return apply_observation(self, sol, t_offset=t_offset)


def _prepare_balloon_windkessel(
    monitor,
    grid: SimulationGrid,
    sample: jax.ShapeDtypeStruct,
    variable_names: tuple | None,
) -> PreparedObservation:
    """Prepare stateful BW integration for aligned native-solver blocks."""
    _validate_monitor_input(sample, variable_names)
    _integer_stride(monitor.period, grid.dt, label="BOLD period")

    prepared_downsample = None
    alignment_steps = None
    if monitor.downsample is None:
        input_dt = grid.dt
        input_shape = sample
        input_names = variable_names
    else:
        prepared_downsample = prepare_observation(
            monitor.downsample, grid, sample, variable_names
        )
        if not isinstance(prepared_downsample.output, ObservationOutput):
            raise ValueError(
                "BalloonWindkesselBold downsample must produce one time series"
            )
        downsample_output = prepared_downsample.output
        input_dt = downsample_output.period
        downsample_alignment = observation_alignment_steps(prepared_downsample, grid)
        input_shape = prepared_single_output_sample(
            prepared_downsample, sample, downsample_alignment
        )
        input_names = downsample_output.variable_names
        alignment_steps = math.lcm(
            downsample_alignment,
            _integer_stride(monitor.period, grid.dt, label="BOLD period"),
        )

    return _prepare_bw_drive(
        monitor,
        input_dt,
        input_shape,
        input_names,
        downsample=prepared_downsample,
        alignment_steps=alignment_steps,
    )


def _prepare_bw_drive(
    monitor,
    input_dt,
    input_shape,
    input_names,
    *,
    downsample=None,
    alignment_steps=None,
):
    """Prepare BW numerics on the resolved drive grid, shared by both paths."""
    _validate_monitor_input(input_shape, input_names)

    indices, names = _resolve_selection(monitor.voi, input_shape.shape[0], input_names)
    if len(indices) != 1:
        raise ValueError(
            "BalloonWindkesselBold requires exactly one input channel after "
            f"downsampling; selected {len(indices)}"
        )

    _integer_stride(monitor.period, input_dt, label="BOLD period")
    repeat, decimate = _bw_resampling_factors(input_dt, monitor.dt_bw)
    save_every = _integer_stride(monitor.period, monitor.dt_bw, label="BOLD period")
    prepared = PreparedObservation(
        params=_bw_params(monitor),
        init=_bw_init(input_shape),
        update=_bw_update(
            jnp.asarray(indices, dtype=int),
            repeat,
            decimate,
            save_every,
            float(monitor.dt_bw),
        ),
        output=ObservationOutput(
            period=float(monitor.period),
            first_sample_offset=float(monitor.period),
            variable_names=_bold_names(names),
        ),
        alignment_steps=alignment_steps,
    )
    return _with_downsample(prepared, downsample)


def Bold(*args, **kwargs):
    """Deprecated: use HRFBold or BalloonWindkesselBold explicitly."""
    warnings.warn(
        "Bold is deprecated and will be removed in a future version. "
        "Use HRFBold (HRF convolution) or BalloonWindkesselBold (ODE integration) explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return HRFBold(*args, **kwargs)
