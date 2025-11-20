import jax
import jax.numpy as jnp
import equinox as eqx
import jax.scipy as jsp
import matplotlib.pyplot as plt
import math

from tvboptim.experimental.network_dynamics.result import NativeSolution

from .downsampling import AbstractMonitor, TemporalAverage


class HRFKernel(eqx.Module):
    """Base class for hemodynamic response function kernels.

    A kernel is a function of time that defines the hemodynamic response.
    Subclasses must implement the kernel computation and specify its duration.

    Attributes:
        duration: Duration of kernel support in milliseconds
    """
    duration: float

    def __call__(self, t: jax.Array, downsample_dt: float) -> jax.Array:
        """Compute kernel values at time points.

        Args:
            t: Time points at which to evaluate the kernel
            downsample_dt: Time step of downsampled signal (for internal use)

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
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Kernel value')
        ax.set_title(f'{self.__class__.__name__}')
        ax.grid(True, alpha=0.3)

        return ax


class LotkaVolterraHRFKernel(HRFKernel):
    """Canonical hemodynamic response function based on Lotka-Volterra dynamics.

    This implements the oscillatory HRF kernel used in standard BOLD signal modeling.

    Attributes:
        tau_s: Signal decay time constant in seconds (default: 0.8 s)
        tau_f: Feedback time constant in seconds (default: 0.4 s)
        scaling: Kernel amplitude scaling factor (default: 1/3)
        duration: Kernel support duration in ms (default: 20,000 ms = 20 s)

    Note:
        The tau parameters are in seconds (not ms) to match the standard HRF formulation.
        Time input to __call__ is expected in milliseconds and converted internally.
    """
    tau_s: float = 0.8  # seconds
    tau_f: float = 0.4  # seconds
    scaling: float = 1.0 / 3.0
    duration: float = 20_000.0  # ms (20 seconds)

    def __init__(self, tau_s=0.8, tau_f=0.4, scaling=1.0/3.0, duration=20_000.0):
        """Initialize Lotka-Volterra HRF kernel.

        Args:
            tau_s: Signal decay time constant in seconds (default: 0.8 s)
            tau_f: Feedback time constant in seconds (default: 0.4 s)
            scaling: Kernel amplitude scaling factor (default: 1/3)
            duration: Kernel support duration in ms (default: 20,000 ms)
        """
        self.tau_s = tau_s
        self.tau_f = tau_f
        self.scaling = scaling
        self.duration = duration

    def __call__(self, t: jax.Array, downsample_dt: float) -> jax.Array:
        """Compute Lotka-Volterra HRF kernel.

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


class Bold(AbstractMonitor):
    """BOLD signal monitor using hemodynamic response function convolution.

    This monitor simulates the Blood Oxygen Level Dependent (BOLD) signal by:
    1. Downsampling the neural activity
    2. Convolving with a hemodynamic response function kernel
    3. Downsampling to the final BOLD sampling period
    """
    # BOLD model parameters
    k_1: float = 5.6           # Signal scaling factor
    V_0: float = 0.02          # Resting blood volume fraction

    # Sampling parameters
    period: float = 1000.0     # ms, final BOLD sampling period
    downsample_period: float = 4.0  # ms, intermediate downsampling period

    # Processing configuration
    kernel: HRFKernel = eqx.field(static=True)
    downsample: eqx.Module = eqx.field(static=True)
    convolution_mode: str = eqx.field(static=True)

    # History buffer for continuous monitoring
    history: jax.Array = None

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
            kernel: HRF kernel to use (default: LotkaVolterraHRFKernel())
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
            self.kernel = LotkaVolterraHRFKernel()
        else:
            self.kernel = kernel

        # Set up downsampling
        if downsample is None:
            # Pass the already normalized voi to the downsampler
            self.downsample = TemporalAverage(voi=self.voi, period=downsample_period)
        else:
            self.downsample = downsample

        # Process history buffer
        self.history = self._process_history(history)

    def _process_history(self, history):
        """Process history input into standardized buffer.

        Args:
            history: None (default to zeros), jax.Array, or Solution object (Native or Diffrax)

        Returns:
            Processed history array or None
        """
        if history is None:
            return None
        elif hasattr(history, 'ys') and hasattr(history, 'ts'):
            # Duck typing: any solution-like object with .ys and .ts attributes
            # Works with both NativeSolution and Diffrax solutions
            # Extract the required history length based on kernel duration
            # Use Python int() and math.ceil() to keep concrete during JIT
            n_samples = int(math.ceil(self.kernel.duration / self.downsample_period))
            # Downsample the history first
            downsampled_history = self.downsample(history)
            # Take the last n_samples
            return downsampled_history.ys[-n_samples:]
        else:
            # Assume it's already a jax.Array
            return history

    def __call__(self, sol, t_offset=0.0):
        """Apply BOLD monitor to simulation results.

        Args:
            sol: Simulation solution with .ys, .ts, and .dt attributes
                 Works with NativeSolution (requires dt as auxiliary data)
            t_offset: Time offset to add to output timestamps (default: 0.0)

        Returns:
            NativeSolution with BOLD signal timeseries
        """
        ts = sol.ts
        dt = sol.dt  # Use dt from auxiliary data

        # --- Downsample neural activity ---
        downsampled = self.downsample(sol)
        ys_downsampled = downsampled.ys

        # --- Create HRF kernel ---
        # Compute kernel sample points using Python int() and math.ceil()
        kernel_samples = int(math.ceil(self.kernel.duration / self.downsample_period))
        kernel_time = jnp.linspace(0.0, self.kernel.duration, kernel_samples)
        hrf = self.kernel(kernel_time, self.downsample_period)

        # --- Prepare signal with history buffer ---
        if self.history is None:
            # Prepend zeros for warm-up
            ys_with_history = jnp.vstack([
                jnp.zeros((kernel_samples, *ys_downsampled.shape[1:])),
                ys_downsampled
            ])
        else:
            # Use provided history
            ys_with_history = jnp.vstack([self.history, ys_downsampled])

        # --- Convolution with HRF ---
        def convolve_single(x):
            return jsp.signal.fftconvolve(x, hrf, mode=self.convolution_mode)

        # Vectorized convolution over nodes and state variables
        bold = jax.vmap(
            lambda y: jax.vmap(convolve_single, in_axes=1, out_axes=1)(y),
            in_axes=1, out_axes=1
        )(ys_with_history)

        # Apply BOLD scaling
        bold = self.k_1 * self.V_0 * (bold - 1.0)

        # --- Final downsampling to BOLD sampling period ---
        # Compute index step for final sampling using Python int() and round()
        final_samples_per_period = self.period / self.downsample_period
        final_idx_step = int(round(final_samples_per_period))

        # Sample at the specified period
        bold_indices = jnp.arange(0, bold.shape[0], final_idx_step)
        bold_signal = bold[bold_indices, ...]

        # Create time points for BOLD signal using Python int() and round()
        bold_time = ts[::int(round(self.period / dt))] + t_offset

        # Ensure time and signal arrays match in length
        min_len = min(len(bold_time), len(bold_signal))
        bold_time = bold_time[:min_len]
        bold_signal = bold_signal[:min_len]

        return NativeSolution(ts=bold_time, ys=bold_signal, dt=self.period)
