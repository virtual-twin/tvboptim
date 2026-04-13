import math
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

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
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Kernel value")
        ax.set_title(f"{self.__class__.__name__}")
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

    def __init__(self, tau_s=0.8, tau_f=0.4, scaling=1.0 / 3.0, duration=20_000.0):
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


class HRFBold(AbstractMonitor):
    """BOLD signal monitor using hemodynamic response function convolution.

    This monitor simulates the Blood Oxygen Level Dependent (BOLD) signal by:
    1. Downsampling the neural activity
    2. Convolving with a hemodynamic response function kernel
    3. Downsampling to the final BOLD sampling period
    """

    # BOLD model parameters
    k_1: float = 5.6  # Signal scaling factor
    V_0: float = 0.02  # Resting blood volume fraction

    # Sampling parameters
    period: float = 1000.0  # ms, final BOLD sampling period
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
            # Sync downsample_period with the actual monitor's period
            # so kernel sampling and final BOLD subsampling use the correct grid
            if hasattr(downsample, "period"):
                self.downsample_period = downsample.period

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
        elif hasattr(history, "ys") and hasattr(history, "ts"):
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
        dt = self._resolve_dt(sol)

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
            ys_with_history = jnp.vstack(
                [jnp.zeros((kernel_samples, *ys_downsampled.shape[1:])), ys_downsampled]
            )
        else:
            # Use provided history
            ys_with_history = jnp.vstack([self.history, ys_downsampled])

        # --- Convolution with HRF ---
        def convolve_single(x):
            return jsp.signal.fftconvolve(x, hrf, mode=self.convolution_mode)

        # Vectorized convolution over nodes and state variables
        bold = jax.vmap(
            lambda y: jax.vmap(convolve_single, in_axes=1, out_axes=1)(y),
            in_axes=1,
            out_axes=1,
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
        bold_time = ts[:: int(round(self.period / dt))] + t_offset

        # Ensure time and signal arrays match in length
        min_len = min(len(bold_time), len(bold_signal))
        bold_time = bold_time[:min_len]
        bold_signal = bold_signal[:min_len]

        return NativeSolution(ts=bold_time, ys=bold_signal, dt=self.period)


class BalloonWindkesselBold(AbstractMonitor):
    """BOLD signal monitor using Balloon-Windkessel hemodynamic ODE.

    Integrates a four-variable ODE system (vasodilatory signal, blood flow,
    blood volume, deoxyhemoglobin) driven by neural firing rates, then
    computes BOLD signal from the hemodynamic state.

    The user-facing interface uses milliseconds for time parameters (period,
    dt_bw). Internally the BW ODE is integrated in seconds, matching the
    standard reference implementation (Friston 2000, Deco 2014).

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

    # BOLD signal parameters
    k1: float
    k2: float
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

        self.k1 = 4.3 * 40.3 * Eo * TE
        self.k2 = 25.0 * Eo * TE
        self.k3 = 1.0

        self.downsample = downsample

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
            sol = self.downsample(sol)

        ys = sol.ys[:, self.voi, :]
        ys = ys.squeeze(axis=1)  # [T, N]
        input_dt = sol.dt

        steps_per_input = int(round(input_dt / self.dt_bw))
        if steps_per_input > 1:
            ys = jnp.repeat(ys, steps_per_input, axis=0)
        elif steps_per_input < 1:
            step = int(round(self.dt_bw / input_dt))
            ys = ys[::step]

        T, N = ys.shape
        save_every = int(round(self.period / self.dt_bw))

        dt_s = self.dt_bw / 1000.0

        itaus = 1.0 / self.taus
        itauf = 1.0 / self.tauf
        itauo = 1.0 / self.tauo
        ialpha = 1.0 / self.alpha
        Eo = self.Eo
        k1, k2, k3, vo = self.k1, self.k2, self.k3, self.vo

        def bw_step(state, r):
            s, f, v, q = state

            ds = r - itaus * s - itauf * (f - 1.0)
            df = s
            dv = itauo * (f - v**ialpha)
            dq = itauo * (
                f * (1.0 - (1.0 - Eo) ** (1.0 / f)) / Eo - v ** (ialpha - 1.0) * q
            )

            s = s + dt_s * ds
            f = f + dt_s * df
            v = v + dt_s * dv
            q = q + dt_s * dq

            bold = vo * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))
            return (s, f, v, q), bold

        init = (
            jnp.zeros(N),  # s: vasodilatory signal
            jnp.ones(N),  # f: blood flow
            jnp.ones(N),  # v: blood volume
            jnp.ones(N),  # q: deoxyhemoglobin
        )

        _, bold_all = jax.lax.scan(bw_step, init, ys)  # [T, N]

        bold_signal = bold_all[save_every - 1 :: save_every]  # [T_bold, N]
        bold_signal = bold_signal[:, jnp.newaxis, :]  # [T_bold, 1, N]

        n_bold = bold_signal.shape[0]
        bold_ts = jnp.arange(n_bold) * self.period + self.period + t_offset

        return NativeSolution(ts=bold_ts, ys=bold_signal, dt=self.period)


def Bold(*args, **kwargs):
    """Deprecated: use HRFBold or BalloonWindkesselBold explicitly."""
    warnings.warn(
        "Bold is deprecated and will be removed in a future version. "
        "Use HRFBold (HRF convolution) or BalloonWindkesselBold (ODE integration) explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return HRFBold(*args, **kwargs)
