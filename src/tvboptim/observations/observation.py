import jax
import jax.numpy as jnp


def _as_2d(timeseries, s_var=0, mode=0, skip_t=0):
    """Reduce a monitor result (or raw array) to a 2D [time, regions] array.

    Mirrors the dispatch in `compute_fc`: accepts NativeSolution/DiffraxSolution
    (via `.data` → `.ys`), or a raw 2D/3D/4D array; selects `s_var` and `mode`
    where present; drops the first `skip_t` samples.
    """
    if hasattr(timeseries, "data"):
        timeseries = timeseries.data
    if timeseries.ndim < 3:
        return timeseries[skip_t:, :]
    if timeseries.ndim < 4:
        return timeseries[skip_t:, s_var, :]
    return timeseries[skip_t:, s_var, :, mode]


def compute_fc(timeseries, s_var=0, mode=0, skip_t=0):
    """Compute functional connectivity matrix from timeseries data.

    Args:
        timeseries: Monitor result (with `.data`) or raw array.
        s_var: State variable index (default: 0)
        mode: Mode index for 4D data (default: 0)
        skip_t: Number of initial timepoints to skip (default: 0)

    Returns:
        Functional connectivity matrix with diagonal set to 0
    """
    bold = _as_2d(timeseries, s_var=s_var, mode=mode, skip_t=skip_t)
    _fc = jnp.corrcoef(bold, rowvar=False)
    return _fc.at[jnp.diag_indices(_fc.shape[0])].set(0)


def compute_fcd(timeseries, t_window, step_size, s_var=0, mode=0, skip_t=0):
    """Compute the functional connectivity dynamics (FCD) matrix.

    Slides a window over the timeseries, computes per-window FC vectors from
    the upper-triangle off-diagonal, and correlates them into the FCD matrix.

    Args:
        timeseries: Monitor result (with `.data`) or raw array.
        t_window: Window length in samples.
        step_size: Window step in samples.
        s_var: State variable index for 3D/4D inputs (default: 0).
        mode: Mode index for 4D inputs (default: 0).
        skip_t: Number of initial timepoints to skip (default: 0).

    Returns:
        Tuple ``(fcd, fcs)`` where ``fcd`` is the [n_windows, n_windows]
        correlation between per-window FC vectors and ``fcs`` is the
        [n_windows, n_regions, n_regions] tensor of per-window FC matrices.
    """
    bold = _as_2d(timeseries, s_var=s_var, mode=mode, skip_t=skip_t)
    n_samples, n_regions = bold.shape
    start_idx = jnp.arange(0, n_samples - t_window + 1, step_size)

    def window_fc(start_time):
        window = jax.lax.dynamic_slice_in_dim(bold, start_time, t_window, axis=0)
        return jnp.corrcoef(window, rowvar=False)

    fcs = jax.vmap(window_fc)(start_idx)
    iu = jnp.triu_indices(n_regions, k=1)
    fc_flat = fcs[:, iu[0], iu[1]]
    fcd = jnp.corrcoef(fc_flat)
    return fcd, fcs


_DEFAULT_FCD_MIDPOINTS = jnp.linspace(-0.99, 0.99, 100)


def fcd_distribution(fcd, midpoints=None, n_diag=1, bw_method=None, normalize=True):
    """KDE density of the FCD off-diagonal values, evaluated on `midpoints`.

    Args:
        fcd: FCD matrix [n_windows, n_windows].
        midpoints: Evaluation grid (default: 100 points on [-0.99, 0.99]).
        n_diag: Diagonal offset for upper-triangle extraction (default: 1,
            excludes the main diagonal).
        bw_method: Forwarded to `jax.scipy.stats.gaussian_kde` (default
            Scott's rule). Smaller values give sharper densities.
        normalize: Renormalise the evaluated density so it integrates to 1
            over `midpoints` (using the midpoint spacing). KDE on a finite
            grid never exactly integrates to 1; this fixes that.
    """
    if midpoints is None:
        midpoints = _DEFAULT_FCD_MIDPOINTS
    fcd_vals = fcd[jnp.triu_indices(fcd.shape[0], k=n_diag)]
    kde = jax.scipy.stats.gaussian_kde(fcd_vals, bw_method=bw_method)
    density = kde.evaluate(midpoints)
    if normalize:
        dx = midpoints[1] - midpoints[0]
        density = density / (jnp.sum(density) * dx)
    return density


def wasserstein_1d(p, q, x):
    """Wasserstein-1 distance between two 1D densities on a shared uniform grid.

    Both inputs are normalised internally (CDFs run 0..1), so `p` and `q` may
    be densities or unnormalised histograms; the returned value lives in the
    same units as `x`. Smooth and differentiable in both inputs, making it a
    good loss-function surrogate for the classical KS statistic.
    """
    dx = x[1] - x[0]
    p_cdf = jnp.cumsum(p)
    q_cdf = jnp.cumsum(q)
    p_cdf = p_cdf / p_cdf[-1]
    q_cdf = q_cdf / q_cdf[-1]
    return jnp.sum(jnp.abs(p_cdf - q_cdf)) * dx


def ks_distance(p, q):
    """Two-sample Kolmogorov-Smirnov statistic from histograms/densities.

    Normalises each cumulative sum to a proper CDF before taking the
    supremum, so the result lies in [0, 1] independent of bin width and is
    directly comparable to literature values.
    """
    p_cdf = jnp.cumsum(p)
    q_cdf = jnp.cumsum(q)
    p_cdf = p_cdf / p_cdf[-1]
    q_cdf = q_cdf / q_cdf[-1]
    return jnp.max(jnp.abs(p_cdf - q_cdf))


def fc_corr(fc1, fc2):
    return jnp.corrcoef(fc1.flatten(), fc2.flatten())[0, 1]


def rmse(matrix1, matrix2, axis=None):
    rmse_value = jnp.sqrt(jnp.square(matrix1 - matrix2).mean(axis=axis))
    return rmse_value
