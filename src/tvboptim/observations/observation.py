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


def welford_cov(s_var=0):
    """Online functional-connectivity reducer for the network solver.

    Returns a ``(init, update, finalize)`` triple for the ``reduce=`` kwarg of
    ``prepare`` / ``solve``. It maintains a running mean and co-moment of the
    chosen state variable's region time series via a block-wise Welford / Chan
    merge (one batched ``X^T X`` per block, not a per-step rank-1 update), so
    the accumulator is ``O(N^2)`` in the region count ``N`` and independent of
    ``n_steps``. ``finalize`` returns the correlation matrix with a zeroed
    diagonal, matching ``compute_fc(result, s_var=s_var)`` on the full
    trajectory (the equivalence reference).

    Args:
        s_var: Index into the variables-of-interest axis (axis 1 of the stacked
            trajectory), matching ``compute_fc``'s ``s_var``.
    """

    def init(template, n_steps):
        # template is one step's output [n_vois, n_nodes]; size from the
        # region (node) axis. n_steps is unused (the state is O(1) in time).
        n = template.shape[-1]
        return (jnp.array(0.0), jnp.zeros(n), jnp.zeros((n, n)))

    def update(acc, block):
        # block is [block_len, n_vois, n_nodes]; pick the chosen variable's
        # region series and merge its block mean / co-moment into the running
        # accumulator (Chan's parallel formula, exact up to float error).
        count, mean, comoment = acc
        x = block[:, s_var, :]
        nb = x.shape[0]
        mean_b = jnp.mean(x, axis=0)
        xc = x - mean_b
        comoment_b = xc.T @ xc
        delta = mean_b - mean
        new_count = count + nb
        new_mean = mean + delta * (nb / new_count)
        new_comoment = comoment + comoment_b + jnp.outer(delta, delta) * (
            count * nb / new_count
        )
        return (new_count, new_mean, new_comoment)

    def finalize(acc):
        count, _mean, comoment = acc
        cov = comoment / count
        d = jnp.sqrt(jnp.diag(cov))
        corr = cov / jnp.outer(d, d)
        return corr.at[jnp.diag_indices(corr.shape[0])].set(0.0)

    return (init, update, finalize)


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
