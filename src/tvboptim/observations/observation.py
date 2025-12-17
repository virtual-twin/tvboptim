import jax.numpy as jnp


def compute_fc(timeseries, s_var=0, mode=0, skip_t=0):
    """Compute functional connectivity matrix from timeseries data.

    Args:
        timeseries: Timeseries data with .data attribute
        s_var: State variable index (default: 0)
        mode: Mode index for 4D data (default: 0)
        skip_t: Number of initial timepoints to skip (default: 0)

    Returns:
        Functional connectivity matrix with diagonal set to 0
    """
    if hasattr(timeseries, "data"):
        timeseries = timeseries.data
    if timeseries.ndim < 3:
        _fc = jnp.corrcoef(timeseries[skip_t:, :], rowvar=False)
    elif timeseries.ndim < 4:
        _fc = jnp.corrcoef(timeseries[skip_t:, s_var, :], rowvar=False)
    else:
        _fc = jnp.corrcoef(timeseries[skip_t:, s_var, :, mode], rowvar=False)
    # diag elements 0
    return _fc.at[jnp.diag_indices(_fc.shape[0])].set(0)


def fc_corr(fc1, fc2):
    return jnp.corrcoef(fc1.flatten(), fc2.flatten())[0, 1]


def rmse(matrix1, matrix2, axis=None):
    rmse_value = jnp.sqrt(jnp.square(matrix1 - matrix2).mean(axis=axis))
    return rmse_value
