"""Local identifiability analysis for tvboptim parameter estimates.

The curvature of the loss around a fitted parameter point tells you which
parameter combinations the data actually constrains. A near-zero eigenvalue
of the loss Hessian (equivalently, of the Fisher information) marks a
*sloppy* direction: a combination of parameters that can be changed without
moving the loss -- i.e. a practically non-identifiable direction.

Two curvature objects are available:

* the **Hessian of the scalar loss** (:func:`loss_hessian`) -- needs only
  the loss you already optimize;
* the **Gauss-Newton Fisher information** ``J^T J / sigma**2``
  (:func:`fisher_information`) -- needs a vector-valued model output (the
  residual / prediction before it is reduced to a scalar). It is always
  positive semi-definite and is the object the Cramer-Rao bound refers to.

At a good fit the two nearly coincide. Either way the curvature is taken in
the same coordinates the optimizer works in -- the flattened ``Parameter``
leaves of a config -- so transformed parameters (``LogPositiveParameter``,
``NormalizedParameter``, ...) are handled automatically: the spectrum comes
out in their stored (log / normalized) coordinates.

References
----------
Gutenkunst et al. 2007, *PLoS Comput Biol* -- "Universally sloppy parameter
sensitivities in systems biology models" (the eigenspectrum / sloppiness
framework).
Raue et al. 2009, *Bioinformatics* -- profile likelihood and practical
identifiability.

Notes
-----
The Hessian eigenspectrum is a *local* diagnostic: it describes the basin
around the supplied point only. It reliably flags flat ridges (practical
non-identifiability), but it does **not** see disconnected alternative
optima (multimodality) -- use profile likelihood or posterior sampling for
that.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from tvboptim.types.parameter import is_parameter
from tvboptim.types.stateutils import combine_state, partition_state

_Kind = Literal["hessian", "fisher"]

__all__ = [
    "IdentifiabilityResult",
    "analyze_identifiability",
    "eigendecompose_curvature",
    "fisher_information",
    "loss_hessian",
]

_KIND_LABELS = {"hessian": "loss Hessian", "fisher": "Fisher information"}


# --------------------------------------------------------------------------
# Parameter-path labelling
# --------------------------------------------------------------------------
def _key_str(key) -> str:
    """Render a single JAX pytree key as a plain string."""
    for attr in ("key", "name", "idx"):
        if hasattr(key, attr):
            return str(getattr(key, attr))
    return str(key)


def _path_str(path) -> str:
    """Join a JAX pytree key path into a dotted string."""
    return ".".join(_key_str(k) for k in path)


def _parameter_labels(diff_state) -> list[str]:
    """One human-readable label per scalar entry of the flattened diff_state.

    The order matches ``ravel_pytree(diff_state)``: parameters are visited in
    pytree-traversal order, and an array-valued ``Parameter`` of size m
    expands to m consecutive ``name[i]`` labels.
    """
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(
        diff_state, is_leaf=is_parameter
    )
    labels: list[str] = []
    for path, param in leaves_with_path:
        base = _path_str(path) or "param"
        if param.size == 1:
            labels.append(base)
        else:
            labels.extend(f"{base}[{i}]" for i in range(param.size))
    return labels


def _flatten_state(state):
    """Partition + ravel the Parameter leaves of a state.

    Returns the pieces needed to plug callables into a flat-theta interface,
    without committing to which callable will be wrapped. Use this when more
    than one callable (e.g. both ``loss`` and ``model``) needs to share the
    same flatten.

    Returns
    -------
    theta0 : jnp.ndarray
        The flattened current parameter values.
    unravel : callable
        Inverse of the flattening, reconstructing the diff_state pytree.
    static_state : pytree
        The non-Parameter pieces of ``state``.
    labels : list[str]
        Parameter name per entry of ``theta0``.
    """
    diff_state, static_state = partition_state(state)
    theta0, unravel = ravel_pytree(diff_state)

    if theta0.size == 0:
        raise ValueError(
            "No Parameter leaves found in `state`. Wrap the quantities you "
            "want analysed in tvboptim.types.Parameter(...) before calling "
            "an identifiability function."
        )

    labels = _parameter_labels(diff_state)
    if len(labels) != theta0.size:
        # Should not happen; guards against a label/flatten ordering drift.
        raise RuntimeError(
            f"Internal error: {len(labels)} labels for {theta0.size} "
            "parameters. Please report this."
        )
    return theta0, unravel, static_state, labels


def _flatten_problem(fn, state):
    """Convenience: flatten ``state`` and wrap ``fn`` to take a flat theta."""
    theta0, unravel, static_state, labels = _flatten_state(state)

    def fn_flat(theta):
        return fn(combine_state(unravel(theta), static_state))

    return fn_flat, theta0, unravel, labels


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass(eq=False)
class IdentifiabilityResult:
    """Eigendecomposition of a loss-curvature matrix at a parameter point.

    Attributes
    ----------
    eigenvalues : jnp.ndarray
        Eigenvalues in ascending order. Small (near-zero) eigenvalues are
        sloppy / non-identifiable directions; large eigenvalues are stiff /
        well-constrained ones.
    eigenvectors : jnp.ndarray
        Eigenvectors as columns; ``eigenvectors[:, i]`` belongs to
        ``eigenvalues[i]``. Each column is a unit-norm combination of the
        parameters in ``labels``.
    matrix : jnp.ndarray
        The curvature matrix that was decomposed (loss Hessian or Fisher
        information).
    labels : list[str]
        Parameter name per row/column of ``matrix``.
    theta : jnp.ndarray
        The flattened parameter point the curvature was evaluated at.
    gradient_norm : float or None
        L2 norm of the loss gradient at ``theta``. Only meaningful as a
        check that ``theta`` is a genuine optimum.
    kind : {"hessian", "fisher"}
        Which curvature object ``matrix`` is.
    """

    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray
    matrix: jnp.ndarray
    labels: list
    theta: jnp.ndarray
    gradient_norm: float | None = None
    kind: _Kind = "hessian"

    @property
    def n_params(self) -> int:
        return len(self.labels)

    def condition_number(self) -> float:
        """Ratio of largest to smallest |eigenvalue|. Large => ill-posed."""
        a = jnp.abs(self.eigenvalues)
        lo = float(a.min())
        hi = float(a.max())
        return float("inf") if lo == 0.0 else hi / lo

    def rank(self, rtol: float = 1e-8) -> int:
        """Number of eigenvalues above ``rtol * max|eigenvalue|``.

        A rank below ``n_params`` means that many parameter combinations are
        practically non-identifiable at this point.
        """
        a = jnp.abs(self.eigenvalues)
        return int(jnp.sum(a > rtol * float(a.max())))

    def _direction(self, idx: int) -> dict:
        vec = np.asarray(self.eigenvectors[:, idx])
        order = np.argsort(-np.abs(vec))
        return {
            "eigenvalue": float(self.eigenvalues[idx]),
            "loadings": {self.labels[j]: float(vec[j]) for j in order},
        }

    def sloppy_directions(self, n: int = 3) -> list[dict]:
        """The ``n`` flattest directions (smallest eigenvalues first).

        Each entry is ``{"eigenvalue": float, "loadings": {label: coeff}}``
        with loadings sorted by descending |coeff|.
        """
        n = min(n, self.n_params)
        return [self._direction(i) for i in range(n)]

    def stiff_directions(self, n: int = 3) -> list[dict]:
        """The ``n`` stiffest directions (largest eigenvalues first)."""
        n = min(n, self.n_params)
        return [self._direction(self.n_params - 1 - i) for i in range(n)]

    def summary(self) -> str:
        """Human-readable one-screen report."""
        label = _KIND_LABELS.get(self.kind, self.kind)
        lines = [
            f"Identifiability analysis  --  {label}  "
            f"({self.n_params} parameters)",
            "-" * 60,
        ]
        if self.gradient_norm is not None:
            lines.append(f"gradient norm at point    : {self.gradient_norm:.3e}")
        lines += [
            f"eigenvalue range          : {float(self.eigenvalues.min()):.3e}"
            f"  ...  {float(self.eigenvalues.max()):.3e}",
            f"condition number          : {self.condition_number():.3e}",
            f"numerical rank (rtol=1e-8): {self.rank()} / {self.n_params}",
            "",
            "Flattest (least identifiable) direction:",
        ]
        d = self.sloppy_directions(1)[0]
        lines.append(f"  eigenvalue = {d['eigenvalue']:.3e}")
        for lab, coeff in list(d["loadings"].items())[:5]:
            lines.append(f"    {lab:<28s} {coeff:+.4f}")
        return "\n".join(lines)

    def plot_spectrum(self, ax=None):
        """Log-scale plot of |eigenvalues|, largest to smallest."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ev = np.sort(np.abs(np.asarray(self.eigenvalues)))[::-1]
        ax.semilogy(np.arange(1, len(ev) + 1), ev, "o-")
        ax.set_xlabel("eigenvalue index (stiff -> sloppy)")
        ax.set_ylabel("|eigenvalue|")
        ax.set_title(f"{_KIND_LABELS.get(self.kind, self.kind)} eigenspectrum")
        ax.grid(True, which="both", alpha=0.3)
        return ax

    def __repr__(self) -> str:
        return (
            f"IdentifiabilityResult(kind={self.kind!r}, "
            f"n_params={self.n_params}, "
            f"rank={self.rank()}, "
            f"condition_number={self.condition_number():.3e})"
        )


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def _hessian_at(loss_flat, theta0):
    """Symmetric dense Hessian of a flat-theta loss at ``theta0``."""
    H = jax.hessian(loss_flat)(theta0)
    return 0.5 * (H + H.T)  # symmetrize against floating-point asymmetry


def _fisher_at(model_flat, theta0, sigma, mode: str = "auto"):
    """Gauss-Newton Fisher information ``J^T J / sigma**2`` at ``theta0``.

    Internal helper: ``model_flat`` must already take a 1-D theta and return
    a JAX array (any shape). Used by both ``fisher_information`` and the
    Fisher path of ``analyze_identifiability``.
    """
    n_params = theta0.size
    out_shape = jax.eval_shape(model_flat, theta0).shape
    n_obs = int(np.prod(out_shape, dtype=int))
    if n_obs <= 1:
        warnings.warn(
            "`model` returns a scalar (or length-1) output. The Fisher "
            "information will be rank-1 and uninformative -- pass the "
            "vector-valued model output (residuals / predictions before the "
            "reduction to a scalar loss).",
            stacklevel=3,
        )

    if mode == "auto":
        mode = "fwd" if n_params <= n_obs else "rev"
    if mode == "fwd":
        J = jax.jacfwd(model_flat)(theta0)
    elif mode == "rev":
        J = jax.jacrev(model_flat)(theta0)
    else:
        raise ValueError(f"Unknown mode {mode!r}, expected 'auto', 'fwd', 'rev'.")
    J = jnp.reshape(J, (n_obs, n_params))

    sig = jnp.asarray(sigma, dtype=J.dtype)
    if sig.ndim == 0:
        Jw = J / sig
    elif sig.shape == (n_obs,):
        Jw = J / sig[:, None]
    else:
        raise ValueError(
            f"sigma must be a scalar or a 1-D array of length n_obs={n_obs} "
            "(the flattened model output, in C-order — flatten yourself if "
            f"`model(state)` is multi-dimensional). Got shape {sig.shape}."
        )

    return Jw.T @ Jw


def loss_hessian(loss, state, *, check_gradient: bool = True):
    """Dense Hessian of a scalar loss w.r.t. the ``Parameter`` leaves of ``state``.

    Parameters
    ----------
    loss : callable
        ``loss(state) -> scalar``. Same contract as ``OptaxOptimizer``.
    state : pytree
        A config whose optimizable quantities are wrapped in ``Parameter``.
        Pass the *fitted* config to analyse a found estimate.
    check_gradient : bool, optional
        Also compute the loss-gradient norm at the point (default True). It
        is cheap and lets callers verify the point is a genuine optimum.

    Returns
    -------
    H : jnp.ndarray
        Symmetric ``(n_params, n_params)`` Hessian.
    theta0 : jnp.ndarray
        Flattened parameter values the Hessian was evaluated at.
    labels : list[str]
        Parameter name per row/column of ``H``.
    grad_norm : float or None
        L2 norm of the gradient at ``theta0`` (None if ``check_gradient`` is
        False).
    """
    loss_flat, theta0, _, labels = _flatten_problem(loss, state)
    H = _hessian_at(loss_flat, theta0)
    grad_norm = None
    if check_gradient:
        grad_norm = float(jnp.linalg.norm(jax.grad(loss_flat)(theta0)))
    return H, theta0, labels, grad_norm


def fisher_information(model, state, *, sigma=1.0, mode: str = "auto"):
    """Gauss-Newton Fisher information ``J^T J / sigma**2`` from a vector model.

    ``model`` returns the *vector-valued* output -- the residual or the raw
    prediction, before the reduction to a scalar loss. ``J`` is the Jacobian
    of that vector w.r.t. the ``Parameter`` leaves of ``state``; ``J^T J`` is
    identical whether ``model`` returns the residual or the prediction.

    Unlike the loss Hessian, this object is always positive semi-definite and
    drops the noise-dependent residual-curvature term, so it stays meaningful
    away from a perfect fit. It is the matrix whose inverse is the
    Cramer-Rao lower bound on the parameter covariance.

    Parameters
    ----------
    model : callable
        ``model(state) -> array``. The pre-reduction model output. May be
        multi-dimensional; it is flattened in C-order internally.
    state : pytree
        Config with optimizable quantities wrapped in ``Parameter``.
    sigma : float or array, optional
        Observation noise scale. Either a scalar, or a 1-D array of length
        ``prod(model(state).shape)`` giving per-observation noise scales for
        the model output flattened in C-order. If your model returns a
        multi-D output, flatten ``sigma`` yourself to match. Default 1.0.
    mode : {"auto", "fwd", "rev"}, optional
        Jacobian mode. ``"auto"`` picks forward-mode when there are at most
        as many parameters as observations, reverse-mode otherwise.

    Returns
    -------
    FIM : jnp.ndarray
        Symmetric ``(n_params, n_params)`` Fisher information matrix.
    theta0 : jnp.ndarray
        Flattened parameter values the FIM was evaluated at.
    labels : list[str]
        Parameter name per row/column of ``FIM``.
    """
    model_flat, theta0, _, labels = _flatten_problem(model, state)
    FIM = _fisher_at(model_flat, theta0, sigma, mode=mode)
    return FIM, theta0, labels


def eigendecompose_curvature(
    matrix, labels, theta0, *, kind: _Kind, gradient_norm=None
) -> IdentifiabilityResult:
    """Eigendecompose a (symmetric) curvature matrix into an IdentifiabilityResult.

    The matrix is assumed to be a loss Hessian or Fisher information matrix —
    i.e. a *curvature* of a loss / log-likelihood in parameter space. The name
    avoids the word "spectrum" because that overloads with frequency-domain
    spectra in time-series and imaging contexts.

    ``kind`` is required and not defaulted: passing a Fisher matrix with the
    default ``"hessian"`` (or vice versa) would silently mislabel results
    downstream in ``summary()`` / ``__repr__``.
    """
    if kind not in ("hessian", "fisher"):
        raise ValueError(
            f"`kind` must be 'hessian' or 'fisher', got {kind!r}."
        )
    matrix = jnp.asarray(matrix)
    matrix = 0.5 * (matrix + matrix.T)  # symmetrize against floating-point asymmetry
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return IdentifiabilityResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        matrix=matrix,
        labels=list(labels),
        theta=jnp.asarray(theta0),
        gradient_norm=gradient_norm,
        kind=kind,
    )


def analyze_identifiability(
    loss,
    state,
    *,
    model=None,
    sigma=None,
    check_gradient: bool = True,
    warn_gradient_tol: float = 1e-3,
) -> IdentifiabilityResult:
    """End-to-end local identifiability analysis from a loss and a config.

    Computes a curvature matrix at the parameter point encoded in ``state``,
    eigendecomposes it, and returns an :class:`IdentifiabilityResult`.

    With ``model=None`` (default) the curvature is the **Hessian of the
    scalar loss** -- nothing beyond the loss you already optimize is needed.
    Pass ``model`` to instead use the **Gauss-Newton Fisher information**
    ``J^T J / sigma**2`` of a vector-valued model output; that object is
    always positive semi-definite and is what the Cramer-Rao bound refers
    to. ``loss`` is still used either way for the at-an-optimum check.

    Parameters
    ----------
    loss : callable
        ``loss(state) -> scalar``. Used for the gradient / optimum check,
        and as the curvature source when ``model`` is None.
    state : pytree
        Fitted config with optimizable quantities wrapped in ``Parameter``.
    model : callable, optional
        ``model(state) -> array``: the pre-reduction model output. If given,
        the Fisher information is used instead of the loss Hessian.
    sigma : float or array, optional
        Observation noise scale for the Fisher information. Only meaningful
        with ``model`` set; passing a value while ``model`` is None emits a
        warning (the loss-Hessian path does not use it). Defaults to 1.0
        internally.
    check_gradient : bool, optional
        Compute the loss-gradient norm at the point (default True).
    warn_gradient_tol : float, optional
        If the gradient norm exceeds this, emit a warning -- the eigenspectrum
        is only an identifiability diagnostic *at an optimum*. Default 1e-3.

    Returns
    -------
    IdentifiabilityResult
    """
    if sigma is not None and model is None:
        warnings.warn(
            "`sigma` was passed but `model` is None, so the loss-Hessian "
            "path is used and `sigma` is ignored. Pass a vector-valued "
            "`model` to use the Fisher information.",
            stacklevel=2,
        )
    sigma_eff = 1.0 if sigma is None else sigma

    # Single flatten of state — shared between gradient check, loss-Hessian,
    # and Fisher-information paths.
    theta0, unravel, static_state, labels = _flatten_state(state)

    def loss_flat(theta):
        return loss(combine_state(unravel(theta), static_state))

    grad_norm = None
    if check_gradient:
        grad_norm = float(jnp.linalg.norm(jax.grad(loss_flat)(theta0)))
        if grad_norm > warn_gradient_tol:
            warnings.warn(
                f"Gradient norm at the supplied point is {grad_norm:.3e} "
                f"(> {warn_gradient_tol:.0e}). The eigenspectrum is an "
                "identifiability diagnostic only at a genuine optimum; away "
                "from one it mixes curvature with slope. Re-fit, or pass a "
                "larger warn_gradient_tol if this is intentional.",
                stacklevel=2,
            )

    if model is None:
        H = _hessian_at(loss_flat, theta0)
        return eigendecompose_curvature(
            H, labels, theta0, gradient_norm=grad_norm, kind="hessian"
        )

    def model_flat(theta):
        return model(combine_state(unravel(theta), static_state))

    FIM = _fisher_at(model_flat, theta0, sigma_eff)
    return eigendecompose_curvature(
        FIM, labels, theta0, gradient_norm=grad_norm, kind="fisher"
    )
