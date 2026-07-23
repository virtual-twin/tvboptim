"""Solving system for network architecture.

This module provides the prepare-solve pattern for Network with multi-coupling
support. The prepare() function sets up the integration with all coupling state
management, and returns a pure function for execution.
"""

import copy
import warnings
from typing import Callable, Tuple

import diffrax
import jax
import jax.numpy as jnp
from plum import dispatch

from .core.bunch import Bunch
from .core.heterogeneous import HeterogeneousNetwork
from .core.network import Network
from .coupling.base import DelayedCoupling, InstantaneousCoupling
from .dynamics.base import AbstractDynamics
from .graph.base import delay_steps_bound, effective_max_delay
from .graph.topology import prepare_graph_topology, validate_graph_topology
from .result import DiffraxSolution, HeterogeneousSolution, wrap_native_result
from .solvers.diffrax import DiffraxSolver
from .solvers.native import BoundedSolver, NativeSolver
from .utils.history import extract_history_window


def _snapshot(tree):
    """Structurally copy a PyTree's containers, sharing leaves.

    Rebuilds every PyTree container (Bunch, dataclass, equinox Module, ...)
    so mutations to the returned tree don't leak back to ``tree``. Leaves
    (arrays, scalars, PRNG keys) are immutable in JAX, so sharing them is
    safe and avoids duplicating large arrays (graph weights, history
    buffers, pre-sampled noise tensors).
    """
    return jax.tree.map(lambda x: x, tree)


def _partition_jax_params(params):
    """Split a named parameter mapping into dynamic and static entries.

    Strings and other non-JAX values cannot be leaves of a jitted solve config.
    Keep such construction metadata in the prepared closure while exposing all
    array-compatible values as live config leaves.
    """
    dynamic = Bunch()
    static = Bunch()
    for name, value in params.items():
        try:
            jax.eval_shape(lambda item: item, value)
        except (TypeError, ValueError):
            static[name] = value
        else:
            dynamic[name] = value
    return dynamic, static


def _blocked_scan(runner, state0, scan_inputs, n_steps, block_size):
    """Split the leading axis into ``(n_blocks, block_size)`` plus a tail, scan
    ``runner`` over the main blocks, run ``runner`` once on the tail, and stitch
    the outputs back to leading shape ``(n_steps, ...)``.

    ``runner(state, block_inputs, block_len) -> (state, outs)`` is the only
    per-block behaviour that varies between callers (checkpointed inner scan,
    truncated window, per-block streaming/reduce); this helper owns the
    reshape/tail/stitch skeleton they all share. ``block_len`` is a static
    Python int (``block_size`` for the main blocks, the remainder for the tail).

    When ``runner`` emits ``None`` outputs (the reduce/fold case) the ``None``
    threads through ``jax.tree.map`` and the concatenate untouched, so the
    accumulator-in-carry path needs no special handling here.
    """
    n_blocks = n_steps // block_size
    remainder = n_steps - n_blocks * block_size

    if n_blocks > 0:
        main = jax.tree.map(
            lambda x: x[: n_blocks * block_size].reshape(
                (n_blocks, block_size) + x.shape[1:]
            ),
            scan_inputs,
        )
        state_mid, outs_main = jax.lax.scan(
            lambda s, b: runner(s, b, block_size), state0, main
        )
        outs_main_flat = jax.tree.map(
            lambda x: x.reshape((n_blocks * block_size,) + x.shape[2:]),
            outs_main,
        )
    else:
        state_mid = state0
        outs_main_flat = None

    if remainder == 0:
        return state_mid, outs_main_flat

    tail = jax.tree.map(lambda x: x[n_blocks * block_size :], scan_inputs)
    state_final, outs_tail = runner(state_mid, tail, remainder)

    if outs_main_flat is None:
        return state_final, outs_tail

    outs = jax.tree.map(
        lambda a, b: jnp.concatenate([a, b], axis=0), outs_main_flat, outs_tail
    )
    return state_final, outs


def _block_scan(op, state0, scan_inputs, n_steps, block_size):
    """Run ``op`` over ``scan_inputs`` as an outer-checkpointed nested scan.

    The leading axis of every leaf in ``scan_inputs`` is split into
    ``(n_blocks, block_size)``; an outer ``jax.lax.scan`` runs over blocks
    with each block wrapped in ``jax.checkpoint``, and an inner
    ``jax.lax.scan`` runs the ``block_size`` steps inside a block. When
    ``n_steps`` is not a multiple of ``block_size`` the remainder runs through
    a checkpointed tail scan; the tail length is fixed at trace time and is at
    most ``block_size - 1``.

    Output of the scanned computation is stitched back to leading shape
    ``(n_steps, ...)`` so the result is indistinguishable from a single
    ``jax.lax.scan(op, state0, scan_inputs)`` call to downstream code.
    """
    # One checkpointed inner scan serves both the main blocks and the tail, so
    # every block's activation tape is rematerialized on the backward pass
    # rather than held live across the whole rollout; only block-boundary
    # carries are retained. Wrapping the tail too keeps peak memory for
    # non-divisor block sizes from spiking above the no-checkpoint baseline.
    block = jax.checkpoint(
        lambda state, block_inputs: jax.lax.scan(op, state, block_inputs)
    )
    return _blocked_scan(
        lambda state, block_inputs, block_len: block(state, block_inputs),
        state0,
        scan_inputs,
        n_steps,
        block_size,
    )


def _truncated_scan(op, state0, scan_inputs, n_steps, window_size, block_size):
    """Run ``op`` over ``scan_inputs`` as a windowed scan with truncated BPTT.

    The leading axis of every leaf in ``scan_inputs`` is split into
    ``(n_windows, window_size)``; an outer ``jax.lax.scan`` runs over windows
    and the carry gradient is severed with ``jax.lax.stop_gradient`` at the
    entry to each window, so credit is assigned only within a window
    (truncated backpropagation through time). Within a window the steps run as
    a plain inner ``jax.lax.scan`` when ``block_size`` is None, or as a
    ``_block_scan`` over ``block_size`` blocks when it is set (the memory
    granularity nests inside the gradient window). When ``n_steps`` is not a
    multiple of ``window_size`` the remainder runs as a final shorter window
    with the same carry severing and inner runner.

    The forward computation is identical to a single ``jax.lax.scan(op, ...)``
    (``stop_gradient`` is the identity on the forward); only the backward pass
    is truncated. Output is stitched back to leading shape ``(n_steps, ...)`` so
    the result is indistinguishable from the untruncated scan to downstream code.
    """

    def run_window(state, window_inputs, window_len):
        # Sever cross-window credit: the window sees a detached start state, so
        # gradient flows within the window (and to op's closed-over parameters)
        # but not back to prior windows. Forward value is unchanged.
        state = jax.lax.stop_gradient(state)
        if block_size is None:
            return jax.lax.scan(op, state, window_inputs)
        return _block_scan(op, state, window_inputs, window_len, block_size)

    return _blocked_scan(run_window, state0, scan_inputs, n_steps, window_size)


def _split_voi(dynamics):
    """Split variables-of-interest into state and auxiliary index arrays.

    Returns ``(state_voi_indices, aux_voi_indices, record_auxiliaries,
    variable_names)``. ``variable_names`` labels axis 1 of the output
    trajectory: selected states first, then selected auxiliaries, matching
    the concatenation order in ``_assemble_output``.
    """
    voi_indices = dynamics.get_variables_of_interest_indices()
    n_states = dynamics.N_STATES
    state_voi_indices = jnp.array([i for i in voi_indices if i < n_states], dtype=int)
    aux_voi_indices = jnp.array(
        [i - n_states for i in voi_indices if i >= n_states], dtype=int
    )
    record_auxiliaries = len(aux_voi_indices) > 0
    all_variable_names = dynamics.all_variable_names
    variable_names = tuple(
        all_variable_names[i] for i in voi_indices if i < n_states
    ) + tuple(all_variable_names[i] for i in voi_indices if i >= n_states)
    return state_voi_indices, aux_voi_indices, record_auxiliaries, variable_names


def _materialize_noise(noise_key, injected, shape):
    """Draw the full per-call noise tensor, or pass through an injected override.

    A single fused ``jax.random.normal`` draw of shape ``[n_steps,
    n_noise_states, n_nodes]`` when no override is present (XLA fuses this with
    the downstream scan); otherwise the caller-supplied tensor verbatim (the
    NumPyro-over-increments replay path). Callers gate this on the network
    actually having noise.
    """
    if injected is None:
        return jax.random.normal(noise_key, shape)
    return injected


def _streaming_noise_gen(noise_key, per_node_shape, provider=None):
    """Per-block streaming noise generator for the block scan.

    Returns ``noise_gen(block_idx, block_len) -> [block_len, *per_node_shape]``.
    The default draws ``jax.random.normal(jax.random.fold_in(noise_key,
    block_idx), ...)``, so the realization is a pure function of
    ``(key, block_idx)`` and the block grain, regenerated (not stored) on the
    backward pass. ``per_node_shape`` is ``(n_noise_states, n_nodes)``. A
    ``provider`` (``config._internal.noise_provider``) overrides the draw for the
    streaming injection workflow.
    """
    if provider is not None:
        return lambda block_idx, block_len: provider(
            block_idx, noise_key, (block_len,) + per_node_shape
        )
    return lambda block_idx, block_len: jax.random.normal(
        jax.random.fold_in(noise_key, block_idx), (block_len,) + per_node_shape
    )


def _assemble_output(
    next_state, auxiliaries, state_voi_indices, aux_voi_indices, record_auxiliaries
):
    """Select one step's variables-of-interest slice of the output.

    Concatenates selected state variables and (optionally) selected
    auxiliaries along axis 0, matching the ordering of ``variable_names``
    from ``_split_voi``.
    """
    if len(state_voi_indices) > 0:
        selected_states = next_state[state_voi_indices]
    else:
        selected_states = jnp.array([]).reshape(0, next_state.shape[1])

    if record_auxiliaries and auxiliaries.size > 0:
        selected_aux = auxiliaries[aux_voi_indices]
        return jnp.concatenate([selected_states, selected_aux], axis=0)
    return selected_states


def _composed_scan(
    block_step, carry0, scan_inputs, n_steps, block_size, window_size=None
):
    """Run a block-wise scan with a pluggable per-block ``block_step``.

    ``block_step(carry, block_inputs) -> (carry, outs)`` owns the per-block work
    (inner step scan, optional noise generation, stack-or-fold of the output);
    it carries whatever state it needs (e.g. ``(state, acc)`` or
    ``(state, counter)``). This helper owns only the composition: blocks of
    ``block_size`` via ``_blocked_scan``, and, when ``window_size`` is set
    (truncated BPTT), an outer window scan that severs the whole carry with
    ``jax.lax.stop_gradient`` at each window entry and nests the blocks inside.
    Returns ``(final_carry, outs)``; ``outs`` is the stitched trajectory for a
    stacking ``block_step`` and ``None`` for a folding one.
    """

    def run_blocks(carry, inputs, length):
        return _blocked_scan(
            lambda c, b, _len: block_step(c, b), carry, inputs, length, block_size
        )

    if window_size is None:
        return run_blocks(carry0, scan_inputs, n_steps)

    def run_window(carry, window_inputs, window_len):
        # Sever cross-window credit on the whole carry: the window contributes
        # its local d(.)/d(theta) with the window-start carry detached. Forward
        # value is unchanged (stop_gradient is the identity on the forward).
        carry = jax.lax.stop_gradient(carry)
        return run_blocks(carry, window_inputs, window_len)

    return _blocked_scan(run_window, carry0, scan_inputs, n_steps, window_size)


def _fold_block(op, update):
    """Folding block step (carry ``(state, acc)``), presampled/no-noise path.

    Runs the block's steps through an inner ``jax.lax.scan(op, ...)`` to a
    stacked ``[block_len, ...]`` output and folds it into ``acc`` at block
    granularity (one batched ``update`` per block, not per step). The whole
    block, including the ``update``, is ``jax.checkpoint``-wrapped so it is
    rematerialized on the backward pass; only the block-boundary ``(state, acc)``
    is retained. Emits ``None`` (no trajectory stacked).
    """

    @jax.checkpoint
    def step(carry, block_inputs):
        state, acc = carry
        state, block_out = jax.lax.scan(op, state, block_inputs)
        return (state, update(acc, block_out)), None

    return step


def _stream_block(op, noise_gen):
    """Streaming stacking block step (carry ``(state, counter)``).

    Generates this block's noise from the absolute block ordinal ``counter``
    (``noise_gen(counter, block_len)``), threads ``(time_chunk, noise)`` into the
    inner step scan, and stacks the block output. ``counter`` increments per
    block so the noise is a pure function of ``(key, absolute_block_idx)``,
    independent of how truncation windows tile the rollout. ``jax.checkpoint``
    wraps the block so the draw is *regenerated*, not stored, on the backward
    pass (the streaming memory win).
    """

    @jax.checkpoint
    def step(carry, time_chunk):
        state, counter = carry
        noise = noise_gen(counter, time_chunk.shape[0])
        state, out = jax.lax.scan(op, state, (time_chunk, noise))
        return (state, counter + 1), out

    return step


def _stream_fold_block(op, update, noise_gen):
    """Streaming folding block step (carry ``(state, acc, counter)``).

    Combines per-block streaming noise (as in ``_stream_block``) with the
    block-level fold of ``_fold_block``: generates the block noise from
    ``counter``, runs the inner step scan over ``(time_chunk, noise)``, folds the
    block output into ``acc``, and increments ``counter``. ``jax.checkpoint``
    rematerializes the whole block (noise draw + inner scan + update) on the
    backward pass. Emits ``None``.
    """

    @jax.checkpoint
    def step(carry, time_chunk):
        state, acc, counter = carry
        noise = noise_gen(counter, time_chunk.shape[0])
        state, block_out = jax.lax.scan(op, state, (time_chunk, noise))
        return (state, update(acc, block_out), counter + 1), None

    return step


def _snap_window(window, block_size):
    """Snap a truncation window to the nearest multiple of ``block_size``.

    Window boundaries must align with block boundaries so the blocks nest and
    (under streaming noise) the absolute block grid is independent of the window
    tiling. A non-multiple window is rounded to the nearest multiple (floored at
    one block) with a warning, rather than silently changing the gradient
    horizon.
    """
    if window % block_size == 0:
        return window
    snapped = max(block_size, round(window / block_size) * block_size)
    warnings.warn(
        f"grad_horizon={window} is not a multiple of block_size={block_size}; "
        f"snapping to {snapped} so window boundaries align with block "
        "boundaries.",
        stacklevel=2,
    )
    return snapped


def _reduce_fold(reduce, variable_names, n_nodes, n_steps):
    """Build the ``(acc0, update)`` fold pair for ``run_scan``, or None.

    A reducer is the ``(init, update, finalize)`` triple passed as the ``reduce``
    kwarg. ``init(template, n_steps)`` sizes the accumulator: the per-step output
    template is ``[n_vois, n_nodes]`` (the selected variables of interest,
    matching the leading-after-time shape of a stacked trajectory) and
    ``n_steps`` is the rollout length. Time-grid reducers that need ``dt`` (e.g.
    BOLD decimation) take it at construction instead, since their per-block
    update closes over static strides built before the framework calls ``init``.
    """
    if reduce is None:
        return None
    init, update, _finalize = reduce
    template = jnp.zeros((len(variable_names), n_nodes))
    return (init(template, n_steps), update)


def run_scan(op, state0, scan_inputs, n_steps, solver, fold=None, noise_gen=None):
    """Run the integration scan, dispatching on the solver's gradient/memory knobs.

    The single seam where scan-level features live. Independent, nullable knobs
    select the path:

    - ``noise_gen`` (streaming noise): ``noise_gen(block_idx, block_len)`` or
      None. Set only when ``block_size`` is set and the network has noise with no
      injected tensor; the per-block noise is generated in-scan from the absolute
      block ordinal, so ``scan_inputs`` carries only the time signal (no noise
      leaf) and the block step combines them.
    - ``fold`` (the reduce output handler): ``(acc0, update)`` or None. When set
      with ``block_size``, the trajectory is folded block-wise into ``acc`` and
      the final carry exposes ``acc`` at index 1; the caller reads it and applies
      ``finalize``. Requires ``block_size``; with ``block_size=None`` the caller
      folds the stacked trajectory once instead (the degenerate single-block /
      post-hoc case).
    - ``grad_horizon`` (gradient horizon): if set, run a windowed scan that
      severs the carry gradient every ``W`` steps. Snapped to a multiple of
      ``block_size`` when both are set so window and block boundaries align.
    - ``block_size`` (block granularity): with no truncation, None is the
      plain single ``jax.lax.scan`` (the monolithic default, no-regression path)
      and an int is the outer block scan that trades recompute for backward
      memory and (with ``noise_gen``) streams the per-block noise.

    ``op`` consumes its per-step driving signals (time, and for SDEs the noise
    slice) from its block inputs and is agnostic to how they were produced.
    """
    block_size = solver.block_size
    window = solver.grad_horizon
    if window is not None and block_size is not None:
        window = _snap_window(window, block_size)

    # Streaming and/or fold both ride on the block scan, which requires
    # block_size. The block step encapsulates the streaming-vs-presampled and
    # stack-vs-fold choices; _composed_scan owns the window/block composition.
    if block_size is not None and (fold is not None or noise_gen is not None):
        if fold is not None:
            acc0, update = fold
            if noise_gen is not None:
                block_step = _stream_fold_block(op, update, noise_gen)
                carry0 = (state0, acc0, jnp.array(0))
            else:
                block_step = _fold_block(op, update)
                carry0 = (state0, acc0)
            return _composed_scan(
                block_step, carry0, scan_inputs, n_steps, block_size, window
            )
        # streaming stack: unwrap the (state, counter) carry to the bare state.
        block_step = _stream_block(op, noise_gen)
        (state, _counter), outs = _composed_scan(
            block_step,
            (state0, jnp.array(0)),
            scan_inputs,
            n_steps,
            block_size,
            window,
        )
        return state, outs

    # Non-streaming, non-fold paths (presampled noise or none).
    if window is not None:
        return _truncated_scan(op, state0, scan_inputs, n_steps, window, block_size)
    if block_size is None:
        return jax.lax.scan(op, state0, scan_inputs)
    return _block_scan(op, state0, scan_inputs, n_steps, block_size)


def _make_diffusion_matrix_fn(diffusion_fn, state_indices, n_states, n_nodes):
    """Build a vectorized diffusion-matrix closure for Diffrax SDE integration.

    Returns ``compute_diffusion_matrix(t, y, noise_params) -> [n_states, n_nodes, n_brownian]``
    where ``n_brownian = len(state_indices) * n_nodes``. The Brownian vector is laid
    out so that index ``i * n_nodes + j`` drives state ``state_indices[i]`` on node ``j``.

    Accepts a diffusion callable returning either a scalar, a per-noise-state 1-D
    array, or a full ``[n_noise_states, n_nodes]`` array; broadcasts accordingly.
    """
    state_indices_arr = jnp.asarray(state_indices)
    n_noise_states = len(state_indices)
    n_brownian = n_noise_states * n_nodes
    i_idx = jnp.arange(n_noise_states)[:, None]
    j_idx = jnp.arange(n_nodes)[None, :]
    brownian_idx = i_idx * n_nodes + j_idx
    state_idx_b = state_indices_arr[:, None]

    def compute_diffusion_matrix(t, y, noise_params):
        g_raw = diffusion_fn(t, y, noise_params)
        if jnp.ndim(g_raw) == 0:
            g = jnp.full((n_noise_states, n_nodes), g_raw)
        elif jnp.ndim(g_raw) == 1:
            g = jnp.broadcast_to(g_raw[..., None], (n_noise_states, n_nodes))
        else:
            g = g_raw
        diffusion_matrix = jnp.zeros((n_states, n_nodes, n_brownian))
        return diffusion_matrix.at[state_idx_b, j_idx, brownian_idx].set(g)

    return compute_diffusion_matrix


_PREPARE_DOC = """Prepare a dynamics model for simulation.

Transforms a model into a JAX-compiled simulation function and a corresponding
configuration PyTree. Dispatches on the first two arguments via ``plum``:

==========================  ================  ======================================
model                       solver            supports
==========================  ================  ======================================
``Network``                 ``NativeSolver``  full feature set: delays, noise,
                                              external inputs, auxiliaries, VOI
``Network``                 ``DiffraxSolver`` stateless couplings only; no delays,
                                              no auxiliaries, no VOI filtering
``AbstractDynamics``        ``NativeSolver``  uncoupled multi-node with optional
                                              noise and external inputs
``AbstractDynamics``        ``DiffraxSolver`` uncoupled multi-node with optional
                                              noise and external inputs, no VOI
==========================  ================  ======================================

Parameters
----------
model : Network or AbstractDynamics
    Either a full ``Network`` (dynamics + couplings + graph + optional noise/
    externals) or a bare ``AbstractDynamics`` for uncoupled simulation.
solver : NativeSolver or DiffraxSolver
    Integration method. ``NativeSolver`` (Euler, Heun) uses fixed-step
    ``jax.lax.scan`` and supports every feature. ``DiffraxSolver`` wraps
    ``diffrax.diffeqsolve`` for adaptive stepping but is restricted to
    stateless couplings (see Diffrax limitations below).
t0 : float, optional
    Start time. Default ``0.0``.
t1 : float, optional
    End time. Default ``1.0``. For native solvers ``t1`` is included in the
    save grid; for Diffrax the save grid is governed by ``solver.saveat``.
dt : float, optional
    Time step. Default ``0.1``. Fixed step for ``NativeSolver``; initial step
    ``dt0`` for ``DiffraxSolver``'s adaptive controller.
n_nodes : int, optional
    **Bare-dynamics dispatch only.** Number of uncoupled nodes. Default ``1``.
    Passing this with a ``Network`` raises a dispatch error.
noise : AbstractNoise, optional
    **Bare-dynamics dispatch only.** Stochastic process. For ``Network``,
    noise is attached to the network itself.
externals : dict, optional
    **Bare-dynamics dispatch only.** Mapping ``{name: AbstractExternalInput}``.
    For ``Network``, externals live on the network.

Returns
-------
solve_function : Callable
    Pure JAX function ``solve_function(config) -> result``. JIT-safe and
    compatible with ``jax.grad``, ``jax.vmap``, and ``jax.pmap``.
config : Bunch
    Configuration PyTree. Keys depend on dispatch but always include
    ``dynamics`` (params), ``initial_state``, and ``_internal`` (precomputed
    static data). Network dispatches additionally carry ``coupling``,
    ``external``, ``graph``, and — if stochastic — ``noise`` (params plus
    a ``key`` field driving in-scan noise generation on the native path),
    with an optional ``_internal.noise_samples`` slot for injecting a
    pre-sampled trajectory.

    For stochastic networks on the native path, ``config.noise`` carries
    both the parameter Bunch (e.g. ``sigma``) and ``config.noise.key`` —
    the PRNG key consumed by the in-scan noise generator. The optional
    slot ``config._internal.noise_samples`` defaults to ``None`` and can
    be set to a pre-sampled trajectory of shape ``[n_steps,
    n_noise_states, n_nodes]`` to override generation (used by NumPyro
    workflows that treat the Brownian increments as latents).

    The returned ``result`` is a ``NativeSolution`` (native solvers) or a
    ``DiffraxSolution`` (Diffrax). Both expose ``.ts``, ``.ys``,
    ``.variable_names``, and ``.dt``.

Raises
------
ValueError
    If ``DiffraxSolver`` is used with a network whose ``max_delay > 0``.
    Diffrax cannot maintain history buffers across its internal loop.

Notes
-----
**Config isolation from the source model.** The returned ``config`` is a
structural snapshot: PyTree containers are rebuilt fresh, leaves are
shared. Mutating ``config`` (e.g. ``config.dynamics.G = 0.5``, or
attaching a ``GridAxis`` for a sweep) does not leak back into the source
``Network``/``AbstractDynamics``, and a subsequent ``prepare()`` returns
the original values. Sharing leaves is safe (JAX arrays are immutable)
and avoids duplicating large data like graph weights or history buffers.

**Native solver time grid.** Native solvers scan over
``arange(t0, t1, dt)`` and emit the post-step state on each iteration, so
the returned ``result.ts`` is the half-open grid ``(t0, t1]``:
``[t0 + dt, t0 + 2*dt, ..., t1]`` with ``(t1 - t0) / dt`` samples. The
initial state at ``t0`` is *not* included; ``t1`` *is*. ``t1 - t0`` must be
an integer multiple of ``dt`` for the grid to land exactly on ``t1``.
Diffrax uses ``solver.saveat`` instead.

**Diffrax limitations.** The Diffrax dispatches are experimental and
intentionally narrower than the native ones:

- Delayed couplings are rejected (``max_delay > 0`` raises ``ValueError``).
- Auxiliary outputs from ``dynamics_fn`` are discarded — Diffrax vector
  fields must be pure ``dy/dt``.
- ``VARIABLES_OF_INTEREST`` is **ignored**. The returned ``ys`` always has
  shape ``[n_time, N_STATES, n_nodes]`` and ``variable_names`` always equals
  ``STATE_NAMES``. Use ``NativeSolver`` if you need VOI filtering or
  auxiliaries.
- ``effective_save_dt`` is inferred as ``saveat.ts[1] - saveat.ts[0]`` and
  is only meaningful for uniform ``SaveAt(ts=...)``. Downstream monitors
  that rely on a scalar ``dt`` will be wrong for non-uniform save grids.

**Noise generation (native path).** The full Brownian-increment tensor
of shape ``[n_steps, n_noise_states, n_nodes]`` is materialised inside
``solve_function`` on every call, via a single
``jax.random.normal(config.noise.key, ...)`` that XLA fuses with the
downstream scan. To scan over seeds, vary ``config.noise.key`` (e.g.
with ``eqx.tree_at`` and ``jax.vmap``); no re-``prepare`` needed.
Reseeding the noise object after ``prepare`` has no effect —
``config.noise.key`` is the live source of randomness.

For workflows that supply increments explicitly (NumPyro inference over
Brownian increments, replay of a recorded trajectory, common random
numbers across a sequential parameter sweep), set
``config._internal.noise_samples`` to an array of shape ``[n_steps,
n_noise_states, n_nodes]``. The trace-time branch on this field
bypasses the in-call PRNG; flipping between ``None`` and an array
triggers a one-time JIT retrace per ``solve_function``.

**Noise generation (Diffrax path).** The ``VirtualBrownianTree`` is
constructed inside ``solve_function`` from ``config.noise.key``, so
the seed-swap workflow used on the native path works identically here:
vary ``config.noise.key`` (e.g. with ``eqx.tree_at`` and ``jax.vmap``)
to scan over noise realisations without re-``prepare``-ing. There is
no injection slot on this dispatch — ``config._internal.noise_samples``
is native-only because Diffrax controls are evaluated lazily by the
solver (including at sub-step times for adaptive stages), and a fixed
pre-sampled array cannot service those queries.

**optax / grad note.** Because ``config.noise.key`` is a PRNG-key leaf
sitting inside ``config.noise`` next to numeric parameters, any
optimiser or gradient transformation applied to that subtree must
either skip non-float leaves or operate on a filtered view (e.g.
``eqx.partition``). The diffusion callback itself only touches numeric
fields, so this is purely a concern for callers that pass
``config.noise`` wholesale into ``optax``.

**Preparation steps (native).**

1. Prepare all couplings, building history buffers for delays if needed.
2. Prepare external inputs.
3. Pre-generate noise samples if stochastic (one per timestep).
4. Pre-compile coupling/external compute and state-update closures to
   avoid dict iteration inside the scan.
5. Return a pure function wrapping ``jax.lax.scan``.

**Preparation steps (Diffrax).**

1. Validate no delays.
2. Build stateless coupling/external data.
3. Construct the ``diffrax.ODETerm`` drift and, if stochastic, a
   ``ControlTerm`` over a ``VirtualBrownianTree``.
4. Return a pure function wrapping ``diffrax.diffeqsolve``.

**Solver selection.** Use ``NativeSolver`` for anything with delays, noise
interacting with history, VOI filtering, or auxiliary recording. Use
``DiffraxSolver`` for stiff stateless systems where adaptive stepping pays
off.

Examples
--------
Network with native solver (all features):

>>> from tvboptim.experimental.network_dynamics import Network, prepare
>>> from tvboptim.experimental.network_dynamics.dynamics import ReducedWongWang
>>> from tvboptim.experimental.network_dynamics.solvers import Euler
>>> from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
>>> from tvboptim.experimental.network_dynamics.graph import DenseGraph
>>> import jax.numpy as jnp
>>> network = Network(ReducedWongWang(),
...                   LinearCoupling(incoming_states='S', G=1.0),
...                   DenseGraph(jnp.ones((68, 68))))
>>> model_fn, config = prepare(network, Euler(), t0=0, t1=100, dt=0.1)
>>> result = model_fn(config)

Network with Diffrax (no delays, no VOI, no auxiliaries):

>>> import diffrax
>>> from tvboptim.experimental.network_dynamics.solvers import DiffraxSolver
>>> solver = DiffraxSolver(diffrax.Tsit5(),
...                        saveat=diffrax.SaveAt(ts=jnp.arange(0, 100, 0.1)))
>>> model_fn, config = prepare(network, solver, t0=0, t1=100, dt=0.1)

Bare dynamics, uncoupled multi-node:

>>> from tvboptim.experimental.network_dynamics.dynamics import JansenRit
>>> from tvboptim.experimental.network_dynamics.solvers import Heun
>>> model_fn, config = prepare(JansenRit(), Heun(), t0=0, t1=1.0, dt=1e-3,
...                            n_nodes=3)

Modifying parameters between runs (config is a PyTree):

>>> import copy
>>> cfg2 = copy.deepcopy(config)
>>> cfg2.dynamics.G = 2.5
>>> result2 = model_fn(cfg2)

See Also
--------
solve : Thin wrapper that calls ``prepare`` and executes immediately.
Network : Network dynamics container.
NativeSolver : Fixed-step integrators (Euler, Heun).
DiffraxSolver : Adaptive-step integrators via Diffrax.
"""


def solve(
    model,
    solver,
    t0: float = 0.0,
    t1: float = 100.0,
    dt: float = 0.1,
    **kwargs,
):
    """Main entry point for simulation.

    Accepts either a Network or a bare AbstractDynamics instance.
    Dispatches to the appropriate prepare() overload via plum.

    Args:
        model: Network or AbstractDynamics instance
        solver: NativeSolver or DiffraxSolver instance
        t0: Start time
        t1: End time (inclusive for native solvers — see note on time grid)
        dt: Time step
        **kwargs: Additional arguments forwarded to prepare()
            (e.g. n_nodes for bare dynamics)

    Returns:
        Simulation results wrapped in result object

    Notes:
        Native solvers use the half-open scan grid ``arange(t0, t1, dt)`` and
        emit the post-step state on each iteration, so the returned save grid
        is ``(t0, t1]``: ``result.ts = [t0 + dt, t0 + 2*dt, ..., t1]``, with the
        initial state at ``t0`` excluded and the endpoint ``t1`` included.
        The number of saved samples is ``(t1 - t0) / dt``. ``t1 - t0`` must be
        an integer multiple of ``dt`` for the grid to land exactly on ``t1``.

    Examples:
        >>> # With Network
        >>> result = solve(network, Euler(), t0=0, t1=10, dt=0.01)

        >>> # With bare dynamics (single node)
        >>> result = solve(JansenRit(), Heun(), t0=0, t1=1.0, dt=0.001)

        >>> # With bare dynamics (multi-node uncoupled)
        >>> result = solve(JansenRit(), Heun(), t0=0, t1=1.0, dt=0.001, n_nodes=3)
    """
    solve_fn, params = prepare(model, solver, t0=t0, t1=t1, dt=dt, **kwargs)
    return solve_fn(params)


@dispatch
def prepare(
    network: HeterogeneousNetwork,
    solver: NativeSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
    reduce=None,
) -> Tuple[Callable, Bunch]:
    """Prepare heterogeneous groups and instantaneous routes."""
    unsupported_routes = [
        name
        for name in network.route_names
        if not isinstance(
            network.routes[name].coupling,
            (InstantaneousCoupling, DelayedCoupling),
        )
    ]
    if unsupported_routes:
        raise NotImplementedError(
            "Heterogeneous routes require an instantaneous or delayed "
            f"PrePostCoupling implementation: {unsupported_routes}"
        )
    if reduce is not None:
        raise NotImplementedError(
            "HeterogeneousNetwork does not support reduce yet: groups have "
            "different variable and node axes, so the homogeneous reducer "
            "template is ambiguous. Reduce result.groups.<name> post-hoc, or "
            "omit reduce until an explicit grouped-observation contract exists."
        )
    if isinstance(solver, BoundedSolver):
        raise NotImplementedError(
            "BoundedSolver needs explicit per-group bounds for heterogeneous state"
        )

    time_steps = jnp.arange(t0, t1, dt)
    prepared_topology = prepare_graph_topology(network.graph)
    group_specs = []
    variable_names = {}
    initial_state = Bunch()
    group_config = Bunch()
    noise_specs = []
    noise_samples_init = Bunch()
    external_specs = Bunch()
    external_data = Bunch()
    external_state_init = Bunch()
    zero_externals = Bunch()
    for name in network.group_names:
        group = network.groups[name]
        n_group_nodes = len(network.group_nodes[name])
        state_indices, aux_indices, record_aux, names = _split_voi(group.dynamics)
        group_specs.append(
            (
                name,
                group.dynamics,
                n_group_nodes,
                state_indices,
                aux_indices,
                record_aux,
            )
        )
        variable_names[name] = names
        initial_state[name] = network.initial_state_for(name)
        group_config[name] = Bunch(dynamics=_snapshot(group.dynamics.params))

        zero_externals[name] = Bunch(
            {
                input_name: jnp.zeros((n_dims, n_group_nodes))
                for input_name, n_dims in group.dynamics.EXTERNAL_INPUTS.items()
            }
        )

        if group.noise is not None:
            # A noise instance owns its resolved state indices. Copy it so the
            # same construction object can safely be reused by groups with
            # different dynamics signatures.
            noise = copy.copy(group.noise)
            noise._state_indices = noise._resolve_state_indices(group.dynamics)
            state_indices_noise = noise._state_indices
            noise_specs.append(
                (
                    name,
                    noise.diffusion,
                    state_indices_noise,
                    len(state_indices_noise),
                    n_group_nodes,
                )
            )
            group_config[name].noise = _snapshot(noise.params)
            group_config[name].noise.key = noise.key
            noise_samples_init[name] = None

        if group.externals:
            group_config[name].external = Bunch()
            external_data[name] = Bunch()
            external_state_init[name] = Bunch()
            prepared_externals = []
            # External inputs operate in group-local node space. The small
            # context preserves the established prepare(network, dt) contract
            # without pretending the shared graph has only this many nodes.
            context = Bunch(
                graph=Bunch(n_nodes=n_group_nodes),
                dynamics=group.dynamics,
                initial_state=initial_state[name],
            )
            for input_name in group.dynamics.EXTERNAL_INPUTS:
                external = group.externals.get(input_name)
                if external is None:
                    prepared_externals.append((input_name, None, None, Bunch()))
                    continue
                data, state = external.prepare(context, dt)
                runtime_params, static_params = _partition_jax_params(external.params)
                external_data[name][input_name] = data
                external_state_init[name][input_name] = state
                group_config[name].external[input_name] = _snapshot(runtime_params)
                prepared_externals.append((input_name, external, data, static_params))
            external_specs[name] = tuple(prepared_externals)

    route_config = Bunch()
    route_specs = []
    route_history_shapes = {}

    def pack_readouts(grouped_state, specs, width, dtype, params):
        signal = jnp.zeros((width, network.n_nodes), dtype=dtype)
        for group_name, nodes, readout, indices in specs:
            values = (
                grouped_state[group_name][indices]
                if readout is None
                else readout(grouped_state[group_name], params[group_name])
            )
            signal = signal.at[:, nodes].set(values)
        return signal

    def pack_history_readouts(history, specs, width, dtype, params):
        n_times = history.ts.shape[0]
        signal = jnp.zeros((n_times, width, network.n_nodes), dtype=dtype)
        for group_name, nodes, readout, indices in specs:
            states = history.groups[group_name]
            values = (
                states[:, indices, :]
                if readout is None
                else jax.vmap(readout, in_axes=(0, None))(states, params[group_name])
            )
            signal = signal.at[:, :, nodes].set(values)
        return signal

    for route_name in network.route_names:
        route = network.routes[route_name]
        source_params = Bunch(
            {
                name: _snapshot(route.source_params.get(name, Bunch()))
                for name in set(route.source) | set(route.local)
            }
        )
        target_params = Bunch(
            {
                name: _snapshot(route.target_params.get(name, Bunch()))
                for name in route.target
            }
        )
        route_config[route_name] = Bunch(
            coupling=_snapshot(route.coupling.params),
            source_params=source_params,
            target_params=target_params,
        )

        def prepare_readouts(readouts, params, role):
            prepared = []
            widths = set()
            dtypes = []
            for group_name in sorted(readouts):
                readout = readouts[group_name]
                state = initial_state[group_name]
                nodes = jnp.asarray(network.group_nodes[group_name], dtype=int)
                if callable(readout):
                    try:
                        shaped = jax.eval_shape(readout, state, params[group_name])
                    except Exception as exc:
                        raise ValueError(
                            f"route {route_name!r} {role} readout for group "
                            f"{group_name!r} could not be evaluated as "
                            "readout(state, params)"
                        ) from exc
                    if not hasattr(shaped, "shape"):
                        raise ValueError(
                            f"route {route_name!r} {role} readout for group "
                            f"{group_name!r} must return one array"
                        )
                    expected_nodes = len(network.group_nodes[group_name])
                    if len(shaped.shape) != 2 or shaped.shape[1] != expected_nodes:
                        raise ValueError(
                            f"route {route_name!r} {role} readout for group "
                            f"{group_name!r} returned shape {shaped.shape}; "
                            f"expected [Q, {expected_nodes}]"
                        )
                    width = shaped.shape[0]
                    dtype = shaped.dtype
                    prepared.append((group_name, nodes, readout, None))
                else:
                    indices = network.groups[group_name].dynamics.name_to_index(readout)
                    width = len(readout)
                    dtype = state.dtype
                    prepared.append((group_name, nodes, None, indices))
                widths.add(width)
                dtypes.append(dtype)
            if len(widths) != 1:
                raise ValueError(
                    f"route {route_name!r} {role} readouts must share one "
                    f"channel width, got {sorted(widths)}"
                )
            return tuple(prepared), widths.pop(), tuple(dtypes)

        source_specs, source_width, source_dtypes = prepare_readouts(
            route.source, source_params, "source"
        )
        if route.local:
            local_specs, local_width, local_dtypes = prepare_readouts(
                route.local, source_params, "local"
            )
        else:
            local_specs, local_width, local_dtypes = (), 0, ()

        signal_dtype = jnp.result_type(*(source_dtypes + local_dtypes))
        route.coupling._validate_pre_contract(
            network.graph,
            tuple(range(source_width)),
            tuple(range(local_width)),
            signal_dtype,
        )
        is_delayed = isinstance(route.coupling, DelayedCoupling)
        if is_delayed and not hasattr(network.graph, "delays"):
            raise ValueError(
                f"delayed route {route_name!r} requires a graph with delays"
            )
        if is_delayed:
            max_delay = effective_max_delay(network.graph)
            if network._history is None:
                initial_signal = pack_readouts(
                    initial_state,
                    source_specs,
                    source_width,
                    signal_dtype,
                    source_params,
                )
                history_rows = delay_steps_bound(max_delay, dt) + 1
                initial_history = jnp.broadcast_to(
                    initial_signal,
                    (history_rows,) + initial_signal.shape,
                )
            else:
                packed_history = pack_history_readouts(
                    network._history,
                    source_specs,
                    source_width,
                    signal_dtype,
                    source_params,
                )
                initial_history = extract_history_window(
                    network._history.ts,
                    packed_history,
                    max_delay,
                    dt,
                )
            coupling_data, coupling_state = route.coupling._prepare_history(
                network.graph,
                initial_history,
                dt,
                t0,
                t1,
                incoming_indices=tuple(range(source_width)),
                local_indices=tuple(range(local_width)),
            )
            route_config[route_name].history = coupling_state.history
            route_history_shapes[route_name] = tuple(coupling_state.history.shape)
            if "write_idx" in coupling_state:
                route_config[route_name].write_idx = coupling_state.write_idx
        else:
            coupling_data = Bunch()
        coupling_data._prepared_topology = prepared_topology
        coupling_data.stage_time_centroid = solver.stage_time_centroid
        coupling_data.recompute_coupling_per_stage = solver.recompute_coupling_per_stage
        source_node_count = sum(nodes.shape[0] for _, nodes, _, _ in source_specs)
        if source_node_count != network.n_nodes:
            source_mask = jnp.zeros((network.n_nodes,), dtype=signal_dtype)
            for _group_name, nodes, _readout, _indices in source_specs:
                source_mask = source_mask.at[nodes].set(1)
            coupling_data._source_mask = source_mask

        target_specs = []
        for group_name in sorted(route.target):
            input_name, conversion = route.target[group_name]
            nodes = jnp.asarray(network.group_nodes[group_name], dtype=int)
            n_group_nodes = len(network.group_nodes[group_name])
            if conversion is not None:
                signal = jax.ShapeDtypeStruct(
                    (route.coupling.N_OUTPUT_STATES, n_group_nodes), signal_dtype
                )
                try:
                    converted = jax.eval_shape(
                        conversion, signal, target_params[group_name]
                    )
                except Exception as exc:
                    raise ValueError(
                        f"route {route_name!r} conversion for group "
                        f"{group_name!r} could not be evaluated as "
                        "conversion(signal, params)"
                    ) from exc
                expected = (
                    network.groups[group_name].dynamics.COUPLING_INPUTS[input_name],
                    n_group_nodes,
                )
                actual = getattr(converted, "shape", None)
                if actual != expected:
                    raise ValueError(
                        f"route {route_name!r} conversion for group "
                        f"{group_name!r} returned shape {actual}; expected {expected}"
                    )
            target_specs.append((group_name, nodes, input_name, conversion))

        route_specs.append(
            (
                route_name,
                route.coupling,
                coupling_data,
                source_width,
                local_width,
                signal_dtype,
                source_specs,
                local_specs,
                tuple(target_specs),
                is_delayed,
            )
        )

    config = Bunch(
        groups=group_config,
        routes=route_config,
        graph=_snapshot(network.graph),
        initial_state=initial_state,
        _internal=Bunch(time=Bunch(t0=t0, t1=t1, dt=dt)),
    )
    has_noise = bool(noise_specs)
    has_externals = bool(external_specs)
    if has_noise:
        config._internal.noise_samples = noise_samples_init
    if has_externals:
        config._internal.external = external_data
        config._internal.external_state = external_state_init
    n_steps = len(time_steps)

    def precompute_routes(config):
        enriched = Bunch()
        for route_name, coupling, data, *_ in route_specs:
            enriched[route_name] = coupling.precompute(
                data, config.routes[route_name].coupling, config.graph
            )
        return enriched

    def compute_routes(grouped_state, route_states, config, enriched):
        coupling_inputs = Bunch()
        for name, dynamics, n_group_nodes, *_ in group_specs:
            coupling_inputs[name] = Bunch(
                {
                    input_name: jnp.zeros(
                        (n_dims, n_group_nodes), dtype=grouped_state[name].dtype
                    )
                    for input_name, n_dims in dynamics.COUPLING_INPUTS.items()
                }
            )

        for (
            route_name,
            coupling,
            _data,
            source_width,
            local_width,
            signal_dtype,
            source_specs,
            local_specs,
            target_specs,
            is_delayed,
        ) in route_specs:
            route_params = config.routes[route_name]
            local_signal = None
            if local_width:
                local_signal = pack_readouts(
                    grouped_state,
                    local_specs,
                    local_width,
                    signal_dtype,
                    route_params.source_params,
                )

            if is_delayed:
                transported = coupling._compute_from_history(
                    local_signal,
                    enriched[route_name],
                    route_states[route_name],
                    route_params.coupling,
                    config.graph,
                )
            else:
                source_signal = pack_readouts(
                    grouped_state,
                    source_specs,
                    source_width,
                    signal_dtype,
                    route_params.source_params,
                )
                transported = coupling._compute_from_signals(
                    source_signal,
                    local_signal,
                    enriched[route_name],
                    route_params.coupling,
                    config.graph,
                )
            for group_name, nodes, input_name, conversion in target_specs:
                values = transported[:, nodes]
                if conversion is not None:
                    values = conversion(values, route_params.target_params[group_name])
                coupling_inputs[group_name][input_name] = (
                    coupling_inputs[group_name][input_name] + values
                )
        return coupling_inputs

    delayed_route_specs = tuple(spec for spec in route_specs if spec[-1])
    has_delays = bool(delayed_route_specs)

    def initial_route_states(config):
        states = Bunch()
        for spec in delayed_route_specs:
            route_name = spec[0]
            actual_shape = tuple(config.routes[route_name].history.shape)
            expected_shape = route_history_shapes[route_name]
            if actual_shape != expected_shape:
                raise ValueError(
                    f"history for route {route_name!r} has shape {actual_shape}; "
                    f"expected {expected_shape}"
                )
            state = Bunch(history=config.routes[route_name].history)
            if "write_idx" in config.routes[route_name]:
                state.write_idx = config.routes[route_name].write_idx
            states[route_name] = state
        return states

    def update_route_states(route_states, grouped_state, config, enriched):
        updated = Bunch()
        for (
            route_name,
            coupling,
            _data,
            source_width,
            _local_width,
            signal_dtype,
            source_specs,
            _local_specs,
            _target_specs,
            _is_delayed,
        ) in delayed_route_specs:
            route_params = config.routes[route_name]
            transmitted = pack_readouts(
                grouped_state,
                source_specs,
                source_width,
                signal_dtype,
                route_params.source_params,
            )
            updated[route_name] = coupling._update_history_from_signal(
                enriched[route_name], route_states[route_name], transmitted
            )
        return updated

    def _f(config):
        grouped_state0 = config.initial_state.copy()
        topology_anchor_group = network.group_names[0]
        grouped_state0[topology_anchor_group] = validate_graph_topology(
            prepared_topology,
            config.graph,
            grouped_state0[topology_anchor_group],
        )
        enriched = precompute_routes(config)
        dynamics_params = Bunch(
            {name: config.groups[name].dynamics for name in network.group_names}
        )

        streaming = (
            has_noise
            and solver.block_size is not None
            and all(
                config._internal.noise_samples[name] is None for name, *_ in noise_specs
            )
        )
        noise_gen = None
        if streaming:

            def noise_gen(block_idx, block_len):
                return Bunch(
                    {
                        name: jax.random.normal(
                            jax.random.fold_in(
                                config.groups[name].noise.key, block_idx
                            ),
                            (block_len, n_noise_states, n_group_nodes),
                        )
                        for (
                            name,
                            _diffusion,
                            _indices,
                            n_noise_states,
                            n_group_nodes,
                        ) in noise_specs
                    }
                )

            noise_samples_all = None
        elif has_noise:
            noise_samples_all = Bunch(
                {
                    name: _materialize_noise(
                        config.groups[name].noise.key,
                        config._internal.noise_samples[name],
                        (n_steps, n_noise_states, n_group_nodes),
                    )
                    for (
                        name,
                        _diffusion,
                        _indices,
                        n_noise_states,
                        n_group_nodes,
                    ) in noise_specs
                }
            )
        else:
            noise_samples_all = None

        def compute_group_externals(t, grouped_state, external_state):
            values = zero_externals.copy()
            for group_name, prepared_externals in external_specs.items():
                for input_name, external, data, static_params in prepared_externals:
                    if external is not None:
                        params = static_params.copy()
                        params.update(config.groups[group_name].external[input_name])
                        values[group_name][input_name] = external.compute(
                            t,
                            grouped_state[group_name],
                            data,
                            external_state[group_name][input_name],
                            params,
                        )
            return values

        def update_group_externals(external_state, grouped_state):
            updated = Bunch()
            for group_name, prepared_externals in external_specs.items():
                updated[group_name] = Bunch()
                for input_name, external, data, _static_params in prepared_externals:
                    if external is not None:
                        updated[group_name][input_name] = external.update_state(
                            data,
                            external_state[group_name][input_name],
                            grouped_state[group_name],
                        )
            return updated

        def grouped_dynamics(
            t,
            grouped_state,
            grouped_params,
            external_state,
            route_states,
            frozen_coupling_inputs=None,
        ):
            coupling_inputs = (
                compute_routes(grouped_state, route_states, config, enriched)
                if frozen_coupling_inputs is None
                else frozen_coupling_inputs
            )
            derivatives = Bunch()
            auxiliaries = Bunch()
            external_inputs = (
                compute_group_externals(t, grouped_state, external_state)
                if has_externals
                else zero_externals
            )
            for name, dynamics, n_group_nodes, *_ in group_specs:
                result = dynamics.dynamics(
                    t,
                    grouped_state[name],
                    grouped_params[name],
                    coupling_inputs[name],
                    external_inputs[name],
                )
                if isinstance(result, tuple):
                    derivatives[name], auxiliaries[name] = result
                else:
                    derivatives[name] = result
                    auxiliaries[name] = jnp.empty(
                        (0, n_group_nodes), dtype=grouped_state[name].dtype
                    )
            return derivatives, auxiliaries

        def op(carry, scan_input):
            if has_noise:
                t, noise_raw = scan_input
            else:
                t = scan_input

            if has_externals or has_delays:
                grouped_state = carry.dynamics
                external_state = carry.external
                route_states = carry.routes
            else:
                grouped_state = carry
                external_state = Bunch()
                route_states = Bunch()

            frozen_coupling_inputs = None
            if not solver.recompute_coupling_per_stage:
                frozen_coupling_inputs = compute_routes(
                    grouped_state, route_states, config, enriched
                )

            def wrapped_dynamics(t_inner, grouped_state, grouped_params):
                return grouped_dynamics(
                    t_inner,
                    grouped_state,
                    grouped_params,
                    external_state,
                    route_states,
                    frozen_coupling_inputs,
                )

            noise_sample = Bunch(
                {
                    name: jnp.zeros_like(grouped_state[name])
                    for name in network.group_names
                }
            )
            if has_noise:
                for (
                    name,
                    diffusion,
                    state_indices_noise,
                    n_noise_states,
                    n_group_nodes,
                ) in noise_specs:
                    raw_diffusion = jnp.asarray(
                        diffusion(t, grouped_state[name], config.groups[name].noise)
                    )
                    if raw_diffusion.ndim == 0:
                        diffusion_values = jnp.full(
                            (n_noise_states, n_group_nodes), raw_diffusion
                        )
                    elif raw_diffusion.ndim == 1:
                        diffusion_values = jnp.broadcast_to(
                            raw_diffusion[:, None],
                            (n_noise_states, n_group_nodes),
                        )
                    else:
                        diffusion_values = jnp.broadcast_to(
                            raw_diffusion, (n_noise_states, n_group_nodes)
                        )
                    scaled_noise = diffusion_values * jnp.sqrt(dt) * noise_raw[name]
                    noise_sample[name] = (
                        noise_sample[name].at[state_indices_noise].set(scaled_noise)
                    )

            next_state, auxiliaries = solver.step(
                wrapped_dynamics,
                t,
                grouped_state,
                dt,
                dynamics_params,
                noise_sample,
            )
            output = Bunch()
            for (
                name,
                _dynamics,
                _n_group_nodes,
                state_indices,
                aux_indices,
                record_aux,
            ) in group_specs:
                output[name] = _assemble_output(
                    next_state[name],
                    auxiliaries[name],
                    state_indices,
                    aux_indices,
                    record_aux,
                )
            if has_externals or has_delays:
                next_external = (
                    update_group_externals(external_state, next_state)
                    if has_externals
                    else Bunch()
                )
                next_routes = (
                    update_route_states(route_states, next_state, config, enriched)
                    if has_delays
                    else Bunch()
                )
                next_carry = Bunch(
                    dynamics=next_state,
                    external=next_external,
                    routes=next_routes,
                )
            else:
                next_carry = next_state
            return next_carry, output

        state0 = (
            Bunch(
                dynamics=grouped_state0,
                external=(
                    config._internal.external_state.copy() if has_externals else Bunch()
                ),
                routes=initial_route_states(config) if has_delays else Bunch(),
            )
            if has_externals or has_delays
            else grouped_state0
        )
        scan_inputs = (
            time_steps
            if not has_noise or streaming
            else (time_steps, noise_samples_all)
        )
        _final_state, trajectories = run_scan(
            op,
            state0,
            scan_inputs,
            n_steps,
            solver,
            noise_gen=noise_gen,
        )
        ts = t0 + (jnp.arange(n_steps) + 1) * dt
        return HeterogeneousSolution(
            ts,
            trajectories,
            dt=dt,
            variable_names=variable_names,
            group_nodes=network.group_nodes,
            n_nodes=network.n_nodes,
        )

    return _f, config


@dispatch
def prepare(
    network: HeterogeneousNetwork,
    solver: DiffraxSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
    reduce=None,
) -> Tuple[Callable, Bunch]:
    del network, solver, t0, t1, dt, reduce
    raise NotImplementedError(
        "HeterogeneousNetwork currently supports native fixed-step solvers only"
    )


@dispatch
def prepare(
    network: Network,
    solver: NativeSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
    reduce=None,
) -> Tuple[Callable, Bunch]:
    """Compile a model into a pure JAX solve function and a config PyTree.

    Builds per-dispatch data (coupling buffers, noise samples, external
    inputs) and returns ``(solve_fn, config)`` where ``solve_fn(config)``
    runs the integration. Dispatches on the first two arguments via
    ``plum``: ``Network``/``AbstractDynamics`` paired with
    ``NativeSolver``/``DiffraxSolver``.

    Parameters
    ----------
    t0, t1, dt : float
        Integration interval and step size. ``dt`` is the fixed step for
        native solvers and the initial step for Diffrax.

    Returns
    -------
    (Callable, Bunch)
        Pure solve function and its runtime configuration PyTree.

    See ``help(prepare)`` or ``prepare.__doc__`` for the full reference,
    including per-dispatch parameters (``n_nodes``, ``noise``, ``externals``
    for bare dynamics) and Diffrax limitations (no delays, no auxiliaries,
    no VOI filtering).
    """
    # Prepare all couplings (creates history buffers, computes indices, etc.).
    # The solver's stage-time centroid rides along so delayed couplings can
    # undo the delay bias that freezing the coupling across stages introduces.
    prepared_topology = prepare_graph_topology(network.graph)
    coupling_data_dict, coupling_state_dict_init = network.prepare(
        dt,
        t0,
        t1,
        stage_time_centroid=solver.stage_time_centroid,
        recompute_coupling_per_stage=solver.recompute_coupling_per_stage,
        _prepared_topology=prepared_topology,
    )

    # Prepare all external inputs
    external_data_dict, external_state_dict_init = network.prepare_external(dt)

    # Time array
    time_steps = jnp.arange(t0, t1, dt)

    # Build new config structure. Param/graph subtrees are snapshotted so
    # user mutations to the returned config don't leak back into the source
    # network (or contaminate later prepare() calls).
    config = Bunch(
        # Parameters (flattened - no params. prefix)
        dynamics=_snapshot(network.dynamics.params),
        coupling=Bunch(),
        external=Bunch(),
        # Graph (PyTree object)
        graph=_snapshot(network.graph),
        # Initial state
        initial_state=Bunch(
            dynamics=network.initial_state,
            coupling=coupling_state_dict_init,
            external=external_state_dict_init,
        ),
        # Internal (static precomputed data)
        _internal=Bunch(
            coupling=coupling_data_dict,
            external=external_data_dict,
            time=Bunch(t0=t0, t1=t1, dt=dt),
        ),
    )

    # Add coupling params
    for name, coupling in network.coupling.items():
        config.coupling[name] = _snapshot(coupling.params)

    # Add external input params
    for name, external in network.externals.items():
        config.external[name] = _snapshot(external.params)

    # Add noise params and (optional) sample-injection slot if stochastic.
    # By default the full Brownian-increment tensor is materialised inside
    # _f via a single jax.random.normal(config.noise.key, ...) call, which
    # XLA fuses with the downstream scan. Users who want to inject a
    # pre-sampled trajectory (e.g. for NumPyro inference over Brownian
    # increments) can populate config._internal.noise_samples; the scan
    # branches on its presence at trace time.
    if network.noise is not None:
        config.noise = _snapshot(network.noise.params)
        config.noise.key = network.noise.key
        config._internal.noise_samples = None

    # =========================================================================
    # PRE-COMPILE COUPLING COMPUTATION CLOSURE
    # =========================================================================
    # Build a list of (name, coupling, data) tuples to avoid dict iteration in scan
    coupling_list = []
    coupling_names_ordered = []
    for name in network.dynamics.COUPLING_INPUTS.keys():
        coupling_names_ordered.append(name)
        if name in network.coupling:
            coupling = network.coupling[name]
            data = coupling_data_dict[name]
            coupling_list.append((name, coupling, data))
        else:
            coupling_list.append((name, None, None))

    n_nodes = network.graph.n_nodes
    n_noise_states = (
        len(network.noise._state_indices) if network.noise is not None else 0
    )

    def compute_all_couplings(
        t, network_state, coupling_state_dict, config, coupling_data_dict
    ):
        """Pre-compiled closure for coupling computation.

        Avoids method calls and dict iterations in scan loop.

        Args:
            config: Config containing coupling parameters to use
            coupling_data_dict: Per-coupling data (enriched by precompute if applicable)
        """
        coupling_inputs = Bunch()

        for name, coupling, _ in coupling_list:
            if coupling is None:
                # Missing coupling - use zeros
                n_dims = network.dynamics.COUPLING_INPUTS[name]
                coupling_inputs[name] = jnp.zeros((n_dims, n_nodes))
            else:
                # Compute coupling using enriched data and graph
                data = coupling_data_dict[name]
                state_data = coupling_state_dict[name]
                coupling_inputs[name] = coupling.compute(
                    t,
                    network_state,
                    data,
                    state_data,
                    config.coupling[name],
                    config.graph,
                )

        return coupling_inputs

    # =========================================================================
    # PRE-COMPILE COUPLING STATE UPDATE CLOSURE
    # =========================================================================
    # Build list of couplings that need state updates (avoid dict iteration)
    update_list = [
        (name, network.coupling[name], coupling_data_dict[name])
        for name in network.coupling.keys()
    ]

    def update_all_coupling_states(
        coupling_state_dict, new_network_state, coupling_data_dict
    ):
        """Pre-compiled closure for coupling state updates.

        Avoids method calls and dict iterations in scan loop.
        """
        new_states = Bunch()

        for name, coupling, _ in update_list:
            data = coupling_data_dict[name]
            new_states[name] = coupling.update_state(
                data,
                coupling_state_dict[name],
                new_network_state,
            )

        return new_states

    def precompute_all_couplings(config):
        """Call precompute() for each coupling. Runs inside JIT, once per forward pass."""
        enriched = {}
        for name, coupling, static_data in coupling_list:
            if coupling is not None:
                enriched[name] = coupling.precompute(
                    static_data, config.coupling[name], config.graph
                )
            else:
                enriched[name] = None
        return enriched

    # =========================================================================
    # PRE-COMPILE EXTERNAL INPUT COMPUTATION CLOSURE
    # =========================================================================
    # Build a list of (name, external, data) tuples to avoid dict iteration in scan
    external_list = []
    for name in network.dynamics.EXTERNAL_INPUTS.keys():
        if name in network.externals:
            external_obj = network.externals[name]
            data = external_data_dict[name]
            external_list.append((name, external_obj, data))
        else:
            external_list.append((name, None, None))

    def compute_all_externals(t, network_state, external_state_dict, config):
        """Pre-compiled closure for external input computation.

        Avoids method calls and dict iterations in scan loop.

        Args:
            config: Config containing external input parameters to use
        """
        external_inputs = Bunch()

        for name, external_obj, data in external_list:
            if external_obj is None:
                # Missing external input - use zeros
                n_dims = network.dynamics.EXTERNAL_INPUTS[name]
                external_inputs[name] = jnp.zeros((n_dims, n_nodes))
            else:
                # Compute external input using pre-fetched data
                state_data = external_state_dict[name]
                external_inputs[name] = external_obj.compute(
                    t, network_state, data, state_data, config.external[name]
                )

        return external_inputs

    # =========================================================================
    # PRE-COMPILE EXTERNAL STATE UPDATE CLOSURE
    # =========================================================================
    # Build list of external inputs that need state updates (avoid dict iteration)
    external_update_list = [
        (name, network.externals[name], external_data_dict[name])
        for name in network.externals.keys()
    ]

    def update_all_external_states(external_state_dict, new_network_state):
        """Pre-compiled closure for external state updates.

        Avoids method calls and dict iterations in scan loop.
        """
        new_states = Bunch()

        for name, external_obj, data in external_update_list:
            new_states[name] = external_obj.update_state(
                data,
                external_state_dict[name],
                new_network_state,
            )

        return new_states

    # =========================================================================
    # PRE-COMPILE DYNAMICS + COUPLING COMPUTATION
    # =========================================================================
    # Store dynamics function reference to avoid attribute lookup
    dynamics_fn = network.dynamics.dynamics

    # Pre-allocate solver step function reference
    solver_step = solver.step
    recompute_coupling_per_stage = solver.recompute_coupling_per_stage

    # =========================================================================
    # VARIABLES OF INTEREST - Determine what to record
    # =========================================================================
    (
        state_voi_indices,
        aux_voi_indices,
        record_auxiliaries,
        variable_names,
    ) = _split_voi(network.dynamics)

    # Static shape for the full per-call noise tensor.
    n_steps = len(time_steps)
    noise_samples_shape = (n_steps, n_noise_states, n_nodes)

    def _f(config):
        """Pure integration function."""
        state0 = config.initial_state.copy()
        state0.dynamics = validate_graph_topology(
            prepared_topology, config.graph, state0.dynamics
        )

        # Run precompute() for all couplings once before the scan.
        # This allows parameter-dependent quantities (e.g. W_eff = W * wLRE) to
        # be computed with gradient flow while avoiding per-step redundancy.
        enriched = precompute_all_couplings(config)

        # Noise source. Streaming (per-block fold_in) activates when blocking is
        # on, the network has noise, and no full tensor is injected: it skips the
        # O(n_steps) draw and regenerates each block's noise in-scan (and on the
        # backward pass). Otherwise the full tensor is drawn once and fused with
        # the scan (or the injected tensor is used verbatim).
        streaming = (
            network.noise is not None
            and solver.block_size is not None
            and config._internal.noise_samples is None
        )
        noise_gen = None
        if streaming:
            noise_samples_all = None
            noise_gen = _streaming_noise_gen(
                config.noise.key,
                (n_noise_states, n_nodes),
                provider=config._internal.get("noise_provider", None),
            )
        elif network.noise is not None:
            noise_samples_all = _materialize_noise(
                config.noise.key,
                config._internal.noise_samples,
                noise_samples_shape,
            )
        else:
            noise_samples_all = None

        def op(state, inputs):
            """Single integration step.

            Args:
                state: Bunch(dynamics=network_state, coupling=coupling_state_dict, external=external_state_dict)
                inputs: (t, noise_slice) for SDE or just t for ODE

            Returns:
                (next_state, output) tuple for scan
            """
            # Unpack per-step driving signals from the scan inputs.
            if network.noise is not None:
                t, noise = inputs
            else:
                t = inputs

            # By default compute all coupling inputs ONCE per step at the
            # step-start point (t, state.dynamics) and freeze them across
            # every solver stage (Heun's predictor+corrector, RK4's k1..k4).
            # For delayed couplings the history buffer is a step-level carry
            # that does not depend on the stage state, so freezing is exact
            # while avoiding the (expensive) delay gather 2x for Heun / 4x
            # for RK4. When solver.recompute_coupling_per_stage is True the
            # coupling is instead re-evaluated inside wrapped_dynamics at each
            # stage's (time, state); see NativeSolver for the order trade-off.
            if not recompute_coupling_per_stage:
                frozen_coupling_inputs = compute_all_couplings(
                    t, state.dynamics, state.coupling, config, enriched
                )

            # Inline dynamics wrapper to avoid extra function creation.
            # External inputs are always evaluated per stage so time-varying
            # stimuli see the stage time/state.
            def wrapped_dynamics(t_inner, network_state, params_dynamics):
                if recompute_coupling_per_stage:
                    coupling_inputs = compute_all_couplings(
                        t_inner, network_state, state.coupling, config, enriched
                    )
                else:
                    coupling_inputs = frozen_coupling_inputs
                external_inputs = compute_all_externals(
                    t_inner, network_state, state.external, config
                )
                # Call dynamics with coupling and external inputs
                # Returns (derivatives, auxiliaries) or just derivatives
                return dynamics_fn(
                    t_inner,
                    network_state,
                    params_dynamics,
                    coupling_inputs,
                    external_inputs,
                )

            # Prepare noise sample if stochastic
            noise_sample = jnp.zeros_like(state.dynamics)
            if network.noise is not None:
                # ``noise`` is the per-step slice handed in via scan inputs.
                # Compute diffusion coefficient
                diffusion = network.noise.diffusion(t, state.dynamics, config.noise)

                # Scale noise: g(t,x) * sqrt(dt) * dW
                # The sqrt(dt) factor is essential for SDEs - Brownian increments scale with sqrt(dt)
                scaled_noise = diffusion * jnp.sqrt(dt) * noise

                # Insert into correct state indices
                noise_sample = noise_sample.at[network.noise._state_indices].set(
                    scaled_noise
                )

            # Solver integration step using pre-compiled function references
            # Returns (next_state, auxiliaries)
            next_dynamics_state, auxiliaries = solver_step(
                wrapped_dynamics, t, state.dynamics, dt, config.dynamics, noise_sample
            )

            # Use pre-compiled closure for coupling state updates
            next_coupling_state_dict = update_all_coupling_states(
                state.coupling, next_dynamics_state, enriched
            )

            # Use pre-compiled closure for external state updates
            next_external_state_dict = update_all_external_states(
                state.external, next_dynamics_state
            )

            # Build next state Bunch
            next_state = Bunch(
                dynamics=next_dynamics_state,
                coupling=next_coupling_state_dict,
                external=next_external_state_dict,
            )

            # Apply VARIABLES_OF_INTEREST filtering to build output
            output = _assemble_output(
                next_dynamics_state,
                auxiliaries,
                state_voi_indices,
                aux_voi_indices,
                record_auxiliaries,
            )

            # Return (carry, output)
            return next_state, output

        # Prepare scan inputs. The per-step driving signals are the scan xs:
        # ODE/DDE (and streaming SDE) carry just time; presampled SDE carries
        # (time, noise_slice). When streaming, ``op`` still consumes (time,
        # noise) per step but the noise is generated per block in-scan via
        # ``noise_gen`` rather than sliced from a global tensor.
        if network.noise is None or streaming:
            scan_inputs = time_steps
        else:
            scan_inputs = (time_steps, noise_samples_all)

        # Run integration through the single scan seam, which dispatches on
        # the solver's block knob (block_size), the streaming noise source,
        # and the reduce fold.
        fold = _reduce_fold(reduce, variable_names, n_nodes, n_steps)
        final_carry, res = run_scan(
            op, state0, scan_inputs, n_steps, solver, fold=fold, noise_gen=noise_gen
        )

        # With a reducer, return the finalized aggregate rather than a
        # trajectory. Blocked: acc is threaded in the (state, acc) carry.
        # Monolithic (block_size=None): fold the stacked trajectory once (the
        # degenerate single-block / post-hoc case, no memory win).
        if reduce is not None:
            _init, update, finalize = reduce
            if solver.block_size is None:
                acc = update(fold[0], res)
            else:
                acc = final_carry[1]
            return finalize(acc)

        # Wrap result for consistency
        return wrap_native_result(res, t0, t1, dt, variable_names=variable_names)

    return _f, config


@dispatch
def prepare(
    network: Network,
    solver: DiffraxSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
    reduce=None,
) -> Tuple[Callable, Bunch]:
    """Compile a model into a pure JAX solve function and a config PyTree.

    Builds per-dispatch data (coupling buffers, noise samples, external
    inputs) and returns ``(solve_fn, config)`` where ``solve_fn(config)``
    runs the integration. Dispatches on the first two arguments via
    ``plum``: ``Network``/``AbstractDynamics`` paired with
    ``NativeSolver``/``DiffraxSolver``.

    Parameters
    ----------
    t0, t1, dt : float
        Integration interval and step size. ``dt`` is the fixed step for
        native solvers and the initial step for Diffrax.

    Returns
    -------
    (Callable, Bunch)
        Pure solve function and its runtime configuration PyTree.

    See ``help(prepare)`` or ``prepare.__doc__`` for the full reference,
    including per-dispatch parameters (``n_nodes``, ``noise``, ``externals``
    for bare dynamics) and Diffrax limitations (no delays, no auxiliaries,
    no VOI filtering).
    """
    # =========================================================================
    # VALIDATION: Check for unsupported features
    # =========================================================================

    # reduce is a native-only feature (it rides on the native block scan).
    # plum dispatches on the first two positional args only, so a reduce= meant
    # for the native path can land here; reject it explicitly rather than via a
    # bare TypeError on an unexpected keyword.
    if reduce is not None:
        raise ValueError(
            "reduce is only supported by NativeSolver, not DiffraxSolver "
            "(it rides on the native block scan). Use a NativeSolver."
        )

    # Check for delayed coupling (stateful)
    if network.max_delay > 0.0:
        raise ValueError(
            f"Diffrax solver does not support delayed coupling (max_delay={network.max_delay}). "
            "Delayed couplings require stateful history buffers that cannot be maintained "
            "with Diffrax's internal integration loop. Use NativeSolver instead."
        )

    # Note on solver compatibility with SDEs:
    # We let Diffrax handle any incompatibility errors rather than checking solver types here.
    # Diffrax will raise appropriate errors if a solver doesn't support SDEs.

    # Warn about potential stateful couplings
    # Note: We can't easily detect if update_state() is non-trivial without running it,
    # but delayed coupling is the main stateful case, which we've already checked above.

    # =========================================================================
    # PREPARE COUPLING AND EXTERNAL INPUT DATA
    # =========================================================================

    # Prepare all couplings (get read-only data, ignore state since we can't maintain it)
    prepared_topology = prepare_graph_topology(network.graph)
    coupling_data_dict, _ = network.prepare(
        dt, t0, t1, _prepared_topology=prepared_topology
    )

    # Prepare all external inputs (get read-only data, ignore state)
    external_data_dict, _ = network.prepare_external(dt)

    # Build config structure. Param/graph subtrees are snapshotted so user
    # mutations to the returned config don't leak back into the source
    # network (or contaminate later prepare() calls).
    config = Bunch(
        # Parameters
        dynamics=_snapshot(network.dynamics.params),
        coupling=Bunch(),
        external=Bunch(),
        # Graph
        graph=_snapshot(network.graph),
        # Initial state [n_states, n_nodes]
        initial_state=network.initial_state,
        # Internal data
        _internal=Bunch(
            coupling=coupling_data_dict,
            external=external_data_dict,
            time=Bunch(t0=t0, t1=t1, dt=dt),
        ),
    )

    # Add coupling params
    for name, coupling in network.coupling.items():
        config.coupling[name] = _snapshot(coupling.params)

    # Add external input params
    for name, external in network.externals.items():
        config.external[name] = _snapshot(external.params)

    # Add noise params if present
    if network.noise is not None:
        config.noise = _snapshot(network.noise.params)

    # =========================================================================
    # PRE-COMPILE COUPLING COMPUTATION CLOSURE
    # =========================================================================

    # Build coupling list for fast iteration (avoid dict lookups in vector field)
    coupling_list = []
    for name in network.dynamics.COUPLING_INPUTS.keys():
        if name in network.coupling:
            coupling = network.coupling[name]
            data = coupling_data_dict[name]
            coupling_list.append((name, coupling, data))
        else:
            coupling_list.append((name, None, None))

    n_nodes = network.graph.n_nodes

    def compute_all_couplings(t, network_state, config, coupling_data_dict):
        """Compute all coupling inputs (stateless - no coupling state).

        Args:
            config: Config containing coupling parameters to use
            coupling_data_dict: Per-coupling data (enriched by precompute if applicable)
        """
        coupling_inputs = Bunch()

        for name, coupling, _ in coupling_list:
            if coupling is None:
                # Missing coupling - use zeros
                n_dims = network.dynamics.COUPLING_INPUTS[name]
                coupling_inputs[name] = jnp.zeros((n_dims, n_nodes))
            else:
                # Compute coupling (stateless - pass empty state)
                # For stateless couplings, coupling_state should be ignored
                data = coupling_data_dict[name]
                empty_state = Bunch()
                coupling_inputs[name] = coupling.compute(
                    t,
                    network_state,
                    data,
                    empty_state,
                    config.coupling[name],
                    config.graph,
                )

        return coupling_inputs

    def precompute_all_couplings(config):
        """Call precompute() for each coupling. Runs inside JIT, once per forward pass."""
        enriched = {}
        for name, coupling, static_data in coupling_list:
            if coupling is not None:
                enriched[name] = coupling.precompute(
                    static_data, config.coupling[name], config.graph
                )
            else:
                enriched[name] = None
        return enriched

    # =========================================================================
    # PRE-COMPILE EXTERNAL INPUT COMPUTATION CLOSURE
    # =========================================================================

    # Build external input list
    external_list = []
    for name in network.dynamics.EXTERNAL_INPUTS.keys():
        if name in network.externals:
            external_obj = network.externals[name]
            data = external_data_dict[name]
            external_list.append((name, external_obj, data))
        else:
            external_list.append((name, None, None))

    def compute_all_externals(t, network_state, config):
        """Compute all external inputs (stateless).

        Args:
            config: Config containing external input parameters to use
        """
        external_inputs = Bunch()

        for name, external_obj, data in external_list:
            if external_obj is None:
                # Missing external input - use zeros
                n_dims = network.dynamics.EXTERNAL_INPUTS[name]
                external_inputs[name] = jnp.zeros((n_dims, n_nodes))
            else:
                # Compute external input (stateless - pass empty state)
                empty_state = Bunch()
                external_inputs[name] = external_obj.compute(
                    t, network_state, data, empty_state, config.external[name]
                )

        return external_inputs

    # =========================================================================
    # DYNAMICS REFERENCES (captured statically for closures)
    # =========================================================================

    dynamics_fn = network.dynamics.dynamics
    n_states = network.dynamics.N_STATES

    # =========================================================================
    # DIFFUSION HELPER (outside _f — captures only static objects)
    # =========================================================================

    has_noise = network.noise is not None
    brownian_tol = None
    n_brownian = None
    if has_noise:
        noise_state_indices = network.noise._state_indices
        n_noise_states = len(noise_state_indices)
        n_brownian = n_noise_states * n_nodes
        brownian_tol = dt * 0.01

        compute_diffusion_matrix = _make_diffusion_matrix_fn(
            network.noise.diffusion, noise_state_indices, n_states, n_nodes
        )

        # Expose the PRNG key on the config so callers can swap seeds per
        # call (mirroring the native dispatch). The VirtualBrownianTree is
        # rebuilt inside _f from config.noise.key — see the rationale below.
        config.noise.key = network.noise.key

    # =========================================================================
    # COMPUTE EFFECTIVE SAVE DT FROM SAVEAT
    # =========================================================================
    # Infer the effective output dt from saveat.ts if available (ts-based saveat).
    # This is a Python-level computation done once at prepare time, so it is
    # concrete and usable as static data in monitors (int/round patterns, etc.).
    # None signals that dt cannot be determined; monitors must handle this case.
    effective_save_dt = None
    saveat_ts = getattr(getattr(solver.saveat, "subs", solver.saveat), "ts", None)
    if saveat_ts is not None and len(saveat_ts) > 1:
        effective_save_dt = float(saveat_ts[1] - saveat_ts[0])

    # =========================================================================
    # CREATE SOLVE FUNCTION
    # =========================================================================

    def _f(config):
        """Pure integration function using Diffrax."""
        initial_state = validate_graph_topology(
            prepared_topology, config.graph, config.initial_state
        )

        # Run precompute() for all couplings once before the solve.
        # This allows parameter-dependent quantities (e.g. W_eff = W * wLRE) to
        # be computed with gradient flow while avoiding per-step redundancy.
        enriched = precompute_all_couplings(config)

        def vector_field(t, y, args):
            """Diffrax-compatible vector field: f(t, y, args) -> dy/dt."""
            # Compute coupling inputs using enriched (precomputed) data
            coupling_inputs = compute_all_couplings(t, y, config, enriched)

            # Compute external inputs
            external_inputs = compute_all_externals(t, y, config)

            # Call dynamics
            result = dynamics_fn(
                t, y, config.dynamics, coupling_inputs, external_inputs
            )

            # Extract derivatives (discard auxiliaries if present)
            if isinstance(result, tuple):
                derivatives, _ = result
            else:
                derivatives = result

            return derivatives

        # Create drift term (deterministic dynamics)
        drift_term = diffrax.ODETerm(vector_field)

        # Combine terms
        if has_noise:
            # Rebuild the VirtualBrownianTree inside _f so the PRNG key is
            # drawn from the runtime config rather than captured as a
            # closure variable. This lets callers swap config.noise.key
            # per call (e.g. via eqx.tree_at + jax.vmap) without
            # re-`prepare`-ing the entire solve function.
            brownian_motion = diffrax.VirtualBrownianTree(
                t0=t0,
                t1=t1,
                tol=brownian_tol,
                shape=(n_brownian,),
                key=config.noise.key,
            )

            # SDE: build diffusion term inside _f so it uses runtime config
            def diffusion_vector_field(t, y, args):
                return compute_diffusion_matrix(t, y, config.noise)

            diffusion_term = diffrax.ControlTerm(
                diffusion_vector_field, brownian_motion
            )
            terms = diffrax.MultiTerm(drift_term, diffusion_term)
        else:
            # ODE: just drift
            terms = drift_term

        # Solve using diffrax with 2D state [n_states, n_nodes]
        solution = diffrax.diffeqsolve(
            terms,
            solver.solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=initial_state,
            saveat=solver.saveat,
            stepsize_controller=solver.stepsize_controller,
            max_steps=solver.max_steps,
            **solver.diffrax_kwargs,
        )

        # NOTE: Diffrax may pad solution arrays with inf when max_steps is specified.
        # Users should filter finite values in post-processing if needed:
        #   finite_mask = jnp.isfinite(solution.ts)
        #   solution_filtered = solution.ts[finite_mask], solution.ys[finite_mask]

        return DiffraxSolution(
            solution,
            dt=effective_save_dt,
            variable_names=tuple(network.dynamics.STATE_NAMES),
        )

    return _f, config


@dispatch
def prepare(
    dynamics: AbstractDynamics,
    solver: NativeSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
    n_nodes: int = 1,
    noise=None,
    externals=None,
    reduce=None,
) -> Tuple[Callable, Bunch]:
    """Compile a model into a pure JAX solve function and a config PyTree.

    Builds per-dispatch data (coupling buffers, noise samples, external
    inputs) and returns ``(solve_fn, config)`` where ``solve_fn(config)``
    runs the integration. Dispatches on the first two arguments via
    ``plum``: ``Network``/``AbstractDynamics`` paired with
    ``NativeSolver``/``DiffraxSolver``.

    Parameters
    ----------
    t0, t1, dt : float
        Integration interval and step size. ``dt`` is the fixed step for
        native solvers and the initial step for Diffrax.

    Returns
    -------
    (Callable, Bunch)
        Pure solve function and its runtime configuration PyTree.

    See ``help(prepare)`` or ``prepare.__doc__`` for the full reference,
    including per-dispatch parameters (``n_nodes``, ``noise``, ``externals``
    for bare dynamics) and Diffrax limitations (no delays, no auxiliaries,
    no VOI filtering).
    """
    # Initial state [N_STATES, n_nodes]
    state0 = dynamics.get_default_initial_state(n_nodes)

    # Zero coupling (always — bare dynamics has no coupling)
    zero_coupling = Bunch()
    for name, n_dims in dynamics.COUPLING_INPUTS.items():
        zero_coupling[name] = jnp.zeros((n_dims, n_nodes))

    # Time array
    time_steps = jnp.arange(t0, t1, dt)
    n_steps = len(time_steps)

    # Config. Param subtrees are snapshotted so user mutations to the
    # returned config don't leak back into the source dynamics/noise/externals.
    config = Bunch(
        dynamics=_snapshot(dynamics.params),
        initial_state=state0,
        _internal=Bunch(time=Bunch(t0=t0, t1=t1, dt=dt)),
    )

    # ---- Noise setup ----
    # Noise increments are materialised inside _f via a single
    # jax.random.normal(config.noise.key, ...) call, which XLA fuses
    # with the downstream scan. Users can override by populating
    # config._internal.noise_samples with a pre-sampled trajectory; the
    # scan branches on its presence at trace time.
    has_noise = noise is not None
    if has_noise:
        noise._state_indices = noise._resolve_state_indices(dynamics)
        noise_state_indices = noise._state_indices
        config.noise = _snapshot(noise.params)
        config.noise.key = noise.key
        config._internal.noise_samples = None
        n_noise_states = len(noise_state_indices)
        noise_diffusion = noise.diffusion

    # ---- External inputs setup ----
    has_externals = externals is not None and len(externals) > 0
    if has_externals:
        config.external = Bunch()
        external_list = []
        external_data_dict = Bunch()
        external_state_dict_init = Bunch()

        for name in dynamics.EXTERNAL_INPUTS.keys():
            if name in externals:
                ext_obj = externals[name]
                config.external[name] = _snapshot(ext_obj.params)
                # Pass None as network — parametric externals don't use it
                ext_data, ext_state = ext_obj.prepare(None, dt)
                external_data_dict[name] = ext_data
                external_state_dict_init[name] = ext_state
                external_list.append((name, ext_obj, ext_data))
            else:
                external_list.append((name, None, None))

        config._internal.external = external_data_dict
        config.initial_state = Bunch(
            dynamics=state0,
            external=external_state_dict_init,
        )

        def compute_all_externals(t, state, external_state_dict, config):
            external_inputs = Bunch()
            for name, ext_obj, data in external_list:
                if ext_obj is None:
                    n_dims = dynamics.EXTERNAL_INPUTS[name]
                    external_inputs[name] = jnp.zeros((n_dims, n_nodes))
                else:
                    state_data = external_state_dict[name]
                    external_inputs[name] = ext_obj.compute(
                        t, state, data, state_data, config.external[name]
                    )
            return external_inputs

        def update_all_external_states(external_state_dict, new_state):
            new_states = Bunch()
            for name, ext_obj, data in external_list:
                if ext_obj is not None:
                    new_states[name] = ext_obj.update_state(
                        data, external_state_dict[name], new_state
                    )
            return new_states
    else:
        # Zero external inputs
        zero_external = Bunch()
        for name, n_dims in dynamics.EXTERNAL_INPUTS.items():
            zero_external[name] = jnp.zeros((n_dims, n_nodes))

    # References
    dynamics_fn = dynamics.dynamics
    solver_step = solver.step

    # VOI filtering
    (
        state_voi_indices,
        aux_voi_indices,
        record_auxiliaries,
        variable_names,
    ) = _split_voi(dynamics)

    # Static shape for the full per-call noise tensor.
    noise_samples_shape = (n_steps, n_noise_states, n_nodes) if has_noise else None

    def _f(config):
        """Pure integration function for bare dynamics."""

        # Noise source. Streaming (per-block fold_in) activates when blocking is
        # on, the dynamics has noise, and no full tensor is injected; otherwise
        # the full tensor is drawn once (or the injected tensor used). See the
        # network+native dispatch for the rationale.
        streaming = (
            has_noise
            and solver.block_size is not None
            and config._internal.noise_samples is None
        )
        noise_gen = None
        if streaming:
            noise_samples_all = None
            noise_gen = _streaming_noise_gen(
                config.noise.key,
                (n_noise_states, n_nodes),
                provider=config._internal.get("noise_provider", None),
            )
        elif has_noise:
            noise_samples_all = _materialize_noise(
                config.noise.key,
                config._internal.noise_samples,
                noise_samples_shape,
            )
        else:
            noise_samples_all = None

        def op(carry, scan_input):
            # Unpack per-step driving signals from the scan inputs.
            if has_noise:
                t, noise_raw = scan_input
            else:
                t = scan_input

            # Unpack carry
            if has_externals:
                state = carry.dynamics
                external_state_dict = carry.external
            else:
                state = carry

            def wrapped_dynamics(t_inner, s, params):
                if has_externals:
                    ext_inputs = compute_all_externals(
                        t_inner, s, external_state_dict, config
                    )
                else:
                    ext_inputs = zero_external
                return dynamics_fn(t_inner, s, params, zero_coupling, ext_inputs)

            # Noise: ``noise_raw`` is the per-step slice from the scan inputs.
            if has_noise:
                diffusion = noise_diffusion(t, state, config.noise)
                scaled_noise = diffusion * jnp.sqrt(dt) * noise_raw
                noise_sample = jnp.zeros_like(state)
                noise_sample = noise_sample.at[noise_state_indices].set(scaled_noise)
            else:
                noise_sample = jnp.zeros_like(state)

            next_state, auxiliaries = solver_step(
                wrapped_dynamics, t, state, dt, config.dynamics, noise_sample
            )

            # VOI filtering
            output = _assemble_output(
                next_state,
                auxiliaries,
                state_voi_indices,
                aux_voi_indices,
                record_auxiliaries,
            )

            # Update carry
            if has_externals:
                next_external = update_all_external_states(
                    external_state_dict, next_state
                )
                next_carry = Bunch(dynamics=next_state, external=next_external)
            else:
                next_carry = next_state

            return next_carry, output

        # Per-step driving signals as the scan xs: time alone for ODE (and
        # streaming SDE, whose noise is generated per block in-scan), or
        # (time, noise_slice) for presampled SDE.
        if not has_noise or streaming:
            scan_inputs = time_steps
        else:
            scan_inputs = (time_steps, noise_samples_all)
        # Single scan seam; dispatches on the solver's block knob, the streaming
        # noise source, and the reduce fold.
        fold = _reduce_fold(reduce, variable_names, n_nodes, n_steps)
        final_carry, res = run_scan(
            op,
            config.initial_state,
            scan_inputs,
            n_steps,
            solver,
            fold=fold,
            noise_gen=noise_gen,
        )
        if reduce is not None:
            _init, update, finalize = reduce
            if solver.block_size is None:
                acc = update(fold[0], res)
            else:
                acc = final_carry[1]
            return finalize(acc)
        return wrap_native_result(res, t0, t1, dt, variable_names=variable_names)

    return _f, config


@dispatch
def prepare(
    dynamics: AbstractDynamics,
    solver: DiffraxSolver,
    t0: float = 0.0,
    t1: float = 1.0,
    dt: float = 0.1,
    n_nodes: int = 1,
    noise=None,
    externals=None,
    reduce=None,
) -> Tuple[Callable, Bunch]:
    """Compile a model into a pure JAX solve function and a config PyTree.

    Builds per-dispatch data (coupling buffers, noise samples, external
    inputs) and returns ``(solve_fn, config)`` where ``solve_fn(config)``
    runs the integration. Dispatches on the first two arguments via
    ``plum``: ``Network``/``AbstractDynamics`` paired with
    ``NativeSolver``/``DiffraxSolver``.

    Parameters
    ----------
    t0, t1, dt : float
        Integration interval and step size. ``dt`` is the fixed step for
        native solvers and the initial step for Diffrax.

    Returns
    -------
    (Callable, Bunch)
        Pure solve function and its runtime configuration PyTree.

    See ``help(prepare)`` or ``prepare.__doc__`` for the full reference,
    including per-dispatch parameters (``n_nodes``, ``noise``, ``externals``
    for bare dynamics) and Diffrax limitations (no delays, no auxiliaries,
    no VOI filtering).
    """
    # reduce is a native-only feature (it rides on the native block scan); plum
    # dispatches on positional args only, so reject a stray reduce= explicitly.
    if reduce is not None:
        raise ValueError(
            "reduce is only supported by NativeSolver, not DiffraxSolver "
            "(it rides on the native block scan). Use a NativeSolver."
        )

    # Initial state [N_STATES, n_nodes]
    state0 = dynamics.get_default_initial_state(n_nodes)

    # Zero coupling (always)
    zero_coupling = Bunch()
    for name, n_dims in dynamics.COUPLING_INPUTS.items():
        zero_coupling[name] = jnp.zeros((n_dims, n_nodes))

    # Config. Param subtrees are snapshotted so user mutations to the
    # returned config don't leak back into the source dynamics/noise/externals.
    config = Bunch(
        dynamics=_snapshot(dynamics.params),
        initial_state=state0,
        _internal=Bunch(time=Bunch(t0=t0, t1=t1, dt=dt)),
    )

    # References
    dynamics_fn = dynamics.dynamics
    n_states = dynamics.N_STATES

    # ---- Noise setup ----
    has_noise = noise is not None
    brownian_tol = None
    n_brownian = None
    if has_noise:
        noise._state_indices = noise._resolve_state_indices(dynamics)
        noise_state_indices = noise._state_indices
        config.noise = _snapshot(noise.params)
        config.noise.key = noise.key
        n_noise_states = len(noise_state_indices)
        n_brownian = n_noise_states * n_nodes
        brownian_tol = dt * 0.01

        compute_diffusion_matrix = _make_diffusion_matrix_fn(
            noise.diffusion, noise_state_indices, n_states, n_nodes
        )

    # ---- External inputs setup ----
    has_externals = externals is not None and len(externals) > 0
    if has_externals:
        config.external = Bunch()
        external_list = []
        external_data_dict = Bunch()

        for name in dynamics.EXTERNAL_INPUTS.keys():
            if name in externals:
                ext_obj = externals[name]
                config.external[name] = _snapshot(ext_obj.params)
                ext_data, _ = ext_obj.prepare(None, dt)
                external_data_dict[name] = ext_data
                external_list.append((name, ext_obj, ext_data))
            else:
                external_list.append((name, None, None))

        config._internal.external = external_data_dict

        def compute_all_externals(t, state, config):
            external_inputs = Bunch()
            for name, ext_obj, data in external_list:
                if ext_obj is None:
                    n_dims = dynamics.EXTERNAL_INPUTS[name]
                    external_inputs[name] = jnp.zeros((n_dims, n_nodes))
                else:
                    empty_state = Bunch()
                    external_inputs[name] = ext_obj.compute(
                        t, state, data, empty_state, config.external[name]
                    )
            return external_inputs
    else:
        zero_external = Bunch()
        for name, n_dims in dynamics.EXTERNAL_INPUTS.items():
            zero_external[name] = jnp.zeros((n_dims, n_nodes))

    # Effective save dt from saveat
    effective_save_dt = None
    saveat_ts = getattr(getattr(solver.saveat, "subs", solver.saveat), "ts", None)
    if saveat_ts is not None and len(saveat_ts) > 1:
        effective_save_dt = float(saveat_ts[1] - saveat_ts[0])

    def _f(config):
        """Pure integration function using Diffrax for bare dynamics."""

        def vector_field(t, y, args):
            if has_externals:
                ext_inputs = compute_all_externals(t, y, config)
            else:
                ext_inputs = zero_external
            result = dynamics_fn(t, y, config.dynamics, zero_coupling, ext_inputs)
            if isinstance(result, tuple):
                derivatives, _ = result
            else:
                derivatives = result
            return derivatives

        drift_term = diffrax.ODETerm(vector_field)

        if has_noise:
            # Rebuild the VirtualBrownianTree per call so it draws its key
            # from config.noise.key — see the Network+Diffrax dispatch for
            # the rationale.
            brownian_motion = diffrax.VirtualBrownianTree(
                t0=t0,
                t1=t1,
                tol=brownian_tol,
                shape=(n_brownian,),
                key=config.noise.key,
            )

            def diffusion_vector_field(t, y, args):
                return compute_diffusion_matrix(t, y, config.noise)

            diffusion_term = diffrax.ControlTerm(
                diffusion_vector_field, brownian_motion
            )
            terms = diffrax.MultiTerm(drift_term, diffusion_term)
        else:
            terms = drift_term

        solution = diffrax.diffeqsolve(
            terms,
            solver.solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=config.initial_state,
            saveat=solver.saveat,
            stepsize_controller=solver.stepsize_controller,
            max_steps=solver.max_steps,
            **solver.diffrax_kwargs,
        )

        return DiffraxSolution(
            solution,
            dt=effective_save_dt,
            variable_names=tuple(dynamics.STATE_NAMES),
        )

    return _f, config


prepare.__doc__ = _PREPARE_DOC
