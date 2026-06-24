"""Solving system for network architecture.

This module provides the prepare-solve pattern for Network with multi-coupling
support. The prepare() function sets up the integration with all coupling state
management, and returns a pure function for execution.
"""

import warnings
from typing import Callable, Tuple

import diffrax
import jax
import jax.numpy as jnp
from plum import dispatch

from .core.bunch import Bunch
from .core.network import Network
from .dynamics.base import AbstractDynamics
from .result import DiffraxSolution, wrap_native_result
from .solvers.diffrax import DiffraxSolver
from .solvers.native import NativeSolver


def _snapshot(tree):
    """Structurally copy a PyTree's containers, sharing leaves.

    Rebuilds every PyTree container (Bunch, dataclass, equinox Module, ...)
    so mutations to the returned tree don't leak back to ``tree``. Leaves
    (arrays, scalars, PRNG keys) are immutable in JAX, so sharing them is
    safe and avoids duplicating large arrays (graph weights, history
    buffers, pre-sampled noise tensors).
    """
    return jax.tree.map(lambda x: x, tree)


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
    block = jax.checkpoint(lambda state, block_inputs: jax.lax.scan(op, state, block_inputs))
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
        return _block_scan(
            op, state, window_inputs, window_len, block_size
        )

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


def _composed_scan(block_step, carry0, scan_inputs, n_steps, block_size,
                   window_size=None):
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
        f"tbptt_window={window} is not a multiple of block_size={block_size}; "
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
    - ``tbptt_window`` (gradient horizon): if set, run a windowed scan that
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
    window = solver.tbptt_window
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
            block_step, (state0, jnp.array(0)), scan_inputs, n_steps,
            block_size, window,
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
    # Prepare all couplings (creates history buffers, computes indices, etc.)
    coupling_data_dict, coupling_state_dict_init = network.prepare(dt, t0, t1)

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
        state0 = config.initial_state

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
    coupling_data_dict, _ = network.prepare(dt, t0, t1)

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
            y0=config.initial_state,
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
            op, config.initial_state, scan_inputs, n_steps, solver, fold=fold,
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
