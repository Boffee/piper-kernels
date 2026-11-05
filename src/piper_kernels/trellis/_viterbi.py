"""Exact Viterbi search over bitshift-trellis walks, the encoder's minimizer.

``compute_walks`` finds, for each sequence of weights, the walk and the choice among
a few candidate scales that minimize the squared error of its reconstruction. Every
scale applies to a whole sequence. The value of state ``s`` at scale ``p1`` with
multiplier ``p2`` is ``(c(s)·p1)·p2`` in float mode, where ``c`` is the 1MAD code
value, or the format's INT8 decode ``clamp(rint(n(s)·alpha), ±127)·p2`` in INT8
mode, where ``n`` is the code numerator and ``alpha`` the folded block scale.

Step ``t`` (``k_t`` new bits, indices cyclic; ``_compute_step_fields``: ``k_t``, ``k_{t+1}``,
``k_{t+2}``) reads ``M[j]``, the least cost of a walk prefix whose last state's low
``L - k_t`` bits are ``j``. For each ``j'``, the low ``L - k_{t+1}`` bits of
``s_t = (a, j')``, it writes the minimum over the ``k_{t+1}`` bits ``a`` the next
shift drops of ``M[s_t >> k_t] + (w_t - value(s_t))²``. A prefix is stored as
``[h, top]`` (``_store_index``), with ``top`` the ``k_{t+2}`` bits the next step
reduces over lowest. A step gathers each state's prefix and code value, which needs
``L >= k_t + k_{t+1}`` for every two steps.

The forward runs as one Triton launch per step (the step loop, ``_step_kernel``) or,
on NVIDIA GPUs that support it, as one persistent Gluon launch for all steps
(``_gluon_forward``). Both pin the same FP32 arithmetic and produce identical
backpointers; a traceback kernel then walks them back. Tail-biting uses QTIP's
Algorithm 4 (``compute_overlap``).
"""

# Triton's JIT launcher and constexpr arguments are not fully represented by its
# Python typing surface.
# pyright: reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917

from __future__ import annotations

import functools
from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.runtime import device_context

from ._layout import CODE_DIVISOR, CODE_RECIPROCAL
from ._reference import compute_code_numerators


@dataclass(frozen=True, slots=True)
class SearchTrellis:
    """A bitshift trellis to search: ``state_bits``-bit states, step ``t`` shifting
    in ``step_widths[t]`` bits, cyclically.

    The stored format's trellis has 16-bit states and its layout's step widths. The
    encoder also searches the same widths with 12-bit states, and windows of them.
    """

    state_bits: int
    step_widths: tuple[int, ...]

    @property
    def min_step_bits(self) -> int:
        """The narrowest step, which sets the widest prefix (``2^(L - min k)`` costs)."""
        return min(self.step_widths)

    def with_state_bits(self, state_bits: int) -> SearchTrellis:
        """The same step widths with ``state_bits``-bit states."""
        return replace(self, state_bits=state_bits)

    def rotate(self, shift: int) -> SearchTrellis:
        """This trellis for sequences rotated right by ``shift``, as ``Tensor.roll``."""
        shift %= len(self.step_widths)
        if shift == 0:
            return self
        widths = self.step_widths
        return replace(self, step_widths=widths[-shift:] + widths[:-shift])

    def crop(self, start_step: int, end_step: int) -> SearchTrellis:
        """The trellis of steps ``start_step`` to ``end_step`` (exclusive), as a sequence."""
        return replace(self, step_widths=self.step_widths[start_step:end_step])


@functools.cache
def _build_code_table(state_bits: int, numerators: bool, device: torch.device) -> torch.Tensor:
    """Every state's 1MAD numerator or code value (FP32), built once per device.

    The code value ``n / CODE_DIVISOR`` is the correctly rounded FP32 quotient, so it
    is divided on the CPU: CUDA divides a tensor by a scalar with a reciprocal
    multiply, which differs in the last bit for many states.
    """
    table = compute_code_numerators(torch.arange(1 << state_bits)).float()
    return (table if numerators else table / CODE_DIVISOR).to(device)


def build_value_table(state_bits: int, device: torch.device) -> torch.Tensor:
    """``c(s)`` for every ``state_bits``-bit state (FP32 ``[2^L]``), for float mode."""
    return _build_code_table(state_bits, False, device)


def build_numerator_table(state_bits: int, device: torch.device) -> torch.Tensor:
    """``n(s)`` for every ``state_bits``-bit state (FP32 ``[2^L]``), for INT8 mode."""
    return _build_code_table(state_bits, True, device)


def fold_block_scales(block_scales: torch.Tensor) -> torch.Tensor:
    """The INT8 decode multiplier ``alpha = fp32(block_scale · R)`` of FP8-valued scales."""
    return block_scales.float() * CODE_RECIPROCAL


def _compute_step_fields(widths: tuple[int, ...], t: int) -> tuple[int, int, int]:
    """``(k_t, k_{t+1}, k_{t+2})`` of step ``t``, cyclically."""
    n = len(widths)
    return widths[t % n], widths[(t + 1) % n], widths[(t + 2) % n]


def _store_index(j: torch.Tensor, bits: int, top: int) -> torch.Tensor:
    """Stored position of the ``bits``-bit prefix index ``j`` whose ``top`` high bits
    the next step reduces over."""
    low = bits - top
    return ((j & ((1 << low) - 1)) << top) | (j >> low)


def _load_index(position: torch.Tensor, bits: int, top: int) -> torch.Tensor:
    """Inverse of ``_store_index``."""
    low = bits - top
    return ((position & ((1 << top) - 1)) << low) | (position >> top)


@triton.jit(do_not_specialize=["rows", "mod"])
def _step_kernel(
    m_ptr, c_ptr, w_ptr, p1_ptr, p2_ptr, out_ptr, bp_ptr, rows, mod,
    int8: tl.constexpr, state_bits: tl.constexpr, e_bits: tl.constexpr, a_bits: tl.constexpr,
    ap_bits: tl.constexpr, bp_width: tl.constexpr, block: tl.constexpr,
    positions: tl.constexpr,
):  # fmt: skip
    """One step for ``block`` rows and ``positions`` stored output positions.

    ``*_bits`` are the widths ``k_t``, ``k_{t+1}``, ``k_{t+2}`` and ``c`` the state
    table in natural order; row ``r`` reads prefix row ``r mod mod``. Each candidate
    state's prefix and table entry are gathered. The arithmetic is pinned (launched
    with ``enable_fp_fusion=False``): ``d = fma(v, -p2, w)`` and
    ``cost = fma(d, d, prefix)``, where ``v`` is ``c·p1`` rounded to FP32 (INT8:
    then rint and clamped to ±127). The minimum over ``a`` is a strict less-than
    scan in ascending ``a``, so ties take the lowest. ``rows`` and ``mod`` are not
    specialized: Triton's variant for values divisible by 16 ran up to 2.1x slower.
    """
    in_bits: tl.constexpr = state_bits - e_bits
    in_low: tl.constexpr = in_bits - a_bits
    out_bits: tl.constexpr = state_bits - a_bits
    out_low: tl.constexpr = out_bits - ap_bits
    r = tl.program_id(0) * block + tl.arange(0, block)
    valid = r < rows
    src = r % mod
    position = tl.program_id(1) * positions + tl.arange(0, positions)
    j = ((position & ((1 << ap_bits) - 1)) << out_low) | (position >> ap_bits)
    w = tl.load(w_ptr + r, mask=valid, other=0.0)[:, None]
    p1 = tl.load(p1_ptr + r, mask=valid, other=0.0)[:, None]
    minus_p2 = -tl.load(p2_ptr + r, mask=valid, other=0.0)[:, None]
    best = tl.zeros((block, positions), tl.float32)
    arg = tl.zeros((block, positions), tl.int32)
    # Triton's typing does not declare static_range iterable.
    for a in tl.static_range(1 << a_bits):  # pyright: ignore[reportGeneralTypeIssues]
        state = (a << out_bits) | j
        prior = state >> e_bits
        stored = ((prior & ((1 << in_low) - 1)) << a_bits) | (prior >> in_low)
        prefix = tl.load(
            m_ptr + src[:, None] * (1 << in_bits) + stored[None, :], mask=valid[:, None], other=0.0
        )
        c = tl.load(c_ptr + state)
        scaled = p1 * c[None, :]
        if int8:
            scaled = tl.minimum(tl.maximum(libdevice.nearbyint(scaled), -127.0), 127.0)
        d = tl.fma(scaled, minus_p2, w)
        total = tl.fma(d, d, prefix)
        if a == 0:
            best = total
        else:
            take = total < best
            best = tl.where(take, total, best)
            arg = tl.where(take, a, arg)
    out = out_ptr + r[:, None] * (1 << out_bits) + position[None, :]
    tl.store(out, best, mask=valid[:, None])
    bp = bp_ptr + r[:, None] * bp_width + position[None, :]
    tl.store(bp, arg.to(tl.uint8), mask=valid[:, None])


# The fastest of 2-8 rows x 64-256 positions x 2-8 warps at L = 12 (4/5-bit steps,
# 32768 sequences, RTX 5090).
_STEP_ROWS = 4
_STEP_POSITIONS = 128
_STEP_WARPS = 4


def _run_step(
    m: torch.Tensor,
    table: torch.Tensor,
    w: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    bp: torch.Tensor,
    int8: bool,
    state_bits: int,
    fields: tuple[int, int, int],
) -> torch.Tensor:
    """One step of the loop: new prefix costs ``[rows, 2^(L - k_{t+1})]`` from ``m``.

    ``w``, ``p1`` and ``p2`` are ``[rows]``; ``bp`` is ``[rows, width]``, whose first
    ``2^(L - k_{t+1})`` columns receive the minimizing ``a``.
    """
    out_width = 1 << (state_bits - fields[1])
    rows = w.shape[0]
    out = torch.empty(rows, out_width, device=w.device)
    positions = min(_STEP_POSITIONS, out_width)
    grid = (triton.cdiv(rows, _STEP_ROWS), out_width // positions)
    e, a, ap = fields
    with device_context(w.device):
        _step_kernel[grid](
            m, table, w, p1, p2, out, bp, rows, m.shape[0],
            int8=int8, state_bits=state_bits, e_bits=e, a_bits=a, ap_bits=ap,
            bp_width=bp.stride(0), block=_STEP_ROWS, positions=positions,
            num_warps=_STEP_WARPS, enable_fp_fusion=False,
        )  # fmt: skip
    return out


@triton.jit
def _traceback_kernel(
    bp_ptr, pick_ptr, j_ptr, widths_ptr, states_ptr, chosen_ptr, batch, rows,
    length: tl.constexpr, picked: tl.constexpr, state_bits: tl.constexpr,
    width: tl.constexpr, block: tl.constexpr,
):  # fmt: skip
    """The whole traceback of ``block`` sequences in one program.

    ``picked``: the sequences chose among several candidates, recorded in ``pick``
    at the last state; ``widths``: ``k_t`` for ``t < length + 2``, cyclic;
    ``width``: the row stride of ``bp`` and ``pick``.
    """
    seq = (tl.program_id(0) * block + tl.arange(0, block)).to(tl.int64)
    valid = seq < batch
    j = tl.load(j_ptr + seq, mask=valid, other=0)
    last = length - 1
    chosen = tl.zeros_like(j)
    if picked:
        low = state_bits - tl.load(widths_ptr + last + 1) - tl.load(widths_ptr + last + 2)
        top = tl.load(widths_ptr + last + 2)
        position = ((j - ((j >> low) << low)) << top) | (j >> low)
        pick = tl.load(pick_ptr + seq * width + position, mask=valid, other=0)
        chosen = pick.to(tl.int64)
    tl.store(chosen_ptr + seq, chosen, mask=valid)
    row = chosen * batch + seq
    for i in range(length):
        t = last - i
        bits = state_bits - tl.load(widths_ptr + t + 1)
        top = tl.load(widths_ptr + t + 2)
        low = bits - top
        position = ((j - ((j >> low) << low)) << top) | (j >> low)
        a = tl.load(bp_ptr + (t * rows + row) * width + position, mask=valid, other=0)
        state = (a.to(tl.int64) << bits) | j
        tl.store(states_ptr + t * batch + seq, state, mask=valid)
        j = state >> tl.load(widths_ptr + t)


_TRACEBACK_SEQUENCES = 64


@functools.cache
def _build_widths_tensor(widths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """``k_t`` for ``t < T + 2``, cyclic, made once per schedule and device."""
    extended = [widths[t % len(widths)] for t in range(len(widths) + 2)]
    return torch.tensor(extended, dtype=torch.int64, device=device)


def _trace_back(
    trellis: SearchTrellis,
    bp: torch.Tensor,
    pick: torch.Tensor,
    j: torch.Tensor,
    num_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(states [T, batch], chosen [batch])`` walked back from the final ``j``.

    ``j`` holds the low ``L - k_0`` bits of each sequence's last state. The chosen
    candidate is read from ``pick [batch, width]`` there, then that candidate's
    backpointers ``bp [T, num_candidates · batch, width]``.
    """
    length, rows, width = bp.shape
    batch = rows // num_candidates
    device = bp.device
    states = torch.empty(length, batch, dtype=torch.int64, device=device)
    chosen = torch.empty(batch, dtype=torch.int64, device=device)
    widths = _build_widths_tensor(trellis.step_widths, device)
    with device_context(device):
        _traceback_kernel[(triton.cdiv(batch, _TRACEBACK_SEQUENCES),)](
            bp, pick, j, widths, states, chosen, batch, rows,
            length=length, picked=num_candidates > 1, state_bits=trellis.state_bits, width=width,
            block=_TRACEBACK_SEQUENCES, num_warps=2,
        )  # fmt: skip
    return states, chosen


def _select_persistent_forward(trellis: SearchTrellis, device: torch.device) -> bool:
    """Whether searches of ``trellis`` on ``device`` take the persistent forward
    rather than the step loop; both give the same results."""
    try:
        from . import _gluon_forward  # noqa: PLC0415 - Gluon is optional
    except ImportError:
        return False
    return _gluon_forward.supports_trellis(
        trellis.state_bits, trellis.step_widths
    ) and _gluon_forward.supports_device(device)


_BACKPOINTER_BYTES = 1 << 32
"""Backpointer bytes per search batch, ``T · rows · 2^(L - min k)``, before rounding to
whole waves: 4 GiB, which holds 4 waves of the persistent forward at L = 16 with 3-
and 4-bit steps (4.28 GB at T = 256). Batching does not change results."""

_PERSISTENT_WAVES = 4
"""Fewest waves per persistent batch with enough rows. A launch's first waves run
slower per wave (RTX 5090, L = 16, 4/5-bit steps: 3.2 ms per wave in a 2-wave launch,
2.7 ms at 3 to 8 waves); 8 waves saved only 1.1-2.1% for 4.3-5.2 GiB more memory."""


def _compute_batch_sequences(
    trellis: SearchTrellis,
    length: int,
    num_sequences: int,
    num_candidates: int,
    wave_rows: int | None,
) -> int:
    """Sequences per batch of ``num_sequences`` with ``num_candidates`` rows each.

    As many as ``_BACKPOINTER_BYTES`` holds; on the persistent forward (``wave_rows``:
    rows per wave) in whole waves, at least ``_PERSISTENT_WAVES``, but no more than
    the sequences need.
    """
    rows = _BACKPOINTER_BYTES // (length << (trellis.state_bits - trellis.min_step_bits))
    if wave_rows is None:
        return max(1, min(num_sequences, rows // num_candidates))
    waves = min(
        max(_PERSISTENT_WAVES, rows // wave_rows), -(-num_sequences * num_candidates // wave_rows)
    )
    return max(1, waves * wave_rows // num_candidates)


def _build_initial_prefix(
    trellis: SearchTrellis, batch: int, overlap: torch.Tensor | None, device: torch.device
) -> torch.Tensor:
    """Prefix costs before step 0, over the low ``L - k_0`` bits of the last state.

    Free (all 0), or constrained to ``overlap [batch]`` (0 there, infinity elsewhere).
    """
    _, a, ap = _compute_step_fields(trellis.step_widths, len(trellis.step_widths) - 1)
    bits = trellis.state_bits - a
    if overlap is None:
        return torch.zeros(batch, 1 << bits, device=device)
    costs = torch.full((batch, 1 << bits), float("inf"), device=device)
    return costs.scatter_(1, _store_index(overlap, bits, ap).unsqueeze(1), 0.0)


def _minimize_candidates(
    costs: torch.Tensor, num_candidates: int, pick: torch.Tensor
) -> torch.Tensor:
    """Final costs ``[num_candidates · batch, width]`` minimized over each sequence's
    candidates, the lowest minimizing candidate stored in ``pick [batch, width]``."""
    if num_candidates == 1:
        return costs
    rows, width = costs.shape
    costs, index = costs.view(num_candidates, rows // num_candidates, width).min(0)
    pick[:, :width].copy_(index)
    return costs


def _search_batch(
    trellis: SearchTrellis,
    int8: bool,
    w: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    overlap: torch.Tensor | None,
    num_candidates: int,
    persistent: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One batch of ``rows = num_candidates · batch`` rows, row ``candidate · batch + sequence``.

    ``w`` is ``[T, rows]``; ``p1`` and ``p2`` are ``[rows]``; ``overlap`` is
    ``[batch]`` or ``None``. Returns ``(states [T, batch], chosen [batch])``.
    """
    length, rows = w.shape
    batch = rows // num_candidates
    widths = trellis.step_widths
    width = 1 << (trellis.state_bits - trellis.min_step_bits)
    bp = torch.empty(length, rows, width, dtype=torch.uint8, device=w.device)
    pick = torch.zeros(batch, width, dtype=torch.uint8, device=w.device)
    if persistent:
        from ._gluon_forward import run_forward  # noqa: PLC0415 - Gluon is optional

        costs = run_forward(
            trellis.state_bits, widths, int8=int8, w=w, p1=p1, p2=p2, overlap=overlap, bp=bp
        )
    else:
        costs = _build_initial_prefix(trellis, batch, overlap, w.device)
        table = (build_numerator_table if int8 else build_value_table)(trellis.state_bits, w.device)
        for t in range(length):
            costs = _run_step(
                costs,
                table,
                w[t],
                p1,
                p2,
                bp[t],
                int8,
                trellis.state_bits,
                _compute_step_fields(widths, t),
            )
    costs = _minimize_candidates(costs, num_candidates, pick)
    _, a, ap = _compute_step_fields(widths, length - 1)
    j = overlap if overlap is not None else _load_index(costs.argmin(1), trellis.state_bits - a, ap)
    return _trace_back(trellis, bp, pick, j, num_candidates)


def compute_walks(
    trellis: SearchTrellis,
    w: torch.Tensor,
    candidates: torch.Tensor,
    p2: torch.Tensor,
    int8: bool,
    overlap: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly optimal walks and scale choices of ``N`` sequences.

    ``w`` is FP32 ``[N, T]`` and finite; ``candidates`` ``[N, S]`` holds each
    sequence's allowed scales ``p1`` (INT8 mode: FP8-valued block scales, folded
    here); ``p2`` is ``[N]``. Returns ``(states [N, T], choice [N])`` minimizing the
    total squared error over every walk and candidate. With ``overlap [N]``, only
    tail-biting walks are allowed whose first state's high ``L - k_0`` bits and last
    state's low ``L - k_0`` bits both equal it. Costs accumulate in FP32; ties take
    the lowest dropped bits and the lowest candidate. Each sequence's result depends
    only on its own inputs.
    """
    num_sequences, length = w.shape
    num_candidates = candidates.shape[1]
    device = w.device
    states_out = torch.empty(num_sequences, length, dtype=torch.int64, device=device)
    choice_out = torch.empty(num_sequences, dtype=torch.int64, device=device)
    if num_sequences == 0:
        return states_out, choice_out
    if int8:
        candidates = fold_block_scales(candidates)
    persistent = _select_persistent_forward(trellis, device)
    wave_rows = None
    if persistent:
        from ._gluon_forward import compute_wave_rows  # noqa: PLC0415 - Gluon is optional

        wave_rows = compute_wave_rows(trellis.state_bits, trellis.step_widths, int8, device)
    batch = _compute_batch_sequences(trellis, length, num_sequences, num_candidates, wave_rows)
    for start_sequence in range(0, num_sequences, batch):
        end_sequence = min(num_sequences, start_sequence + batch)
        span = slice(start_sequence, end_sequence)
        w_batch = w[span].float().T.repeat(1, num_candidates)
        p1 = candidates[span].float().T.reshape(-1)
        p2_batch = p2[span].float().repeat(num_candidates)
        batch_overlap = None if overlap is None else overlap[span].contiguous()
        states, chosen = _search_batch(
            trellis, int8, w_batch, p1, p2_batch, batch_overlap, num_candidates, persistent
        )
        states_out[span] = states.T
        choice_out[span] = chosen
    return states_out, choice_out


def compute_overlap(
    trellis: SearchTrellis,
    w: torch.Tensor,
    scale: torch.Tensor,
    p2: torch.Tensor,
    int8: bool,
    window: int | None = None,
) -> torch.Tensor:
    """QTIP Algorithm 4's overlap of each sequence for a tail-biting search.

    The unconstrained optimal walk of the sequence rotated right by ``T/2``, at
    ``scale [N]``; the result is the high ``L - k_0`` bits of its state at original
    position 0. ``window``, from 1 to ``T/2``, limits that walk to the ``2·window``
    original positions around the wrap point.
    """
    length = w.shape[1]
    half = length // 2
    span = half if window is None else window
    start_step, end_step = half - span, half + span
    rolled = trellis.rotate(half).crop(start_step, end_step)
    rolled_w = w.roll(half, 1)[:, start_step:end_step]
    states, _ = compute_walks(rolled, rolled_w, scale[:, None], p2, int8)
    return states[:, span] >> trellis.step_widths[0]


__all__ = [
    "SearchTrellis",
    "build_numerator_table",
    "build_value_table",
    "compute_overlap",
    "compute_walks",
    "fold_block_scales",
]
