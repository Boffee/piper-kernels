"""Coarse-to-fine trellis encoder: rotated FP32 rows to trellis-coded blocks.

The encoder minimizes each block's squared weight error (data-free, no Hessian)
against the format's exact INT8 decode in three stages:

1. **Analytical scale selection** (fused Triton): a 17-point log-grid search of a
   Bennett-style 1MAD distortion + tail-clipping proxy picks each block's target
   scale ``best_s`` and capped peak, then scores all 29 row-scale multipliers
   ``m · s_c0`` (INT8 error plus trellis error at FP8-snapped block scales) to fix
   the row scale ``p2`` and initial FP8 ``block_scale``.
2. **Coarse L=12 cost check:** one cost-only ``FP16x2`` Viterbi pass (``COST_ONLY``,
   no backpointers or traceback) over the 3 nearest E4M3 values (``radius=1``) picks
   each block's winning FP8 scale against actual 1MAD trellis dynamics.
3. **Final L=16 pass & FP8 refit:** QTIP Algorithm 4's wrap overlap over the ``2 · 32``
   positions around the boundary (skipping backpointer writes on the first 32 steps),
   one constrained 16-bit ``FP16x2`` Viterbi pass, and a fused FP32 refit over the 17
   nearest E4M3 scales for the winning walk.
"""

# Triton's JIT launcher and constexpr arguments are not fully represented by its
# Python typing surface.
# pyright: reportCallIssue=false, reportIndexIssue=false, reportArgumentType=false
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.runtime import device_context

from ._gluon_forward import FP16_SCALE, run_viterbi
from ._layout import (
    CODE_RECIPROCAL,
    STATE_BITS,
    WRAP_GUARD_BYTES,
    TrellisLayout,
)
from ._reference import decode_states
from .triton import _decode_lcg_numerator_i32

_MIN_SCALE = 1e-8
_FP8_MIN_POSITIVE = 2.0**-9
_COARSE_STATE_BITS = 12
_OVERLAP_WINDOW = 32
_FP8_SEARCH_RADIUS = 1
_FP8_REFIT_RADIUS = 8
_ROW_MULTIPLIERS = tuple(2.0 ** (i / 16) for i in range(-4, 25))
"""Row-scale candidates ``m · s_c0``: 29 values in 4.4% steps from 0.84 to 2.83."""
_MAX_CHUNK_WEIGHTS = 1 << 24
_MAX_LDLQ_CHUNK_ROWS = 4096
_MAX_LDLQ_CHUNK_WEIGHTS = 1 << 27

_CODE_RECIPROCAL = tl.constexpr(CODE_RECIPROCAL)
_MIN_SCALE_TL = tl.constexpr(_MIN_SCALE)


@functools.cache
def _build_fp8_values(device: torch.device) -> torch.Tensor:
    """All 126 finite positive E4M3 values, ascending (FP32)."""
    values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    values = values[(values > 0) & torch.isfinite(values)].unique().sort().values
    return values.to(device)


def _round_fp8(x: torch.Tensor) -> torch.Tensor:
    """The nearest E4M3 value (half-even, saturating at 448), as FP32."""
    return x.float().to(torch.float8_e4m3fn).float()


def _compute_fp8_window(center: torch.Tensor, radius: int) -> torch.Tensor:
    """Positive E4M3 values within ``radius`` grid steps of ``_round_fp8(center)``."""
    grid = _build_fp8_values(center.device)
    nearest = _round_fp8(center).clamp(min=_FP8_MIN_POSITIVE)
    index = torch.searchsorted(grid, nearest.contiguous())
    offsets = torch.arange(-radius, radius + 1, device=center.device)
    positions = (index.unsqueeze(-1) + offsets).clamp(0, grid.numel() - 1)
    return grid[positions]


@triton.jit
def _estimate_block_scales_kernel(
    w_ptr,
    out_scale_ptr,
    out_cap_amax_ptr,
    num_blocks,
    c_q,
    block_size: tl.constexpr,
    tile_t: tl.constexpr,
):
    """Per-block 17-point log-grid search of the 1MAD Bennett + tail-clipping proxy."""
    bid = tl.program_id(0)
    if bid >= num_blocks:
        return
    t = tl.arange(0, tile_t)
    mask = t < block_size
    w = tl.load(w_ptr + bid * block_size + t, mask=mask, other=0.0).to(tl.float32)
    w_abs = tl.abs(w)
    amax = tl.max(w_abs, axis=0)
    # Normalize by block amax before squaring so large finite weights never overflow FP32.
    u_scale = tl.maximum(amax, _MIN_SCALE_TL)
    u_abs = w_abs / u_scale
    u_amax = amax / u_scale  # pyright: ignore[reportOperatorIssue]
    u_rms = tl.sqrt(tl.sum(u_abs * u_abs, axis=0) / block_size)
    u_floor = _MIN_SCALE_TL / u_scale
    u_lo = tl.maximum(u_rms * 0.85, u_floor)
    u_hi = tl.maximum(tl.maximum(u_rms * 1.35, u_amax / 2.7), u_lo)
    step_mult = tl.exp(tl.log(u_hi / u_lo) * (1.0 / 16.0))

    best_u = tl.maximum(u_rms * 1.05, u_lo)
    best_e = tl.full((), float("inf"), tl.float32)
    u = u_lo
    for _ in range(17):
        z = tl.minimum(u_abs / u, 3.05)
        q_term = tl.sum(tl.where(mask, tl.exp(0.37 * z * z), 0.0), axis=0)
        clip = tl.maximum(u_abs - 3.05 * u, 0.0)
        clip_term = tl.sum(tl.where(mask, clip * clip, 0.0), axis=0)
        e = 0.62 * c_q * (u * u) * q_term + clip_term
        better = e < best_e
        best_e = tl.where(better, e, best_e)
        best_u = tl.where(better, u, best_u)
        u = u * step_mult

    best_s = tl.maximum(best_u * u_scale, _MIN_SCALE_TL)
    tl.store(out_scale_ptr + bid, best_s)
    tl.store(out_cap_amax_ptr + bid, tl.minimum(amax, 3.05 * best_s))


@triton.jit
def _score_row_multipliers_kernel(
    w_ptr,
    sc_ptr,
    eff_s_ptr,
    out_err_ptr,
    num_blocks,
    blocks_per_row,
    c_q,
    block_size: tl.constexpr,
    tile_t: tl.constexpr,
    num_mults: tl.constexpr,
):
    """Score all ``num_mults`` candidate row scales per block in registers."""
    bid = tl.program_id(0)
    if bid >= num_blocks:
        return
    row = bid // blocks_per_row
    # Normalize by a constant positive scale for this row so argmin is invariant and
    # large finite weights never overflow FP32 squaring.
    s0 = tl.load(sc_ptr + row * num_mults).to(tl.float32)
    t = tl.arange(0, tile_t)
    mask = t < block_size
    w = tl.load(w_ptr + bid * block_size + t, mask=mask, other=0.0).to(tl.float32)
    wn = w / s0
    wn_abs = tl.abs(wn)
    for m in range(num_mults):
        sc = tl.load(sc_ptr + row * num_mults + m)
        eff_s = tl.load(eff_s_ptr + bid * num_mults + m)
        q = tl.minimum(tl.maximum(libdevice.nearbyint(w / sc), -127.0), 127.0)
        sc_n = sc / s0
        eff_n = eff_s / s0
        d_i8 = tl.where(mask, wn - q * sc_n, 0.0)
        i8_e = tl.sum(d_i8 * d_i8, axis=0)
        z = tl.minimum(wn_abs / eff_n, 3.05)
        q_term = tl.sum(tl.where(mask, tl.exp(0.37 * z * z), 0.0), axis=0)
        clip = tl.maximum(wn_abs - 3.05 * eff_n, 0.0)
        clip_term = tl.sum(tl.where(mask, clip * clip, 0.0), axis=0)
        t_e = 0.62 * c_q * (eff_n * eff_n) * q_term + clip_term
        tl.store(out_err_ptr + bid * num_mults + m, i8_e + t_e)


@triton.jit
def _refit_fp8_kernel(
    w_ptr,
    states_ptr,
    p2_ptr,
    window_ptr,
    out_scale_ptr,
    num_blocks,
    block_size: tl.constexpr,
    tile_t: tl.constexpr,
    num_cands: tl.constexpr,
):
    """Best FP8 block scale in ``window`` for the fixed ``states`` walk (FP32)."""
    bid = tl.program_id(0)
    if bid >= num_blocks:
        return
    t = tl.arange(0, tile_t)
    mask = t < block_size
    w = tl.load(w_ptr + bid * block_size + t, mask=mask, other=0.0).to(tl.float32)
    st = tl.load(states_ptr + bid * block_size + t, mask=mask, other=0).to(tl.uint32)
    n = _decode_lcg_numerator_i32(st, use_dp4a=True).to(tl.float32)
    p2 = tl.load(p2_ptr + bid).to(tl.float32)
    w_scaled = w / p2

    best_err = tl.full((), float("inf"), tl.float32)
    best_scale = tl.load(window_ptr + bid * num_cands + (num_cands // 2))
    for c in range(num_cands):
        sc = tl.load(window_ptr + bid * num_cands + c)
        q = tl.minimum(tl.maximum(libdevice.rint(n * (sc * _CODE_RECIPROCAL)), -127.0), 127.0)
        d = tl.where(mask, w_scaled - q, 0.0)
        err = tl.sum(d * d, axis=0)
        better = err < best_err
        best_err = tl.where(better, err, best_err)
        best_scale = tl.where(better, sc, best_scale)
    tl.store(out_scale_ptr + bid, best_scale)


def _estimate_initial_scales(
    w_rows: torch.Tensor, layout: TrellisLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytical per-block and per-row scale estimation for ``w_rows [R, K]``."""
    rows, cols = w_rows.shape
    block_size = layout.block_size
    blocks_per_row = cols // block_size
    w = w_rows.reshape(-1, block_size)
    num_blocks = w.shape[0]
    device = w.device
    tile_t = triton.next_power_of_2(block_size)
    c_q = 2.0 ** (-2.0 * layout.block_bits / block_size)

    best_s = torch.empty(num_blocks, dtype=torch.float32, device=device)
    cap_amax = torch.empty(num_blocks, dtype=torch.float32, device=device)
    with device_context(device):
        _estimate_block_scales_kernel[(num_blocks,)](
            w, best_s, cap_amax, num_blocks, c_q, block_size=block_size, tile_t=tile_t, num_warps=4
        )

    s_c0 = (cap_amax.view(rows, blocks_per_row).amax(-1) / 127.0).clamp(min=_MIN_SCALE)
    multipliers = torch.tensor(_ROW_MULTIPLIERS, dtype=torch.float32, device=device)
    sc_all = (s_c0[:, None] * multipliers[None, :]).contiguous()
    sc_blk = sc_all.repeat_interleave(blocks_per_row, dim=0)
    ratio = (best_s[:, None] / sc_blk).clamp(min=_FP8_MIN_POSITIVE)
    eff_s = (_round_fp8(ratio) * sc_blk).contiguous()

    blk_errs = torch.empty(num_blocks, len(_ROW_MULTIPLIERS), dtype=torch.float32, device=device)
    with device_context(device):
        _score_row_multipliers_kernel[(num_blocks,)](
            w,
            sc_all,
            eff_s,
            blk_errs,
            num_blocks,
            blocks_per_row,
            c_q,
            block_size=block_size,
            tile_t=tile_t,
            num_mults=len(_ROW_MULTIPLIERS),
            num_warps=4,
        )
    best_m = blk_errs.view(rows, blocks_per_row, len(_ROW_MULTIPLIERS)).double().sum(1).argmin(-1)
    row_scale = sc_all.gather(1, best_m[:, None]).squeeze(1)
    p2 = row_scale.repeat_interleave(blocks_per_row)
    block_scale = _round_fp8((best_s / p2).clamp(min=_FP8_MIN_POSITIVE))
    return row_scale, p2, block_scale


def _search_and_refit_blocks(
    w_flat: torch.Tensor,
    p2_flat: torch.Tensor,
    block_scale: torch.Tensor,
    layout: TrellisLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stages 2 and 3: L=12 scale search, wrap-overlap L=16 Viterbi, and FP8 refit."""
    block_size = layout.block_size
    num_blocks = w_flat.shape[0]
    widths = layout.compute_step_widths()
    # Reversing step order aligns Viterbi's left-shift transitions with little-endian windows.
    viterbi_widths = (widths[0], *widths[:0:-1])
    w_norm_h = ((w_flat / p2_flat[:, None]).clamp(-256.0, 256.0) * FP16_SCALE).to(torch.float16)
    w_rev_h = w_norm_h.flip(1).contiguous()

    # 2. Cost-only L=12 FP8 block-scale refinement (no backpointers).
    cands = _compute_fp8_window(block_scale, _FP8_SEARCH_RADIUS)
    num_cands = cands.shape[1]
    costs_12 = run_viterbi(
        _COARSE_STATE_BITS,
        viterbi_widths,
        w_rev_h.repeat(num_cands, 1),
        (cands.T.reshape(-1) * CODE_RECIPROCAL).contiguous(),
        cost_only=True,
    ).view(num_cands, num_blocks)
    block_scale = cands.gather(1, costs_12.argmin(0)[:, None]).squeeze(1).contiguous()
    alpha_16 = (block_scale * CODE_RECIPROCAL).contiguous()

    # 3. L=16 wrap overlap, constrained L=16 walk, and fused FP8 refit.
    half = block_size // 2
    span = min(_OVERLAP_WINDOW, half)
    rolled_widths = viterbi_widths[-half:] + viterbi_widths[:-half]
    ov_widths = rolled_widths[half - span : half + span]
    ov_w_h = w_rev_h.roll(half, 1)[:, half - span : half + span].contiguous()
    ov_states = run_viterbi(
        STATE_BITS, ov_widths, ov_w_h, alpha_16, cost_only=False, write_bp_from=span
    )
    overlap = (ov_states[:, 0] >> viterbi_widths[0]).contiguous()

    v_states = run_viterbi(
        STATE_BITS, viterbi_widths, w_rev_h, alpha_16, overlap, cost_only=False, write_bp_from=0
    )
    states = v_states.flip(1).contiguous()

    refit_win = _compute_fp8_window(block_scale, _FP8_REFIT_RADIUS).contiguous()
    final_scale = torch.empty_like(block_scale)
    with device_context(w_flat.device):
        _refit_fp8_kernel[(num_blocks,)](
            w_flat,
            states,
            p2_flat,
            refit_win,
            final_scale,
            num_blocks,
            block_size=block_size,
            tile_t=triton.next_power_of_2(block_size),
            num_cands=refit_win.shape[1],
            num_warps=4,
            enable_fp_fusion=False,
        )
    return states, final_scale


def _encode_rows(
    w_rows: torch.Tensor, layout: TrellisLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode finite rotated FP32 rows ``[R, K]`` on an SM80+ CUDA device."""
    row_scale, p2, block_scale = _estimate_initial_scales(w_rows, layout)
    states, final_scale = _search_and_refit_blocks(
        w_rows.reshape(-1, layout.block_size), p2, block_scale, layout
    )
    return states, final_scale, row_scale


def _encode_rows_ldlq(
    w_rows: torch.Tensor,
    layout: TrellisLayout,
    feedback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode ``w_rows [R, K]`` via Block-LDLQ from right to left using ``feedback``."""
    rows, cols = w_rows.shape
    block_size = layout.block_size
    blocks_per_row = cols // block_size
    device = w_rows.device
    tile_t = triton.next_power_of_2(block_size)
    c_q = 2.0 ** (-2.0 * layout.block_bits / block_size)

    row_scale, _, _ = _estimate_initial_scales(w_rows, layout)
    bitstreams = torch.empty(
        (rows, blocks_per_row, layout.bytes_per_block), dtype=torch.uint8, device=device
    )
    block_scales = torch.empty((rows, blocks_per_row), dtype=torch.float8_e4m3fn, device=device)

    w_work = w_rows.clone()
    best_s = torch.empty(rows, dtype=torch.float32, device=device)
    cap_amax = torch.empty(rows, dtype=torch.float32, device=device)

    for block_idx in range(blocks_per_row - 1, -1, -1):
        start = block_idx * block_size
        end = start + block_size
        w_block = w_work[:, start:end].contiguous()
        with device_context(device):
            _estimate_block_scales_kernel[(rows,)](
                w_block,
                best_s,
                cap_amax,
                rows,
                c_q,
                block_size=block_size,
                tile_t=tile_t,
                num_warps=4,
            )
        block_scale = _round_fp8((best_s / row_scale).clamp(min=_FP8_MIN_POSITIVE))
        states, final_scale = _search_and_refit_blocks(w_block, row_scale, block_scale, layout)
        bitstreams[:, block_idx, :] = pack_states(states, layout)
        final_scale_fp8 = final_scale.to(torch.float8_e4m3fn)
        block_scales[:, block_idx] = final_scale_fp8
        if block_idx > 0:
            q_block = decode_states(states, final_scale_fp8).to(torch.float32)
            # Use original rotated weights so (W_orig - W_hat) @ L_tilde = W_target - W_hat.
            delta_block = w_rows[:, start:end] - q_block * row_scale[:, None]
            w_work[:, :start].addmm_(delta_block, feedback[start:end, :start])

    return bitstreams, block_scales, row_scale


@functools.cache
def _build_pack_tables(
    layout: TrellisLayout, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached ``(source_index, shift_amount)`` little-endian bit-packing tables on ``device``."""
    total = layout.block_bits
    sources = [0] * total
    shifts = [0] * total
    widths = layout.compute_step_widths()
    for t, (offset, width) in enumerate(zip(layout.compute_state_offsets(), widths, strict=True)):
        for i in range(width):
            position = (offset + STATE_BITS - width + i) % total
            sources[position] = t
            shifts[position] = STATE_BITS - width + i
    return (
        torch.tensor(sources, device=device).view(-1, 8),
        torch.tensor(shifts, dtype=torch.int32, device=device).view(-1, 8),
    )


def pack_states(states: torch.Tensor, layout: TrellisLayout) -> torch.Tensor:
    """Little-endian bitstreams ``[N, bytes_per_block]`` uint8 with trailing guard bytes."""
    source_index, shift_amount = _build_pack_tables(layout, states.device)
    narrow = states.to(torch.int32)
    code_bytes = layout.code_bytes
    packed = torch.zeros(
        states.shape[0], layout.bytes_per_block, dtype=torch.int32, device=states.device
    )
    for bit in range(8):
        value = (narrow[:, source_index[:, bit]] >> shift_amount[:, bit]) & 1
        packed[:, :code_bytes] |= value << bit
    packed[:, code_bytes : code_bytes + WRAP_GUARD_BYTES] = packed[:, :WRAP_GUARD_BYTES]
    return packed.to(torch.uint8)


def encode_matrix(
    rotated: torch.Tensor,
    layout: TrellisLayout,
    *,
    feedback: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trellis-coded storage of a rotated FP32 matrix ``[R, K]`` on an SM80+ CUDA device.

    Returns ``(bitstreams [R, K/T, bytes] uint8, block_scales [R, K/T] float8_e4m3fn,
    scale [R, 1] float32)``. When ``feedback`` is supplied, encodes blocks from right to left
    via Block-LDLQ error propagation. The values must be finite.
    """
    rows, cols = rotated.shape
    blocks_per_row = cols // layout.block_size
    device = rotated.device
    bitstreams = torch.empty(
        rows, blocks_per_row, layout.bytes_per_block, dtype=torch.uint8, device=device
    )
    block_scales = torch.empty(rows, blocks_per_row, dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty(rows, 1, dtype=torch.float32, device=device)
    if feedback is None:
        chunk_rows = max(1, _MAX_CHUNK_WEIGHTS // cols)
        for start_row in range(0, rows, chunk_rows):
            end_row = min(rows, start_row + chunk_rows)
            states, chunk_block_scales, chunk_scale = _encode_rows(
                rotated[start_row:end_row].contiguous(), layout
            )
            packed = pack_states(states, layout)
            bitstreams[start_row:end_row] = packed.view(end_row - start_row, blocks_per_row, -1)
            block_scales[start_row:end_row] = chunk_block_scales.view(-1, blocks_per_row).to(
                torch.float8_e4m3fn
            )
            scale[start_row:end_row] = chunk_scale[:, None]
    else:
        chunk_rows = max(1, min(_MAX_LDLQ_CHUNK_ROWS, _MAX_LDLQ_CHUNK_WEIGHTS // cols))
        for start_row in range(0, rows, chunk_rows):
            end_row = min(rows, start_row + chunk_rows)
            chunk_bits, chunk_bscales, chunk_scale = _encode_rows_ldlq(
                rotated[start_row:end_row].contiguous(), layout, feedback
            )
            bitstreams[start_row:end_row] = chunk_bits
            block_scales[start_row:end_row] = chunk_bscales
            scale[start_row:end_row] = chunk_scale[:, None]
    return bitstreams, block_scales, scale


__all__ = ["encode_matrix", "pack_states"]
