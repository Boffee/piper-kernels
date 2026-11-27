"""Triton hybrid execution for ConvRot Trellis linear.

Small batches (``M <= 4``, and Triton GPUs without an optimized ConvRot INT8
W8A8 backend) execute a fused in-register ``Trellis -> A16`` Split-K GEMV
kernel without materializing decoded INT8 weights or per-weight INT8 rounding.
Larger batches with an INT8 backend rotate/quantize inputs once and stream
bounded scratch-decoded N chunks through ``ConvRotInt8`` W8A8 execution.
"""

# Triton's JIT launcher and constexpr options are not fully represented by its typing surface.
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false
# pyright: reportOperatorIssue=false, reportOptionalSubscript=false

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton.primitives import (
    TRITON_FLOAT_POINTER_TYPES,
    TritonCompileFn,
    default_triton_compile,
    splitk_tile_bounds,
)
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._dispatch import empty_linear_output
from piper_kernels.linear._splitk import device_sm_count, select_split_k
from piper_kernels.trellis._layout import (
    WRAP_GUARD_BYTES,
    TrellisLayout,
    default_layout_for_block_size,
    validate_block_size,
)
from piper_kernels.trellis.triton import (
    _CODE_RECIPROCAL,
    _decode_e4m3,
    _decode_group_4_nums,
    _le_step_bits_4,
    compute_row_offsets,
    decode_int8_out,
    decode_variable_rows_int8_out,
)
from piper_kernels.weights.convrot.triton import rotate_input

from . import _backend as convrot_int8_backend

_GEMV_MAX_ROWS = 4
_A16_MAX_BLOCK_M = 8
_SCRATCH_DECODE_MAX_BYTES = 32 * 1024 * 1024
_LARGE_BATCH_SCRATCH_DECODE_MAX_BYTES = 64 * 1024 * 1024
_LARGE_BATCH_SCRATCH_MIN_ROWS = 1024
_REDUCE_BLOCK_N = 256
_SPLITK_MIN_WEIGHT_BYTES = 2 * 1024 * 1024
_SM120_SPLITK_MIN_WEIGHT_BYTES = 4 * 1024 * 1024
_SPLITK_MAX = 8
_SM120_SPLITK_MAX = 16
_SM120_FP32_WORKSPACE_BYTES = 64 * 1024 * 1024
_WRAP_GUARD_BYTES = tl.constexpr(WRAP_GUARD_BYTES)


@triton.jit
def _fma_4steps_block_scaled(
    acc,
    x_all,
    n0,
    n1,
    n2,
    n3,
    mult,
    tile_steps: tl.constexpr,
    block_m: tl.constexpr,
):
    """Accumulate block-scaled 4-step dot products into ``acc`` without per-weight rounding."""
    num_groups: tl.constexpr = tile_steps // 4
    if block_m == 1:
        xe, xo = tl.split(tl.reshape(x_all, (tile_steps // 2, 2)))
        x0, x2 = tl.split(tl.reshape(xe, (num_groups, 2)))
        x1, x3 = tl.split(tl.reshape(xo, (num_groups, 2)))
        acc += (x0 * n0 + x1 * n1 + x2 * n2 + x3 * n3) * mult
    else:
        xe, xo = tl.split(tl.reshape(x_all, (block_m, tile_steps // 2, 2)))
        x0, x2 = tl.split(tl.reshape(xe, (block_m, num_groups, 2)))
        x1, x3 = tl.split(tl.reshape(xo, (block_m, num_groups, 2)))
        acc += (x0 * n0[None, :] + x1 * n1[None, :] + x2 * n2[None, :] + x3 * n3[None, :]) * mult
    return acc


@triton.jit
def _gemv_accumulate_k_blocks(
    x_rot_ptr,
    bitstreams_ptr,
    block_scales_ptr,
    row_blk_base,
    row_base_bytes,
    width64,
    sbit0,
    sbit1,
    sbit2,
    sbit3,
    first_byte,
    m0,
    ms,
    m_mask,
    pid_k,
    blocks_per_row,
    limit_bytes,
    block_size,
    stride_xm,
    split_k,
    non_empty,
    tile_steps: tl.constexpr,
    block_m: tl.constexpr,
    exact_pow2: tl.constexpr,
    needs_w2: tl.constexpr,
    use_dp4a: tl.constexpr,
    has_tail_bytes: tl.constexpr,
    ptr_aligned: tl.constexpr,
    x_mul4: tl.constexpr,
    unroll_k: tl.constexpr,
):
    """Run the unified LE unrounded block-scaled K-block accumulation loop for one CTA split."""
    b_start, b_end = splitk_tile_bounds(blocks_per_row, pid_k, split_k)
    u32_ptr = bitstreams_ptr.to(tl.pointer_type(tl.uint32))
    full_words = limit_bytes >> 2
    num_groups: tl.constexpr = tile_steps // 4
    steps_all = tl.arange(0, tile_steps)
    x_step_mask = steps_all < block_size
    x_row_base = m0.to(tl.int64) * stride_xm

    acc = (
        tl.zeros((num_groups,), dtype=tl.float32)
        if block_m == 1
        else tl.zeros((block_m, num_groups), dtype=tl.float32)
    )
    for b in tl.range(  # pyright: ignore[reportGeneralTypeIssues]
        b_start, b_end, loop_unroll_factor=unroll_k
    ):
        blk_idx = row_blk_base + b
        base_byte = row_base_bytes + tl.full((), b, tl.int64) * width64
        n0, n1, n2, n3 = _decode_group_4_nums(
            u32_ptr, bitstreams_ptr, base_byte, sbit0, sbit1, sbit2, sbit3, first_byte,
            full_words, limit_bytes, non_empty, needs_w2=needs_w2, use_dp4a=use_dp4a,
            has_tail_bytes=has_tail_bytes, ptr_aligned=ptr_aligned, evict_first=True,
        )  # fmt: skip
        sc = tl.load(block_scales_ptr + blk_idx, eviction_policy="evict_first")
        mult = _decode_e4m3(sc.to(tl.uint32)) * _CODE_RECIPROCAL

        if block_m == 1:
            base_x = x_row_base + b.to(tl.int64) * block_size + steps_all
            x_offs = tl.multiple_of(base_x, 4) if x_mul4 else base_x
            if exact_pow2:
                x_all = tl.load(x_rot_ptr + x_offs).to(tl.float32)
            else:
                x_all = tl.load(x_rot_ptr + x_offs, mask=x_step_mask, other=0.0).to(tl.float32)
        else:
            x_mask = m_mask[:, None] if exact_pow2 else (m_mask[:, None] & x_step_mask[None, :])
            base_x = (
                ms[:, None].to(tl.int64) * stride_xm
                + (b.to(tl.int64) * block_size + steps_all)[None, :]
            )
            x_offs = tl.multiple_of(base_x, [1, 4]) if x_mul4 else base_x
            x_all = tl.load(x_rot_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)
        acc = _fma_4steps_block_scaled(
            acc, x_all, n0, n1, n2, n3, mult, tile_steps=tile_steps, block_m=block_m
        )
    return acc[None, :] if block_m == 1 else acc


@triton.jit(
    do_not_specialize=[
        "rows",
        "out_features",
        "blocks_per_row",
        "bitstreams_bytes",
        "block_size",
        "step_bits",
        "num_wide_steps",
        "bytes_per_block",
        "stride_xm",
        "stride_om",
        "split_k",
    ]
)
def trellis_gemv_splitk_kernel(
    x_rot_ptr,
    bitstreams_ptr,
    block_scales_ptr,
    row_scale_ptr,
    bias_ptr,
    row_widths_ptr,
    row_offsets_ptr,
    output_ptr,
    rows,
    out_features,
    blocks_per_row,
    bitstreams_bytes,
    block_size,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    stride_xm,
    stride_om,
    split_k,
    tile_steps: tl.constexpr,
    block_m: tl.constexpr,
    is_splitk: tl.constexpr,
    variable_rows: tl.constexpr,
    exact_pow2: tl.constexpr,
    has_bias: tl.constexpr,
    needs_w2: tl.constexpr,
    use_dp4a: tl.constexpr,
    has_tail_bytes: tl.constexpr,
    ptr_aligned: tl.constexpr,
    x_mul4: tl.constexpr,
    unroll_k: tl.constexpr = 1,
):
    """Fused in-register LE Trellis -> A16 Split-K GEMV over one output row per CTA."""
    pid_mn = tl.program_id(0)
    pid_n = pid_mn % out_features
    pid_m = pid_mn // out_features
    pid_k = tl.program_id(1)
    ms = pid_m * block_m + tl.arange(0, block_m)
    m_mask = ms < rows

    denom = tile_steps if exact_pow2 else block_size
    if variable_rows:
        width = tl.load(row_widths_ptr + pid_n).to(tl.int32)
        row_offset = tl.load(row_offsets_ptr + pid_n).to(tl.int64)
        row_base_bytes = row_offset * blocks_per_row
        width64 = width.to(tl.int64)
        eff_code_bits = tl.maximum(width - _WRAP_GUARD_BYTES, 0) * 8
        eff_step_bits = eff_code_bits // denom
        eff_wide_steps = eff_code_bits - eff_step_bits * denom
        non_empty = width > _WRAP_GUARD_BYTES
    else:
        width64 = bytes_per_block.to(tl.int64)
        row_base_bytes = pid_n.to(tl.int64) * blocks_per_row * width64
        eff_step_bits = step_bits
        eff_wide_steps = num_wide_steps
        non_empty = True

    first_step = tl.arange(0, tile_steps // 4) * 4
    sbit0, sbit1, sbit2, sbit3 = _le_step_bits_4(
        first_step, eff_step_bits, eff_wide_steps, denom, block_size, exact_pow2
    )
    first_byte = sbit0 >> 3
    row_blk_base = pid_n.to(tl.int64) * blocks_per_row

    acc = _gemv_accumulate_k_blocks(
        x_rot_ptr, bitstreams_ptr, block_scales_ptr, row_blk_base, row_base_bytes, width64,
        sbit0, sbit1, sbit2, sbit3, first_byte, pid_m, ms, m_mask, pid_k, blocks_per_row,
        bitstreams_bytes.to(tl.int64), block_size, stride_xm, split_k, non_empty,
        tile_steps=tile_steps, block_m=block_m, exact_pow2=exact_pow2, needs_w2=needs_w2,
        use_dp4a=use_dp4a, has_tail_bytes=has_tail_bytes, ptr_aligned=ptr_aligned,
        x_mul4=x_mul4, unroll_k=unroll_k,
    )  # fmt: skip

    sums = tl.sum(acc, axis=1)
    n_valid = m_mask & (pid_n < out_features)
    if is_splitk:
        partial_offsets = (pid_k.to(tl.int64) * rows + ms.to(tl.int64)) * out_features + pid_n
        tl.store(output_ptr + partial_offsets, sums, mask=n_valid)
    else:
        row_scale = tl.load(row_scale_ptr + pid_n, mask=pid_n < out_features, other=0.0).to(
            tl.float32
        )
        result = sums * row_scale
        if has_bias:
            result += tl.load(bias_ptr + pid_n, mask=pid_n < out_features, other=0.0).to(tl.float32)
        tl.store(
            output_ptr + ms.to(tl.int64) * stride_om + pid_n,
            result.to(output_ptr.dtype.element_ty),
            mask=n_valid,
        )


@triton.jit(do_not_specialize=["rows", "out_features", "split_k"])
def trellis_splitk_reduce_kernel(
    partial_ptr,
    row_scale_ptr,
    bias_ptr,
    output_ptr,
    rows,
    out_features,
    split_k,
    block_n: tl.constexpr,
    has_bias: tl.constexpr,
):
    """Reduce Split-K partial sums in fixed K order and apply row scale + bias epilogue."""
    pid_n = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features

    acc = tl.zeros((block_n,), dtype=tl.float32)
    split_stride = tl.full((), rows, tl.int64) * out_features
    ptr = partial_ptr + row * out_features + n_offs_i64
    for _ in range(split_k):
        acc += tl.load(ptr, mask=n_valid, other=0.0)
        ptr += split_stride

    acc *= tl.load(row_scale_ptr + n_offs_i64, mask=n_valid, other=0.0).to(tl.float32)
    if has_bias:
        acc += tl.load(bias_ptr + n_offs_i64, mask=n_valid, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * out_features + n_offs_i64,
        acc.to(output_ptr.dtype.element_ty),
        mask=n_valid,
    )


@dataclass(frozen=True, slots=True)
class _TrellisSchedule:
    """Resolved ConvRot Trellis A16 GEMV execution schedule for one target and workload."""

    block_m: int
    num_warps: int
    split_k: int
    unroll_k: int
    use_dp4a: bool
    num_stages: int = 1


def _select_block_m(rows: int) -> int:
    if rows == 1:
        return 1
    if rows == 2:
        return 2
    if rows <= 4:
        return 4
    return _A16_MAX_BLOCK_M


def _select_generic_split_k(
    num_sms: int,
    rows: int,
    out_features: int,
    blocks_per_row: int,
    weight_bytes: int,
    block_m: int,
) -> int:
    if rows > _A16_MAX_BLOCK_M:
        # Large-M fallback (devices without an optimized W8A8 backend) already fills SMs on M x N.
        return 1
    return select_split_k(
        num_sms,
        output_tiles=out_features * triton.cdiv(rows, block_m),
        k_tiles=blocks_per_row,
        weight_bytes=weight_bytes,
        min_weight_bytes=_SPLITK_MIN_WEIGHT_BYTES,
        max_split_k=_SPLITK_MAX,
        cta_multiplier=4,
        min_k_tiles=4,
        min_tiles_per_split=2,
    )


def _select_trellis_schedule(
    target: AcceleratorTarget,
    num_sms: int,
    rows: int,
    out_features: int,
    blocks_per_row: int,
    weight_bytes: int,
) -> _TrellisSchedule:
    block_m = _select_block_m(rows)
    use_dp4a = target.is_nvidia_cuda
    is_sm120 = target.is_cuda_capability(12)
    unroll_k = 2 if (is_sm120 and block_m <= 2) else 1
    if is_sm120:
        if rows > _A16_MAX_BLOCK_M:
            return _TrellisSchedule(
                block_m=block_m,
                num_warps=2,
                split_k=1,
                unroll_k=1,
                use_dp4a=use_dp4a,
            )
        max_split_k = min(
            _SM120_SPLITK_MAX,
            max(1, _SM120_FP32_WORKSPACE_BYTES // max(1, rows * out_features * 4)),
        )
        output_tiles = out_features * triton.cdiv(rows, block_m)
        min_weight_bytes = (
            _SPLITK_MIN_WEIGHT_BYTES if output_tiles <= num_sms else _SM120_SPLITK_MIN_WEIGHT_BYTES
        )
        split_k = (
            select_split_k(
                num_sms,
                output_tiles=output_tiles,
                k_tiles=blocks_per_row,
                weight_bytes=weight_bytes,
                min_weight_bytes=min_weight_bytes,
                max_split_k=max_split_k,
                cta_multiplier=100,
                min_k_tiles=8,
                min_tiles_per_split=8,
                round_up=False,
            )
            if max_split_k > 1
            else 1
        )
        return _TrellisSchedule(
            block_m=block_m,
            num_warps=1,
            split_k=split_k,
            unroll_k=unroll_k,
            use_dp4a=use_dp4a,
        )
    split_k = _select_generic_split_k(
        num_sms, rows, out_features, blocks_per_row, weight_bytes, block_m
    )
    return _TrellisSchedule(
        block_m=block_m,
        num_warps=1,
        split_k=split_k,
        unroll_k=1,
        use_dp4a=use_dp4a,
    )


def linear_a16(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Execute ConvRot Trellis linear via fused in-register A16 Split-K GEMV."""
    out_features, blocks_per_row = block_scales.shape
    in_features = blocks_per_row * block_size
    empty = empty_linear_output(input, out_features, in_features, bias)
    if empty is not None:
        return empty
    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)

    input_2d = input.reshape(rows, in_features).contiguous()
    output = torch.empty((rows, out_features), device=input.device, dtype=input.dtype)
    rotated = torch.empty((rows, in_features), device=input.device, dtype=torch.float32)
    rotate_input(input_2d, rotated, group_size, num_warps=4)

    variable_rows = row_widths is not None
    if variable_rows:
        step_bits = num_wide_steps = bytes_per_block = 0
        needs_w2 = True
        bitstreams_ptr = bitstreams
    else:
        layout = TrellisLayout.from_storage(block_size, bitstreams.shape[2])
        step_bits = layout.step_bits
        num_wide_steps = layout.num_wide_steps
        bytes_per_block = layout.bytes_per_block
        needs_w2 = layout.block_bits > 5 * layout.block_size
        bitstreams_ptr = bitstreams.view(-1)

    tile_steps = triton.next_power_of_2(block_size)
    schedule = _select_trellis_schedule(
        AcceleratorTarget.from_device(input.device),
        device_sm_count(input.device),
        rows,
        out_features,
        blocks_per_row,
        bitstreams_ptr.numel(),
    )
    ptr_aligned = bitstreams_ptr.is_contiguous() and bitstreams_ptr.data_ptr() % 4 == 0
    has_tail_bytes = bitstreams_ptr.numel() % 4 != 0
    x_mul4 = block_size % 4 == 0 and rotated.stride(0) % 4 == 0
    is_splitk = schedule.split_k > 1
    exact_pow2 = block_size == tile_steps
    has_bias = bias is not None
    row_scale = scale.reshape(-1)
    target = (
        torch.empty(
            (schedule.split_k, rows, out_features),
            device=input.device,
            dtype=torch.float32,
        )
        if is_splitk
        else output
    )
    grid = (out_features * triton.cdiv(rows, schedule.block_m), schedule.split_k)
    with device_context(input.device):
        row_offsets = compute_row_offsets(row_widths) if row_widths is not None else None
        trellis_gemv_splitk_kernel[grid](
            rotated,
            bitstreams_ptr,
            block_scales.view(torch.uint8),
            row_scale,
            bias,
            row_widths,
            row_offsets,
            target,
            rows,
            out_features,
            blocks_per_row,
            bitstreams_ptr.numel(),
            block_size,
            step_bits,
            num_wide_steps,
            bytes_per_block,
            rotated.stride(0),
            output.stride(0),
            schedule.split_k,
            tile_steps=tile_steps,
            block_m=schedule.block_m,
            is_splitk=is_splitk,
            variable_rows=variable_rows,
            exact_pow2=exact_pow2,
            has_bias=has_bias,
            needs_w2=needs_w2,
            use_dp4a=schedule.use_dp4a,
            has_tail_bytes=has_tail_bytes,
            ptr_aligned=ptr_aligned,
            x_mul4=x_mul4,
            unroll_k=schedule.unroll_k,
            num_warps=schedule.num_warps,
            num_stages=schedule.num_stages,
        )
        if is_splitk:
            reduce_grid = (triton.cdiv(out_features, _REDUCE_BLOCK_N), rows)
            trellis_splitk_reduce_kernel[reduce_grid](
                target,
                row_scale,
                bias,
                output,
                rows,
                out_features,
                schedule.split_k,
                block_n=_REDUCE_BLOCK_N,
                has_bias=has_bias,
                num_warps=4,
            )
    return output.reshape(*leading_shape, out_features)


def _linear_w8a8(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
    *,
    int8_backend: convrot_int8_backend.LinearBackend,
) -> torch.Tensor:
    """Execute prefill ConvRot Trellis linear via bounded N-chunked scratch decode + W8A8."""
    out_features, blocks_per_row = block_scales.shape
    in_features = blocks_per_row * block_size
    empty = empty_linear_output(input, out_features, in_features, bias)
    if empty is not None:
        return empty
    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)

    input_2d = input.reshape(rows, in_features).contiguous()
    output = torch.empty((rows, out_features), device=input.device, dtype=input.dtype)
    input_qdata, input_scale = int8_backend.prepare_input(input_2d, group_size)

    scratch_budget = (
        _LARGE_BATCH_SCRATCH_DECODE_MAX_BYTES
        if rows >= _LARGE_BATCH_SCRATCH_MIN_ROWS
        else _SCRATCH_DECODE_MAX_BYTES
    )
    raw_chunk_rows = max(1, scratch_budget // in_features)
    chunk_align = 256 if raw_chunk_rows >= 256 else 128
    if raw_chunk_rows < out_features and raw_chunk_rows >= chunk_align:
        chunk_rows = (raw_chunk_rows // chunk_align) * chunk_align
    else:
        chunk_rows = min(out_features, raw_chunk_rows)
    q_scratch = torch.empty((chunk_rows, in_features), device=input.device, dtype=torch.int8)
    block_scales_u8 = block_scales.view(torch.uint8)

    if row_widths is None:
        layout = TrellisLayout.from_storage(block_size, bitstreams.shape[2])
        row_offsets = None
    else:
        layout = None
        with device_context(input.device):
            row_offsets = compute_row_offsets(row_widths)

    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        q_view = q_scratch[: end - start]
        if row_widths is None:
            assert layout is not None
            decode_int8_out(
                bitstreams[start:end].view(-1, layout.bytes_per_block),
                block_scales_u8[start:end].view(-1),
                layout,
                q_view,
            )
        else:
            assert row_offsets is not None
            decode_variable_rows_int8_out(
                bitstreams,
                block_scales_u8[start:end],
                row_widths[start:end],
                block_size,
                q_view,
                row_offsets=row_offsets[start:end],
            )
        chunk_bias = None if bias is None else bias[start:end]
        int8_backend.linear_prepared(
            input_qdata,
            input_scale,
            q_view,
            scale[start:end],
            chunk_bias,
            input.dtype,
            out=output[:, start:end],
        )
    return output.reshape(*leading_shape, out_features)


def linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch ConvRot Trellis linear between fused A16 GEMV and W8A8 prefill."""
    rows = math.prod(input.shape[:-1])
    int8_backend = (
        convrot_int8_backend.select_linear_backend(input) if rows > _GEMV_MAX_ROWS else None
    )
    if int8_backend is not None:
        return _linear_w8a8(
            input,
            bitstreams,
            block_scales,
            scale,
            row_widths,
            bias,
            block_size,
            group_size,
            int8_backend=int8_backend,
        )
    return linear_a16(
        input,
        bitstreams,
        block_scales,
        scale,
        row_widths,
        bias,
        block_size,
        group_size,
    )


def compile_trellis_kernels(
    target: GPUTarget,
    *,
    rows: int,
    out_features: int,
    input_features: int,
    block_size: int,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype | None = None,
    variable_rows: bool = False,
    bytes_per_block: int | None = None,
    num_sms: int = 64,
    compile_fn: TritonCompileFn = default_triton_compile,
) -> dict[str, triton.compiler.CompiledKernel]:
    """Compile the production Trellis A16 GEMV (and Split-K reduce) kernels for a target."""
    if rows <= 0 or out_features <= 0 or input_features <= 0 or num_sms <= 0:
        raise ValueError("Workload dimensions and SM count must be positive integers")
    validate_block_size(block_size)
    if input_features % block_size:
        raise ValueError(
            f"input_features {input_features} must be divisible by block_size {block_size}"
        )
    if input_dtype not in TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported Trellis input dtype: {input_dtype}")
    if bias_dtype is not None and bias_dtype not in TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported Trellis bias dtype: {bias_dtype}")

    out_ty = TRITON_FLOAT_POINTER_TYPES[input_dtype]
    has_bias = bias_dtype is not None
    bias_signature = (
        {} if bias_dtype is None else {"bias_ptr": TRITON_FLOAT_POINTER_TYPES[bias_dtype]}
    )
    bias_constexpr: dict[str, object] = {"bias_ptr": None} if bias_dtype is None else {}
    blocks_per_row = input_features // block_size
    layout = default_layout_for_block_size(block_size, bytes_per_block)
    effective_bytes = layout.bytes_per_block
    weight_bytes = out_features * blocks_per_row * effective_bytes
    needs_w2 = True if variable_rows else (layout.block_bits > 5 * layout.block_size)

    tile_steps = triton.next_power_of_2(block_size)
    schedule = _select_trellis_schedule(
        AcceleratorTarget.from_compiler_target(target),
        num_sms,
        rows,
        out_features,
        blocks_per_row,
        weight_bytes,
    )
    is_splitk = schedule.split_k > 1
    exact_pow2 = block_size == tile_steps

    gemv_signature: dict[str, str] = {
        "x_rot_ptr": "*fp32",
        "bitstreams_ptr": "*u8",
        "block_scales_ptr": "*u8",
        "row_scale_ptr": "*fp32",
        **bias_signature,
        "output_ptr": "*fp32" if is_splitk else out_ty,
        "rows": "i32",
        "out_features": "i32",
        "blocks_per_row": "i32",
        "bitstreams_bytes": "i64",
        "block_size": "i32",
        "step_bits": "i32",
        "num_wide_steps": "i32",
        "bytes_per_block": "i32",
        "stride_xm": "i32",
        "stride_om": "i32",
        "split_k": "i32",
    }
    gemv_constexprs: dict[str, object] = {
        **bias_constexpr,
        "tile_steps": tile_steps,
        "block_m": schedule.block_m,
        "is_splitk": is_splitk,
        "variable_rows": variable_rows,
        "exact_pow2": exact_pow2,
        "has_bias": has_bias,
        "needs_w2": needs_w2,
        "use_dp4a": schedule.use_dp4a,
        "has_tail_bytes": (weight_bytes % 4) != 0 or variable_rows,
        "ptr_aligned": True,
        "x_mul4": (block_size % 4 == 0 and input_features % 4 == 0),
        "unroll_k": schedule.unroll_k,
    }
    if variable_rows:
        gemv_signature["row_widths_ptr"] = "*u8"
        gemv_signature["row_offsets_ptr"] = "*i64"
    else:
        gemv_constexprs["row_widths_ptr"] = None
        gemv_constexprs["row_offsets_ptr"] = None

    compile_options: dict[str, int] = {
        "num_warps": schedule.num_warps,
        "num_stages": schedule.num_stages,
    }
    compiled: dict[str, triton.compiler.CompiledKernel] = {
        "gemv": compile_fn(
            ASTSource(trellis_gemv_splitk_kernel, gemv_signature, constexprs=gemv_constexprs),
            target,
            compile_options,
        )
    }
    if is_splitk:
        compiled["reduce"] = compile_fn(
            ASTSource(
                trellis_splitk_reduce_kernel,
                {
                    "partial_ptr": "*fp32",
                    "row_scale_ptr": "*fp32",
                    **bias_signature,
                    "output_ptr": out_ty,
                    "rows": "i32",
                    "out_features": "i32",
                    "split_k": "i32",
                },
                constexprs={
                    **bias_constexpr,
                    "block_n": _REDUCE_BLOCK_N,
                    "has_bias": has_bias,
                },
            ),
            target,
            {"num_warps": 4},
        )
    return compiled


__all__ = [
    "compile_trellis_kernels",
    "linear",
    "linear_a16",
    "trellis_gemv_splitk_kernel",
    "trellis_splitk_reduce_kernel",
]
