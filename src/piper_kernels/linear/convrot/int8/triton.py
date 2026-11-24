"""Portable INT8 arithmetic shared by accelerator-owned launchers."""

# Triton specializes Python literals as constexpr arguments inside JIT functions.
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import triton
import triton.language as tl

from piper_kernels._residual.triton import add_residual
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget, rdna4_int8_accumulator_seed
from piper_kernels.linear._splitk import device_sm_count, select_split_k

_INT8_SPLITK_MAX_ROWS = 16
_INT8_MIN_SPLITK_WEIGHT_BYTES = 8 * 1024 * 1024
_INT8_MAX_SPLIT_K = 16
INT8_SPLITK_REDUCE_BLOCK_N = 256
_INT8_SPLITK_MIN_K_TILES = 8
_INT8_SPLITK_MIN_TILES_PER_SPLIT = 4
_INT8_SPLITK_PARTIAL_RATIO_DIVISOR = 64
_INT8_SPLITK_MAX_SAFE_REDUCTION = (2**31 - 1) // (128 * 128)


@triton.jit
def _accumulate_int8_tile(
    input_ptr,
    weight_ptr,
    offsets_m,
    offsets_n,
    m,
    n,
    k,
    k_start,
    k_end,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    aligned_m: tl.constexpr,
    aligned_nk: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
):
    """Accumulate an INT32 matmul tile across K-blocks ``[k_start, k_end)``."""
    offsets_k = tl.arange(0, block_k)
    offsets_m_i64 = offsets_m.to(tl.int64)
    offsets_n_i64 = offsets_n.to(tl.int64)
    offsets_k_i64 = offsets_k.to(tl.int64)
    k_stride = tl.full((), k, tl.int64)
    k_start_i64 = tl.full((), k_start, tl.int64) * block_k
    input_pointers = (
        input_ptr + offsets_m_i64[:, None] * k_stride + (k_start_i64 + offsets_k_i64)[None, :]
    )
    weight_pointers = (
        weight_ptr + offsets_n_i64[None, :] * k_stride + (k_start_i64 + offsets_k_i64)[:, None]
    )
    accumulator = tl.full((block_m, block_n), accumulator_seed, tl.int32)

    for k_offset in range(k_start, k_end):
        if aligned_m and aligned_nk:
            input_values = tl.load(input_pointers)
            weight = tl.load(weight_pointers)
        elif aligned_nk:
            input_values = tl.load(input_pointers, mask=offsets_m[:, None] < m, other=0)
            weight = tl.load(weight_pointers)
        else:
            remaining_k = k - k_offset * block_k
            input_mask = (
                offsets_k[None, :] < remaining_k
                if aligned_m
                else (offsets_m[:, None] < m) & (offsets_k[None, :] < remaining_k)
            )
            input_values = tl.load(input_pointers, mask=input_mask, other=0)
            weight = tl.load(
                weight_pointers,
                mask=(offsets_n[None, :] < n) & (offsets_k[:, None] < remaining_k),
                other=0,
            )
        if accumulator_seed:
            accumulator = tl.dot(input_values, weight, accumulator)
        else:
            accumulator += tl.dot(input_values, weight)
        input_pointers += block_k
        weight_pointers += block_k
    return accumulator


@triton.jit
def _row_scaled_int8_matmul(
    input_ptr,
    weight_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    offsets_m,
    offsets_n,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    aligned_tiles: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
):
    """Return the exact INT32 tile product scaled by input rows, and the weight scales."""
    accumulator = _accumulate_int8_tile(
        input_ptr,
        weight_ptr,
        offsets_m,
        offsets_n,
        m,
        n,
        k,
        0,
        tl.cdiv(k, block_k),
        block_m,
        block_n,
        block_k,
        aligned_tiles,
        aligned_tiles,
        accumulator_seed,
    )

    if aligned_tiles:
        input_scale = tl.load(input_scale_ptr + offsets_m)
        weight_scale = tl.load(weight_scale_ptr + offsets_n)
    else:
        input_scale = tl.load(
            input_scale_ptr + offsets_m,
            mask=offsets_m < m,
            other=0.0,
        )
        weight_scale = tl.load(
            weight_scale_ptr + offsets_n,
            mask=offsets_n < n,
            other=0.0,
        )
    return (accumulator - accumulator_seed).to(tl.float32) * input_scale[:, None], weight_scale


@triton.jit
def scaled_int8_matmul(
    input_ptr,
    weight_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    offsets_m,
    offsets_n,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    aligned_tiles: tl.constexpr,
):
    """Return one FP32 ConvRot projection tile before its output epilogue.

    Inputs are the prepared rowwise-INT8 activation and the rotated rowwise-INT8
    weight. Their FP32 scales are applied after the exact INT32 dot product. The
    caller owns bias handling, logical-dtype rounding, and the final store so the
    same projection can feed either the ordinary linear epilogue or a fused
    attention epilogue.
    """
    row_scaled, weight_scale = _row_scaled_int8_matmul(
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        offsets_m,
        offsets_n,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        aligned_tiles,
    )
    return row_scaled * weight_scale[None, :]


@triton.jit
def _add_bias(result, row_scaled, scaled_weight, scaled_bias, explicit_bias_fma: tl.constexpr):
    if explicit_bias_fma:
        # Per-tile branches can let the compiler sink an identical bias add below the
        # branch, away from its scale multiply, which would round twice.
        return tl.fma(row_scaled, scaled_weight, scaled_bias.to(tl.float32))
    return result + scaled_bias


@triton.jit
def _scale_and_add_bias(
    row_scaled,
    weight_scale,
    bias_ptr,
    second_bias_ptr,
    offsets_n,
    second,
    n,
    block_n: tl.constexpr,
    has_bias: tl.constexpr,
    paired: tl.constexpr,
    second_has_bias: tl.constexpr,
    aligned_n: tl.constexpr,
    explicit_bias_fma: tl.constexpr,
    broadcast_row: tl.constexpr,
):
    scaled_weight = weight_scale[None, :] if broadcast_row else weight_scale
    result = row_scaled * scaled_weight
    if paired and (has_bias or second_has_bias):
        # Select FP32 values rather than pointers: biases may have different dtypes.
        bias = tl.full((block_n,), 0, tl.float32)
        second_bias = tl.full((block_n,), 0, tl.float32)
        if has_bias:
            bias = tl.load(bias_ptr + offsets_n, (offsets_n < n) & ~second, other=0.0).to(
                tl.float32
            )
        if second_has_bias:
            second_bias = tl.load(
                second_bias_ptr + offsets_n, (offsets_n < n) & second, other=0.0
            ).to(tl.float32)
        bias = tl.where(second, second_bias, bias)
        scaled_bias = bias[None, :] if broadcast_row else bias
        result = _add_bias(result, row_scaled, scaled_weight, scaled_bias, explicit_bias_fma)
    elif has_bias:
        if aligned_n:
            bias = tl.load(bias_ptr + offsets_n)
        else:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < n, other=0.0)
        scaled_bias = bias[None, :] if broadcast_row else bias
        result = _add_bias(result, row_scaled, scaled_weight, scaled_bias, explicit_bias_fma)
    return result


@triton.jit
def _int8_matmul_with_bias(
    input_ptr,
    weight_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    second_bias_ptr,
    offsets_m,
    offsets_n,
    second,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
    paired: tl.constexpr,
    second_has_bias: tl.constexpr,
    aligned_tiles: tl.constexpr,
    explicit_bias_fma: tl.constexpr,
    accumulator_seed: tl.constexpr,
):
    # Keep scaling and bias together in each branch to preserve FMA rounding.
    row_scaled, weight_scale = _row_scaled_int8_matmul(
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        offsets_m,
        offsets_n,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        aligned_tiles,
        accumulator_seed,
    )
    return _scale_and_add_bias(
        row_scaled,
        weight_scale,
        bias_ptr,
        second_bias_ptr,
        offsets_n,
        second,
        n,
        block_n,
        has_bias,
        paired,
        second_has_bias,
        aligned_tiles,
        explicit_bias_fma,
        True,
    )


@triton.jit
def _project_tile(
    input_ptr,
    weight_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    second_weight_ptr,
    second_scale_ptr,
    second_bias_ptr,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
    paired: tl.constexpr,
    second_has_bias: tl.constexpr,
    aligned_m: tl.constexpr,
    aligned_nk: tl.constexpr,
    group_m: tl.constexpr,
    explicit_bias_fma: tl.constexpr,
    per_tile_tail: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
):
    if group_m:
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(n, block_n) * (2 if paired else 1)
        row_block_count = tl.num_programs(0) // num_pid_n
        num_pid_in_group = group_m * num_pid_n
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * group_m
        actual_group_m = tl.minimum(row_block_count - first_pid_m, group_m)
        pid_m = first_pid_m + pid_in_group % actual_group_m
        pid_n = pid_in_group // actual_group_m
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    second = pid_n >= tl.cdiv(n, block_n)
    if paired:
        pid_n %= tl.cdiv(n, block_n)
        weight_ptr = tl.where(second, second_weight_ptr, weight_ptr)
        weight_scale_ptr = tl.where(second, second_scale_ptr, weight_scale_ptr)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    tile_args = (
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        second_bias_ptr,
        offsets_m,
        offsets_n,
        second,
        m,
        n,
        k,
    )
    # Large tiles share one launch. With aligned N/K, complete M tiles take an
    # unmasked branch around the entire projection. Other widths use the masked
    # loop in the same launch; branching for those widths can regress performance.
    if per_tile_tail and aligned_nk:
        tile_aligned = aligned_m or (pid_m + 1) * block_m <= m
    else:
        tile_aligned = aligned_m and aligned_nk
    if tile_aligned:
        result = _int8_matmul_with_bias(
            *tile_args,
            block_m,
            block_n,
            block_k,
            has_bias,
            paired,
            second_has_bias,
            True,
            explicit_bias_fma,
            accumulator_seed,
        )
    else:
        result = _int8_matmul_with_bias(
            *tile_args,
            block_m,
            block_n,
            block_k,
            has_bias,
            paired,
            second_has_bias,
            False,
            explicit_bias_fma,
            accumulator_seed,
        )
    return result, offsets_m, offsets_n, second, tile_aligned


@triton.jit
def _store_tile(
    output_ptr,
    result,
    offsets_m,
    offsets_n,
    m,
    n,
    output_row_stride,
    second,
    paired: tl.constexpr,
    tile_aligned,
):
    output_pointers = (
        output_ptr
        + offsets_m.to(tl.int64)[:, None] * output_row_stride
        + offsets_n.to(tl.int64)[None, :]
    )
    if paired:
        output_pointers += second * n
    if tile_aligned:
        tl.store(output_pointers, result)
    else:
        tl.store(
            output_pointers,
            result,
            mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
        )


@triton.jit
def int8_matmul_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    second_weight_ptr,
    second_scale_ptr,
    second_bias_ptr,
    m,
    n,
    k,
    output_row_stride,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
    paired: tl.constexpr,
    second_has_bias: tl.constexpr,
    aligned_m: tl.constexpr,
    aligned_nk: tl.constexpr,
    group_m: tl.constexpr,
    explicit_bias_fma: tl.constexpr,
    per_tile_tail: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
):
    """Project to a logical-dtype tensor without residual launch arguments."""
    result, offsets_m, offsets_n, second, tile_aligned = _project_tile(
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        second_weight_ptr,
        second_scale_ptr,
        second_bias_ptr,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        has_bias,
        paired,
        second_has_bias,
        aligned_m,
        aligned_nk,
        group_m,
        explicit_bias_fma,
        per_tile_tail,
        accumulator_seed,
    )
    _store_tile(
        output_ptr,
        result,
        offsets_m,
        offsets_n,
        m,
        n,
        output_row_stride,
        second,
        paired,
        tile_aligned,
    )


@triton.jit
def int8_residual_matmul_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
    aligned_m: tl.constexpr,
    aligned_nk: tl.constexpr,
    group_m: tl.constexpr,
    explicit_bias_fma: tl.constexpr,
    per_tile_tail: tl.constexpr,
    residual_base_ptr,
    residual_gate_ptr,
    residual_indices_ptr,
    residual_reusable_gate_ptr,
    residual_row_offset,
    residual_gate_row_stride: tl.constexpr,
    residual_gate_rows,
    residual_reusable_gate_row_stride: tl.constexpr,
    residual_reusable_gate_rows,
    residual_has_reusable_update: tl.constexpr,
    residual_python_indexing: tl.constexpr,
):
    """Project and add a validated residual without a materialized projection.

    Rounds the projection to its logical 16-bit dtype before FP32 residual
    arithmetic, preserving the materialized projection's rounding boundary while
    keeping the tile compact before residual loads. On RTX 5090 (SM120),
    retaining FP32 in the 128x256 Triton tile spilled registers and regressed
    attention blocks by up to ~1.6% and FFNs by 1-5% (splitting the Triton
    epilogue removed spills but stayed slower); keep this rounding unless a
    different schedule demonstrates a measured win across workloads.
    """
    result, offsets_m, offsets_n, _, tile_aligned = _project_tile(
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        weight_ptr,
        weight_scale_ptr,
        bias_ptr,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        has_bias,
        False,
        False,
        aligned_m,
        aligned_nk,
        group_m,
        explicit_bias_fma,
        per_tile_tail,
    )
    # Preserve materialized-projection rounding. On SM120 this also lets the
    # compiler pack the tile before residual loads; retaining FP32 spills.
    result = result.to(output_ptr.dtype.element_ty).to(tl.float32)
    result = add_residual(
        result,
        residual_base_ptr,
        output_ptr,
        residual_gate_ptr,
        residual_reusable_gate_ptr,
        residual_indices_ptr,
        offsets_m[:, None],
        offsets_n[None, :],
        (offsets_m[:, None] < m) & (offsets_n[None, :] < n),
        offsets_m[:, None] < m,
        n,
        residual_row_offset,
        residual_gate_row_stride,
        residual_gate_rows,
        residual_reusable_gate_row_stride,
        residual_reusable_gate_rows,
        residual_has_reusable_update,
        residual_python_indexing,
    )
    _store_tile(
        output_ptr,
        result,
        offsets_m,
        offsets_n,
        m,
        n,
        n,
        0,
        False,
        tile_aligned,
    )


def _is_int8_splitk_eligible(
    *,
    rows: int,
    out_features: int,
    in_features: int,
    block_k: int,
    paired: bool = False,
    group_m: int = 0,
) -> bool:
    """Return whether a small-batch INT8 projection qualifies for Split-K."""
    k_tiles = (in_features + block_k - 1) // block_k
    weight_bytes = out_features * in_features * (2 if paired else 1)
    return (
        0 < rows <= _INT8_SPLITK_MAX_ROWS
        and group_m == 0
        and out_features > 0
        and 0 < in_features <= _INT8_SPLITK_MAX_SAFE_REDUCTION
        and weight_bytes >= _INT8_MIN_SPLITK_WEIGHT_BYTES
        and k_tiles >= _INT8_SPLITK_MIN_K_TILES
    )


def _compute_int8_split_k(
    num_sms: int,
    *,
    rows: int,
    out_features: int,
    in_features: int,
    block_m: int,
    block_n: int,
    block_k: int,
    paired: bool = False,
) -> int:
    m_tiles = (rows + block_m - 1) // block_m
    n_tiles = ((out_features + block_n - 1) // block_n) * (2 if paired else 1)
    k_tiles = (in_features + block_k - 1) // block_k
    weight_bytes = out_features * in_features * (2 if paired else 1)
    max_for_partial = max(1, in_features // (_INT8_SPLITK_PARTIAL_RATIO_DIVISOR * rows))
    return select_split_k(
        num_sms,
        output_tiles=m_tiles * n_tiles,
        k_tiles=k_tiles,
        weight_bytes=weight_bytes,
        min_weight_bytes=_INT8_MIN_SPLITK_WEIGHT_BYTES,
        max_split_k=min(_INT8_MAX_SPLIT_K, max_for_partial),
        cta_multiplier=1,
        min_k_tiles=_INT8_SPLITK_MIN_K_TILES,
        min_tiles_per_split=_INT8_SPLITK_MIN_TILES_PER_SPLIT,
        round_up=False,
    )


def select_int8_split_k(
    num_sms: int,
    *,
    rows: int,
    out_features: int,
    in_features: int,
    block_m: int,
    block_n: int,
    block_k: int,
    paired: bool = False,
    group_m: int = 0,
) -> int:
    """Select the Split-K factor for an INT8 matmul workload given an SM count."""
    if not _is_int8_splitk_eligible(
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        block_k=block_k,
        paired=paired,
        group_m=group_m,
    ):
        return 1
    return _compute_int8_split_k(
        num_sms,
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        paired=paired,
    )


@triton.jit(do_not_specialize=["m", "split_k"])
def int8_matmul_splitk_kernel(
    input_ptr,
    weight_ptr,
    second_weight_ptr,
    partial_ptr,
    m,
    n,
    k,
    split_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    paired: tl.constexpr,
    aligned_nk: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
):
    """Accumulate K-sliced INT32 partial sums across one or two paired weights."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    n_tiles = tl.cdiv(n, block_n)
    second = pid_n >= n_tiles
    if paired:
        pid_n %= n_tiles
        weight_ptr = tl.where(second, second_weight_ptr, weight_ptr)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    total_k_tiles = tl.cdiv(k, block_k)
    base_tiles = total_k_tiles // split_k
    remainder_tiles = total_k_tiles % split_k
    k_start = pid_k * base_tiles + tl.minimum(pid_k, remainder_tiles)
    k_end = k_start + base_tiles + tl.where(pid_k < remainder_tiles, 1, 0)

    acc = _accumulate_int8_tile(
        input_ptr,
        weight_ptr,
        offs_m,
        offs_n,
        m,
        n,
        k,
        k_start,
        k_end,
        block_m,
        block_n,
        block_k,
        False,
        aligned_nk,
        accumulator_seed,
    )

    m_i64 = offs_m.to(tl.int64)
    n_i64 = offs_n.to(tl.int64)
    out_cols = tl.full((), n, tl.int64) * (2 if paired else 1)
    col_offs = n_i64 + (second.to(tl.int64) * n if paired else 0)
    out_base = (pid_k.to(tl.int64) * m + m_i64[:, None]) * out_cols + col_offs[None, :]
    tl.store(
        partial_ptr + out_base,
        acc - accumulator_seed,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


@triton.jit(do_not_specialize=["m", "split_k"])
def int8_splitk_reduce_kernel(
    partial_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    second_scale_ptr,
    second_bias_ptr,
    output_ptr,
    m,
    n,
    split_k,
    output_row_stride,
    block_n: tl.constexpr,
    has_bias: tl.constexpr,
    paired: tl.constexpr,
    second_has_bias: tl.constexpr,
    explicit_bias_fma: tl.constexpr,
):
    """Deterministically reduce INT32 Split-K partials and apply scale/bias epilogue."""
    pid_n = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    n_tiles = tl.cdiv(n, block_n)
    second = pid_n >= n_tiles
    if paired:
        pid_n %= n_tiles
        weight_scale_ptr = tl.where(second, second_scale_ptr, weight_scale_ptr)

    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_n_i64 = offs_n.to(tl.int64)
    n_valid = offs_n < n
    out_cols = tl.full((), n, tl.int64) * (2 if paired else 1)
    col_offs_i64 = offs_n_i64 + (second.to(tl.int64) * n if paired else 0)

    acc = tl.zeros((block_n,), tl.int32)
    split_stride = tl.full((), m, tl.int64) * out_cols
    partial_tile_ptr = partial_ptr + row * out_cols + col_offs_i64
    for _ in range(split_k):
        acc += tl.load(partial_tile_ptr, mask=n_valid, other=0)
        partial_tile_ptr += split_stride

    input_scale = tl.load(input_scale_ptr + row)
    weight_scale = tl.load(weight_scale_ptr + offs_n_i64, mask=n_valid, other=0.0)
    row_scaled = acc.to(tl.float32) * input_scale
    result = _scale_and_add_bias(
        row_scaled,
        weight_scale,
        bias_ptr,
        second_bias_ptr,
        offs_n,
        second,
        n,
        block_n,
        has_bias,
        paired,
        second_has_bias,
        False,
        explicit_bias_fma,
        False,
    )

    tl.store(output_ptr + row * output_row_stride + col_offs_i64, result, mask=n_valid)


def launch_int8_matmul_splitk(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    output: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    second_weight: torch.Tensor,
    second_scale: torch.Tensor,
    second_bias: torch.Tensor | None,
    *,
    paired: bool,
    split_k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    explicit_bias_fma: bool,
    accumulator_seed: int = 0,
    num_warps: int,
    num_stages: int,
    compiler_options: Mapping[str, Any] | None = None,
) -> None:
    """Execute a 2-stage Split-K INT8 projection and deterministic INT32 reduction."""
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    out_cols = n * (2 if paired else 1)
    partial = torch.empty((split_k, m, out_cols), device=input_qdata.device, dtype=torch.int32)
    m_tiles = triton.cdiv(m, block_m)
    n_tiles = triton.cdiv(n, block_n) * (2 if paired else 1)
    reduce_tiles = triton.cdiv(n, INT8_SPLITK_REDUCE_BLOCK_N) * (2 if paired else 1)
    extra_options = {} if compiler_options is None else dict(compiler_options)
    with device_context(input_qdata.device):
        int8_matmul_splitk_kernel[(m_tiles, n_tiles, split_k)](
            input_qdata,
            weight_qdata,
            second_weight,
            partial,
            m,
            n,
            k,
            split_k,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            paired=paired,
            aligned_nk=n % block_n == 0 and k % block_k == 0,
            accumulator_seed=accumulator_seed,
            **extra_options,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        int8_splitk_reduce_kernel[(reduce_tiles, m)](
            partial,
            input_scale,
            weight_scale,
            bias if bias is not None else output,
            second_scale,
            second_bias if second_bias is not None else output,
            output,
            m,
            n,
            split_k,
            output.stride(0),
            block_n=INT8_SPLITK_REDUCE_BLOCK_N,
            has_bias=bias is not None,
            paired=paired,
            second_has_bias=second_bias is not None,
            explicit_bias_fma=explicit_bias_fma,
            num_warps=4,
        )


def try_launch_int8_matmul_splitk(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    output: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    second_weight: torch.Tensor,
    second_scale: torch.Tensor,
    second_bias: torch.Tensor | None,
    *,
    paired: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    explicit_bias_fma: bool,
    num_warps: int,
    num_stages: int,
    target: AcceleratorTarget | None = None,
    compiler_options: Mapping[str, Any] | None = None,
) -> bool:
    """Resolve Split-K once for NVIDIA/AMD launchers and run it if beneficial."""
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    if not _is_int8_splitk_eligible(
        rows=m,
        out_features=n,
        in_features=k,
        block_k=block_k,
        paired=paired,
        group_m=group_m,
    ):
        return False
    split_k = _compute_int8_split_k(
        device_sm_count(input_qdata.device),
        rows=m,
        out_features=n,
        in_features=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        paired=paired,
    )
    if split_k <= 1:
        return False
    accumulator_seed = splitk_accumulator_seed(
        target,
        rows=m,
        out_features=n,
        in_features=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        paired=paired,
        split_k=split_k,
    )
    launch_int8_matmul_splitk(
        input_qdata,
        weight_qdata,
        output,
        input_scale,
        weight_scale,
        bias,
        second_weight,
        second_scale,
        second_bias,
        paired=paired,
        split_k=split_k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        explicit_bias_fma=explicit_bias_fma,
        accumulator_seed=accumulator_seed,
        num_warps=num_warps,
        num_stages=num_stages,
        compiler_options=compiler_options,
    )
    return True


def splitk_accumulator_seed(
    target: AcceleratorTarget | None,
    *,
    rows: int,
    out_features: int,
    in_features: int,
    block_m: int,
    block_n: int,
    block_k: int,
    paired: bool,
    split_k: int,
) -> int:
    """Return the RDNA4 accumulator zero-crossing seed for the largest K slice."""
    if target is None or not target.is_amd_hip:
        return 0
    output_tiles = (
        triton.cdiv(rows, block_m) * triton.cdiv(out_features, block_n) * (2 if paired else 1)
    )
    max_tiles_per_split = triton.cdiv(triton.cdiv(in_features, block_k), split_k)
    split_reduction = min(in_features, max_tiles_per_split * block_k)
    return rdna4_int8_accumulator_seed(target, split_reduction, output_tiles * split_k)
