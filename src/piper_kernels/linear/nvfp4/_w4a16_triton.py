"""Shared Triton kernels and GPU runtime launcher for NVFP4 W4A16 linear execution."""

# Triton's JIT launcher and constexpr options are not fully represented in its typing surface.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton.primitives import splitk_tile_bounds
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._splitk import device_sm_count
from piper_kernels.weights.nvfp4 import _layout
from piper_kernels.weights.nvfp4.triton import _decode_fp4_code, swizzled_scale_offsets

from ._amd import policy as amd_policy
from ._nvidia import policy as nvidia_policy
from ._nvidia.isa import decode_scale_sm120_e2m1x2, decode_sm120_e2m1x2_f32
from ._w4a16_schedule import (
    W4A16_DEQUANT_BLOCK_K,
    W4A16_DEQUANT_BLOCK_N,
    W4A16_FP32_WORKSPACE_BYTES,
    W4A16_MAX_REDUCE_ROWS,
    W4A16_REDUCE_BLOCK_N,
    W4A16Schedule,
    default_w4a16_schedule,
)

_NVFP4_BLOCK_SIZE_TL = tl.constexpr(_layout.BLOCK_SIZE)
_NVFP4_QDATA_BLOCK_SIZE_TL = tl.constexpr(_layout.QDATA_BLOCK_SIZE)


@triton.jit
def _decode_e4m3_u8(code):
    """Exact FP32 value of FP8 E4M3FN byte bits without backend FP8 lowering quirks."""
    sign = code >> 7
    exponent = (code >> 3) & 15
    mantissa = code & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    subnormal = mantissa.to(tl.float32) * 0.001953125
    magnitude = tl.where(exponent == 0, subnormal, normal)
    return tl.where(sign != 0, -magnitude, magnitude)


@triton.jit
def _load_nvfp4_raw_tile(
    weight_ptr,
    weight_scale_ptr,
    n_offs_i64,
    n_valid,
    k_base,
    packed_k,
    scale_k,
    scale_column_blocks,
    block_k: tl.constexpr,
):
    """Load one swizzled NVFP4 weight tile as packed uint8 bytes and raw E4M3FN bytes."""
    block_packed_k: tl.constexpr = block_k // 2
    block_scale_k: tl.constexpr = block_k // _NVFP4_BLOCK_SIZE_TL
    k_byte_offs = (k_base // 2) + tl.arange(0, block_packed_k)
    packed = tl.load(
        weight_ptr + n_offs_i64[:, None] * packed_k + k_byte_offs[None, :],
        mask=n_valid[:, None] & (k_byte_offs[None, :] < packed_k),
        other=0,
    )
    scale_cols = (k_base // _NVFP4_BLOCK_SIZE_TL) + tl.arange(0, block_scale_k)
    scale_offsets = swizzled_scale_offsets(
        n_offs_i64[:, None],
        scale_cols[None, :],
        scale_column_blocks,
    )
    scale_u8 = tl.load(
        weight_scale_ptr + scale_offsets,
        mask=n_valid[:, None] & (scale_cols[None, :] < scale_k),
        other=0,
    )
    return packed, scale_u8


@triton.jit
def _load_nvfp4_tile(
    weight_ptr,
    weight_scale_ptr,
    n_offs_i64,
    n_valid,
    k_base,
    packed_k,
    scale_k,
    scale_column_blocks,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr = False,
):
    """Load one swizzled NVFP4 weight tile as decoded E2M1 pairs and FP32 block scales."""
    packed, scale_u8 = _load_nvfp4_raw_tile(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_k,
    )
    if use_sm120_e2m1x2:
        w_even, w_odd = decode_sm120_e2m1x2_f32(packed, high_first)
        scales = scale_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        return w_even, w_odd, scales
    w_lo = _decode_fp4_code(packed & 0xF)
    w_hi = _decode_fp4_code(packed >> 4)
    w_even = w_hi if high_first else w_lo
    w_odd = w_lo if high_first else w_hi
    scales = _decode_e4m3_u8(scale_u8.to(tl.uint32))
    return w_even, w_odd, scales


@triton.jit
def _expand_nvfp4_scales(
    scales,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    block_packed_k: tl.constexpr = block_k // 2
    block_scale_k: tl.constexpr = block_k // _NVFP4_BLOCK_SIZE_TL
    return tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (block_n, block_scale_k, 1)),
            (block_n, block_scale_k, _NVFP4_QDATA_BLOCK_SIZE_TL),
        ),
        (block_n, block_packed_k),
    )


@triton.jit
def _split_global_scale(global_scale):
    apply_early = tl.abs(global_scale) < 1.0
    pre_scale = tl.where(apply_early, global_scale, 1.0)
    post_scale = tl.where(apply_early, 1.0, global_scale)
    return pre_scale, post_scale


@triton.jit
def _accumulate_scaled_row_blocks(
    x_vec,
    w_even,
    w_odd,
    scales,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Compute scaled per-block dot partials ``[block_n, block_k // 16]`` for one row."""
    block_packed_k: tl.constexpr = block_k // 2
    block_scale_k: tl.constexpr = block_k // _NVFP4_BLOCK_SIZE_TL
    x_even, x_odd = tl.split(tl.reshape(x_vec.to(tl.float32), (block_packed_k, 2)))
    prod = w_even * x_even[None, :] + w_odd * x_odd[None, :]
    grouped = tl.reshape(prod, (block_n, block_scale_k, _NVFP4_QDATA_BLOCK_SIZE_TL))
    block_sum = tl.sum(grouped, axis=2)
    return tl.where(scales == 0.0, 0.0, block_sum * scales)


@triton.jit
def _accumulate_scaled_row(
    x_vec,
    w_even,
    w_odd,
    scales,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    return tl.sum(
        _accumulate_scaled_row_blocks(x_vec, w_even, w_odd, scales, block_n, block_k),
        axis=1,
    )


@triton.jit
def _store_w4a16_result(
    acc,
    epilogue_scale,
    bias_ptr,
    output_ptr,
    pid_k,
    m_offs,
    m_offs_i64,
    n_offs_i64,
    n_valid,
    rows,
    out_features,
    block_m: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    is_splitk: tl.constexpr,
):
    if block_m == 1:
        if is_splitk:
            tl.store(
                output_ptr + pid_k.to(tl.int64) * out_features + n_offs_i64,
                acc,
                mask=n_valid,
            )
            return
        if has_weight_per_tensor_scale:
            acc *= epilogue_scale
        if has_bias:
            acc += tl.load(bias_ptr + n_offs_i64, mask=n_valid, other=0.0).to(tl.float32)
        tl.store(output_ptr + n_offs_i64, acc, mask=n_valid)
        return

    row_valid = (m_offs[:, None] < rows) & n_valid[None, :]
    if is_splitk:
        partial_base = (pid_k.to(tl.int64) * rows + m_offs_i64[:, None]) * out_features
        tl.store(output_ptr + partial_base + n_offs_i64[None, :], acc, mask=row_valid)
        return
    if has_weight_per_tensor_scale:
        acc *= epilogue_scale
    if has_bias:
        bias_vec = tl.load(bias_ptr + n_offs_i64, mask=n_valid, other=0.0).to(tl.float32)
        acc += bias_vec[None, :]
    tl.store(
        output_ptr + m_offs_i64[:, None] * out_features + n_offs_i64[None, :],
        acc,
        mask=row_valid,
    )


@triton.jit
def _w4a16_program_coords(
    rows,
    out_features,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    group_m: tl.constexpr,
):
    """Map program_id(0) to 2D output tile coordinates with optional L2 group swizzling."""
    pid_mn = tl.program_id(0)
    if group_m > 0:
        num_pid_m = tl.cdiv(rows, block_m)
        num_pid_n = tl.cdiv(out_features, block_n)
        num_pid_in_group = group_m * num_pid_n
        group_id = pid_mn // num_pid_in_group
        first_pid_m = group_id * group_m
        group_size_m = tl.minimum(num_pid_m - first_pid_m, group_m)
        pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
        pid_n = (pid_mn % num_pid_in_group) // group_size_m
        return pid_m, pid_n
    return 0, pid_mn


@triton.jit
def _load_scaled_nvfp4_pairs(
    weight_ptr,
    weight_scale_ptr,
    n_offs_i64,
    n_valid,
    k_base,
    packed_k,
    scale_k,
    scale_column_blocks,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    out_dtype: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr,
    half_scales: tl.constexpr = False,
):
    """Load, decode, and scale one NVFP4 K-tile into even/odd pairs."""
    if use_sm120_e2m1x2:
        # Every finite (E2M1, E4M3FN) product fits exactly in FP16 (max 2688, min 2^-10,
        # 6 significand bits), so f16x2 decode+scale is bit-exact for both FP16 and FP32.
        is_fp16_scale: tl.constexpr = tl.constexpr(out_dtype != tl.bfloat16)
        scale_dtype = tl.float16 if is_fp16_scale else tl.bfloat16
        packed, scale_u8 = _load_nvfp4_raw_tile(
            weight_ptr,
            weight_scale_ptr,
            n_offs_i64,
            n_valid,
            k_base,
            packed_k,
            scale_k,
            scale_column_blocks,
            block_k,
        )
        scales_16 = _expand_nvfp4_scales(
            scale_u8.to(tl.float8e4nv, bitcast=True).to(scale_dtype),
            block_n,
            block_k,
        )
        w_e, w_o = decode_scale_sm120_e2m1x2(packed, scales_16, high_first, is_fp16_scale)
        return w_e.to(out_dtype), w_o.to(out_dtype)

    w_even, w_odd, scales = _load_nvfp4_tile(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_k,
        high_first,
    )
    if half_scales:
        scales_expanded = _expand_nvfp4_scales(scales.to(out_dtype), block_n, block_k)
        return w_even.to(out_dtype) * scales_expanded, w_odd.to(out_dtype) * scales_expanded
    scales_expanded = _expand_nvfp4_scales(scales, block_n, block_k)
    return (w_even * scales_expanded).to(out_dtype), (w_odd * scales_expanded).to(out_dtype)


@triton.jit
def _w4a16_accumulate_dot_tile(
    acc,
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    m_offs,
    m_offs_i64,
    n_offs_i64,
    n_valid,
    k_base,
    rows,
    input_features,
    packed_k,
    scale_k,
    scale_column_blocks,
    pre_scale,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    pre_scale_in_gemv: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr,
    single_dot: tl.constexpr,
):
    """Accumulate one NVFP4 K-tile via tensor-core ``tl.dot``."""
    block_packed_k: tl.constexpr = block_k // 2
    # Decode weights before loading X so RDNA3/3.5 does not hold live X tiles across nibble unpacks.
    w_e, w_o = _load_scaled_nvfp4_pairs(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_n,
        block_k,
        high_first,
        tl.constexpr(input_ptr.dtype.element_ty),
        use_sm120_e2m1x2,
    )
    k_elem_offs = k_base + tl.arange(0, block_k)
    x_tile = tl.load(
        input_ptr + m_offs_i64[:, None] * input_features + k_elem_offs[None, :],
        mask=(m_offs[:, None] < rows) & (k_elem_offs[None, :] < input_features),
        other=0.0,
    )
    if pre_scale_in_gemv:
        x_tile = (x_tile.to(tl.float32) * pre_scale).to(x_tile.dtype)

    if single_dot:
        w_full = tl.reshape(tl.join(w_e, w_o), (block_n, block_k))
        return tl.dot(x_tile, tl.trans(w_full), acc=acc)
    x_even, x_odd = tl.split(tl.reshape(x_tile, (block_m, block_packed_k, 2)))
    acc = tl.dot(x_even, tl.trans(w_e), acc=acc)
    return tl.dot(x_odd, tl.trans(w_o), acc=acc)


@triton.jit
def _w4a16_accumulate_scalar_tile(
    acc,
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    m_offs,
    n_offs_i64,
    n_valid,
    k_base,
    rows,
    input_features,
    packed_k,
    scale_k,
    scale_column_blocks,
    pre_scale,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    pre_scale_in_gemv: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr,
):
    """Accumulate one NVFP4 K-tile via FP32 scalar decode and reduction."""
    w_even, w_odd, scales = _load_nvfp4_tile(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_k,
        high_first,
        use_sm120_e2m1x2,
    )

    k_elem_offs = k_base + tl.arange(0, block_k)
    k_elem_valid = k_elem_offs < input_features
    if block_m == 1:
        x_vec = tl.load(input_ptr + k_elem_offs, mask=k_elem_valid, other=0.0).to(tl.float32)
        if pre_scale_in_gemv:
            x_vec = x_vec * pre_scale
        acc += _accumulate_scaled_row_blocks(x_vec, w_even, w_odd, scales, block_n, block_k)
    else:
        # Triton's typing does not declare static_range iterable.
        for row_idx in tl.static_range(block_m):  # pyright: ignore[reportGeneralTypeIssues]
            if row_idx < rows:
                x_vec = tl.load(
                    input_ptr + tl.full((), row_idx, tl.int64) * input_features + k_elem_offs,
                    mask=k_elem_valid,
                    other=0.0,
                ).to(tl.float32)
                if pre_scale_in_gemv:
                    x_vec = x_vec * pre_scale
                row_acc = _accumulate_scaled_row(x_vec, w_even, w_odd, scales, block_n, block_k)
                acc += tl.where((m_offs == row_idx)[:, None], row_acc[None, :], 0.0)
    return acc


@triton.jit(do_not_specialize=["rows", "split_k"])
def w4a16_gemv_splitk_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_per_tensor_scale_ptr,
    bias_ptr,
    output_ptr,
    rows,
    out_features,
    packed_k,
    scale_column_blocks,
    split_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    is_splitk: tl.constexpr,
    use_dot: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr = False,
    single_dot: tl.constexpr = False,
    group_m: tl.constexpr = 0,
):
    """Stream swizzled NVFP4 weights with Split-K/2D tiling and accumulate in FP32."""
    tl.static_assert(use_dot or group_m == 0)
    tl.static_assert(not (block_m == 1 and use_dot))
    pid_m, pid_n = _w4a16_program_coords(rows, out_features, block_m, block_n, group_m)
    pid_k = tl.program_id(1)
    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features
    m_offs = pid_m * block_m + tl.arange(0, block_m)
    m_offs_i64 = m_offs.to(tl.int64)

    # Pass packed_k so Triton keeps 16-byte divisibility for vectorized weight loads.
    input_features = packed_k * 2
    scale_k = packed_k // _NVFP4_QDATA_BLOCK_SIZE_TL
    # Keep FP16 tl.dot unscaled (cannot overflow FP32, would underflow FP16); BF16 pre-scales
    # against FP32 overflow, trading away deep subnormal |x * pre_scale| < 2^-133 products.
    pre_scale_in_gemv: tl.constexpr = tl.constexpr(
        has_weight_per_tensor_scale and not (use_dot and input_ptr.dtype.element_ty == tl.float16)
    )
    global_scale = (
        tl.load(weight_per_tensor_scale_ptr).to(tl.float32) if has_weight_per_tensor_scale else 1.0
    )
    pre_scale, post_scale = _split_global_scale(global_scale)
    epilogue_scale = post_scale if pre_scale_in_gemv else global_scale
    k_tile_start, k_tile_end = splitk_tile_bounds(tl.cdiv(input_features, block_k), pid_k, split_k)

    block_scale_k: tl.constexpr = block_k // _NVFP4_BLOCK_SIZE_TL
    acc = (
        tl.zeros((block_n, block_scale_k), dtype=tl.float32)
        if block_m == 1
        else tl.zeros((block_m, block_n), dtype=tl.float32)
    )
    for k_tile in range(k_tile_start, k_tile_end):
        k_base = k_tile * block_k
        if use_dot:
            acc = _w4a16_accumulate_dot_tile(
                acc,
                input_ptr,
                weight_ptr,
                weight_scale_ptr,
                m_offs,
                m_offs_i64,
                n_offs_i64,
                n_valid,
                k_base,
                rows,
                input_features,
                packed_k,
                scale_k,
                scale_column_blocks,
                pre_scale,
                block_m,
                block_n,
                block_k,
                high_first,
                pre_scale_in_gemv,
                use_sm120_e2m1x2,
                single_dot,
            )
        else:
            acc = _w4a16_accumulate_scalar_tile(
                acc,
                input_ptr,
                weight_ptr,
                weight_scale_ptr,
                m_offs,
                n_offs_i64,
                n_valid,
                k_base,
                rows,
                input_features,
                packed_k,
                scale_k,
                scale_column_blocks,
                pre_scale,
                block_m,
                block_n,
                block_k,
                high_first,
                pre_scale_in_gemv,
                use_sm120_e2m1x2,
            )

    if block_m == 1:
        acc = tl.sum(acc, axis=1)

    _store_w4a16_result(
        acc,
        epilogue_scale,
        bias_ptr,
        output_ptr,
        pid_k,
        m_offs,
        m_offs_i64,
        n_offs_i64,
        n_valid,
        rows,
        out_features,
        block_m,
        has_weight_per_tensor_scale,
        has_bias,
        is_splitk,
    )


@triton.jit(do_not_specialize=["rows", "split_k"])
def w4a16_splitk_reduce_kernel(
    partial_ptr,
    weight_per_tensor_scale_ptr,
    bias_ptr,
    output_ptr,
    rows,
    out_features,
    split_k,
    block_n: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    post_scale_only: tl.constexpr = False,
):
    """Deterministically reduce Split-K partials in fixed order and apply affine epilogue."""
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

    if has_weight_per_tensor_scale:
        global_scale = tl.load(weight_per_tensor_scale_ptr).to(tl.float32)
        if post_scale_only:
            _, post_scale = _split_global_scale(global_scale)
            acc *= post_scale
        else:
            acc *= global_scale
    if has_bias:
        acc += tl.load(bias_ptr + n_offs_i64, mask=n_valid, other=0.0).to(tl.float32)
    tl.store(output_ptr + row * out_features + n_offs_i64, acc, mask=n_valid)


@triton.jit
def dequantize_nvfp4_swizzled_kernel(
    weight_ptr,
    weight_scale_ptr,
    output_ptr,
    out_features,
    packed_k,
    scale_column_blocks,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr = False,
):
    """Dequantize packed swizzled NVFP4 weights into a contiguous floating matrix."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features

    # Pass packed_k so Triton keeps 16-byte divisibility for vectorized weight loads.
    input_features = packed_k * 2
    scale_k = packed_k // _NVFP4_QDATA_BLOCK_SIZE_TL
    k_base = pid_k * block_k

    w_e, w_o = _load_scaled_nvfp4_pairs(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_n,
        block_k,
        high_first,
        tl.constexpr(output_ptr.dtype.element_ty),
        use_sm120_e2m1x2,
    )
    values = tl.reshape(tl.join(w_e, w_o), (block_n, block_k))
    k_elem_offs = k_base + tl.arange(0, block_k)
    tl.store(
        output_ptr + n_offs_i64[:, None] * input_features + k_elem_offs[None, :],
        values,
        mask=n_valid[:, None] & (k_elem_offs[None, :] < input_features),
    )


def select_w4a16_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
    *,
    target: AcceleratorTarget | None = None,
) -> W4A16Schedule:
    """Resolve the architecture-specific W4A16 schedule shared by runtime and offline compile."""
    if target is not None and target.is_nvidia_cuda:
        schedule = nvidia_policy.select_w4a16_schedule(
            target, num_sms, rows, input_dtype, out_features, input_features
        )
    elif target is not None and target.is_amd_hip:
        schedule = amd_policy.select_w4a16_schedule(
            target, num_sms, rows, input_dtype, out_features, input_features
        )
    else:
        schedule = default_w4a16_schedule(num_sms, rows, input_dtype, out_features, input_features)
    if schedule.mode == "gemv" and schedule.group_m == 0 and rows > schedule.block_m:
        raise ValueError(
            f"W4A16 schedule requires group_m > 0 when rows ({rows}) > block_m ({schedule.block_m})"
        )
    return schedule


def _dequantize_nvfp4_swizzled(
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    high_first: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
    block_n: int = W4A16_DEQUANT_BLOCK_N,
    block_k: int = W4A16_DEQUANT_BLOCK_K,
    num_warps: int = 4,
    use_sm120_e2m1x2: bool = False,
) -> torch.Tensor:
    """Dequantize a 2-D swizzled NVFP4 weight matrix in Triton without global scale."""
    out_features, packed_k = weight_qdata.shape
    input_features = 2 * packed_k
    output = torch.empty(
        (out_features, input_features),
        device=weight_qdata.device,
        dtype=output_dtype,
    )
    if out_features == 0 or input_features == 0:
        return output
    scale_column_blocks = int(triton.cdiv(input_features, _layout.SCALE_COLUMN_TILE))
    grid = (
        triton.cdiv(out_features, block_n),
        triton.cdiv(input_features, block_k),
    )
    with device_context(weight_qdata.device):
        dequantize_nvfp4_swizzled_kernel[grid](
            weight_qdata,
            weight_scale.view(torch.uint8),
            output,
            out_features,
            packed_k,
            scale_column_blocks,
            block_n=block_n,
            block_k=block_k,
            high_first=high_first,
            use_sm120_e2m1x2=use_sm120_e2m1x2,
            num_warps=num_warps,
        )
    return output


def _linear_w4a16_large_m(
    active_2d: torch.Tensor,
    input_dtype: torch.dtype,
    input_shape: torch.Size,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    high_first: bool,
    schedule: W4A16Schedule,
) -> torch.Tensor:
    rows, out_features = active_2d.shape[0], weight_qdata.shape[0]
    dequant_fp32 = schedule.dequant_as_fp32
    dequant_weight = _dequantize_nvfp4_swizzled(
        weight_qdata,
        weight_scale,
        high_first=high_first,
        output_dtype=torch.float32 if dequant_fp32 else active_2d.dtype,
        block_n=schedule.block_n,
        block_k=schedule.block_k,
        num_warps=schedule.num_warps,
        use_sm120_e2m1x2=schedule.use_sm120_e2m1x2,
    )
    with torch.amp.autocast(active_2d.device.type, enabled=False):
        weight_t = dequant_weight.T
        if weight_per_tensor_scale is None and bias is None and active_2d.dtype is input_dtype:
            product = (
                torch.mm(active_2d.float(), weight_t).to(input_dtype)
                if dequant_fp32
                else torch.mm(active_2d, weight_t)
            )
            return product.reshape(*input_shape[:-1], out_features)
        output = torch.empty((rows, out_features), device=active_2d.device, dtype=input_dtype)
        post_scale_only = (
            weight_per_tensor_scale is not None and active_2d.dtype is not torch.float16
        )
        pre_scale = None
        if post_scale_only:
            assert weight_per_tensor_scale is not None
            gs_f32 = weight_per_tensor_scale.to(device=active_2d.device, dtype=torch.float32)
            pre_scale = torch.where(gs_f32.abs() < 1.0, gs_f32, torch.ones_like(gs_f32))
        chunk_rows = min(
            rows,
            max(1, W4A16_FP32_WORKSPACE_BYTES // (out_features * 4)),
            W4A16_MAX_REDUCE_ROWS,
        )
        with device_context(active_2d.device):
            for start in range(0, rows, chunk_rows):
                end = min(start + chunk_rows, rows)
                active_chunk = active_2d[start:end]
                if pre_scale is not None:
                    active_chunk = (active_chunk.float() * pre_scale).to(active_2d.dtype)
                if active_2d.dtype is torch.float32 or dequant_fp32:
                    partial = torch.mm(
                        active_chunk.float() if dequant_fp32 else active_chunk, weight_t
                    )
                else:
                    partial = torch.mm(active_chunk, weight_t, out_dtype=torch.float32)
                chunk_len = end - start
                reduce_grid = (triton.cdiv(out_features, W4A16_REDUCE_BLOCK_N), chunk_len)
                w4a16_splitk_reduce_kernel[reduce_grid](
                    partial,
                    weight_per_tensor_scale,
                    bias,
                    output[start:end],
                    chunk_len,
                    out_features,
                    1,
                    block_n=W4A16_REDUCE_BLOCK_N,
                    has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
                    has_bias=bias is not None,
                    post_scale_only=post_scale_only,
                    num_warps=4,
                )
    return output.reshape(*input_shape[:-1], out_features)


def linear_w4a16(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    high_first: bool = False,
    *,
    group_size: int = 0,
) -> torch.Tensor:
    """Apply a swizzled NVFP4 weight to a floating activation, optionally rotated."""
    out_features, packed_k = weight_qdata.shape
    input_features = 2 * packed_k
    if input.numel() == 0 or out_features == 0:
        return input.new_empty((*input.shape[:-1], out_features))

    input_2d = input.reshape(-1, input_features).contiguous()
    rows = int(input_2d.shape[0])
    if group_size > 0:
        from piper_kernels.weights.convrot.triton import rotate_input  # noqa: PLC0415

        rotated = torch.empty(
            (rows, input_features),
            device=input_2d.device,
            dtype=torch.float32 if input.dtype is torch.float16 else input.dtype,
        )
        rotate_input(input_2d, rotated, group_size, num_warps=4)
        active_2d = rotated
    else:
        active_2d = input_2d

    target_info = AcceleratorTarget.from_device(input.device)
    schedule = select_w4a16_schedule(
        device_sm_count(input.device),
        rows,
        active_2d.dtype,
        out_features,
        input_features,
        target=target_info,
    )
    if schedule.mode == "dequant":
        return _linear_w4a16_large_m(
            active_2d,
            input.dtype,
            input.shape,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            bias,
            high_first,
            schedule,
        )

    output = torch.empty((rows, out_features), device=input.device, dtype=input.dtype)
    scale_column_blocks = int(triton.cdiv(input_features, _layout.SCALE_COLUMN_TILE))
    is_splitk = schedule.split_k > 1
    target = (
        torch.empty(
            (schedule.split_k, rows, out_features), device=input.device, dtype=torch.float32
        )
        if is_splitk
        else output
    )
    grid = (
        int(triton.cdiv(rows, schedule.block_m)) * int(triton.cdiv(out_features, schedule.block_n)),
        schedule.split_k,
    )
    launch_kwargs: dict[str, int] = {"num_warps": schedule.num_warps}
    if schedule.num_stages is not None:
        launch_kwargs["num_stages"] = schedule.num_stages
    with device_context(input.device):
        w4a16_gemv_splitk_kernel[grid](
            active_2d,
            weight_qdata,
            weight_scale.view(torch.uint8),
            weight_per_tensor_scale,
            bias,
            target,
            rows,
            out_features,
            packed_k,
            scale_column_blocks,
            schedule.split_k,
            block_m=schedule.block_m,
            block_n=schedule.block_n,
            block_k=schedule.block_k,
            high_first=high_first,
            has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
            has_bias=bias is not None,
            is_splitk=is_splitk,
            use_dot=schedule.use_dot,
            use_sm120_e2m1x2=schedule.use_sm120_e2m1x2,
            single_dot=schedule.single_dot,
            group_m=schedule.group_m,
            **launch_kwargs,
        )
        if is_splitk:
            reduce_grid = (triton.cdiv(out_features, W4A16_REDUCE_BLOCK_N), rows)
            w4a16_splitk_reduce_kernel[reduce_grid](
                target,
                weight_per_tensor_scale,
                bias,
                output,
                rows,
                out_features,
                schedule.split_k,
                block_n=W4A16_REDUCE_BLOCK_N,
                has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
                has_bias=bias is not None,
                post_scale_only=(
                    weight_per_tensor_scale is not None
                    and not (schedule.use_dot and active_2d.dtype is torch.float16)
                ),
                num_warps=4,
            )
    return output.reshape(*input.shape[:-1], out_features)
