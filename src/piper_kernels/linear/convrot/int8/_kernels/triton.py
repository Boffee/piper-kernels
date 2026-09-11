"""Portable INT8 arithmetic shared by accelerator-owned launchers."""

# Triton constexpr defaults are Python constants before JIT specialization.
# pyright: reportArgumentType=false

import triton
import triton.language as tl


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
    offsets_k = tl.arange(0, block_k)
    offsets_m_i64 = offsets_m.to(tl.int64)
    offsets_n_i64 = offsets_n.to(tl.int64)
    offsets_k_i64 = offsets_k.to(tl.int64)
    input_pointers = input_ptr + offsets_m_i64[:, None] * k + offsets_k_i64[None, :]
    weight_pointers = weight_ptr + offsets_n_i64[None, :] * k + offsets_k_i64[:, None]
    accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)

    for k_offset in range(tl.cdiv(k, block_k)):
        if aligned_tiles:
            input_values = tl.load(input_pointers)
            weight = tl.load(weight_pointers)
        else:
            remaining_k = k - k_offset * block_k
            input_values = tl.load(
                input_pointers,
                mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < remaining_k),
                other=0,
            )
            weight = tl.load(
                weight_pointers,
                mask=(offsets_n[None, :] < n) & (offsets_k[:, None] < remaining_k),
                other=0,
            )
        accumulator += tl.dot(input_values, weight)
        input_pointers += block_k
        weight_pointers += block_k

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
    return accumulator.to(tl.float32) * input_scale[:, None] * weight_scale[None, :]


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
    row_block_offset,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
    paired: tl.constexpr,
    second_has_bias: tl.constexpr,
    aligned_tiles: tl.constexpr,
    group_m: tl.constexpr,
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
        pid_m = first_pid_m + pid_in_group % actual_group_m + row_block_offset
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
    offsets_m_i64 = offsets_m.to(tl.int64)
    offsets_n_i64 = offsets_n.to(tl.int64)
    result = scaled_int8_matmul(
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
        result += tl.where(second, second_bias, bias)[None, :]
    elif has_bias:
        if aligned_tiles:
            bias = tl.load(bias_ptr + offsets_n)
        else:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < n, other=0.0)
        result += bias[None, :]

    output_pointers = (
        output_ptr + offsets_m_i64[:, None] * output_row_stride + offsets_n_i64[None, :]
    )
    if paired:
        output_pointers += second * n
    if aligned_tiles:
        tl.store(output_pointers, result)
    else:
        tl.store(
            output_pointers,
            result,
            mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
        )
