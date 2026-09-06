"""Portable INT8 arithmetic shared by accelerator-owned launchers."""

# Triton constexpr defaults are Python constants before JIT specialization.
# pyright: reportArgumentType=false

import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.stochastic_quantization import stochastic_round_to_int
from piper_kernels.gguf import triton as gguf_backend
from piper_kernels.linear.convrot import triton as convrot_backend


@triton.jit
def int8_scale_from_max(absolute_max, reciprocal_scale: tl.constexpr = False):
    """Construct an unclamped scale with the backend's scalar-division lowering.

    HIP PyTorch divides by a scalar using a rounded FP32 reciprocal. Request
    that explicitly on HIP; retain the established CUDA expression by default.
    """
    if reciprocal_scale:
        return absolute_max * (1.0 / 127.0)
    return absolute_max / 127.0


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
def normalize_for_int8(values, scale, logical_dtype_code: tl.constexpr):
    """Normalize values without dividing by an underflowed logical scale."""
    if logical_dtype_code == 1:
        logical_scale = scale.to(tl.float16)
        safe_logical_scale = tl.where(logical_scale == 0, 1.0, logical_scale).to(tl.float16)
        scaled = (values / safe_logical_scale).to(tl.float16)
        return tl.where(
            logical_scale == 0,
            values.to(tl.float32) / scale,
            scaled.to(tl.float32),
        )
    elif logical_dtype_code == 2:
        return (values / scale.to(tl.bfloat16)).to(tl.bfloat16)
    else:
        return values / scale


@triton.jit
def _quantize_int8(values, scale, logical_dtype_code: tl.constexpr):
    """Apply the shared deterministic rounding and saturation policy."""
    scaled = normalize_for_int8(values, scale, logical_dtype_code)
    return tl.clamp(libdevice.rint(scaled.to(tl.float32)), -128.0, 127.0).to(tl.int8)


@triton.jit
def _store_quantized_chunk(
    q_ptr,
    output_row_offset,
    row_width,
    chunk_start: tl.constexpr,
    chunk_offsets,
    values,
    scale,
    logical_dtype_code: tl.constexpr,
):
    offsets = chunk_start + chunk_offsets
    quantized = _quantize_int8(values, scale, logical_dtype_code)
    tl.store(
        q_ptr + output_row_offset + offsets,
        quantized,
        mask=offsets < row_width,
    )


@triton.jit
def _load_weight_chunk(
    x_ptr,
    input_row_offset,
    packed_row_offset,
    row_width,
    chunk_start: tl.constexpr,
    chunk_offsets,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    logical_dtype_code: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    gguf_quant_type: tl.constexpr,
):
    if gguf_quant_type >= 0:
        return gguf_backend.load_rotated_chunk(
            x_ptr,
            packed_row_offset,
            row_width,
            chunk_start,
            chunk_offsets,
            chunk_size,
            group_size,
            inverse_sqrt_group,
            logical_dtype_code,
            gguf_quant_type,
        )
    return convrot_backend.load_activated_rotated_chunk(
        x_ptr,
        input_row_offset,
        row_width,
        chunk_start,
        chunk_offsets,
        chunk_size,
        group_size,
        inverse_sqrt_group,
        logical_dtype_code,
        activation_fn,
        accelerator_backend,
    )


@triton.jit
def rotate_quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    chunk_size: tl.constexpr,
    chunk_count: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    logical_dtype_code: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    gguf_quant_type: tl.constexpr,
):
    """Rotate and quantize a row held as one, two, or three equal chunks.

    Keep every rotated chunk live until the shared row scale is known, avoiding
    both recomputation and a global-memory intermediate.
    """
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    chunk_offsets = tl.arange(0, chunk_size)
    input_row_width = row_width * (2 if activation_fn == "swiglu" else 1)
    input_row_offset = row_i64 * input_row_width
    packed_row_offset = 0
    if gguf_quant_type >= 0:
        packed_row_offset = row_i64 * gguf_backend.packed_row_size(
            row_width,
            gguf_quant_type,
        )
    output_row_offset = row_i64 * row_width

    values0 = _load_weight_chunk(
        x_ptr,
        input_row_offset,
        packed_row_offset,
        row_width,
        0,
        chunk_offsets,
        chunk_size,
        group_size,
        inverse_sqrt_group,
        logical_dtype_code,
        activation_fn,
        accelerator_backend,
        gguf_quant_type,
    )
    row_max = tl.max(tl.abs(values0).to(tl.float32), axis=0)
    if chunk_count >= 2:
        values1 = _load_weight_chunk(
            x_ptr,
            input_row_offset,
            packed_row_offset,
            row_width,
            chunk_size,
            chunk_offsets,
            chunk_size,
            group_size,
            inverse_sqrt_group,
            logical_dtype_code,
            activation_fn,
            accelerator_backend,
            gguf_quant_type,
        )
        row_max = tl.maximum(
            row_max,
            tl.max(tl.abs(values1).to(tl.float32), axis=0),
        )
    if chunk_count >= 3:
        values2 = _load_weight_chunk(
            x_ptr,
            input_row_offset,
            packed_row_offset,
            row_width,
            2 * chunk_size,
            chunk_offsets,
            chunk_size,
            group_size,
            inverse_sqrt_group,
            logical_dtype_code,
            activation_fn,
            accelerator_backend,
            gguf_quant_type,
        )
        row_max = tl.maximum(
            row_max,
            tl.max(tl.abs(values2).to(tl.float32), axis=0),
        )

    scale = tl.maximum(int8_scale_from_max(row_max, accelerator_backend == "hip"), 1e-30)
    _store_quantized_chunk(
        q_ptr,
        output_row_offset,
        row_width,
        0,
        chunk_offsets,
        values0,
        scale,
        logical_dtype_code,
    )
    if chunk_count >= 2:
        _store_quantized_chunk(
            q_ptr,
            output_row_offset,
            row_width,
            chunk_size,
            chunk_offsets,
            values1,
            scale,
            logical_dtype_code,
        )
    if chunk_count >= 3:
        _store_quantized_chunk(
            q_ptr,
            output_row_offset,
            row_width,
            2 * chunk_size,
            chunk_offsets,
            values2,
            scale,
            logical_dtype_code,
        )
    tl.store(scale_ptr + row_i64, scale)


@triton.jit
def convert_gguf_tiles_kernel(
    data_ptr,
    q_ptr,
    scale_ptr,
    maxima_ptr,
    row_width,
    tiles_per_row,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    logical_dtype_code: tl.constexpr,
    quant_type: tl.constexpr,
    write_maxima: tl.constexpr,
):
    """Decode/rotate twice, retaining only tile maxima between the two passes."""
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    offsets = tl.arange(0, block_size)
    start = tile * block_size
    row_bytes = gguf_backend.packed_row_size(row_width, quant_type)
    tile_bytes = gguf_backend.packed_row_size(block_size, quant_type)
    values = gguf_backend.load_rotated_chunk(
        data_ptr,
        row * row_bytes + tile.to(tl.int64) * tile_bytes,
        row_width - start,
        0,
        offsets,
        block_size,
        group_size,
        group_size**-0.5,
        logical_dtype_code,
        quant_type,
    )
    if write_maxima:
        maximum = tl.max(tl.abs(values).to(tl.float32), axis=0)
        tl.store(maxima_ptr + row * tiles_per_row + tile, maximum)
    else:
        scale = tl.load(scale_ptr + row)
        quantized = _quantize_int8(values, scale, logical_dtype_code)
        tl.store(q_ptr + row * row_width + start + offsets, quantized, start + offsets < row_width)


@triton.jit
def gguf_row_scales_kernel(
    maxima_ptr,
    scale_ptr,
    tiles_per_row,
    block_size: tl.constexpr,
    reciprocal_scale: tl.constexpr,
):
    """Reduce decoded/rotated tile maxima to the weight's single scale per row."""
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block_size)
    maxima = tl.load(maxima_ptr + row * tiles_per_row + offsets, offsets < tiles_per_row, 0.0)
    scale = tl.maximum(int8_scale_from_max(tl.max(maxima, axis=0), reciprocal_scale), 1e-30)
    tl.store(scale_ptr + row, scale)


@triton.jit
def quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    block_size: tl.constexpr,
    logical_dtype_code: tl.constexpr,
    reciprocal_scale: tl.constexpr = False,
):
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < row_width
    row_offset = row_i64 * row_width
    values = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    scale = tl.maximum(int8_scale_from_max(tl.max(tl.abs(values), axis=0), reciprocal_scale), 1e-30)
    quantized = _quantize_int8(values, scale, logical_dtype_code)
    tl.store(q_ptr + row_offset + offsets, quantized, mask=mask)
    tl.store(scale_ptr + row_i64, scale)


@triton.jit
def requantize_update_rows_kernel(
    q_ptr,
    scale_ptr,
    update_ptr,
    row_width,
    stride_q_row,
    stride_q_col,
    stride_scale_row,
    stride_update_row,
    stride_update_col,
    beta,
    alpha,
    rounding_seed,
    block_size: tl.constexpr,
    logical_dtype_code: tl.constexpr,
    has_base: tl.constexpr,
    has_update: tl.constexpr,
    stochastic: tl.constexpr,
    reciprocal_scale: tl.constexpr = False,
):
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    offsets_i64 = offsets.to(tl.int64)
    mask = offsets < row_width
    if has_base:
        quantized = tl.load(
            q_ptr + row_i64 * stride_q_row + offsets_i64 * stride_q_col,
            mask=mask,
            other=0,
        )
        old_scale = tl.load(scale_ptr + row_i64 * stride_scale_row)
        values = beta * quantized.to(tl.float32) * old_scale
    else:
        values = tl.zeros((block_size,), dtype=tl.float32)
    if has_update:
        update = tl.load(
            update_ptr + row_i64 * stride_update_row + offsets_i64 * stride_update_col,
            mask=mask,
            other=0.0,
        )
        values += alpha * update.to(tl.float32)
    if logical_dtype_code == 1:
        values = values.to(tl.float16)
    elif logical_dtype_code == 2:
        values = values.to(tl.bfloat16)
    scale = tl.maximum(
        int8_scale_from_max(tl.max(tl.abs(values).to(tl.float32), axis=0), reciprocal_scale),
        1e-30,
    )
    quantized = _quantize_int8(values, scale, logical_dtype_code)
    if stochastic:
        stochastic_scaled = values.to(tl.float32) / scale
        logical_offsets = row_i64 * row_width + offsets_i64
        quantized = stochastic_round_to_int(
            stochastic_scaled,
            quantized,
            rounding_seed,
            logical_offsets,
            -128,
            127,
        ).to(tl.int8)
    tl.store(
        q_ptr + row_i64 * stride_q_row + offsets_i64 * stride_q_col,
        quantized,
        mask=mask,
    )
    tl.store(scale_ptr + row_i64 * stride_scale_row, scale)


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
    if paired:
        second = pid_n >= tl.cdiv(n, block_n)
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
