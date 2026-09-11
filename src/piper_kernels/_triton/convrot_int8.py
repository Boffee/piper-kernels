"""Shared INT8 rotation, quantization, and storage update kernels."""

# pyright: reportArgumentType=false
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton import convrot as convrot_backend
from piper_kernels._triton.stochastic_quantization import stochastic_round_to_int
from piper_kernels.gguf import triton as gguf_backend


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
def normalize_for_int8(values, scale):
    """Normalize in FP32 without rounding the scale or quotient."""
    return values.to(tl.float32) / scale


@triton.jit
def round_to_int8(scaled, accelerator_backend: tl.constexpr):
    """Saturate and round once to nearest-even INT8 using the target's conversion."""
    if accelerator_backend == "cuda":
        return libdevice.float2int_rn(tl.clamp(scaled, -128.0, 127.0)).to(tl.int8)  # pyright: ignore[reportAttributeAccessIssue]
    return tl.clamp(libdevice.rint(scaled), -128.0, 127.0).to(tl.int8)


@triton.jit
def _quantize_int8(values, scale, accelerator_backend: tl.constexpr):
    """Apply terminal INT8 rounding after FP32 normalization."""
    return round_to_int8(normalize_for_int8(values, scale), accelerator_backend)


@triton.jit
def _store_quantized_chunk(
    q_ptr,
    output_row_offset,
    row_width,
    chunk_start: tl.constexpr,
    chunk_offsets,
    values,
    scale,
    accelerator_backend: tl.constexpr,
):
    offsets = chunk_start + chunk_offsets
    quantized = _quantize_int8(values, scale, accelerator_backend)
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
        accelerator_backend,
    )
    if chunk_count >= 2:
        _store_quantized_chunk(
            q_ptr,
            output_row_offset,
            row_width,
            chunk_size,
            chunk_offsets,
            values1,  # pyright: ignore[reportPossiblyUnboundVariable]
            scale,
            accelerator_backend,
        )
    if chunk_count >= 3:
        _store_quantized_chunk(
            q_ptr,
            output_row_offset,
            row_width,
            2 * chunk_size,
            chunk_offsets,
            values2,  # pyright: ignore[reportPossiblyUnboundVariable]
            scale,
            accelerator_backend,
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
    quant_type: tl.constexpr,
    write_maxima: tl.constexpr,
    accelerator_backend: tl.constexpr,
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
        quant_type,
    )
    if write_maxima:
        maximum = tl.max(tl.abs(values).to(tl.float32), axis=0)
        tl.store(maxima_ptr + row * tiles_per_row + tile, maximum)
    else:
        scale = tl.load(scale_ptr + row)
        quantized = _quantize_int8(values, scale, accelerator_backend)
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
    accelerator_backend: tl.constexpr,
    reciprocal_scale: tl.constexpr = False,
):
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < row_width
    row_offset = row_i64 * row_width
    values = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    scale = tl.maximum(
        int8_scale_from_max(tl.max(tl.abs(values).to(tl.float32), axis=0), reciprocal_scale), 1e-30
    )
    quantized = _quantize_int8(values, scale, accelerator_backend)
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
    has_base: tl.constexpr,
    has_update: tl.constexpr,
    stochastic: tl.constexpr,
    accelerator_backend: tl.constexpr,
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
    scale = tl.maximum(
        int8_scale_from_max(tl.max(tl.abs(values).to(tl.float32), axis=0), reciprocal_scale),
        1e-30,
    )
    quantized = _quantize_int8(values, scale, accelerator_backend)
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
