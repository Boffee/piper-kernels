"""ConvRot NVFP4 packing kernels shared by activations and weights."""

# pyright: reportCallIssue=false
from __future__ import annotations

import triton
import triton.language as tl

from piper_kernels._triton import convrot as convrot_backend
from piper_kernels._triton import nvfp4 as nvfp4_backend
from piper_kernels.gguf import triton as gguf_backend
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

_NVFP4_BLOCK_SIZE = nvfp4_layout.BLOCK_SIZE
_NVFP4_BLOCK_SIZE_TL = tl.constexpr(_NVFP4_BLOCK_SIZE)
_MAX_ROTATION_CHUNK_SIZE = 16_384


@triton.jit
def _load_source_chunk(
    input_ptr,
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
            input_ptr,
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
        input_ptr,
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
def _rotated_chunk_amax(
    input_ptr,
    input_row_offset,
    packed_row_offset,
    row_width,
    chunk_start: tl.constexpr,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    gguf_quant_type: tl.constexpr,
):
    chunk_offsets = tl.arange(0, chunk_size)
    values = _load_source_chunk(
        input_ptr,
        input_row_offset,
        packed_row_offset,
        row_width,
        chunk_start,
        chunk_offsets,
        chunk_size,
        group_size,
        inverse_sqrt_group,
        activation_fn,
        accelerator_backend,
        gguf_quant_type,
    )
    return tl.max(tl.abs(values).to(tl.float32), axis=0)


@triton.jit
def _rotated_row_amax_kernel(
    input_ptr,
    row_amax_ptr,
    row_width,
    chunk_count: tl.constexpr,
    chunk_size0: tl.constexpr,
    chunk_size1: tl.constexpr,
    chunk_size2: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    gguf_quant_type: tl.constexpr,
):
    """Compute one exact post-rotation absolute maximum per activation row."""
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    input_row_width = row_width * (2 if activation_fn == "swiglu" else 1)
    input_row_offset = row_i64 * input_row_width
    packed_row_offset = 0
    if gguf_quant_type >= 0:
        packed_row_offset = row_i64 * gguf_backend.packed_row_size(
            row_width,
            gguf_quant_type,
        )

    row_amax = _rotated_chunk_amax(
        input_ptr,
        input_row_offset,
        packed_row_offset,
        row_width,
        tl.constexpr(0),
        chunk_size0,
        group_size,
        inverse_sqrt_group,
        activation_fn,
        accelerator_backend,
        gguf_quant_type,
    )
    if chunk_count >= 2:
        row_amax = tl.maximum(
            row_amax,
            _rotated_chunk_amax(
                input_ptr,
                input_row_offset,
                packed_row_offset,
                row_width,
                chunk_size0,
                chunk_size1,
                group_size,
                inverse_sqrt_group,
                activation_fn,
                accelerator_backend,
                gguf_quant_type,
            ),
        )
    if chunk_count >= 3:
        row_amax = tl.maximum(
            row_amax,
            _rotated_chunk_amax(
                input_ptr,
                input_row_offset,
                packed_row_offset,
                row_width,
                chunk_size0 + chunk_size1,
                chunk_size2,
                group_size,
                inverse_sqrt_group,
                activation_fn,
                accelerator_backend,
                gguf_quant_type,
            ),
        )
    tl.store(row_amax_ptr + row_i64, row_amax)


@triton.jit
def _rotate_quantize_chunk(
    input_ptr,
    per_tensor_scale_ptr,
    qdata_ptr,
    scale_ptr,
    input_row_offset,
    packed_row_offset,
    qdata_row_offset,
    row,
    row_width,
    chunk_start: tl.constexpr,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    gguf_quant_type: tl.constexpr,
    has_per_tensor_scale: tl.constexpr,
    swizzled_scales: tl.constexpr,
    high_first: tl.constexpr,
):
    """Rotate and encode one power-of-two row chunk into standard NVFP4 storage."""
    chunk_offsets = tl.arange(0, chunk_size)
    values = _load_source_chunk(
        input_ptr,
        input_row_offset,
        packed_row_offset,
        row_width,
        chunk_start,
        chunk_offsets,
        chunk_size,
        group_size,
        inverse_sqrt_group,
        activation_fn,
        accelerator_backend,
        gguf_quant_type,
    )
    block_count: tl.constexpr = chunk_size // _NVFP4_BLOCK_SIZE_TL
    blocked = tl.reshape(values, (block_count, _NVFP4_BLOCK_SIZE_TL))
    per_tensor_scale = tl.load(per_tensor_scale_ptr).to(tl.float32)
    if not has_per_tensor_scale:
        per_tensor_scale = 1.0
    packed, encoded_scale = nvfp4_backend.encode_nvfp4_blocks(  # pyright: ignore[reportGeneralTypeIssues]
        blocked,
        per_tensor_scale,
        block_count,
        high_first,
    )
    packed_offsets = tl.arange(0, chunk_size // 2)
    logical_packed_offsets = chunk_start // 2 + packed_offsets
    tl.store(
        qdata_ptr + qdata_row_offset + logical_packed_offsets,
        tl.reshape(packed, (chunk_size // 2,)),
        mask=logical_packed_offsets * 2 < row_width,
    )

    scale_columns = chunk_start // _NVFP4_BLOCK_SIZE_TL + tl.arange(0, block_count)
    if swizzled_scales:
        scale_offsets = nvfp4_backend.swizzled_scale_offsets(
            row,
            scale_columns,
            scale_column_blocks,
        )
    else:
        scale_offsets = row * (row_width // _NVFP4_BLOCK_SIZE_TL) + scale_columns
    tl.store(
        scale_ptr + scale_offsets,
        encoded_scale,
        mask=scale_columns * _NVFP4_BLOCK_SIZE_TL < row_width,
    )


@triton.jit
def _rotate_quantize_nvfp4_kernel(
    input_ptr,
    per_tensor_scale_ptr,
    qdata_ptr,
    scale_ptr,
    row_width,
    chunk_count: tl.constexpr,
    chunk_size0: tl.constexpr,
    chunk_size1: tl.constexpr,
    chunk_size2: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    gguf_quant_type: tl.constexpr,
    has_per_tensor_scale: tl.constexpr,
    swizzled_scales: tl.constexpr,
    high_first: tl.constexpr,
):
    """Apply the second exact rotation and write hardware-ready NVFP4 storage."""
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    input_row_width = row_width * (2 if activation_fn == "swiglu" else 1)
    input_row_offset = row_i64 * input_row_width
    packed_row_offset = 0
    if gguf_quant_type >= 0:
        packed_row_offset = row_i64 * gguf_backend.packed_row_size(
            row_width,
            gguf_quant_type,
        )
    qdata_row_offset = row_i64 * (row_width // 2)

    _rotate_quantize_chunk(
        input_ptr,
        per_tensor_scale_ptr,
        qdata_ptr,
        scale_ptr,
        input_row_offset,
        packed_row_offset,
        qdata_row_offset,
        row,
        row_width,
        tl.constexpr(0),
        chunk_size0,
        group_size,
        inverse_sqrt_group,
        scale_column_blocks,
        activation_fn,
        accelerator_backend,
        gguf_quant_type,
        has_per_tensor_scale,
        swizzled_scales,
        high_first,
    )
    if chunk_count >= 2:
        _rotate_quantize_chunk(
            input_ptr,
            per_tensor_scale_ptr,
            qdata_ptr,
            scale_ptr,
            input_row_offset,
            packed_row_offset,
            qdata_row_offset,
            row,
            row_width,
            chunk_size0,
            chunk_size1,
            group_size,
            inverse_sqrt_group,
            scale_column_blocks,
            activation_fn,
            accelerator_backend,
            gguf_quant_type,
            has_per_tensor_scale,
            swizzled_scales,
            high_first,
        )
    if chunk_count >= 3:
        _rotate_quantize_chunk(
            input_ptr,
            per_tensor_scale_ptr,
            qdata_ptr,
            scale_ptr,
            input_row_offset,
            packed_row_offset,
            qdata_row_offset,
            row,
            row_width,
            chunk_size0 + chunk_size1,
            chunk_size2,
            group_size,
            inverse_sqrt_group,
            scale_column_blocks,
            activation_fn,
            accelerator_backend,
            gguf_quant_type,
            has_per_tensor_scale,
            swizzled_scales,
            high_first,
        )


def _rotation_chunk_sizes(input_features: int, group_size: int) -> tuple[int, int, int]:
    """Plan at most three power-of-two chunks, padding only when necessary."""
    remaining = input_features
    exact_chunk_sizes: list[int] = []
    while remaining:
        chunk_size = 1 << (min(remaining, _MAX_ROTATION_CHUNK_SIZE).bit_length() - 1)
        if chunk_size < group_size:
            raise ValueError(
                f"ConvRot NVFP4 cannot align row width {input_features} to group size {group_size}"
            )
        exact_chunk_sizes.append(chunk_size)
        remaining -= chunk_size
    if len(exact_chunk_sizes) <= 3:
        exact_chunk_sizes.extend(0 for _ in range(3 - len(exact_chunk_sizes)))
        return exact_chunk_sizes[0], exact_chunk_sizes[1], exact_chunk_sizes[2]

    padded_plans: list[tuple[int, tuple[int, ...]]] = []
    for chunk_count in (2, 3):
        minimum_chunk = (input_features + chunk_count - 1) // chunk_count
        chunk_size = 1 << ((minimum_chunk - 1).bit_length())
        if (
            chunk_size <= _MAX_ROTATION_CHUNK_SIZE
            and chunk_size >= group_size
            and chunk_size % group_size == 0
            and (chunk_count - 1) * chunk_size < input_features
        ):
            padded_plans.append(
                (chunk_count * chunk_size, (chunk_size,) * chunk_count),
            )
    if not padded_plans:
        raise ValueError(
            f"ConvRot NVFP4 row width {input_features} exceeds three "
            f"{_MAX_ROTATION_CHUNK_SIZE}-element chunks"
        )
    chunk_sizes = list(min(padded_plans)[1])
    chunk_sizes.extend(0 for _ in range(3 - len(chunk_sizes)))
    return chunk_sizes[0], chunk_sizes[1], chunk_sizes[2]


def _preparation_num_warps(
    chunk_sizes: tuple[int, int, int],
    group_size: int,
) -> tuple[int, int]:
    """Choose local amax/packing schedules without depending on an INT8 GEMM plan."""
    chunk_count = sum(chunk_size > 0 for chunk_size in chunk_sizes)
    amax_num_warps = 2 if chunk_count > 1 and chunk_sizes[0] <= 4_096 else 4
    packing_num_warps = 4 if group_size == 16 else amax_num_warps
    if (
        group_size == 16
        and chunk_sizes[0] >= 4_096
        and (chunk_count == 1 or (chunk_count == 3 and chunk_sizes[2] >= 2_048))
    ):
        packing_num_warps = 8
    return amax_num_warps, packing_num_warps
