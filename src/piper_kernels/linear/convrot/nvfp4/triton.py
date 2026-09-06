"""Exact static and dynamic ConvRot NVFP4 activation preparation."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.gguf import triton as gguf_backend
from piper_kernels.linear._input_activations import input_activation_width
from piper_kernels.linear.convrot import triton as convrot_backend
from piper_kernels.linear.convrot._rotation import validate_group_size
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend

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
        0,
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
        0,
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


type _ValidatedInput = tuple[
    torch.Tensor,
    int,
    int,
    tuple[int, int, int],
    AcceleratorTarget,
]


def _validate_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
) -> _ValidatedInput:
    validate_group_size(group_size)
    if input.ndim == 0 or input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("ConvRot NVFP4 input must be a non-scalar FP16 or BF16 tensor")
    source_features = int(input.shape[-1])
    if source_features < 1:
        raise ValueError("ConvRot NVFP4 requires a nonempty feature dimension")
    activation_width = input_activation_width(activation_fn)
    if source_features % activation_width:
        raise ValueError("ConvRot NVFP4 input activation requires equal feature partitions")
    input_features = source_features // activation_width
    rows = int(input.numel() // source_features)
    if rows < 1 or input_features % group_size or input_features % _NVFP4_BLOCK_SIZE:
        raise ValueError(
            "ConvRot NVFP4 requires nonempty rows divisible by the rotation and FP4 blocks"
        )
    if input.device.type != "cuda":
        raise ValueError("ConvRot NVFP4 currently requires CUDA")
    target = AcceleratorTarget.from_device(input.device)
    if not target.is_cuda_capability(12, 0):
        raise ValueError("ConvRot NVFP4 requires exact NVIDIA SM120")
    return (
        input.contiguous(),
        rows,
        input_features,
        _rotation_chunk_sizes(input_features, group_size),
        target,
    )


def _prepare_dynamic_scale(
    validated_input: _ValidatedInput,
    group_size: int,
    out: torch.Tensor | None = None,
    *,
    activation_fn: str | None = None,
) -> torch.Tensor:
    contiguous_input, rows, input_features, chunk_sizes, target = validated_input
    chunk_count = sum(chunk_size > 0 for chunk_size in chunk_sizes)
    chunk_size0, chunk_size1, chunk_size2 = chunk_sizes
    amax_num_warps, _ = _preparation_num_warps(chunk_sizes, group_size)
    row_amax = torch.empty(rows, device=contiguous_input.device, dtype=torch.float32)
    with device_context(contiguous_input.device):
        _rotated_row_amax_kernel[(rows,)](
            contiguous_input,
            row_amax,
            input_features,
            chunk_count=chunk_count,
            chunk_size0=chunk_size0,
            chunk_size1=chunk_size1,
            chunk_size2=chunk_size2,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            activation_fn=activation_fn,
            accelerator_backend=target.backend,
            gguf_quant_type=-1,
            num_warps=amax_num_warps,
        )
        return nvfp4_backend.dynamic_scale(row_amax, out=out)


def _prepare_static_storage(
    validated_input: _ValidatedInput,
    per_tensor_scale: torch.Tensor,
    group_size: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    activation_fn: str | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    contiguous_input, rows, input_features, chunk_sizes, target = validated_input
    if (
        per_tensor_scale.shape != ()
        or per_tensor_scale.dtype is not torch.float32
        or per_tensor_scale.device != contiguous_input.device
        or not per_tensor_scale.is_contiguous()
    ):
        raise ValueError("ConvRot NVFP4 scale must be a contiguous FP32 scalar on the input device")
    qdata, scale = nvfp4_layout.prepare_activation_storage(
        contiguous_input,
        rows,
        input_features,
        out,
    )
    chunk_count = sum(chunk_size > 0 for chunk_size in chunk_sizes)
    chunk_size0, chunk_size1, chunk_size2 = chunk_sizes
    _, packing_num_warps = _preparation_num_warps(chunk_sizes, group_size)
    with device_context(contiguous_input.device):
        _rotate_quantize_nvfp4_kernel[(rows,)](
            contiguous_input,
            per_tensor_scale,
            qdata,
            scale,
            input_features,
            chunk_count=chunk_count,
            chunk_size0=chunk_size0,
            chunk_size1=chunk_size1,
            chunk_size2=chunk_size2,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            scale_column_blocks=(input_features + nvfp4_layout.SCALE_COLUMN_TILE - 1)
            // nvfp4_layout.SCALE_COLUMN_TILE,
            activation_fn=activation_fn,
            accelerator_backend=target.backend,
            gguf_quant_type=-1,
            has_per_tensor_scale=True,
            swizzled_scales=True,
            high_first=high_first,
            num_warps=packing_num_warps,
        )
        return qdata, scale


def _gguf_dynamic_scale(
    data: torch.Tensor,
    quant_type: int,
    group_size: int,
    rows: int,
    row_width: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode GGUF rows through the existing post-rotation amax kernel."""
    chunk_sizes = _rotation_chunk_sizes(row_width, group_size)
    chunk_count = sum(chunk_size > 0 for chunk_size in chunk_sizes)
    chunk_size0, chunk_size1, chunk_size2 = chunk_sizes
    amax_num_warps, _ = _preparation_num_warps(chunk_sizes, group_size)
    row_amax = torch.empty(rows, device=data.device, dtype=torch.float32)
    target = AcceleratorTarget.from_device(data.device)
    with device_context(data.device):
        _rotated_row_amax_kernel[(rows,)](
            data,
            row_amax,
            row_width,
            chunk_count=chunk_count,
            chunk_size0=chunk_size0,
            chunk_size1=chunk_size1,
            chunk_size2=chunk_size2,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            activation_fn=None,
            accelerator_backend=target.backend,
            gguf_quant_type=quant_type,
            num_warps=amax_num_warps,
        )
        return nvfp4_backend.dynamic_scale(row_amax, out=out)


def _gguf_prepare_out(
    data: torch.Tensor,
    quant_type: int,
    group_size: int,
    per_tensor_scale: torch.Tensor | None,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    *,
    is_swizzled_scales: bool,
    high_first: bool,
) -> None:
    """Decode GGUF rows through the existing ConvRot NVFP4 packing kernel."""
    rows, packed_width = qdata.shape
    row_width = packed_width * 2
    chunk_sizes = _rotation_chunk_sizes(row_width, group_size)
    chunk_count = sum(chunk_size > 0 for chunk_size in chunk_sizes)
    chunk_size0, chunk_size1, chunk_size2 = chunk_sizes
    _, packing_num_warps = _preparation_num_warps(chunk_sizes, group_size)
    target = AcceleratorTarget.from_device(data.device)
    with device_context(data.device):
        _rotate_quantize_nvfp4_kernel[(rows,)](
            data,
            per_tensor_scale if per_tensor_scale is not None else data,
            qdata,
            scale,
            row_width,
            chunk_count=chunk_count,
            chunk_size0=chunk_size0,
            chunk_size1=chunk_size1,
            chunk_size2=chunk_size2,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            scale_column_blocks=(row_width + nvfp4_layout.SCALE_COLUMN_TILE - 1)
            // nvfp4_layout.SCALE_COLUMN_TILE,
            activation_fn=None,
            accelerator_backend=target.backend,
            gguf_quant_type=quant_type,
            has_per_tensor_scale=per_tensor_scale is not None,
            swizzled_scales=is_swizzled_scales,
            high_first=high_first,
            num_warps=packing_num_warps,
        )


def dynamic_scale(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate the exact dynamic scale after grouped rotation."""
    validated_input = _validate_input(input, group_size)
    return _prepare_dynamic_scale(validated_input, group_size, out)


def prepare_static(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    group_size: int,
    activation_fn: str | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then rotate and pack using a supplied scale."""
    validated_input = _validate_input(input, group_size, activation_fn)
    qdata, scale = _prepare_static_storage(
        validated_input,
        per_tensor_scale,
        group_size,
        activation_fn=activation_fn,
        high_first=high_first,
    )
    return qdata, scale, per_tensor_scale.clone()


def prepare_static_out(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    group_size: int,
    out: tuple[torch.Tensor, torch.Tensor],
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and pack into reusable caller-owned NVFP4 storage."""
    validated_input = _validate_input(input, group_size)
    return _prepare_static_storage(
        validated_input,
        per_tensor_scale,
        group_size,
        out,
        high_first=high_first,
    )


def prepare_dynamic(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then prepare exact dynamic ConvRot NVFP4 storage."""
    validated_input = _validate_input(input, group_size, activation_fn)
    per_tensor_scale = _prepare_dynamic_scale(
        validated_input,
        group_size,
        activation_fn=activation_fn,
    )
    qdata, scale = _prepare_static_storage(
        validated_input,
        per_tensor_scale,
        group_size,
        out,
        activation_fn=activation_fn,
        high_first=high_first,
    )
    return qdata, scale, per_tensor_scale


__all__ = [
    "dynamic_scale",
    "prepare_dynamic",
    "prepare_static",
    "prepare_static_out",
]
