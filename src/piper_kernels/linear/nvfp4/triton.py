"""Triton kernels for NVFP4 activation preparation and projection."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.nvfp4 import (
    _decode_fp4,
    _decode_fp4_code,
    dynamic_scale,
    encode_nvfp4_blocks,
    pack_e2m1_pairs,
    swizzled_scale_offsets,
)
from piper_kernels._triton.runtime import device_context
from piper_kernels.linear import _bias
from piper_kernels.linear.nvfp4._storage import prepare_activation_storage
from piper_kernels.weights.nvfp4 import _layout

_NVFP4_BLOCK_SIZE = _layout.BLOCK_SIZE
_NVFP4_QDATA_BLOCK_SIZE = _layout.QDATA_BLOCK_SIZE
_NVFP4_BLOCK_SIZE_TL = tl.constexpr(_NVFP4_BLOCK_SIZE)
_NVFP4_QDATA_BLOCK_SIZE_TL = tl.constexpr(_NVFP4_QDATA_BLOCK_SIZE)
_PREPARE_BLOCKS = 32
_BIAS_BLOCK_SIZE = 1_024
_MEAN_BLOCK_M = 256
_MEAN_BLOCK_K = 128
_PROJECTION_BLOCK_N = 64
_PROJECTION_BLOCK_K = 128


@triton.jit
def _prepare_static_kernel(
    input_ptr,
    per_tensor_scale_ptr,
    qdata_ptr,
    scale_ptr,
    block_count,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    swiglu: tl.constexpr,
    blocks_per_program: tl.constexpr,
    high_first: tl.constexpr,
):
    """Quantize static-scale activations directly into both hardware layouts."""
    block_offsets = tl.program_id(0) * blocks_per_program + tl.arange(0, blocks_per_program)
    valid_blocks = block_offsets < block_count
    scale_columns = block_offsets % (output_features // _NVFP4_BLOCK_SIZE_TL)
    rows = block_offsets // (output_features // _NVFP4_BLOCK_SIZE_TL)
    element_offsets = tl.arange(0, _NVFP4_BLOCK_SIZE_TL)
    input_offsets = (
        rows[:, None] * input_features
        + scale_columns[:, None] * _NVFP4_BLOCK_SIZE_TL
        + element_offsets[None, :]
    )
    values = tl.load(
        input_ptr + input_offsets,
        mask=valid_blocks[:, None],
        other=0.0,
    ).to(tl.float32)
    gate = values
    if swiglu:
        gate = tl.load(
            input_ptr + input_offsets + output_features,
            mask=valid_blocks[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
    if swiglu:
        values *= gate / (1.0 + libdevice.exp(-gate))  # pyright: ignore[reportOperatorIssue]

    per_tensor_scale = tl.load(per_tensor_scale_ptr).to(tl.float32)
    packed, encoded_scale = encode_nvfp4_blocks(  # pyright: ignore[reportGeneralTypeIssues]
        values,
        per_tensor_scale,
        blocks_per_program,
        high_first,
    )
    qdata_offsets = (
        rows[:, None] * (output_features // 2)
        + scale_columns[:, None] * _NVFP4_QDATA_BLOCK_SIZE_TL
        + tl.arange(0, _NVFP4_QDATA_BLOCK_SIZE_TL)[None, :]
    )
    tl.store(qdata_ptr + qdata_offsets, packed, mask=valid_blocks[:, None])
    scale_offsets = swizzled_scale_offsets(
        rows,
        scale_columns,
        scale_column_blocks,
    )
    tl.store(scale_ptr + scale_offsets, encoded_scale, mask=valid_blocks)


@triton.jit
def _add_bias_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    elements,
    features: tl.constexpr,
    input_row_stride: tl.constexpr,
    bias_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    valid = offsets < elements
    rows = offsets // features
    columns = offsets % features
    values = tl.load(
        input_ptr + rows * input_row_stride + columns,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    values += tl.load(bias_ptr + columns * bias_stride, mask=valid, other=0.0).to(tl.float32)
    tl.store(output_ptr + rows * output_row_stride + columns, values, mask=valid)


def _prepare_static_storage(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    *,
    swiglu: bool,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    contiguous_input = input.contiguous()
    input_features = int(contiguous_input.shape[-1])
    output_features = input_features // 2 if swiglu else input_features
    rows = int(contiguous_input.numel() // input_features)
    qdata, scale = prepare_activation_storage(input, rows, output_features, out)
    block_count = rows * (output_features // _NVFP4_BLOCK_SIZE)
    with device_context(input.device):
        _prepare_static_kernel[(triton.cdiv(block_count, _PREPARE_BLOCKS),)](
            contiguous_input,
            per_tensor_scale,
            qdata,
            scale,
            block_count,
            input_features=input_features,
            output_features=output_features,
            scale_column_blocks=(output_features + _layout.SCALE_COLUMN_TILE - 1)
            // _layout.SCALE_COLUMN_TILE,
            swiglu=swiglu,
            blocks_per_program=_PREPARE_BLOCKS,
            high_first=high_first,
            num_warps=2,
        )
        return qdata, scale


def prepare_static(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    swiglu: bool = False,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a static-scale NVFP4 activation without intermediate tensors."""
    qdata, scale = _prepare_static_storage(
        input,
        per_tensor_scale,
        swiglu=swiglu,
        high_first=high_first,
    )
    return qdata, scale, per_tensor_scale.clone()


def prepare_static_out(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    out: tuple[torch.Tensor, torch.Tensor],
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare static NVFP4 storage into reusable caller-owned buffers."""
    return _prepare_static_storage(
        input,
        per_tensor_scale,
        swiglu=False,
        out=out,
        high_first=high_first,
    )


def add_bias_out(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bias: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Add bias in FP32 and cast into caller-owned storage, which may alias input."""
    if (
        input.ndim != 2
        or output.shape != input.shape
        or input.stride(1) != 1
        or output.stride(1) != 1
        or bias.ndim != 1
        or bias.shape[0] != input.shape[1]
    ):
        raise ValueError("NVFP4 bias addition requires matching row-major matrices and bias width")
    elements = input.numel()
    with device_context(input.device):
        _add_bias_kernel[(triton.cdiv(elements, _BIAS_BLOCK_SIZE),)](
            input,
            bias,
            output,
            elements,
            features=input.shape[-1],
            input_row_stride=input.stride(0),
            bias_stride=bias.stride(0),
            output_row_stride=output.stride(0),
            block_size=_BIAS_BLOCK_SIZE,
            num_warps=4,
        )


@triton.jit
def _dequantized_input_mean_partial_kernel(
    input_ptr,
    input_scale_ptr,
    partial_ptr,
    block_lengths_ptr,
    sequence_length,
    input_features: tl.constexpr,
    row_block_count: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """Sum one sequence block represented by packed FP4 activation storage."""
    row_block = tl.program_id(0)
    feature_block = tl.program_id(1)
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    feature_start = feature_block * block_k
    feature_offsets = feature_start + tl.arange(0, block_k)
    rows = batch * sequence_length + sequence_offsets
    valid_rows = sequence_offsets < sequence_length
    if mask_block_lengths:
        block_lengths = tl.load(
            block_lengths_ptr + sequence_offsets // 64,
            mask=valid_rows,
            other=0,
        )
        valid_rows &= sequence_offsets % 64 < block_lengths
    packed_feature_offsets = feature_start // 2 + tl.arange(0, block_k // 2)
    valid_packed = valid_rows[:, None] & (packed_feature_offsets[None, :] * 2 < input_features)
    packed = tl.load(
        input_ptr + rows[:, None] * (input_features // 2) + packed_feature_offsets[None, :],
        mask=valid_packed,
        other=0,
    )
    values = tl.interleave(
        _decode_fp4_code(packed & 0xF),
        _decode_fp4_code(packed >> 4),
    )
    scale_columns = feature_start // _NVFP4_BLOCK_SIZE_TL + tl.arange(
        0,
        block_k // _NVFP4_BLOCK_SIZE_TL,
    )
    scale_offsets = swizzled_scale_offsets(
        rows[:, None],
        scale_columns[None, :],
        scale_column_blocks,
    )
    scales = tl.load(
        input_scale_ptr + scale_offsets,
        mask=valid_rows[:, None] & (scale_columns[None, :] * _NVFP4_BLOCK_SIZE_TL < input_features),
        other=0.0,
    ).to(tl.float32)
    scales = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (block_m, block_k // _NVFP4_BLOCK_SIZE_TL, 1)),
            (block_m, block_k // _NVFP4_BLOCK_SIZE_TL, _NVFP4_BLOCK_SIZE_TL),
        ),
        (block_m, block_k),
    )
    valid = valid_rows[:, None] & (feature_offsets[None, :] < input_features)
    values = tl.where(valid, values * scales, 0.0)
    partial_offsets = (batch * row_block_count + row_block) * input_features + feature_offsets
    tl.store(
        partial_ptr + partial_offsets,
        tl.sum(values, axis=0),
        mask=feature_offsets < input_features,
    )


@triton.jit
def _dequantized_input_mean_reduce_kernel(
    partial_ptr,
    input_per_tensor_scale_ptr,
    mean_ptr,
    valid_count_ptr,
    sequence_length,
    input_features: tl.constexpr,
    row_block_count: tl.constexpr,
    reduction_rows: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    block_k: tl.constexpr,
):
    """Reduce represented-activation partial sums into one FP32 mean per batch."""
    feature_block = tl.program_id(0)
    batch = tl.program_id(1)
    row_offsets = tl.arange(0, reduction_rows)
    feature_offsets = feature_block * block_k + tl.arange(0, block_k)
    values = tl.load(
        partial_ptr
        + (batch * row_block_count + row_offsets[:, None]) * input_features
        + feature_offsets[None, :],
        mask=(row_offsets[:, None] < row_block_count) & (feature_offsets[None, :] < input_features),
        other=0.0,
    )
    per_tensor_scale = tl.load(input_per_tensor_scale_ptr).to(tl.float32)
    valid_count = tl.load(valid_count_ptr) if mask_block_lengths else sequence_length
    mean = tl.sum(values, axis=0) * per_tensor_scale / valid_count
    tl.store(
        mean_ptr + batch * input_features + feature_offsets,
        mean,
        mask=feature_offsets < input_features,
    )


@triton.jit
def _project_input_mean_kernel(
    input_mean_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_per_tensor_scale_ptr,
    bias_ptr,
    output_ptr,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Project a represented FP32 input mean through one packed NVFP4 weight."""
    output_block = tl.program_id(0)
    batch = tl.program_id(1)
    output_offsets = output_block * block_n + tl.arange(0, block_n)
    feature_offsets = tl.arange(0, block_k)
    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    for feature_block in range(tl.cdiv(input_features, block_k)):
        logical_features = feature_block * block_k + feature_offsets
        valid = (output_offsets[:, None] < output_features) & (
            logical_features[None, :] < input_features
        )
        input_mean = tl.load(
            input_mean_ptr + batch * input_features + logical_features,
            mask=logical_features < input_features,
            other=0.0,
        )
        packed = tl.load(
            weight_ptr
            + output_offsets[:, None] * (input_features // 2)
            + logical_features[None, :] // 2,
            mask=valid,
            other=0,
        )
        scale_offsets = swizzled_scale_offsets(
            output_offsets[:, None],
            logical_features[None, :] // _NVFP4_BLOCK_SIZE_TL,
            scale_column_blocks,
        )
        scales = tl.load(weight_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.float32)
        weight = _decode_fp4(packed, logical_features[None, :]) * scales
        accumulator += tl.sum(weight * input_mean[None, :], axis=1)
    if has_weight_per_tensor_scale:
        accumulator *= tl.load(weight_per_tensor_scale_ptr).to(tl.float32)
    if has_bias:
        accumulator += tl.load(
            bias_ptr + output_offsets,
            mask=output_offsets < output_features,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        output_ptr + batch * output_features + output_offsets,
        accumulator,
        mask=output_offsets < output_features,
    )


def _validate_linear_mean(  # noqa: PLR0912
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    batch: int,
    sequence_length: int,
    block_lengths: torch.Tensor | None,
) -> tuple[int, int]:
    if input_qdata.ndim != 2 or input_qdata.dtype is not torch.uint8:
        raise ValueError("NVFP4 mean input must be a two-dimensional packed UINT8 tensor")
    rows, packed_input_features = input_qdata.shape
    input_features = 2 * packed_input_features
    if batch <= 0 or sequence_length <= 0 or rows != batch * sequence_length:
        raise ValueError("NVFP4 mean batch and sequence dimensions must match its input rows")
    if input_features % _NVFP4_BLOCK_SIZE:
        raise ValueError(f"NVFP4 mean input features must be divisible by {_NVFP4_BLOCK_SIZE}")
    expected_input_scale_shape = _layout.scale_shape(rows, input_features)
    if (
        input_scale.shape != expected_input_scale_shape
        or input_scale.dtype is not torch.float8_e4m3fn
    ):
        raise ValueError("NVFP4 mean input scale has an incompatible swizzled layout")
    if input_per_tensor_scale.shape != () or input_per_tensor_scale.dtype is not torch.float32:
        raise ValueError("NVFP4 mean input per-tensor scale must be an FP32 scalar")
    if (
        weight_qdata.ndim != 2
        or weight_qdata.dtype is not torch.uint8
        or weight_qdata.shape[1] != packed_input_features
    ):
        raise ValueError("NVFP4 mean weight must be a compatible packed UINT8 matrix")
    output_features = weight_qdata.shape[0]
    expected_weight_scale_shape = _layout.scale_shape(output_features, input_features)
    if (
        weight_scale.shape != expected_weight_scale_shape
        or weight_scale.dtype is not torch.float8_e4m3fn
    ):
        raise ValueError("NVFP4 mean weight scale has an incompatible swizzled layout")
    if weight_per_tensor_scale is not None and (
        weight_per_tensor_scale.shape != () or weight_per_tensor_scale.dtype is not torch.float32
    ):
        raise ValueError("NVFP4 mean weight per-tensor scale must be an FP32 scalar")
    if bias is not None:
        if bias.shape != (output_features,):
            raise ValueError("NVFP4 mean bias must contain one value per output feature")
        _bias.validate_dtype(bias, "NVFP4 mean")
    if block_lengths is not None and (
        sequence_length % 64
        or block_lengths.shape != (sequence_length // 64,)
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != input_qdata.device
        or not block_lengths.is_contiguous()
    ):
        raise ValueError("NVFP4 mean block lengths must be one contiguous device INT32 per K64")
    operands = [
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
    ]
    operands.extend(operand for operand in (weight_per_tensor_scale, bias) if operand is not None)
    if block_lengths is not None:
        operands.append(block_lengths)
    if input_qdata.device.type != "cuda" or any(
        operand.device != input_qdata.device for operand in operands
    ):
        raise ValueError("NVFP4 mean operands must share a CUDA device")
    if any(not operand.is_contiguous() for operand in operands):
        raise ValueError("NVFP4 mean operands must be contiguous")
    return input_features, output_features


@torch.library.custom_op("piper_kernels::nvfp4_linear_mean", mutates_args=())
def linear_mean(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    batch: int,
    sequence_length: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project the compact or valid-front padded mean represented by NVFP4 storage."""
    input_features, output_features = _validate_linear_mean(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        batch,
        sequence_length,
        block_lengths,
    )
    row_block_count = int(triton.cdiv(sequence_length, _MEAN_BLOCK_M))
    partial = torch.empty(
        (batch, row_block_count, input_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    input_mean = torch.empty(
        (batch, input_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    output = torch.empty(
        (batch, output_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    scale_column_blocks = int(triton.cdiv(input_features, 64))
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else input_scale
    valid_count = block_lengths.sum(dtype=torch.float32) if has_block_lengths else input_scale
    with device_context(input_qdata.device):
        _dequantized_input_mean_partial_kernel[
            (row_block_count, triton.cdiv(input_features, _MEAN_BLOCK_K), batch)
        ](
            input_qdata,
            input_scale,
            partial,
            block_lengths_ptr,
            sequence_length,
            input_features=input_features,
            row_block_count=row_block_count,
            scale_column_blocks=scale_column_blocks,
            mask_block_lengths=has_block_lengths,
            block_m=_MEAN_BLOCK_M,
            block_k=_MEAN_BLOCK_K,
            num_warps=8,
        )
        _dequantized_input_mean_reduce_kernel[(triton.cdiv(input_features, _MEAN_BLOCK_K), batch)](
            partial,
            input_per_tensor_scale,
            input_mean,
            valid_count,
            sequence_length,
            input_features=input_features,
            row_block_count=row_block_count,
            reduction_rows=triton.next_power_of_2(row_block_count),
            mask_block_lengths=has_block_lengths,
            block_k=_MEAN_BLOCK_K,
            num_warps=8,
        )
        _project_input_mean_kernel[(triton.cdiv(output_features, _PROJECTION_BLOCK_N), batch)](
            input_mean,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            bias,
            output,
            input_features=input_features,
            output_features=output_features,
            scale_column_blocks=scale_column_blocks,
            has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
            has_bias=bias is not None,
            block_n=_PROJECTION_BLOCK_N,
            block_k=_PROJECTION_BLOCK_K,
            num_warps=8,
        )
        return output


@linear_mean.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _linear_mean_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    batch: int,
    _sequence_length: int,
    _block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    return input_qdata.new_empty((batch, weight_qdata.shape[0]), dtype=torch.float32)


__all__ = [
    "add_bias_out",
    "dynamic_scale",
    "encode_nvfp4_blocks",
    "linear_mean",
    "pack_e2m1_pairs",
    "prepare_static",
    "prepare_static_out",
    "swizzled_scale_offsets",
]
