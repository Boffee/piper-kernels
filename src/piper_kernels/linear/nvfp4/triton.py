"""Triton utilities for prepared NVFP4 activation storage."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

_MEAN_BLOCK_M = 256
_MEAN_BLOCK_K = 128
_PROJECTION_BLOCK_N = 64
_PROJECTION_BLOCK_K = 128


@triton.jit
def _decode_fp4(packed, logical_offsets):
    code = tl.where(logical_offsets % 2 == 0, packed & 0xF, packed >> 4)
    magnitude = code & 0x7
    value = tl.where(
        magnitude <= 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
    )
    return tl.where(code & 0x8 == 0, value, -value)


@triton.jit
def _swizzled_scale_offsets(rows, scale_columns, column_blocks: tl.constexpr):
    row_block = rows // 128
    row_inner = rows % 128
    column_block = scale_columns // 4
    return (
        ((row_block * column_blocks + column_block) * 32 + row_inner % 32) * 16
        + (row_inner // 32) * 4
        + scale_columns % 4
    )


@triton.jit
def _dequantized_input_mean_partial_kernel(
    input_ptr,
    input_scale_ptr,
    partial_ptr,
    sequence_length,
    input_features: tl.constexpr,
    row_block_count: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """Sum one sequence block represented by packed FP4 activation storage."""
    row_block = tl.program_id(0)
    feature_block = tl.program_id(1)
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    feature_offsets = feature_block * block_k + tl.arange(0, block_k)
    rows = batch * sequence_length + sequence_offsets
    valid = (sequence_offsets[:, None] < sequence_length) & (
        feature_offsets[None, :] < input_features
    )
    packed = tl.load(
        input_ptr + rows[:, None] * (input_features // 2) + feature_offsets[None, :] // 2,
        mask=valid,
        other=0,
    )
    scale_offsets = _swizzled_scale_offsets(
        rows[:, None],
        feature_offsets[None, :] // 16,
        scale_column_blocks,
    )
    scales = tl.load(input_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.float32)
    values = _decode_fp4(packed, feature_offsets[None, :]) * scales
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
    sequence_length,
    input_features: tl.constexpr,
    row_block_count: tl.constexpr,
    reduction_rows: tl.constexpr,
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
    mean = tl.sum(values, axis=0) * per_tensor_scale / sequence_length
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
        scale_offsets = _swizzled_scale_offsets(
            output_offsets[:, None],
            logical_features[None, :] // 16,
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


def _validate_linear_mean(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    batch: int,
    sequence_length: int,
) -> tuple[int, int]:
    if input_qdata.ndim != 2 or input_qdata.dtype is not torch.uint8:
        raise ValueError("NVFP4 mean input must be a two-dimensional packed UINT8 tensor")
    rows, packed_input_features = input_qdata.shape
    input_features = 2 * packed_input_features
    if batch <= 0 or sequence_length <= 0 or rows != batch * sequence_length:
        raise ValueError("NVFP4 mean batch and sequence dimensions must match its input rows")
    if input_features % 16:
        raise ValueError("NVFP4 mean input features must be divisible by 16")
    expected_input_scale_shape = (
        triton.cdiv(rows, 128) * 32,
        triton.cdiv(input_features, 64) * 16,
    )
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
    expected_weight_scale_shape = (
        triton.cdiv(output_features, 128) * 32,
        triton.cdiv(input_features, 64) * 16,
    )
    if (
        weight_scale.shape != expected_weight_scale_shape
        or weight_scale.dtype is not torch.float8_e4m3fn
    ):
        raise ValueError("NVFP4 mean weight scale has an incompatible swizzled layout")
    if weight_per_tensor_scale is not None and (
        weight_per_tensor_scale.shape != () or weight_per_tensor_scale.dtype is not torch.float32
    ):
        raise ValueError("NVFP4 mean weight per-tensor scale must be an FP32 scalar")
    if bias is not None and (bias.shape != (output_features,) or bias.dtype is not torch.bfloat16):
        raise ValueError("NVFP4 mean bias must be one BF16 value per output feature")
    operands = [
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
    ]
    operands.extend(operand for operand in (weight_per_tensor_scale, bias) if operand is not None)
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
) -> torch.Tensor:
    """Project the sequence mean represented by prepared NVFP4 storage."""
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
    _dequantized_input_mean_partial_kernel[
        (row_block_count, triton.cdiv(input_features, _MEAN_BLOCK_K), batch)
    ](
        input_qdata,
        input_scale,
        partial,
        sequence_length,
        input_features=input_features,
        row_block_count=row_block_count,
        scale_column_blocks=scale_column_blocks,
        block_m=_MEAN_BLOCK_M,
        block_k=_MEAN_BLOCK_K,
        num_warps=8,
    )
    _dequantized_input_mean_reduce_kernel[(triton.cdiv(input_features, _MEAN_BLOCK_K), batch)](
        partial,
        input_per_tensor_scale,
        input_mean,
        sequence_length,
        input_features=input_features,
        row_block_count=row_block_count,
        reduction_rows=triton.next_power_of_2(row_block_count),
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
) -> torch.Tensor:
    return input_qdata.new_empty((batch, weight_qdata.shape[0]), dtype=torch.float32)


__all__ = ["linear_mean"]
