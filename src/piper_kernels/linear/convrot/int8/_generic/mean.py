"""Portable reduction of the activation represented by prepared INT8 rows."""

# Triton's JIT launch options are not ordinary Python parameters.
# pyright: reportCallIssue=false
# ruff: noqa: ANN001, ANN202

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context

_MEAN_BLOCK_M = 256
_MEAN_BLOCK_K = 128


@triton.jit
def _dequantized_input_mean_partial_kernel(
    input_ptr,
    input_scale_ptr,
    partial_ptr,
    block_lengths_ptr,
    sequence_length,
    input_features: tl.constexpr,
    row_block_count: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """Sum one row block of the activation represented by prepared ConvRot storage."""
    row_block = tl.program_id(0)
    feature_block = tl.program_id(1)
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    feature_offsets = feature_block * block_k + tl.arange(0, block_k)
    valid = (sequence_offsets[:, None] < sequence_length) & (
        feature_offsets[None, :] < input_features
    )
    if mask_block_lengths:
        block_lengths = tl.load(
            block_lengths_ptr + sequence_offsets // 64,
            mask=sequence_offsets < sequence_length,
            other=0,
        )
        valid &= sequence_offsets[:, None] % 64 < block_lengths[:, None]
    values = tl.load(
        input_ptr
        + (batch * sequence_length + sequence_offsets[:, None]) * input_features
        + feature_offsets[None, :],
        mask=valid,
        other=0,
    ).to(tl.float32)
    scales = tl.load(
        input_scale_ptr + batch * sequence_length + sequence_offsets,
        mask=sequence_offsets < sequence_length,
        other=0.0,
    )
    partial = tl.sum(values * scales[:, None], axis=0)
    tl.store(
        partial_ptr + (batch * row_block_count + row_block) * input_features + feature_offsets,
        partial,
        mask=feature_offsets < input_features,
    )


@triton.jit
def _dequantized_input_mean_reduce_kernel(
    partial_ptr,
    output_ptr,
    valid_count_ptr,
    sequence_length,
    input_features: tl.constexpr,
    row_block_count: tl.constexpr,
    reduction_rows: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    block_k: tl.constexpr,
):
    """Reduce prepared-input partial sums to one FP32 feature mean per batch."""
    feature_block = tl.program_id(0)
    batch = tl.program_id(1)
    row_offsets = tl.arange(0, reduction_rows)
    feature_offsets = feature_block * block_k + tl.arange(0, block_k)
    partial = tl.load(
        partial_ptr
        + (batch * row_block_count + row_offsets[:, None]) * input_features
        + feature_offsets[None, :],
        mask=(row_offsets[:, None] < row_block_count) & (feature_offsets[None, :] < input_features),
        other=0.0,
    )
    valid_count = tl.load(valid_count_ptr) if mask_block_lengths else sequence_length
    mean = tl.sum(partial, axis=0) / valid_count
    tl.store(
        output_ptr + batch * input_features + feature_offsets,
        mean,
        mask=feature_offsets < input_features,
    )


def dequantized_input_mean(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the FP32 compact or valid-front padded prepared-input mean."""
    if input_qdata.ndim != 3 or input_qdata.dtype is not torch.int8:
        raise ValueError("ConvRot mean input must be [batch,sequence,features] INT8")
    batch, sequence_length, input_features = input_qdata.shape
    if input_scale.shape != (batch, sequence_length) or input_scale.dtype is not torch.float32:
        raise ValueError("ConvRot mean scale must be a batch/sequence FP32 matrix")
    if input_scale.device != input_qdata.device:
        raise ValueError("ConvRot mean operands must share a device")
    if not input_qdata.is_contiguous() or not input_scale.is_contiguous():
        raise ValueError("ConvRot mean operands must be contiguous")
    if block_lengths is not None and (
        sequence_length % 64
        or block_lengths.shape != (sequence_length // 64,)
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != input_qdata.device
        or not block_lengths.is_contiguous()
    ):
        raise ValueError("ConvRot mean block lengths must be one contiguous device INT32 per K64")
    row_block_count = int(triton.cdiv(sequence_length, _MEAN_BLOCK_M))
    partial = torch.empty(
        (batch, row_block_count, input_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    output = torch.empty(
        (batch, input_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
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
            mask_block_lengths=has_block_lengths,
            block_m=_MEAN_BLOCK_M,
            block_k=_MEAN_BLOCK_K,
            num_warps=8,
        )
        _dequantized_input_mean_reduce_kernel[(triton.cdiv(input_features, _MEAN_BLOCK_K), batch)](
            partial,
            output,
            valid_count,
            sequence_length,
            input_features=input_features,
            row_block_count=row_block_count,
            reduction_rows=triton.next_power_of_2(row_block_count),
            mask_block_lengths=has_block_lengths,
            block_k=_MEAN_BLOCK_K,
            num_warps=8,
        )
        return output
