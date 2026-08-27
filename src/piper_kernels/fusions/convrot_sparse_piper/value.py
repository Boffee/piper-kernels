"""ConvRot projection and tile-scaled INT8 V preparation for sparse Piper."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

# Triton device functions cannot carry ordinary Python type annotations.
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

_BLOCK_M = 128
_VALUE_TILE_ROWS = 64
_BLOCK_K = 128
_HEAD_DIM = 128
_HEADS_PER_PROGRAM = 2
_BLOCK_N = _HEAD_DIM * _HEADS_PER_PROGRAM
_P_UINT8_RANGE = tl.constexpr(255.0)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)
_JIT_VALUE_TILE_ROWS = tl.constexpr(64)


@triton.jit
def _project_prepared_input_mean_kernel(
    input_mean_ptr,
    weight_ptr,
    weight_scale_ptr,
    value_mean_ptr,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Project one represented-input mean without quantizing that compact row."""
    output_block = tl.program_id(0)
    batch = tl.program_id(1)
    output_offsets = output_block * block_n + tl.arange(0, block_n)
    feature_offsets = tl.arange(0, block_k)
    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    for feature_block in range(tl.cdiv(input_features, block_k)):
        remaining_features = input_features - feature_block * block_k
        represented_mean = tl.load(
            input_mean_ptr + batch * input_features + feature_block * block_k + feature_offsets,
            mask=feature_offsets < remaining_features,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + output_offsets[:, None] * input_features
            + feature_block * block_k
            + feature_offsets[None, :],
            mask=(output_offsets[:, None] < output_features)
            & (feature_offsets[None, :] < remaining_features),
            other=0,
        ).to(tl.float32)
        accumulator += tl.sum(weight * represented_mean[None, :], axis=1)
    weight_scale = tl.load(
        weight_scale_ptr + output_offsets,
        mask=output_offsets < output_features,
        other=0.0,
    )
    tl.store(
        value_mean_ptr + batch * output_features + output_offsets,
        accumulator * weight_scale,
        mask=output_offsets < output_features,
    )


@triton.jit
def _convrot_project_quantize_sparse_value_kernel(  # noqa: PLR0913, PLR0917
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    value_mean_ptr,
    value_ptr,
    value_scale_ptr,
    rows,
    sequence_length,
    row_block_offset,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    aligned_projection: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Project two heads over two K64 tiles and emit sparse Piper's V format."""
    tl.static_assert(block_m == 2 * _JIT_VALUE_TILE_ROWS)
    tl.static_assert(heads_per_program == 2)
    tl.static_assert(block_n == heads_per_program * head_dim)
    tl.static_assert(head_dim == 128)

    row_block = row_block_offset + tl.program_id(0)
    head_block = tl.program_id(1)
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    row_offsets = batch * sequence_length + sequence_offsets
    feature_offsets = tl.arange(0, head_dim)
    projection_feature_offsets = tl.arange(0, block_n)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    weight_offsets = head_block * block_n + projection_feature_offsets
    projection = convrot_backend.scaled_int8_matmul(
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        row_offsets,
        weight_offsets,
        rows,
        heads * head_dim,
        input_features,
        block_m,
        block_n,
        block_k,
        aligned_projection,
    )
    projection = tl.reshape(projection, (block_m, heads_per_program, head_dim))
    value_mean = tl.load(
        value_mean_ptr
        + (batch * heads + head_offsets[:, None]) * head_dim
        + feature_offsets[None, :],
        mask=head_offsets[:, None] < heads,
        other=0.0,
    )
    valid_rows = sequence_offsets < sequence_length
    centered = tl.where(
        valid_rows[:, None, None],
        projection - value_mean[None, :, :],
        0.0,
    )
    centered = tl.permute(centered, (1, 0, 2))
    grouped = tl.reshape(
        centered,
        (
            heads_per_program,
            block_m // _JIT_VALUE_TILE_ROWS,
            _JIT_VALUE_TILE_ROWS,
            head_dim,
        ),
    )
    maximum = tl.max(tl.max(tl.abs(grouped), axis=3), axis=2)
    value_scale = maximum / _V_INT8_RANGE + _SCALE_EPSILON
    quantized = qk_quantization.round_to_int8(grouped / value_scale[:, :, None, None])
    quantized = tl.reshape(quantized, (heads_per_program, block_m, head_dim))

    value_offsets = (
        (batch * heads + head_offsets[:, None, None]) * head_dim * sequence_length
        + feature_offsets[None, None, :] * sequence_length
        + sequence_offsets[None, :, None]
    )
    tl.store(
        value_ptr + value_offsets,
        quantized,
        mask=(head_offsets[:, None, None] < heads)
        & (sequence_offsets[None, :, None] < sequence_length),
    )

    tile_count = sequence_length // _JIT_VALUE_TILE_ROWS
    local_tile_offsets = tl.arange(0, block_m // _JIT_VALUE_TILE_ROWS)
    tile_offsets = row_block * (block_m // _JIT_VALUE_TILE_ROWS) + local_tile_offsets
    scale_offsets = (batch * heads + head_offsets[:, None]) * tile_count + tile_offsets[None, :]
    tl.store(
        value_scale_ptr + scale_offsets,
        value_scale * _P_UINT8_RANGE,
        mask=(head_offsets[:, None] < heads) & (tile_offsets[None, :] < tile_count),
    )


def _validate_inputs(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[int, int, int]:
    if input_qdata.ndim != 3 or input_qdata.dtype is not torch.int8:
        raise ValueError("V projection input must be [batch,sequence,features] INT8")
    batch, sequence_length, input_features = input_qdata.shape
    if input_scale.shape != (batch, sequence_length) or input_scale.dtype is not torch.float32:
        raise ValueError("V projection input scale must be a batch/sequence FP32 matrix")
    if input_mean.shape != (batch, input_features) or input_mean.dtype is not torch.float32:
        raise ValueError("V projection represented-input mean must be a batch/feature FP32 matrix")
    if weight_qdata.ndim != 2 or weight_qdata.dtype is not torch.int8:
        raise ValueError("V projection weight must be a two-dimensional INT8 tensor")
    if weight_qdata.shape[1] != input_features or weight_qdata.shape[0] % _HEAD_DIM:
        raise ValueError("V projection weight must map the input to complete D128 heads")
    if weight_scale.shape != (weight_qdata.shape[0], 1) or weight_scale.dtype is not torch.float32:
        raise ValueError("V projection weight scale must be one FP32 value per output feature")
    operands = input_qdata, input_scale, input_mean, weight_qdata, weight_scale
    if any(operand.device != input_qdata.device for operand in operands):
        raise ValueError("V projection operands must share a device")
    if input_qdata.device.type != "cuda":
        raise ValueError("V projection fusion currently requires CUDA")
    if any(not operand.is_contiguous() for operand in operands):
        raise ValueError("V projection operands must be contiguous")
    if sequence_length < _VALUE_TILE_ROWS or sequence_length % _VALUE_TILE_ROWS:
        raise ValueError("V projection sequence must be K64 aligned")
    return batch, sequence_length, weight_qdata.shape[0] // _HEAD_DIM


def _launch_value_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, heads = _validate_inputs(
        input_qdata,
        input_scale,
        input_mean,
        weight_qdata,
        weight_scale,
    )
    value = torch.empty(
        (batch, heads, _HEAD_DIM, sequence_length),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    value_scale_multiplier = torch.empty(
        (batch, heads, sequence_length // _VALUE_TILE_ROWS, 1),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    value_mean = torch.empty(
        (batch, heads, _HEAD_DIM),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    _project_prepared_input_mean_kernel[(triton.cdiv(heads * _HEAD_DIM, _BLOCK_N), batch)](
        input_mean,
        weight_qdata,
        weight_scale,
        value_mean,
        input_features=input_qdata.shape[2],
        output_features=heads * _HEAD_DIM,
        block_n=_BLOCK_N,
        block_k=_BLOCK_K,
        num_warps=8,
    )

    def launch(row_block_count: int, row_block_offset: int, *, aligned_rows: bool) -> None:
        _convrot_project_quantize_sparse_value_kernel[
            (
                row_block_count,
                triton.cdiv(heads, _HEADS_PER_PROGRAM),
                batch,
            )
        ](
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            value_mean,
            value,
            value_scale_multiplier,
            batch * sequence_length,
            sequence_length,
            row_block_offset,
            input_features=input_qdata.shape[2],
            heads=heads,
            heads_per_program=_HEADS_PER_PROGRAM,
            head_dim=_HEAD_DIM,
            aligned_projection=(
                aligned_rows
                and input_qdata.shape[2] % _BLOCK_K == 0
                and heads % _HEADS_PER_PROGRAM == 0
            ),
            block_m=_BLOCK_M,
            block_n=_BLOCK_N,
            block_k=_BLOCK_K,
            num_warps=8,
            num_stages=3,
        )

    full_row_blocks = sequence_length // _BLOCK_M
    if full_row_blocks:
        launch(full_row_blocks, 0, aligned_rows=True)
    if sequence_length % _BLOCK_M:
        launch(1, full_row_blocks, aligned_rows=False)
    return value, value_scale_multiplier, value_mean


@torch.library.custom_op(
    "piper_kernels::convrot_sparse_piper_project_value",
    mutates_args=(),
)
def _project_value_op(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_value_projection(
        input_qdata,
        input_scale,
        input_mean,
        weight_qdata,
        weight_scale,
    )


@_project_value_op.register_fake
def _project_value_op_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, _input_features = input_qdata.shape
    heads = weight_qdata.shape[0] // _HEAD_DIM
    return (
        input_qdata.new_empty((batch, heads, _HEAD_DIM, sequence_length)),
        input_qdata.new_empty(
            (batch, heads, sequence_length // _VALUE_TILE_ROWS, 1),
            dtype=torch.float32,
        ),
        input_qdata.new_empty((batch, heads, _HEAD_DIM), dtype=torch.float32),
    )
