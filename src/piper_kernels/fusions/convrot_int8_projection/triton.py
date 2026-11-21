"""Shared ConvRot projection/bias arithmetic, compute configuration, indexing, and chunk means."""

# Triton JIT launch options and constexpr parameters are not ordinary Python types.
# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.fusions.attention import triton as projected_qk
from piper_kernels.linear.convrot.int8 import triton as convrot_int8_kernels

_CHUNK_ROWS = 1024
_TL_CHUNK_ROWS = tl.constexpr(_CHUNK_ROWS)
_MEAN_TILE_ROWS = tl.constexpr(64)
_MEAN_BLOCK_K = 128


@triton.jit
def projection_tile_ids(group_m: tl.constexpr):
    """Optionally group row/head tiles for reuse without changing the launch grid."""
    row, head = tl.program_id(0), tl.program_id(1)
    if group_m:
        rows, heads = tl.num_programs(0), tl.num_programs(1)
        program = row + head * rows
        group = program // (group_m * heads)
        first_row = group * group_m
        group_rows = tl.minimum(rows - first_row, group_m)
        within_group = program % (group_m * heads)
        row = first_row + within_group % group_rows
        head = within_group // group_rows
    return row, head


@triton.jit
def project_tile(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    row_offsets,
    weight_offsets,
    rows,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    aligned_projection: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    bias_ptr=None,
):
    """Return an FP32 ConvRot projection tile with optional bias, before transforms."""
    projection = convrot_int8_kernels.scaled_int8_matmul(
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        row_offsets,
        weight_offsets,
        rows,
        output_features,
        input_features,
        block_m,
        block_n,
        block_k,
        aligned_projection,
    )
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + weight_offsets, weight_offsets < output_features, 0).to(
            tl.float32
        )
        projection += bias[None, :]
    return projection


@triton.jit
def project_rmsnorm_rope_tile(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    row_offsets,
    weight_offsets,
    sequence_offsets,
    rows,
    sequence_length,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    aligned_projection: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    round_rsqrt_to_nearest: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
    bias_ptr=None,
):
    """Return one FP32 normalized and rotated projection tile."""
    projection = project_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        row_offsets,
        weight_offsets,
        rows,
        input_features,
        output_features,
        aligned_projection,
        block_m,
        block_n,
        block_k,
        bias_ptr=bias_ptr,
    )
    projection = tl.reshape(projection, (block_m, heads_per_program, head_dim))
    return projected_qk.rmsnorm_rope_tile(
        projection,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        sequence_offsets,
        sequence_length,
        heads_per_program,
        head_dim,
        rotary_dim,
        norm_epsilon,
        mask_ragged_tail,
        block_m,
        round_rsqrt_to_nearest,
    )


@triton.jit(do_not_specialize=["sequence_length", "chunk_count"])
def _chunk_dequantized_input_mean_kernel(
    input_ptr,
    input_scale_ptr,
    chunk_mean_ptr,
    sequence_length,
    chunk_count,
    input_features: tl.constexpr,
    block_k: tl.constexpr,
):
    """Reduce one 1024-token chunk of dequantized ConvRot input rows per CTA."""
    chunk = tl.program_id(0).to(tl.int64)
    feature_block = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    chunk_start = chunk * _TL_CHUNK_ROWS
    valid_count = tl.minimum(_TL_CHUNK_ROWS, sequence_length - chunk_start)
    active_tiles = tl.cdiv(valid_count, _MEAN_TILE_ROWS)
    feature_offsets = feature_block * block_k + tl.arange(0, block_k)
    valid_features = feature_offsets < input_features
    row_offsets = tl.arange(0, _MEAN_TILE_ROWS)
    accumulator = tl.zeros((block_k,), dtype=tl.float32)
    for tile_idx in tl.range(0, active_tiles, disable_licm=True):
        rows = chunk_start + tile_idx * _MEAN_TILE_ROWS + row_offsets
        valid_rows = rows < sequence_length
        values = tl.load(
            input_ptr
            + (batch * sequence_length + rows[:, None]) * input_features
            + feature_offsets[None, :],
            mask=valid_rows[:, None] & valid_features[None, :],
            other=0,
        ).to(tl.float32)
        scales = tl.load(
            input_scale_ptr + batch * sequence_length + rows,
            mask=valid_rows,
            other=0.0,
        )
        accumulator += tl.sum(values * scales[:, None], axis=0)
    mean = accumulator / tl.maximum(valid_count, 1).to(tl.float32)
    tl.store(
        chunk_mean_ptr + (batch * chunk_count + chunk) * input_features + feature_offsets,
        mean,
        mask=valid_features,
    )


@triton.jit(do_not_specialize=["chunk_count"])
def project_prepared_chunk_means_kernel(
    chunk_mean_ptr,
    weight_ptr,
    weight_scale_ptr,
    value_mean_ptr,
    chunk_count,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    bias_ptr=None,
):
    """Project each 1024-token input mean directly into ``value_mean[B, H, C, D]``."""
    output_block = tl.program_id(0)
    chunk = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    output_offsets = output_block * block_n + tl.arange(0, block_n)
    feature_offsets = tl.arange(0, block_k)
    input_base = chunk_mean_ptr + (batch * chunk_count + chunk) * input_features
    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    for feature_block in range(tl.cdiv(input_features, block_k)):
        remaining_features = input_features - feature_block * block_k
        represented_mean = tl.load(
            input_base + feature_block * block_k + feature_offsets,
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
    projected_mean = accumulator * weight_scale
    if bias_ptr is not None:
        projected_mean += tl.load(
            bias_ptr + output_offsets, output_offsets < output_features, 0
        ).to(tl.float32)
    head_offsets = output_offsets // head_dim
    dim_offsets = output_offsets % head_dim
    out_offsets = (
        (batch * heads + head_offsets.to(tl.int64)) * chunk_count + chunk
    ) * head_dim + dim_offsets
    tl.store(
        value_mean_ptr + out_offsets,
        projected_mean,
        mask=output_offsets < output_features,
    )


def project_value_chunk_means(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    block_n: int,
    block_k: int,
    out: torch.Tensor,
) -> None:
    """Compute per-1024-token represented input means and project them into ``out[B, H, C, D]``."""
    batch, sequence_length, input_features = input_qdata.shape
    if batch == 0:
        return
    _, heads, chunk_count, head_dim = out.shape
    output_features = weight_qdata.shape[0]
    chunk_input_means = torch.empty(
        (batch, chunk_count, input_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    with device_context(input_qdata.device):
        _chunk_dequantized_input_mean_kernel[
            (chunk_count, triton.cdiv(input_features, _MEAN_BLOCK_K), batch)
        ](
            input_qdata,
            input_scale,
            chunk_input_means,
            sequence_length,
            chunk_count,
            input_features=input_features,
            block_k=_MEAN_BLOCK_K,
            num_warps=4,
        )
        project_prepared_chunk_means_kernel[
            (triton.cdiv(output_features, block_n), chunk_count, batch)
        ](
            chunk_input_means,
            weight_qdata,
            weight_scale,
            out,
            chunk_count,
            input_features=input_features,
            output_features=output_features,
            heads=heads,
            head_dim=head_dim,
            block_n=block_n,
            block_k=block_k,
            bias_ptr=bias,
            num_warps=4,
        )
