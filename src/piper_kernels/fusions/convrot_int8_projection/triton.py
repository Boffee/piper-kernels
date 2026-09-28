"""Shared ConvRot projection/bias arithmetic, compute configuration, indexing, and means."""

# pyright: reportArgumentType=false, reportCallIssue=false

import triton
import triton.language as tl

from piper_kernels.linear.convrot.int8._kernels import triton as convrot_int8_kernels


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
def project_prepared_input_mean_kernel(
    input_mean_ptr,
    weight_ptr,
    weight_scale_ptr,
    value_mean_ptr,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    bias_ptr=None,
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
    projected_mean = accumulator * weight_scale
    if bias_ptr is not None:
        projected_mean += tl.load(
            bias_ptr + output_offsets, output_offsets < output_features, 0
        ).to(tl.float32)
    tl.store(
        value_mean_ptr + batch * output_features + output_offsets,
        projected_mean,
        mask=output_offsets < output_features,
    )
